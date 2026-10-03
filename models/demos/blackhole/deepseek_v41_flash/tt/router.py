# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision-preserving router for DeepSeek-V4.1-Flash.

``TTMoEGate`` feeds ``generalized_moe_gate`` bf16 scores and a bf16 selection bias. V4.1's
correction bias reaches |bias| ~ 10-14 while the 6th/7th-ranked ``score + bias`` margin is often
0.001-0.02, so bf16 rounding of the bias alone (ulp 0.0625 above 8) flips experts: measured on real layer
0/2/20 activations the bf16 path agrees with the fp32 reference expert set for only 12-39% of tokens.

Fix (selection is by ``score + bias``, weights come from the unbiased ``score``):

  * shift: bias <- bias - K, K = the layer's calibrated top-6 cutoff (a constant per layer, ranking-invariant).
    On real activations this alone lifts exact-set agreement from 12-39% to 89-92%, with or without a bf16 sum.

  * bias = b_hi + b_lo with b_hi = bf16(bias) (exact in the op's bf16 bias tensor) and b_lo = bias - b_hi.
  * b_lo is added to the *fp32* score before it is cast to bf16 for the op, so the op ranks
    ``bf16(score + b_lo) + b_hi`` -- error ~0.003 instead of ~0.03 (92-93% exact-set agreement).
  * the routing weights are then recomputed from the unfolded fp32 score at the selected indices,
    so the folded b_lo does not perturb them.

The router matmul and the score transform run in fp32 for the same reason.
"""

import os

import torch

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate

# moe_compute accumulates its three matmuls in bf16 DEST (fp32_dest_acc_en=false), which has a systematic +5% gain per
# matmul (measured: tests/probe_matmul_gain.py) -> routed expert outputs come out ~1.14x too large (tests/
# test_moe_compute_isolated.py). Until the kernel accumulates in fp32, the routing weights are divided by this gain.
ROUTED_GAIN = float(os.environ.get("DSV41_ROUTED_GAIN", "1.0"))


class DSV41Gate(TTMoEGate):
    """sqrtsoftplus, ungrouped, kernel path (257-512 experts), selection bias split into hi/lo."""

    def __init__(self, mesh_device, config, torch_gate_weight, torch_gate_bias, bias_shift: float = 0.0):
        assert config.score_func == "sqrtsoftplus" and config.n_group == 1 and config.score_correction_bias
        # Ranking is invariant to a constant added to every expert's bias. Subtracting the layer's typical
        # selection cutoff (the 6th-ranked score+bias, std across tokens ~0.1) moves the values near the
        # top-6 boundary to ~0 where bf16 is accurate (see module docstring / calibrate.py).
        bias = torch_gate_bias.to(torch.float32) - float(bias_shift)
        b_hi = bias.to(torch.bfloat16).to(torch.float32)  # exactly representable in the op's bf16 bias tensor
        b_lo = bias - b_hi
        super().__init__(mesh_device, config, torch_gate_weight=torch_gate_weight, torch_gate_bias=b_hi)
        assert not self.use_fallback and self.score_transform == "sqrtsoftplus"
        padded = self._padded_experts
        lo = torch.zeros(1, 1, 1, padded, dtype=torch.float32)
        lo[0, 0, 0, : bias.numel()] = b_lo
        self._tt_b_lo = ttnn.from_torch(
            lo,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # exact router: full fp32 (shifted) bias for the fp32 ranking, phantom padded experts ranked last
        rank_bias = torch.full((1, 1, 1, padded), -1e9, dtype=torch.float32)
        rank_bias[0, 0, 0, : bias.numel()] = bias
        self._tt_rank_bias = ttnn.from_torch(
            rank_bias,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # fast exact router: unpadded 384-wide weight / bias (padding to 512 only serves the kernel path)
        E = bias.numel()
        self._E = E
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self._w_exact = ttnn.from_torch(
            torch_gate_weight.to(torch.bfloat16).reshape(1, 1, -1, E),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self._bias_exact = ttnn.from_torch(
            bias.reshape(1, 1, 1, E),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self._arange = ttnn.from_torch(
            torch.arange(E, dtype=torch.float32).reshape(1, 1, 1, E),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        # The kernel-path tensors TTMoEGate keeps in L1 (12 KB per bank per layer, ~490 KB over 40 layers) are only used by
        # DSV41_ROUTER=kernel; the exact router does not need them, and they would sit in the middle of L1 next to the
        # static circular buffers of SDPA decode. Free them unless the kernel router was asked for.
        self._kernel_tensors = os.environ.get("DSV41_ROUTER", "exact") == "kernel"
        if not self._kernel_tensors:
            for name in ("tt_bias", "tt_input_indices", "tt_output", "tt_output_indices"):
                ttnn.deallocate(getattr(self, name))
                setattr(self, name, None)
        # constants of _forward_exact_fast2
        const = lambda t: ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self._zero_exact = const(torch.zeros(1, 1, 1, E))
        self._arange_col = const(torch.arange(E, dtype=torch.float32).reshape(1, 1, E, 1))
        self._ones_k = const(torch.ones(1, 1, self.k, self.k))
        self._ckc_exact = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, tt_x: ttnn.Tensor):
        """``tt_x`` [1, 1, batch, hidden] -> (weights, indices), each [batch, 1, 1, k].

        DSV41_ROUTER=exact (default): everything in fp32 and ``ttnn.topk`` on fp32 ranks (exact expert sets, ~13 ops).
        DSV41_ROUTER=kernel: the bf16 ``generalized_moe_gate`` op with the shifted-bias trick (faster, ~95% exact sets).
        """
        mode = os.environ.get("DSV41_ROUTER", "fused")
        if mode in ("exact2", "exact"):  # exact2 (default): same expert ids as exact_fast, ~40 us/call faster
            return self._forward_exact_fast2(tt_x)
        if mode == "fused":
            return self._forward_fused(tt_x)
        if mode == "exact_fast":
            return self._forward_exact_fast(tt_x)
        if mode == "exact_ref":
            return self._forward_exact(tt_x)
        if mode == "exact2":
            return self._forward_exact_fast2(tt_x)
        return self._forward_kernel(tt_x)

    def _forward_exact_fast(self, tt_x: ttnn.Tensor):
        """Same maths as ``_forward_exact`` (identical expert sets and weights) in ~9 ops: unpadded 384-wide matmul on a
        2x8 core grid (24 us vs 76), top-k over 384 (70 us vs 88) and a one-hot select instead of ``ttnn.gather`` (28 us vs 49).
        """
        T, E, k = tt_x.shape[2], self._E, self.k
        logits = ttnn.matmul(
            tt_x,
            self._w_exact,
            compute_kernel_config=self._ckc_exact,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=2, x=8),
        )
        score = ttnn.sqrt(ttnn.softplus(logits, beta=1.0, threshold=20.0))  # fp32 [1, 1, T, E]
        ttnn.deallocate(logits)
        _, idx = ttnn.topk(ttnn.add(score, self._bias_exact), k=k, dim=-1, largest=True, sorted=True)  # [1, 1, T, k]
        onehot = ttnn.eq(ttnn.reshape(ttnn.typecast(idx, ttnn.float32), [T, 1, k, 1]), self._arange)  # [T, 1, k, E]
        sel = ttnn.sum(ttnn.multiply(onehot, ttnn.reshape(score, [T, 1, 1, E])), dim=-1, keepdim=True)  # [T, 1, k, 1]
        ttnn.deallocate(score)
        denom = ttnn.add(ttnn.sum(sel, dim=2, keepdim=True), 1e-20)
        w = ttnn.typecast(ttnn.multiply(ttnn.div(sel, denom), self.scaling_factor / ROUTED_GAIN), ttnn.bfloat16)
        weights = ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k))
        indices = ttnn.view(ttnn.to_layout(ttnn.typecast(idx, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, k))
        return weights, indices

    def _forward_exact_fast2(self, tt_x: ttnn.Tensor, exact_select=None, l1: bool = True, grid=(1, 8)):
        """Same expert ids as ``_forward_exact_fast`` (bit-identical rank values -> identical top-k incl. ties) and routing
        weights equal to it within one bf16 ulp, ~35-50 us cheaper per call (measured in a trace, T = 4..32) by cutting dependent ops:

        * softplus -> sqrt (-> + bias) are lhs activations of one binary add each (3 us, vs ~5 us per unary op);
        * the select of the 6 chosen scores is ONE batched matmul ``score[T,1,1,E] @ onehot[T,1,E,k]`` (fp32; one-hot x score has a
          single non-zero product per output, the only error is the matmul unpacker's tf32 truncation of the score, < 2^-10
          relative, which flips a bf16 rounding of ~3% of the weights by one ulp) and the normaliser ``sum(sel)`` is a second
          matmul with a ones[k,k] matrix (``ttnn.sum`` costs 12-15 us, the matmul 3);
        * ``sel / (den + 1e-20) * scale`` -> bf16 is ONE div (eps as b-activation, scale as post-activation, bf16 output);
        * intermediates live in L1 (freed on return; the returned tensors are in DRAM like those of the old path).

        ``exact_select=True`` selects with multiply + ``ttnn.sum`` in the [1,k,T,E] orientation instead (bit-identical weights;
        15-25 us slower at T<=16, the faster one at T=32; default: used for T > 16). ``grid`` is the matmul core grid (y, x):
        1x8 gives bit-identical logits to 2x8 and is 1-3 us faster.
        """
        T, E, k = tt_x.shape[2], self._E, self.k
        if exact_select is None:
            exact_select = T > 16
        mc = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
        U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
        acts = [U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)]
        logits = ttnn.matmul(
            tt_x,
            self._w_exact,
            compute_kernel_config=self._ckc_exact,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]),
            memory_config=mc,
        )  # [1,1,T,E] fp32
        score = ttnn.add(
            logits, self._zero_exact, input_tensor_a_activations=acts, memory_config=mc
        )  # sqrt(softplus(logits))
        ttnn.deallocate(logits)
        rank = ttnn.add(score, self._bias_exact, memory_config=mc)
        _, idx = ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True, memory_config=mc)  # [1,1,T,k] uint32
        ttnn.deallocate(rank)
        idxf = ttnn.typecast(idx, ttnn.float32, memory_config=mc)
        scale = self.scaling_factor / ROUTED_GAIN
        div_kw = dict(
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, scale)],
            dtype=ttnn.bfloat16,
            memory_config=mc,
        )
        if not exact_select:
            onehot = ttnn.eq(self._arange_col, ttnn.reshape(idxf, [T, 1, 1, k]), memory_config=mc)  # [T,1,E,k]
            sel = ttnn.matmul(
                ttnn.reshape(score, [T, 1, 1, E]),
                onehot,
                compute_kernel_config=self._ckc_exact,
                dtype=ttnn.float32,
                memory_config=mc,
            )  # [T,1,1,k]
            den = ttnn.matmul(
                sel, self._ones_k, compute_kernel_config=self._ckc_exact, dtype=ttnn.float32, memory_config=mc
            )
            w = ttnn.div(sel, den, **div_kw)
            weights = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [T,1,1,k]
        else:
            onehot = ttnn.eq(
                ttnn.permute(idxf, (0, 3, 2, 1), memory_config=mc), self._arange, memory_config=mc
            )  # [1,k,T,E]
            sel = ttnn.sum(
                ttnn.multiply(onehot, score, memory_config=mc), dim=-1, keepdim=True, memory_config=mc
            )  # [1,k,T,1]
            den = ttnn.sum(sel, dim=1, keepdim=True, memory_config=mc)  # [1,1,T,1]
            w = ttnn.permute(ttnn.div(sel, den, **div_kw), (0, 3, 2, 1), memory_config=mc)  # [1,1,T,k]
            weights = ttnn.view(
                ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG), (T, 1, 1, k)
            )
        indices = ttnn.view(
            ttnn.to_layout(
                ttnn.typecast(idx, ttnn.uint16, memory_config=mc),
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            (T, 1, 1, k),
        )
        return weights, indices

    def _forward_fused(self, tt_x: ttnn.Tensor, grid=(1, 8)):
        """DSV41_ROUTER=fused: fp32 matmul, score = sqrt(softplus) and rank = score + bias as in ``_forward_exact_fast2``
        (bit-identical rank values), then ONE JIT program (``router_select``: one core per token) replaces ``ttnn.topk`` (single
        core, 63-68 us), the one-hot select, the normalisation and the layout/typecast ops: exact fp32 top-k (lowest index wins
        ties), fp32 weights s_i / (sum s + 1e-20) * scale -> bf16, written as the RM [T,1,1,k] weights / uint16 indices.
        """
        from models.demos.blackhole.deepseek_v41_flash.tt.router_select import router_select

        T, E, k = tt_x.shape[2], self._E, self.k
        L1 = ttnn.L1_MEMORY_CONFIG
        U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
        logits = ttnn.matmul(
            tt_x,
            self._w_exact,
            compute_kernel_config=self._ckc_exact,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]),
            memory_config=L1,
        )
        score = ttnn.add(
            logits,
            self._zero_exact,
            input_tensor_a_activations=[U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)],
            memory_config=L1,
        )
        ttnn.deallocate(logits)
        rank = ttnn.add(score, self._bias_exact, memory_config=L1)
        weights, indices = router_select(rank, score, k, 1e-20, self.scaling_factor / ROUTED_GAIN)
        ttnn.deallocate(rank)
        ttnn.deallocate(score)
        return weights, indices

    def _forward_exact(self, tt_x: ttnn.Tensor):
        logits = ttnn.matmul(tt_x, self.tt_gate_weight, compute_kernel_config=self._ckc_exact, dtype=ttnn.float32)
        score = ttnn.sqrt(ttnn.softplus(logits, beta=1.0, threshold=20.0))  # fp32 [1, 1, B, padded]
        ttnn.deallocate(logits)
        rank = ttnn.add(score, self._tt_rank_bias)
        _, idx = ttnn.topk(rank, k=self.k, dim=-1, largest=True, sorted=True)  # [1, 1, B, k]
        ttnn.deallocate(rank)
        total_batch = score.shape[2]
        idx32 = ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.TILE_LAYOUT)
        sel = ttnn.gather(score, 3, index=idx32)  # unbiased fp32 scores at the chosen experts
        ttnn.deallocate(score)
        denom = ttnn.add(ttnn.sum(sel, dim=3, keepdim=True), 1e-20)
        w = ttnn.typecast(ttnn.multiply(ttnn.div(sel, denom), self.scaling_factor / ROUTED_GAIN), ttnn.bfloat16)
        weights = ttnn.view(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (total_batch, 1, 1, self.k))
        indices = ttnn.view(
            ttnn.to_layout(ttnn.typecast(idx, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), (total_batch, 1, 1, self.k)
        )
        return weights, indices

    def _forward_kernel(self, tt_x: ttnn.Tensor):
        assert (
            self._kernel_tensors
        ), "DSV41_ROUTER=kernel must be set when the gate is built (its L1 tensors are freed otherwise)"
        # 1) router logits and scores in fp32
        logits = ttnn.matmul(
            tt_x,
            self.tt_gate_weight,
            compute_kernel_config=self.compute_kernel_config,
            program_config=self.matmul_program_config,
            dtype=ttnn.float32,
        )
        score = ttnn.sqrt(ttnn.softplus(logits, beta=1.0, threshold=20.0))  # fp32 [1, 1, B, padded]
        ttnn.deallocate(logits)
        # 2) folded score for the op: bf16(score + b_lo); the bf16 bias b_hi is added inside the op
        folded = ttnn.typecast(ttnn.add(score, self._tt_b_lo), ttnn.bfloat16)

        total_batch = folded.shape[2]
        chunk = min(self.num_device_cores, self._buffer_rows)
        num_iters = (total_batch + chunk - 1) // chunk
        padding = (num_iters - (total_batch % num_iters)) % num_iters
        batch_per_iter = (total_batch + padding) // num_iters
        if padding:
            folded = ttnn.pad(folded, [(0, 0), (0, 0), (0, padding), (0, 0)], 0)

        mem_in = self._sharded_mem_config(batch_per_iter, (self.num_blocks * 32, 32))
        mem_out = self._sharded_mem_config(batch_per_iter, (32, 32))
        shape = (batch_per_iter, self.num_blocks, 16, 16)
        bias = ttnn.slice(self.tt_bias, [0, 0, 0, 0], [batch_per_iter, self.num_blocks, 16, 16], memory_config=mem_in)
        in_idx = ttnn.slice(
            self.tt_input_indices, [0, 0, 0, 0], [batch_per_iter, self.num_blocks, 16, 16], memory_config=mem_in
        )
        out = ttnn.slice(self.tt_output, [0, 0, 0], [batch_per_iter, 32, 32], memory_config=mem_out)
        out_idx = ttnn.slice(self.tt_output_indices, [0, 0, 0], [batch_per_iter, 32, 32], memory_config=mem_out)

        idx_chunks = []
        for start in range(0, total_batch + padding, batch_per_iter):
            cur = ttnn.reshape(folded[:, :, start : start + batch_per_iter, :], shape)
            cur = ttnn.to_memory_config(cur, memory_config=mem_in)
            _, idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(
                cur,
                bias_tensor=bias,
                input_indices_tensor=in_idx,
                output_tensor=out,
                output_indices_tensor=out_idx,
                eps=self.eps,
                scaling_factor=self.scaling_factor,
                enable_sigmoid=self.enable_sigmoid,
                topk=self.k,
                output_softmax=self.output_softmax,
            )
            idx_chunks.append(ttnn.to_memory_config(idx, memory_config=ttnn.L1_MEMORY_CONFIG))
            ttnn.deallocate(cur)
        ttnn.deallocate(folded)
        indices = idx_chunks[0] if num_iters == 1 else ttnn.concat(idx_chunks, dim=0)
        indices = ttnn.view(indices[:total_batch, 0, : self.k], (total_batch, 1, 1, self.k))  # uint16

        # 3) weights from the UNFOLDED fp32 score at the selected experts, renormalised and scaled
        idx32 = ttnn.typecast(ttnn.reshape(indices, (1, 1, total_batch, self.k)), ttnn.uint32)
        idx32 = ttnn.to_layout(idx32, ttnn.TILE_LAYOUT)
        sel = ttnn.gather(score, 3, index=idx32)  # fp32 [1, 1, B, k]
        ttnn.deallocate(score)
        denom = ttnn.add(ttnn.sum(sel, dim=3, keepdim=True), 1e-20)
        w = ttnn.multiply(ttnn.div(sel, denom), self.scaling_factor / ROUTED_GAIN)
        w = ttnn.typecast(w, ttnn.bfloat16)
        weights = ttnn.view(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (total_batch, 1, 1, self.k))
        return weights, indices
