# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2.6-Flash router (HF MiMoV2MoEGate semantics, verified in modeling_mimo_v2.py:135-184):

  logits = fp32(x) @ fp32(W).T ; score = sigmoid(logits) ; select top-8 of (score + e_score_correction_bias) ;
  weights = score at the selected experts (bias NOT used), normalised to sum 1 (+1e-20) ; * routed_scaling_factor
  (config null -> 1.0). n_group = topk_group = 1 (ungrouped), noaux_tc, no shared expert, no clamp.

MiMo selection biases are ~1.7-2.2 (L1), ~10.4 (L20), ~11.7 (L46) with std 0.02-0.05 across experts, vs bf16 ulp
0.0625 above 8: a plain bf16 gate may flip experts. Three device modes (same technique as DSV41Gate, sigmoid instead of
sqrtsoftplus):
  plain : TTMoEGate.forward (bf16 bias, in-op sigmoid)                              -- baseline to measure
  hilo  : fp32 matmul + fp32 sigmoid, bias = b_hi (bf16, in op) + b_lo folded into the fp32 score before the bf16 cast,
          weights recomputed from the unfolded fp32 score                           -- kernel path
  exact : fp32 matmul, sigmoid, bias add, ttnn.topk on fp32, fp32 gather            -- reference-quality device path
"""

import torch

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig

N_EXPERTS, TOPK, HIDDEN = 256, 8, 4096


def mimo_gate_config(batch_per_device=None) -> TTMoEGateConfig:
    return TTMoEGateConfig(
        num_routed_experts=N_EXPERTS,
        select_experts_k=TOPK,
        hidden_size=HIDDEN,
        batch_per_device=batch_per_device,
        n_group=1,
        score_func="sigmoid",
        routed_scaling_factor=1.0,
        score_correction_bias=True,
    )


def cpu_route(x, w_hf, bias, mode="fp32", shift=0.0):
    """CPU routing. x [T,H], w_hf [E,H] (HF layout), bias [E]. Returns (idx [T,k] sorted by rank, weights [T,k]).
    mode fp32 = HF reference. 'bf16_emul' = rough emulation of the plain bf16 op (bf16 logits/score/bias/sum), NOT a
    substitute for the device measurement."""
    if mode == "fp32":
        s = (x.float() @ w_hf.float().T).sigmoid()
        rank = s + bias.float()
    elif mode == "bf16_emul":
        s = (x.float() @ w_hf.float().T).to(torch.bfloat16).sigmoid().to(torch.bfloat16).float()
        rank = (s + (bias.float() - shift).to(torch.bfloat16).float()).to(torch.bfloat16).float()
    else:
        raise ValueError(mode)
    idx = torch.topk(rank, TOPK, dim=-1).indices
    sel = s.gather(1, idx)
    return idx, sel / (sel.sum(-1, keepdim=True) + 1e-20)


def calibrate_shift(x, w_hf, bias):
    """Layer constant K = mean 8th-ranked (score+bias) over the calibration tokens (ranking-invariant shift)."""
    rank = (x.float() @ w_hf.float().T).sigmoid() + bias.float()
    return float(torch.topk(rank, TOPK, dim=-1).values[:, -1].mean())


class MiMoGate(TTMoEGate):
    def __init__(self, mesh_device, torch_gate_weight_hf, torch_gate_bias, bias_shift=0.0, batch_per_device=None):
        """torch_gate_weight_hf: [E, H] (HF); torch_gate_bias: [E] fp32."""
        bias = torch_gate_bias.to(torch.float32) - float(bias_shift)
        b_hi = bias.to(torch.bfloat16).to(torch.float32)
        b_lo = bias - b_hi
        super().__init__(
            mesh_device,
            mimo_gate_config(batch_per_device),
            torch_gate_weight=torch_gate_weight_hf.T.contiguous(),
            torch_gate_bias=b_hi,
        )
        assert not self.use_fallback
        self._plain_enable_sigmoid = self.enable_sigmoid
        rep = ttnn.ReplicateTensorToMesh(mesh_device)

        def const(t):
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        P = self._padded_experts
        lo = torch.zeros(1, 1, 1, P)
        lo[0, 0, 0, :N_EXPERTS] = b_lo
        self._tt_b_lo = const(lo)
        rb = torch.full((1, 1, 1, P), -1e9)
        rb[0, 0, 0, :N_EXPERTS] = bias
        self._tt_rank_bias = const(rb)
        self._ckc32 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def route(self, tt_x, mode="hilo"):
        """tt_x [1,1,T,H] -> (weights [T,1,1,k], indices [T,1,1,k])."""
        if mode == "plain":
            self.enable_sigmoid = True
            return self.forward(tt_x)
        logits = ttnn.matmul(tt_x, self.tt_gate_weight, compute_kernel_config=self._ckc32, dtype=ttnn.float32)
        score = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)
        T = score.shape[2]
        if mode == "exact":
            rank = ttnn.add(score, self._tt_rank_bias)
            _, idx = ttnn.topk(rank, k=self.k, dim=-1, largest=True, sorted=True)
            ttnn.deallocate(rank)
            indices = ttnn.view(
                ttnn.to_layout(ttnn.typecast(idx, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, self.k)
            )
            idx32 = ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.TILE_LAYOUT)
        elif mode == "hilo":
            folded = ttnn.typecast(ttnn.add(score, self._tt_b_lo), ttnn.bfloat16)
            # reuse the base class op path with the sigmoid already applied: run the gate op with enable_sigmoid=False
            # and linear renorm (weights are recomputed below anyway; only the indices are used).
            self.enable_sigmoid = False
            try:
                indices = self._run_gate_op_on_scores(folded, T)
            finally:
                self.enable_sigmoid = self._plain_enable_sigmoid
            idx32 = ttnn.to_layout(
                ttnn.typecast(ttnn.reshape(indices, (1, 1, T, self.k)), ttnn.uint32), ttnn.TILE_LAYOUT
            )
        else:
            raise ValueError(mode)
        sel = ttnn.gather(score, 3, index=idx32)
        ttnn.deallocate(score)
        denom = ttnn.add(ttnn.sum(sel, dim=3, keepdim=True), 1e-20)
        w = ttnn.typecast(ttnn.multiply(ttnn.div(sel, denom), self.scaling_factor), ttnn.bfloat16)
        weights = ttnn.view(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (T, 1, 1, self.k))
        return weights, indices

    def _run_gate_op_on_scores(self, folded, T):
        """Chunked call of generalized_moe_gate on bf16 scores [1,1,T,P] (same buffers/sharding as
        TTMoEGate._forward_kernel in the DSV4 router). Returns uint16 indices [T,1,1,k]."""
        chunk = min(self.num_device_cores, self._buffer_rows)
        n_it = (T + chunk - 1) // chunk
        pad = (n_it - (T % n_it)) % n_it
        bpi = (T + pad) // n_it
        if pad:
            folded = ttnn.pad(folded, [(0, 0), (0, 0), (0, pad), (0, 0)], 0)
        nb = self.num_blocks
        mem_in = self._sharded_mem_config(bpi, (nb * 32, 32))
        mem_out = self._sharded_mem_config(bpi, (32, 32))
        shape = (bpi, nb, 16, 16) if nb > 1 else (bpi, 16, 16)
        sl = [bpi, nb, 16, 16] if nb > 1 else [bpi, 16, 16]
        z = [0] * len(sl)
        bias = ttnn.slice(self.tt_bias, z, sl, memory_config=mem_in)
        in_idx = ttnn.slice(self.tt_input_indices, z, sl, memory_config=mem_in)
        out = ttnn.slice(self.tt_output, [0, 0, 0], [bpi, 32, 32], memory_config=mem_out)
        out_idx = ttnn.slice(self.tt_output_indices, [0, 0, 0], [bpi, 32, 32], memory_config=mem_out)
        chunks = []
        for s in range(0, T + pad, bpi):
            cur = ttnn.to_memory_config(ttnn.reshape(folded[:, :, s : s + bpi, :], shape), memory_config=mem_in)
            _, idx = ttnn.experimental.deepseek.moe.generalized_moe_gate(
                cur,
                bias_tensor=bias,
                input_indices_tensor=in_idx,
                output_tensor=out,
                output_indices_tensor=out_idx,
                eps=self.eps,
                scaling_factor=self.scaling_factor,
                enable_sigmoid=False,
                topk=self.k,
                output_softmax=False,
            )
            chunks.append(ttnn.to_memory_config(idx, memory_config=ttnn.L1_MEMORY_CONFIG))
            ttnn.deallocate(cur)
        ttnn.deallocate(folded)
        indices = chunks[0] if n_it == 1 else ttnn.concat(chunks, dim=0)
        return ttnn.view(indices[:T, 0, : self.k], (T, 1, 1, self.k))
