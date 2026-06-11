# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTMoEGate — the routing front-end that feeds ``TTMoEDecode``.

``TTMoEDecode.forward`` consumes a pre-computed routing decision per token
(``tt_scores``/``tt_indices`` of shape ``[1, 1, batch, select_experts_k]``); this
module PRODUCES it from the router logits:

    hidden ──[router matmul: ttnn.matmul(h, W_gate)]──▶ logits[1,1,batch,num_experts]
           ──[reshape→16×16 height-shard, one token/core]──▶ gate-op input
           ──[generalized_moe_gate: (sigmoid|softmax)+top-k+normalize]──▶ (scores,indices)[batch,32,32]
           ──[slice :k + view]──▶ (weights, indices)[1,1,batch,k]

The device flow mirrors ``models/demos/deepseek_v3/tt/moe_gate.py`` (decode path):
the gate op is single-token-per-core, so a variable ``batch`` is padded up to a
multiple of the core count and processed in ``batch_per_iter`` chunks, reusing the
bias / input-indices / output buffers. External tensors are plain L1-interleaved;
the height-sharded layout the op needs is produced inside ``forward``.

Gate-op selection (via ``n_group``):
  - ``n_group=1`` → ``generalized_moe_gate`` (ungrouped global top-k; configurable ``topk`` +
    sigmoid/softmax). Softmax-no-bias models use ``enable_sigmoid=False, output_softmax=True``
    (softmax-over-selected ≡ softmax-front + renormalize, since the global Z cancels).
  - ``n_group=8`` → ``deepseek_moe_gate`` (grouped top-8; sigmoid + bias). Only the 256-expert
    select-8 case is allowed.

Scope: ``num_experts`` ≤ 512. ≤256 runs on one 256-face (fewer experts padded up with phantom
``_PAD_NEG`` experts that rank last); 257-512 uses the op's 2-block combine path (generalized op only).
``n_group=8`` (deepseek grouped) is single-256-block only.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3.tt.deepseek_moe_gate.op import DeepseekMoeGateOp
from models.demos.deepseek_v3.tt.generalized_moe_gate.op import GeneralizedMoeGateOp

_TOKEN_SHAPE = (16, 16)  # 256 experts laid out as a 16×16 face per token
_SHARD_SHAPE = (32, 32)  # one 32×32 tile per core (the op's per-token shard)
_FACE_EXPERTS = _TOKEN_SHAPE[0] * _TOKEN_SHAPE[1]  # 256: the op's fixed face size; <256 models pad up to it
# Phantom (padding) experts get this large-negative value in BOTH the logit and the bias. Selection ranks
# by `score + bias` (score = logit for softmax, sigmoid(logit) for sigmoid); driving the bias to -1e9 makes
# phantom experts rank last in EITHER mode (a -1e9 logit alone isn't enough for sigmoid, since sigmoid(-1e9)=0
# could still sit above a real expert with a negative bias). They never enter the top-k, so they never reach
# the output weights or the softmax exp.
_PAD_NEG = -1e9


class TTMoEGate:
    """Produce ``(weights, indices)`` for ``TTMoEDecode`` from hidden states."""

    def __init__(
        self,
        mesh_device,
        *,
        num_experts: int,
        select_experts_k: int,
        hidden_size: int,
        torch_gate_weight: torch.Tensor,  # [hidden_size, num_experts] router projection
        # TWO distinct biases (both [num_experts], both optional):
        #   torch_gate_bias      = score-CORRECTION bias (deepseek/sigmoid): added INSIDE the op to the
        #                          scores for SELECTION ranking only; the output weights stay unbiased.
        #   torch_gate_proj_bias = router LINEAR bias (gpt-oss): added to the LOGITS (Wx + b) BEFORE the op,
        #                          so it flows into BOTH selection AND the softmax/normalized weights.
        torch_gate_bias: torch.Tensor | None = None,
        torch_gate_proj_bias: torch.Tensor | None = None,
        n_group: int = 1,  # 1 → generalized (ungrouped) op; 8 → deepseek (grouped) op
        score_func: str = "softmax",  # "softmax" (no bias) | "sigmoid" | "sqrtsoftplus" (deepseek-v4)
        scaling_factor: float = 1.0,  # routed_scaling_factor applied after normalize
        eps: float = 1e-20,
        matmul_compute_config: dict | object | None = None,  # router-matmul compute kernel config (per-model, optional)
    ):
        # `score_func` maps to (a) the score nonlinearity and (b) two op flags -- enable_sigmoid (apply
        # sigmoid inside the op) and output_softmax (softmax over the selected top-k vs linear renorm):
        #   "softmax"      → score=logit,             enable_sigmoid=False, output_softmax=True   (no bias)
        #   "sigmoid"      → score=sigmoid(logit),    enable_sigmoid=True,  output_softmax=False  (+bias)
        #   "sqrtsoftplus" → score=sqrt(softplus(l)), enable_sigmoid=False, output_softmax=False  (+bias)
        # sqrtsoftplus (deepseek-v4) has no in-kernel op, so it is applied EXTERNALLY in forward() via
        # ttnn.sqrt(ttnn.softplus(.)) and fed to the op with enable_sigmoid=False (the op then just adds
        # bias, ranks, and linearly renormalizes -- same tail as the sigmoid path).
        if score_func not in ("softmax", "sigmoid", "sqrtsoftplus"):
            raise ValueError(f"score_func must be 'softmax' | 'sigmoid' | 'sqrtsoftplus'; got {score_func!r}")
        # n_group picks the backing op: 1 = generalized_moe_gate (configurable top-k, ungrouped),
        # 8 = deepseek_moe_gate (grouped top-8). Only these two are wired today.
        if n_group not in (1, 8):
            raise NotImplementedError(f"n_group must be 1 (generalized) or 8 (deepseek); got {n_group}")
        if n_group == 8 and not (num_experts == 256 and select_experts_k == 8):
            raise NotImplementedError(
                f"n_group==8 (deepseek_moe_gate) only supports 256 experts select-8 (got N={num_experts}, k={select_experts_k})"
            )
        # generalized_moe_gate's rank-mask only covers k=4/6/8 (drop threshold = 16*(k-4)). Ungrouped cases
        # outside that (e.g. qwen3.5: top-10) take a pure-ttnn FALLBACK (matmul → topk → softmax, see
        # _forward_fallback) instead of the kernel op. The grouped (n_group=8) op has no fallback.
        use_fallback = (n_group == 1) and (select_experts_k not in (4, 6, 8))
        # The op works in 256-expert blocks (a 16×16 face). ≤256 experts pad up to ONE block; 257-512 use the
        # op's 2-block COMBINE path (each block → a re-mergeable top-k run → global top-k over both). Either
        # way the phantom experts (num_experts..num_blocks*256) get a large-negative logit + bias (_PAD_NEG)
        # so they rank last. The grouped (n_group=8) op is single-block only.
        if not (1 <= num_experts <= 2 * _FACE_EXPERTS):
            raise NotImplementedError(
                f"TTMoEGate supports up to {2 * _FACE_EXPERTS} experts (1-2 blocks); got {num_experts}"
            )
        if num_experts > _FACE_EXPERTS and n_group != 1:
            raise NotImplementedError(
                f"num_experts>{_FACE_EXPERTS} uses the generalized combine path (n_group=1); got n_group={n_group}"
            )

        self.mesh_device = mesh_device
        self.num_experts = num_experts
        self.num_blocks = (num_experts + _FACE_EXPERTS - 1) // _FACE_EXPERTS  # 1 (≤256) or 2 (257-512)
        self._padded_experts = self.num_blocks * _FACE_EXPERTS  # 256 or 512
        self.k = select_experts_k
        self.hidden_size = hidden_size
        self.n_group = n_group
        self.score_func = score_func
        self.scaling_factor = scaling_factor
        self.eps = eps
        self.enable_sigmoid = score_func == "sigmoid"
        self.output_softmax = score_func == "softmax"
        # external score transform applied to the logits before the op (None → the op's own scoring).
        self.score_transform = "sqrtsoftplus" if score_func == "sqrtsoftplus" else None
        self.use_fallback = use_fallback  # ungrouped + k∉{4,6,8} → pure-ttnn path (no kernel op)

        grid = mesh_device.compute_with_storage_grid_size()
        self.num_device_cores = grid.x * grid.y
        self._grid = grid

        self.compute_kernel_config = self._matmul_compute_config(matmul_compute_config)

        # --- router weight (replicated, DRAM) ---
        self.tt_gate_weight = ttnn.from_torch(
            torch_gate_weight,  # [hidden, num_experts]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- router LINEAR bias (gpt-oss): logits = Wx + b, added by ttnn.linear in forward (None → matmul) ---
        self.tt_gate_proj_bias = None
        if torch_gate_proj_bias is not None:
            self.tt_gate_proj_bias = ttnn.from_torch(
                torch_gate_proj_bias.reshape(1, num_experts),  # [1, num_experts] (ttnn.linear bias)
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # --- fallback path: ttnn.topk works on the full logits directly, so none of the op's face/combine/
        #     output buffers are needed. Build only the (optional) score-correction bias for selection
        #     ([1, num_experts] broadcast-add), then return. ---
        if self.use_fallback:
            self.tt_fallback_bias = None
            if torch_gate_bias is not None:
                self.tt_fallback_bias = ttnn.from_torch(
                    torch_gate_bias.reshape(1, num_experts).to(torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            return

        # --- reusable per-core buffers sized to num_device_cores (sliced to batch in forward) ---
        cores = self.num_device_cores
        mem = self._sharded_mem_config(cores)

        # bias + input-indices buffers. Real experts [0:num_experts] get torch_gate_bias (or 0 for no-bias —
        # a constant shift changes neither selection nor output, so zeros is the correct no-bias case);
        # phantom experts [num_experts:num_blocks*256] get _PAD_NEG so they always rank last (see _PAD_NEG).
        real_bias = (torch_gate_bias if torch_gate_bias is not None else torch.zeros(num_experts)).to(torch.float32)
        bias = torch.full((self._padded_experts,), _PAD_NEG, dtype=torch.float32)
        bias[:num_experts] = real_bias
        if self.num_blocks == 1:
            # single 256-face: (1,16,16) → pad to (1,32,32) → transpose → per core. indices = arange(256).
            bias_face = torch.nn.functional.pad(bias.reshape(1, 16, 16), (0, 16, 0, 16, 0, 0), "constant", 0).transpose(
                1, 2
            )  # (1, 32, 32)
            self.tt_bias = ttnn.from_torch(
                bias_face.repeat(cores, 1, 1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mem,
            )
            idx = torch.arange(_FACE_EXPERTS, dtype=torch.int32).reshape(1, 16, 16).transpose(1, 2)
            self.tt_input_indices = ttnn.from_torch(
                idx.repeat(cores, 1, 1),
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mem,
            )
        else:
            # combine path (num_blocks×256): num_blocks 16×16 faces stacked, each block transposed; indices
            # carry GLOBAL expert ids (block b = arange(256)+b*256). Shard (num_blocks*32, 32), tile (32,32).
            # Mirrors test_generalized_moe_gate.py::test_generalized_moe_gate_512_global.
            tile = ttnn.Tile((32, 32))
            mem_blocks = self._sharded_mem_config(cores, (self.num_blocks * 32, 32))
            bias_blocks = torch.transpose(bias.reshape(self.num_blocks, 16, 16), -2, -1).contiguous()  # (nb,16,16)
            self.tt_bias = ttnn.from_torch(
                bias_blocks.unsqueeze(0).repeat(cores, 1, 1, 1),  # (cores, nb, 16, 16)
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mem_blocks,
                tile=tile,
            )
            ar = torch.arange(_FACE_EXPERTS, dtype=torch.int32).reshape(1, 16, 16)
            offs = (torch.arange(self.num_blocks, dtype=torch.int32) * _FACE_EXPERTS).reshape(self.num_blocks, 1, 1)
            idx_blocks = torch.transpose(ar + offs, -2, -1).contiguous().to(torch.int32)  # (nb, 16, 16) global ids
            self.tt_input_indices = ttnn.from_torch(
                idx_blocks.unsqueeze(0).repeat(cores, 1, 1, 1),
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=mem_blocks,
                tile=tile,
            )

        # preallocated output buffers (filled in place by the op), per core. ROW_MAJOR (like moe_gate.py)
        # so the final ttnn.view is metadata-only — a TILE buffer routes view through the device
        # reshape_view op, which rejects uint16.
        self.tt_output = ttnn.from_torch(
            torch.zeros((cores, *_SHARD_SHAPE), dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mem,
        )
        self.tt_output_indices = ttnn.from_torch(
            torch.zeros((cores, *_SHARD_SHAPE), dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mem,
        )

    # ------------------------------------------------------------------ helpers
    def _matmul_compute_config(self, cfg):
        """Build the router-matmul ``compute_kernel_config`` from a per-model dict (or pass through an
        already-built config / ``None``).

        The gate matmul is tiny but its reduction is ``hidden_size``-deep; on a deep, tie-sensitive gate
        (e.g. deepseek 7168→256, grouped top-8) the DEFAULT matmul (LoFi + bf16 accumulate) builds up
        enough error to flip near-tied experts at the top-k boundary (output weights stay ~right, but the
        selected expert *ids* drift). There is no universal setting — deepseek_v3 uses HiFi2+fp32-accumulate,
        mixtral uses LoFi, gpt_oss keeps the default — so this is per-model config (yaml ``gate_matmul_compute``)
        rather than hardcoded. Built device-agnostically (``init_device_compute_kernel_config`` selects the
        right config class per arch), unlike a hardcoded ``WormholeComputeKernelConfig`` (wrong on Blackhole).

          - ``None``        → ttnn's default matmul fidelity (fine for shallow / tie-insensitive gates).
          - ``dict``        → keys ``math_fidelity`` ("HiFi2"|…), ``math_approx_mode``, ``fp32_dest_acc_en``,
                              ``packer_l1_acc`` (each optional; the defaults below suit a deep, tie-sensitive gate).
          - ttnn config     → used as-is.
        """
        if cfg is None or not isinstance(cfg, dict):
            return cfg
        cfg = dict(cfg)
        fidelity = cfg.pop("math_fidelity", "HiFi2")
        if isinstance(fidelity, str):
            fidelity = getattr(ttnn.MathFidelity, fidelity)
        return ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=fidelity,
            math_approx_mode=cfg.pop("math_approx_mode", True),
            fp32_dest_acc_en=cfg.pop("fp32_dest_acc_en", True),
            packer_l1_acc=cfg.pop("packer_l1_acc", True),
        )

    def _sharded_mem_config(self, num_cores: int, shard_shape: tuple = _SHARD_SHAPE) -> ttnn.MemoryConfig:
        # one token per core; shard_shape is (32,32) for a single 256-block, (num_blocks*32, 32) for the
        # combine path (num_blocks tiles stacked per core).
        core_grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(self._grid.x, self._grid.y), row_wise=True)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    # ------------------------------------------------------------------ device
    def forward(self, tt_x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``tt_x``: hidden states ``[1, 1, batch, hidden_size]`` (L1/DRAM interleaved) →
        ``(weights, indices)`` each ``[1, 1, batch, select_experts_k]``."""
        if self.use_fallback:
            return self._forward_fallback(tt_x)
        # 1) router projection → logits [1, 1, batch, num_experts]. With a router LINEAR bias (gpt-oss) it's
        #    Wx + b via ttnn.linear; otherwise the proven bias-free matmul (unchanged for every other model).
        #    compute_kernel_config is per-model (see __init__): a deep, tie-sensitive gate wants HiFi2 + fp32.
        if self.tt_gate_proj_bias is None:
            logits = ttnn.matmul(tt_x, self.tt_gate_weight, compute_kernel_config=self.compute_kernel_config)
        else:
            logits = ttnn.linear(
                tt_x, self.tt_gate_weight, bias=self.tt_gate_proj_bias, compute_kernel_config=self.compute_kernel_config
            )

        # 1a) external score transform (sqrtsoftplus, deepseek-v4): score = sqrt(softplus(logit)). Applied
        #     here (no in-kernel op) and fed to the gate op with enable_sigmoid=False; the op then adds bias,
        #     ranks, and linearly renormalizes. Done before the phantom-pad so phantom logits stay _PAD_NEG.
        if self.score_transform == "sqrtsoftplus":
            # standard softplus log(1+e^x): beta=1, threshold=20 (matches torch's defaults / the golden).
            logits = ttnn.sqrt(ttnn.softplus(logits, beta=1.0, threshold=20.0))

        # 1b) pad the experts dim up to num_blocks×256 (256 single-face, or 512 for the 2-block combine).
        #     Phantom experts get _PAD_NEG so they rank last (their bias in tt_bias is also _PAD_NEG); they
        #     never enter the top-k.
        if self.num_experts < self._padded_experts:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, self._padded_experts - self.num_experts)], _PAD_NEG)
        total_batch = logits.shape[2]

        # 2) the gate op is one-token-per-core: pad batch to a multiple of the core count and
        #    process in equal chunks, reusing the bias/indices/output buffers. (Mirrors moe_gate.py.)
        cores = self.num_device_cores
        num_iters = (total_batch + cores - 1) // cores
        padding = (num_iters - (total_batch % num_iters)) % num_iters
        batch_per_iter = (total_batch + padding) // num_iters
        if padding != 0:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, padding), (0, 0)], 0)

        # input/bias/indices live on (num_blocks*32, 32) shards (1 token/core, num_blocks tiles); the output
        # is always a single (32,32) tile per token (the combine writes the final global top-k into row 0).
        mem_in = self._sharded_mem_config(batch_per_iter, (self.num_blocks * 32, 32))
        mem_out = self._sharded_mem_config(batch_per_iter, _SHARD_SHAPE)
        if self.num_blocks == 1:
            reshaped_shape = (batch_per_iter, *_TOKEN_SHAPE)  # (bpi, 16, 16)
            bias = ttnn.slice(self.tt_bias, [0, 0, 0], [batch_per_iter, 16, 16], memory_config=mem_in)
            in_idx = ttnn.slice(self.tt_input_indices, [0, 0, 0], [batch_per_iter, 16, 16], memory_config=mem_in)
        else:
            reshaped_shape = (batch_per_iter, self.num_blocks, *_TOKEN_SHAPE)  # (bpi, nb, 16, 16)
            bias = ttnn.slice(
                self.tt_bias, [0, 0, 0, 0], [batch_per_iter, self.num_blocks, 16, 16], memory_config=mem_in
            )
            in_idx = ttnn.slice(
                self.tt_input_indices, [0, 0, 0, 0], [batch_per_iter, self.num_blocks, 16, 16], memory_config=mem_in
            )
        out = ttnn.slice(self.tt_output, [0, 0, 0], [batch_per_iter, 32, 32], memory_config=mem_out)
        out_idx = ttnn.slice(self.tt_output_indices, [0, 0, 0], [batch_per_iter, 32, 32], memory_config=mem_out)

        weights_chunks, indices_chunks = [], []
        for start in range(0, total_batch + padding, batch_per_iter):
            cur = logits[:, :, start : start + batch_per_iter, :]
            cur = ttnn.reshape(cur, reshaped_shape)  # (bpi,16,16) or (bpi,num_blocks,16,16) for combine
            cur = ttnn.to_memory_config(cur, memory_config=mem_in)  # height-shard: one token/core

            if self.n_group == 8:
                # deepseek grouped top-8 (sigmoid + bias); no top-k / output-softmax knobs.
                w, idx = DeepseekMoeGateOp.op(
                    cur, bias, out, in_idx, out_idx, self.eps, self.scaling_factor, self.enable_sigmoid
                )
            else:
                w, idx = GeneralizedMoeGateOp.op(
                    cur,
                    bias,
                    out,
                    in_idx,
                    out_idx,
                    self.eps,
                    self.scaling_factor,
                    self.enable_sigmoid,
                    self.k,
                    self.output_softmax,
                )
            weights_chunks.append(ttnn.to_memory_config(w, memory_config=ttnn.L1_MEMORY_CONFIG))
            indices_chunks.append(ttnn.to_memory_config(idx, memory_config=ttnn.L1_MEMORY_CONFIG))
            ttnn.deallocate(cur)
        ttnn.deallocate(logits)

        weights = weights_chunks[0] if num_iters == 1 else ttnn.concat(weights_chunks, dim=0)
        indices = indices_chunks[0] if num_iters == 1 else ttnn.concat(indices_chunks, dim=0)

        # 3) take the top-k (ranks 0..k-1 are the first k cols of row 0) → [1,1,batch,k].
        # ROW_MAJOR output buffers make these views metadata-only (no device reshape_view), so the
        # uint16 indices reshape without a dtype round-trip — same as moe_gate.py.
        weights = ttnn.view(weights[:total_batch, 0, : self.k], (1, 1, total_batch, self.k))
        indices = ttnn.view(indices[:total_batch, 0, : self.k], (1, 1, total_batch, self.k))
        return weights, indices

    def _forward_fallback(self, tt_x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Pure-ttnn ungrouped gate for cases the kernel op can't do (k∉{4,6,8}, e.g. qwen3.5 top-10).
        Mirrors the op's logic with stock ttnn ops, no 256-face / combine / padding: matmul → score
        transform → (+ score-correction bias for selection) → topk → gather unbiased scores → normalize
        (softmax-over-selected if output_softmax, else linear renorm) at the END → × scale. Output
        ``(weights, indices)`` each ``[1, 1, batch, k]``, indices uint16 (like the op path)."""
        # 1) logits = Wx (+ router LINEAR bias). compute_kernel_config per-model (HiFi2 for deep gates).
        if self.tt_gate_proj_bias is None:
            logits = ttnn.matmul(tt_x, self.tt_gate_weight, compute_kernel_config=self.compute_kernel_config)
        else:
            logits = ttnn.linear(
                tt_x, self.tt_gate_weight, bias=self.tt_gate_proj_bias, compute_kernel_config=self.compute_kernel_config
            )

        # 2) score = transform(logits): sqrt(softplus) | sigmoid | identity(softmax ranks by raw logit).
        if self.score_transform == "sqrtsoftplus":
            scores = ttnn.sqrt(ttnn.softplus(logits, beta=1.0, threshold=20.0))
        elif self.enable_sigmoid:
            scores = ttnn.sigmoid(logits)
        else:
            scores = logits

        # 3) add the score-CORRECTION bias for SELECTION ranking (None → skip; qwen3.5 is bias-free).
        rank_key = scores if self.tt_fallback_bias is None else ttnn.add(scores, self.tt_fallback_bias)

        # 4) top-k over ALL experts → [1, 1, batch, k]. ttnn.topk handles any k / expert count, ungrouped.
        topk_vals, topk_idx = ttnn.topk(rank_key, self.k, dim=-1)

        # 5) weights = the UNBIASED scores at the selected experts. With a correction bias the topk values are
        #    the BIASED ranking key, so gather the unbiased scores; without one they're already the scores.
        sel = topk_vals if self.tt_fallback_bias is None else ttnn.gather(scores, dim=-1, index=topk_idx)

        # 6) normalize at the END (unified): softmax-over-selected (output_softmax) or linear renorm; × scale.
        if self.output_softmax:
            weights = ttnn.softmax(sel, dim=-1)
        else:
            weights = ttnn.div(sel, ttnn.sum(sel, dim=-1, keepdim=True))
        if self.scaling_factor != 1.0:
            weights = ttnn.multiply(weights, self.scaling_factor)

        indices = ttnn.typecast(topk_idx, ttnn.uint16)
        return weights, indices

    # ------------------------------------------------------------------ golden
    @staticmethod
    def golden(
        hidden: torch.Tensor,  # [batch, hidden]
        gate_weight: torch.Tensor,  # [hidden, num_experts]
        gate_bias: torch.Tensor | None,  # [num_experts] score-correction bias (selection-only) or None
        *,
        select_experts_k: int,
        score_func: str = "softmax",
        scaling_factor: float = 1.0,
        eps: float = 1e-20,
        n_group: int = 1,
        gate_proj_bias: torch.Tensor | None = None,  # [num_experts] router LINEAR bias (gpt-oss): logits = Wx + b
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch reference: hidden → logits → gate → (scores[batch,k], indices[batch,k]),
        delegating to the matching op golden (``DeepseekMoeGateOp`` for n_group==8 grouped top-8,
        ``GeneralizedMoeGateOp`` for n_group==1 ungrouped top-k)."""
        batch = hidden.shape[0]
        logits = hidden.float() @ gate_weight.float()  # [batch, num_experts]
        # router LINEAR bias (gpt-oss) is part of the projection: logits = Wx + b. It flows into BOTH
        # selection and the softmax weights (unlike the score-correction `gate_bias`, which is selection-only).
        if gate_proj_bias is not None:
            logits = logits + gate_proj_bias.float()
        bias = gate_bias.float() if gate_bias is not None else torch.zeros(logits.shape[-1])
        enable_sigmoid = score_func == "sigmoid"

        # external score transform (sqrtsoftplus, deepseek-v4): mirror forward() — apply sqrt∘softplus up
        # front and feed the op golden with enable_sigmoid=False, so it just adds bias / ranks / renorms.
        if score_func == "sqrtsoftplus":
            logits = torch.sqrt(torch.nn.functional.softplus(logits))

        if n_group == 8:
            # deepseek grouped top-8: 8 groups of 32 experts (top-2-sum per group → top-4 groups → top-8).
            # The golden's grouping is driven by shape[-2], so reshape to the LOGICAL (8, 32) — NOT the
            # (16, 16) tile face (that's only the device's physical layout; the kernel still groups the 256
            # experts as 8×32). Bias likewise reshapes to (8, 32) (the device uploads it transposed in the
            # 16×16 tile, but the golden works in logical expert order). See the op unit test:
            # models/demos/deepseek_v3_b1/tests/unit_tests/test_deepseek_moe_gate.py (input_shape=(b,8,32)).
            return DeepseekMoeGateOp.golden(
                logits.reshape(batch, 8, 32),
                bias.reshape(8, 32),
                eps,
                scaling_factor,
                enable_sigmoid,
            )
        return GeneralizedMoeGateOp.golden(
            logits,
            bias.expand_as(logits),
            eps,
            scaling_factor,
            enable_sigmoid,
            select_experts_k,
            score_func == "softmax",  # output_softmax
        )
