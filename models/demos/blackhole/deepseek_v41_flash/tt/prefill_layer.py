# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4.1-Flash decoder layer for PREFILL, built around an existing decode ``DSV41Layer`` (weights shared).

The token-wise blocks (mHC, router, MoE, shared expert) run on chunks of ``T`` = 32 consecutive tokens of the mesh row (the
decode kernels are verified up to 32 tokens per device); attention runs once over all R = U*Sp tokens of the row
(``DSV41PrefillAttention``). The routed experts are the decode-format ``moe_compute`` weights: ``DSV41PrefillMoE`` is a second
``TTMoEDecode`` front-end with batch_per_device = T that SHARES the expert weights of the decode MoE (no second copy).
"""

import os
from pathlib import Path

from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode, _TTMoEDecodeBuffers
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig

_FREE = set(os.environ.get("DSV41_PF_FREE", "a,a_c,h,hh,h_tok,m,sh,x2,hs").split(","))


def _free(tag, t, keep=False):
    if tag in _FREE and t is not None:
        ttnn.deallocate(t)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "deepseek_v41_flash.yaml"


class DSV41PrefillMoE:
    """``DSV41MoEBlock.forward`` at batch_per_device = T over the expert weights of an existing decode MoE block."""

    def __init__(self, moe_block, T=32, buffers=None):
        md = moe_block.mesh_device
        text = CONFIG_PATH.read_text()
        text = text.replace("batch_per_device: 4 ", f"batch_per_device: {T} ", 1)
        text = text.replace("num_shared_experts: 1", "num_shared_experts: 0").replace(
            "  shared_expert_ids_to_devices: fully_replicated\n", ""
        )
        cfg = TTMoEDecodeConfig.from_yaml(text, topology=ttnn.Topology.Linear)
        if cfg.mesh_shape != tuple(md.shape):
            cfg = cfg.with_mesh_shape(tuple(md.shape))
        if cfg.num_fast_reduce_outputs == 1:
            cfg = cfg.model_copy(
                update={"reduce": cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})}
            )
        self.T, self.md, self.gate = T, md, moe_block.gate
        dec = object.__new__(TTMoEDecode)
        dec.config = cfg
        dec.expert_state = moe_block.decode.expert_state  # shared weights
        if buffers is None:
            bd = cfg.buffers.model_dump()
            bd["compute_tilize_drain_core"] = ttnn.experimental.get_moe_tilize_drain_core(
                md,
                cfg.compute.output_height_shard_dim,
                auto_output_width_shard_dim(cfg.hidden_size, matmul_ring_size=effective_matmul_ring_size(md)),
                cfg.hidden_size,
                mux_core_range_set=cfg.compute.mux_core_range_set,
            )
            buffers = _TTMoEDecodeBuffers(md, **bd)
        dec.buffers = buffers
        self.decode = dec

    def forward(self, tt_x_gate, tt_x_tokens):
        scores, indices = self.gate.forward(tt_x_gate)
        if indices.dtype != ttnn.uint16:
            indices = ttnn.typecast(indices, ttnn.uint16)
        if indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        if scores.dtype != ttnn.bfloat16:
            scores = ttnn.typecast(scores, ttnn.bfloat16)
        if scores.layout != ttnn.ROW_MAJOR_LAYOUT:
            scores = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        return self.decode.forward(tt_x=tt_x_tokens, tt_scores=scores, tt_indices=indices, layer_id=0)


class DSV41PrefillLayer:
    def __init__(self, layer, prefill_attn, pmoe, T=32):
        self.L, self.pa, self.pmoe, self.T = layer, prefill_attn, pmoe, T
        self.debug = None  # set to a dict: per-chunk lists of a_c / hh / m / sh are kept (not freed) for diagnostics

    def forward(self, xs, pres, S, s0=0):
        """xs: list of n chunks [T,1,4,D] fp32 (the mesh row's R = n*T tokens, user-major), pres: list of [T,1,1,4] fp32.
        -> (list of new streams, list of ffn_pre)."""
        L, T = self.L, self.T
        n = len(xs)
        eps = L.eps
        mix = [L.mhc_attn.mixes(x) for x in xs]  # (pre, post, comb)
        hs = [L.mhc_attn.collapse_norm(x, p, L.attn_norm_w, eps) for x, p in zip(xs, pres)]
        h = ttnn.concat(hs, dim=2) if n > 1 else hs[0]
        for t in hs:
            _free("hs", t)
        if self.debug is not None:
            self.debug["hs"], self.debug["h"] = list(hs), h
        a = self.pa.forward_dyn(h) if self.pa.dyn is not None else self.pa.forward(h, S, s0=s0)
        _free("h", h)
        if self.debug is not None:
            ttnn.synchronize_device(L.mesh_device)
            cols = tuple(L.mesh_device.shape)[1]
            self.debug["a_full"] = a
            self.debug["a_early"] = [
                ttnn.to_torch(ttnn.get_device_tensors(a)[r * cols]).float().reshape(-1, a.shape[3])
                for r in range(tuple(L.mesh_device.shape)[0])
            ]
        outs, pres_out = [], []
        for c in range(n):
            a_c = ttnn.slice(a, [0, 0, c * T, 0], [1, 1, (c + 1) * T, a.shape[3]])
            if self.debug is not None:
                self.debug.setdefault("a", []).append(a_c)
            x2 = L.mhc_attn.expand(a_c, xs[c], mix[c][1], mix[c][2])
            _free("a_c", a_c)
            f_pre, f_post, f_comb = L.mhc_ffn.mixes(x2)
            hh, h_tok = L.mhc_ffn.collapse_norm_rm(x2, mix[c][0], L.ffn_norm_w, eps)
            m = self.pmoe.forward(hh, h_tok)
            m = L.mesh_config.allgather(m, L.ccl, axis=1, dim=3)
            sh = L.shared.forward(hh)
            if self.debug is not None:
                for k_, v_ in (("hh", hh), ("m", m), ("sh", sh), ("x2", x2)):
                    self.debug.setdefault(k_, []).append(v_)
            x3 = L.mhc_ffn.expand(m, x2, f_post, f_comb, sh)
            outs.append(ttnn.to_memory_config(x3, ttnn.DRAM_MEMORY_CONFIG))
            pres_out.append(ttnn.to_memory_config(f_pre, ttnn.DRAM_MEMORY_CONFIG))
            for tag, t in (
                ("x2", x2),
                ("hh", hh),
                ("h_tok", h_tok),
                ("m", m),
                ("sh", sh),
            ):  # x3 / f_pre may alias the outputs: left to the caller
                _free(tag, t)
        _free("a", a)
        return outs, pres_out
