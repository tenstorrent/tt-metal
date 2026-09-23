# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5-9B gated full-attention layer — composes config/weights/prefill/decode."""

import os

import ttnn
from models.demos.blackhole.qwen36.tt.attention.config import AttentionConfig
from models.demos.blackhole.qwen36.tt.attention.decode import decode_forward
from models.demos.blackhole.qwen36.tt.attention.prefill import prefill_forward
from models.demos.blackhole.qwen36.tt.attention.weights import load_attention_weights


class Qwen36GatedAttention:
    """Gated Full Attention layer for Qwen3.5-9B with KV cache.

    Uses softmax SDPA with GQA (16 Q heads, 4 KV heads, head_dim=256)
    plus a sigmoid output gate derived from the 2x wide q_proj.
    Q and K are normalized with zero-centered RMSNorm before attention.
    """

    def __init__(self, mesh_device, config: AttentionConfig, state_dict, tensor_cache_path=None):
        self.device = mesh_device
        self.config = config

        self.weights = load_attention_weights(mesh_device, state_dict, tensor_cache_path)

        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._prefill_progcfg_fn = tpc.make_prefill_progcfg_fn(mesh_device)

        # step2 (2026-09-22): routed through tpc.prefill_matmul_ckc() -- see QWEN36_PREFILL_MM_*
        # flags in tp_common.py. Legacy values (packer_l1_acc=False, fp32_dest_acc_en=True) unless
        # overridden.
        self.compute_kernel_config = tpc.prefill_matmul_ckc()
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # KV cache state (concat-based prefill)
        self.past_key = None
        self.past_value = None
        # Paged attention state (for vLLM integration)
        self.paged_kv_cache_key = None
        self.paged_kv_cache_value = None
        self.use_paged_attention = False

    def forward(
        self,
        x,
        cos,
        sin,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
    ):
        T = x.shape[1]
        # F3/F10A (G1): short prefills (T <= QWEN36_ATTN_L1_MAX_T, default 2048 — raised from 1024
        # for the T=2048 production chunk size, step1 task F10A) also get L1 placement for the
        # attention-layer glue ops (linears, rms_norm, fused rotary, concatenate_heads, gate,
        # chunked SDPA output); longer prefill keeps DRAM (memory_config=None) exactly as before.
        # QWEN36_ATTN_L1_MAX_T=0 restores the pre-F10A default (1024) exactly.
        _attn_l1_max_t = int(os.environ.get("QWEN36_ATTN_L1_MAX_T", "2048"))
        if _attn_l1_max_t == 0:
            _attn_l1_max_t = 1024
        # Decode (T==1) is unconditionally L1, same as before this change.
        mc = ttnn.L1_MEMORY_CONFIG if (T == 1 or T <= _attn_l1_max_t) else None
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        # Branches are mutually exclusive on T; decode (T==1) is checked first to keep the hot path short.
        if self.use_paged_attention and T == 1:
            # Branch B — paged decode
            return decode_forward(
                x=x,
                cos=cos,
                sin=sin,
                weights=self.weights,
                config=self.config,
                device=self.device,
                ckc=ckc,
                mc=mc,
                position_tensor=position_tensor,
                page_table=page_table,
                paged_kv_cache_key=self.paged_kv_cache_key,
                paged_kv_cache_value=self.paged_kv_cache_value,
            )
        elif self.use_paged_attention and T > 1 and chunk_page_table is not None:
            # Branch A — paged prefill
            return prefill_forward(
                x=x,
                cos=cos,
                sin=sin,
                weights=self.weights,
                config=self.config,
                device=self.device,
                ckc=ckc,
                mc=mc,
                paged_kv_cache_key=self.paged_kv_cache_key,
                paged_kv_cache_value=self.paged_kv_cache_value,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                use_paged_attention=True,
                prefill_progcfg_fn=getattr(self, "_prefill_progcfg_fn", None),
            )
        else:
            # Branch C — concat prefill
            output, new_key, new_value = prefill_forward(
                x=x,
                cos=cos,
                sin=sin,
                weights=self.weights,
                config=self.config,
                device=self.device,
                ckc=ckc,
                mc=mc,
                past_key=self.past_key,
                past_value=self.past_value,
                use_paged_attention=False,
                prefill_progcfg_fn=getattr(self, "_prefill_progcfg_fn", None),
            )
            self.past_key = new_key
            self.past_value = new_value
            return output

    def reset_cache(self):
        """Clear the concat KV cache for a new sequence."""
        self.past_key = None
        self.past_value = None

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach externally-allocated paged KV cache (called once after allocate_kv_cache)."""
        self.paged_kv_cache_key = k_cache
        self.paged_kv_cache_value = v_cache
        self.use_paged_attention = True
