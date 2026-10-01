# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5-9B gated full-attention layer — composes config/weights/prefill/decode."""

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

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
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
        mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None
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
