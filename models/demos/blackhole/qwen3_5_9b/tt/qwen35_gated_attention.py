# models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py
"""Gated Attention wrapper for Qwen3.5-9B full attention layers.

Wraps the experimental `gated_attention_forward_ttnn()` into a module
that manages weight tensors, KV cache, and integrates with the model framework.
"""

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import gated_attention_forward_ttnn


class Qwen35GatedAttention:
    """Gated Full Attention layer for Qwen3.5-9B with KV cache.

    Uses softmax SDPA with GQA (16 Q heads, 4 KV heads, head_dim=256)
    plus a sigmoid output gate derived from the 2× wide q_proj.
    Q and K are normalized with zero-centered RMSNorm before attention.
    """

    def __init__(self, args, state_dict, layer_num, device, weight_cache_path=None):
        self.args = args
        self.device = device
        self.layer_num = layer_num
        self.num_heads = args.n_heads
        self.num_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.norm_eps = args.norm_eps

        prefix = f"layers.{layer_num}.self_attn"

        def load_weight(name):
            """Load 2D weight, transpose to [in, out] for ttnn.linear."""
            t = state_dict[f"{prefix}.{name}"].T.contiguous()
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / f"{prefix}.{name}" if weight_cache_path else None,
            )

        def load_norm_1d(name):
            """Load 1D norm weight with +1 offset for zero-centered RMSNorm fused path."""
            t = state_dict[f"{prefix}.{name}"] + 1.0
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / f"{prefix}.{name}_offset" if weight_cache_path else None,
            )

        self.q_proj_weight = load_weight("q_proj.weight")
        self.k_proj_weight = load_weight("k_proj.weight")
        self.v_proj_weight = load_weight("v_proj.weight")
        self.o_proj_weight = load_weight("o_proj.weight")
        self.q_norm_weight = load_norm_1d("q_norm.weight")
        self.k_norm_weight = load_norm_1d("k_norm.weight")

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
        self.max_seq_len = args.max_seq_len
        # Paged attention state (for vLLM integration)
        self.paged_kv_cache_key = None
        self.paged_kv_cache_value = None
        self.use_paged_attention = False

    def forward(self, x, cos, sin, position_tensor=None, page_table=None, chunk_page_table=None, chunk_start_idx=None):
        T = x.shape[1]
        mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        if self.use_paged_attention and T > 1 and chunk_page_table is not None:
            # Paged prefill: fill K/V into paged cache + chunked SDPA
            output, _, _ = gated_attention_forward_ttnn(
                hidden_states=x,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                o_proj_weight=self.o_proj_weight,
                q_norm_weight=self.q_norm_weight,
                k_norm_weight=self.k_norm_weight,
                cos=cos,
                sin=sin,
                num_attention_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                device=self.device,
                norm_eps=self.norm_eps,
                compute_kernel_config=ckc,
                use_optimized_concat=True,
                norm_weights_pre_offset=True,
                page_table=page_table,
                paged_kv_cache_key=self.paged_kv_cache_key,
                paged_kv_cache_value=self.paged_kv_cache_value,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
            )
            return output
        elif self.use_paged_attention and T == 1:
            # Paged decode: use paged_update_cache + paged_sdpa_decode via page_table
            output, _, _ = gated_attention_forward_ttnn(
                hidden_states=x,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                o_proj_weight=self.o_proj_weight,
                q_norm_weight=self.q_norm_weight,
                k_norm_weight=self.k_norm_weight,
                cos=cos,
                sin=sin,
                num_attention_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                device=self.device,
                norm_eps=self.norm_eps,
                compute_kernel_config=ckc,
                use_optimized_concat=True,
                memory_config=mc,
                norm_weights_pre_offset=True,
                cur_pos_tensor=position_tensor,
                page_table=page_table,
                paged_kv_cache_key=self.paged_kv_cache_key,
                paged_kv_cache_value=self.paged_kv_cache_value,
            )
            return output
        else:
            # Concat path: used by non-paged prefill and short-sequence paged prefill (T<=1024).
            # For T>1024 paged prefill, the paged branch above is used instead.
            output, new_key, new_value = gated_attention_forward_ttnn(
                hidden_states=x,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                o_proj_weight=self.o_proj_weight,
                q_norm_weight=self.q_norm_weight,
                k_norm_weight=self.k_norm_weight,
                cos=cos,
                sin=sin,
                num_attention_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                device=self.device,
                norm_eps=self.norm_eps,
                past_key=self.past_key,
                past_value=self.past_value,
                compute_kernel_config=ckc,
                use_optimized_concat=True,
                memory_config=mc,
                norm_weights_pre_offset=True,
            )
            self.past_key = new_key
            self.past_value = new_value
            return output

    def reset_cache(self):
        """Clear KV cache for new sequence."""
        self.past_key = None
        self.past_value = None

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach externally-allocated paged KV cache (called once after allocate_kv_cache)."""
        self.paged_kv_cache_key = k_cache
        self.paged_kv_cache_value = v_cache
        self.use_paged_attention = True
