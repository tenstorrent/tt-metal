# models/demos/blackhole/qwen3_5_9b/tt/qwen35_gated_attention.py
"""Gated Attention wrapper for Qwen3.5-9B full attention layers.

Wraps the experimental `gated_attention_forward_ttnn()` into a module
that manages weight tensors, KV cache, and integrates with the model framework.
"""
import torch

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

        def load_1d(name):
            """Load 1D param (norm weight) — TILE_LAYOUT on device."""
            t = state_dict[f"{prefix}.{name}"]
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
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

        # KV cache state (concat-based, default)
        self.past_key = None
        self.past_value = None
        # Pre-allocated KV cache for trace capture
        self.max_seq_len = args.max_seq_len
        self.kv_cache_key = None
        self.kv_cache_value = None
        self.cache_pos = 0
        self.use_preallocated_cache = False

    def enable_preallocated_cache(self, batch_size=1):
        """Allocate fixed-size KV cache for trace-compatible decode."""
        self.kv_cache_key = ttnn.from_torch(
            torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.kv_cache_value = ttnn.from_torch(
            torch.zeros(batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Buffers to capture new K/V during trace (read by host between replays)
        _dram = ttnn.DRAM_MEMORY_CONFIG
        self.trace_new_k_buf = ttnn.from_torch(
            torch.zeros(batch_size, self.num_kv_heads, 1, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_dram,
        )
        self.trace_new_v_buf = ttnn.from_torch(
            torch.zeros(batch_size, self.num_kv_heads, 1, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_dram,
        )
        # Padded buffers for update_cache (needs dim[-2]=32)
        self.trace_new_k_padded = ttnn.from_torch(
            torch.zeros(batch_size, self.num_kv_heads, 32, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_dram,
        )
        self.trace_new_v_padded = ttnn.from_torch(
            torch.zeros(batch_size, self.num_kv_heads, 32, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_dram,
        )
        # Attention mask [1, 1, 1, max_seq_len] — 0 for valid, -inf for invalid
        mask = torch.zeros(1, 1, 1, self.max_seq_len, dtype=torch.bfloat16)
        mask[:, :, :, :] = -10000.0  # All invalid initially
        self.trace_attn_mask = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_dram,
        )
        # Staging position: last slot in cache, always unmasked, holds current step's K/V
        self.staging_pos = self.max_seq_len - 1
        # Padding zeros for update_cache (needs dim[-2]=32)
        self.trace_kv_pad_zeros = ttnn.from_torch(
            torch.zeros(batch_size, self.num_kv_heads, 31, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_dram,
        )
        self.use_preallocated_cache = True
        self.use_trace_mode = False
        self.cache_pos = 0

    def enable_trace_mode(self):
        """Switch to trace-compatible mode (must call enable_preallocated_cache first)."""
        self.use_trace_mode = True

    def update_cache_after_trace(self, pos):
        """Write captured K/V into cache at the correct position (called between trace replays)."""
        # Read new K/V from trace buffers, pad to [B, H, 32, D] for update_cache
        new_k = ttnn.to_torch(self.trace_new_k_buf)  # [1, H_kv, 1, D]
        new_v = ttnn.to_torch(self.trace_new_v_buf)
        k_padded = torch.zeros(1, self.num_kv_heads, 32, self.head_dim, dtype=torch.bfloat16)
        v_padded = torch.zeros(1, self.num_kv_heads, 32, self.head_dim, dtype=torch.bfloat16)
        k_padded[:, :, 0:1, :] = new_k
        v_padded[:, :, 0:1, :] = new_v
        k_tt = ttnn.from_torch(k_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        v_tt = ttnn.from_torch(v_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        ttnn.update_cache(self.kv_cache_key, k_tt, update_idx=pos)
        ttnn.update_cache(self.kv_cache_value, v_tt, update_idx=pos)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)

    def update_mask_for_pos(self, valid_len):
        """Update attention mask to expose positions 0..valid_len-1 + staging pos."""
        mask = torch.zeros(1, 1, 1, self.max_seq_len, dtype=torch.bfloat16)
        mask[:, :, :, valid_len:] = -10000.0
        # Always unmask the staging position (current token's K/V lives here)
        mask[:, :, :, self.staging_pos] = 0.0
        new_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        ttnn.copy(new_mask, self.trace_attn_mask)
        ttnn.deallocate(new_mask)

    def forward(self, x, cos, sin):
        T = x.shape[1]
        mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None

        if self.use_preallocated_cache and self.use_trace_mode:
            # Trace-compatible mode: save K/V to buffers, write to staging pos, SDPA with full cache + mask
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
                kv_cache_key=self.kv_cache_key,
                kv_cache_value=self.kv_cache_value,
                compute_kernel_config=self.compute_kernel_config,
                use_optimized_concat=True,
                memory_config=mc,
                norm_weights_pre_offset=True,
                trace_new_k_buf=self.trace_new_k_buf,
                trace_new_v_buf=self.trace_new_v_buf,
                trace_attn_mask=self.trace_attn_mask,
                trace_kv_pad_zeros=self.trace_kv_pad_zeros,
                trace_staging_pos=self.staging_pos,
            )
            return output
        elif self.use_preallocated_cache:
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
                kv_cache_key=self.kv_cache_key,
                kv_cache_value=self.kv_cache_value,
                cache_pos=self.cache_pos,
                cache_len=self.max_seq_len,
                compute_kernel_config=self.compute_kernel_config,
                use_optimized_concat=True,
                memory_config=mc,
                norm_weights_pre_offset=True,
            )
            self.cache_pos += T
            return output
        else:
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
                compute_kernel_config=self.compute_kernel_config,
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
        self.cache_pos = 0
