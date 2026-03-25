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
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
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
        # Pre-computed tensors for on-device mask computation (avoids host round-trip)
        # Position indices [0, 1, 2, ..., max_seq_len-1] for threshold comparison
        self.mask_indices = ttnn.from_torch(
            torch.arange(self.max_seq_len, dtype=torch.bfloat16).reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_dram,
        )
        # One-hot at staging position (1.0 at staging_pos, 0.0 elsewhere)
        staging_one_hot = torch.zeros(1, 1, 1, self.max_seq_len, dtype=torch.bfloat16)
        staging_one_hot[:, :, :, self.staging_pos] = 1.0
        self.staging_mask = ttnn.from_torch(
            staging_one_hot,
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
        """Write captured K/V into cache at the correct position (called between trace replays).

        All on device — uses the same concat+update_cache pattern as the trace forward
        path (ttnn_gated_attention.py lines 212-215). No host round-trips.
        """
        k_padded = ttnn.concat([self.trace_new_k_buf, self.trace_kv_pad_zeros], dim=2)
        v_padded = ttnn.concat([self.trace_new_v_buf, self.trace_kv_pad_zeros], dim=2)
        ttnn.update_cache(self.kv_cache_key, k_padded, update_idx=pos)
        ttnn.update_cache(self.kv_cache_value, v_padded, update_idx=pos)
        ttnn.deallocate(k_padded)
        ttnn.deallocate(v_padded)

    def update_mask_for_pos(self, valid_len):
        """Update attention mask to expose positions 0..valid_len-1 + staging pos.

        All on device — uses pre-computed index tensor + comparison ops.
        No host round-trips (the old path created a CPU tensor and transferred each step).
        """
        # mask_indices: [0, 1, 2, ..., max_seq_len-1]
        # valid where index < valid_len (1.0) or at staging_pos (1.0), else 0.0
        valid = ttnn.lt(self.mask_indices, float(valid_len))
        # staging_mask is 1.0 at staging_pos only; positions don't overlap since
        # staging_pos = max_seq_len-1 and valid_len <= max_seq_len-1
        combined = ttnn.add(valid, self.staging_mask)
        # non-zero → 0.0 (unmasked), zero → -10000.0 (masked)
        new_mask = ttnn.where(combined, 0.0, -10000.0)
        ttnn.copy(new_mask, self.trace_attn_mask)
        ttnn.deallocate(valid)
        ttnn.deallocate(combined)
        ttnn.deallocate(new_mask)

    def forward(self, x, cos, sin):
        T = x.shape[1]
        mc = ttnn.L1_MEMORY_CONFIG if T == 1 else None
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

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
                compute_kernel_config=ckc,
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
                compute_kernel_config=ckc,
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
        self.cache_pos = 0
