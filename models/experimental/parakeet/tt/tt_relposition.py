import math
import torch
import ttnn


class RelPositionMultiHeadAttentionTTNN:
    """TTNN inference implementation mirroring NeMo's RelPositionMultiHeadAttention.
    Args:
        device: TTNN device.
        config: object with num_heads, dim_head.
    Forward:
        query, key, value: (batch, time, d_model)
        mask: optional (batch, time, time) bool tensor
        pos_emb: (batch, 2*time-1, d_model) from NeMo RelPositionalEncoding
    Returns:
        output: (batch, time, d_model)
    """

    def __init__(self, device, config):
        self.num_heads = config.num_heads
        self.head_dim = config.dim_head
        self.device = device
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self._setup_compute_config()

    def _setup_compute_config(self):
        self.compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

    def prepare_weights(
        self,
        linear_q_weight,
        linear_k_weight,
        linear_v_weight,
        linear_q_bias,
        linear_k_bias,
        linear_v_bias,
        linear_out_weight,
        linear_out_bias,
        linear_pos_weight,
        pos_bias_u,
        pos_bias_v,
    ):
        # Transpose linear weights for TTNN matmul convention
        self.linear_q_weight = ttnn.from_torch(
            linear_q_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_k_weight = ttnn.from_torch(
            linear_k_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_v_weight = ttnn.from_torch(
            linear_v_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_q_bias = ttnn.from_torch(
            linear_q_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_k_bias = ttnn.from_torch(
            linear_k_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_v_bias = ttnn.from_torch(
            linear_v_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_out_weight = ttnn.from_torch(
            linear_out_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_out_bias = ttnn.from_torch(
            linear_out_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        self.linear_pos_weight = ttnn.from_torch(
            linear_pos_weight.transpose(-1, -2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        # Keep pos_bias_u/v as torch for bias addition; shape (heads, head_dim)
        self.pos_bias_u = pos_bias_u
        self.pos_bias_v = pos_bias_v

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch rel_shift matching NeMo: pad, view, slice."""
        b, h, qlen, pos_len = x.size()  # (b, h, t, 2*t-1)
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t, 2*t)
        x = x.view(b, h, -1, qlen)  # (b, h, 2*t, t)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t, t)
        return x

    def forward(self, query, key, value, mask=None, pos_emb=None, cache=None):
        bsz, T, d_model = query.shape
        # Validate pos_emb length matches NeMo's RelPositionalEncoding (2*T-1)
        if pos_emb is not None:
            pos_len = pos_emb.shape[1] if hasattr(pos_emb, "shape") else pos_emb.size(1)
            assert pos_len == 2 * T - 1, f"pos_emb length {pos_len} != 2*{T}-1"
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        key = ttnn.to_layout(key, ttnn.TILE_LAYOUT)
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        if pos_emb is not None:
            pos_emb = ttnn.to_layout(pos_emb, ttnn.TILE_LAYOUT)

        # Linear projections with bias
        q = ttnn.linear(
            query,
            self.linear_q_weight,
            bias=self.linear_q_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        k = ttnn.linear(
            key,
            self.linear_k_weight,
            bias=self.linear_k_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        v = ttnn.linear(
            value,
            self.linear_v_weight,
            bias=self.linear_v_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )

        # Reshape to (batch, heads, time, head_dim) and keep TILE layout
        q = ttnn.reshape(q, (bsz, T, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.reshape(k, (bsz, T, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.reshape(v, (bsz, T, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Positional projection (linear_pos)
        p = ttnn.linear(
            pos_emb,
            self.linear_pos_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        p = ttnn.reshape(p, (bsz, 2 * T - 1, self.num_heads, self.head_dim))
        p = ttnn.permute(p, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Add learnable biases on host (small tensors)
        q_torch = ttnn.to_torch(q)  # (B, H, T, D)
        k_torch = ttnn.to_torch(k)
        bias_u = self.pos_bias_u.unsqueeze(0).unsqueeze(2)  # (1, H, 1, D)
        bias_v = self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        q_with_bias_u = q_torch + bias_u
        q_with_bias_v = q_torch + bias_v

        # matrix_ac (content) and matrix_bd (relative) on host
        matrix_ac = torch.matmul(q_with_bias_u, k_torch.transpose(-2, -1))  # (B, H, T, T)
        p_torch = ttnn.to_torch(p)
        matrix_bd = torch.matmul(q_with_bias_v, p_torch.transpose(-2, -1))  # (B, H, T, 2*T-1)
        matrix_bd = self._rel_shift(matrix_bd)  # (B, H, T, 2*T-1) — rel_shift keeps last dim
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]  # trim to (B, H, T, T) like NeMo

        # Combine and scale
        scores_torch = (matrix_ac + matrix_bd) * self.scale  # matches NeMo's / self.s_d_k

        # Apply input mask if provided (NeMo uses INF_VAL)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T, T)
            scores_torch = scores_torch.masked_fill(mask, -10000.0)

        # Softmax and matmul with V on device
        # Softmax and matmul with V on device (use value heads, not position p)
        scores = ttnn.from_torch(scores_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        attn = ttnn.softmax(scores, dim=-1)
        attn = ttnn.to_layout(attn, ttnn.TILE_LAYOUT)
        value_heads = ttnn.to_layout(v, ttnn.TILE_LAYOUT)  # (B, H, T, D) — do not use p here
        context = ttnn.matmul(attn, value_heads)  # (B, H, T, D)

        # Concatenate heads matching NeMo: (B, H, T, D) -> (B, T, H*D)
        context = ttnn.permute(context, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)
        context = ttnn.reshape(context, (bsz, T, self.num_heads * self.head_dim))

        # Output projection
        output = ttnn.linear(
            context,
            self.linear_out_weight,
            bias=self.linear_out_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        return output
