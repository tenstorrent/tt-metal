"""TTNN implementation of Wav2Vec2-BERT (facebook/w2v-bert-2.0) conformer encoder.

Extracts hidden_states[16] from 16 conformer encoder layers (layers 0-15).
The model is frozen (inference only). The feature extractor (mel filterbank)
stays on CPU; this module handles feature_projection + conformer layers.

Key design:
- Self-attention: separate Q, K, V linears (not fused), relative position bias on device
- Conv module: depthwise Conv1d (groups=1024, k=31) on device via ttnn.conv1d
- GLU: on device (split + sigmoid + mul)
- Macaron half-step: ttnn.multiply by 0.5 scalar + ttnn.add
- LayerNorm weights: [1, 1, dim//32, 32] ROW_MAJOR_LAYOUT (matching existing convention)
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.model_config import (
    W2V_DEPTHWISE_KERNEL,
    W2V_DIM,
    W2V_HEAD_DIM,
    W2V_HEADS,
    W2V_LEFT_MAX,
    W2V_NUM_LAYERS,
    W2V_RIGHT_MAX,
    get_compute_kernel_config_hifi4,
)

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _to_device_weight(tensor_2d: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Convert a 2D weight [out, in] to TTNN format [1, 1, in, out] transposed for linear."""
    return ttnn.from_torch(
        tensor_2d.T.unsqueeze(0).unsqueeze(0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=DRAM,
    )


def _to_device_bias(tensor_1d: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Convert a 1D bias [dim] to TTNN format [1, 1, 1, dim]."""
    return ttnn.from_torch(
        tensor_1d.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=DRAM,
    )


def _to_device_ln_weight(tensor_1d: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Convert LayerNorm weight/bias [dim] to [1, 1, dim//32, 32] ROW_MAJOR."""
    dim = tensor_1d.shape[0]
    return ttnn.from_torch(
        tensor_1d.reshape(1, 1, dim // 32, 32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=DRAM,
    )


class TtW2vFFN(LightweightModule):
    """Conformer FFN: Linear(1024, 4096) -> SiLU -> Linear(4096, 1024)."""

    def __init__(self, device, state_dict, prefix, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device

        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        self.w1 = _to_device_weight(state_dict[prefix + "intermediate_dense.weight"], device, dtype)
        self.b1 = _to_device_bias(state_dict[prefix + "intermediate_dense.bias"], device, dtype)
        self.w2 = _to_device_weight(state_dict[prefix + "output_dense.weight"], device, dtype)
        self.b2 = _to_device_bias(state_dict[prefix + "output_dense.bias"], device, dtype)
        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def forward(self, x):
        """Forward: x -> Linear -> SiLU -> Linear.

        Args:
            x: [1, 1, T, 1024] TILE_LAYOUT
        Returns:
            [1, 1, T, 1024] in L1
        """
        h = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        h = ttnn.silu(h, memory_config=L1)
        h = ttnn.linear(
            h,
            self.w2,
            bias=self.b2,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        return h


class TtW2vSelfAttention(LightweightModule):
    """Wav2Vec2-BERT self-attention with relative position bias.

    Q, K, V are separate linears. Relative position bias computed on host.
    SDPA with position bias cannot use ttnn.transformer.scaled_dot_product_attention
    directly (it doesn't support additive bias), so we do manual attention on host.
    """

    def __init__(self, device, state_dict, prefix, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.n_heads = W2V_HEADS
        self.head_dim = W2V_HEAD_DIM
        self.scale = 1.0 / math.sqrt(self.head_dim)

        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        # Q, K, V, Out projection weights
        self.wq = _to_device_weight(state_dict[prefix + "linear_q.weight"], device, dtype)
        self.bq = _to_device_bias(state_dict[prefix + "linear_q.bias"], device, dtype)
        self.wk = _to_device_weight(state_dict[prefix + "linear_k.weight"], device, dtype)
        self.bk = _to_device_bias(state_dict[prefix + "linear_k.bias"], device, dtype)
        self.wv = _to_device_weight(state_dict[prefix + "linear_v.weight"], device, dtype)
        self.bv = _to_device_bias(state_dict[prefix + "linear_v.bias"], device, dtype)
        self.wo = _to_device_weight(state_dict[prefix + "linear_out.weight"], device, dtype)
        self.bo = _to_device_bias(state_dict[prefix + "linear_out.bias"], device, dtype)

        # Distance embedding on device for position bias computation
        self.distance_embedding_weight = state_dict[prefix + "distance_embedding.weight"]  # [73, 64]
        self.distance_embedding_tt = ttnn.from_torch(
            self.distance_embedding_weight.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=DRAM,
        )

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def forward(self, x):
        """Forward pass with relative position bias.

        Args:
            x: [1, 1, T, 1024] TILE_LAYOUT on device
        Returns:
            [1, 1, T, 1024] in L1
        """
        seq_len = x.shape[2]

        # Q, K, V projections on device
        q = ttnn.linear(
            x,
            self.wq,
            bias=self.bq,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        k = ttnn.linear(
            x,
            self.wk,
            bias=self.bk,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        v = ttnn.linear(
            x,
            self.wv,
            bias=self.bv,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Reshape to multi-head on device: [1, 1, T, dim] -> [1, T, H, D] -> [1, H, T, D]
        q = ttnn.permute(ttnn.reshape(q, [1, seq_len, self.n_heads, self.head_dim]), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, [1, seq_len, self.n_heads, self.head_dim]), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, [1, seq_len, self.n_heads, self.head_dim]), (0, 2, 1, 3))

        # Q @ K^T * scale on device
        k_t = ttnn.permute(k, (0, 1, 3, 2))  # [1, H, D, T]
        scores = ttnn.matmul(q, k_t)  # [1, H, T, T]
        scores = ttnn.multiply(scores, self.scale)

        # Position bias on device via ttnn.embedding + ttnn.matmul
        # 1. Distance indices (static per seq_len)
        positions = torch.arange(seq_len)
        distances = (positions[:, None] - positions[None, :]).clamp(-W2V_LEFT_MAX, W2V_RIGHT_MAX) + W2V_LEFT_MAX
        dist_indices = distances.long().reshape(1, -1)  # [1, T*T] for ttnn.embedding
        dist_tt = ttnn.from_torch(dist_indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        # 2. Embedding lookup: [1, T*T] -> [1, T*T, 64]
        pos_embed = ttnn.embedding(dist_tt, self.distance_embedding_tt)  # [1, T*T, 64]
        pos_embed = ttnn.reshape(pos_embed, [seq_len, seq_len, self.head_dim])  # [T, T, 64]

        # 3. Transpose for matmul: [T, T, 64] -> [T, 64, T]
        pos_embed_t = ttnn.permute(pos_embed, (0, 2, 1))  # [T, 64, T]
        pos_embed_t = ttnn.to_layout(pos_embed_t, ttnn.TILE_LAYOUT)

        # 4. Reshape query: [1, 16, T, 64] -> [T, 16, 64] for batched matmul over T
        q_for_bias = ttnn.permute(q, (0, 2, 1, 3))  # [1, T, 16, 64]
        q_for_bias = ttnn.reshape(q_for_bias, [seq_len, self.n_heads, self.head_dim])  # [T, 16, 64]

        # 5. Batched matmul: [T, 16, 64] @ [T, 64, T] -> [T, 16, T]
        pos_bias = ttnn.matmul(q_for_bias, pos_embed_t)

        # 6. Reshape to [1, 16, T, T] and scale: [T, 16, T] -> [1, T, 16, T] -> [1, 16, T, T]
        pos_bias = ttnn.reshape(pos_bias, [1, seq_len, self.n_heads, seq_len])  # [1, T, 16, T]
        pos_bias = ttnn.permute(pos_bias, (0, 2, 1, 3))  # [1, 16, T, T]
        pos_bias = ttnn.multiply(pos_bias, 1.0 / (self.head_dim**0.5))

        # Add bias + softmax + weighted sum on device
        scores = ttnn.add(scores, pos_bias)
        attn_weights = ttnn.softmax(scores, dim=-1)
        attn_output = ttnn.matmul(attn_weights, v)  # [1, H, T, D]

        # Head merge on device: [1, H, T, D] -> [1, T, H, D] -> [1, 1, T, dim]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (1, 1, seq_len, W2V_DIM))

        # Output projection on device
        out = ttnn.linear(
            attn_output,
            self.wo,
            bias=self.bo,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        return out


class TtW2vConvModule(LightweightModule):
    """Conformer convolution module.

    LN -> pointwise_conv1 (via linear) -> GLU (device) -> depthwise_conv (device) ->
    depthwise_LN (device) -> SiLU (device) -> pointwise_conv2 (via linear)

    All ops run on device. Depthwise conv uses ttnn.conv1d with groups=1024.
    """

    def __init__(self, device, state_dict, prefix, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device

        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        # LayerNorm weights
        self.ln_weight = _to_device_ln_weight(state_dict[prefix + "layer_norm.weight"], device, dtype)
        self.ln_bias = _to_device_ln_weight(state_dict[prefix + "layer_norm.bias"], device, dtype)

        # Pointwise conv1: Conv1d(1024, 2048, k=1) = Linear(1024, 2048), no bias
        # Weight shape from checkpoint: [2048, 1024, 1] -> squeeze to [2048, 1024]
        pw1_w = state_dict[prefix + "pointwise_conv1.weight"].squeeze(-1)  # [2048, 1024]
        self.pw1_weight = _to_device_weight(pw1_w, device, dtype)

        # Depthwise conv weight: [1024, 1, 31] (keep raw torch tensor for reference)
        self.dw_conv_weight = state_dict[prefix + "depthwise_conv.weight"]  # [1024, 1, 31]
        # TTNN host tensor for ttnn.conv1d
        self.dw_conv_weight_tt = ttnn.from_torch(
            self.dw_conv_weight.to(torch.float32),
            dtype=ttnn.float32,
        )

        # Depthwise layer norm weights (keep raw torch tensors for reference)
        self.dw_ln_weight = state_dict[prefix + "depthwise_layer_norm.weight"]
        self.dw_ln_bias = state_dict[prefix + "depthwise_layer_norm.bias"]
        # Device tensors for ttnn.layer_norm
        self.dw_ln_weight_tt = _to_device_ln_weight(self.dw_ln_weight, device, dtype)
        self.dw_ln_bias_tt = _to_device_ln_weight(self.dw_ln_bias, device, dtype)

        # Pointwise conv2: Conv1d(1024, 1024, k=1) = Linear(1024, 1024), no bias
        pw2_w = state_dict[prefix + "pointwise_conv2.weight"].squeeze(-1)  # [1024, 1024]
        self.pw2_weight = _to_device_weight(pw2_w, device, dtype)

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

        # Depthwise conv1d config (BLOCK_SHARDED)
        self.dw_conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        )

        # Cached device weights for depthwise conv (populated on first forward)
        self._dw_cached_weight = None
        self._dw_cached_bias = None

    def forward(self, x):
        """Forward pass.

        Args:
            x: [1, 1, T, 1024] TILE_LAYOUT on device
        Returns:
            [1, 1, T, 1024] in L1
        """
        # LayerNorm on device
        h = ttnn.layer_norm(
            x,
            weight=self.ln_weight,
            bias=self.ln_bias,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Pointwise conv1 as linear on device: [1, 1, T, 1024] -> [1, 1, T, 2048]
        h = ttnn.linear(
            h,
            self.pw1_weight,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # GLU on device: split + sigmoid + mul
        h1 = h[:, :, :, :W2V_DIM]  # [1, 1, T, 1024]
        h2 = h[:, :, :, W2V_DIM:]  # [1, 1, T, 1024]
        h = ttnn.mul(h1, ttnn.sigmoid(h2), memory_config=L1)

        # Transpose to ROW_MAJOR for conv1d
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)

        # Depthwise conv (k=31, groups=1024, causal left-pad=30)
        T = h.shape[2]
        w = self._dw_cached_weight or self.dw_conv_weight_tt
        b = self._dw_cached_bias
        h, out_len, [self._dw_cached_weight, self._dw_cached_bias] = ttnn.conv1d(
            input_tensor=h,
            weight_tensor=w,
            in_channels=W2V_DIM,
            out_channels=W2V_DIM,
            device=self.device,
            bias_tensor=b,
            kernel_size=W2V_DEPTHWISE_KERNEL,
            stride=1,
            padding=(W2V_DEPTHWISE_KERNEL - 1, 0),
            batch_size=1,
            input_length=T,
            dtype=ttnn.bfloat16,
            conv_config=self.dw_conv_config,
            compute_config=self.compute_kernel_config,
            groups=W2V_DIM,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        h = ttnn.sharded_to_interleaved(h, DRAM)
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # LayerNorm + SiLU on device
        h = ttnn.layer_norm(
            h,
            weight=self.dw_ln_weight_tt,
            bias=self.dw_ln_bias_tt,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        h = ttnn.silu(h, memory_config=L1)

        # Pointwise conv2 as linear on device
        h = ttnn.linear(
            h,
            self.pw2_weight,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        return h


class TtW2vConformerLayer(LightweightModule):
    """Single Conformer encoder layer with Macaron half-step FFNs.

    Sub-block 1: FFN1 (half-step) -- x = FFN1(LN(x)) * 0.5 + x
    Sub-block 2: Self-Attention  -- x = Attn(LN(x)) + x
    Sub-block 3: Conv Module     -- x = Conv(x) + x
    Sub-block 4: FFN2 (half-step) -- x = FFN2(LN(x)) * 0.5 + x
    Final: LayerNorm
    """

    def __init__(self, device, state_dict, layer_idx, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        prefix = f"encoder.layers.{layer_idx}."

        # LayerNorm weights for each sub-block
        self.ffn1_ln_w = _to_device_ln_weight(state_dict[prefix + "ffn1_layer_norm.weight"], device, dtype)
        self.ffn1_ln_b = _to_device_ln_weight(state_dict[prefix + "ffn1_layer_norm.bias"], device, dtype)

        self.attn_ln_w = _to_device_ln_weight(state_dict[prefix + "self_attn_layer_norm.weight"], device, dtype)
        self.attn_ln_b = _to_device_ln_weight(state_dict[prefix + "self_attn_layer_norm.bias"], device, dtype)

        self.ffn2_ln_w = _to_device_ln_weight(state_dict[prefix + "ffn2_layer_norm.weight"], device, dtype)
        self.ffn2_ln_b = _to_device_ln_weight(state_dict[prefix + "ffn2_layer_norm.bias"], device, dtype)

        self.final_ln_w = _to_device_ln_weight(state_dict[prefix + "final_layer_norm.weight"], device, dtype)
        self.final_ln_b = _to_device_ln_weight(state_dict[prefix + "final_layer_norm.bias"], device, dtype)

        # Sub-modules
        self.ffn1 = TtW2vFFN(device, state_dict, prefix + "ffn1.", dtype)
        self.self_attn = TtW2vSelfAttention(device, state_dict, prefix + "self_attn.", dtype)
        self.conv_module = TtW2vConvModule(device, state_dict, prefix + "conv_module.", dtype)
        self.ffn2 = TtW2vFFN(device, state_dict, prefix + "ffn2.", dtype)

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def forward(self, x):
        """Forward pass.

        Args:
            x: [1, 1, T, 1024] TILE_LAYOUT on device
        Returns:
            [1, 1, T, 1024] in L1
        """
        # Sub-block 1: FFN1 (half-step)
        h = ttnn.layer_norm(
            x,
            weight=self.ffn1_ln_w,
            bias=self.ffn1_ln_b,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        h = self.ffn1(h)
        h = ttnn.multiply(h, 0.5, memory_config=L1)
        x = ttnn.add(x, h, memory_config=L1)

        # Sub-block 2: Self-Attention
        h = ttnn.layer_norm(
            x,
            weight=self.attn_ln_w,
            bias=self.attn_ln_b,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        h = self.self_attn(h)
        x = ttnn.add(x, h, memory_config=L1)

        # Sub-block 3: Convolution Module
        h = self.conv_module(x)
        x = ttnn.add(x, h, memory_config=L1)

        # Sub-block 4: FFN2 (half-step)
        h = ttnn.layer_norm(
            x,
            weight=self.ffn2_ln_w,
            bias=self.ffn2_ln_b,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
        h = self.ffn2(h)
        h = ttnn.multiply(h, 0.5, memory_config=L1)
        x = ttnn.add(x, h, memory_config=L1)

        # Final LayerNorm
        x = ttnn.layer_norm(
            x,
            weight=self.final_ln_w,
            bias=self.final_ln_b,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x


class TtWav2Vec2Bert(LightweightModule):
    """Wav2Vec2-BERT conformer encoder (facebook/w2v-bert-2.0).

    Extracts hidden_states[16] by running feature_projection + 16 conformer layers.
    The model is frozen (inference only).

    Usage:
        model = TtWav2Vec2Bert(device)
        hidden = model.forward(input_features)  # [B, T, 1024]
    """

    def __init__(self, device, state_dict=None, num_layers=W2V_NUM_LAYERS, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.num_layers = num_layers

        # Load state dict from HuggingFace if not provided
        if state_dict is None:
            state_dict = self._load_hf_state_dict()
        self.state_dict_ref = state_dict

        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        # Feature projection weights
        self.fp_ln_w = _to_device_ln_weight(state_dict["feature_projection.layer_norm.weight"], device, dtype)
        self.fp_ln_b = _to_device_ln_weight(state_dict["feature_projection.layer_norm.bias"], device, dtype)
        self.fp_proj_w = _to_device_weight(state_dict["feature_projection.projection.weight"], device, dtype)
        self.fp_proj_b = _to_device_bias(state_dict["feature_projection.projection.bias"], device, dtype)

        # Conformer layers
        self.layers = []
        for i in range(num_layers):
            layer = TtW2vConformerLayer(device, state_dict, i, dtype)
            self.layers.append(layer)

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    @staticmethod
    def _load_hf_state_dict():
        """Load Wav2Vec2-BERT state dict from HuggingFace, stripping the model prefix."""
        from transformers import Wav2Vec2BertModel

        hf_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        raw_sd = hf_model.state_dict()

        # Strip "wav2vec2_bert." prefix if present
        state_dict = {}
        prefix = "wav2vec2_bert."
        for k, v in raw_sd.items():
            if k.startswith(prefix):
                state_dict[k[len(prefix) :]] = v
            else:
                state_dict[k] = v

        del hf_model
        return state_dict

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Full forward pass: feature_projection + conformer layers.

        Args:
            input_features: [B, T, 160] torch tensor (mel filterbank features)
        Returns:
            [B, T, 1024] torch tensor (hidden_states[num_layers])
        """
        B, T, _ = input_features.shape

        # Feature projection on device (160 = 5*32, tile-aligned)
        x = ttnn.from_torch(
            input_features.to(torch.bfloat16).unsqueeze(0),  # [1, B, T, 160]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=L1,
        )

        # LayerNorm(160) on device
        x = ttnn.layer_norm(
            x,
            weight=self.fp_ln_w,
            bias=self.fp_ln_b,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Linear(160, 1024) on device
        x = ttnn.linear(
            x,
            self.fp_proj_w,
            bias=self.fp_proj_b,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Run conformer layers on device
        for layer in self.layers:
            x = layer(x)

        # Move result back to host
        x_torch = ttnn.to_torch(x).squeeze(0)  # [B, T, 1024]

        return x_torch
