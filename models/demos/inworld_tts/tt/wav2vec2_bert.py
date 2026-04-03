"""TTNN implementation of Wav2Vec2-BERT (facebook/w2v-bert-2.0) conformer encoder.

Extracts hidden_states[16] from 16 conformer encoder layers (layers 0-15).
The model is frozen (inference only). The feature extractor (mel filterbank)
stays on CPU; this module handles feature_projection + conformer layers.

Key design:
- Self-attention: separate Q, K, V linears (not fused), relative position bias on host
- Conv module: depthwise Conv1d (groups=1024, k=31) on host, pointwise as linear on device
- GLU: host roundtrip (split + sigmoid + mul)
- Macaron half-step: ttnn.multiply by 0.5 scalar + ttnn.add
- LayerNorm weights: [1, 1, dim//32, 32] ROW_MAJOR_LAYOUT (matching existing convention)
"""

import math

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.reference.functional import w2v_relative_position_bias
from models.demos.inworld_tts.tt.model_config import (
    W2V_DEPTHWISE_KERNEL,
    W2V_DIM,
    W2V_HEAD_DIM,
    W2V_HEADS,
    W2V_INPUT_DIM,
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

        # Distance embedding stays on host (used for position bias computation)
        self.distance_embedding_weight = state_dict[prefix + "distance_embedding.weight"]  # [73, 64]

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

        # Host roundtrip for attention with relative position bias
        q_torch = ttnn.to_torch(q).squeeze(0)  # [1, T, 1024]
        k_torch = ttnn.to_torch(k).squeeze(0)
        v_torch = ttnn.to_torch(v).squeeze(0)

        B = q_torch.shape[0]
        T = q_torch.shape[1]

        # Reshape to [B, H, T, D]
        q_torch = q_torch.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k_torch = k_torch.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v_torch = v_torch.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        scores = torch.matmul(q_torch.float(), k_torch.float().transpose(-2, -1)) * self.scale

        # Relative position bias
        pos_bias = w2v_relative_position_bias(
            q_torch.float(),
            self.distance_embedding_weight.float(),
            seq_len,
            left_max=W2V_LEFT_MAX,
            right_max=W2V_RIGHT_MAX,
        )
        scores = scores + pos_bias

        # Softmax + weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_torch.float())

        # Merge heads: [B, H, T, D] -> [B, T, dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, W2V_DIM)

        # Move back to device for output projection
        attn_output = ttnn.from_torch(
            attn_output.to(torch.bfloat16).unsqueeze(0),  # [1, B, T, 1024]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=L1,
        )

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

    LN -> pointwise_conv1 (via linear) -> GLU (host) -> depthwise_conv (host) ->
    depthwise_LN (host) -> SiLU (host) -> pointwise_conv2 (via linear)

    The depthwise conv (groups=1024, k=31) runs on host because ttnn.conv1d with
    groups=channels is not efficient. Pointwise convs (k=1) are just linears.
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

        # Depthwise conv stays on host: [1024, 1, 31]
        self.dw_conv_weight = state_dict[prefix + "depthwise_conv.weight"]  # [1024, 1, 31]

        # Depthwise layer norm stays on host
        self.dw_ln_weight = state_dict[prefix + "depthwise_layer_norm.weight"]
        self.dw_ln_bias = state_dict[prefix + "depthwise_layer_norm.bias"]

        # Pointwise conv2: Conv1d(1024, 1024, k=1) = Linear(1024, 1024), no bias
        pw2_w = state_dict[prefix + "pointwise_conv2.weight"].squeeze(-1)  # [1024, 1024]
        self.pw2_weight = _to_device_weight(pw2_w, device, dtype)

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

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

        # Host roundtrip for GLU + depthwise conv + depthwise LN + SiLU
        h_torch = ttnn.to_torch(h).float()  # [1, 1, T, 2048] -> float32 for conv
        h_torch = h_torch.squeeze(0)  # [1, T, 2048]

        # GLU: split along last dim, sigmoid gate
        h1, h2 = h_torch.chunk(2, dim=-1)  # each [1, T, 1024]
        h_torch = h1 * torch.sigmoid(h2)

        # Transpose to [B, C, T] for conv
        h_torch = h_torch.transpose(1, 2)  # [1, 1024, T]

        # Causal left-pad + depthwise conv
        h_torch = F.pad(h_torch, (W2V_DEPTHWISE_KERNEL - 1, 0))  # pad 30 left
        h_torch = F.conv1d(h_torch, self.dw_conv_weight, groups=W2V_DIM)

        # Depthwise layer norm
        h_torch = h_torch.transpose(1, 2)  # [1, T, 1024]
        h_torch = F.layer_norm(h_torch, [W2V_DIM], self.dw_ln_weight, self.dw_ln_bias)

        # SiLU
        h_torch = F.silu(h_torch)

        # Back to device for pointwise conv2
        h = ttnn.from_torch(
            h_torch.to(torch.bfloat16).unsqueeze(0),  # [1, 1, T, 1024]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=L1,
        )

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

        # Feature projection on host first (input dim 160 is not tile-friendly)
        # LayerNorm(160) + Linear(160, 1024) on host
        x = F.layer_norm(
            input_features.float(),
            [W2V_INPUT_DIM],
            self.state_dict_ref["feature_projection.layer_norm.weight"],
            self.state_dict_ref["feature_projection.layer_norm.bias"],
        )
        x = F.linear(
            x,
            self.state_dict_ref["feature_projection.projection.weight"],
            self.state_dict_ref["feature_projection.projection.bias"],
        )

        # Move to device: [B, T, 1024] -> [1, B, T, 1024]
        x = ttnn.from_torch(
            x.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=L1,
        )

        # Run conformer layers on device
        for layer in self.layers:
            x = layer(x)

        # Move result back to host
        x_torch = ttnn.to_torch(x).squeeze(0)  # [B, T, 1024]

        return x_torch
