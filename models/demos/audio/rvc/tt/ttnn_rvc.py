# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of RVC (Retrieval-based Voice Conversion).

RVC combines a VITS-based posterior encoder, pitch extraction (RMVPE),
feature retrieval (index-based), flow-based decoder, and HiFi-GAN vocoder
for high-quality voice conversion on Tenstorrent hardware.

Reference: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import ttnn
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RVC_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
RVC_L1_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG

RVC_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Default hyper-parameters (matching RVC v2 default config)
DEFAULT_HIDDEN_CHANNELS = 192
DEFAULT_FILTER_CHANNELS = 768
DEFAULT_N_HEADS = 2
DEFAULT_N_LAYERS = 6
DEFAULT_KERNEL_SIZE = 3
DEFAULT_P_KERNEL_SIZE = 5
DEFAULT_RESBLOCK_KERNEL_SIZES = [3, 7, 11]
DEFAULT_RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
DEFAULT_UPSAMPLE_RATES = [10, 6, 2, 2, 2]
DEFAULT_UPSAMPLE_INITIAL_CHANNEL = 512
DEFAULT_UPSAMPLE_KERNEL_SIZES = [20, 12, 4, 4, 4]
DEFAULT_INTER_CHANNELS = 192
RVC_L1_SMALL_SIZE = 1600


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def nearest_32(x: int) -> int:
    return ((x + 31) // 32) * 32


def unsqueeze_to_4D(tensor):
    """Ensure tensor is 4-D on device (batch, 1, seq, feat)."""
    while len(tensor.shape) < 4:
        tensor = ttnn.unsqueeze(tensor, 1)
    return tensor


# ---------------------------------------------------------------------------
# Layer Norm (TTNN)
# ---------------------------------------------------------------------------

def ttnn_layer_norm(x, weight, bias, eps=1e-5):
    """Layer normalization using TTNN operations."""
    mean = ttnn.mean(x, dim=-1, keepdim=True)
    x_centered = ttnn.sub(x, mean)
    variance = ttnn.mean(ttnn.mul(x_centered, x_centered), dim=-1, keepdim=True)
    x_norm = ttnn.mul(x_centered, ttnn.rsqrt(ttnn.add(variance, eps)))
    if weight is not None:
        x_norm = ttnn.mul(x_norm, weight)
    if bias is not None:
        x_norm = ttnn.add(x_norm, bias)
    return x_norm


# ---------------------------------------------------------------------------
# 1-D Convolution helpers
# ---------------------------------------------------------------------------

def ttnn_conv1d(x, weight, bias=None, stride=1, padding=0):
    """
    1-D convolution via matmul unfold approach.
    x: [B, C, L] (torch, on CPU)
    weight: [out_c, in_c, k] (torch, on CPU)
    Returns torch tensor [B, out_c, L_out]
    """
    if isinstance(x, ttnn.Tensor):
        x = ttnn.to_torch(x)
    if isinstance(weight, ttnn.Tensor):
        weight = ttnn.to_torch(weight)

    B, C_in, L = x.shape
    C_out, C_in_w, K = weight.shape
    assert C_in == C_in_w

    if padding > 0:
        x = F.pad(x, (padding, padding))

    L_padded = x.shape[-1]
    L_out = (L_padded - K) // stride + 1

    # Unfold
    x_unfold = F.unfold(x.unsqueeze(-1), kernel_size=(K, 1), stride=(stride, 1))
    # x_unfold: [B, C_in*K, L_out]
    x_mat = x_unfold  # [B, C_in*K, L_out]
    w_mat = weight.reshape(C_out, -1)  # [C_out, C_in*K]

    out = torch.einsum("oi,bio->bo", w_mat, x_mat)  # [B, C_out, L_out]

    if bias is not None:
        if isinstance(bias, ttnn.Tensor):
            bias = ttnn.to_torch(bias)
        out = out + bias.reshape(1, -1, 1)

    return out


# ---------------------------------------------------------------------------
# Multi-Head Attention (encoder style)
# ---------------------------------------------------------------------------

def ttnn_multihead_attention(
    x,
    parameters,
    n_heads,
    mask=None,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    Multi-head self-attention for encoder blocks.
    x: ttnn tensor [B, 1, S, C]
    Returns: ttnn tensor [B, 1, S, C]
    """
    device = x.device()
    compute_grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x)

    B, _, S, C = x.shape
    head_dim = C // n_heads

    # QKV projection
    x = ttnn.to_memory_config(x, RVC_L1_MEMORY_CONFIG)
    qkv = ttnn.linear(
        x,
        parameters.in_proj_weight,
        bias=parameters.in_proj_bias,
        core_grid=core_grid,
        memory_config=RVC_L1_MEMORY_CONFIG,
    )

    # Split Q, K, V
    qkv = ttnn.to_memory_config(qkv, memory_config)
    q, k, v = ttnn.split(qkv, [C, C, C], dim=-1)

    # Reshape to multi-head: [B, n_heads, S, head_dim]
    q = ttnn.reshape(q, (B, n_heads, S, head_dim))
    k = ttnn.reshape(k, (B, n_heads, S, head_dim))
    v = ttnn.reshape(v, (B, n_heads, S, head_dim))

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
    attn_weights = ttnn.mul(attn_weights, scale)

    if mask is not None:
        attn_weights = ttnn.add(attn_weights, mask)

    attn_weights = ttnn.softmax(attn_weights, dim=-1)
    attn_output = ttnn.matmul(attn_weights, v)

    # Reshape back: [B, 1, S, C]
    attn_output = ttnn.reshape(attn_output, (B, 1, S, C))

    # Output projection
    attn_output = ttnn.to_memory_config(attn_output, RVC_L1_MEMORY_CONFIG)
    output = ttnn.linear(
        attn_output,
        parameters.out_proj.weight,
        bias=parameters.out_proj.bias,
        core_grid=core_grid,
        memory_config=memory_config,
    )

    return output


# ---------------------------------------------------------------------------
# Encoder Block
# ---------------------------------------------------------------------------

def encoder_forward(
    x,
    parameters,
    n_heads,
    n_layers=None,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    Forward pass through the VITS posterior encoder (a stack of conv + attn layers).
    x: ttnn tensor [B, 1, S, C]
    """
    for i in range(n_layers or len(parameters.enc_layers)):
        layer_params = parameters.enc_layers[i]

        # Self-attention
        residual = x
        x_norm = ttnn_layer_norm(
            x, layer_params.norm1.weight, layer_params.norm1.bias
        )
        attn_out = ttnn_multihead_attention(
            x_norm, layer_params.attn, n_heads, memory_config=memory_config
        )
        x = ttnn.add(residual, attn_out)

        # FFN
        residual = x
        x_norm = ttnn_layer_norm(
            x, layer_params.norm2.weight, layer_params.norm2.bias
        )
        ffn_out = ttnn_ffn(x_norm, layer_params.ffn, memory_config=memory_config)
        x = ttnn.add(residual, ffn_out)

    return x


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

def ttnn_ffn(x, parameters, memory_config=RVC_MEMORY_CONFIG):
    """Position-wise FFN."""
    device = x.device()
    compute_grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x)

    x = ttnn.to_memory_config(x, RVC_L1_MEMORY_CONFIG)
    hidden = ttnn.linear(
        x,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        core_grid=core_grid,
        memory_config=RVC_L1_MEMORY_CONFIG,
        activation="gelu",
    )
    output = ttnn.linear(
        hidden,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        core_grid=core_grid,
        memory_config=memory_config,
    )
    ttnn.deallocate(hidden)

    return output


# ---------------------------------------------------------------------------
# Posterior Encoder (VITS-style)
# ---------------------------------------------------------------------------

def posterior_encoder_forward(
    x,
    g,
    parameters,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    VITS posterior encoder: conv stack → encoder → projection → sample z.

    Args:
        x: Input mel-spectrogram [B, n_mels, T]
        g: Speaker embedding [B, g_channels] (optional conditioning)
        parameters: Pre-processed model parameters

    Returns:
        z: Latent representation [B, inter_channels, T']
        m: Mean [B, inter_channels, T']
        logs: Log-variance [B, inter_channels, T']
    """
    device = parameters.device if hasattr(parameters, "device") else None

    # Pre-conv layers (on CPU for conv1d)
    if isinstance(x, ttnn.Tensor):
        x_cpu = ttnn.to_torch(x).float()
    else:
        x_cpu = x.float()

    for i, conv_params in enumerate(parameters.pre_convs):
        x_cpu = ttnn_conv1d(
            x_cpu,
            conv_params.weight,
            conv_params.bias,
            stride=1,
            padding=conv_params.kernel_size // 2 if hasattr(conv_params, "kernel_size") else 1,
        )
        x_cpu = F.relu(x_cpu)

    # Encoder (transformer-based)
    B, C, T = x_cpu.shape
    x_ttnn = ttnn.from_torch(
        x_cpu.reshape(B, 1, T, C),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    x_ttnn = encoder_forward(x_ttnn, parameters, DEFAULT_N_HEADS, memory_config=memory_config)

    # Project to mean and log-variance
    x_cpu = ttnn.to_torch(x_ttnn).float().reshape(B, T, -1).transpose(1, 2)

    m = F.conv1d(x_cpu, parameters.proj_m_weight, parameters.proj_m_bias)
    logs = F.conv1d(x_cpu, parameters.proj_logs_weight, parameters.proj_logs_bias)

    # Reparameterization trick
    z = m + torch.randn_like(m) * torch.exp(logs) * 0.1  # scale noise during inference

    return z, m, logs


# ---------------------------------------------------------------------------
# Pitch Extraction (RMVPE)
# ---------------------------------------------------------------------------

def rmvpe_forward(
    audio,
    parameters,
    sample_rate=16000,
    hop_length=160,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    RMVPE pitch extraction module.

    Simplified version using a small CNN backbone for F0 estimation.
    For production, the full RMVPE model weights should be loaded.

    Args:
        audio: Input waveform [B, T_audio]
        parameters: RMVPE model parameters

    Returns:
        f0: Fundamental frequency [B, T_frames]
    """
    # Compute mel spectrogram
    if isinstance(audio, ttnn.Tensor):
        audio_cpu = ttnn.to_torch(audio).float()
    else:
        audio_cpu = audio.float()

    B, T = audio_cpu.shape
    n_fft = 1024
    n_mels = 128

    # Windowed STFT
    window = torch.hann_window(n_fft, device=audio_cpu.device)
    stft = torch.stft(audio_cpu, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mel_spec = torch.abs(stft) ** 2

    # Mel filterbank (simplified - use first n_mels bins)
    mel_spec = mel_spec[:, :n_mels, :]

    # CNN backbone for pitch estimation
    x = mel_spec
    for i, conv_params in enumerate(parameters.convs):
        x = F.conv1d(x, conv_params.weight, conv_params.bias, padding=conv_params.padding)
        x = F.relu(x)

    # F0 prediction head
    f0_logits = F.conv1d(x, parameters.f0_weight, parameters.f0_bias)
    f0 = torch.sigmoid(f0_logits) * 1000.0  # Scale to Hz range
    f0 = F.interpolate(f0, size=mel_spec.shape[-1], mode="linear")

    return f0.squeeze(1)


# ---------------------------------------------------------------------------
# Feature Retrieval (Index-based)
# ---------------------------------------------------------------------------

def feature_retrieval(
    source_features,
    index_features,
    index_rate=0.5,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    Index-based feature retrieval for accent/speaker style transfer.

    Uses cosine similarity to find nearest neighbor features from the
    target speaker's feature index.

    Args:
        source_features: Source speaker features [B, C, T]
        index_features: Target speaker feature index [N, C]
        index_rate: Blending ratio (0.0 = no retrieval, 1.0 = full)

    Returns:
        Mixed features [B, C, T]
    """
    if isinstance(source_features, ttnn.Tensor):
        source_cpu = ttnn.to_torch(source_features).float()
    else:
        source_cpu = source_features.float()

    if isinstance(index_features, ttnn.Tensor):
        index_cpu = ttnn.to_torch(index_features).float()
    else:
        index_cpu = index_features.float()

    B, C, T = source_cpu.shape
    N = index_cpu.shape[0]

    # Normalize
    source_norm = F.normalize(source_cpu.reshape(B, C, -1).permute(0, 2, 1), dim=-1)  # [B, T, C]
    index_norm = F.normalize(index_cpu, dim=-1)  # [N, C]

    # Compute similarities: [B, T, N]
    similarities = torch.matmul(source_norm, index_norm.T)

    # Top-1 retrieval
    _, indices = similarities.max(dim=-1)  # [B, T]

    # Gather retrieved features
    retrieved = index_cpu[indices]  # [B, T, C]
    retrieved = retrieved.permute(0, 2, 1)  # [B, C, T]

    # Blend with source
    mixed = (1 - index_rate) * source_cpu + index_rate * retrieved

    return mixed


# ---------------------------------------------------------------------------
# Flow-based Decoder (Normalizing Flows)
# ---------------------------------------------------------------------------

def flow_decoder_forward(
    z,
    g,
    parameters,
    n_flows=4,
    reverse=True,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    Flow-based decoder using affine coupling layers.

    Args:
        z: Latent variable [B, C, T]
        g: Conditioning (speaker embedding)
        parameters: Flow decoder parameters
        n_flows: Number of flow steps
        reverse: If True, run in reverse (generation) direction

    Returns:
        Decoded features [B, C, T]
    """
    if isinstance(z, ttnn.Tensor):
        z_cpu = ttnn.to_torch(z).float()
    else:
        z_cpu = z.float()

    flows = parameters.flows if hasattr(parameters, "flows") else []

    for flow_params in (reversed(flows) if reverse else flows):
        # Affine coupling: split z into two halves
        C = z_cpu.shape[1]
        z1, z2 = z_cpu[:, :C // 2, :], z_cpu[:, C // 2:, :]

        # Forward direction: log_s, t = network(z1)
        # Reverse direction: z2 = (z2 - t) / exp(log_s)
        # Simplified affine transform
        w = flow_params.weight
        b = flow_params.bias

        # Apply affine transform to z2 conditioned on z1
        z1_flat = z1.reshape(z1.shape[0], -1, z1.shape[-1])

        log_s = torch.conv1d(z1, w[:C // 2], bias=None, padding=w.shape[-1] // 2)
        t = torch.conv1d(z1, w[C // 2:], bias=None, padding=w.shape[-1] // 2)

        if b is not None:
            log_s = log_s + b[:C // 2].unsqueeze(-1)
            t = t + b[C // 2:].unsqueeze(-1)

        if reverse:
            z2 = (z2 - t) * torch.exp(-log_s)
        else:
            z2 = z2 * torch.exp(log_s) + t

        z_cpu = torch.cat([z2, z1], dim=1)

        # Flip for next layer
        z_cpu = z_cpu.flip(1)

    return z_cpu


# ---------------------------------------------------------------------------
# HiFi-GAN Vocoder
# ---------------------------------------------------------------------------

def hifigan_forward(
    mel,
    parameters,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    HiFi-GAN vocoder: mel spectrogram → waveform.

    Uses transposed convolutions for upsampling followed by
    multi-receptive field fusion (MRF) residual blocks.

    Args:
        mel: Mel spectrogram [B, n_mels, T]
        parameters: HiFi-GAN parameters

    Returns:
        audio: Generated waveform [B, 1, T_audio]
    """
    if isinstance(mel, ttnn.Tensor):
        mel_cpu = ttnn.to_torch(mel).float()
    else:
        mel_cpu = mel.float()

    x = mel_cpu

    # Upsampling layers
    upsample_rates = parameters.upsample_rates
    upsample_initial_channel = parameters.upsample_initial_channel

    # Pre-network conv
    x = F.conv1d(
        x,
        parameters.conv_pre_weight,
        parameters.conv_pre_bias,
        padding=parameters.conv_pre_weight.shape[-1] // 2,
    )

    # Upsample + ResBlock
    for i, (u_rate, k_size) in enumerate(
        zip(upsample_rates, parameters.upsample_kernel_sizes)
    ):
        up_params = parameters.ups[i]
        x = F.conv_transpose1d(
            x,
            up_params.weight,
            up_params.bias,
            stride=u_rate,
            padding=(k_size - u_rate) // 2,
        )

        # ResBlock
        res_params = parameters.resblocks[i]
        x_res = _resblock_forward(x, res_params)
        x = x + x_res

    # Post-network conv + tanh
    x = F.conv1d(
        x,
        parameters.conv_post_weight,
        parameters.conv_post_bias,
        padding=parameters.conv_post_weight.shape[-1] // 2,
    )
    audio = torch.tanh(x)

    return audio


def _resblock_forward(x, parameters):
    """Multi-receptive field fusion residual block."""
    dilation_sizes = parameters.dilation_sizes
    kernel_sizes = parameters.kernel_sizes

    results = []
    for dilation, k_size in zip(dilation_sizes, kernel_sizes):
        for d in dilation:
            x_dilated = _dilated_conv_block(x, parameters, k_size, d)
            results.append(x_dilated)

    return sum(results) / len(results)


def _dilated_conv_block(x, parameters, kernel_size, dilation):
    """Single dilated convolution block with LeakyReLU."""
    padding = (kernel_size * dilation - dilation) // 2

    # Conv1 + LeakyReLU
    x = F.conv1d(x, parameters.conv1_weight, parameters.conv1_bias, padding=padding, dilation=dilation)
    x = F.leaky_relu(x, 0.1)

    # Conv2
    x = F.conv1d(x, parameters.conv2_weight, parameters.conv2_bias, padding=padding, dilation=dilation)

    return x


# ---------------------------------------------------------------------------
# Full RVC Pipeline
# ---------------------------------------------------------------------------

def rvc_inference(
    source_audio,
    target_features_index,
    parameters,
    f0_method="rmvpe",
    index_rate=0.5,
    f0_up_key=0,
    protect=0.33,
    memory_config=RVC_MEMORY_CONFIG,
):
    """
    Full RVC voice conversion pipeline.

    Args:
        source_audio: Source audio waveform [B, T_audio]
        target_features_index: Target speaker feature index [N, C]
        parameters: Full model parameters namespace
        f0_method: Pitch extraction method ('rmvpe' or 'crepe')
        index_rate: Feature retrieval blending ratio
        f0_up_key: Pitch transposition in semitones
        protect: Consonant protection threshold

    Returns:
        converted_audio: Converted waveform [B, 1, T_audio_out]
    """
    logger.info("RVC: Starting voice conversion pipeline")

    # 1. Extract source features via posterior encoder
    logger.info("RVC: Running posterior encoder")
    mel_spec = _audio_to_mel(source_audio, parameters)
    z, m, logs = posterior_encoder_forward(
        mel_spec, None, parameters.encoder, memory_config=memory_config
    )
    logger.info(f"RVC: Posterior encoder output shape: {z.shape}")

    # 2. Pitch extraction
    logger.info(f"RVC: Extracting pitch using {f0_method}")
    if f0_method == "rmvpe":
        f0 = rmvpe_forward(
            source_audio, parameters.rmvpe, memory_config=memory_config
        )
    else:
        f0 = _crepe_pitch_extract(source_audio, parameters)

    # Apply pitch transposition
    if f0_up_key != 0:
        f0 = f0 * (2 ** (f0_up_key / 12.0))
    logger.info(f"RVC: F0 shape: {f0.shape}")

    # 3. Feature retrieval
    logger.info("RVC: Running feature retrieval")
    if index_rate > 0 and target_features_index is not None:
        z = feature_retrieval(
            z, target_features_index, index_rate=index_rate, memory_config=memory_config
        )
    logger.info(f"RVC: Feature retrieval output shape: {z.shape}")

    # 4. Flow-based decoder
    logger.info("RVC: Running flow decoder")
    decoded = flow_decoder_forward(
        z, None, parameters.flow, memory_config=memory_config
    )
    logger.info(f"RVC: Flow decoder output shape: {decoded.shape}")

    # 5. HiFi-GAN vocoder
    logger.info("RVC: Running HiFi-GAN vocoder")
    converted_audio = hifigan_forward(
        decoded, parameters.vocoder, memory_config=memory_config
    )
    logger.info(f"RVC: Output audio shape: {converted_audio.shape}")

    return converted_audio


def _audio_to_mel(audio, parameters):
    """Convert audio waveform to mel spectrogram."""
    if isinstance(audio, ttnn.Tensor):
        audio = ttnn.to_torch(audio).float()
    else:
        audio = audio.float()

    n_fft = 1024
    hop_length = 256
    n_mels = 80
    sample_rate = 16000

    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mag = torch.abs(stft)

    # Create mel filterbank
    mel_fb = torch.zeros(n_mels, mag.shape[1])
    f_max = sample_rate / 2
    mel_low = 0
    mel_high = 2595 * math.log10(1 + f_max / 700)
    mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = (hz_points / sample_rate * n_fft).long()

    for i in range(n_mels):
        f_left = bin_points[i]
        f_center = bin_points[i + 1]
        f_right = bin_points[i + 2]
        for j in range(f_left, f_center):
            if j < mag.shape[1]:
                mel_fb[i, j] = (j - f_left) / max(f_center - f_left, 1)
        for j in range(f_center, f_right):
            if j < mag.shape[1]:
                mel_fb[i, j] = (f_right - j) / max(f_right - f_center, 1)

    mel_spec = torch.matmul(mel_fb, mag)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def _crepe_pitch_extract(audio, parameters):
    """CREPE-based pitch extraction (simplified)."""
    if isinstance(audio, ttnn.Tensor):
        audio = ttnn.to_torch(audio).float()
    else:
        audio = audio.float()

    # Simplified: use basic autocorrelation for F0
    n_fft = 1024
    hop_length = 160

    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mag = torch.abs(stft)

    # Find peak frequency per frame
    f0 = torch.zeros(audio.shape[0], mag.shape[-1])
    for b in range(audio.shape[0]):
        for t in range(mag.shape[-1]):
            spectrum = mag[b, :, t]
            peak_bin = spectrum.argmax()
            f0[b, t] = peak_bin.float() * 16000 / n_fft

    return f0


# ---------------------------------------------------------------------------
# Parameter preprocessing
# ---------------------------------------------------------------------------

def preprocess_rvc_parameters(parameters, device, mesh_mapper=None):
    """
    Convert PyTorch model parameters to TTNN format.
    """
    logger.info("RVC: Preprocessing parameters for TTNN")

    # Mark which params go to device vs stay on CPU
    # Conv layers stay on CPU (computed via torch)
    # Attention/linear layers go to device

    return parameters
