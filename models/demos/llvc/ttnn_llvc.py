# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import json
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from loguru import logger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

import ttnn
from models.common.utility_functions import nearest_32

LLVC_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
LLVC_BATCH_SIZE = 1
LLVC_L1_SMALL_SIZE = 1024
LLVC_TRACE_REGION_SIZE = 100000000


def gelu(tensor):
    return ttnn.gelu(tensor, memory_config=LLVC_MEMORY_CONFIG)


def dropout(hidden_states, p, training):
    # ignored for inference
    return hidden_states


def unsqueeze_to_4D_at_dim_1(tensor):
    rank = len(tensor.shape)
    if rank == 4:
        return tensor
    elif rank == 3:
        return ttnn.unsqueeze(tensor, 1)
    elif rank == 2:
        return ttnn.unsqueeze(ttnn.unsqueeze(tensor, 1), 1)
    else:
        raise ValueError(f"Unsupported shape: {tensor.shape}")


class TtDilatedCausalConvEncoder:
    """TTNN implementation of the Dilated Causal Convolution Encoder"""

    def __init__(self, channels, num_layers, kernel_size=3, device=None):
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.device = device

        # Compute buffer lengths for each layer
        self.buf_lengths = [(kernel_size - 1) * 2**i for i in range(num_layers)]

        # Compute buffer start indices for each layer
        self.buf_indices = [0]
        for i in range(num_layers - 1):
            self.buf_indices.append(self.buf_indices[-1] + self.buf_lengths[i])

        # Create depthwise separable conv layers
        self.dcc_layers = []
        for i in range(num_layers):
            dilation = 2**i
            # Depthwise conv
            self.dcc_layers.append(
                ttnn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, kernel_size),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, dilation),
                    groups=channels,
                    bias=False,
                    device=device,
                    memory_config=LLVC_MEMORY_CONFIG,
                )
            )
            # Pointwise conv
            self.dcc_layers.append(
                ttnn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                    device=device,
                    memory_config=LLVC_MEMORY_CONFIG,
                )
            )

    def init_ctx_buf(self, batch_size):
        """Initialize context buffer"""
        total_buf_len = (self.kernel_size - 1) * (2**self.num_layers - 1)
        return ttnn.zeros(
            (batch_size, self.channels, total_buf_len),
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=LLVC_MEMORY_CONFIG,
        )

    def forward(self, x, ctx_buf):
        """Forward pass with context buffer management"""
        for i in range(self.num_layers):
            buf_start_idx = self.buf_indices[i]
            buf_end_idx = self.buf_indices[i] + self.buf_lengths[i]

            # Extract context for this layer
            ctx_slice = ttnn.slice(ctx_buf, [0, 0, buf_start_idx], [ctx_buf.shape[0], ctx_buf.shape[1], buf_end_idx])

            # Concatenate context with input
            dcc_in = ttnn.concat([ctx_slice, x], dim=-1)

            # Update context buffer
            ctx_buf = ttnn.concat([
                ttnn.slice(ctx_buf, [0, 0, 0], [ctx_buf.shape[0], ctx_buf.shape[1], buf_start_idx]),
                ttnn.slice(dcc_in, [0, 0, dcc_in.shape[-1] - self.buf_lengths[i]], dcc_in.shape),
                ttnn.slice(ctx_buf, [0, 0, buf_end_idx], ctx_buf.shape)
            ], dim=-1)

            # Apply depthwise separable conv
            # Depthwise
            x_dw = self.dcc_layers[i*2](ttnn.unsqueeze(dcc_in, 1))
            x_dw = ttnn.squeeze(x_dw, 1)

            # Pointwise
            x_pw = self.dcc_layers[i*2 + 1](ttnn.unsqueeze(x_dw, 1))
            x_pw = ttnn.squeeze(x_pw, 1)

            # Residual connection
            x = ttnn.add(x, x_pw)

        return x, ctx_buf


class TtCausalTransformerDecoder:
    """TTNN implementation of the Causal Transformer Decoder"""

    def __init__(self, model_dim, ctx_len, chunk_size, num_layers, nhead, use_pos_enc, ff_dim, dropout, device=None):
        self.model_dim = model_dim
        self.ctx_len = ctx_len
        self.chunk_size = chunk_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.use_pos_enc = use_pos_enc
        self.device = device

        # Create transformer decoder layers
        self.tf_dec_layers = []
        for _ in range(num_layers):
            # We'll implement a simplified version for now
            # In full implementation, this would include self-attention and cross-attention
            self.tf_dec_layers.append(
                ttnn.Linear(model_dim, model_dim, bias=False, device=device, memory_config=LLVC_MEMORY_CONFIG)
            )

    def init_ctx_buf(self, batch_size):
        """Initialize context buffer"""
        return ttnn.zeros(
            (batch_size, self.num_layers + 1, self.ctx_len, self.model_dim),
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=LLVC_MEMORY_CONFIG,
        )

    def forward(self, tgt, mem, ctx_buf):
        """Forward pass"""
        # Simplified implementation - full version would include proper causal attention
        # For now, just pass through with basic linear transformation
        for layer in self.tf_dec_layers:
            tgt = layer(tgt)

        return tgt, ctx_buf


class TtMaskNet:
    """TTNN implementation of the Mask Network"""

    def __init__(self, enc_dim, num_enc_layers, dec_dim, dec_buf_len, dec_chunk_size,
                 num_dec_layers, use_pos_enc, skip_connection, proj, decoder_dropout, device=None):
        self.skip_connection = skip_connection
        self.proj = proj
        self.device = device

        # Encoder
        self.encoder = TtDilatedCausalConvEncoder(
            channels=enc_dim,
            num_layers=num_enc_layers,
            device=device
        )

        # Decoder
        self.decoder = TtCausalTransformerDecoder(
            model_dim=dec_dim,
            ctx_len=dec_buf_len,
            chunk_size=dec_chunk_size,
            num_layers=num_dec_layers,
            nhead=8,
            use_pos_enc=use_pos_enc,
            ff_dim=2 * dec_dim,
            dropout=decoder_dropout,
            device=device
        )

        if proj:
            # Projection layers would be implemented here
            pass

    def forward(self, x, l, enc_buf, dec_buf):
        """Forward pass"""
        # Encode input
        e, enc_buf = self.encoder(x, enc_buf)

        # Label integration (simplified)
        # In full implementation, this would integrate the label embedding

        # Generate mask
        m, dec_buf = self.decoder(l, e, dec_buf)

        # Apply projections if needed
        if self.proj:
            # Projection logic would go here
            pass

        # Skip connection
        if self.skip_connection:
            # Skip connection logic
            pass

        return m, enc_buf, dec_buf


class TtLLVCModel:
    """TTNN implementation of LLVC model"""

    def __init__(self, label_len=1, L=8, enc_dim=512, num_enc_layers=10, dec_dim=256,
                 dec_buf_len=100, num_dec_layers=2, dec_chunk_size=72, out_buf_len=2,
                 use_pos_enc=True, skip_connection=True, proj=True, lookahead=True,
                 decoder_dropout=0.0, convnet_config=None, device=None):
        self.L = L
        self.dec_chunk_size = dec_chunk_size
        self.out_buf_len = out_buf_len
        self.enc_dim = enc_dim
        self.lookahead = lookahead
        self.device = device

        # Input convolution
        kernel_size = 3 * L if lookahead else L
        self.in_conv = ttnn.Conv2d(
            in_channels=1,
            out_channels=enc_dim,
            kernel_size=(1, kernel_size),
            stride=(1, L),
            padding=(0, 0),
            bias=False,
            device=device,
            memory_config=LLVC_MEMORY_CONFIG,
        )

        # Label embedding (simplified)
        self.label_embedding = ttnn.Linear(label_len, enc_dim, bias=False, device=device, memory_config=LLVC_MEMORY_CONFIG)

        # Mask generator
        self.mask_gen = TtMaskNet(
            enc_dim=enc_dim,
            num_enc_layers=num_enc_layers,
            dec_dim=dec_dim,
            dec_buf_len=dec_buf_len,
            dec_chunk_size=dec_chunk_size,
            num_dec_layers=num_dec_layers,
            use_pos_enc=use_pos_enc,
            skip_connection=skip_connection,
            proj=proj,
            decoder_dropout=decoder_dropout,
            device=device
        )

        # Output convolution
        self.out_conv = ttnn.ConvTranspose2d(
            in_channels=enc_dim,
            out_channels=1,
            kernel_size=(1, (out_buf_len + 1) * L),
            stride=(1, L),
            padding=(0, out_buf_len * L),
            bias=False,
            device=device,
            memory_config=LLVC_MEMORY_CONFIG,
        )

    def init_buffers(self, batch_size):
        """Initialize all buffers"""
        enc_buf = self.mask_gen.encoder.init_ctx_buf(batch_size)
        dec_buf = self.mask_gen.decoder.init_ctx_buf(batch_size)
        out_buf = ttnn.zeros(
            (batch_size, self.enc_dim, self.out_buf_len),
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=LLVC_MEMORY_CONFIG,
        )
        return enc_buf, dec_buf, out_buf

    def forward(self, x, init_enc_buf=None, init_dec_buf=None, init_out_buf=None, pad=True):
        """Forward pass"""
        # Handle padding
        if pad:
            pad_size = (self.L, self.L) if self.lookahead else (0, 0)
            # Padding logic would go here

        # Initialize buffers if not provided
        if init_enc_buf is None or init_dec_buf is None or init_out_buf is None:
            enc_buf, dec_buf, out_buf = self.init_buffers(x.shape[0])
        else:
            enc_buf, dec_buf, out_buf = init_enc_buf, init_dec_buf, init_out_buf

        # Input convolution
        x = ttnn.unsqueeze(x, 1)  # Add channel dimension for 2D conv
        x = self.in_conv(x)
        x = ttnn.squeeze(x, 1)  # Remove channel dimension

        # Generate label embedding
        label = ttnn.zeros((x.shape[0], 1), dtype=ttnn.bfloat16, device=self.device, memory_config=LLVC_MEMORY_CONFIG)
        l = self.label_embedding(label)

        # Generate mask
        m, enc_buf, dec_buf = self.mask_gen(x, l, enc_buf, dec_buf)

        # Apply mask
        x = ttnn.mul(x, m)

        # Concatenate with output buffer
        x = ttnn.concat([out_buf, x], dim=-1)

        # Update output buffer
        out_buf = ttnn.slice(x, [0, 0, x.shape[-1] - self.out_buf_len], x.shape)

        # Output convolution
        x = ttnn.unsqueeze(x, 1)
        x = self.out_conv(x)
        x = ttnn.squeeze(x, 1)

        # Remove padding if needed
        # Padding removal logic would go here

        if init_enc_buf is None:
            return x
        else:
            return x, enc_buf, dec_buf, out_buf


def load_llvc_model(checkpoint_path, config_path, device):
    """Load LLVC model from checkpoint"""
    # Load config
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create model
    model = TtLLVCModel(device=device, **config['model_params'])

    # Load weights (simplified - would need proper weight loading)
    # For now, we'll assume weights are already loaded or use random weights

    return model, config['data']['sr']


def preprocess_audio(audio, sample_rate, target_sr):
    """Preprocess audio for LLVC"""
    # Resample if needed
    if sample_rate != target_sr:
        # Resampling logic would go here
        pass

    # Convert to tensor
    audio_tensor = ttnn.from_torch(audio.unsqueeze(0), dtype=ttnn.bfloat16, device=None, memory_config=LLVC_MEMORY_CONFIG)

    return audio_tensor


def infer_llvc(model, audio, streaming=False, chunk_factor=1):
    """Run LLVC inference"""
    if streaming:
        return infer_stream_llvc(model, audio, chunk_factor)
    else:
        return model(audio)


def infer_stream_llvc(model, audio, chunk_factor):
    """Streaming inference for LLVC"""
    L = model.L
    chunk_len = model.dec_chunk_size * L * chunk_factor

    # Pad audio
    original_len = audio.shape[-1]
    if original_len % chunk_len != 0:
        pad_len = chunk_len - (original_len % chunk_len)
        audio = ttnn.concat([audio, ttnn.zeros((audio.shape[0], pad_len), dtype=audio.dtype, device=audio.device, memory_config=LLVC_MEMORY_CONFIG)], dim=-1)

    # Initialize buffers
    enc_buf, dec_buf, out_buf = model.init_buffers(audio.shape[0])

    # Split into chunks
    audio_chunks = []
    for i in range(0, audio.shape[-1], chunk_len):
        chunk = ttnn.slice(audio, [0, i], [audio.shape[0], min(i + chunk_len, audio.shape[-1])])
        audio_chunks.append(chunk)

    # Add lookahead context
    new_audio_chunks = []
    for i, chunk in enumerate(audio_chunks):
        if i == 0:
            front_ctx = ttnn.zeros((chunk.shape[0], L * 2), dtype=chunk.dtype, device=chunk.device, memory_config=LLVC_MEMORY_CONFIG)
        else:
            front_ctx = ttnn.slice(audio_chunks[i-1], [0, audio_chunks[i-1].shape[-1] - L * 2], audio_chunks[i-1].shape)
        new_audio_chunks.append(ttnn.concat([front_ctx, chunk], dim=-1))
    audio_chunks = new_audio_chunks

    outputs = []
    for chunk in audio_chunks:
        output, enc_buf, dec_buf, out_buf = model(
            chunk.unsqueeze(0),
            enc_buf, dec_buf, out_buf,
            pad=(not model.lookahead)
        )
        outputs.append(output)

    # Concatenate outputs
    result = ttnn.concat(outputs, dim=-1)

    # Remove padding
    result = ttnn.slice(result, [0, 0], [result.shape[0], original_len])

    return result
