# tt_cosyvoice2.py
"""
TT-NN implementation of CosyVoice2's HiFT vocoder + iSTFT + streaming pipeline.
Uses TT-NN APIs for device execution and host-device tensor management.
"""

import torch
import tt_lib
from tt_lib import tensor as tt_tensor
from tt_lib import device as tt_device
from tt_lib import model
import numpy as np
from typing import List, Tuple, Optional
import math

# --- Complex number helpers for TT-NN (using interleaved real/imag layout) ---
def torch_to_tt_complex(torch_tensor: torch.Tensor, device: tt_device.Device) -> tt_tensor.Tensor:
    """
    Convert a complex torch tensor (shape [..., 2]) to TT-NN tensor with interleaved real/imag.
    Assumes torch_tensor is real-imag interleaved (e.g., [..., freq, 2]).
    """
    assert torch_tensor.shape[-1] == 2, "Last dim must be 2 for complex (real, imag)"
    tt_tensor_ = tt_tensor.from_torch(torch_tensor, dtype=tt_tensor.DataType.BFLOAT16)
    return tt_tensor_.to(device)

def tt_complex_to_torch(tt_tensor_: tt_tensor.Tensor) -> torch.Tensor:
    """Convert TT-NN interleaved complex tensor back to torch tensor."""
    return tt_tensor_.to_torch()

# --- HiFT Vocoder Blocks (TT-NN compatible) ---
class HiFTConv1d:
    """TT-NN version of 1D convolution with causal/dilated support."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True, device: tt_device.Device = None):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights/bias as torch tensors first
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        weight_torch = torch.empty(out_channels, in_channels // groups, kernel_size)
        torch.nn.init.uniform_(weight_torch, -scale, scale)
        self.weight = weight_torch
        
        self.bias = torch.zeros(out_channels) if bias else None
        
        # Pre-convert to TT tensors for device
        self.tt_weight = tt_tensor.from_torch(self.weight, dtype=tt_tensor.DataType.BFLOAT16).to(device)
        self.tt_bias = tt_tensor.from_torch(self.bias, dtype=tt_tensor.DataType.BFLOAT16).to(device) if bias else None

    def forward(self, x: tt_tensor.Tensor) -> tt_tensor.Tensor:
        """TT-NN 1D convolution (causal padding applied externally)."""
        return tt_tensor.conv2d(
            input_tensor=x,
            weight_tensor=self.tt_weight,
            bias_tensor=self.tt_bias,
            stride=(1, self.stride),
            dilation=(1, self.dilation),
            groups=self.groups,
            padding=(0, 0, (self.kernel_size - 1) * self.dilation, 0),  # Causal padding
            device=self.device
        )

class HiFTResBlock:
    """HiFT residual block with dilated convolutions."""
    def __init__(self, channels: int, dilations: List[int], device: tt_device.Device = None):
        self.device = device
        self.convs1 = []
        self.convs2 = []
        
        for d in dilations:
            self.convs1.append(HiFTConv1d(channels, channels, kernel_size=3, dilation=d, device=device))
            self.convs2.append(HiFTConv1d(channels, channels, kernel_size=3, dilation=1, device=device))
        
        # Activation: leaky ReLU (approximated via TT-NN ops)
        self.alpha = 0.2

    def forward(self, x: tt_tensor.Tensor) -> tt_tensor.Tensor:
        """Forward pass with residual connections."""
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            # Leaky ReLU
            x = tt_tensor.clamp(x, min_val=0.0) + self.alpha * tt_tensor.clamp(x, max_val=0.0)
            x = conv1.forward(x)
            x = tt_tensor.clamp(x, min_val=0.0) + self.alpha * tt_tensor.clamp(x, max_val=0.0)
            x = conv2.forward(x)
            x = tt_tensor.add(x, residual)
        return x

class HiFTVocoder:
    """TT-NN implementation of HiFT vocoder."""
    def __init__(self, 
                 in_channels: int = 80,  # mel-spectrogram channels
                 out_channels: int = 1,   # audio channels
                 hidden_channels: int = 512,
                 upsample_rates: List[int] = [5, 5, 11],
                 resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 device: tt_device.Device = None):
        self.device = device
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Initial projection
        self.conv_pre = HiFTConv1d(in_channels, hidden_channels, kernel_size=7, device=device)
        
        # Upsampling blocks
        self.ups = []
        for i, (upsample_rate, res_dilations) in enumerate(zip(upsample_rates, resblock_dilations)):
            kernel_size = upsample_rate * 2
            self.ups.append(HiFTConv1d(hidden_channels, hidden_channels // (2 ** (i + 1)),
                                      kernel_size=kernel_size, stride=upsample_rate, device=device))
            self.ups.append(HiFTResBlock(hidden_channels // (2 ** (i + 1)), res_dilations, device=device))
        
        # Final conv
        self.conv_post = HiFTConv1d(hidden_channels // (2 ** len(upsample_rates)), out_channels,
                                   kernel_size=7, device=device)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward pass: mel -> audio (returns torch tensor for compatibility)."""
        # Convert mel to TT tensor
        mel_tt = torch_to_tt_complex(mel.unsqueeze(-1).repeat(1, 1, 1, 2), self.device)  # Add complex dim
        
        # Pre-conv
        x = self.conv_pre.forward(mel_tt)
        
        # Upsampling + residual blocks
        for i in range(0, len(self.ups), 2):
            # Transpose for TT-NN conv2d (N, C, H, W)
            x = x.permute(0, 3, 1, 2)  # (N, 2, C, L) -> (N, 2, C, L)
            x = self.ups[i].forward(x)
            x = x.permute(0, 2, 3, 1)  # Back to (N, C, L, 2)
            x = self.ups[i+1].forward(x)
        
        # Post-conv + tanh
        x = self.conv_post.forward(x)
        x = tt_tensor.tanh(x)
        
        # Convert back to torch
        audio = tt_complex_to_torch(x).squeeze(-1)  # Remove complex dim
        return audio

# --- iSTFT Layer ---
class iSTFT:
    """TT-NN-compatible iSTFT layer."""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 window: Optional[torch.Tensor] = None, device: tt_device.Device = None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.device = device
        
        # Pre-compute window and DFT matrix
        if window is None:
            window = torch.hann_window(win_length)
        self.window = window
        
        # DFT matrix (real-imag)
        k = torch.arange(n_fft).unsqueeze(1)
        n = torch.arange(n_fft)
        dft_matrix = torch.exp(-2j * torch.pi * k * n / n_fft)
        self.dft_real = dft_matrix.real
        self.dft_imag = dft_matrix.imag
        
        # Pre-convert to TT
        self.tt_window = tt_tensor.from_torch(self.window, dtype=tt_tensor.DataType.BFLOAT16).to(device)
        self.tt_dft_real = tt_tensor.from_torch(self.dft_real, dtype=tt_tensor.DataType.BFLOAT16).to(device)
        self.tt_dft_imag = tt_tensor.from_torch(self.dft_imag, dtype=tt_tensor.DataType.BFLOAT16).to(device)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Inverse STFT: complex spectrogram (..., freq, time) -> waveform.
        Uses TT-NN for matrix multiplies.
        """
        # spec shape: (batch, freq, time, 2) where last dim is [real, imag]
        batch, freq, time, _ = spec.shape
        assert freq == self.n_fft // 2 + 1, f"Expected {self.n_fft//2+1} freq bins, got {freq}"
        
        # Convert to TT tensor
        spec_tt = torch_to_tt_complex(spec, self.device)
        
        # Reshape for batched matrix multiply: (batch*time, freq)
        spec_tt = spec_tt.reshape(batch * time, freq, 1, 2)
        
        # Compute inverse DFT: IDFT = (1/N) * DFT^H
        # Multiply by conjugate DFT matrix (dft_real - j*dft_imag)
        # Real part: spec_real * dft_real + spec_imag * dft_imag
        # Imag part: spec_imag * dft_real - spec_real * dft_imag
        spec_real = spec_tt.slice((0, 0, 0, 0), (batch*time, freq, 1, 1))
        spec_imag = spec_tt.slice((0, 0, 0, 1), (batch*time, freq, 1, 2))
        
        # Matrix multiply: (freq, n_fft) x (freq, 1)
        # Use TT-NN matmul (treat as 2D)
        spec_real_2d = spec_real.reshape(batch*time, freq)
        spec_imag_2d = spec_imag.reshape(batch*time, freq)
        
        dft_real_t = self.tt_dft_real.transpose(0, 1)  # (n_fft, freq)
        dft_imag_t = self.tt_dft_imag.transpose(0, 1)
        
        # Real part: DFT_real^T * spec_real + DFT_imag^T * spec_imag
        real_part = tt_tensor.matmul(dft_real_t, spec_real_2d)
        imag_part = tt_tensor.matmul(dft_imag_t, spec_imag_2d)
        real_part = tt_tensor.add(real_part, tt_tensor.matmul(dft_imag_t, spec_imag_2d))
        imag_part = tt_tensor.sub(tt_tensor.matmul(dft_imag_t, spec_real_2d), imag_part)
        
        # Combine and apply window (overlap-add)
        waveform = self._overlap_add(real_part, imag_part, time)
        
        # Normalize
        waveform = tt_tensor.mul(waveform, 1.0 / self.n_fft)
        
        return tt_complex_to_torch(waveform).squeeze(0)  # Return (time,) for single sample

    def _overlap_add(self, real_part: tt_tensor.Tensor, imag_part: tt_tensor.Tensor, time: int) -> tt_tensor.Tensor:
        """Overlap-add for STFT frames."""
        # Reshape to (batch, time, n_fft)
        real_3d = real_part.reshape(1, time, self.n_fft)
        imag_3d = imag_part.reshape(1, time, self.n_fft)
        
        # Create overlap-add matrix (TT-NN doesn't have built-in overlap-add, so manual)
        # Use stride tricks equivalent in TT-NN (slicing and accumulation)
        waveform_length = (time - 1) * self.hop_length + self.win_length
        waveform = torch.zeros(1, waveform_length)
        window_sum = torch.zeros(1, waveform_length)
        
        # Process each frame
        for i in range(time):
            start = i * self.hop_length
            frame_real = real_3d[0, i, :].unsqueeze(0)
            frame_imag = imag_3d[0, i, :].unsqueeze(0)
            
            # Apply window
            frame_real = frame_real * self.window
            frame_imag = frame_imag * self.window
            
            # Accumulate
            waveform[0, start:start+self.win_length] += frame_real
            window_sum[0, start:start+self.win_length] += self.window
        
        # Normalize by window sum (avoid division by zero)
        window_sum = torch.where(window_sum < 1e-8, torch.ones_like(window_sum), window_sum)
        waveform = waveform / window_sum
        
        return torch_to_tt_complex(waveform.unsqueeze(-1), self.device)  # Add complex dim

# --- Streaming Pipeline ---
class StreamingCosyVoice2:
    """Streaming pipeline for CosyVoice2."""
    def __init__(self, 
                 vocoder: HiFTVocoder,
                 istft: iSTFT,
                 chunk_size: int = 1024,
                 hop_size: int = 256,
                 device: tt_device.Device = None):
        self.vocoder = vocoder
        self.istft = istft
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.device = device
        
        # Buffer for streaming
        self.mel_buffer = torch.zeros(1, 0, 80)  # (batch, time, mel_dim)
        self.audio_buffer = torch.zeros(0)
        
        # Pre-compute overlap region
        self.overlap_length = self.chunk_size - self.hop_size

    def process_chunk(self, mel_chunk: torch.Tensor) -> torch.Tensor:
        """
        Process a chunk of mel-spectrograms.
        Args:
            mel_chunk: (batch, time, mel_dim)
        Returns:
            audio_chunk: (batch, time * hop_size)
        """
        # Append to buffer
        self.mel_buffer = torch.cat([self.mel_buffer, mel_chunk], dim=1)
        
        # Process if we have enough frames
        if self.mel_buffer.size(1) >= self.chunk_size:
            # Extract chunk
            mel_chunk_proc = self.mel_buffer[:, :self.chunk_size, :]
            
            # Vocode
            audio_chunk = self.vocoder.forward(mel_chunk_proc)
            
            # iSTFT (if vocoder doesn't include it)
            if self.istft is not None:
                # Reshape for iSTFT: (batch, freq, time, 2)
                # Assume vocoder output is (batch, time) -> convert to spec
                # For simplicity, assume vocoder already does iSTFT or output is waveform
                pass
            
            # Overlap-add with previous chunk
            if self.audio_buffer.numel() > 0:
                # Overlap region
                overlap_audio = (self.audio_buffer[-self.overlap_length:] + 
                               audio_chunk[:self.overlap_length]) / 2
                self.audio_buffer = torch.cat([
                    self.audio_buffer[:-self.overlap_length],
                    overlap_audio,
                    audio_chunk[self.overlap_length:]
                ])
            else:
                self.audio_buffer = audio_chunk
            
            # Remove processed frames
            self.mel_buffer = self.mel_buffer[:, self.hop_size:, :]
            
            return audio_chunk[:self.hop_size]  # Return new audio samples
        else:
            return torch.zeros(0)  # Not enough data yet

    def flush(self) -> torch.Tensor:
        """Flush remaining buffer."""
        if self.mel_buffer.size(1) > 0:
            # Zero-pad to chunk_size
            pad_size = self.chunk_size - self.mel_buffer.size(1)
            mel_padded = torch.cat([self.mel_buffer, 
                                   torch.zeros(1, pad_size, 80)], dim=1)
            audio = self.vocoder.forward(mel_padded)
            self.audio_buffer = torch.cat([self.audio_buffer, audio])
        return self.audio_buffer

# --- Example usage ---
if __name__ == "__main__":
    # Initialize device
    device = tt_device.CreateDevice(0)
    
    # Initialize components
    vocoder = HiFTVocoder(device=device)
    istft = iSTFT(device=device)
    streaming_pipeline = StreamingCosyVoice2(vocoder, istft, device=device)
    
    # Simulate mel-spectrogram chunks
    mel_chunks = [torch.randn(1, 10, 80) for _ in range(5)]  # 5 chunks
    
    # Process chunks
    for i, mel_chunk in enumerate(mel_chunks):
        audio_chunk = streaming_pipeline.process_chunk(mel_chunk)
        if audio_chunk.numel() > 0:
            print(f"Chunk {i}: Generated {audio_chunk.shape[0]} audio samples")
    
    # Flush remaining
    final_audio = streaming_pipeline.flush()
    print(f"Final audio length: {final_audio.shape[0]} samples")
    
    # Cleanup
    tt_device.DisconnectDevice(device)