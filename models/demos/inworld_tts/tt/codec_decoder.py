"""TTNN implementation of the full Inworld TTS codec decoder.

Pipeline: FSQ dequantize (CPU) -> fc_post_a Linear(2048,1024) -> VocosBackbone -> ISTFTHead (CPU).

CPU boundaries:
- FSQ dequantize: codebook lookup, stays on CPU
- ISTFTHead: ISTFT signal processing (FFT), stays on CPU

TTNN accelerated:
- fc_post_a: Linear projection
- VocosBackbone: 12 transformer layers (main compute)
"""

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.model_config import (
    ISTFT_HOP_LENGTH,
    ISTFT_N_FFT,
    VOCOS_DEPTH,
    VOCOS_DIM,
    VOCOS_HEADS,
    VOCOS_POS_EMB_DIM,
    get_compute_kernel_config_hifi4,
)
from models.demos.inworld_tts.tt.vocos_backbone import TtVocosBackbone


class TtCodecDecoder(LightweightModule):
    """Full codec decoder: FSQ dequant -> fc_post_a -> VocosBackbone -> ISTFTHead."""

    def __init__(
        self,
        device,
        state_dict,
        quantizer=None,
        dim=VOCOS_DIM,
        depth=VOCOS_DEPTH,
        n_heads=VOCOS_HEADS,
        pos_emb_dim=VOCOS_POS_EMB_DIM,
        dtype=ttnn.bfloat16,
        backbone_prefix="backbone.",
        head_prefix="head.",
    ):
        super().__init__()
        self.device = device
        self.quantizer = quantizer

        # Full compute grid
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        # fc_post_a: Linear(2048, 1024) -- on device
        fc_w = state_dict["fc_post_a.weight"]
        fc_b = state_dict["fc_post_a.bias"]
        self.fc_post_a_weight = ttnn.from_torch(
            fc_w.T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.fc_post_a_bias = ttnn.from_torch(
            fc_b.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # VocosBackbone
        self.backbone = TtVocosBackbone(
            device=device,
            state_dict=state_dict,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            pos_emb_dim=pos_emb_dim,
            dtype=dtype,
            state_dict_prefix=backbone_prefix,
        )

        # ISTFTHead weights (CPU)
        self.istft_out_weight = state_dict[head_prefix + "out.weight"]
        self.istft_out_bias = state_dict[head_prefix + "out.bias"]

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def fsq_dequantize(self, indices):
        """FSQ dequantize using the quantizer module (CPU).

        Args:
            indices: [B, 1, T] or [B, T] integer VQ codes
        Returns:
            [B, T, 2048] torch tensor
        """
        if self.quantizer is None:
            raise ValueError("Quantizer required for FSQ dequantization")

        if indices.dim() == 2:
            indices = indices.unsqueeze(1)

        codes = indices.transpose(1, 2)  # [B, T, 1]
        return self.quantizer.get_output_from_indices(codes)

    def istft_head(self, x_torch):
        """ISTFTHead: linear -> split mag/phase -> exp -> complex -> ISTFT (CPU).

        Args:
            x_torch: [B, T, 1024] torch tensor
        Returns:
            [B, 1, num_samples] torch tensor
        """
        n_fft = ISTFT_N_FFT
        hop_length = ISTFT_HOP_LENGTH
        win_length = n_fft

        # Linear projection (ensure float32 for FFT math)
        x_torch = x_torch.float()
        x_pred = F.linear(x_torch, self.istft_out_weight.float(), self.istft_out_bias.float())  # [B, T, 1282]
        x_pred = x_pred.transpose(1, 2)  # [B, 1282, T]

        # Split magnitude and phase
        mag, p = x_pred.chunk(2, dim=1)  # each [B, 641, T]

        # Magnitude activation
        mag = torch.exp(mag)
        mag = torch.clamp(mag, max=1e2)

        # Complex spectrogram
        S = mag * (torch.cos(p) + 1j * torch.sin(p))

        # ISTFT with "same" padding
        pad = (win_length - hop_length) // 2
        window = torch.hann_window(win_length, device=S.device)

        B, N, T_frames = S.shape

        # Inverse FFT
        ifft = torch.fft.irfft(S, n_fft, dim=1, norm="backward")
        ifft = ifft * window[None, :, None]

        # Overlap and add
        output_size = (T_frames - 1) * hop_length + win_length
        y = F.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope normalization
        window_sq = window.square().expand(1, T_frames, -1).transpose(1, 2)
        window_envelope = F.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, win_length),
            stride=(1, hop_length),
        ).squeeze()[pad:-pad]

        y = y / window_envelope

        return y.unsqueeze(1)

    def forward(self, vq_codes):
        """Full codec decoder forward.

        Args:
            vq_codes: [B, T] or [B, 1, T] integer VQ codes (torch tensor)
        Returns:
            [B, 1, num_samples] audio waveform (torch tensor)
        """
        # Step 1: FSQ dequantize (CPU)
        vq_emb = self.fsq_dequantize(vq_codes)  # [B, T, 2048]

        # Step 2: fc_post_a projection (TTNN)
        vq_emb_ttnn = ttnn.from_torch(
            vq_emb.unsqueeze(0),  # [1, B, T, 2048]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        projected = ttnn.linear(
            vq_emb_ttnn,
            self.fc_post_a_weight,
            bias=self.fc_post_a_bias,
            core_grid=self.core_grid,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, 1, T, 1024]

        # Step 3: VocosBackbone (TTNN)
        hidden = self.backbone(projected)  # [1, 1, T, 1024]

        # Step 4: ISTFTHead (CPU)
        hidden_torch = ttnn.to_torch(hidden)  # [1, 1, T, 1024]
        if hidden_torch.dim() == 4:
            hidden_torch = hidden_torch.squeeze(0)  # [1, T, 1024]

        audio = self.istft_head(hidden_torch)  # [1, 1, num_samples]

        return audio

    def forward_from_embeddings(self, vq_emb):
        """Forward pass from pre-dequantized embeddings (for testing without quantizer).

        Args:
            vq_emb: [B, T, 1024] torch tensor (already projected by fc_post_a)
        Returns:
            [1, 1, T, 1024] ttnn tensor (VocosBackbone output, before ISTFT)
        """
        vq_emb_ttnn = ttnn.from_torch(
            vq_emb.unsqueeze(0) if vq_emb.dim() == 3 else vq_emb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        return self.backbone(vq_emb_ttnn)
