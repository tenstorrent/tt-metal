# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Main Synthesizer model for OpenVoice V2.

Integrates all components for voice conversion.
"""

from typing import Any, Dict, Optional, Tuple

import torch

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.generator import TTNNGenerator
from models.demos.openvoice.tt.posterior_encoder import TTNNPosteriorEncoder
from models.demos.openvoice.tt.reference_encoder import TTNNReferenceEncoder
from models.demos.openvoice.tt.residual_coupling import TTNNResidualCouplingBlock


class TTNNSynthesizerTrn:
    """
    Main synthesizer for voice conversion.

    For voice conversion (n_speakers=0), uses:
        - ref_enc: ReferenceEncoder to extract speaker embeddings
        - enc_q: PosteriorEncoder to encode mel spectrograms
        - flow: ResidualCouplingBlock for latent transformation
        - dec: Generator to synthesize audio

    Voice conversion pipeline:
        1. Extract source speaker embedding from source audio
        2. Extract target speaker embedding from reference audio
        3. Encode source audio to latent (enc_q)
        4. Transform latent: source → neutral (flow forward)
        5. Transform latent: neutral → target (flow reverse)
        6. Decode to audio (dec)
    """

    def __init__(
        self,
        n_speakers: int,
        gin_channels: int,
        inter_channels: int = 192,
        hidden_channels: int = 192,
        spec_channels: int = 513,
        enc_q: Optional[TTNNPosteriorEncoder] = None,
        dec: Optional[TTNNGenerator] = None,
        flow: Optional[TTNNResidualCouplingBlock] = None,
        ref_enc: Optional[TTNNReferenceEncoder] = None,
        zero_g: bool = False,
        device: Optional[Any] = None,
        **kwargs,  # Accept extra config fields
    ):
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.spec_channels = spec_channels
        self.zero_g = zero_g
        self.device = device

        self.enc_q = enc_q
        self.dec = dec
        self.flow = flow
        self.ref_enc = ref_enc

    def voice_conversion(
        self,
        y: Any,
        y_lengths: Any,
        sid_src: Any,
        sid_tgt: Any,
        tau: float = 1.0,
    ) -> Tuple[Any, Any, Tuple]:
        """
        Perform voice conversion.

        Args:
            y: Source mel spectrogram [B, spec_channels, T]
            y_lengths: Lengths [B]
            sid_src: Source speaker embedding [B, gin_channels, 1]
            sid_tgt: Target speaker embedding [B, gin_channels, 1]
            tau: Temperature for sampling

        Returns:
            Tuple of (converted_audio, mask, intermediate_tensors)
        """
        g_src = sid_src
        g_tgt = sid_tgt

        # Use zero conditioning if specified
        # Check if inputs are PyTorch tensors
        is_torch = isinstance(y, torch.Tensor)

        if self.zero_g:
            if not TTNN_AVAILABLE or is_torch:
                g_src_enc = torch.zeros_like(g_src)
                g_tgt_dec = torch.zeros_like(g_tgt)
            else:
                g_src_enc = ttnn.zeros_like(g_src)
                g_tgt_dec = ttnn.zeros_like(g_tgt)
        else:
            g_src_enc = g_src
            g_tgt_dec = g_tgt

        # Encode source audio
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src_enc, tau=tau)

        # Forward flow: source → neutral latent
        z_p = self.flow(z, y_mask, g=g_src, reverse=False)

        # Reverse flow: neutral → target latent
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)

        # Decode to audio
        if not TTNN_AVAILABLE or is_torch:
            o_hat = self.dec(z_hat * y_mask, g=g_tgt_dec)
        else:
            masked_z = ttnn.multiply(z_hat, y_mask)
            o_hat = self.dec(masked_z, g=g_tgt_dec)

        return o_hat, y_mask, (z, z_p, z_hat)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any],
        config: Dict[str, Any],
        device: Optional[Any] = None,
    ) -> "TTNNSynthesizerTrn":
        """
        Create SynthesizerTrn from state dict and config.

        Args:
            state_dict: Model weights (with weight norm fused)
            config: Model hyperparameters
            device: TTNN device

        Returns:
            Initialized TTNNSynthesizerTrn
        """
        # Extract config values
        model_cfg = config.get("model", config)
        data_cfg = config.get("data", {})

        n_speakers = data_cfg.get("n_speakers", 0)
        gin_channels = model_cfg.get("gin_channels", 256)
        inter_channels = model_cfg.get("inter_channels", 192)
        hidden_channels = model_cfg.get("hidden_channels", 192)
        spec_channels = data_cfg.get("filter_length", 1024) // 2 + 1
        zero_g = model_cfg.get("zero_g", False)

        # Build components for voice conversion (n_speakers=0)
        # PosteriorEncoder
        enc_q = TTNNPosteriorEncoder.from_state_dict(
            state_dict,
            prefix="enc_q",
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels,
            device=device,
        )

        # Generator
        dec = TTNNGenerator.from_state_dict(
            state_dict,
            prefix="dec",
            initial_channel=inter_channels,
            resblock=model_cfg.get("resblock", "1"),
            resblock_kernel_sizes=model_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
            resblock_dilation_sizes=model_cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            upsample_rates=model_cfg.get("upsample_rates", [8, 8, 2, 2]),
            upsample_initial_channel=model_cfg.get("upsample_initial_channel", 512),
            upsample_kernel_sizes=model_cfg.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            gin_channels=gin_channels,
            device=device,
        )

        # Flow
        flow = TTNNResidualCouplingBlock.from_state_dict(
            state_dict,
            prefix="flow",
            channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            n_flows=4,
            gin_channels=gin_channels,
            device=device,
        )

        # ReferenceEncoder (for n_speakers=0)
        ref_enc = None
        if n_speakers == 0:
            ref_enc = TTNNReferenceEncoder.from_state_dict(
                state_dict,
                prefix="ref_enc",
                spec_channels=spec_channels,
                gin_channels=gin_channels,
                device=device,
            )

        return cls(
            n_speakers=n_speakers,
            gin_channels=gin_channels,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            spec_channels=spec_channels,
            enc_q=enc_q,
            dec=dec,
            flow=flow,
            ref_enc=ref_enc,
            zero_g=zero_g,
            device=device,
        )
