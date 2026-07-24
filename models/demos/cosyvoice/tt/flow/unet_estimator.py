"""CausalConditionalDecoder UNet1D estimator (Stage 1, host-side torch).

Stage 1: wraps the reference CausalConditionalDecoder from CosyVoice source,
loading weights from flow.pt. This is functionally correct and matches golden
fixtures exactly. Device port (TTNN) is Stage 2 optimization.

Architecture (§4.2.1):
  - SinusoidalPosEmb(320) → TimestepEmbedding(320→1024, silu)
  - Input assembly: pack([x, mu, spks_expanded, cond]) → [B, 320, T]
  - down_blocks(1): CausalResnetBlock1D(320→256) + 4× BasicTransformerBlock + CausalConv1d(256,256,k=3)
  - mid_blocks(12): CausalResnetBlock1D(256→256) + 4× BasicTransformerBlock each
  - up_blocks(1): skip concat → CausalResnetBlock1D(512→256) + 4× BasicTransformerBlock + CausalConv1d(256,256,k=3)
  - final_block: CausalBlock1D(256, 256) + final_proj: Conv1d(256→80, k=1)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

_COSYVOICE_SRC = str(Path(__file__).resolve().parents[3] / "model_data" / "CosyVoice_src")
_MATCHA = str(Path(_COSYVOICE_SRC) / "third_party" / "Matcha-TTS")
if _COSYVOICE_SRC not in sys.path:
    sys.path.insert(0, _COSYVOICE_SRC)
if _MATCHA not in sys.path:
    sys.path.append(_MATCHA)


class UNetEstimator:
    """Wraps the reference CausalConditionalDecoder with flow.pt weights.

    Stage 1: host-side torch (correctness-first). Device port = Stage 2.
    """

    def __init__(self, decoder_weights: Dict[str, torch.Tensor]):
        from cosyvoice.flow.decoder import CausalConditionalDecoder

        self.model = CausalConditionalDecoder(
            in_channels=320,
            out_channels=80,
            channels=(256,),
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
            static_chunk_size=50,
            num_decoding_left_chunks=-1,
        )

        est_sd = {
            k.replace("decoder.estimator.", ""): v
            for k, v in decoder_weights.items()
            if k.startswith("decoder.estimator.")
        }
        missing, unexpected = self.model.load_state_dict(est_sd, strict=False)
        assert not unexpected, f"Unexpected keys: {unexpected[:5]}..."
        assert not missing, f"Missing keys: {missing[:5]}..."
        self.model.eval()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        """Estimator forward.

        Args:
            x: [B, 80, T] noisy mel
            mask: [B, 1, T]
            mu: [B, 80, T] conditioning mel
            t: [B] timestep
            spks: [B, 80] speaker embedding
            cond: [B, 80, T] prompt mel condition

        Returns:
            [B, 80, T] predicted velocity
        """
        return self.model(x, mask, mu, t, spks=spks, cond=cond, streaming=streaming)
