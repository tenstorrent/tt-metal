"""HiFTGenerator vocoder — Stage 1 implementation (host-side torch).

Architecture (from cosyvoice2.yaml + source):
  1. f0_predictor: ConvRNNF0Predictor — 5× [weight_norm(Conv1d k=3 pad=1) + ELU] (80→512) + Linear(512→1) + abs()
  2. f0_upsamp: Upsample(scale_factor=480) on f0
  3. m_source: SourceModuleHnNSF(SineGen2, 24kHz, 8 harmonics) → sine_merge + noise
  4. decode(mel, source):
     - conv_pre: weight_norm(Conv1d(80→512, k=7, pad=3))
     - 3× upsample stages: leaky_relu → ConvTranspose1d → [reflection_pad on last] → source fusion → 3× ResBlock(Snake)
     - leaky_relu → conv_post(64→18, k=7) → exp(magnitude) + sin(phase) → iSTFT(n_fft=16, hop=4) → clamp(±0.99)

Stage 1: wraps the reference HiFTGenerator from CosyVoice source with weight-norm
folded (torch 2.x parametrizations API). Runs entirely on host (torch). The vocoder
is NOT perf-critical (runs once per utterance). Device optimization = Stage 2.

Weight-norm fold (U15 RESOLVED):
  hift.pt stores weight-norm as torch 2.x `parametrizations.weight.original0/original1`.
  We fold via `torch.nn.utils.parametrize.remove_parametrizations(module, 'weight',
  leave_parametrized=True)` which computes the final weight (g * v/||v||) and stores
  it as a plain parameter. 328 keys → 246 keys after fold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.nn.utils.parametrize import remove_parametrizations

_COSYVOICE_SRC = str(Path(__file__).resolve().parents[3] / "model_data" / "CosyVoice_src")
_MATCHA = str(Path(_COSYVOICE_SRC) / "third_party" / "Matcha-TTS")
if _COSYVOICE_SRC not in sys.path:
    sys.path.insert(0, _COSYVOICE_SRC)
if _MATCHA not in sys.path:
    sys.path.append(_MATCHA)


def _fold_weight_norm(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            remove_parametrizations(module, "weight", leave_parametrized=True)


class HiFTVocoder:
    """Wraps the reference HiFTGenerator with hift.pt weights (weight-norm folded).

    Stage 1: host-side torch (correctness-first). Device port = Stage 2.
    """

    def __init__(self, hift_weights: Dict[str, torch.Tensor]):
        from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
        from cosyvoice.hifigan.generator import HiFTGenerator

        f0_pred = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
        self.model = HiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=24000,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            f0_predictor=f0_pred,
        )

        missing, unexpected = self.model.load_state_dict(hift_weights, strict=False)
        assert not unexpected, f"Unexpected keys: {unexpected[:5]}..."
        assert not missing, f"Missing keys: {missing[:5]}..."

        _fold_weight_norm(self.model)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, hift_pt_path: str | Path) -> "HiFTVocoder":
        sd = torch.load(str(hift_pt_path), map_location="cpu", weights_only=True)
        return cls(sd)

    @torch.inference_mode()
    def inference(
        self,
        mel: torch.Tensor,
        cache_source: torch.Tensor = torch.zeros(1, 1, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mel → waveform.

        Args:
            mel: [B, 80, T] mel-spectrogram (50 Hz frame rate).
            cache_source: [B, 1, S] optional cached source for glitch avoidance.

        Returns:
            (waveform [B, T*480], source [B, 1, T*480])
        """
        return self.model.inference(mel, cache_source=cache_source)
