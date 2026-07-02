# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SongEval aesthetic scorer wrapper.

SongEval (ASLP-lab/SongEval, arXiv:2505.10793) is the reference aesthetic-evaluation toolkit
used in the ACE-Step v1.5 paper. It scores full-length songs across five perceptual dimensions
aligned with professional-musician judgments:

    Coherence · Musicality · Memorability · Clarity · Naturalness   (each in [1, 5])

Pipeline (from the upstream `eval.py`):
    24 kHz waveform -> MuQ SSL encoder (hidden_states[6], 1024-d) -> Generator -> 5 scores

This wrapper packages that pipeline as a reusable `SongEvalScorer` so the ACE-Step demo can score
audio the same way the paper does. Upstream assets (`model.py`, `config.yaml`, `ckpt/model.safetensors`)
live alongside this file; the MuQ encoder is fetched from the HF hub on first use.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML = os.path.join(HERE, "config.yaml")
SCORER_CKPT = os.path.join(HERE, "ckpt", "model.safetensors")
MUQ_ID = "OpenMuQ/MuQ-large-msd-iter"
MUQ_HIDDEN_LAYER = 6  # SongEval uses the 7th hidden state (index 6)
MUQ_SAMPLE_RATE = 24000
DIMENSIONS = ("Coherence", "Musicality", "Memorability", "Clarity", "Naturalness")


def songeval_available() -> bool:
    """True iff deps + upstream assets are present (scorer ckpt is git-LFS, must be pulled)."""
    if not (os.path.isfile(CONFIG_YAML) and os.path.isfile(SCORER_CKPT)):
        return False
    if os.path.getsize(SCORER_CKPT) < 1024:  # LFS pointer / empty placeholder, not the real weights
        return False
    for mod in ("muq", "omegaconf", "hydra", "librosa"):
        if importlib.util.find_spec(mod) is None:
            return False
    return True


def _load_generator(device):
    """Instantiate the upstream Generator from config.yaml + load the scorer checkpoint."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    # The upstream config.yaml uses `_target_: model.Generator`; make that import resolvable.
    import sys

    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    cfg = OmegaConf.load(CONFIG_YAML)
    gen = instantiate(cfg.generator).to(device).eval()
    gen.load_state_dict(load_file(SCORER_CKPT, device="cpu"), strict=False)
    return gen


@dataclass
class SongEvalScorer:
    """Loaded SongEval scorer: MuQ SSL encoder + aesthetic Generator head."""

    generator: torch.nn.Module
    muq: torch.nn.Module
    device: torch.device

    @classmethod
    def load(cls, use_cpu: bool = True) -> "SongEvalScorer":
        from muq import MuQ

        device = torch.device("cpu") if use_cpu or not torch.cuda.is_available() else torch.device("cuda")
        gen = _load_generator(device)
        muq = MuQ.from_pretrained(MUQ_ID).to(device).eval()
        return cls(generator=gen, muq=muq, device=device)

    @torch.no_grad()
    def score_waveform(self, waveform: torch.Tensor) -> dict[str, float]:
        """Score a mono 24 kHz waveform tensor [T] or [1, T]. Returns the 5 aesthetic scores."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        feat = self.muq(waveform, output_hidden_states=True)["hidden_states"][MUQ_HIDDEN_LAYER]
        scores = self.generator(feat).squeeze(0)
        return {dim: round(scores[i].item(), 4) for i, dim in enumerate(DIMENSIONS)}

    @torch.no_grad()
    def score_file(self, audio_path: str) -> dict[str, float]:
        """Load an audio file at 24 kHz and score it."""
        import librosa

        wav, _ = librosa.load(audio_path, sr=MUQ_SAMPLE_RATE)
        return self.score_waveform(torch.tensor(wav))
