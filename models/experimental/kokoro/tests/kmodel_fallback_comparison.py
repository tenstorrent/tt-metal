# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC sweep: one new op fallback at a time on top of baseline E.

Baseline E = ``use_torch_stft_fallback`` + ``use_torch_phase_fallback`` +
``use_torch_sinegen_fallback`` (vocoder harmonic path stabilized).

Each row toggles exactly one additional op flag. Per-stage breakdown:
``kmodel_decode_stack_diagnostic.py``.

Run (from repo root, with kokoro package + checkpoint + TT device):

    python models/experimental/kokoro/tests/kmodel_fallback_comparison.py
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tt.tt_kmodel import TTKModel, preprocess_tt_kmodel

_TEST_TEXT = "Hello from Tenstorrent."
_VOICE = "af_heart"
_LANG_CODE = "a"

_CKPT_CANDIDATES = (
    Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
)


def _find_checkpoint() -> Optional[Path]:
    for p in _CKPT_CANDIDATES:
        if p.is_file():
            return p
        if p.is_dir():
            for child in p.rglob("kokoro-v1_0.pth"):
                return child
    return None


@contextmanager
def _zero_noise():
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    torch.randn_like = lambda t, **kwargs: torch.zeros_like(t, **kwargs)
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _phonemize(text: str) -> tuple[str, torch.Tensor]:
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=_LANG_CODE, model=False)
    results = list(pipe(text, voice=_VOICE))
    phonemes = results[0].phonemes
    pack = pipe.load_voice(_VOICE)
    ref_s = pack[len(phonemes) - 1]
    if not isinstance(ref_s, torch.Tensor):
        ref_s = torch.tensor(ref_s)
    ref_s = ref_s.float().cpu()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    return phonemes, ref_s


@dataclass
class Config:
    name: str
    stft: bool = False
    stft_conv: bool = False
    atan2: bool = False
    phase: bool = False
    sinegen: bool = False
    linear: bool = False
    tanh: bool = False


_E = dict(stft=True, phase=True, sinegen=True)

CONFIGS: list[Config] = [
    Config(name="A. No fallback"),
    Config(name="E. STFT+Phase+SineGen (baseline)", **_E),
    Config(name="E + linear+tanh", **_E, linear=True, tanh=True),
    Config(name="F. conv+atan2+Phase+SineGen", stft_conv=True, atan2=True, phase=True, sinegen=True),
]


def _run_tt(
    device,
    ref: KModel,
    params,
    phonemes: str,
    ref_s: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    with _zero_noise():
        tt_model = TTKModel(
            device,
            ref,
            params,
            use_torch_stft_fallback=cfg.stft,
            use_torch_stft_conv_fallback=cfg.stft_conv,
            use_torch_atan2_fallback=cfg.atan2,
            use_torch_phase_fallback=cfg.phase,
            use_torch_sinegen_fallback=cfg.sinegen,
            use_torch_linear_fallback=cfg.linear,
            use_torch_tanh_fallback=cfg.tanh,
        )
    out = tt_model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
    return out.audio.detach().float().squeeze()


def main() -> None:
    ckpt = _find_checkpoint()
    if ckpt is None:
        sys.exit("Kokoro-82M checkpoint not found. Update _CKPT_CANDIDATES.")

    phonemes, ref_s = _phonemize(_TEST_TEXT)
    print(f"Text: {_TEST_TEXT!r}")
    print(f"Phonemes ({len(phonemes)}): {phonemes!r}")

    ref = KModel(repo_id="hexgrad/Kokoro-82M", model=str(ckpt), disable_complex=True).eval()
    with torch.no_grad(), _zero_noise():
        y_ref = ref.forward(phonemes=phonemes, ref_s=ref_s, speed=1.0, return_output=False).detach().float().squeeze()

    device = ttnn.open_device(device_id=0)
    try:
        params = preprocess_tt_kmodel(ref, device)
        baseline_pcc: Optional[float] = None
        for cfg in CONFIGS:
            try:
                y_hat = _run_tt(device, ref, params, phonemes, ref_s, cfg)
                _, pcc = comp_pcc(y_ref.unsqueeze(0), y_hat.unsqueeze(0), pcc=0.0)
                pcc_f = float(pcc)
                if cfg.name.startswith("E. STFT"):
                    baseline_pcc = pcc_f
                delta = ""
                if baseline_pcc is not None and not cfg.name.startswith("A.") and not cfg.name.startswith("E. STFT"):
                    delta = f"  (Δ vs E: {pcc_f - baseline_pcc:+.6f})"
                print(f"{cfg.name:<52} PCC = {pcc_f:.6f}{delta}")
            except Exception as e:  # noqa: BLE001
                print(f"{cfg.name:<52} ERROR: {e}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
