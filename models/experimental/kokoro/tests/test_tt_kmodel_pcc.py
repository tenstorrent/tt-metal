# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_kmodel.TTKModel` vs reference
:class:`~models.experimental.kokoro.reference.model.KModel`.

Runs the full on-device pipeline with real Kokoro-82M checkpoint weights and
actual phoneme inputs produced by the upstream G2P pipeline.

Test structure
--------------
``test_tt_kmodel_generator_no_torch_fallback_pcc``
Full pipeline with all torch fallbacks disabled.  Validates shape, finiteness,
non-trivial signal, and a minimum PCC floor against reference ``KModel.forward``.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))


from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tt.tt_kmodel import TTKModel, preprocess_tt_kmodel

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_TEST_TEXT = "Hello from Tenstorrent."
_VOICE = "af_heart"
_LANG_CODE = "a"

_CKPT_CANDIDATES = (
    Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _find_checkpoint() -> Path | None:
    for p in _CKPT_CANDIDATES:
        if p.is_file():
            return p
        if p.is_dir():
            for child in p.rglob("kokoro-v1_0.pth"):
                return child
    return None


def _load_pipeline():
    """Return a quiet KPipeline (G2P only, no model)."""
    try:
        from kokoro import KPipeline  # upstream package: pip install "kokoro>=0.9.2"

        return KPipeline(lang_code=_LANG_CODE, model=False)
    except ImportError:
        return None


def _phonemize(text: str) -> tuple[str, torch.Tensor]:
    """Return ``(phonemes, ref_s)`` for the first chunk of ``text``.

    Loads the voice pack from the ``kokoro`` package (which must be installed).
    Skips the test if the package or the voice cannot be loaded.
    """
    pipe = _load_pipeline()
    if pipe is None:
        pytest.skip("kokoro package not installed: pip install 'kokoro>=0.9.2'")

    results = list(pipe(text, voice=_VOICE))
    if not results:
        pytest.skip(f"KPipeline produced no chunks for: {text!r}")

    phonemes = results[0].phonemes
    if not phonemes:
        pytest.skip("KPipeline produced empty phonemes for first chunk.")

    pack = pipe.load_voice(_VOICE)
    ref_s = pack[len(phonemes) - 1]
    if not isinstance(ref_s, torch.Tensor):
        ref_s = torch.tensor(ref_s)
    ref_s = ref_s.float().cpu()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)

    return phonemes, ref_s


@contextmanager
def _zero_noise():
    """Context that zeros all stochastic torch ops for deterministic reference output."""
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    torch.randn_like = lambda t, **kwargs: torch.zeros_like(t, **kwargs)
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _ref_audio(ref: KModel, phonemes: str, ref_s: torch.Tensor, speed: float = 1.0) -> torch.Tensor:
    """Run the reference KModel forward and return the audio (1-D float32)."""
    with torch.no_grad(), _zero_noise():
        out = ref.forward(phonemes=phonemes, ref_s=ref_s, speed=speed, return_output=False)
    return out.detach().float().squeeze()


def _tt_audio(
    device,
    ref: KModel,
    params,
    phonemes: str,
    ref_s: torch.Tensor,
    *,
    use_torch_stft_fallback: bool,
    use_torch_phase_fallback: bool,
    use_torch_linear_fallback: bool = False,
    use_torch_tanh_fallback: bool = False,
) -> torch.Tensor:
    """Run one TTKModel variant and return 1-D audio."""
    tt_model = TTKModel(
        device,
        ref,
        params,
        use_torch_stft_fallback=use_torch_stft_fallback,
        use_torch_phase_fallback=use_torch_phase_fallback,
        use_torch_linear_fallback=use_torch_linear_fallback,
        use_torch_tanh_fallback=use_torch_tanh_fallback,
    )
    out = tt_model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
    return out.audio.detach().float().squeeze()


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup(ckpt_path: Path, device) -> tuple[KModel, TTKModel, str, torch.Tensor]:
    """Load KModel, build TTKModel, return (ref, tt_model, phonemes, ref_s)."""
    phonemes, ref_s = _phonemize(_TEST_TEXT)

    ref = KModel(
        repo_id="hexgrad/Kokoro-82M",
        model=str(ckpt_path),
        disable_complex=True,
    ).eval()

    params = preprocess_tt_kmodel(ref, device)
    # Return the params; callers pass fallback flags when constructing TTKModel.
    return ref, params, phonemes, ref_s


# ---------------------------------------------------------------------------
# No torch fallback path
# ---------------------------------------------------------------------------


def test_tt_kmodel_generator_no_torch_fallback_pcc(device):
    """TTKModel without torch fallbacks — shape, finiteness, signal, and PCC floor."""
    ckpt_path = _find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    ref, params, phonemes, ref_s = _setup(ckpt_path, device)
    y_ref = _ref_audio(ref, phonemes, ref_s)

    y_hat = _tt_audio(
        device,
        ref,
        params,
        phonemes,
        ref_s,
        use_torch_stft_fallback=False,
        use_torch_phase_fallback=False,
    )

    assert y_hat.shape == y_ref.shape, (y_hat.shape, y_ref.shape)
    assert torch.isfinite(y_hat).all(), "TTKModel (no torch fallback) produced NaN/Inf"
    assert y_hat.abs().max().item() > 1e-3, "TTKModel (no torch fallback) produced ~zero output"

    _, pcc = comp_pcc(y_ref.unsqueeze(0), y_hat.unsqueeze(0), pcc=0.0)
    print(f"\nTTKModel no-torch-fallback PCC: {pcc:.6f}  phonemes={len(phonemes)}")
    assert pcc > 0.25, f"PCC {pcc:.6f} is below the no-fallback minimum floor"

    debug_mode = os.getenv("KOKORO_PCC_DEBUG_MODE", "").strip().lower()
    if debug_mode in {"stft", "phase", "both", "full"}:
        use_stft = debug_mode in {"stft", "both", "full"}
        use_phase = debug_mode in {"phase", "both", "full"}
        use_linear = debug_mode == "full"
        use_tanh = debug_mode == "full"
        y_debug = _tt_audio(
            device,
            ref,
            params,
            phonemes,
            ref_s,
            use_torch_stft_fallback=use_stft,
            use_torch_phase_fallback=use_phase,
            use_torch_linear_fallback=use_linear,
            use_torch_tanh_fallback=use_tanh,
        )
        _, pcc_debug = comp_pcc(y_ref.unsqueeze(0), y_debug.unsqueeze(0), pcc=0.0)
        print("TTKModel PCC debug: " f"mode={debug_mode}, no_fallback={pcc:.6f}, debug_mode_pcc={pcc_debug:.6f}")
