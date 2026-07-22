# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 4 (HiFi-GAN vocoder generator) reference validation.

Checks our self-contained reference (reference/xtts_hifigan_ref.py, weight-norm folded)
reproduces coqui's real HifiganGenerator on the same input:
    z [1,1024,L] + d-vector g [1,512,1] -> waveform [1,1,L*256].

Skips when the checkpoint or the vendored coqui source is absent (see reference/PROVENANCE.md).
Run: python -m pytest models/experimental/xtts_v2/tests/test_hifigan_pcc.py
"""

import pytest

from models.experimental.xtts_v2.reference import xtts_hifigan_ref as ref
from models.experimental.xtts_v2.reference.xtts_gpt_ref import pcc
from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

PCC_THRESHOLD = 0.999


def test_hifigan_reference_matches_coqui():
    ckpt = gt.checkpoint_path()

    # Our reference (weight-norm folded, functional)
    w = ref.load_hifigan_state(ckpt)
    z, g = ref.make_synthetic_inputs(n_latent=32)
    our_wav = ref.generator(z, w, g)  # [1, 1, L*256]

    # Coqui ground truth (needs the vendored source)
    if not gt.have_vendored_coqui():
        pytest.skip(f"vendored coqui source not found at {gt.VENDORED_TTS} (see reference/PROVENANCE.md)")
    gen = gt.build_coqui_hifigan(gt.load_hifigan_weights(ckpt))
    coqui_wav = gt.coqui_hifigan(gen, z, g)

    assert our_wav.shape == coqui_wav.shape
    wav_pcc = pcc(our_wav, coqui_wav)
    assert wav_pcc > PCC_THRESHOLD, f"vocoder waveform PCC {wav_pcc:.6f} <= {PCC_THRESHOLD}"
