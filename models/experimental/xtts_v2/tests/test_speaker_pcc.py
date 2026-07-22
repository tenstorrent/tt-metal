# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 2 (ResNet speaker encoder) reference validation.

Checks our self-contained reference (reference/xtts_speaker_ref.py) reproduces coqui's real
ResNetSpeakerEncoder CORE on the same logmel input: logmel [1,64,T] -> d-vector [1,512].
The mel front-end (torchaudio) runs on host and is not part of the port boundary.

Skips when the checkpoint or the vendored coqui source is absent (see reference/PROVENANCE.md).
Run: python -m pytest models/experimental/xtts_v2/tests/test_speaker_pcc.py
"""

import pytest
import torch

from models.experimental.xtts_v2.reference import xtts_speaker_ref as ref
from models.experimental.xtts_v2.reference.xtts_gpt_ref import pcc
from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

PCC_THRESHOLD = 0.999


def test_speaker_reference_matches_coqui():
    ckpt = gt.checkpoint_path()

    # Our reference core: logmel -> d-vector
    core = ref.build_reference(ckpt)
    logmel = ref.make_synthetic_logmel(n_frames=128)
    with torch.no_grad():
        our_dvec = core(logmel, l2_norm=True)  # [1, 512]

    # Coqui ground truth (needs the vendored source)
    if not gt.have_vendored_coqui():
        pytest.skip(f"vendored coqui source not found at {gt.VENDORED_TTS} (see reference/PROVENANCE.md)")
    enc = gt.build_coqui_speaker(gt.load_speaker_weights(ckpt))
    coqui_dvec = gt.coqui_speaker(enc, logmel)

    assert our_dvec.shape == coqui_dvec.shape
    dvec_pcc = pcc(our_dvec, coqui_dvec)
    assert dvec_pcc > PCC_THRESHOLD, f"speaker d-vector PCC {dvec_pcc:.6f} <= {PCC_THRESHOLD}"
