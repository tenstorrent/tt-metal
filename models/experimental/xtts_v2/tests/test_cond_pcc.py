# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 1 (conditioning encoder + Perceiver resampler) reference validation.

Checks our self-contained reference (reference/xtts_cond_ref.py) reproduces coqui's real
conditioning branch on the same mel input, at two boundaries:
  - conditioning-encoder output  (mel -> [1,1024,T])
  - full style embedding         (get_style_emb -> [1,1024,32], = gpt_cond_latent transposed)

Skips when the checkpoint or the vendored coqui source is absent (see reference/PROVENANCE.md).
Run: python -m pytest models/experimental/xtts_v2/tests/test_cond_pcc.py
"""

import pytest

from models.experimental.xtts_v2.reference import xtts_cond_ref as ref
from models.experimental.xtts_v2.reference.xtts_gpt_ref import pcc
from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

PCC_THRESHOLD = 0.999


def test_cond_reference_matches_coqui():
    ckpt = gt.checkpoint_path()

    # Our reference
    enc_w, perc_w = ref.load_cond_state(ckpt)
    mel = ref.make_synthetic_mel(n_frames=128)
    our_enc, our_perc = ref.get_style_emb(mel, enc_w, perc_w)
    our_style = our_perc.transpose(1, 2)  # [1, 1024, 32] to match coqui get_style_emb

    # Coqui ground truth (needs the vendored source)
    if not gt.have_vendored_coqui():
        pytest.skip(f"vendored coqui source not found at {gt.VENDORED_TTS} (see reference/PROVENANCE.md)")
    coqui_gpt = gt.build_coqui_gpt(gt.load_gpt_weights(ckpt))
    coqui_enc, coqui_style = gt.coqui_cond(coqui_gpt, mel)

    enc_pcc = pcc(our_enc, coqui_enc)
    style_pcc = pcc(our_style, coqui_style)
    assert enc_pcc > PCC_THRESHOLD, f"conditioning-encoder PCC {enc_pcc:.6f} <= {PCC_THRESHOLD}"
    assert style_pcc > PCC_THRESHOLD, f"style-embedding PCC {style_pcc:.6f} <= {PCC_THRESHOLD}"
