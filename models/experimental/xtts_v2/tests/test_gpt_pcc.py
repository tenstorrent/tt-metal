# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 3 (GPT core) PREFILL reference validation.

Checks our self-contained reference (reference/xtts_gpt_ref.py) reproduces coqui's real GPT
forward on the same inputs_embeds: inputs_embeds -> GPT2 stack -> ln_f -> final_norm = latents.

Skips when the checkpoint or the vendored coqui source is absent (see reference/PROVENANCE.md).
Run: python -m pytest models/experimental/xtts_v2/tests/test_gpt_pcc.py
"""

import pytest

from models.experimental.xtts_v2.reference import xtts_gpt_ref as ref
from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

PCC_THRESHOLD = 0.999


def test_gpt_prefill_reference_matches_coqui():
    ckpt = gt.checkpoint_path()

    # Our reference
    gpt, final_norm = ref.build_reference(ckpt)
    inputs_embeds = ref.make_synthetic_inputs_embeds(ckpt, n_text=16, n_mel=48)
    _, our_latents = ref.reference_forward(gpt, final_norm, inputs_embeds)

    # Coqui ground truth (needs the vendored source)
    if not gt.have_vendored_coqui():
        pytest.skip(f"vendored coqui source not found at {gt.VENDORED_TTS} (see reference/PROVENANCE.md)")
    coqui_gpt = gt.build_coqui_gpt(gt.load_gpt_weights(ckpt))
    coqui_latents = gt.coqui_prefill_latents(coqui_gpt, inputs_embeds)

    assert our_latents.shape == coqui_latents.shape
    pcc = ref.pcc(our_latents, coqui_latents)
    assert pcc > PCC_THRESHOLD, f"prefill latents PCC {pcc:.6f} <= {PCC_THRESHOLD}"
