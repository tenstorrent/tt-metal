# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 3 (GPT core) DECODE reference validation.

Checks our greedy KV-cache decode reference reproduces coqui's real GPT2InferenceModel decode
from the identical prefix: exact generated code sequence + per-step logits PCC. Coqui's decode
is driven step-by-step (its get_fixed_embedding position logic) rather than via HF .generate().

Skips when the checkpoint or the vendored coqui source is absent (see reference/PROVENANCE.md).
Run: python -m pytest models/experimental/xtts_v2/tests/test_gpt_decode_pcc.py
"""

import pytest
import torch

from models.experimental.xtts_v2.reference import xtts_gpt_ref as ref
from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

MAX_NEW = 24
PCC_THRESHOLD = 0.999


def test_gpt_decode_reference_matches_coqui():
    ckpt = gt.checkpoint_path()

    # Our reference decode
    gpt, final_norm = ref.build_reference(ckpt)
    heads = ref.load_gen_head(ckpt)
    prefix = ref.make_synthetic_prefix(heads, n_text=8)
    our = ref.reference_generate(gpt, final_norm, heads, prefix, max_new=MAX_NEW)

    # Coqui ground truth (needs the vendored source)
    if not gt.have_vendored_coqui():
        pytest.skip(f"vendored coqui source not found at {gt.VENDORED_TTS} (see reference/PROVENANCE.md)")
    coqui_gpt = gt.build_coqui_gpt(gt.load_gpt_weights(ckpt))
    coqui_codes, coqui_logits = gt.coqui_decode(coqui_gpt, prefix, n_steps=our["codes"].numel())

    k = coqui_codes.numel()
    assert torch.equal(our["codes"][:k], coqui_codes), (
        f"code sequences differ:\n  ours = {our['codes'][:k].tolist()}\n  coqui= {coqui_codes.tolist()}"
    )
    pcc = ref.pcc(our["logits"][:, :k], coqui_logits)
    assert pcc > PCC_THRESHOLD, f"per-step logits PCC {pcc:.6f} <= {PCC_THRESHOLD}"
