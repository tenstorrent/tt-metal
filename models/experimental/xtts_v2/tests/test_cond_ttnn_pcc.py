# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 1 (conditioning encoder + Perceiver resampler) — TTNN-vs-reference on device.

Runs the on-device port (tt/ttnn_xtts_cond.py: TtConditioningEncoder) and compares to the CPU
reference (reference/xtts_cond_ref.py: get_style_emb) on the same synthetic mel, at both boundaries:
  - conditioning-encoder output  (mel [1,80,T] -> enc [1,1024,T])
  - Perceiver output             (perc [1,32,1024], = gpt_cond_latent)

fp32 throughout (fidelity-first, one-shot block). Two subtleties matter for accuracy and are baked
into the port: the attention softmax must be numeric_stable, and GroupNorm is done manually in fp32
(native ttnn.group_norm is bf16-only) — otherwise the resampler amplifies the residual error.

Skips cleanly without ttnn / a device / the checkpoint.
Run: TT_METAL_HOME=<repo> PYTHONPATH=<repo> python -m pytest \
        models/experimental/xtts_v2/tests/test_cond_ttnn_pcc.py
"""

import os

import pytest

from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(gt.XTTS_DIR)))
os.environ.setdefault("TT_METAL_HOME", _REPO)

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable (build tt-metal from source)")

from models.experimental.xtts_v2.reference import xtts_cond_ref as ref  # noqa: E402
from models.experimental.xtts_v2.reference.xtts_gpt_ref import pcc  # noqa: E402
from models.experimental.xtts_v2.tt import ttnn_xtts_cond as port  # noqa: E402

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

PCC_THRESHOLD = 0.999  # fp32 on device vs fp32 CPU reference


def test_cond_ttnn_matches_reference():
    ckpt = gt.checkpoint_path()

    enc_w, perc_w = ref.load_cond_state(ckpt)
    mel = ref.make_synthetic_mel(n_frames=128)
    ref_enc, ref_perc = ref.get_style_emb(mel, enc_w, perc_w)

    try:
        device = ttnn.open_device(device_id=0, l1_small_size=131072)
    except Exception as e:
        pytest.skip(f"could not open a Tenstorrent device: {e}")
    try:
        gen = port.TtConditioningEncoder(device, ckpt)
        our_enc, our_perc = gen(mel)
    finally:
        ttnn.close_device(device)

    enc_pcc = pcc(our_enc, ref_enc)
    perc_pcc = pcc(our_perc, ref_perc)
    assert enc_pcc > PCC_THRESHOLD, f"conditioning-encoder PCC {enc_pcc:.6f} <= {PCC_THRESHOLD}"
    assert perc_pcc > PCC_THRESHOLD, f"Perceiver (gpt_cond_latent) PCC {perc_pcc:.6f} <= {PCC_THRESHOLD}"
