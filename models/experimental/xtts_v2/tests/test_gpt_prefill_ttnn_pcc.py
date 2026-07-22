# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 3 (GPT core) PREFILL — TTNN-vs-reference PCC on device.

Runs the TTNN port (tt/ttnn_xtts_gpt.py) on the Tenstorrent device and compares its latents to
the CPU reference (reference/xtts_gpt_ref.py) on the same synthetic inputs_embeds. The port runs
fp32 (HiFi3 + fp32 accumulation) to match the reference precision, but Tensix matmuls are not
bit-exact IEEE fp32, so the bar is 0.999 (measured ~0.9997 over the full 30-layer stack).

Skips cleanly when: ttnn isn't importable, no device is available, or the checkpoint is absent.
Run: TT_METAL_HOME=<repo> PYTHONPATH=<repo> python -m pytest \
        models/experimental/xtts_v2/tests/test_gpt_prefill_ttnn_pcc.py
"""

import os

import pytest

from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

# ttnn needs TT_METAL_HOME to locate kernels; default it to the repo root if unset.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(gt.XTTS_DIR)))
os.environ.setdefault("TT_METAL_HOME", _REPO)

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable (build tt-metal from source)")

from models.experimental.xtts_v2.reference import xtts_gpt_ref as ref  # noqa: E402
from models.experimental.xtts_v2.tt import ttnn_xtts_gpt as port  # noqa: E402

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

PCC_THRESHOLD = 0.999


def test_gpt_prefill_ttnn_matches_reference():
    ckpt = gt.checkpoint_path()

    # CPU reference golden
    gpt2, final_norm = ref.build_reference(ckpt)
    inputs_embeds = ref.make_synthetic_inputs_embeds(ckpt, n_text=16, n_mel=48)
    _, golden = ref.reference_forward(gpt2, final_norm, inputs_embeds)

    # TTNN on device
    try:
        device = ttnn.open_device(device_id=0)
    except Exception as e:  # no device / driver issue
        pytest.skip(f"could not open a Tenstorrent device: {e}")
    try:
        out = port.run_prefill(device, inputs_embeds, ckpt_path=ckpt)
    finally:
        ttnn.close_device(device)

    p = ref.pcc(out, golden)
    assert p > PCC_THRESHOLD, f"TTNN prefill PCC {p:.6f} <= {PCC_THRESHOLD}"
