# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block 3 (GPT core) TRACED decode — TTNN-vs-reference on device.

Runs the fast trace-captured bf16 decoder (tt/ttnn_xtts_gpt.py: generate_traced) — one 30-layer
step captured into a device trace and replayed per token, with the position in a device tensor
threaded through paged_update_cache + scaled_dot_product_attention_decode. Compares to the CPU
reference (reference_generate) from the same prefix: the generated code sequence must match
exactly (greedy), and latents PCC must clear 0.99 (bf16 flash-decode vs the fp32 CPU reference).

Requires the device opened with a trace_region_size. Skips cleanly without ttnn / a device /
the checkpoint.
Run: TT_METAL_HOME=<repo> PYTHONPATH=<repo> python -m pytest \
        models/experimental/xtts_v2/tests/test_gpt_trace_ttnn_pcc.py
"""

import os

import pytest
import torch

from models.experimental.xtts_v2.tests import _coqui_groundtruth as gt

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(gt.XTTS_DIR)))
os.environ.setdefault("TT_METAL_HOME", _REPO)

ttnn = pytest.importorskip("ttnn", reason="ttnn not importable (build tt-metal from source)")

from models.experimental.xtts_v2.reference import xtts_gpt_ref as ref  # noqa: E402
from models.experimental.xtts_v2.tt import ttnn_xtts_gpt as port  # noqa: E402

pytestmark = pytest.mark.skipif(
    not gt.have_checkpoint(),
    reason=f"XTTS-v2 checkpoint not found at {gt.checkpoint_path()} (see reference/PROVENANCE.md)",
)

MAX_NEW = 24
PCC_THRESHOLD = 0.99  # bf16 flash-decode vs fp32 reference


def test_gpt_traced_decode_matches_reference():
    ckpt = gt.checkpoint_path()

    # CPU reference decode from a fixed synthetic prefix
    heads = ref.load_gen_head(ckpt)
    prefix = ref.make_synthetic_prefix(heads, n_text=8)
    gpt2, final_norm = ref.build_reference(ckpt)
    ref_out = ref.reference_generate(gpt2, final_norm, heads, prefix, max_new=MAX_NEW)

    # Traced bf16 decode on device (needs a trace region)
    try:
        device = ttnn.open_device(device_id=0, trace_region_size=50_000_000)
    except Exception as e:
        pytest.skip(f"could not open a Tenstorrent device with a trace region: {e}")
    try:
        our = port.generate_traced(device, prefix, heads, max_new=MAX_NEW, use_trace=True)
    finally:
        ttnn.close_device(device)

    k = ref_out["codes"].numel()
    assert torch.equal(our["codes"][:k], ref_out["codes"][:k]), (
        f"code sequences differ:\n  ours = {our['codes'][:k].tolist()}\n  ref  = {ref_out['codes'][:k].tolist()}"
    )
    latents_pcc = ref.pcc(our["latents"][:, :k], ref_out["latents"][:, :k])
    assert latents_pcc > PCC_THRESHOLD, f"traced decode latents PCC {latents_pcc:.6f} <= {PCC_THRESHOLD}"
