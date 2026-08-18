# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the request path's vocoder length bucketing (tt/ttnn_xtts_model.py).

The vocoder is compiled at a fixed set of frame counts (VOC_BUCKETS); each request pads z up
to its bucket and trims the waveform back. Two properties: the bucket a length maps to, and
that the padding is inert — the trimmed waveform must not depend on how much zero padding
followed it, which is what lets a short utterance skip the 2634-frame cap.
"""
import types

import pytest
import torch.nn.functional as F
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.tests.reference_helpers import hifigan_reference
from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import (
    TTNNHifiganGenerator,
    preprocess_hifigan_parameters,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_model import HOP, VOC_BUCKETS, VOC_L, XttsV2, _voc_bucket

TARGET_PCC = 0.999


def run_voc_bucket_invariance(device):
    refs = hifigan_reference()
    z, g = refs["z"], refs["g"]  # [1,1024,L], [1,512,1]
    L = z.shape[-1]
    assert _voc_bucket(L) < VOC_L, f"reference z (L={L}) must sit below the cap or this compares nothing"

    # _vocode reads only self.mesh_device and self.vocoder, so drive it on a stand-in rather
    # than building the whole model (all four blocks + traced decoder) for a Block-4 property.
    model = types.SimpleNamespace(
        mesh_device=device, vocoder=TTNNHifiganGenerator(device, preprocess_hifigan_parameters(device))
    )
    bucketed = XttsV2._vocode(model, z, g)  # pads L -> _voc_bucket(L)
    at_cap = XttsV2._vocode(model, F.pad(z, (0, VOC_L - L)), g)[:, :, : L * HOP]  # pre-bucketing behaviour

    passed, msg = comp_pcc(at_cap, bucketed, pcc=TARGET_PCC)
    print(f"L={L} -> bucket {_voc_bucket(L)} vs cap {VOC_L}, wav {tuple(bucketed.shape)}  pcc: {msg}")
    return passed, msg


def test_voc_bucket_selection():
    assert VOC_BUCKETS[-1] == VOC_L, "the top bucket must be the model cap, or long utterances have none"
    assert list(VOC_BUCKETS) == sorted(VOC_BUCKETS), "_voc_bucket scans in order, so buckets must ascend"
    lo, nxt = VOC_BUCKETS[0], VOC_BUCKETS[1]
    for L, want in ((1, lo), (lo - 1, lo), (lo, lo), (lo + 1, nxt), (VOC_L, VOC_L)):
        assert _voc_bucket(L) == want, f"_voc_bucket({L}) should be {want}"
    with pytest.raises(AssertionError):
        _voc_bucket(VOC_L + 1)


# l1_small_size sizes the per-bank L1 region conv shapes take their halo config from. Match
# XttsV2's own 262144: this test compiles a second vocoder shape and OOMs at the 65536
# test_hifigan_pcc opens with.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 262144}], indirect=True)
def test_voc_bucket_invariance(device):
    passed, msg = run_voc_bucket_invariance(device)
    assert passed, f"bucketed waveform differs from the cap-padded one below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys

    test_voc_bucket_selection()
    dev = ttnn.open_device(device_id=0, l1_small_size=262144)
    try:
        dev.enable_program_cache()
        ok, msg = run_voc_bucket_invariance(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
