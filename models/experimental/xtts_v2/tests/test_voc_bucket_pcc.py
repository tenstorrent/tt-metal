# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the request path's vocoder length bucketing (tt/ttnn_xtts_model.py).

The vocoder is compiled and traced at a fixed set of frame counts (VOC_BUCKETS); each request
pads z up to its bucket and trims the waveform back. Three properties: the bucket a length maps
to, that the padding is inert (the trimmed waveform must not depend on how much zero padding
followed it, which is what lets a short utterance skip the 2634-frame cap), that replaying one
bucket's trace leaves the other buckets' output unchanged, and that the prepared conv weights the
buckets share are deduplicated rather than held per length.
"""
import types

import pytest
import torch
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
NUM_CONVS = 78  # conv_pre + conv_post + 4 upsamples + 12 resblocks x 6
MAX_LAYOUTS = 84  # measured: only 6 convs prepare a second layout, at one length boundary each
MAX_LAYOUT_MB = 100  # 70.4 MB measured; without deduplication it is 276.8 MB


def run_voc_bucket_invariance(device):
    refs = hifigan_reference()
    z, g = refs["z"], refs["g"]  # [1,1024,L], [1,512,1]
    L = z.shape[-1]
    assert _voc_bucket(L) < VOC_L, f"reference z (L={L}) must sit below the cap or this compares nothing"

    # _vocode reads only these three, so drive it on a stand-in rather than building the whole
    # model (all four blocks + traced decoder) for a Block-4 property. No traces -> eager path.
    model = types.SimpleNamespace(
        mesh_device=device,
        vocoder=TTNNHifiganGenerator(device, preprocess_hifigan_parameters(device)),
        _voc_traces={},
    )
    bucketed = XttsV2._vocode(model, z, g)  # pads L -> _voc_bucket(L)
    at_cap = XttsV2._vocode(model, F.pad(z, (0, VOC_L - L)), g)[:, :, : L * HOP]  # pre-bucketing behaviour

    passed, msg = comp_pcc(at_cap, bucketed, pcc=TARGET_PCC)
    print(f"L={L} -> bucket {_voc_bucket(L)} vs cap {VOC_L}, wav {tuple(bucketed.shape)}  pcc: {msg}")
    return passed, msg


def run_voc_trace_replay(device):
    """Replay every bucket, then re-check the largest: a capture frees its intermediates, so a
    later shape's persistent buffers can land there and a replay would scribble over them."""
    g = torch.randn(1, 512, 1, generator=torch.Generator().manual_seed(0))
    model = types.SimpleNamespace(
        mesh_device=device,
        vocoder=TTNNHifiganGenerator(device, preprocess_hifigan_parameters(device)),
        _voc_traces={},
    )
    XttsV2._capture_vocoder(model, g)
    assert sorted(model._voc_traces) == sorted(VOC_BUCKETS), "every bucket should be traced"

    gen = torch.Generator().manual_seed(1)
    z = {Lb: torch.randn(1, 1024, Lb, generator=gen) for Lb in VOC_BUCKETS}
    first = XttsV2._vocode(model, z[VOC_L], g)
    for Lb in VOC_BUCKETS[:-1]:
        XttsV2._vocode(model, z[Lb], g)
    again = XttsV2._vocode(model, z[VOC_L], g)

    maxabs = (first - again).abs().max().item()
    print(f"largest bucket re-replayed after {len(VOC_BUCKETS) - 1} others, maxabs diff {maxabs:.3e}")
    return maxabs == 0.0, f"maxabs {maxabs:.3e}"


def run_voc_prepared_weight_dedup(device):
    """Every bucket eagerly, then check the prepared weights collapsed to distinct layouts."""
    g = torch.randn(1, 512, 1, generator=torch.Generator().manual_seed(0))
    voc = TTNNHifiganGenerator(device, preprocess_hifigan_parameters(device))
    model = types.SimpleNamespace(mesh_device=device, vocoder=voc, _voc_traces={})
    for Lb in VOC_BUCKETS:
        XttsV2._vocode(model, torch.zeros(1, 1024, Lb), g)

    pairs, distinct, mb = voc.prepared_weight_stats()
    print(f"prepared weights: {pairs} (conv, length) pairs -> {distinct} layouts, {mb:.1f} MB")
    expected = NUM_CONVS * len(VOC_BUCKETS)
    passed = pairs == expected and distinct <= MAX_LAYOUTS and mb < MAX_LAYOUT_MB
    return passed, f"{pairs} pairs (expected {expected}), {distinct} layouts, {mb:.1f} MB"


def run_voc_bucket_selection():
    assert VOC_BUCKETS[-1] == VOC_L, "the top bucket must be the model cap, or long utterances have none"
    assert list(VOC_BUCKETS) == sorted(VOC_BUCKETS), "_voc_bucket scans in order, so buckets must ascend"
    lo, nxt = VOC_BUCKETS[0], VOC_BUCKETS[1]
    for L, want in ((1, lo), (lo - 1, lo), (lo, lo), (lo + 1, nxt), (VOC_L, VOC_L)):
        assert _voc_bucket(L) == want, f"_voc_bucket({L}) should be {want}"


def test_voc_bucket_selection(expect_error):
    run_voc_bucket_selection()
    with expect_error(AssertionError, "exceeds the fixed cap"):
        _voc_bucket(VOC_L + 1)


# l1_small_size sizes the per-bank L1 region conv shapes take their halo config from. Match
# XttsV2's own 262144: this test compiles a second vocoder shape and OOMs at the 65536
# test_hifigan_pcc opens with.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 262144}], indirect=True)
def test_voc_bucket_invariance(device):
    passed, msg = run_voc_bucket_invariance(device)
    assert passed, f"bucketed waveform differs from the cap-padded one below {TARGET_PCC}: {msg}"


# Traces need their own region on top of the L1_SMALL the conv shapes want.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 262144, "trace_region_size": 120_000_000}], indirect=True)
def test_voc_trace_replay(device):
    passed, msg = run_voc_trace_replay(device)
    assert passed, f"replaying other buckets changed the largest bucket's output: {msg}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 262144}], indirect=True)
def test_voc_prepared_weight_dedup(device):
    passed, msg = run_voc_prepared_weight_dedup(device)
    assert passed, f"prepared conv weights are not deduplicated as expected: {msg}"


if __name__ == "__main__":
    import sys

    run_voc_bucket_selection()
    dev = ttnn.open_device(device_id=0, l1_small_size=262144, trace_region_size=120_000_000)
    try:
        dev.enable_program_cache()
        ok, msg = run_voc_bucket_invariance(dev)
        ok2, msg2 = run_voc_trace_replay(dev)
        ok3, msg3 = run_voc_prepared_weight_dedup(dev)
    finally:
        ttnn.close_device(dev)
    all_ok = ok and ok2 and ok3
    print(("PASSED " if all_ok else "FAILED ") + f"{msg}; {msg2}; {msg3}")
    sys.exit(0 if all_ok else 1)
