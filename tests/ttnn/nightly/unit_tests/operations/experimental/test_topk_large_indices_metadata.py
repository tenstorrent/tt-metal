# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-safe metadata path of topk_large_indices (valid_length_tensor / valid_length_offset).

`valid_length` is a host runtime argument. A ttnn trace replay cannot re-run the host code that sets it, so
a captured value stays frozen and every later chunk ranks the wrong prefix -- too small drops real keys, too
large ranks a stale tail. The tensor form has the reader read the bound on-device as
`valid_length_tensor[0] + valid_length_offset`.

The offset is not decoration: the DSA indexer's bound is `chunk_start + chunk_global`, where only
`chunk_start` varies per chunk. Carrying the base in the tensor and the structural part as the offset lets
ONE metadata tensor drive both this op and `ring_indexer_score_dsa`'s `kv_len`, so the two bounds cannot
drift apart -- which is what makes a traced run bit-identical to an untraced one rather than merely close.

Note the failure mode this op has that most do not: the reader pushes `num_chunks * tiles_per_chunk` CB
pages and compute pops `num_chunks`. If the two derived `num_chunks` independently and disagreed, the op
would HANG rather than return wrong data. The kernels therefore derive once, in the reader, and publish
through a mailbox CB -- `test_metadata_trace_replay` exercises exactly the case that would expose a
mismatch (a replay whose derived num_chunks differs from the captured one).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.nightly.unit_tests.operations.experimental.test_topk_large_indices import (
    _make_bf16_exact_input,
    _to_device,
    _assert_indices,
)

# (k, n, valid_length) -- chosen so the awkward cases are covered, not just the aligned one:
#   1024/4096/1024  aligned: valid_length is exactly one LLK window
#   256/2048/600    valid_length is NOT a multiple of the LLK window
#   512/4096/513    odd valid_length -> a 1-element tail chunk
CASES = [(1024, 4096, 1024), (256, 2048, 600), (512, 4096, 513)]
CASE_IDS = [f"k{k}_n{n}_v{v}" for k, n, v in CASES]


def _meta(device, value: int, *, on_device: bool = True):
    t = torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1)
    kw = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    if on_device:
        kw.update(device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.from_torch(t, **kw)


@pytest.mark.parametrize("k,n,valid_length", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("offset", [0, 128], ids=["off0", "off128"])
def test_metadata_matches_scalar(device, k, n, valid_length, offset):
    """The tensor path must reproduce the scalar path exactly, including when the bound is split between the
    tensor and the offset (base + offset == the same total)."""
    if valid_length - offset <= 0:
        pytest.skip("offset exceeds the bound under test")
    num_rows = 2
    torch_input = _make_bf16_exact_input(num_rows, n)

    scalar = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=valid_length)
    meta = ttnn.experimental.topk_large_indices(
        _to_device(torch_input, device),
        k=k,
        valid_length_tensor=_meta(device, valid_length - offset),
        valid_length_offset=offset,
    )
    assert_equal(
        ttnn.to_torch(meta, dtype=torch.uint32).to(torch.int64),
        ttnn.to_torch(scalar, dtype=torch.uint32).to(torch.int64),
    )

    # And still the core guarantee: rows are strictly increasing, so the largest values live in the stale
    # tail -- any index >= valid_length means the tail leaked.
    _, ref = torch.topk(torch_input[:, :valid_length].float(), k, dim=-1, largest=True, sorted=True)
    assert int(ref.max()) < valid_length
    _assert_indices(meta, ref, [num_rows, k])


def test_metadata_rejects_scalar_alongside_tensor(device, expect_error):
    """Mutually exclusive rather than silently preferring one -- the on-device value would win and the
    caller's scalar would be quietly ignored."""
    torch_input = _make_bf16_exact_input(2, 2048)
    with expect_error(RuntimeError, "mutually exclusive"):
        ttnn.experimental.topk_large_indices(
            _to_device(torch_input, device),
            k=256,
            valid_length=600,
            valid_length_tensor=_meta(device, 600),
        )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_metadata_trace_replay(device):
    """Capture ONCE, then replay with a DIFFERENT valid_length each time.

    This is the test that actually proves traceability. A replay that only re-runs the CAPTURED bound passes
    even if the kernel ignores the metadata tensor entirely, so each replay here uses its own value and is
    checked against that value's own eager result. The order includes a DESCENDING pass: a stale bound that
    is too LARGE reads the stale tail (wrong indices), while too small drops real keys -- ascending-only
    replay can miss one of those directions.
    """
    num_rows, n, k = 2, 4096, 1024
    bounds = [1024, 2048, 1536, 4096]
    torch_input = _make_bf16_exact_input(num_rows, n)

    # Eager references, one per bound.
    refs = {}
    for v in bounds:
        out = ttnn.experimental.topk_large_indices(_to_device(torch_input, device), k=k, valid_length=v)
        refs[v] = ttnn.to_torch(out, dtype=torch.uint32).to(torch.int64)

    tt_input = _to_device(torch_input, device)
    meta = _meta(device, bounds[0])
    host_meta = {v: _meta(device, v, on_device=False) for v in bounds}

    # Warm the metadata program before capture: a program-cache miss inside begin_trace_capture would
    # compile into the capture instead of being replayed from it.
    ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length_tensor=meta)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    traced_out = ttnn.experimental.topk_large_indices(tt_input, k=k, valid_length_tensor=meta)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    order = bounds + bounds[::-1] + [bounds[-1], bounds[0]]
    for i, v in enumerate(order):
        ttnn.copy_host_to_device_tensor(host_meta[v], meta)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(traced_out, dtype=torch.uint32).to(torch.int64)
        if not torch.equal(got, refs[v]):
            # Matching a DIFFERENT bound's reference is the signature of a stale/ignored metadata read;
            # say so rather than just reporting inequality.
            matches = [b for b in bounds if torch.equal(got, refs[b])]
            raise AssertionError(
                f"replay {i} (valid_length={v}) does not match its eager reference; "
                f"it matches bound(s) {matches or 'none'} instead "
                f"-- {'stale metadata read' if matches else 'unrelated divergence'}"
            )
    ttnn.release_trace(device, tid)
