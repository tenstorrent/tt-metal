# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.transformer.sdpa_k_split_merge (the merge of ring joint SDPA's K-split partitions) vs an fp64 host reference,
over split counts (1 to 6: the fast two-session path for <= 3, the row-sum pre-session above), batch, heads, rows,
value widths (odd tile counts exercise the partial column group) and padded statistics rows."""

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler


def _ref(o, m, l, S, scale):
    """o [B, S NH, N, DV], m [B, S NH, N] row max, l [B, S NH, N, 32] per-column partial sums."""
    B, SNH, N, DV = o.shape
    NH = SNH // S
    o, m, L = o.double().view(B, S, NH, N, DV), m.double().view(B, S, NH, N), l.double().sum(-1).view(B, S, NH, N)
    M = m.max(1, keepdim=True).values
    a = torch.exp(scale * (m - M))
    return (a[..., None] * o).sum(1) / (a * L).sum(1)[..., None]


@pytest.mark.parametrize(
    "S, B, NH, N, DV, pad_rows",
    [
        (1, 1, 2, 64, 128, 0),
        (2, 1, 32, 256, 128, 0),
        (3, 1, 32, 128, 128, 64),
        (3, 2, 3, 96, 160, 0),
        (2, 1, 5, 64, 32, 32),
        (4, 1, 8, 128, 128, 0),
        (5, 1, 4, 64, 192, 0),
        (6, 2, 3, 64, 96, 32),
    ],
    ids=lambda v: str(v),
)
def test_sdpa_k_split_merge(device, S, B, NH, N, DV, pad_rows):
    torch.manual_seed(S * 1000 + NH)
    scale = 192**-0.5
    o = torch.randn(B, S * NH, N, DV) * 20
    m = torch.randn(B, S * NH, N) * 8  # running max (pre-scale scores): partitions differ widely
    l = torch.rand(B, S * NH, N, 32) * 4 + 0.05
    o, m, l = o.bfloat16().float(), m.bfloat16().float(), l.bfloat16().float()
    rows = N + pad_rows  # the ring op's stats scratch can be taller than the output
    stats = torch.zeros(B, S * NH, 2 * rows, 32)
    stats[:, :, :N, 0] = m
    stats[:, :, :N, 1:] = torch.randn(B, S * NH, N, 31)  # max tile columns past 0 are ignored
    stats[:, :, rows : rows + N, :] = l
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.transformer.sdpa_k_split_merge(tt(o), tt(stats), S, scale)
    assert list(out.shape) == [B, NH, N, DV]
    got = ttnn.to_torch(out).double()
    ref = _ref(o, m, l, S, scale)
    rel = ((got - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
    worst = ((got - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    print(f"S={S} B={B} NH={NH} N={N} DV={DV}: pcc {pcc:.6f} rel {rel:.2e} worst {worst:.2e}")
    assert pcc > 0.9999 and rel < 8e-3, (pcc, rel)


def test_sdpa_k_split_merge_program_cache(device):
    """Second call with new buffers must hit the program cache and patch the addresses."""
    S, NH, N, DV = 2, 4, 64, 128
    scale = 0.1
    outs = []
    for it in range(3):
        torch.manual_seed(it)
        o = torch.randn(1, S * NH, N, DV).bfloat16().float()
        m = torch.randn(1, S * NH, N).bfloat16().float()
        l = (torch.rand(1, S * NH, N, 32) + 0.1).bfloat16().float()
        stats = torch.zeros(1, S * NH, 2 * N, 32)
        stats[:, :, :N, 0] = m
        stats[:, :, N:, :] = l
        keep = ttnn.from_torch(torch.zeros(1, 1, 32 * (it + 1), 64), layout=ttnn.TILE_LAYOUT, device=device)
        tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.transformer.sdpa_k_split_merge(tt(o), tt(stats), S, scale)
        got, ref = ttnn.to_torch(out).double(), _ref(o, m, l, S, scale)
        assert ((got - ref).norm() / ref.norm()).item() < 8e-3
        outs.append(keep)
    assert device.num_program_cache_entries() == 1


def _inputs(S, B, NH, N, DV, pad_rows=0, seed=0):
    torch.manual_seed(seed)
    o = (torch.randn(B, S * NH, N, DV) * 20).bfloat16().float()
    m = (torch.randn(B, S * NH, N) * 8).bfloat16().float()
    l = (torch.rand(B, S * NH, N, 32) * 4 + 0.05).bfloat16().float()
    rows = N + pad_rows
    stats = torch.zeros(B, S * NH, 2 * rows, 32)
    stats[:, :, :N, 0] = m
    stats[:, :, rows : rows + N, :] = l
    return o, m, l, stats


def _mismatch_marker(reference, actual):
    """On-device exact compare (as tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py): a one-element tensor, non-zero
    iff any element differs; outputs never leave the device."""
    return ttnn.max(ttnn.ne(reference, actual, dtype=ttnn.bfloat16))


@pytest.mark.parametrize("S", [2, 3, 5], ids=lambda s: f"s{s}")
def test_sdpa_k_split_merge_determinism(device, S):
    """50 merges of the same inputs are bitwise identical (compared on device)."""
    o, _, _, stats = _inputs(S, 1, 32, 256, 128)
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    to, ts = tt(o), tt(stats)
    reference = ttnn.transformer.sdpa_k_split_merge(to, ts, S, 0.07)
    marker = None
    for _ in range(49):
        out = ttnn.transformer.sdpa_k_split_merge(to, ts, S, 0.07)
        m = _mismatch_marker(reference, out)
        marker = m if marker is None else ttnn.maximum(marker, m)
        out.deallocate(True)
    assert float(ttnn.to_torch(marker).item()) == 0.0, "sdpa_k_split_merge is not deterministic"


# Device duration (ns) per (k_split, heads, rows, dv) on a BH p150, median of 3 dispatches; MiMo-V2 GA shapes
# (QuietBox SP2 x TP2: 32 local heads at 2048 / 640 tokens per chip; Galaxy TP4: 16 heads at 640). Recalibrate from
# the "RT-CAL" lines. The merge is DRAM-bound: bytes = (S + 1) x output + stats.
_PERF_EXPECTED_NS = {  # BH p150 (QuietBox), 2026-09-29
    (2, 32, 2048, 128): 181_319,
    (3, 32, 2048, 128): 258_043,
    (3, 32, 640, 128): 86_430,
    (3, 16, 640, 128): 43_849,
}
_PERF_MARGIN = 0.05


@pytest.mark.parametrize(
    "S, NH, N, DV",
    [(2, 32, 2048, 128), (3, 32, 2048, 128), (3, 32, 640, 128), (3, 16, 640, 128)],
    ids=lambda v: str(v),
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_sdpa_k_split_merge_perf(device, S, NH, N, DV):
    require_realtime_profiler("sdpa_k_split_merge perf checks")
    o, _, _, stats = _inputs(S, 1, NH, N, DV)
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    to, ts = tt(o), tt(stats)
    run = lambda: ttnn.transformer.sdpa_k_split_merge(to, ts, S, 0.07).deallocate(True)
    key = (S, NH, N, DV)
    expected = _PERF_EXPECTED_NS.get(key)
    ns = assert_op_duration_merged(
        device,
        run,
        "/ksplit_merge",
        expected_ns=expected or 1,
        margin=_PERF_MARGIN if expected else float("inf"),
        label=f"{key}",
        iters=3,
    )
    gbytes = ((S + 1) * NH * N * DV + S * NH * 2 * N * 32) * 2 / 1e9
    print(f"sdpa_k_split_merge {key}: {ns / 1e3:.1f} us, {gbytes / (ns * 1e-9):.0f} GB/s")
    if expected is None:
        pytest.skip(f"no baseline for {key}; add it to _PERF_EXPECTED_NS")


# Device duration (ns) per (k_split, heads, rows, dv) on a BH p150, median of 3 dispatches; MiMo-V2 GA shapes
# (QuietBox SP2 x TP2: 32 local heads at 2048 / 640 tokens per chip; Galaxy TP4: 16 heads at 640). Recalibrate from
# the "RT-CAL" lines. The merge is DRAM-bound: bytes = (S + 1) x output + stats.
_PERF_EXPECTED_NS = {  # BH p150 (QuietBox), 2026-09-29
    (2, 32, 2048, 128): 181_319,
    (3, 32, 2048, 128): 258_043,
    (3, 32, 640, 128): 86_430,
    (3, 16, 640, 128): 43_849,
}
_PERF_MARGIN = 0.05


@pytest.mark.parametrize(
    "S, NH, N, DV",
    [(2, 32, 2048, 128), (3, 32, 2048, 128), (3, 32, 640, 128), (3, 16, 640, 128)],
    ids=lambda v: str(v),
)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_sdpa_k_split_merge_perf(device, S, NH, N, DV):
    require_realtime_profiler("sdpa_k_split_merge perf checks")
    o, _, _, stats = _inputs(S, 1, NH, N, DV)
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    to, ts = tt(o), tt(stats)
    run = lambda: ttnn.transformer.sdpa_k_split_merge(to, ts, S, 0.07).deallocate(True)
    key = (S, NH, N, DV)
    expected = _PERF_EXPECTED_NS.get(key)
    ns = assert_op_duration_merged(
        device,
        run,
        "/ksplit_merge",
        expected_ns=expected or 1,
        margin=_PERF_MARGIN if expected else float("inf"),
        label=f"{key}",
        iters=3,
    )
    gbytes = ((S + 1) * NH * N * DV + S * NH * 2 * N * 32) * 2 / 1e9
    print(f"sdpa_k_split_merge {key}: {ns / 1e3:.1f} us, {gbytes / (ns * 1e-9):.0f} GB/s")
    if expected is None:
        pytest.skip(f"no baseline for {key}; add it to _PERF_EXPECTED_NS")
