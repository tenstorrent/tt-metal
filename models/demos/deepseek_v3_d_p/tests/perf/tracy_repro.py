# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
#44928 - ttnn.typecast(uint16 -> int32) determinism / corruption repro suite.

Background
----------
The MoE forward path was:
    gate (moe_grouped_topk -> uint16 / TILE indices)
      -> to_layout(TILE) -> typecast(int32) -> to_layout(ROW_MAJOR)
      -> dispatch (expects int32 indices)

With the profiler OFF the typecast op produced byte-identical int32 outputs
for byte-identical uint16 inputs. With the profiler ON it produced a permuted
~24-token cluster (~0.07% of elements). The shuffled indices fed bad routing
into dispatch and combine, eventually deadlocking the fabric router on a
bogus dst_chip (visible as a hang right after `Expert 8/8`).

The variants below isolate the bug along the two relevant axes:
  - Input layout to typecast (TILE vs ROW_MAJOR), which selects different
    program factories (TypecastProgramFactory vs TypecastRowMajorChunkedProgramFactory).
  - Output dtype (INT32 vs UINT32), which selects different LLK lowerings.
  - typecast op (ttnn.typecast LLK path vs ttnn.experimental.typecast prim::copy path).

Shape / value distribution mirrors the failing production config:
    indices.shape = (1, 1, 3840, 8)   # one chip's view (sharded)
    indices.dtype = uint16
    values        = top-k expert IDs in [0, 256)

How to run
----------
Each test should pass with the profiler OFF and is expected to expose the
LLK bug under tracy when the corresponding kernel path is buggy.

Plain pytest (profiler OFF) - sanity baseline:
    pytest models/demos/deepseek_v3_d_p/tests/perf/tracy_repro.py

Under tracy (profiler ON) - this is the failing configuration:
    python -m tracy -r -p -v -m pytest \
        models/demos/deepseek_v3_d_p/tests/perf/tracy_repro.py

Run one variant at a time with -k, e.g.:
    python -m tracy -r -p -v -m pytest \
        models/demos/deepseek_v3_d_p/tests/perf/tracy_repro.py \
        -k "tile_int32"

Expected (hypothesis - confirm by running):
  - tile_int32_baseline       : FAIL under tracy (known)
  - tile_uint32               : ?     - is the bug INT32-specific or all TILE outputs?
  - rm_int32                  : PASS  - different program factory (chunked RM)
  - rm_uint32                 : PASS  - different program factory + widening cast
  - experimental_tile_int32   : PASS  - prim::copy path, no LLK compute kernel
  - tile_int32_loop           : Same as baseline but iterated; failure rate amplifies.
"""

from collections import OrderedDict

import pytest
import torch

import ttnn

# Per-test outcomes recorded by the makereport hook, rendered by the
# terminal_summary hook at the very end of the run. Plain pytest output gets
# buried under kernel-level logs when running under tracy, so we emit a fenced
# banner that is trivially greppable.
_TEST_RESULTS: "OrderedDict[str, dict]" = OrderedDict()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    entry = _TEST_RESULTS.setdefault(item.nodeid, {"outcome": None, "longrepr": None})
    entry["outcome"] = report.outcome
    if report.failed:
        entry["longrepr"] = str(report.longrepr).splitlines()[0] if report.longrepr else ""


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _TEST_RESULTS:
        return

    bar = "=" * 88
    lines = ["", bar, "  tracy_repro.py - typecast determinism summary  (#44928)", bar]

    name_width = max(len(nodeid.split("::", 1)[-1]) for nodeid in _TEST_RESULTS)
    pass_count = fail_count = skip_count = 0
    for nodeid, entry in _TEST_RESULTS.items():
        short = nodeid.split("::", 1)[-1]
        outcome = (entry["outcome"] or "unknown").upper()
        if outcome == "PASSED":
            pass_count += 1
        elif outcome == "FAILED":
            fail_count += 1
        elif outcome == "SKIPPED":
            skip_count += 1
        line = f"  {short:<{name_width}}  {outcome}"
        if entry.get("longrepr"):
            line += f"  ::  {entry['longrepr']}"
        lines.append(line)

    lines.append(bar)
    lines.append(f"  totals: {pass_count} passed, {fail_count} failed, {skip_count} skipped")
    lines.append(bar)
    lines.append("")
    terminalreporter.write_line("\n".join(lines))


# Shape and value range pulled straight from the failing deepseek MoE config.
# 3840 = isl_6k4 / tp_factor = 7680 / 2.
# 8    = num_experts_per_tok (top-k).
# Values in [0, 256) match actual gate output range.
INDICES_SHAPE = (1, 1, 3840, 8)
EXPERT_ID_RANGE = 256


def _build_uint16_indices(device, seed=0):
    """Return a uint16 ROW_MAJOR DRAM tensor with the same shape and value
    distribution as the deepseek MoE gate output that exposed the bug.

    The actual gate kernel writes UINT16 / TILE; we build ROW_MAJOR here so each
    test below can decide whether to tilize before typecasting. That keeps the
    repro inputs byte-identical across variants without baking in a layout
    assumption.
    """
    torch.manual_seed(seed)
    values = torch.randint(0, EXPERT_ID_RANGE, INDICES_SHAPE, dtype=torch.int32).to(torch.int16)
    return ttnn.from_torch(
        values,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_typecast_matches_torch(out_tt, ref_int32):
    """Compare a ttnn typecast output against the torch reference and emit a
    diagnostic AssertionError when they disagree. Surfaces the first ~15
    differing positions so the LLK owner can see WHERE and HOW the kernel
    diverges (typically a permuted 24-token cluster)."""
    out_host = ttnn.to_torch(out_tt).to(torch.int32)  # int32 view regardless of uint32/int32 storage

    if torch.equal(out_host, ref_int32):
        return

    mismatch = out_host != ref_int32
    n_diff = int(mismatch.sum().item())
    diff_positions = mismatch.nonzero(as_tuple=False)[:15].tolist()
    lines = [
        f"typecast disagrees with torch reference: "
        f"{n_diff}/{out_host.numel()} elements differ "
        f"({100 * n_diff / out_host.numel():.4f}%)",
        f"first {len(diff_positions)} divergences (position -> ttnn vs torch):",
    ]
    for pos in diff_positions:
        ttnn_val = int(out_host[tuple(pos)].item())
        torch_val = int(ref_int32[tuple(pos)].item())
        lines.append(f"  {pos}: ttnn={ttnn_val} torch={torch_val}")
    raise AssertionError("\n".join(lines))


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def test_tile_int32_baseline(device):
    """Original failing chain from tt_moe.py before the fix:
        to_layout(TILE) -> typecast(int32) -> to_layout(ROW_MAJOR)

    Selects TypecastProgramFactory (TILE LLK path), output INT32.
    Expected: PASS without tracy, FAIL under tracy."""
    tt_in = _build_uint16_indices(device)
    ref = ttnn.to_torch(tt_in).to(torch.int32)

    x = ttnn.to_layout(tt_in, ttnn.TILE_LAYOUT)
    x = ttnn.typecast(x, ttnn.int32)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    _assert_typecast_matches_torch(x, ref)
    ttnn.deallocate(x)


def test_tile_uint32(device):
    """TILE input, UINT32 output. Isolates whether the bug is specific to the
    INT32 output lowering or affects all TILE typecast outputs from UINT16.

    Selects TypecastProgramFactory (TILE LLK path), output UINT32.
    Expected: tells us whether switching only the output dtype is sufficient."""
    tt_in = _build_uint16_indices(device)
    ref = ttnn.to_torch(tt_in).to(torch.int32)

    x = ttnn.to_layout(tt_in, ttnn.TILE_LAYOUT)
    x = ttnn.typecast(x, ttnn.uint32)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    _assert_typecast_matches_torch(x, ref)
    ttnn.deallocate(x)


def test_rm_int32(device):
    """ROW_MAJOR input, INT32 output. Untilize BEFORE typecast so the dispatcher
    selects TypecastRowMajorChunkedProgramFactory (a different kernel entirely).

    Expected: PASS under tracy if the bug is layout-scoped to the TILE LLK."""
    tt_in = _build_uint16_indices(device)
    ref = ttnn.to_torch(tt_in).to(torch.int32)

    x = ttnn.to_layout(tt_in, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.typecast(x, ttnn.int32)

    _assert_typecast_matches_torch(x, ref)
    ttnn.deallocate(x)


def test_rm_uint32(device):
    """ROW_MAJOR input, UINT32 output. Same kernel as test_rm_int32 but a pure
    widening cast (uint16 -> uint32), which is the cheapest correct conversion
    for dispatch (which accepts INT32 || UINT32).

    Expected: PASS under tracy. This is the recommended Option A workaround."""
    tt_in = _build_uint16_indices(device)
    ref = ttnn.to_torch(tt_in).to(torch.int32)

    x = ttnn.to_layout(tt_in, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.typecast(x, ttnn.uint32)

    _assert_typecast_matches_torch(x, ref)
    ttnn.deallocate(x)


def test_experimental_typecast_tile_int32(device):
    """ttnn.experimental.typecast routes through prim::copy (data-movement op,
    no LLK compute kernel). Same input layout (TILE) and output dtype (INT32)
    as the failing baseline, so a PASS here pinpoints the bug to the LLK
    compute path used by ttnn.typecast.

    Expected: PASS under tracy."""
    tt_in = _build_uint16_indices(device)
    ref = ttnn.to_torch(tt_in).to(torch.int32)

    x = ttnn.to_layout(tt_in, ttnn.TILE_LAYOUT)
    x = ttnn.experimental.typecast(x, ttnn.int32)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    _assert_typecast_matches_torch(x, ref)
    ttnn.deallocate(x)


@pytest.mark.parametrize("iterations", [16])
def test_tile_int32_loop(device, iterations):
    """Re-run the failing chain N times in one test to amplify the failure rate
    under tracy. Useful when the single-shot baseline is intermittent or when
    triaging fix candidates."""
    tt_in = _build_uint16_indices(device)
    ref = ttnn.to_torch(tt_in).to(torch.int32)

    failures = []
    for i in range(iterations):
        x = ttnn.to_layout(tt_in, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(x, ttnn.int32)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        out_host = ttnn.to_torch(x).to(torch.int32)
        ttnn.deallocate(x)
        if not torch.equal(out_host, ref):
            n_diff = int((out_host != ref).sum().item())
            failures.append((i, n_diff))

    if failures:
        msg_lines = [
            f"{len(failures)}/{iterations} iterations diverged from torch reference:",
        ]
        for idx, n_diff in failures[:10]:
            msg_lines.append(f"  iter {idx}: {n_diff} differing elements")
        raise AssertionError("\n".join(msg_lines))
