# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.untilize with gathered_prefix_tensor: only the gathered prefix of each row segment is untilized, the extent read
on the device (trace-safe). g = min(round_up(start + slab, slab), H) rows over num_segments segments, the first
g / num_segments rows of each, bit-equal to a full untilize there."""

import pytest
import torch
import ttnn

SLAB, SEGMENTS = 5120, 8


def _extent(device, start):
    return ttnn.from_torch(
        torch.tensor([[[[start]]]], dtype=torch.int64),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _gathered_rows(h, start):
    g = min(-(-(start + SLAB) // SLAB) * SLAB, h)
    return g // SEGMENTS


def _check(out, ref, h, start):
    per_seg, keep = h // SEGMENTS, _gathered_rows(h, start)
    o = out.reshape(SEGMENTS, per_seg, -1)[:, :keep]
    r = ref.reshape(SEGMENTS, per_seg, -1)[:, :keep]
    assert torch.equal(o, r), f"start {start}: gathered rows differ ({(o != r).sum().item()} values)"


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
@pytest.mark.parametrize("h, w", [(4 * SLAB, 576), (8 * SLAB, 128)])
@pytest.mark.parametrize("start_kind", ["zero", "mid", "full"])
def test_untilize_gathered_prefix(device, dtype, h, w, start_kind):
    torch.manual_seed(0)
    start = {"zero": 0, "mid": (h // SLAB // 2) * SLAB, "full": h - SLAB}[start_kind]
    x = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ref = ttnn.to_torch(ttnn.untilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ext = _extent(device, start)
    out = ttnn.untilize(
        t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        gathered_prefix_tensor=ext,
        gathered_slab_global=SLAB,
        num_segments=SEGMENTS,
    )
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT and list(out.shape) == [1, 1, h, w]
    _check(ttnn.to_torch(out), ref, h, start)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_untilize_gathered_prefix_trace(device):
    """One capture, the extent tensor rewritten in place between replays: each replay untilizes its own prefix."""
    torch.manual_seed(1)
    h, w = 4 * SLAB, 576
    x = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ref = ttnn.to_torch(ttnn.untilize(t, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    ext = _extent(device, 0)
    kw = dict(
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        gathered_prefix_tensor=ext,
        gathered_slab_global=SLAB,
        num_segments=SEGMENTS,
    )
    ttnn.deallocate(ttnn.untilize(t, **kw))  # compile before the capture
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = ttnn.untilize(t, **kw)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        for start in (0, SLAB, 3 * SLAB, 2 * SLAB):
            host = ttnn.from_torch(
                torch.tensor([[[[start]]]], dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(host, ext)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            _check(ttnn.to_torch(out), ref, h, start)
    finally:
        ttnn.release_trace(device, tid)
