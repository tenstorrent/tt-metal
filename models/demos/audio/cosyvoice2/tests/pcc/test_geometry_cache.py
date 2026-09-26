# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`GeometryWeightCache` (tt/geometry_cache.py) and its use in `TtConv1d`/
`TtConvTranspose1d` for the DRAM leak this round's item closes: prepared conv weights
were cached per `(input_length, batch_size)` geometry with no eviction, so DRAM grew
without bound as a session saw more distinct lengths (measured: 90.9 MB/bank -> 134.1
MB/bank across four utterance lengths in the 2026-09-21 regression run). Threshold-based
eviction (real free-DRAM pressure, not a per-utterance schedule) is designed for reuse by
streaming's own chunk-shape churn later, not as a one-off patch -- see geometry_cache.py's
module docstring.
"""

from __future__ import annotations

import pytest
import torch

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)


# --------------------------------------------------------------------------
# host tier -- GeometryWeightCache's own LRU/eviction logic, no conv involved
# --------------------------------------------------------------------------
@needs_l1_small
def test_geometry_cache_evicts_lru_when_threshold_forces_it(device):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.geometry_cache import GeometryWeightCache

    def free_bytes():
        return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_free_per_bank

    def alloc():
        return ttnn.from_torch(torch.randn(1, 1, 256), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Threshold = free bytes right now, in exact bytes (not MB-rounded, which would
    # swallow a tensor this small in its rounding slack) -- so ANY further allocation
    # trips eviction on the next put(), deterministically, regardless of how much DRAM
    # happens to be free on this run.
    cache = GeometryWeightCache(device, threshold_mb=1)
    cache._threshold_bytes = free_bytes()

    t1 = alloc()
    cache.put("a", [t1], "meta-a")
    assert len(cache) == 1
    assert cache.get("a") == "meta-a"

    # Inserting a second entry should evict "a" (the only other key) to satisfy the
    # threshold, since "a" is now the least-recently-used entry.
    t2 = alloc()
    cache.put("b", [t2], "meta-b")
    assert cache.get("a") is None, "the LRU entry should have been evicted"
    assert cache.get("b") == "meta-b", "the just-inserted entry must never evict itself"
    assert len(cache) == 1


@needs_l1_small
def test_geometry_cache_get_refreshes_lru_order(device):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.geometry_cache import GeometryWeightCache

    def free_bytes():
        return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_free_per_bank

    def alloc():
        return ttnn.from_torch(torch.randn(1, 1, 256), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    cache = GeometryWeightCache(device, threshold_mb=1)
    # Measure one allocation's real DRAM footprint, then set the threshold so that
    # exactly two of these small entries fit but a third does not -- deterministic
    # regardless of how much DRAM happens to be free on this run.
    free0 = free_bytes()
    probe = alloc()
    consumed = free0 - free_bytes()
    ttnn.deallocate(probe)
    cache._threshold_bytes = free_bytes() - int(consumed * 2.5)

    cache.put("a", [alloc()], "meta-a")
    cache.put("b", [alloc()], "meta-b")
    # Touch "a" so it becomes MRU -- "b" is now the least-recently-used.
    assert cache.get("a") == "meta-a"
    cache.put("c", [alloc()], "meta-c")
    # "b" (LRU) must be the one evicted, if anything was; "a" (freshly touched) and "c"
    # (just inserted) must both survive.
    assert cache.get("b") is None, "the LRU entry should have been evicted, not a or c"
    assert cache.get("a") == "meta-a"
    assert cache.get("c") == "meta-c"


@needs_l1_small
def test_geometry_cache_release_deallocates_everything(device):
    import ttnn
    from models.demos.audio.cosyvoice2.tt.geometry_cache import GeometryWeightCache

    cache = GeometryWeightCache(device, threshold_mb=1)  # threshold never trips on its own here
    t1 = ttnn.from_torch(torch.randn(1, 1, 256), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cache.put("a", [t1], "meta-a")
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None


# --------------------------------------------------------------------------
# device tier -- TtConv1d/TtConvTranspose1d actually evicting and re-preparing correctly
# --------------------------------------------------------------------------
@needs_l1_small
def test_conv1d_stays_correct_across_forced_eviction(device):
    """Real regression for the ownership-transfer logic in TtConv1d._resolve: force a
    LOW threshold so the second geometry's `put()` evicts the first geometry's cache
    entries, then come back to the FIRST geometry -- it must re-prepare and re-verify
    from scratch, and still be numerically correct, not silently reuse a freed tensor."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.conv import TtConv1d

    g = torch.Generator().manual_seed(0)
    in_ch, out_ch, k = 8, 8, 3
    w = torch.randn(out_ch, in_ch, k, generator=g) / (in_ch * k) ** 0.5
    b = torch.randn(out_ch, generator=g) * 0.1

    conv = TtConv1d(device, w, b, stride=1, padding=1, dtype=ttnn.float32)

    def free_bytes():
        return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_free_per_bank

    def check(length):
        x = torch.randn(1, length, in_ch, generator=g) * 0.5
        want = torch.nn.functional.conv1d(
            x.transpose(1, 2).double(), w.double(), b.double(), stride=1, padding=1
        ).transpose(1, 2)
        x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        out, out_len = conv(x_dev, length, 1)
        got = ttnn.to_torch(out).float().reshape(1, out_len, out_ch).double()
        rel = float((got - want).norm() / want.norm())
        assert rel < 0.01, f"length={length}: rel err {rel}"
        return rel

    r1 = check(32)  # first geometry: fresh prepare + verify
    assert conv._verified_config.get((32, 1)) is not None, "geometry 1 should have resolved and cached"

    # Force eviction on the NEXT cache write, deterministically: a threshold set to
    # "current free plus one byte" guarantees ANY further allocation trips it, regardless
    # of that allocation's actual size -- an artificially tight threshold for this test
    # only, not a realistic operating point.
    conv._verified_config._threshold_bytes = free_bytes() + 1

    r2 = check(64)  # second geometry: its own cache write evicts geometry 1's entry
    assert conv._verified_config.get((32, 1)) is None, "geometry 1 should have been evicted"
    assert conv._verified_config.get((64, 1)) is not None, "the just-resolved geometry must survive its own write"

    r1b = check(32)  # BACK to the first geometry: must re-prepare + re-verify correctly
    print(f"\n  rel err: L=32 first={r1:.4f}  L=64={r2:.4f}  L=32 after eviction+re-prepare={r1b:.4f}")


@needs_l1_small
def test_conv_transpose1d_stays_correct_across_forced_eviction(device):
    """Same regression, for TtConvTranspose1d."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.upsample import TtConvTranspose1d

    g = torch.Generator().manual_seed(1)
    in_ch, out_ch, k, stride = 8, 8, 4, 2
    w = torch.randn(in_ch, out_ch, k, generator=g) / (in_ch * k) ** 0.5
    b = torch.randn(out_ch, generator=g) * 0.1

    conv = TtConvTranspose1d(device, w, b, stride=stride, padding=1, dtype=ttnn.float32)

    def free_bytes():
        return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_free_per_bank

    def check(length):
        x = torch.randn(1, length, in_ch, generator=g) * 0.5
        want = torch.nn.functional.conv_transpose1d(
            x.transpose(1, 2).double(), w.double(), b.double(), stride=stride, padding=1
        ).transpose(1, 2)
        x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        out, out_len = conv(x_dev, length, 1)
        got = ttnn.to_torch(out).float().reshape(1, out_len, out_ch).double()
        rel = float((got - want).norm() / want.norm())
        assert rel < 0.01, f"length={length}: rel err {rel}"
        return rel

    r1 = check(32)
    assert conv._verified_config.get((32, 1)) is not None, "geometry 1 should have resolved and cached"

    conv._verified_config._threshold_bytes = free_bytes() + 1

    r2 = check(64)
    assert conv._verified_config.get((32, 1)) is None, "geometry 1 should have been evicted"
    assert conv._verified_config.get((64, 1)) is not None, "the just-resolved geometry must survive its own write"

    r1b = check(32)
    print(f"\n  rel err: L=32 first={r1:.4f}  L=64={r2:.4f}  L=32 after eviction+re-prepare={r1b:.4f}")


@needs_l1_small
def test_release_caches_then_reuse_stays_correct(device):
    """release_caches() (the explicit, all-at-once counterpart to threshold eviction)
    must leave the instance fully usable afterward -- the next call re-prepares from
    scratch and is still correct."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.hifigan.conv import TtConv1d

    g = torch.Generator().manual_seed(2)
    in_ch, out_ch, k = 8, 8, 3
    w = torch.randn(out_ch, in_ch, k, generator=g) / (in_ch * k) ** 0.5
    b = torch.randn(out_ch, generator=g) * 0.1
    length = 40

    conv = TtConv1d(device, w, b, stride=1, padding=1, dtype=ttnn.float32)
    x = torch.randn(1, length, in_ch, generator=g) * 0.5
    want = torch.nn.functional.conv1d(x.transpose(1, 2).double(), w.double(), b.double(), stride=1, padding=1).transpose(
        1, 2
    )
    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out, out_len = conv(x_dev, length, 1)
    got1 = ttnn.to_torch(out).float().reshape(1, out_len, out_ch).double()
    assert float((got1 - want).norm() / want.norm()) < 0.01

    conv.release_caches()
    assert len(conv._prep_cache) == 0
    assert len(conv._verified_config) == 0

    x_dev2 = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out2, out_len2 = conv(x_dev2, length, 1)
    got2 = ttnn.to_torch(out2).float().reshape(1, out_len2, out_ch).double()
    rel = float((got2 - want).norm() / want.norm())
    print(f"\n  rel err after release_caches() + reuse: {rel:.4f}")
    assert rel < 0.01, rel
