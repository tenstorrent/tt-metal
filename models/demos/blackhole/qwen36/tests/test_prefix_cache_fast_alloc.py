# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of QWEN36_PREFIX_CACHE_FAST_ALLOC (GdnPrefixStateCache slots allocated on device, uninitialized).

No weights: the stub GDN layers of test_prefix_cache_host_tier.py (48 layers, rec_state [1, 48/TP, 128, 128] fp32,
conv_carry [1, 3, 10240/TP] bf16, DIFFERENT random bytes on every device). With the flag on:
* every slot / staging tensor has the scratch tensor's shape, padded shape, dtype, layout and memory config, i.e. the
  spec the flag-off allocation (alloc_zeros_like) gives;
* save -> junk -> restore returns the saved state bit for bit on every device, from a device slot and from the host tier
  (QWEN36_PREFIX_HOST_SLOTS) through the staging set;
* allocation time per slot with the flag on vs off is logged ("FAST_ALLOC ... ms/slot").

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_prefix_cache_fast_alloc.py -svq
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_prefix_cache_host_tier import N_LAYERS, _bind_random, _check, _StubModel
from models.demos.blackhole.qwen36.tt.prefix_cache import GdnPrefixStateCache, HostSnap

ENV = "QWEN36_PREFIX_CACHE_FAST_ALLOC"


def _spec(t):
    return (tuple(t.shape), tuple(t.padded_shape), t.dtype, t.layout, t.memory_config())


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fast_alloc_spec_roundtrip_and_time(mesh_device, monkeypatch):
    for k in ("QWEN36_PREFIX_HOST_SPILL_MIN_POS", "QWEN36_PREFIX_HOST_MIN_GAIN", "QWEN36_PREFIX_HOST_SPILL_LOW"):
        monkeypatch.delenv(k, raising=False)
    mesh = mesh_device
    g = torch.Generator().manual_seed(0)
    model = _StubModel(mesh, N_LAYERS)
    tokens = torch.arange(40000, dtype=torch.int32) % 991

    # allocation time, flag off vs on (8 slots each)
    ms = {}
    caches = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(ENV, flag)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        caches[flag] = GdnPrefixStateCache(model, num_slots=8, name=f"fast-alloc {flag}")
        ttnn.synchronize_device(mesh)
        ms[flag] = (time.perf_counter() - t0) * 1e3 / 8
    assert caches["0"].fast_alloc is False and caches["1"].fast_alloc is True
    logger.info(f"FAST_ALLOC layers={N_LAYERS} off={ms['0']:.1f} ms/slot on={ms['1']:.2f} ms/slot")

    # same tensor spec as today's allocation, for every layer's rec / carry
    for (rec0, car0), (rec1, car1), (dn, rec_s, _, car_s, *_) in zip(
        caches["0"]._slots[0], caches["1"]._slots[0], model._gdn_prefill_scratch
    ):
        assert _spec(rec1) == _spec(rec0) and _spec(car1) == _spec(car0)
        assert tuple(rec1.shape) == tuple(rec_s.shape) and tuple(car1.shape) == tuple(car_s.shape)
    for c in caches.values():  # free them (8 x 48 x 2 tensors each) before the round trip below
        for pairs in c._slots:
            for a, b in pairs:
                ttnn.deallocate(a)
                ttnn.deallocate(b)

    # byte-exact round trip from device slots and from the host tier, slots / staging allocated on device
    monkeypatch.setenv(ENV, "1")
    cache = GdnPrefixStateCache(model, num_slots=2, host_slots=2, name="fast-alloc roundtrip")
    st1 = _bind_random(mesh, model, g)
    assert cache.save(tokens, 4096)
    st2 = _bind_random(mesh, model, g)
    assert cache.save(tokens, 8192)
    st3 = _bind_random(mesh, model, g)
    assert cache.save(tokens, 12288)  # evicts 4096 -> host (its slot is reused for 12288)
    assert cache.stats.get("spills", 0) == 1

    _bind_random(mesh, model, g)  # junk in the bound state
    pos, h = cache.lookup(tokens, 8000)
    assert pos == 4096 and isinstance(h, HostSnap)
    cache.restore(h)
    _check(mesh, model, st1, "host restore (fast-alloc staging)")
    for want, p in ((st2, 8192), (st3, 12288)):
        _bind_random(mesh, model, g)
        pos, slot = cache.lookup(tokens, p)
        assert pos == p and not isinstance(slot, HostSnap)
        cache.restore(slot)
        _check(mesh, model, want, f"device restore {p} (fast-alloc slot)")
