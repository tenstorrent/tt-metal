# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for QWEN36_PREFIX_CACHE_FAST_ALLOC (GdnPrefixStateCache slot allocation):

* flag off: every slot tensor (and the host-tier staging set) comes from alloc_zeros_like, as today; the on-device
  allocator is never called;
* flag on: every one comes from ttnn.allocate_tensor_on_device with the scratch tensor's shape and dtype, TILE layout,
  DRAM interleaved, on the GDN layer's mesh; alloc_zeros_like is never called;
* the cache built either way saves / looks up / restores identically (device copies recorded).
The device check (allocation time, byte-exact save / restore, host tier) is test_prefix_cache_fast_alloc.py.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefix_cache_fast_alloc_host.py -q
"""
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import prefix_cache as pc

ENV = "QWEN36_PREFIX_CACHE_FAST_ALLOC"


class _Mesh:
    def get_num_devices(self):
        return 8


class _Model:
    """The B=1 GDN scratch tuples (dn, rec, conv_states, carry, ...) GdnPrefixStateCache mirrors."""

    def __init__(self, layers=3):
        self.mesh = _Mesh()
        self._gdn_prefill_scratch = []
        for i in range(layers):
            rec = types.SimpleNamespace(shape=(1, 6, 128, 128), dtype=ttnn.float32, tag=("rec", i))
            carry = types.SimpleNamespace(shape=(1, 3, 1280), dtype=ttnn.bfloat16, tag=("carry", i))
            dn = types.SimpleNamespace(rec_state=("bound_rec", i), conv_carry=("bound_carry", i), mesh=self.mesh)
            self._gdn_prefill_scratch.append((dn, rec, None, carry, None, None))

    def _ensure_gdn_prefill_scratch(self):
        pass


@pytest.fixture
def rec(monkeypatch):
    calls = {"zeros": [], "empty": [], "copy": []}
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(
        pc, "alloc_zeros_like", lambda t, mesh: calls["zeros"].append((t, mesh)) or ("Z", len(calls["zeros"]))
    )
    monkeypatch.setattr(
        ttnn,
        "allocate_tensor_on_device",
        lambda *a: calls["empty"].append(a) or ("E", len(calls["empty"])),
    )
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: calls["copy"].append((src, dst)))
    return calls


@pytest.mark.parametrize("host_slots", [0, 2])
def test_flag_off_allocates_as_today(rec, host_slots):
    m = _Model(3)
    c = pc.GdnPrefixStateCache(m, num_slots=4, host_slots=host_slots)
    assert c.fast_alloc is False
    n = 2 * 3 * (4 + (1 if host_slots else 0))  # (rec, carry) x layers x (slots + staging)
    assert len(rec["zeros"]) == n and not rec["empty"]
    assert all(mesh is m.mesh for _, mesh in rec["zeros"])


@pytest.mark.parametrize("host_slots", [0, 2])
def test_flag_on_allocates_on_device_with_the_scratch_spec(rec, monkeypatch, host_slots):
    monkeypatch.setenv(ENV, "1")
    m = _Model(3)
    c = pc.GdnPrefixStateCache(m, num_slots=4, host_slots=host_slots)
    assert c.fast_alloc is True
    assert not rec["zeros"]
    assert len(rec["empty"]) == 2 * 3 * (4 + (1 if host_slots else 0))
    want = []
    for _, rect, _, carry, *_ in m._gdn_prefill_scratch:
        want += [(tuple(rect.shape), rect.dtype), (tuple(carry.shape), carry.dtype)]
    for k, (shape, dtype, layout, mesh, memcfg) in enumerate(rec["empty"]):
        assert (tuple(shape), dtype) == want[k % len(want)]
        assert layout == ttnn.TILE_LAYOUT and memcfg == ttnn.DRAM_MEMORY_CONFIG and mesh is m.mesh


def test_cache_behaves_the_same_either_way(rec, monkeypatch):
    tokens = torch.arange(20000, dtype=torch.int32) % 991
    seqs = []
    for flag in ("0", "1"):
        monkeypatch.setenv(ENV, flag)
        rec["copy"].clear()
        c = pc.GdnPrefixStateCache(_Model(2), num_slots=2)
        assert c.save(tokens, 4096) and c.save(tokens, 8192) and c.save(tokens, 12288)  # third evicts 4096
        pos, slot = c.lookup(tokens, 10000)
        c.restore(slot)
        seqs.append(
            (
                pos,
                [k[0] for k in c._lru],
                c.stats["saves"],
                c.stats["evictions"],
                len(rec["copy"]),
                [
                    (s if isinstance(s, tuple) and s[0] in ("bound_rec", "bound_carry") else "slot")
                    for s, _ in rec["copy"]
                ],
            )
        )
    assert seqs[0] == seqs[1]
    assert seqs[0][0] == 8192 and seqs[0][1] == [12288, 8192]
