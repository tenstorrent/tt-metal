# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests (no device, no real ttnn) of ``Qwen36KVTransfer``'s export-mirror bookkeeping: the begin_export /
end_export window, the chunk observer's copy-once-per-chunk into the open export's sinks, the pump gating by the
per-chunk sync, ``_needs_chunk_sync`` and the failure paths. ``tt/kv_transfer.py`` is loaded under a stub ``ttnn``
into its own module name, so the device test modules of this directory are untouched.

Run: pytest models/demos/blackhole/qwen36/tests/test_kv_transfer_hook_host.py (any host with torch + loguru)."""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_hook_module():
    stub = types.ModuleType("ttnn")
    for name in ("bfloat8_b", "bfloat16", "float32", "int32", "uint32"):
        setattr(stub, name, name)
    stub.TILE_LAYOUT, stub.ROW_MAJOR_LAYOUT, stub.DRAM_MEMORY_CONFIG = "TILE", "ROW_MAJOR", "DRAM"
    real = sys.modules.get("ttnn")
    sys.modules["ttnn"] = stub
    try:
        path = Path(__file__).resolve().parents[1] / "tt" / "kv_transfer.py"
        spec = importlib.util.spec_from_file_location("qwen36_kv_transfer_host_stub", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if real is not None:
            sys.modules["ttnn"] = real
        else:
            sys.modules.pop("ttnn", None)
    return mod


kvt = _load_hook_module()


class _T:
    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype


class _Sink:
    def __init__(self):
        self.writes = []
        self.raise_on = None

    def write_from_device(self, t, *, chunk, blocking=True, cq_id=None):
        if self.raise_on is not None and chunk == self.raise_on:
            raise RuntimeError("device op failed")
        self.writes.append((t, chunk))


def _model(n_attn=2, n_gdn=2):
    attn = [
        SimpleNamespace(
            paged_k=_T((1025, 4, 64, 256), "bfloat8_b"),
            paged_v=_T((1025, 4, 64, 256), "bfloat8_b"),
            _pd_export_k=None,
            _pd_export_v=None,
        )
        for _ in range(n_attn)
    ]
    gdn = [
        SimpleNamespace(
            rec_state=_T((1, 48, 128, 128), "float32"), conv_states=[_T((1, 1, 10240), "bfloat16")] * 4, K=4, B=1
        )
        for _ in range(n_gdn)
    ]
    layers, idx = [], []
    for i, a in enumerate(attn):
        idx.append(len(layers))
        layers.append(SimpleNamespace(is_full_attention=True, attention=a))
        layers.append(SimpleNamespace(is_full_attention=False, attention=gdn[i]))
    m = SimpleNamespace(
        mesh_device=SimpleNamespace(get_num_devices=lambda: 1, num_program_cache_entries=lambda: 0),
        layers=layers,
        _attention_layer_indices=idx,
        _pad_kv_block=1024,
        mtp=None,
        args=SimpleNamespace(max_batch_size=1),
        observer=None,
    )

    def set_obs(fn, sync_each_chunk=False, needs_sync=None):
        m.observer = (fn, sync_each_chunk, needs_sync)

    m.set_prefill_chunk_observer = set_obs
    return m


def _hook(model=None, *, bound=True, sync=True):
    h = kvt.Qwen36KVTransfer(model or _model())
    h._chunk_tokens = 2048
    if bound:
        h._export_staging = {name: object() for name, _ in h._kv_tensors()}
    h._chunk_sync = sync
    return h


def _sinks(h):
    return {name: _Sink() for name, _ in h._kv_tensors()}


def test_begin_export_window_copies_each_chunk_once_and_pumps_between_chunks():
    h = _hook()
    pumps = []
    h.set_chunk_pump(lambda: pumps.append(1))
    sinks = _sinks(h)
    # T-1 = 2560 tokens -> 40 blocks -> 2 wire chunks
    assert h.begin_export(list(range(1, 41)) + [99, 98], 2560, sinks) is True
    exp = h._active_export
    assert (exp.nblk, exp.nchunks, exp.block_ids[-1], exp.written) == (40, 2, 40, set())
    h._on_prefill_chunk(chunk_start=0, n_tokens=2048, final=False)
    for name, s in sinks.items():
        assert s.writes == [(h._export_staging[name], 0)]
    assert exp.written == {0} and pumps == [1] and h.mirrored_chunks == 1
    h._on_prefill_chunk(chunk_start=0, n_tokens=2048, final=False)  # a repeat copies nothing, still pumps
    assert all(len(s.writes) == 1 for s in sinks.values()) and pumps == [1, 1]
    h._on_prefill_chunk(chunk_start=2048, n_tokens=512, final=True)  # the tail: chunk 1, no pump (step end pumps)
    for name, s in sinks.items():
        assert s.writes[1] == (h._export_staging[name], 1)
    assert exp.written == {0, 1} and pumps == [1, 1]
    h._on_prefill_chunk(chunk_start=4096, n_tokens=64, final=True)  # beyond the export: nothing
    h._on_prefill_chunk(chunk_start=100, n_tokens=64, final=False)  # not a chunk boundary: nothing copied, pumps
    assert all(len(s.writes) == 2 for s in sinks.values()) and pumps == [1, 1, 1]
    # the window closes only for its own sinks
    h.end_export(_sinks(h))
    assert h._active_export is exp
    h.end_export(sinks)
    assert h._active_export is None
    h._on_prefill_chunk(chunk_start=0, n_tokens=2048, final=False)  # no window: nothing copied, pump still runs
    assert all(len(s.writes) == 2 for s in sinks.values()) and pumps == [1, 1, 1, 1]


def test_begin_export_replaces_an_open_window_with_a_warning(caplog):
    h = _hook()
    a, b = _sinks(h), _sinks(h)
    h.begin_export([1, 2], 100, a)
    h._on_prefill_chunk(chunk_start=0, n_tokens=100, final=True)
    with caplog.at_level("WARNING"):
        h.begin_export([3, 4], 128, b)
    assert h._active_export.sinks is b and h._active_export.written == set()
    # the same sinks again is not a second window (a re-armed pre-open): no warning, fresh bookkeeping
    h.begin_export([3, 4], 128, b)
    assert h._active_export.sinks is b
    h.end_export(None)
    assert h._active_export is None


def test_unbound_mirror_gathers_everything_and_copies_nothing():
    h = _hook(bound=False)
    sinks = _sinks(h)
    assert h.begin_export([1], 10, sinks) is False
    h._on_prefill_chunk(chunk_start=0, n_tokens=10, final=True)
    assert all(s.writes == [] for s in sinks.values()) and h._active_export.written == set()


def test_pump_gating_needs_sync_and_the_no_sync_mode():
    h = _hook(sync=True)
    assert h._needs_chunk_sync() is False  # no pump wired
    pumps = []
    h.set_chunk_pump(lambda: pumps.append(1))
    assert h._needs_chunk_sync() is True  # pump wired, no wants(): conservative
    h.set_chunk_pump(lambda: pumps.append(1), wants=lambda: False)
    assert h._needs_chunk_sync() is False
    h.set_chunk_pump(lambda: pumps.append(1), wants=lambda: True)
    assert h._needs_chunk_sync() is True
    # TT_PD_CHUNK_SYNC=0: never a boundary pump (host-time boundaries run ahead of the device), never a sync
    h._chunk_sync = False
    assert h._needs_chunk_sync() is False
    sinks = _sinks(h)
    h.begin_export(list(range(1, 33)), 2048, sinks)
    h._on_prefill_chunk(chunk_start=0, n_tokens=2048, final=False)
    assert pumps == [] and all(len(s.writes) == 1 for s in sinks.values())  # copies still happen

    def boom():
        raise ValueError("pump broke")

    h._chunk_sync = True
    h.set_chunk_pump(boom)
    h._on_prefill_chunk(chunk_start=0, n_tokens=2048, final=False)  # a raising pump never takes the prefill down


def test_mirror_copy_failure_leaves_the_chunk_unwritten_for_the_gather():
    h = _hook()
    sinks = _sinks(h)
    names = list(sinks)
    sinks[names[-1]].raise_on = 0  # the last part's device copy raises
    h.begin_export(list(range(1, 33)), 2048, sinks)
    h._on_prefill_chunk(chunk_start=0, n_tokens=2048, final=True)  # no raise out of the observer
    assert h._active_export.written == set() and h.mirrored_chunks == 0
    assert len(sinks[names[0]].writes) == 1 and sinks[names[-1]].writes == []


def test_install_observer_hands_the_model_the_sync_and_needs_sync_seams(monkeypatch):
    monkeypatch.setenv("TT_PD_CHUNK_SYNC", "1")
    h = _hook()
    h._install_observer()
    fn, sync, needs = h.model.observer
    assert fn == h._on_prefill_chunk and sync is True and needs == h._needs_chunk_sync
    monkeypatch.setenv("TT_PD_CHUNK_SYNC", "0")
    h2 = _hook()
    h2._install_observer()
    assert h2.model.observer[1] is False and h2._chunk_sync is False


def test_taps_split_bounds(expect_error):
    h = _hook()
    g = {"conv_dim": 10240}
    h.taps_split = 8  # slab 1280: measured 3.4 MiB of L1 per core, overflows
    with expect_error(AssertionError, "overflows"):
        h._alloc_taps_stage(g, 48, 4)
    h.taps_split = 7  # not tile-aligned
    with expect_error(AssertionError, "tile-aligned"):
        h._alloc_taps_stage(g, 48, 4)
