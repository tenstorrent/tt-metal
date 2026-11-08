# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for the r67 merge of QWEN36_PREFILL_GDN_GLUE (prefill-r6) with QWEN36_PREFILL_INTRACE_SNAP
(tail-planner):

* the merge guard predicate (gdn/tp.py snap_tile_gather_ok): a snapshot-carrying prefill body keeps the GDN glue's
  TILE conv only with QWEN36_PREFILL_SNAP_TILE's "conv_idx_tile" rows and T a multiple of 128;
* QWEN36_PREFILL_SNAP_TILE's gather (gdn/tp.py snap_rows_from_tile with prefill_plan.snap_tile_index_words), run on a
  torch stand-in for the five ttnn calls it makes, gathers exactly the rows the row-major gather (kda_conv_prefill's
  embedding over the untilized conv input with snap_index_words' conv words) gathers, for every staged snapshot;
* snap_tile_index_words rejects row offsets the 128-row block gather cannot serve.

Run (worktree first on PYTHONPATH, ttnn from a built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_r67_snap_tile_host.py -q
"""
import random
import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt import prefill_plan as pp
from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp
from models.demos.blackhole.qwen36.tt.tp_common import GDN_SNAP_SLOTS


class _TorchTTNN:
    """The ttnn surface snap_rows_from_tile touches, on torch tensors (layouts are no-ops: values are what matter)."""

    TILE_SIZE = 32
    DRAM_MEMORY_CONFIG = "dram"
    ROW_MAJOR_LAYOUT = "rm"

    def __init__(self):
        self.calls = []

    def reshape(self, t, shape):
        self.calls.append(("reshape", tuple(shape)))
        return t.reshape(tuple(shape))

    def slice(self, t, start, end, memory_config=None):
        self.calls.append(("slice", tuple(start), tuple(end)))
        return t[tuple(slice(s, e) for s, e in zip(start, end))].clone()

    def to_layout(self, t, layout, memory_config=None):
        self.calls.append(("to_layout", layout))
        return t.clone()

    def embedding(self, idx, weight, layout=None, memory_config=None):
        self.calls.append(("embedding", tuple(weight.shape)))
        return weight[idx.long()]  # [1, R] ids -> [1, R, C]

    def copy(self, src, dst):
        self.calls.append(("copy",))
        dst.copy_(src)

    def deallocate(self, t):
        self.calls.append(("deallocate",))


def _row_major_gather(x, rel_rows):
    """kda_conv_prefill's snapshot gather: embedding over the [T, C] conv input with snap_index_words' conv words."""
    _, conv = pp.snap_index_words(rel_rows, GDN_SNAP_SLOTS)
    return x[0][torch.tensor(conv)]


@pytest.mark.parametrize("T", [128, 256, 1024, 2048, 8192])
def test_snap_rows_from_tile_matches_row_major_gather(monkeypatch, T):
    fake = _TorchTTNN()
    monkeypatch.setattr(gdn_tp, "ttnn", fake)
    rng = random.Random(T)
    C, W = 96, 160  # conv columns [0, C) of a wider (qkvzab-like) projection
    x = torch.randn(1, T, W, dtype=torch.float32)
    candidates = list(range(128, T, 128))  # in-replay points: strictly inside the replay, 128-aligned
    cases = [[]] + [sorted(rng.sample(candidates, k)) for k in range(1, min(GDN_SNAP_SLOTS, len(candidates)) + 1)]
    if candidates:
        cases.append([candidates[-1]])  # the last block boundary before the replay end
    for rel_rows in cases:
        out = torch.full((1, 4 * GDN_SNAP_SLOTS, C), float("nan"))
        words = pp.snap_tile_index_words(rel_rows, GDN_SNAP_SLOTS)
        gdn_tp.snap_rows_from_tile(x, T, C, torch.tensor([words]), out)
        ref = _row_major_gather(x[..., :C], rel_rows)
        n = 3 * len(rel_rows)  # rows 3k..3k+2 = snapshot k; the rest are never read (copy_snapshot_to)
        assert torch.equal(out[0, :n], ref[:n]), (T, rel_rows)
        for k, r in enumerate(rel_rows):  # and they are the three conv-input rows before the snapshot position
            assert torch.equal(out[0, 3 * k : 3 * k + 3], x[0, r - 3 : r, :C]), (T, r)
    # the work is a quarter of the rows: one view, one tile-aligned slice of every block's last tile row, one untilize
    nb = T // 128
    assert ("reshape", (nb, 128, W)) in fake.calls
    assert ("slice", (0, 96, 0), (nb, 128, C)) in fake.calls
    assert sum(1 for c in fake.calls if c[0] == "to_layout") == len(cases)


def test_snap_tile_index_words_layout():
    # block i's last tile row (rows 128i+96 .. 128i+127) becomes untilized rows 32i .. 32i+31
    assert pp.snap_tile_index_words([128], 4) == [29, 30, 31] + [0] * 13
    assert pp.snap_tile_index_words([256, 1024], 4) == [61, 62, 63, 253, 254, 255] + [0] * 10
    assert pp.snap_tile_index_words([], 4) == [0] * 16
    with pytest.raises(AssertionError):
        pp.snap_tile_index_words([160], 4)  # 32-aligned but not 128-aligned: not a block's last tile row
    with pytest.raises(AssertionError):
        pp.snap_tile_index_words([128, 256, 384, 512, 640], 4)  # more than the slots


def test_snap_tile_gather_ok():
    ok = gdn_tp.snap_tile_gather_ok
    assert ok(None, 8192) and ok(None, 100)  # no snapshot ops: the glue conv is unaffected
    snap = {"rec_idx": object(), "conv_idx": object()}
    assert not ok(snap, 8192)  # SNAP_TILE off: the merge guard keeps the row-major conv
    snap["conv_idx_tile"] = object()
    assert ok(snap, 8192) and ok(snap, 2048) and ok(snap, 128)
    assert not ok(snap, 96) and not ok(snap, 8192 + 32)  # not whole 128-row blocks


def test_forward_prefill_guard_source():
    """forward_prefill applies the guard right after the glue plan and dispatches the snapshot to whichever conv runs
    (a static check: building a TPGatedDeltaNet needs a device)."""
    import inspect

    src = inspect.getsource(gdn_tp.TPGatedDeltaNet.forward_prefill)
    plan = src.index("self._prefill_glue_plan(T, valid_len, prefill_masks)")
    guard = src.index("if glue_conv and not snap_tile_gather_ok(snap, T):")
    reset = src.index("glue_conv, keep = False, False", guard)
    proj = src.index("self._project_qkvzab(x, T")
    assert plan < guard < reset < proj
    assert "_conv_fn = self._kda_conv_prefill_tile if glue_conv else self._kda_conv_prefill" in src
    assert '**({} if snap is None else {"snap": snap})' in src


def test_snap_tile_flag_default_off(monkeypatch):
    from models.demos.blackhole.qwen36.tt import tp_common

    monkeypatch.setattr(tp_common, "_PREFILL_SNAP_TILE", None)
    monkeypatch.delenv("QWEN36_PREFILL_SNAP_TILE", raising=False)
    assert tp_common.prefill_snap_tile_enabled() is False
    monkeypatch.setattr(tp_common, "_PREFILL_SNAP_TILE", None)
    monkeypatch.setenv("QWEN36_PREFILL_SNAP_TILE", "1")
    assert tp_common.prefill_snap_tile_enabled() is True
    monkeypatch.setattr(tp_common, "_PREFILL_SNAP_TILE", None)


def test_model_stages_tile_rows(monkeypatch):
    """Qwen36Model._snap_stage writes the conv_idx_tile words with the other two index tensors (SNAP_TILE on) and
    skips everything when the staged rows do not change."""
    from models.demos.blackhole.qwen36.tt import model as model_mod

    writes = []
    fake = types.SimpleNamespace(
        ReplicateTensorToMesh=lambda mesh: None,
        uint32="u32",
        ROW_MAJOR_LAYOUT="rm",
        from_torch=lambda t, **kw: t.clone(),
        copy_host_to_device_tensor=lambda h, dev: writes.append((dev, h.reshape(-1).tolist())),
    )
    monkeypatch.setattr(model_mod, "ttnn", fake)
    m = types.SimpleNamespace(
        _gdn_snap={"rec_idx": "rec", "conv_idx": "conv", "conv_idx_tile": "tile"}, _snap_staged=(), mesh_device=None
    )
    refs = model_mod.Qwen36Model._snap_stage(m, [256, 1024])
    assert len(refs) == 3 and [d for d, _ in writes] == ["rec", "conv", "tile"]
    assert writes[2][1] == pp.snap_tile_index_words([256, 1024], GDN_SNAP_SLOTS)
    assert writes[1][1] == pp.snap_index_words([256, 1024], GDN_SNAP_SLOTS)[1]
    assert model_mod.Qwen36Model._snap_stage(m, [256, 1024]) == []  # unchanged rows: no copies
    del m._gdn_snap["conv_idx_tile"]  # SNAP_TILE off: today's two writes
    writes.clear()
    model_mod.Qwen36Model._snap_stage(m, [])
    assert [d for d, _ in writes] == ["rec", "conv"]
