# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host test of the lazy key hand-off (DSV41_PF_LAZY_EXPORT, tt/dsv41_model.py ``flush_key_exports``): the export of a prompt's index keys into the decode slabs is recorded by a chunk step and
runs once, before the keys are read (decode / draft seeding), before the carried state of the slot is overwritten (slot eviction) and not at all for a superseded / released prompt.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_lazy_key_export.py
"""

import ast
import os

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model():
    pytest.importorskip("ttnn")
    from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import Model

    m = Model.__new__(Model)
    m.calls = []
    m.log = lambda *a, **k: None
    m._export_keys_users = lambda users, ends, s_end, slot_of: m.calls.append(
        (sorted(users), {b: ends[b] for b in users}, s_end, {b: slot_of(b) for b in users})
    )
    return m


def test_flush_groups_by_window_end_and_clears():
    m = _model()
    m._kx_pending = {1: (1024, 1000, 0), 2: (1024, 990, 1), 3: (2048, 2000, 0)}
    m.flush_key_exports(why="t")
    assert m.calls == [([1, 2], {1: 1000, 2: 990}, 1024, {1: 0, 2: 1}), ([3], {3: 2000}, 2048, {3: 0})]
    assert not m._kx_pending
    m.calls.clear()
    assert m.flush_key_exports() == 0.0 and not m.calls  # nothing pending: free
    assert m.kx_stats["users"] == 3 and m.kx_stats["flushes"] == 1


def test_flush_selected_users_only():
    m = _model()
    m._kx_pending = {1: (512, 500, 0), 2: (512, 500, 1)}
    m.flush_key_exports([2, 7])
    assert m.calls == [([2], {2: 500}, 512, {2: 1})] and list(m._kx_pending) == [1]


def test_drop_and_release_forget_the_record():
    m = _model()
    m._kx_pending = {1: (512, 500, 0), 2: (512, 500, 1), 3: (512, 500, 1)}
    m.drop_key_exports([1])
    m.flush_key_exports()
    assert [c[0] for c in m.calls] == [[2, 3]]
    m._kx_pending = {4: (512, 500, 0)}
    m.drop_key_exports()
    assert not m._kx_pending


def test_slot_eviction_exports_the_old_owner_first():
    m = _model()
    m.Up, m.U, m.rows = 1, 8, 1
    m._kx_pending = {5: (4096, 4000, 0)}
    m.pf_slot_owner, m.pf_slot_of = [5], {5: 0}
    resume = {}  # owner 5 is finished (not resumable): a new prompt of user 6 takes its slot
    out = m._slot_alloc([(6, None, 0, 100)], resume)
    assert out == {6: 0}
    assert m.calls == [
        ([5], {5: 4000}, 4096, {5: 0})
    ], "the old owner's keys must be exported before its FIFO slot is reused"
    assert 5 not in m.pf_slot_of


def _fn(path, cls, name):
    tree = ast.parse(open(os.path.join(PKG, "tt", path)).read())
    for c in tree.body:
        if isinstance(c, ast.ClassDef) and c.name == cls:
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return ast.unparse(f)
    raise AssertionError(name)


@pytest.mark.parametrize(
    "path,cls,name",
    [
        ("dsv41_model.py", "Model", "decode_forward"),
        ("dsv41_model.py", "Model", "decode_forward_bucket"),
        ("dsv41_model.py", "Model", "prefill_forward_dyn"),
        ("generator_vllm.py", "DeepseekV41ForCausalLM", "decode_forward"),
        ("generator_vllm.py", "DeepseekV41ForCausalLM", "propose_draft_tokens"),
    ],
)
def test_every_reader_of_the_slab_flushes_first(path, cls, name):
    src = _fn(path, cls, name)
    assert (
        "flush_key_exports" in src or "_flush_keys" in src
    ), f"{cls}.{name} reads / overwrites the key state: it must flush the recorded hand-offs"


def test_prefill_records_instead_of_exporting():
    src = _fn("dsv41_model.py", "Model", "_prefill_interleaved_one")
    assert "lazy_kx" in src and "_kx_pending" in src


def test_window_sync_rules():
    pytest.importorskip("ttnn")
    from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import window_needs_sync as w

    C = 512
    assert w([(0, 0, 4096)], C, False)  # asynchronous windows off: always wait
    assert not w([(0, 512, 4096), (1, 0, 8192)], C, True)  # nobody ends here, no snapshot policy
    assert w([(0, 3584, 4096)], C, True)  # the window of a user's last token (end 4096 = 3584 + 512)
    assert w([(0, 3700, 4000)], C, True)  # padded last window
    wants = lambda B: B in (1024, 2048)  # noqa: E731
    assert w([(0, 512, 4096)], C, True, wants)  # window [512, 1024) ends on a wanted boundary
    assert not w([(0, 1024, 4096)], C, True, wants)  # [1024, 1536) -> 1536: not wanted
    assert not w([(0, 512, 4096)], C, True, wants, vis={0: object()})  # image prompts are never snapshotted
