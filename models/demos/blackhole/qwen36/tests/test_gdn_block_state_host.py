# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the bookkeeping of gdn_block_state.GdnBlockState with the ttnn calls patched out.

Checks which block every GDN layer reads for each row (mamba_state groups -> layers -> page index), that a row move
only rewrites the page index, that the index is copied only when its blocks change, which block a prefill row writes,
and how zero_blocks batches vLLM's block ids.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
from models.demos.blackhole.qwen36.tt.kv_cache_spec import GdnDims, plan_gdn_block_layout

QWEN_27B_TP8 = GdnDims(nk=2, nv=6, dk=128, dv=128, kconv=4)
ATTENTION = [i % 4 == 3 for i in range(16)]  # 4 attention layers, 12 GDN layers in 3 groups
NUM_BLOCKS = 50
BMAX = 4


@pytest.fixture
def ttnn_calls(monkeypatch):
    """Record the host-to-device copies and cache fills GdnBlockState issues."""
    calls = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: SimpleNamespace(host=t.clone()))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: calls.append(("copy", host.host, dev)))
    monkeypatch.setattr(
        ttnn.experimental,
        "paged_fill_cache",
        lambda cache, src, table, batch_idx=0: calls.append(("fill", cache, src, table)),
        raising=False,
    )
    return calls


def _state(block_size=512):
    """A GdnBlockState over 12 stand-in GDN layers, without a device."""
    layout = plan_gdn_block_layout(block_size=block_size, kv_heads=1, head_dim=256, kv_bf8=False, dims=QWEN_27B_TP8)
    st = object.__new__(gbs.GdnBlockState)
    st.mesh = None
    st.bmax = BMAX
    st.num_blocks = NUM_BLOCKS
    st.dims = QWEN_27B_TP8
    st.layout = layout
    st.layers = {
        i: SimpleNamespace(name=i, state_blocks=SimpleNamespace(idx=f"idx{i}"))
        for i, a in enumerate(ATTENTION)
        if not a
    }
    st.kv_caches = [(f"k{j}", f"v{j}") for j in range(4)]
    st._group_names = None
    st._group_layers = None
    st._decode_blocks = {}
    st._index_refs = {}
    st._zero = "zero"
    st._zero_table = SimpleNamespace(shape=[1, 2])
    st._zero_refs = []
    st._write_refs = []
    return st


def _plugin():
    return PluginLayout(ATTENTION, 1, 256, QWEN_27B_TP8, False, num_blocks=NUM_BLOCKS)


def _copies(calls):
    return {dev: host for kind, host, dev in calls if kind == "copy"}


def test_decode_index_names_each_rows_blocks(ttnn_calls):
    st, plugin = _state(), _plugin()
    rows = [(10, 11, 12), (20, 21, 22), None, (30, 31, 32)]  # row 2 is a padding row
    st.set_decode_blocks(plugin.step_state(rows, BMAX))
    copies = _copies(ttnn_calls)
    assert len(ttnn_calls) == 12, "one copy per GDN layer"
    for g, layers in enumerate(plugin.gdn_groups):
        blocks = [r[g] if r is not None else 0 for r in rows]
        want = st.layout.index_host(blocks, NUM_BLOCKS)
        for i in layers:
            assert torch.equal(copies[f"idx{i}"], want), f"layer {i} of group {g}"


def test_row_move_rewrites_only_the_index(ttnn_calls):
    """A batch condense (request D moves from row 3 to row 1) changes which blocks each row names; nothing else."""
    st, plugin = _state(), _plugin()
    before = [(10, 11, 12), (20, 21, 22), (40, 41, 42), (30, 31, 32)]
    st.set_decode_blocks(plugin.step_state(before, BMAX))
    first = _copies(ttnn_calls)
    ttnn_calls.clear()
    after = [before[0], before[3], before[2], None]
    st.set_decode_blocks(plugin.step_state(after, BMAX))
    second = _copies(ttnn_calls)
    assert {kind for kind, *_ in ttnn_calls} == {"copy"}
    for i in plugin.gdn_groups[0]:
        old, new = first[f"idx{i}"], second[f"idx{i}"]
        assert torch.equal(new[:, 0], old[:, 0]) and torch.equal(new[:, 1], old[:, 3])
        assert torch.equal(new[:, 2], old[:, 2])
        assert torch.equal(new[:, 3], st.layout.index_host([0], NUM_BLOCKS)[:, 0])


def test_index_is_copied_only_when_its_blocks_change(ttnn_calls):
    st, plugin = _state(), _plugin()
    rows = [(10, 11, 12), (20, 21, 22), None, None]
    st.set_decode_blocks(plugin.step_state(rows, BMAX))
    assert len(ttnn_calls) == 12
    ttnn_calls.clear()
    st.set_decode_blocks(plugin.step_state(rows, BMAX))
    assert ttnn_calls == []
    # Only group 1's blocks change: only its 4 layers get a new index.
    st.set_decode_blocks(plugin.step_state([(10, 11, 12), (20, 25, 22), None, None], BMAX))
    assert sorted(dev for _, _, dev in ttnn_calls) == sorted(f"idx{i}" for i in plugin.gdn_groups[1])


def test_mamba_state_must_name_the_models_gdn_layers(ttnn_calls, expect_error):
    st, plugin = _state(), _plugin()
    state = plugin.step_state([(1, 2, 3)], BMAX)
    bad = SimpleNamespace(groups=(SimpleNamespace(layer_names=("model.layers.3.linear_attn",), windows=None),))
    with expect_error(ValueError, "not a GDN layer of this model"):
        st.set_decode_blocks(bad)
    twice = SimpleNamespace(groups=(state.groups[0], state.groups[0], state.groups[2]))
    with expect_error(ValueError, "in more than one group"):
        st.set_decode_blocks(twice)
    with expect_error(ValueError, "has no group for GDN layers"):
        st.set_decode_blocks(SimpleNamespace(groups=state.groups[:2]))


def test_prefill_writes_each_rows_blocks(monkeypatch, expect_error):
    st, plugin = _state(), _plugin()
    writes, target = [], []
    monkeypatch.setattr(st, "_set_write_tables", target.append)
    monkeypatch.setattr(st, "_write_state", lambda dn: writes.append((dn.name, target[-1])))
    state = plugin.step_state([(10, 11, 12), (20, 21, 22)], BMAX)
    st.write_prefill_state(state, 1)
    want = [(i, 20 + g) for g, layers in enumerate(plugin.gdn_groups) for i in layers]
    assert writes == want
    with expect_error(ValueError, "has the null block 0"):
        st.write_prefill_state(state, 2)


def test_prefill_checkpoint_writes_each_layers_own_state(monkeypatch, expect_error):
    """write_checkpoint writes the state every GDN layer computed at the prefill checkpoint (not its workspace) into
    its group's checkpoint block, and frees it."""
    st, plugin = _state(), _plugin()
    writes, target, freed = [], [], []
    monkeypatch.setattr(st, "_set_write_tables", target.append)
    monkeypatch.setattr(
        st, "_write_state", lambda dn, rec=None, carry=None: writes.append((dn.name, rec, carry, target[-1]))
    )
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    for i, dn in st.layers.items():
        dn.take_prefill_checkpoint = lambda i=i: (f"rec{i}", f"conv{i}")
    state = plugin.step_state([(10, 11, 12)], BMAX)
    st.write_checkpoint(state, (30, 31, 32))
    want = [(i, f"rec{i}", f"conv{i}", 30 + g) for g, layers in enumerate(plugin.gdn_groups) for i in layers]
    assert writes == want
    assert freed == [t for i, rec, conv, _ in want for t in (rec, conv)]
    with expect_error(ValueError, "has the null block 0"):
        st.write_checkpoint(state, (30, 0, 32))


def test_zero_blocks_in_padded_groups(ttnn_calls):
    st = _state()
    st.zero_blocks([])
    assert ttnn_calls == []
    st.zero_blocks([5, 6, 7])
    tables = [c[1] for c in ttnn_calls if c[0] == "copy"]
    assert [t.tolist() for t in tables] == [[[5, 6]], [[7, 0]]]
    fills = [c for c in ttnn_calls if c[0] == "fill"]
    assert len(fills) == 2 * 2 * len(st.kv_caches), "every K and V buffer, per group of blocks"
    assert {c[1] for c in fills} == {name for kv in st.kv_caches for name in kv}
