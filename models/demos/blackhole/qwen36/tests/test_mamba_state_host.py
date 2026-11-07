# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the vLLM classes turn the plugin's ``mamba_state`` keyword (QWEN36_MAMBA_SPEC=1) into the per-group
inputs of the GDN block state, exactly.

``mamba_state`` holds one group per MambaSpec KV cache group: ``layer_names``, ``windows`` [R, 1 + K],
``checkpoint_blocks`` [R, W] or None and ``chain`` (``copies``, ``start_blocks``, ``windows``) or None; beside them
``accepted_counts`` [R] and ``checkpoint_positions`` [R, W]. The tests build one step of 3 GDN groups of Qwen3.8's 64
layers, with the groups in another order than SpecGdnBlockState binds them (layer slot rank) and the names of each
group shuffled, and check what Qwen36MTPContractForCausalLM.prefill_forward / decode_forward pass on (prefill blocks,
checkpoint points, gdn_windows with counts and chain tables, the staged chain copies) against the same values built
directly in SpecGdnBlockState's group order.

    PYTHONPATH=<worktree>:<plugin>/src:<tools>/ttnn-host-stub pytest --noconftest \
        models/demos/blackhole/qwen36/tests/test_mamba_state_host.py
"""

import random
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.config")
pytest.importorskip("vllm_tt_plugin.spec_decode")

from models.demos.blackhole.qwen36.tt import qwen36_vllm as qv  # noqa: E402
from models.demos.blackhole.qwen36.tt import spec_block_state as sbs  # noqa: E402
from models.demos.blackhole.qwen36.tt.kv_cache_spec import gdn_layer_name  # noqa: E402
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine  # noqa: E402

K, T = 3, 4
G = 3
R = 3  # rows the plugin sends
B = 4  # rows of the step (the runner's padded batch)
BLOCK = 1024
ORDER = (2, 0, 1)  # mamba_state group k holds the layers of SpecGdnBlockState group ORDER[k]


def _i32(x):
    return torch.as_tensor(x, dtype=torch.int32)


def _per_layer_specs():
    """(shapes, dtypes, tensor_idx) of Qwen3.8's 64 layers: attention every 4th, slot = rank within its type."""
    specs, n_attn, n_gdn = [], 0, 0
    for i in range(64):
        if i % 4 == 3:
            specs.append(([1, 1, 2048, 256], "bf8", n_attn))
            n_attn += 1
        else:
            specs.append((((3, 2560), (12, 128, 128)), ("bf16", "fp32"), n_gdn % 16))
            n_gdn += 1
    return specs


LAYER_GROUPS = sbs.gdn_groups_by_slot(_per_layer_specs(), lambda e: isinstance(e[0], tuple))


def _names(g, seed=0):
    names = [gdn_layer_name(i) for i, h in sorted(LAYER_GROUPS.items()) if h == g]
    random.Random(seed + g).shuffle(names)
    return tuple(names)


def _mamba_state(windows, counts=None, ckpt=None, positions=None, chain=None, chain_field=True):
    """The plugin's mamba_state from per-group values in SpecGdnBlockState's group order, packed in ORDER.
    ``chain_field=False``: the groups have no ``chain`` attribute, as vllm-tt-plugin main's MambaGroupState."""

    def group(g):
        fields = dict(layer_names=_names(g), windows=windows[g], checkpoint_blocks=None if ckpt is None else ckpt[g])
        if chain is not None:
            fields["chain"] = SimpleNamespace(copies=chain[0][g], start_blocks=chain[1][g], windows=chain[2][g])
        elif chain_field:
            fields["chain"] = None
        return SimpleNamespace(**fields)

    groups = tuple(group(g) for g in ORDER)
    return SimpleNamespace(groups=groups, accepted_counts=counts, checkpoint_positions=positions)


def _layout():
    nv, C = 48 // 8, (2 * 16 * 128 + 48 * 128) // 8
    S = sbs.kv_block_bytes_per_bank([2000, 1, BLOCK, 256], "bfloat8_b", 8)
    return sbs.SpecBlockLayout(S, 8, 2000, nv, sbs.conv_carry_pages(4, C))


def _spec_state():
    """SpecGdnBlockState's host part over a SpecBlockBook (the device uploads do nothing)."""
    st = sbs.SpecGdnBlockState.__new__(sbs.SpecGdnBlockState)
    st.layer_groups, st.G, st.T, st.bmax = LAYER_GROUPS, G, T, B
    st.book = sbs.SpecBlockBook(_layout(), G, B, T, "align", BLOCK)
    st._upload_idx = lambda groups: None
    st._upload_tables = lambda: None
    st.moves_counts, st.chain_copies, st._tables_moved = None, [], False
    return st


class _EngineStub:
    """What MTPSpecEngine._stage_gdn_windows / _paged_plan read; runner slot s is engine row PHYS[s]."""

    PHYS = (2, 0, 3, 1)
    _to_rows = MTPSpecEngine._to_rows
    _set_slot_map = MTPSpecEngine._set_slot_map
    _stage_gdn_windows = MTPSpecEngine._stage_gdn_windows
    _paged_plan = MTPSpecEngine._paged_plan

    def __init__(self, st):
        self.B, self.T, self.block_size, self.st = B, T, BLOCK, st
        self.model = SimpleNamespace(_spec_blocks=st)
        self._paged_step_rows = None
        self._set_slot_map(self.PHYS)

    def _paged_state(self):
        return self.st


def _pad(x):
    out = torch.zeros((B,) + tuple(x.shape[1:]), dtype=torch.int64)
    out[: x.shape[0]] = x
    return out


def test_layer_names_map_each_group_and_wrong_groupings_raise():
    st = _spec_state()
    w = [_i32([[10 * g + r] for r in range(R)]) for g in range(G)]
    groups = qv._spec_mamba_groups(_mamba_state(w), st)
    assert [int(grp.windows[0, 0]) for grp in groups] == [0, 10, 20]

    def broken(edit):
        ms = _mamba_state(w)
        names = [list(grp.layer_names) for grp in ms.groups]
        edit(names)
        ms.groups = tuple(SimpleNamespace(**{**vars(grp), "layer_names": tuple(n)}) for grp, n in zip(ms.groups, names))
        with pytest.raises(ValueError) as e:  # allow-pytest.raises: --noconftest host test
            qv._spec_mamba_groups(ms, st)
        return str(e.value)

    first = _names(ORDER[0])[0]
    assert first in broken(lambda n: n[0].remove(first))  # in no group
    assert first in broken(lambda n: n[1].append(first))  # in two groups
    assert "model.layers.3.self_attn" in broken(lambda n: n[2].append("model.layers.3.self_attn"))  # no GDN layer
    # every layer in one group, but not the layer-slot grouping SpecGdnBlockState bound
    swapped = broken(lambda n: (n[0].__setitem__(0, n[1][0]), n[1].__setitem__(0, first)))
    assert "differ from the groups bound by layer slot" in swapped


def test_prefill_with_checkpoints_passes_each_groups_blocks_and_checkpoint_points():
    windows = [_i32([[100 * g + 10 * r + t for t in range(T)] for r in range(R)]) for g in range(G)]
    ckpt = [_i32([[500 + 100 * g + 1, 500 + 100 * g + 2], [500 + 100 * g + 3, 0], [0, 0]]) for g in range(G)]
    positions = _i32([[2048, 3072], [1024, 0], [0, 0]])
    st = _spec_state()
    prefills = []
    eng = SimpleNamespace(prefill=lambda slot, toks, table, **kw: prefills.append((slot, kw)) or torch.zeros(8))
    me = SimpleNamespace(
        model=[SimpleNamespace(_spec_blocks=st)],
        _ensure_ready=lambda kv=None: eng,
        _table=lambda pt: pt,
        _has_visual=qv.Qwen36ForCausalLM._has_visual,
    )
    L = 5000
    qv.Qwen36MTPContractForCausalLM.prefill_forward(
        me,
        torch.zeros(R, L, dtype=torch.int64),
        torch.zeros(R, 8, dtype=torch.int32),
        None,
        [L] * R,
        empty_slots=[0, 1, 2],
        page_tables_per_layer=["the per-layer tables"],
        mamba_state=_mamba_state(windows, ckpt=ckpt, positions=positions),
    )
    assert prefills == [(u, {"gdn_req": u}) for u in range(R)]
    for g in range(G):
        assert torch.equal(st.prefill_blocks[g], windows[g][:, 0].to(torch.int64))
    want = {
        0: [(2048, [501, 601, 701]), (3072, [502, 602, 702])],
        1: [(1024, [503, 603, 703])],
    }
    assert {u: st.prefill_checkpoints(u) for u in range(R) if st.prefill_checkpoints(u)} == want
    written = []
    st.write_checkpoint = written.append
    es = _EngineStub(st)
    for u in range(R):
        plan = es._paged_plan(0, L, None, True, u)
        if u not in want:
            assert plan is None
            continue
        s, src, points = plan
        assert (s, src, list(points)) == (0, None, [n for n, _ in want[u]])
        for n in points:
            points.save(n)
    assert written == [blocks for u in sorted(want) for _, blocks in want[u]]


def _decode(st, mamba_state, chained):
    """Qwen36MTPContractForCausalLM.decode_forward on B rows; returns the gdn_windows it passes to the engine."""
    calls = []

    def decode_forward(tokens, start_pos, **kw):
        calls.append(kw)
        if kw["spec_mode"] == "fused_sample":
            return SimpleNamespace(spec_mode="fused_sample", handle="pending step", logits=None)
        return SimpleNamespace(spec_mode="argmax_ids", argmax_ids=torch.zeros(B, T, dtype=torch.int32), logits=None)

    me = SimpleNamespace(
        model=[SimpleNamespace(_spec_blocks=st)],
        model_capabilities={"supports_async_spec_decode": True},
        _ensure_ready=lambda kv=None: SimpleNamespace(decode_forward=decode_forward),
        _table=lambda pt: pt,
        _SPEC_K=K,
        _check_spec_sampling=lambda ss, mode, b: ss,
        _gdn_windows=qv.Qwen36MTPContractForCausalLM._gdn_windows,
    )
    kw = dict(
        tokens=torch.zeros(B, T, dtype=torch.int64),
        start_pos=_i32([[100 + t for t in range(T)]] * B),
        page_table=torch.zeros(B, 8, dtype=torch.int32),
        num_valid_drafts=_i32([K] * B),
        page_tables_per_layer=["the per-layer tables"],
        mamba_state=mamba_state,
    )
    if chained:
        kw.update(
            accepted_counts=_i32([-1, 1, -1, 1]),
            spec_mode="fused_sample",
            defer_read=True,
            chain_rows=torch.tensor([True, False, True, False]),
            chain_tip=7,
        )
    else:
        kw.update(accepted_counts=_i32([1] * B), spec_mode="argmax_ids")
    qv.Qwen36MTPContractForCausalLM.decode_forward(me, **kw)
    (call,) = calls
    return call["gdn_windows"]


DEC = [_i32([[1000 + 100 * g + 10 * r + t for t in range(T)] for r in range(R)]) for g in range(G)]
START0 = torch.tensor([2047, 500, 700, 0])
ACTIVE = torch.tensor([True, True, True, False])


@pytest.mark.parametrize("chain_field", [True, False])
def test_decode_passes_each_groups_window_and_the_counts(chain_field):
    st = _spec_state()
    counts = _i32([1, 3, 4])
    blocks, c, moves = _decode(st, _mamba_state(DEC, counts=counts, chain_field=chain_field), chained=False)
    assert len(blocks) == G and moves is None
    for g in range(G):
        assert torch.equal(blocks[g], _pad(DEC[g]))
    assert torch.equal(c, torch.tensor([1, 3, 4, 1]))


def test_chained_decode_passes_the_chain_tables_and_runs_the_chain_choices():
    st = _spec_state()
    es = _EngineStub(st)
    es._stage_gdn_windows(_decode(st, _mamba_state(DEC, counts=_i32([1, 3, 4])), False), *es._to_rows(START0, ACTIVE))
    wins = [_i32([[1300 + 100 * g + 10 * r + t for t in range(T)] for r in range(R)]) for g in range(G)]
    copies = [torch.zeros(R, T, 2, dtype=torch.int32) for _ in range(G)]
    for g in range(G):
        copies[g][0, 1:] = _i32([int(DEC[g][0, 1]), 1700 + g])  # row 0: counts 2 .. 4 copy window column 1 on
        copies[g][2, 3] = _i32([int(DEC[g][2, 3]), 1750 + g])  # row 2: count 4 copies column 3 on
    starts = [_i32([[1400 + 100 * g + 10 * r + c for c in range(T)] for r in range(R)]) for g in range(G)]
    chain_wins = [
        _i32([[[1500 + 100 * g + 10 * r + c + t for t in range(T)] for c in range(T)] for r in range(R)])
        for g in range(G)
    ]
    ms = _mamba_state(wins, counts=_i32([-1, 2, -1]), chain=(copies, starts, chain_wins))
    gdn_windows = _decode(st, ms, chained=True)
    blocks, c, (got_starts, got_wins, got_copies) = gdn_windows
    assert torch.equal(c, torch.tensor([-1, 2, -1, 1]))
    for g in range(G):
        assert torch.equal(blocks[g], _pad(wins[g]))
        assert torch.equal(got_starts[g], _pad(starts[g]))
        assert torch.equal(got_wins[g], _pad(chain_wins[g]))
        assert torch.equal(got_copies[g], _pad(copies[g]))
    es._stage_gdn_windows(gdn_windows, *es._to_rows(START0, ACTIVE))
    row = _EngineStub.PHYS  # runner slot -> engine row
    assert es._paged_unknown == [row[0], row[2]] and sorted(st.book.moves) == sorted([row[0], row[2]])
    # each distinct (src, dst) of a chained row: the counts that select it and src's column in the pending window
    want = []
    for g in range(G):
        want += [
            (row[0], g, int(DEC[g][0, 1]), 1700 + g, 1, (2, 3, 4)),
            (row[2], g, int(DEC[g][2, 3]), 1750 + g, 3, (4,)),
        ]
    assert sorted(st.chain_copies) == sorted(want)


def test_chain_tables_without_a_copy_pass_no_copies():
    """Every chain copy (0, 0) (prefix_cache_retention_interval 0): the engine gets no copies to run."""
    st = _spec_state()
    zero = [torch.zeros(R, T, 2, dtype=torch.int32) for _ in range(G)]
    starts = [torch.ones(R, T, dtype=torch.int32) for _ in range(G)]
    wins = [torch.ones(R, T, T, dtype=torch.int32) for _ in range(G)]
    _, _, moves = qv.Qwen36MTPContractForCausalLM._gdn_windows(
        _mamba_state(DEC, counts=_i32([-1, 1, 1]), chain=(zero, starts, wins)), st, B
    )
    assert moves[2] is None and [tuple(s.shape) for s in moves[0]] == [(B, T)] * G


def test_non_spec_class_prefill_gives_each_gdn_layer_its_groups_block():
    """Qwen36ForCausalLM (GdnBlockState): prefill_forward passes (window column 0 of each mamba_state group,
    {GDN layer: its group}); every GDN layer then reads the block of the group whose layer_names hold it."""
    recorded = {}
    me = SimpleNamespace(
        model=[SimpleNamespace(num_devices=4, args=SimpleNamespace(max_batch_size=4))],
        _gdn_blocks=SimpleNamespace(layers=[(i, None) for i in sorted(LAYER_GROUPS)]),
        _prefill_forward_tp_batched=lambda *a, **k: recorded.update(k),
    )
    ids = [_i32([[200 + 100 * g + r] for r in range(R)]) for g in range(G)]
    qv.Qwen36ForCausalLM.prefill_forward(
        me,
        torch.zeros(R, 64, dtype=torch.int64),
        torch.zeros(R, 8, dtype=torch.int32),
        None,
        [64] * R,
        empty_slots=[0, 1, 2],
        mamba_state=_mamba_state(ids),
    )
    blocks, group_of = recorded["state_block_ids"]
    assert sorted(group_of) == sorted(LAYER_GROUPS)
    for i, g in LAYER_GROUPS.items():
        assert blocks[group_of[i]].tolist() == ids[g][:, 0].tolist(), i
