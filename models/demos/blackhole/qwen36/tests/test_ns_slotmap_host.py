# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the non-spec host slot map (QWEN36_NS_SLOTMAP) of ``Qwen36ForCausalLM``.

A plugin slot_remap permutes a runner-slot -> device-row map on the host instead of moving GDN state on the device.
No device: Generator.decode_forward is a recorder / tiny fake device, the model is a namespace fake.

Run: pytest models/demos/blackhole/qwen36/tests/test_ns_slotmap_host.py -q
"""

import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM
from models.tt_transformers.tt.generator import Generator

W = 8


@dataclass(frozen=True)
class _SP:
    temperature: object
    top_k: object
    seed: object = 0  # scalar: left alone


class _Rec:
    def __init__(self):
        self.calls = []
        self.gdn_remaps = []
        self.rope_remaps = []
        self.prefills = []  # device rows handed to prefill_paged_slots, per call
        self.fail_next = False


@pytest.fixture
def ns(monkeypatch):
    monkeypatch.delenv("QWEN36_SERVE_DEVICE_DECODE", raising=False)
    monkeypatch.delenv("TT_DECODE_BUCKETING", raising=False)
    monkeypatch.setenv("QWEN36_NS_SLOTMAP", "1")
    rec = _Rec()

    def fake_decode_forward(self, *args, **kwargs):
        if rec.fail_next:
            rec.fail_next = False
            raise RuntimeError("device error")
        rec.calls.append(kwargs)
        return "out"

    monkeypatch.setattr(Generator, "decode_forward", fake_decode_forward)
    monkeypatch.setattr(Generator, "process_decode_output_host", lambda self, tt_out, **kw: tt_out)
    monkeypatch.setattr(Qwen36ForCausalLM, "_has_visual", staticmethod(lambda kwargs, key, u=0: False))
    model = SimpleNamespace(
        num_devices=4,
        args=SimpleNamespace(max_batch_size=W),
        _serve_device_decode=True,
        _remap_gdn_slots=lambda r, live=None: rec.gdn_remaps.append(list(r)),
        remap_slot_rope_delta=lambda r: rec.rope_remaps.append(list(r)),
        sampling=None,
    )

    def prefill_paged_slots(token_ids_list, pt, slots, **kw):
        rec.prefills.append(list(slots))
        return [torch.zeros(4) for _ in token_ids_list]

    model.prefill_paged_slots = prefill_paged_slots
    gen = Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)
    gen.model = [model]
    return gen, rec


def _decode(gen, pos, *, remap=None, reload=True, width=W, tokens=None, sp=None, pt=None, **extra):
    """pos: {runner row: position}; token value of runner row i is 100 + i, page-table row i is [i]*4."""
    start_pos = torch.full((width,), -1, dtype=torch.int32)
    for r, p in pos.items():
        start_pos[r] = p
    kw = dict(
        tokens=tokens if tokens is not None else (100 + torch.arange(width, dtype=torch.int32)).reshape(width, 1),
        start_pos=start_pos,
        page_table=pt if pt is not None else torch.arange(width, dtype=torch.int32).reshape(width, 1).repeat(1, 4),
        kv_cache=None,
        enable_trace=True,
        read_from_device=False,
        reload_inputs=reload,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        sampling_params=sp if sp is not None else _SP(list(range(width)), torch.arange(width)),
    )
    if remap is not None:
        kw["slot_remap"] = remap
    kw.update(extra)
    return gen.decode_forward(**kw)


def _prefill(gen, slots):
    n = len(slots)
    return gen._prefill_forward_tp_batched(
        gen.model[0], torch.ones(n, 4, dtype=torch.int32), torch.zeros(n, 4, dtype=torch.int32), [4] * n, slots
    )


ROT = [1, 2, 3, 4, 5, 6, 7, 0]  # runner row i takes the request of old runner row i+1


def test_remap_step_moves_no_state_and_forwards_none(ns):
    gen, rec = ns
    _decode(gen, {0: 5, 1: 5, 2: 5}, remap=ROT)
    assert rec.gdn_remaps == [] and rec.rope_remaps == []
    assert rec.calls[-1]["slot_remap"] is None
    assert gen._ns_phys == slot_after(list(range(W)), ROT)


def slot_after(phys, remap):
    return [phys[j] for j in remap]


def test_inputs_scattered_and_outputs_gathered(ns):
    gen, rec = ns
    remap = [2, 0, 1, 3, 4, 5, 6, 7]
    phys = slot_after(list(range(W)), remap)  # runner row i -> device row phys[i]
    pos = {0: 7, 1: 8, 2: 9}
    _decode(
        gen,
        pos,
        remap=remap,
        prompt_tokens=torch.arange(W * 3).reshape(W, 3),
        output_tokens=torch.arange(W * 2).reshape(W, 2),
        grammar_bitmask=torch.arange(W * 2).reshape(W, 2),
    )
    c = rec.calls[-1]
    live = sorted(phys[i] for i in pos)
    bucket = c["tokens"].shape[0]
    assert bucket == 4 and live == [0, 1, 2]
    for i in range(W):
        d = phys[i]
        if d >= bucket:
            continue
        assert int(c["tokens"][d, 0]) == 100 + i
        assert int(c["start_pos"][d]) == pos.get(i, -1)
        assert c["page_table"][d].tolist() == [i] * 4
        assert c["sampling_params"].temperature[d] == i
        assert int(c["sampling_params"].top_k[d]) == i
        for name, n in (("prompt_tokens", 3), ("output_tokens", 2), ("grammar_bitmask", 2)):
            assert c[name][d].tolist() == list(range(i * n, i * n + n))
    assert c["sampling_params"].seed == 0
    assert isinstance(c["sampling_params"].temperature, list) and c["tokens"].dtype == torch.int32
    # the full-width tensors given to the plugin were not modified
    # outputs: device order -> runner order
    toks = torch.arange(W, dtype=torch.int64) + 500  # device row d produced 500 + d
    got = gen.process_decode_output_host(toks, is_tokens=True)
    assert got.dtype == toks.dtype and got.tolist() == [500 + phys[i] for i in range(W)]
    short = toks[:4]  # bucket-length output: rows beyond it are neutral
    got = gen.process_decode_output_host(short, is_tokens=True)
    assert got.tolist() == [500 + phys[i] if phys[i] < 4 else 0 for i in range(W)]
    logits = torch.arange(W, dtype=torch.float32).reshape(W, 1, 1).expand(W, 1, 5).clone()
    got = gen.process_decode_output_host(logits)
    assert got[:, 0, 0].tolist() == [float(phys[i]) for i in range(W)]
    # old-path logprobs [B] and new-path (topk_lp [B, 32], topk_idx [B, 32])
    lp = torch.arange(W, dtype=torch.float32) / 10
    out, got_lp = gen.process_decode_output_host((toks, lp))
    assert out.tolist() == [500 + phys[i] for i in range(W)]
    assert got_lp.tolist() == pytest.approx([phys[i] / 10 for i in range(W)])
    tlp = torch.arange(W, dtype=torch.float32).reshape(W, 1).expand(W, 32).clone()
    tidx = torch.arange(W, dtype=torch.int32).reshape(W, 1).expand(W, 32).clone()
    out, (glp, gidx) = gen.process_decode_output_host((toks, (tlp, tidx)))
    assert glp.shape == (W, 32) and gidx.dtype == torch.int32
    assert glp[:, 0].tolist() == [float(phys[i]) for i in range(W)]
    assert gidx[:, 5].tolist() == [phys[i] for i in range(W)]
    assert gen.process_decode_output_host((toks, None))[1] is None


@pytest.mark.parametrize("live_rows,expect", [([], 1), ([0], 1), ([1], 2), ([0, 2], 4), ([3], 4), ([4], 8), ([7], 8)])
def test_bucket_is_next_pow2_above_highest_live_device_row(ns, live_rows, expect):
    gen, rec = ns
    _decode(gen, {r: 3 for r in live_rows})  # identity map: device row == runner row
    assert rec.calls[-1]["tokens"].shape[0] == expect
    assert rec.calls[-1]["page_table"].shape[0] == expect


def test_bucket_uses_device_rows_not_live_count(ns):
    gen, rec = ns
    remap = [0, 1, 2, 3, 4, 5, 6, 7]
    remap[0], remap[5] = 5, 0  # runner row 0 now holds the request that sat at row 5
    _decode(gen, {5: 3}, remap=remap)  # one live request, in device row 0 after the swap... row 5 -> runner 0
    # live runner row 5 after remap? start_pos is in NEW runner order: row 5 holds old row 0's request -> device row 0
    assert gen._ns_phys[5] == 0
    assert rec.calls[-1]["tokens"].shape[0] == 1


def test_bucket_change_still_needs_reload(ns, expect_error):
    gen, rec = ns
    _decode(gen, {0: 3, 1: 3, 2: 3})
    with expect_error(ValueError, "bucket changed"):
        _decode(gen, {0: 3, 5: 3}, reload=False)
    n = len(rec.calls)
    _decode(gen, {0: 3, 5: 3})
    assert len(rec.calls) == n + 1


@pytest.mark.parametrize("bad", [[0, 0, 1, 2, 3, 4, 5, 6], [1, 2, 3], list(range(9)), [0, 1, 2, 3, 4, 5, 6, 8]])
def test_bad_remap_raises_without_state_change(ns, expect_error, bad):
    gen, rec = ns
    with expect_error(ValueError, "permutation"):
        _decode(gen, {0: 3}, remap=bad)
    assert rec.calls == [] and gen._ns_phys == list(range(W)) and gen._ns_owned == set()


def test_prefill_picks_lowest_free_device_row(ns):
    gen, rec = ns
    _prefill(gen, [3])  # runner slot 3 -> device row 0
    assert rec.prefills[-1] == [0]
    assert gen._ns_owned == {0} and gen._ns_phys[3] == 0 and gen._ns_phys[0] == 3
    _prefill(gen, [5, 0])  # slot 0 holds device row 3, which is free but not the lowest
    assert rec.prefills[-1] == [1, 2]
    assert gen._ns_owned == {0, 1, 2}
    assert sorted(gen._ns_phys) == list(range(W))
    gen.release_request(3)  # runner slot 3 holds device row 0
    assert gen._ns_owned == {1, 2}
    _prefill(gen, [4])
    assert rec.prefills[-1] == [0]


def test_prefill_without_empty_slots_defaults_to_range_n(ns):
    gen, rec = ns
    gen._prefill_forward_tp_batched(
        gen.model[0], torch.ones(2, 4, dtype=torch.int32), torch.zeros(2, 4, dtype=torch.int32), [4, 4], None
    )
    assert rec.prefills[-1] == [0, 1] and gen._ns_owned == {0, 1}


def test_prefill_with_stale_owned_rows_uses_the_slot_row(ns):
    gen, rec = ns
    gen._ns_phys = list(reversed(range(W)))
    gen._ns_owned = set(range(W))  # stale: every row marked owned
    _prefill(gen, [2])
    assert rec.prefills[-1] == [W - 1 - 2] and gen._ns_phys == list(reversed(range(W)))
    assert gen._ns_owned == set(range(W))


def test_prefill_normal_case_picks_lowest_free_row(ns):
    gen, rec = ns
    gen._ns_phys = list(reversed(range(W)))
    gen._ns_owned = {0, 1}
    _prefill(gen, [0])  # slot 0 holds row W-1; the lowest free row is 2
    assert rec.prefills[-1] == [2] and gen._ns_phys[0] == 2 and sorted(gen._ns_phys) == list(range(W))


def test_prefill_two_slots_never_share_a_row(ns):
    gen, rec = ns
    gen._ns_owned = set(range(W))  # stale
    gen._ns_phys = [1, 0] + list(range(2, W))
    _prefill(gen, [0, 1])
    rows = rec.prefills[-1]
    assert len(set(rows)) == 2 and rows == [1, 0]
    gen._ns_owned = set()
    gen._ns_phys = list(range(W))
    _prefill(gen, [3, 5])
    rows = rec.prefills[-1]
    assert rows == [0, 1] and sorted(gen._ns_phys) == list(range(W))


def test_failed_decode_leaves_state_unchanged(ns, expect_error):
    gen, rec = ns
    _decode(gen, {0: 3, 1: 3})
    gen._ns_owned = {0, 1}
    before = (list(gen._ns_phys), set(gen._ns_owned))
    rec.fail_next = True
    with expect_error(RuntimeError, "device error"):
        _decode(gen, {0: 3, 1: 3, 2: 3}, remap=ROT)
    assert (gen._ns_phys, gen._ns_owned) == before


def test_reload_step_takes_ownership_from_start_pos(ns):
    gen, rec = ns
    _decode(gen, {0: 3, 2: 3}, remap=[2, 1, 0, 3, 4, 5, 6, 7])
    assert gen._ns_owned == {0, 2}
    _decode(gen, {0: 3, 2: 4}, reload=False)  # steady step: owned unchanged
    gen.release_request(0)  # runner slot 0 sits in device row 2
    assert gen._ns_owned == {0}
    _decode(gen, {0: 5, 2: 5}, reload=False)  # steady step, same bucket
    assert gen._ns_owned == {0}


def test_identity_map_passes_objects_through(ns):
    gen, rec = ns
    sp = _SP(list(range(W)), torch.arange(W))
    tokens = (100 + torch.arange(W, dtype=torch.int32)).reshape(W, 1)
    _decode(gen, {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3}, tokens=tokens, sp=sp, remap=list(range(W)))
    c = rec.calls[-1]
    assert c["tokens"] is tokens and c["sampling_params"] is sp
    toks = torch.arange(W)
    assert gen.process_decode_output_host(toks) is toks
    lp = (toks, torch.zeros(W))
    assert gen.process_decode_output_host(lp) is lp


def test_flag_off_release_is_a_noop_and_remap_goes_to_the_device(ns, monkeypatch):
    gen, rec = ns
    monkeypatch.setenv("QWEN36_NS_SLOTMAP", "0")
    gen.release_request(2)
    assert not hasattr(gen, "_ns_owned")
    _decode(gen, {0: 3, 1: 3}, remap=ROT)
    assert rec.gdn_remaps == [ROT] and rec.calls[-1]["slot_remap"] == ROT
    assert not hasattr(gen, "_ns_phys")
    toks = torch.arange(W)
    assert gen.process_decode_output_host(toks) is toks


def test_warmup_resets_the_map(ns, monkeypatch):
    gen, rec = ns
    gen._ns_phys = list(reversed(range(W)))
    gen._ns_owned = {1}
    monkeypatch.setattr(Generator, "warmup_model_decode", lambda self, *a, **k: None, raising=False)
    from models.demos.blackhole.qwen36.tt import qwen36_vllm

    monkeypatch.setattr(qwen36_vllm, "warmup_decode_buckets", lambda gen, fn, *a, **k: "ok")
    monkeypatch.setattr(Qwen36ForCausalLM, "_validate_device_sampling_request", lambda self, r: None)
    monkeypatch.setattr(Qwen36ForCausalLM, "_maybe_forbid_post_capture_compile", lambda self, e: None)
    assert gen.warmup_model_decode(enable_trace=False) == "ok"
    assert gen._ns_phys == list(range(W)) and gen._ns_owned == set()


# --------------------------------------------------------------------------------------------- #
# Churn: a fake device keeps per-device-row state written by prefill; every live request must be read from the row its
# prefill wrote, through arbitrary finishes, condense remaps and random permutations.
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
def test_churn_every_request_reads_its_own_device_row(ns, seed):
    gen, rec = ns
    rng = random.Random(seed)
    dev_owner = [-1] * W  # device row -> request id whose prefill wrote it
    dev_count = [0] * W  # device row -> decode steps its state has seen
    ref_count = {}  # request id -> steps (reference, physically independent)
    steps_seen = []

    def fake_decode(self, *args, **kw):
        t, sp, pt, st = kw["tokens"], kw["start_pos"], kw["page_table"], kw["sampling_params"]
        b = t.shape[0]
        out = torch.zeros(b, dtype=torch.int64)
        for r in range(b):
            if int(sp[r]) == -1:
                continue
            rid = int(t[r, 0])
            assert dev_owner[r] == rid, f"device row {r} holds request {dev_owner[r]}, decode fed {rid}"
            assert int(pt[r, 0]) == rid and st.temperature[r] == rid
            dev_count[r] += 1
            out[r] = rid * 1000 + dev_count[r]
        steps_seen.append(b)
        return out

    Generator.decode_forward = fake_decode  # restored by monkeypatch teardown of the fixture
    rows = [None] * W  # runner row -> request id (the plugin's view)
    next_rid = 1
    remaps = 0
    need_reload = True
    for step in range(60):
        changed = need_reload
        # finishes: release by CURRENT runner slot, then condense
        for i in range(W):
            if rows[i] is not None and rng.random() < 0.15:
                gen.release_request(i)
                rows[i] = None
                changed = True
        # arrivals into free runner slots
        free = [i for i in range(W) if rows[i] is None]
        rng.shuffle(free)
        new = sorted(free[: rng.randint(0, min(2, len(free)))])
        if new:
            ids = list(range(next_rid, next_rid + len(new)))
            next_rid += len(new)
            gen.model[0].prefill_paged_slots = lambda tl, pt, slots, ids=ids, **kw: (
                [(dev_owner.__setitem__(d, rid), dev_count.__setitem__(d, 0)) for d, rid in zip(slots, ids)],
                [torch.zeros(4) for _ in tl],
            )[1]
            _prefill(gen, new)
            for i, rid in zip(new, ids):
                rows[i] = rid
                ref_count[rid] = 0
            changed = True
        # a condense (front-pack) and sometimes a random permutation
        remap = None
        if changed or rng.random() < 0.1:
            live_rows = [i for i in range(W) if rows[i] is not None]
            idle_rows = [i for i in range(W) if rows[i] is None]
            remap = live_rows + idle_rows
            if rng.random() < 0.3:
                rng.shuffle(remap)
            if remap == list(range(W)) and rng.random() < 0.5:
                remap = None
            if remap is not None:
                remaps += 1
                rows = [rows[j] for j in remap]
        if not any(rows):
            need_reload = True  # a gap: the next decode is a commanded reload
            continue
        need_reload = False
        tokens = torch.tensor([[r if r is not None else 0] for r in rows], dtype=torch.int32)
        pt = tokens.repeat(1, 4).to(torch.int32)
        sp = _SP([float(r or 0) for r in rows], torch.tensor([r or 0 for r in rows]))
        pos = {i: 10 + step for i, r in enumerate(rows) if r is not None}
        reload = changed or remap is not None
        for _ in range(rng.randint(1, 3)):
            tt_out = _decode(gen, pos, remap=remap, reload=reload, tokens=tokens, sp=sp, pt=pt)
            remap, reload = None, False
            got = gen.process_decode_output_host(tt_out, is_tokens=True)
            for i, rid in enumerate(rows):
                if rid is not None:
                    ref_count[rid] += 1
                    assert int(got[i]) == rid * 1000 + ref_count[rid], f"step {step} row {i} request {rid}"
        assert rec.gdn_remaps == [] and rec.rope_remaps == []
        assert sorted(gen._ns_phys) == list(range(W))
        assert gen._ns_owned == {gen._ns_phys[i] for i, r in enumerate(rows) if r is not None}
    assert remaps > 5 and next_rid > 8
    assert any(b < W for b in steps_seen)


@pytest.mark.parametrize("dtype", [torch.uint32, torch.int32, torch.int64, torch.float32])
def test_gather_and_scatter_rows_are_dtype_safe(dtype):
    from models.demos.blackhole.qwen36.tt.qwen36_vllm import _ns_gather_rows, _ns_scatter_rows

    x = torch.tensor([[220], [220], [0], [198]]).to(dtype)
    out = _ns_gather_rows(x, [0, 2, 1, 3])
    assert out.dtype == dtype and out.flatten().tolist() == [220, 0, 220, 198]
    out = _ns_gather_rows(x[:2], [0, 2, 1, 3])  # phys >= len(x) -> zeros
    assert out.dtype == dtype and out.flatten().tolist() == [220, 0, 220, 0]
    inv = [0, 2, 1, 3]  # device row r takes runner slot inv[r]
    out = _ns_scatter_rows(x[:3], inv, 4, 7)  # short field padded with fill, then reordered
    assert out.dtype == dtype and out.flatten().tolist() == [220, 0, 220, 7]
    out = _ns_scatter_rows(x, inv, 4, 0)
    assert out.dtype == dtype and out.flatten().tolist() == [220, 0, 220, 198]


def test_remap_step_gathers_inside_the_call_with_the_new_map(ns, monkeypatch):
    """Generator.decode_forward(read_from_device=True) gathers the output inside the call: the new map must be live then."""
    gen, rec = ns

    def inside(self, *args, **kwargs):
        dev = 500 + torch.arange(W, dtype=torch.int64)  # device row d produced 500 + d
        return self.process_decode_output_host(dev, is_tokens=True)

    monkeypatch.setattr(Generator, "decode_forward", inside)
    out = _decode(gen, {0: 3, 1: 3, 2: 3}, remap=ROT, read_from_device=True)
    assert out.tolist() == [500 + p for p in slot_after(list(range(W)), ROT)]
    assert gen._ns_phys == slot_after(list(range(W)), ROT)


def test_failed_remap_decode_restores_map_seen_inside_the_call(ns, monkeypatch, expect_error):
    gen, rec = ns
    _decode(gen, {0: 3, 1: 3})
    before = (list(gen._ns_phys), set(gen._ns_owned))
    seen = []

    def boom(self, *args, **kwargs):
        seen.append(list(self._ns_phys))
        raise RuntimeError("device error")

    monkeypatch.setattr(Generator, "decode_forward", boom)
    with expect_error(RuntimeError, "device error"):
        _decode(gen, {0: 3, 1: 3, 2: 3}, remap=ROT)
    assert seen[0] == slot_after(before[0], ROT) and (gen._ns_phys, gen._ns_owned) == before
