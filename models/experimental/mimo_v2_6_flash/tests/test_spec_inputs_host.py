# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed speculative inputs (reference/spec_inputs.py): the torch mirrors of the device programs that derive a verify's
and an MTP draft's trace inputs from one upload give exactly what the host used to build and copy input by input
(TtMiMoModel.prepare_decode_inputs_host, kv.owned_positions / owner_mask, spec_decode.serial_positions, and the
MTP's host inputs from plan_mtp and mtp_cache_position); no device. The device programs (tt/spec_inputs.py) run the
mirrors' ops one for one; test_spec_inputs_tp8.py checks them on a Galaxy row."""

import random

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    SERIAL_WRITES,
    SPEC_SLOTS,
    DraftSource,
    draft_source,
    mtp_cache_position,
    plan_mtp,
    plan_verify_passes,
    serial_positions,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import (
    INT32_MAX,
    as_uint32,
    derive_mtp,
    derive_verify,
    owned_slot_caps,
    owned_slot_columns,
    pack_mtp,
    pack_verify,
    slot_caps,
    tile_padded,
)
from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib

B, TP, REP = 32, 8, 2  # max_batch; a TP=8 replica's global layers: 4 KV heads, chip pairs split the users
MAX_SEQ_LEN = 1 << 20


def verify_host_values(tokens, positions, rope):
    """The torch values of prepare_decode_inputs_host(tokens, positions, serial=True) per chip, tile-layout tensors
    padded to whole tiles as ttnn.from_torch pads them (the cos / sin aside: they are not derived)."""
    out = []
    owned = kv_lib.owned_positions(positions, TP, REP)
    owner = kv_lib.owner_mask(positions, TP, REP)
    for chip in range(TP):
        values = {
            "tokens": tokens.reshape(1, -1).to(torch.int64),
            "cur_pos": positions.to(torch.int32),
            "mask": tile_padded((positions >= 0).to(torch.bfloat16).reshape(1, 1, -1, 1)),
            "pos_owned": owned[chip],
            "owner_mask": tile_padded(owner[chip]),
            "serial": [torch.tensor(s, dtype=torch.int32) for s in serial_positions(positions.tolist(), SPEC_SLOTS)],
        }
        if rope:
            values["rope_pos"] = positions.clamp(min=0).reshape(1, -1).to(torch.int64)
        out.append(values)
    return out


def check_verify(tokens, positions, rope=True):
    packed = pack_verify(tokens, positions)
    caps, own_cap, own_col = slot_caps(B), owned_slot_caps(B, TP, REP), owned_slot_columns(B, TP, REP)
    for chip, want in enumerate(verify_host_values(tokens, positions, rope)):
        got = derive_verify(packed, own_cap[chip], own_col[chip], caps, rope)
        assert set(got) == set(want)
        for name, value in want.items():
            if name == "serial":
                assert len(got[name]) == len(value)
                for g, w in zip(got[name], value):
                    assert torch.equal(g, w), (chip, name)
            else:
                assert got[name].dtype == value.dtype and torch.equal(got[name], value), (chip, name)


def random_positions(g, active, low=0, high=MAX_SEQ_LEN):
    """[B] positions: ``active`` random slots at random positions in [low, high), the others -1."""
    pos = torch.full((B,), -1, dtype=torch.int32)
    slots = torch.randperm(B, generator=g)[:active]
    pos[slots] = torch.randint(low, high, (active,), generator=g, dtype=torch.int32)
    return pos


# ---------------------------------------------------------------------- constants
def test_constants():
    own = owned_slot_caps(B, TP, REP)
    pos = torch.arange(B, dtype=torch.int32) * 1000 - 1
    assert torch.equal(torch.minimum(pos.reshape(1, -1), own), kv_lib.owned_positions(pos, TP, REP))
    assert torch.equal(owned_slot_columns(B, TP, REP), kv_lib.owner_mask(torch.zeros(B, dtype=torch.int32), TP, REP))
    caps = slot_caps(B)
    assert caps.shape == (SERIAL_WRITES, B) and int(caps.max()) == INT32_MAX
    assert [list(map(int, (caps[k] == INT32_MAX).nonzero().flatten())) for k in range(SERIAL_WRITES)] == [
        list(range(k, SPEC_SLOTS, SERIAL_WRITES)) for k in range(SERIAL_WRITES)
    ]


def test_uint32_rows_are_the_same_bits():
    assert as_uint32(torch.tensor([0, 151999, INT32_MAX, -1], dtype=torch.int32)).tolist() == [
        0,
        151999,
        INT32_MAX,
        0xFFFFFFFF,
    ]


# ---------------------------------------------------------------------- verify
def test_verify_idle():
    check_verify(torch.zeros(B, dtype=torch.int32), torch.full((B,), -1, dtype=torch.int32))


@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("seed", range(20))
def test_verify_random_positions(seed, rope):
    g = torch.Generator().manual_seed(seed)
    active = int(torch.randint(0, B + 1, (1,), generator=g))
    tokens = torch.randint(0, 152576, (B,), generator=g, dtype=torch.int32)
    check_verify(tokens, random_positions(g, active), rope)


@pytest.mark.parametrize("seed", range(20))
def test_verify_packed_passes(seed):
    """The slots of the passes plan_verify_passes makes (rows of 1 + n candidates, consecutive positions), in the
    slots the adapter fills (_enqueue_verify_pass: the first SPEC_SLOTS), some starting at 0 or near max_seq_len."""
    rng = random.Random(seed)
    rows = []
    for row in range(rng.randint(1, 4)):
        n = rng.randint(0, MAX_DRAFTS)
        start = rng.choice([0, 1, 255, 256, 4095, MAX_SEQ_LEN - 1 - n, rng.randrange(MAX_SEQ_LEN - n)])
        rows.append((row, [rng.randrange(152576) for _ in range(n + 1)], list(range(start, start + n + 1)), n))
    for vp in plan_verify_passes(rows):
        tokens = torch.zeros(B, dtype=torch.int32)
        positions = torch.full((B,), -1, dtype=torch.int32)
        tokens[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
        positions[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
        check_verify(tokens, positions)


def test_verify_rejects_positions_below_idle(expect_error):
    with expect_error(AssertionError, "positions are >= -1"):
        pack_verify(torch.zeros(B, dtype=torch.int32), torch.full((B,), -2, dtype=torch.int32))


# ---------------------------------------------------------------------- MTP draft
def mtp_host_values(tokens, plan):
    """What TtMiMoMTP built on the host input by input before the packed upload (its ``host_inputs``): tokens, the
    fill / step2 / step3 cache positions, the serial fill positions, and the true positions whose cos / sin it
    uploaded (clamped at 0 there: the RoPE table rows a device run reads)."""
    idle = [-1] * SPEC_SLOTS
    fill, step2, step3 = (plan.fill, plan.step2, plan.step3) if plan is not None else (idle, idle, idle)

    def padded(values, fill_value):
        t = torch.full((B,), fill_value, dtype=torch.int32)
        t[: len(values)] = torch.tensor(values, dtype=torch.int32)
        return t

    def cache(values):
        return padded([mtp_cache_position(p) for p in values], -1)

    tok = padded(tokens if tokens is not None else [0] * SPEC_SLOTS, 0).reshape(1, -1)
    return {
        "tokens": tok.to(torch.int64),
        "fill": cache(fill),
        "serial": [padded(s, -1) for s in serial_positions([mtp_cache_position(p) for p in fill])],
        "step2": cache(step2),
        "step3": cache(step3),
        "rope_idx": [padded(v, -1).clamp(min=0).reshape(1, -1).to(torch.int64) for v in (fill, step2, step3)],
    }


def check_mtp(tokens, plan, rope=True):
    got = derive_mtp(pack_mtp(B, tokens, plan), slot_caps(B), rope)
    want = mtp_host_values(tokens, plan)
    if not rope:
        del want["rope_idx"]
    assert set(got) == set(want)
    for name, value in want.items():
        if isinstance(value, list):
            assert len(got[name]) == len(value)
            for g, w in zip(got[name], value):
                assert g.dtype == w.dtype and torch.equal(g, w), name
        else:
            assert got[name].dtype == value.dtype and torch.equal(got[name], value), name


def test_mtp_idle():
    check_mtp(None, None)
    check_mtp(None, None, rope=False)


# Last committed positions around the ring's wrap (MTP_RING_TOKENS = 256), at 0 and at the end of the context.
BONUS = [1, 2, 3, 4, 254, 255, 256, 257, 258, 511, 512, 513, 4096, MAX_SEQ_LEN - 1 - MAX_DRAFTS]


@pytest.mark.parametrize("rope", [True, False])
@pytest.mark.parametrize("bonus", BONUS)
def test_mtp_after_a_verify(bonus, rope):
    """A verify of one row at positions p .. p + n (slots 0 .. n) committed m tokens, at p + 1 .. p + m = ``bonus``."""
    rng = random.Random(bonus)
    for n in range(MAX_DRAFTS + 1):
        for m in range(1, n + 2):
            p = bonus - m
            if p < 0:
                continue
            source = draft_source(plan_verify_passes([(0, list(range(10, 11 + n)), list(range(p, p + n + 1)), n)]))
            assert source.slots == list(range(n + 1))
            committed = [p + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)
            plan = plan_mtp(source, committed, m)
            tokens = [0] * SPEC_SLOTS
            for c in range(m):
                tokens[source.slots[c]] = rng.randrange(152576)
            check_mtp(tokens, plan, rope)


@pytest.mark.parametrize("bonus", BONUS)
@pytest.mark.parametrize("slot", range(SPEC_SLOTS))
def test_mtp_after_a_plain_step(bonus, slot):
    """A decode step of one request in any decode slot (generator_vllm.decode_forward's DraftSource)."""
    plan = plan_mtp(DraftSource(row=7, slots=[slot]), [bonus] + [-1] * MAX_DRAFTS, 1)
    tokens = [0] * SPEC_SLOTS
    tokens[slot] = 151999
    check_mtp(tokens, plan)
