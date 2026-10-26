# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device RoPE tables (TtMiMoModel.setup_decode_rope_tables, rope.partial_rope_table): no device. Decode, verify and
MTP draft steps read their cos / sin as table rows at their positions instead of the host's per-step
``decode_host_inputs`` (partial_rope_cos_sin of the step's 32 positions, idle slots at 0). The rows, whole or compact
(the 64 rotated dims, followed on device by the constant 1 / 0 tail: rope.rope_table_rows mirrors that read), must be
bit for bit the host's fp32 values; both go through the same bf16 conversion (ttnn.from_torch), so the greedy output
cannot change. Checked for every position up to 1M + the table margin, both of MiMo's thetas, and through
setup_decode_rope_tables itself (from_torch replaced by the identity: the tables, their order, widths, margin and
tails). The tables are built in the host's per-step call shape (rope.partial_rope_table): torch's CPU sin / cos of a
value can change in the last bit with the size of the tensor it is computed in, but not with the step's other
positions (test_rows_do_not_depend_on_the_other_slots)."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tt import model as model_lib
from models.experimental.mimo_v2_6_flash.tt.model import ROPE_TABLE_MARGIN, TtMiMoModel
from models.experimental.mimo_v2_6_flash.tt.rope import partial_rope_cos_sin, partial_rope_table, rope_table_rows

B = 32  # max_batch: a decode step's cos / sin rows
HEAD_DIM, ROPE_DIM = 192, 64  # MiMo-V2.6-Flash: partial RoPE on 64 of the 192 Q / K head dims
THETAS = {"swa": 10000.0, "full": 5000000.0}  # swa_rope_theta (SWA and MTP layers), rope_theta (global layers)
MAX_SEQ_LEN = 1 << 20  # the MTP serving entry's context


def host_step(positions, theta):
    """decode_host_inputs' values: partial_rope_cos_sin of one step's B positions, idle slots (-1) at position 0."""
    return partial_rope_cos_sin(positions.clamp(min=0), HEAD_DIM, ROPE_DIM, theta)


def step_batches(g):
    """Position batches of B slots near 0, 128K, 512K, 1M - 1 and the margin past it, some with idle slots."""
    out = []
    for base in (0, 1, 131072 - 16, 131072, 524288 - 7, MAX_SEQ_LEN - B, MAX_SEQ_LEN - 1):
        out.append(torch.arange(base, base + B).clamp(max=MAX_SEQ_LEN + ROPE_TABLE_MARGIN - 1))
    for _ in range(64):
        p = torch.randint(0, MAX_SEQ_LEN, (B,), generator=g)
        p[torch.rand(B, generator=g) < 0.3] = -1
        out.append(p)
    # A verify block: one request at p .. p + 3 in slots 0-3, the other slots idle.
    for p in (5, 131070, MAX_SEQ_LEN - 4):
        out.append(torch.cat((torch.arange(p, p + 4), torch.full((B - 4,), -1))))
    return out


@pytest.fixture(scope="module", params=sorted(THETAS))
def tables(request):
    """Whole and compact tables of one theta over MAX_SEQ_LEN + ROPE_TABLE_MARGIN positions."""
    theta = THETAS[request.param]
    rows = MAX_SEQ_LEN + ROPE_TABLE_MARGIN
    return theta, {
        "whole": partial_rope_table(rows, HEAD_DIM, ROPE_DIM, theta, batch=B),
        "compact": partial_rope_table(rows, HEAD_DIM, ROPE_DIM, theta, width=ROPE_DIM, batch=B),
    }


def test_rows_equal_the_host_step_at_every_position(tables):
    """Every position to 1M + margin, in consecutive steps of B: rows (whole and compact + tail) == host cos / sin."""
    theta, made = tables
    rows = MAX_SEQ_LEN + ROPE_TABLE_MARGIN
    for start in range(0, rows, B):
        positions = torch.arange(start, start + B)
        want = host_step(positions, theta)
        for kind, (cos_t, sin_t) in made.items():
            got = rope_table_rows(cos_t, sin_t, positions, HEAD_DIM)
            for g, w in zip(got, want):
                assert torch.equal(g, w), f"{kind} table, positions {start} .. {start + B - 1}"


def test_rows_equal_the_host_step_on_served_batches(tables):
    """Random and edge batches with idle slots (the device index is max(position, 0), spec_inputs.derive_verify)."""
    theta, made = tables
    for positions in step_batches(torch.Generator().manual_seed(0)):
        want = host_step(positions, theta)
        for kind, (cos_t, sin_t) in made.items():
            got = rope_table_rows(cos_t, sin_t, positions.clamp(min=0), HEAD_DIM)
            for g, w in zip(got, want):
                assert torch.equal(g, w), f"{kind} table, positions {positions.tolist()}"
                assert torch.equal(g.bfloat16(), w.bfloat16())


def test_rows_do_not_depend_on_the_other_slots():
    """A position's host cos / sin are the same whatever the step's other positions (so one table row serves every
    step): position p alone among idle slots, among random positions, and in every slot."""
    g = torch.Generator().manual_seed(1)
    for theta in THETAS.values():
        for p in (0, 3, 4095, 131071, 131072, 777777, MAX_SEQ_LEN - 1):
            ref = [t[0] for t in host_step(torch.cat((torch.tensor([p]), torch.full((B - 1,), -1))), theta)]
            for _ in range(8):
                positions = torch.randint(0, MAX_SEQ_LEN, (B,), generator=g)
                slot = int(torch.randint(0, B, (1,), generator=g))
                positions[slot] = p
                got = host_step(positions, theta)
                assert all(torch.equal(t[slot], r) for t, r in zip(got, ref)), (theta, p, slot)
            got = host_step(torch.full((B,), p), theta)
            assert all(torch.equal(t[s], r) for t, r in zip(got, ref) for s in range(B)), (theta, p)


def model_stub(max_seq_len):
    setups = {
        name: SimpleNamespace(head_dim=HEAD_DIM, rope_dim=ROPE_DIM, theta=theta, datatype="bf16")
        for name, theta in THETAS.items()
    }
    return SimpleNamespace(
        max_seq_len=max_seq_len,
        max_batch=B,
        decode_trace=None,
        rope_setups=setups,
        mesh_device=None,
        plan=SimpleNamespace(replicate=lambda: None),
    )


@pytest.mark.parametrize("compact", [False, True], ids=["whole", "compact"])
@pytest.mark.parametrize("max_seq_len", [4096, MAX_SEQ_LEN])
def test_setup_decode_rope_tables(monkeypatch, compact, max_seq_len):
    """The tables setup_decode_rope_tables uploads (from_torch replaced by the identity): cos / sin of the SWA theta,
    then of the global theta, max_seq_len + margin rows of 192 (64 compact) dims, and with compact tables the tails
    the reads append; their rows are the host step's values."""
    monkeypatch.setattr(model_lib.ttnn, "from_torch", lambda t, **kwargs: t)
    stub = model_stub(max_seq_len)
    TtMiMoModel.setup_decode_rope_tables(stub, compact=compact)
    rows, width = max_seq_len + ROPE_TABLE_MARGIN, ROPE_DIM if compact else HEAD_DIM
    assert [tuple(t.shape) for t in stub.rope_tables] == [(rows, width)] * 4
    per_chip = sum(t.numel() for t in stub.rope_tables) * 2  # bf16
    if max_seq_len == MAX_SEQ_LEN:
        assert round(per_chip / 1e6) == (537 if compact else 1611)
    if compact:
        ones, zeros = stub.rope_table_tails
        assert torch.equal(ones, torch.ones(1, 1, B, HEAD_DIM - ROPE_DIM))
        assert torch.equal(zeros, torch.zeros(1, 1, B, HEAD_DIM - ROPE_DIM))
    else:
        assert stub.rope_table_tails is None
    positions = torch.tensor([0, 1, 2, max_seq_len // 2, max_seq_len - 1, max_seq_len + ROPE_TABLE_MARGIN - 1])
    for k, name in enumerate(("swa", "full")):
        cos_t, sin_t = stub.rope_tables[2 * k : 2 * k + 2]
        got = rope_table_rows(cos_t, sin_t, positions, HEAD_DIM)
        for g, w in zip(got, host_step(positions, THETAS[name])):
            assert torch.equal(g, w), name
