# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The arguments a layer program uses to put projection groups into a program of its own (program.py ProjProgram:
cores, hub, rect, hub_buffers, cb_base, weights, the units' done groups, combine), on one chip, each against the
stand-alone program of the same group, whose outputs they must reproduce bit for bit; and the form without the
deferred norm (x arrives normalised, no gamma in the weights, scale 1).

    test_embedded          the pre group on caller-given cores inside a small rectangle, hub buffers at caller-given
                           addresses, circular-buffer ids moved, the gate units in a done group of their own
    test_two_groups        the pre and the query group as ONE program (combine): disjoint rectangles and ids
    test_weights_given     the stream as a tensor the caller made (mesh_stream of host-packed shards: one rounding)
    test_normalised        gamma_in = None and scale = 1 on a normalised x, against the deferred form's reference

On the mock Galaxy the programs are compiled only. Log lines start with HY4_MK7."""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.moe.program import hub_tensor
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.projections.tests.harness import (
    FID_KW,
    FID_TAG,
    check,
    random_input,
    random_reference,
    tables,
)
from models.experimental.hy4_preview.fused.projections.tests.test_query import query_inputs


def share(chip=0, gamma=True, seed=0):
    ref, gamma_in, _ = random_reference(seed)
    lw = L.layer_weights(ref.weights(), gamma_in if gamma else None)
    return lw, L.chip_weights(lw, chip)


def pre_x():
    y, scale = random_input()
    _, _, d_cos, d_sin = tables()
    return y, scale, torch.cat((y, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1)


def rect_cores(rect, n):
    """The first n cores of a rectangle in row-major order, and its last core (the hub)."""
    x0, y0, x1, y1 = rect
    cells = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    assert n + 1 <= len(cells)
    return cells[:n], cells[-1]


class CallerBuffers:
    """The five hub buffers as tensors of the caller's on the hub core (a layer program gives ranges of its arena)."""

    def __init__(self, device, hub, op_sizes, x_tiles, out_tiles):
        z = lambda n, w: torch.zeros(n, w)
        core = ttnn.CoreCoord(*hub)
        self.x = hub_tensor(device, core, z(x_tiles, 2 * 16 * L.ROWS), ttnn.bfloat16)
        self.out = hub_tensor(device, core, z(out_tiles, 2 * 16 * L.ROWS), ttnn.bfloat16)
        self.stat = hub_tensor(device, core, z(1, 2 * 16 * L.ROWS), ttnn.float32)
        self.meta = hub_tensor(device, core, torch.zeros(1, L.META_IN_WORDS, dtype=torch.int32), ttnn.uint16)
        self.status = hub_tensor(device, core, torch.zeros(1, P.STATUS_WORDS, dtype=torch.int32), ttnn.uint32)
        self.addresses = {name: getattr(self, name).buffer_address() for name in P.HUB_BUFFERS}

    def write(self, x, scale=None, lanes=None):
        host = ttnn.from_torch(L.to_row_tiles(x.float()), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.x)
        host = ttnn.from_torch(L.meta_words(scale, lanes), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.meta)

    def read(self):
        out = L.from_row_tiles(ttnn.to_torch(self.out).float())
        stat = L.from_row_tiles(ttnn.to_torch(self.stat).float())
        status = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(self.status).flatten().tolist()]
        return out, stat, status


def run_standalone(build, device, w, x, **params):
    op = build(device, w, fmt="bfp8", **FID_KW)
    op.set_x(x)
    op.set_params(**params)
    op()
    if P.MOCK:
        return None
    out, stat, status = op.read()
    op.check_status(status)
    return out, stat


def test_embedded(device):
    _, w = share()
    y, scale, x = pre_x()
    want = run_standalone(P.pre_program, device, w, x, scale=scale)
    rect = (2, 1, 6, 4)
    n_units = len(P.pre_units(w))
    cores, hub = rect_cores(rect, n_units)
    buffers = CallerBuffers(device, hub, None, P.PRE_X_TILES, P.pre_out_tiles(w))
    op = P.pre_program(
        device,
        w,
        fmt="bfp8",
        gate_done=1,
        cores=cores,
        hub=hub,
        rect=rect,
        hub_buffers=buffers.addresses,
        cb_base=24,
        **FID_KW,
    )
    assert op.hub_buffer_bytes()["x"] == P.PRE_X_TILES * P.ROW_TILE
    buffers.write(x, scale=scale)
    op()
    if P.MOCK:
        return
    out, stat, status = buffers.read()
    op.check_status(status)
    payload, gates = op.out_units_done(0), op.out_units_done(1)
    logger.info(
        f"HY4_MK7 interface{FID_TAG} embedded pre: output bit-equal to the stand-alone program "
        f"{torch.equal(out, want[0])}, statistic tile {torch.equal(stat, want[1])}; done group 0 (Xa's payload units) "
        f"{payload}, group 1 (gate units) {gates}; counters at {hex(op.done_address(0))}, {hex(op.done_address(1))}"
    )
    assert torch.equal(out, want[0]) and torch.equal(stat, want[1])
    assert payload == (n_units - 8, n_units - 8) and gates == (8, 8)


def test_two_groups(device):
    lw, w = share()
    y, scale, x = pre_x()
    q_a, lanes = query_inputs(lw)
    _, _, d_cos, d_sin = tables()
    xq = torch.cat((q_a, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1)
    want_pre = run_standalone(P.pre_program, device, w, x, scale=scale)
    want_query = run_standalone(P.query_program, device, w, xq, lanes=lanes)

    rect_a, rect_b = (0, 0, 3, 2), (4, 0, 7, 3)
    cores_a, hub_a = rect_cores(rect_a, len(P.pre_units(w)))
    cores_b, hub_b = rect_cores(rect_b, len(P.query_units(w)))
    pre = P.pre_program(device, w, fmt="bfp8", cores=cores_a, hub=hub_a, rect=rect_a, **FID_KW)
    query = P.query_program(device, w, fmt="bfp8", cores=cores_b, hub=hub_b, rect=rect_b, cb_base=24, **FID_KW)
    pre.set_x(x)
    pre.set_params(scale=scale)
    query.set_x(xq)
    query.set_params(lanes=lanes)
    program, io = P.combine([pre, query])
    if P.MOCK:
        ttnn.experimental.prepare_generic_op(io, program)
        return
    ttnn.generic_op(io, program)
    out_pre, stat_pre, status = pre.read()
    pre.check_status(status)
    out_query, _, status = query.read()
    query.check_status(status)
    same = (
        torch.equal(out_pre, want_pre[0]),
        torch.equal(stat_pre, want_pre[1]),
        torch.equal(out_query, want_query[0]),
    )
    logger.info(
        f"HY4_MK7 interface{FID_TAG} pre and query as one program: pre output, statistic tile and query output "
        f"bit-equal to the stand-alone programs {same}; units done {pre.out_units_done()}, {query.out_units_done()}"
    )
    assert all(same)


def test_weights_given(device):
    _, w = share()
    y, scale, x = pre_x()
    probe = P.pre_program(device, w, fmt="bfp8", **FID_KW)  # for its stream layout
    stream = P.mesh_stream(device, [P.host_shard(probe.host_stream(), "bfp8")])
    op = P.pre_program(device, w, fmt="bfp8", weights=stream, **FID_KW)
    op.set_x(x)
    op.set_params(scale=scale)
    op()
    if P.MOCK:
        return
    out, _, status = op.read()
    op.check_status(status)
    _, _, d_cos, d_sin = tables()
    held = {k: L.stored(v.float().contiguous(), "bfp8") if not isinstance(v, list) else v for k, v in w.items()}
    rounded = L.pre_golden(y, scale, held, d_cos, d_sin)
    exact = L.pre_golden(y, scale, w, d_cos, d_sin)
    check(
        f"interface{FID_TAG} caller's stream (packed on the host) q_a vs exact arithmetic on layout.stored",
        rounded["q_a"],
        out[:, : 2 * L.TILE],
    )
    score, _, _ = check(f"interface{FID_TAG} caller's stream q_a vs the reference", exact["q_a"], out[:, : 2 * L.TILE])
    assert score > 0.9995


def test_normalised(device):
    """x arrives normalised (the root applied the whole norm), the weights carry no gamma, scale = 1."""
    _, w_deferred = share(gamma=True)
    _, w = share(gamma=False)
    y, scale, _ = pre_x()
    _, _, d_cos, d_sin = tables()
    ref, gamma_in, _ = random_reference(0)
    # what the root sends: x = y / rms * gamma, rounded to BF16
    xn = ((y.double() * scale.double()[:, None]) * gamma_in.double()).float().to(torch.bfloat16).float()
    op = P.pre_program(device, w, fmt="bfp8", **FID_KW)
    op.set_x(torch.cat((xn, L.rope_tile(d_cos), L.rope_tile(d_sin)), dim=1))
    op.set_params()  # scale 1
    op()
    if P.MOCK:
        return
    out, _, status = op.read()
    op.check_status(status)
    deferred = L.pre_golden(y, scale, w_deferred, d_cos, d_sin)  # the reference of the deferred form
    ones = torch.ones(L.ROWS)
    held = {k: L.stored_stream(v.float().contiguous(), "bfp8") if not isinstance(v, list) else v for k, v in w.items()}
    here = L.pre_golden(xn, ones, held, d_cos, d_sin)
    q_a, gate = out[:, : 2 * L.TILE], out[:, P.PRE_GATE * L.TILE : P.PRE_IDX * L.TILE]
    check(f"interface{FID_TAG} normalised x, q_a vs exact arithmetic on the stored weights", here["q_a"], q_a)
    check(f"interface{FID_TAG} normalised x, gate vs exact arithmetic on the stored weights", here["gate"], gate)
    s1, _, _ = check(f"interface{FID_TAG} normalised x, q_a vs the reference (deferred norm)", deferred["q_a"], q_a)
    s2, _, _ = check(f"interface{FID_TAG} normalised x, gate vs the reference (deferred norm)", deferred["gate"], gate)
    assert s1 > 0.9995 and s2 > 0.9995


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mesh_stream_mock(mesh_device):
    """One mesh tensor with a shard per chip as the stream of the chips' pre groups: built and compiled on the mock
    Galaxy. On the board this is a 32-chip program (no fabric, but the first of its kind from this piece): it is the
    layer program's to run."""
    if not P.MOCK:
        pytest.skip("mock only: a multi-chip run belongs to the layer program")
    from models.experimental.hy4_preview.fused.projections.tests.test_latency import random_share

    shares = [random_share(chip, False, seed=chip) for chip in range(L.CHIPS)]
    chips_with_pool = [c for c in range(L.CHIPS) if "pool" in shares[c]]
    for w in shares:  # one stream layout on every chip: every chip gets a pool unit (zeros where it has none)
        w.setdefault("pool", torch.zeros(L.HIDDEN, L.TILE))
    shards = [P.host_shard(P.unit_stream(P.pre_units(w), "bfp8"), "bfp8") for w in shares]
    stream = P.mesh_stream(mesh_device, shards)
    rect = (2, 1, 6, 4)
    cores, hub = rect_cores(rect, len(P.pre_units(shares[0])))
    op = P.pre_program(
        mesh_device, shares[0], fmt="bfp8", weights=stream, cores=cores, hub=hub, rect=rect, cb_base=24, **FID_KW
    )
    op()
    logger.info(
        f"HY4_MK7 interface{FID_TAG} mesh stream: {len(shards)} shards of {tuple(shards[0].shape)} as one tensor at "
        f"{hex(stream.buffer_address())}; {len(chips_with_pool)} chips have a pool unit of their own"
    )
