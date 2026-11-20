# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The "hand-over words" form of the collective's sender (fused/collective/kernels/collective_sender.cpp built with
HY4_COLLECTIVE_HANDOVERS; fused/layer_common/sender.py Sender(handovers=...)): an op handed over several times in
one program, each time with its own payload bytes (0: nothing sent, nothing counted) and its own offset from the
op's landing slot. What program P2 of the attention half needs for two sub-passes of 8 token rows with a row mask
(PLAN.md section 19).

Per chip: op 0 is handed over 3 times (the whole payload into slot 0 | 0 bytes | half the payload, FROM A SECOND
STAGING, into slot 2), op 1
once with words (a quarter, slot 1 of its own landing), op 2 once WITHOUT words (the op's own bytes and slot: the
form must leave such an op as it was). On one chip every op lands on the chip itself (own copy); on more chips op 0
and op 1 go to the next chip of the rung by a unicast and op 2 stays an own copy.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k galaxy
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "parse and pair_row"
    (the PARSE-ONLY rung first: nothing is opened or sent; then test_handover solo, pair_row, pair_col, galaxy, each one rung at a time)

Checks: every landing slot holds exactly the pattern of the hand-over that was to write it and the fill everywhere
else, each arrival counter the number of hand-overs with bytes, the ready and served words are zero, no status.
Log lines start with HY4_MK8A."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.layer_common.sender import Sender, unicast
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
KERNEL = os.path.join(os.path.dirname(__file__), "..", "kernels", "attn_handover_test.cpp")
SENDER, LAND = (6, 9), (5, 9)
PAYLOAD = 16_384
# one uint32 arena on every core. Sender core: staging of the 3 ops, a second staging of op 0 | words | parse page.
# Landing core: 3 slots of op 0 | 2 slots of op 1 | 1 slot of op 2 | counters
A_STAGE, A_WORDS, A_PARSE = 0, 4 * PAYLOAD, 4 * PAYLOAD + 1024
W_READY, W_SERVED, W_STATUS, W_TABLE = 0, 1, 2, 16  # words on the sender core (the table: 4 words per op)
L_OP = [0, 3 * PAYLOAD, 5 * PAYLOAD]
A_COUNT = 6 * PAYLOAD  # on the landing core: one control block of 16 words per op (counter first, status at + 12)
ARENA = 6 * PAYLOAD + 4096
FILL = 0x0EEEEEEE
# (op, write the words, payload bytes, landing offset, staging offset) in the order they are handed over
HANDOVERS = [
    (0, 1, PAYLOAD, 0, 0),
    (1, 1, PAYLOAD // 4, PAYLOAD, 0),
    (0, 1, 0, PAYLOAD, 0),
    (2, 0, 0, 0, 0),
    (0, 1, PAYLOAD // 2, 2 * PAYLOAD, 3 * PAYLOAD),
]

galaxy = pytest.mark.parametrize("device_params, mesh_device", [(pb.device_params(), MESH)], indirect=True, ids=["8x4"])
RUNGS = {"solo": [(0, 0)], "pair_row": [(0, 0), (0, 1)], "pair_col": [(0, 0), (1, 0)], "galaxy": None}


def build(mesh_device, rung, parse_only):
    topo = tp.Topology(tuple(mesh_device.shape), (0, 0), RUNGS[rung])
    chips = list(topo.chips)
    arena = tp.core_buffer(mesh_device, tp.grid_set(mesh_device), ARENA, ttnn.uint32)
    base = tp.address(arena)
    word = lambda i: base + A_WORDS + 4 * i
    mesh_desc, parsed = ttnn.MeshProgramDescriptor(), {}
    for i, coord in enumerate(chips):
        nxt = chips[(i + 1) % len(chips)]
        remote = [unicast(nxt)] if len(chips) > 1 else []
        desc = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=[])
        s = Sender(
            mesh_device,
            coord,
            SENDER,
            word(W_READY),
            2_000_000 if parse_only else 30_000_000,
            served=word(W_SERVED),
            parse_page=base + A_PARSE if parse_only else None,
            handovers=word(W_TABLE),
        )
        count = lambda k: base + A_COUNT + 64 * k
        s.op(base + A_STAGE, PAYLOAD, 4096, LAND, base + L_OP[0], count(0), remote, own=not remote, handovers=3)
        s.op(base + A_STAGE + PAYLOAD, PAYLOAD, 4096, LAND, base + L_OP[1], count(1), remote, own=not remote)
        s.op(base + A_STAGE + 2 * PAYLOAD, PAYLOAD, 4096, LAND, base + L_OP[2], count(2), [], own=True)
        s.finish(desc)
        parsed[coord] = s.parsed
        rt = ttnn.RuntimeArgs()
        args = [word(W_READY), word(W_SERVED), word(W_TABLE), word(W_STATUS), len(HANDOVERS)]
        for op, has_words, nbytes, offset, stage in HANDOVERS:
            args += [op, base + A_STAGE + op * PAYLOAD + stage, PAYLOAD, has_words, nbytes, offset, stage]
        rt[SENDER[0]][SENDER[1]] = args
        desc.kernels = list(desc.kernels) + [
            ttnn.KernelDescriptor(
                kernel_source=os.path.abspath(KERNEL),
                core_ranges=tp.core_set([SENDER]),
                compile_time_args=[],
                runtime_args=rt,
                defines=[("HANDOVER_TEST_WAIT", "2000000u" if parse_only else "30000000u")],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
            )
        ]
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
    # (a generic op takes at least two tensors: the arena and a word buffer nothing uses)
    return topo, [arena, tp.core_buffer(mesh_device, [SENDER], 1024)], mesh_desc, parsed


def core_words(mesh_device, arena, chip, core):
    grid = mesh_device.compute_with_storage_grid_size()
    image = ttnn.to_torch(ttnn.get_device_tensors(arena)[chip]).to(torch.int64) & 0xFFFFFFFF
    return image.reshape(grid.x * grid.y, -1)[core[1] * grid.x + core[0]]


def fill_landings(mesh_device, arena):
    """Every word of every core's arena zero, the landing core's three landings FILL."""
    grid = mesh_device.compute_with_storage_grid_size()
    image = torch.zeros(grid.x * grid.y, ARENA // 4, dtype=torch.int32)
    image[LAND[1] * grid.x + LAND[0], : 6 * PAYLOAD // 4] = FILL
    host = ttnn.from_torch(image.reshape(-1, 256), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_host_shards([host] * mesh_device.get_num_devices(), mesh_device.shape), arena
    )


def expected_landing():
    """(the landing core's three landings as words, the three arrival counts) when every op lands here."""
    want = torch.full((6 * PAYLOAD // 4,), FILL, dtype=torch.int64)
    number, counts = [0, 0, 0], [0, 0, 0]
    for op, has_words, nbytes, offset, _ in HANDOVERS:
        if not has_words:
            nbytes, offset = PAYLOAD, 0
        first = (L_OP[op] + offset) // 4
        want[first : first + nbytes // 4] = (op << 24) | (number[op] << 16) | torch.arange(nbytes // 4)
        number[op] += 1
        counts[op] += int(nbytes != 0)
    return want, counts


def test_expected_host():
    want, counts = expected_landing()
    assert counts == [2, 1, 1]
    assert int((want != FILL).sum()) == (PAYLOAD + PAYLOAD // 2 + PAYLOAD // 4 + PAYLOAD) // 4
    assert int(want[PAYLOAD // 4]) == FILL and int(want[2 * PAYLOAD // 4]) == (0 << 24) | (2 << 16)


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS), ids=list(RUNGS))
def test_handover(mesh_device, rung):
    if tp.is_mock() and RUNGS[rung] is not None and not os.environ.get("HY4_XFER_MOCK_ALL"):
        pytest.skip("the mock runs whole-mesh programs only")
    topo, io, desc, _ = build(mesh_device, rung, False)
    arena = io[0]
    if tp.is_mock():
        ttnn.experimental.prepare_generic_op(io, desc)
        return
    want, counts = expected_landing()
    for round_ in range(2):  # (the second launch finds the words as the first left them)
        fill_landings(mesh_device, arena)
        ttnn.synchronize_device(mesh_device)
        ttnn.generic_op(io, desc)
        ttnn.synchronize_device(mesh_device)
        bad = []
        for coord in topo.chips:
            chip = topo.chip(coord)
            sender, land = core_words(mesh_device, arena, chip, SENDER), core_words(mesh_device, arena, chip, LAND)
            words = sender[A_WORDS // 4 : A_WORDS // 4 + 16].tolist()
            got = land[: 6 * PAYLOAD // 4]
            wrong = int((got != want).sum())
            seen = [int(land[(A_COUNT + 64 * k) // 4]) for k in range(3)]
            sender_status = [int(land[(A_COUNT + 64 * k) // 4 + 12]) for k in range(3)]
            # the sender's own count of the hand-overs it closed, per op (word 3 of the op's hand-over words)
            closed = [int(sender[A_WORDS // 4 + W_TABLE + 4 * k + 3]) for k in range(3)]
            asked = [sum(1 for h in HANDOVERS if h[0] == k) for k in range(3)]
            ok = wrong == 0 and seen == counts and words[W_READY] == 0 and words[W_SERVED] == 0 and words[W_STATUS] == 0
            ok = ok and not any(sender_status) and closed == asked
            logger.info(
                f"HY4_MK8A handover {rung} round {round_} chip {coord}: landing words wrong {wrong} of {want.numel()}, "
                f"arrivals {seen} (wanted {counts}), ready {words[W_READY]:#x}, served {words[W_SERVED]:#x}, "
                f"status {words[W_STATUS]:#x}, sender status {[hex(v) for v in sender_status]}, hand-overs the sender "
                f"closed per op {closed} (asked {asked})"
            )
            if not ok:
                first = [
                    (i, hex(int(a)), hex(int(b))) for i, (a, b) in enumerate(zip(got.tolist(), want.tolist())) if a != b
                ][:6]
                bad.append((coord, wrong, seen, words[:3], closed, first))
        assert not bad, f"{rung} round {round_}: {bad}"


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS), ids=list(RUNGS))
def test_parse(mesh_device, rung):
    """PARSE ONLY: every chip's sender reads its argument table in the hand-over form (the table's address after the
    served word's, each op's hand-overs in its routes word), writes what it parsed to a page and returns; no fabric
    connection is opened and nothing is sent. The test kernel's first wait for a served bit runs out (expected)."""
    if tp.is_mock() and RUNGS[rung] is not None and not os.environ.get("HY4_XFER_MOCK_ALL"):
        pytest.skip("the mock runs whole-mesh programs only")
    topo, io, desc, parsed = build(mesh_device, rung, True)
    arena = io[0]
    if tp.is_mock():
        ttnn.experimental.prepare_generic_op(io, desc)
        return
    ttnn.generic_op(io, desc)
    ttnn.synchronize_device(mesh_device)
    wrong = []
    for coord in topo.chips:
        want = parsed[coord]
        sender = core_words(mesh_device, arena, topo.chip(coord), SENDER)
        page = sender[A_PARSE // 4 : A_PARSE // 4 + len(want)].tolist()
        words = sender[A_WORDS // 4 : A_WORDS // 4 + 3].tolist()
        same = page == want
        logger.info(
            f"HY4_MK8A handover parse {rung} chip {coord}: {len(want)} words, "
            f"{'equal to the host table' if same else 'DIFFERENT'}; flags {want[2]}, table {want[4]:#x}, ops {want[5]}, "
            f"connections {want[6]}, routes {want[7]}; ready / served / status words {[hex(v) for v in words]}; "
            f"page {[hex(v) for v in page]}"
        )
        if not same:
            wrong.append((coord, [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(page, want)) if a != b][:8]))
        closed = [int(sender[A_WORDS // 4 + W_TABLE + 4 * k + 3]) for k in range(3)]
        assert words[W_SERVED] == 0 and closed == [
            0,
            0,
            0,
        ], f"{rung} chip {coord}: a parse-only sender served something: {words} {closed}"
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"
