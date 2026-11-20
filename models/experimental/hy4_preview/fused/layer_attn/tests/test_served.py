# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The "served" word of the collective's sender (fused/collective/kernels/collective_sender.cpp, the form
fused/layer_common/sender.py Sender(served=...) asks for), proven on the transport's own program
(Transport.xfer_program: the broadcast X1' and the reduce X6 with no compute between them) with the staging and the
landings as ranges of one arena tensor (Transport(arena=...)), the two things program P2 of the attention half
builds on.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k galaxy
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "parse and pair_row"
    (the PARSE-ONLY rung, before any build of a changed sender sends on those chips: test_parse)
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k solo
    ... -k pair_row, -k pair_col, -k galaxy   (each multi-chip rung after the one-chip rung has passed)

Three rounds on fresh data per rung: x bit-equal on every chip, F the exact sum, every counter at rest, and the served
word of every chip's sender core holds exactly the bits of its ops (root: 1 op; a chip that is not a head: one per
reduce core; a head: one per reduce core); the host then zeroes the words. Log lines start with HY4_MK8A."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
PAYLOAD = 98304
HUB, SENDER = (5, 9), (6, 9)
REDUCE = [(5, 6), (5, 7), (5, 8), (6, 8)]
SITE_ORIGIN, SITE_COLS, SITE_ROWS = (5, 0), 2, 8
SERVED_WORD = 15
ARENA = 393_216  # staging 98,304 | x and the test's source and partial on the hub; the landings on the reduce cores

galaxy = pytest.mark.parametrize("device_params, mesh_device", [(pb.device_params(), MESH)], indirect=True, ids=["8x4"])
RUNGS = {
    "solo": [(0, 0)],
    "pair_row": [(0, 0), (0, 1)],
    "pair_col": [(0, 0), (1, 0)],
    "galaxy": None,
}


def site_cores():
    return [(SITE_ORIGIN[0] + x, SITE_ORIGIN[1] + y) for y in range(SITE_ROWS) for x in range(SITE_COLS)]


def site_set():
    x, y = SITE_ORIGIN
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + SITE_COLS - 1, y + SITE_ROWS - 1))]
    )


def write_per_chip(mesh_device, tensor, data):
    shards = [ttnn.from_torch(data[i], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for i in range(data.shape[0])]
    ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, mesh_device.shape), tensor)


def read_chip(tensor, chip):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).float()


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS), ids=list(RUNGS))
def test_served(mesh_device, rung):
    if tp.is_mock() and RUNGS[rung] is not None and not os.environ.get("HY4_XFER_MOCK_ALL"):
        pytest.skip("the mock runs whole-mesh programs only")
    topo = tp.Topology(tuple(mesh_device.shape), (0, 0), RUNGS[rung])
    arena = tp.core_buffer(mesh_device, tp.grid_set(mesh_device), ARENA)
    base = tp.address(arena)
    # on the sender core: staging at 0; on the reduce cores: the landings at 0 and 98,304; on the hub: x at 0
    t = tp.Transport(
        mesh_device,
        topo,
        HUB,
        SENDER,
        REDUCE,
        PAYLOAD,
        arena=(arena, {"staging": 0, "landing1": 0, "landing2": 98304}),
        x_address=base,
        served_word=SERVED_WORD,
    )
    source = tp.core_buffer(mesh_device, [HUB], PAYLOAD)
    partial = tp.core_buffer(mesh_device, [HUB], PAYLOAD)
    sites = site_cores()
    per = len(sites) // len(REDUCE)
    f = tp.core_buffer(mesh_device, site_set(), PAYLOAD // len(sites), width=PAYLOAD // len(sites) // 2)
    targets = [sites[k * per : (k + 1) * per] for k in range(len(REDUCE))]
    io, desc = t.xfer_program(source, partial, targets, tp.address(f))
    io = [tensor for tensor in io if tensor is not None]
    chips = mesh_device.get_num_devices()
    logger.info(
        f"HY4_MK8A served {rung}: arena {ARENA} B at {base:#x}, L1 allocated per core "
        f"{ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank}"
    )
    if tp.is_mock():
        ttnn.experimental.prepare_generic_op(io, desc)
        return
    hub_index = HUB[1] * t.grid[0] + HUB[0]
    for round_ in range(3):
        g = torch.Generator().manual_seed(100 * round_ + len(rung))
        x = (torch.randn(PAYLOAD // 512, 256, generator=g) * 3).to(torch.bfloat16).float()
        part = torch.randint(-3, 4, (chips, PAYLOAD // 512, 256), generator=g).float()
        write_per_chip(mesh_device, source, x[None].repeat(chips, 1, 1))
        write_per_chip(mesh_device, partial, part)
        ttnn.synchronize_device(mesh_device)
        ttnn.generic_op(io, desc)
        ttnn.synchronize_device(mesh_device)
        words = t.read_words()
        served = {c: int(t.core_words(words[c], SENDER)[SERVED_WORD]) for c in topo.chips}
        want = {c: 1 if c == topo.root else (1 << len(REDUCE)) - 1 for c in topo.chips}
        rest = {
            (c, i): {w: hex(int(words[c][i][w])) for w in (0, 1, 2, 3, 4, 5, 12, 13, 14) if int(words[c][i][w])}
            for c in topo.chips
            for i in range(words[c].shape[0])
        }
        rest = {k: v for k, v in rest.items() if v}
        wrong_x = []
        for c in topo.chips:
            got = read_chip(arena, topo.chip(c)).reshape(t.grid[0] * t.grid[1], -1, 256)[hub_index][: PAYLOAD // 512]
            if not torch.equal(got, x):
                wrong_x.append(c)
        total = sum(part[topo.chip(c)] for c in topo.chips)
        got_f = read_chip(f, topo.chip(topo.root)).reshape(-1, 256)
        diff = int((got_f != total).sum())
        logger.info(
            f"HY4_MK8A served {rung} round {round_}: {len(topo.chips)} chips, x bit-equal on "
            f"{len(topo.chips) - len(wrong_x)}, F elements wrong {diff} of {total.numel()}, served words "
            f"{sorted(set(served.values()))} (wanted {sorted(set(want.values()))}), words not at rest: {rest or 'none'}"
        )
        assert not rest, f"{rung}: words not at rest: {rest}"
        assert not wrong_x, f"{rung}: x differs on {wrong_x}"
        assert diff == 0, f"{rung}: F is not the sum of the partials"
        assert served == want, f"{rung}: served words {served}"
        t.reset()


PARSE_OFFSET = 196_608  # on the sender core, behind the staging


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS), ids=list(RUNGS))
def test_parse(mesh_device, rung):
    """The sender's argument table as the kernel reads it, on the chips of the rung, with NO fabric connection opened
    and nothing sent (collective_sender.cpp HY4_COLLECTIVE_PARSE_ONLY): every chip's sender writes what it parsed
    (head arguments, served word, routes, ops, where the fabric arguments start) to a page, and the page must equal
    the table the host built. The other kernels of the transport's program run into their bounded waits (nothing
    arrives), which is expected; the words are zeroed afterwards."""
    if tp.is_mock() and RUNGS[rung] is not None and not os.environ.get("HY4_XFER_MOCK_ALL"):
        pytest.skip("the mock runs whole-mesh programs only")
    topo = tp.Topology(tuple(mesh_device.shape), (0, 0), RUNGS[rung])
    arena = tp.core_buffer(mesh_device, tp.grid_set(mesh_device), ARENA, ttnn.uint32)
    base = tp.address(arena)
    t = tp.Transport(
        mesh_device,
        topo,
        HUB,
        SENDER,
        REDUCE,
        PAYLOAD,
        wait_limit=2_000_000,
        arena=(arena, {"staging": 0, "landing1": 0, "landing2": 98304}),
        x_address=base,
        served_word=SERVED_WORD,
        parse_page=base + PARSE_OFFSET,
    )
    source = tp.core_buffer(mesh_device, [HUB], PAYLOAD)
    partial = tp.core_buffer(mesh_device, [HUB], PAYLOAD)
    f = tp.core_buffer(mesh_device, site_set(), PAYLOAD // 16, width=PAYLOAD // 32)
    sites = site_cores()
    targets = [sites[4 * k : 4 * k + 4] for k in range(len(REDUCE))]
    io, desc = t.xfer_program(source, partial, targets, tp.address(f))
    io = [tensor for tensor in io if tensor is not None]
    if tp.is_mock():
        ttnn.experimental.prepare_generic_op(io, desc)
        return
    ttnn.generic_op(io, desc)
    ttnn.synchronize_device(mesh_device)
    sender_index = SENDER[1] * t.grid[0] + SENDER[0]
    wrong = []
    for c in topo.chips:
        want = t.parsed[c]
        image = ttnn.to_torch(ttnn.get_device_tensors(arena)[topo.chip(c)]).to(torch.int64) & 0xFFFFFFFF
        page = image.reshape(t.grid[0] * t.grid[1], -1)[sender_index][
            PARSE_OFFSET // 4 : PARSE_OFFSET // 4 + len(want)
        ].tolist()
        same = page == want
        logger.info(
            f"HY4_MK8A parse {rung} chip {c}: {len(want)} words, {'equal to the host table' if same else 'DIFFERENT'}; "
            f"ops {want[4]}, connections {want[5]}, routes {want[6]}, fabric arguments from index {want[-2]}; "
            f"page {[hex(v) for v in page]}"
        )
        if not same:
            wrong.append((c, [(i, hex(a), hex(b)) for i, (a, b) in enumerate(zip(page, want)) if a != b][:8]))
    served = {c: int(t.core_words(t.read_words()[c], SENDER)[SERVED_WORD]) for c in topo.chips}
    logger.info(f"HY4_MK8A parse {rung}: served words {served} (nothing was sent: 0 expected)")
    t.reset()
    assert not wrong, f"{rung}: the sender parsed its table differently: {wrong}"
    assert not any(served.values()), f"{rung}: served words {served}"
