# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The collective's sender core (fused/collective/kernels/collective_sender.cpp) driven by an explicit table of ops,
for the cross-chip steps of a layer program that are not an all-gather over a ring: a broadcast from one chip, a
reduce towards one chip.

An op is one payload in the sender core's staging buffer that goes, once a core of this chip has set the op's
ready bit, into a landing address on a landing core of other chips (the same logical core and L1 address on every
chip: the allocator gives a tensor one address on every core of every chip) along the op's routes, the last packet
of each route counting one arrival in a counter word on that core; with ``own`` it also goes into the same address
on this chip by a NoC copy, counted last. The kernel is the collective's, with three additions it documents (a
persistent ready word in place of a program semaphore, ops without an own slot, a table without routes).

What differs from collective.py's ``Fragment``: no ``Site`` (no rings, no parities: the caller owns the landing and
says why one buffer is enough), any landing core and address per op, any set of routes per op, and no program
semaphore: the ready word is a persistent L1 word on the sender core, zero between programs.

Routes are the collective's tuples: (coord the first hop goes towards, 0 unicast | 1 multicast, destination coord
or None, hops e | w << 8 | n << 16 | s << 24). Only routes the collective or the merges piece ran on the board are
built here: a unicast to any chip (collective "pair", "row0", "col0", "axis" cases) and the four multicast branches
that reach every chip of the mesh (merges/mesh_site.py "mesh" routes, log mk6_topk_ladder). Link 0 only.
"""

from __future__ import annotations

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl

READY_IS_ADDRESS = 1 << 31
READY_HAS_SERVED = 1 << 30
OP_NO_OWN_SLOT = 1 << 8
READY_HAS_HANDOVERS = 1 << 29
OP_MORE_HANDOVERS_SHIFT = 16
HANDOVER_VALID = 1 << 31
HANDOVER_WORDS = 4  # per op in the hand-over table
MAX_OPS = cl.MAX_OPS


def unicast(dst):
    return (tuple(dst), 0, tuple(dst), 0)


def mesh_branches(mesh_shape, coord):
    """The four multicast branches from ``coord`` that between them reach every other chip of a torus mesh: east
    and west along its own row, south and north along its column with east / west teeth at every chip passed
    (merges/mesh_site.py MeshSite.routes_from, "mesh")."""
    rows, cols = mesh_shape
    r, c = coord
    east, west = cols // 2, cols - 1 - cols // 2
    south, north = rows // 2, rows - 1 - rows // 2
    teeth = east | (west << 8)
    out = []
    for hops, towards, word in (
        (east, (r, (c + 1) % cols), east),
        (west, (r, (c - 1) % cols), west << 8),
        (south, ((r + 1) % rows, c), teeth | (south << 24)),
        (north, ((r - 1) % rows, c), teeth | (north << 16)),
    ):
        if hops:
            out.append((towards, 1, None, word))
    return out


class Sender:
    """The sender kernel of ONE chip's program, with the ops the caller adds.

        s = Sender(mesh_device, coord, sender_core, ready_address, wait_limit)
        k = s.op(staging, nbytes, chunk, landing_core, slot, counter, routes, own=...)   # ready bit: s.bit(k)
        s.finish(desc)                                                                  # appends kernel + connections

    ``ready_address``: a persistent, host-zeroed L1 word on the sender core. A core hands an op over by writing the
    payload to ``staging`` on the sender core and then adding ``bit(k)`` to that word (one NoC atomic; several ops
    in one add is fine).

    ``served``: the address of a second persistent, host-zeroed L1 word on the sender core, or None. With it the
    sender adds ``1 << k`` to that word (an atomic on itself) once op k is served: every packet handed to the
    routers and every NoC write out of the op's staging acknowledged, so the staging may be reused for another op.
    The reader takes the bits it has seen off the word again.

    ``parse_page``: an L1 address on the sender core with room for PARSE_PAGE_BYTES. The kernel is then built
    parse-only (HY4_COLLECTIVE_PARSE_ONLY): it reads its argument table as the real build does, writes what it
    parsed to that page and returns without opening a fabric connection or sending anything; ``parsed`` (set by
    finish) is what the page must hold. The rung every change of the sender's argument layout gets on the board.

    ``handovers``: the address of a persistent, host-zeroed table on the sender core, HANDOVER_WORDS words per op
    (the kernel's "hand-over words", built with HY4_COLLECTIVE_HANDOVERS; needs ``served``), or None. An op may then
    be handed over ``op(..., handovers=n)`` times in one program, each time with its own payload bytes (0: nothing
    is sent or counted), its own offset from the op's landing slot and its own offset from the op's staging address:
    whoever hands op k over writes ``HANDOVER_VALID | bytes`` to word 4 k, the landing offset to word 4 k + 1 and
    the staging offset to word 4 k + 2 before it adds the ready bit, and makes the next hand-over of the same op
    only after it has seen the op's served bit and taken it off."""

    PARSE_PAGE_BYTES = 4096
    PARSE_MAGIC, PARSE_END = 0x50415253, 0x454E4421

    def __init__(
        self,
        mesh_device,
        coord,
        sender_core,
        ready_address,
        wait_limit=cl.DEFAULT_WAIT_LIMIT,
        link=0,
        served=None,
        parse_page=None,
        handovers=None,
    ):
        assert handovers is None or served is not None, "hand-over words need the served word"
        self.handovers = None if handovers is None else int(handovers)
        self.op_handovers = []
        self.parse_page = None if parse_page is None else int(parse_page)
        self.parsed = None
        self.mesh_device, self.coord, self.sender_core = mesh_device, tuple(coord), tuple(sender_core)
        self.ready_address, self.wait_limit, self.link = int(ready_address), int(wait_limit), link
        self.served = None if served is None else int(served)
        self.ops = []

    @staticmethod
    def bit(op: int) -> int:
        return 1 << (2 * op)  # parity 0 of the op's bit pair

    def op(self, staging, nbytes, chunk, landing_core, slot, counter, routes, own=True, handovers=1) -> int:
        assert len(self.ops) < MAX_OPS, f"more than {MAX_OPS} sender ops"
        assert handovers == 1 or (self.handovers is not None and 1 <= handovers <= 256), handovers
        self.op_handovers.append(int(handovers))
        assert nbytes % 16 == 0 and chunk % 16 == 0 and staging % 16 == 0 and slot % 16 == 0
        assert own or routes, "an op without an own slot needs a route"
        self.ops.append(
            (int(staging), int(nbytes), int(chunk), tuple(landing_core), int(slot), int(counter), list(routes), own)
        )
        return len(self.ops) - 1

    def finish(self, desc) -> int:
        """Append the sender kernel, its op table and its fabric connections to ``desc``. Returns its index."""
        assert self.ops
        mesh = self.mesh_device
        node = lambda coord: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
        src = node(self.coord)
        directions, connection_nodes, route_keys, route_table, op_table, parsed_ops = [], [], [], [], [], []
        for k, (staging, nbytes, chunk, landing_core, slot, counter, routes, own) in enumerate(self.ops):
            route_ids = []
            for towards, kind, dst, hops in routes:
                direction = ttnn.get_eth_forwarding_direction(src, node(towards))
                assert direction is not None, f"no route from {self.coord} to {towards}"
                if direction not in directions:
                    directions.append(direction)
                    connection_nodes.append(node(towards))
                key = (directions.index(direction), kind, node(dst).chip_id if dst is not None else 0, hops)
                if key not in route_keys:
                    route_keys.append(key)
                    route_table += list(key)
                route_ids.append(route_keys.index(key))
            landing = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*landing_core))
            # parity stride 0 and the counter as the control block: the kernel's parity is always 0 here
            op_table += [staging, nbytes, chunk, landing.x | (landing.y << 16), slot, 0, counter]
            more = (self.op_handovers[k] - 1) << OP_MORE_HANDOVERS_SHIFT
            op_table += [len(route_ids) | (0 if own else OP_NO_OWN_SLOT) | more] + route_ids
            parsed_ops += [
                staging,
                nbytes,
                chunk,
                landing.x | (landing.y << 16),
                slot,
                0,
                counter,
                len(route_ids),
                int(own),
            ]
            parsed_ops += ([self.op_handovers[k]] if self.handovers is not None else []) + route_ids
        num_routes = len(route_keys)
        assert (
            num_routes <= cl.PACKET_HEADERS_PER_RISC
        ), f"{num_routes} routes, {cl.PACKET_HEADERS_PER_RISC} packet headers"
        head = [self.ready_address | READY_IS_ADDRESS, len(self.ops), len(connection_nodes), num_routes]
        if self.served is not None:
            head[0] |= READY_HAS_SERVED
            head.append(self.served)
        if self.handovers is not None:
            head[0] |= READY_HAS_HANDOVERS
            head.append(self.handovers)
        flags = int(self.served is not None) | (2 if self.handovers is not None else 0)
        self.parsed = (
            [self.PARSE_MAGIC, self.ready_address | READY_IS_ADDRESS, flags, self.served or 0]
            + ([self.handovers] if self.handovers is not None else [])
            + [len(self.ops), len(connection_nodes), num_routes]
            + route_table
            + parsed_ops
            + [len(head) + len(route_table) + len(op_table), self.PARSE_END]
        )
        assert 4 * len(self.parsed) <= self.PARSE_PAGE_BYTES
        parse_defines = []
        if self.parse_page is not None:
            parse_defines = [("HY4_COLLECTIVE_PARSE_ONLY", "1"), ("HY4_COLLECTIVE_PARSE_PAGE", f"{self.parse_page}u")]
        if self.handovers is not None:
            parse_defines = parse_defines + [("HY4_COLLECTIVE_HANDOVERS", "1")]
        kernel = ttnn.KernelDescriptor(
            kernel_source=f"{cl.KERNELS}/collective_sender.cpp",
            core_ranges=cl.core_set([self.sender_core]),
            compile_time_args=[num_routes],
            # (with connections ttnn.setup_routing_plane_connection adds the fabric's defines itself)
            defines=cl.kernel_defines(self.wait_limit, False)
            + parse_defines
            + ([] if connection_nodes else [tuple(d) for d in ttnn.get_fabric_kernel_defines("Linear")]),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            compiler_include_paths=[str(cl.KERNEL_DIR)],
        )
        desc.kernels = list(desc.kernels) + [kernel]
        index = len(desc.kernels) - 1
        if connection_nodes:
            fabric_args = list(
                ttnn.setup_routing_plane_connection(
                    mesh, src, connection_nodes, [self.link], desc, index, ttnn.CoreCoord(*self.sender_core)
                )
            )
        else:  # a one-chip run: no connection; the connection manager still reads this chip's id and mesh id
            fabric_args = [int(src.chip_id), int(src.mesh_id)]
        x, y = self.sender_core
        desc.kernels[index].runtime_args[x][y] = head + route_table + op_table + fabric_args
        return index
