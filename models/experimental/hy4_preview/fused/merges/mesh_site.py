# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A collective site (fused/collective/collective.py ``Site``) whose members are any set of chips in a fixed order,
for the two merges of this directory. It is used through the collective's own ``Fragment`` and sender kernel,
unchanged; what it adds to ``Site``:

* one ring of ALL the chips of the mesh, reached with four multicast branches per packet (east, west, south with
  east / west teeth, north with east / west teeth: the decomposition tt_fabric_api.h fabric_set_mcast_route asks of
  a client). Every member ends with every member's payload: an all-gather over the 32 chips in one fabric pass,
  where two ``Site``s (one per mesh axis) take two passes and can only hand on a sum;
* slots of exactly ``part_bytes`` (rounded up to 16) instead of whole tiles: 32 members of a 528-byte part take
  34 KB of landing per core for the two parities, not 128 KB. The landing is read by address, never as a CB.

Members ``coords`` are (row, col) in the order that numbers the slots: member i's payload is in slot i on every
member. Routes by the shape of the member set: the whole mesh -> "mesh" (four branches); a whole mesh row or
column -> the collective's two walks; anything else (two chips, a few chips) -> one unicast route per other
member, at most 12. ``coords`` may also be several rings of one size (a list of lists: the rows or the columns of
a rectangle of chips), as ``Site`` takes them; a chip is in at most one.

``staging=False`` leaves the sender-side staging out: for a landing that a sender of another kind fills (the
log-sum-exp merge's column stage), with only the landing-core protocol of collective.hpp in use.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl


class MeshSite(cl.Site):
    def __init__(
        self,
        mesh_device,
        coords,
        part_bytes: int,
        parts: int = 1,
        landing_core=cl.LANDING_CORE,
        sender_core=cl.SENDER_CORE,
        dtype=ttnn.bfloat16,
        name: str = "mesh_site",
        staging: bool = True,
    ):
        # Not Site.__init__: the slot stride and the buffer sizes differ. Every attribute Site's methods and
        # Fragment read is set here.
        self.mesh_device, self.name, self.dtype = mesh_device, name, dtype
        self.mesh_shape = tuple(mesh_device.shape)
        rows, cols = self.mesh_shape
        coords = list(coords)
        many = isinstance(coords[0][0], (tuple, list))
        self.rings = [[tuple(c) for c in ring] for ring in (coords if many else [coords])]
        ring = self.rings[0]
        self.members = len(ring)
        assert self.members >= 2 and all(len(r) == self.members for r in self.rings), "rings of one size >= 2"
        self.where = {}
        for one in self.rings:
            for index, coord in enumerate(one):
                assert coord not in self.where, f"{coord} is in two rings of {name}"
                self.where[coord] = (one, index)
        whole_mesh = not many and ring == [(r, c) for r in range(rows) for c in range(cols)]
        lines = [[(r, c) for c in range(cols)] for r in range(rows)] + [
            [(r, c) for r in range(rows)] for c in range(cols)
        ]
        line = all(one in lines for one in self.rings)
        self.routes = "mesh" if whole_mesh and rows > 1 and cols > 1 else "multicast" if line else "unicast"
        assert self.routes != "unicast" or self.members - 1 <= cl.PACKET_HEADERS_PER_RISC, "too many unicast routes"

        self.parts = int(parts)
        self.landing_core, self.sender_core = tuple(landing_core), tuple(sender_core)
        self.landing_cores = cl.row_cores(self.landing_core, self.parts)
        grid = mesh_device.compute_with_storage_grid_size()
        assert 1 <= self.parts <= cl.MAX_OPS and self.landing_cores[-1][0] < grid.x, f"{self.parts} landing cores"
        assert self.sender_core not in self.landing_cores

        self.tile_bytes = cl.TILE_BYTES[dtype]
        self.part_bytes = int(part_bytes)
        assert self.part_bytes > 0 and self.part_bytes % 16 == 0, "a part is a multiple of 16 bytes"
        self.payload_bytes = self.part_bytes * self.parts
        self.slot_stride = self.part_bytes
        self.parity_stride = self.members * self.slot_stride
        max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.chunk_bytes = min(self.part_bytes, max_payload - max_payload % 16)
        self.landing_tiles = -(-2 * self.parity_stride // self.tile_bytes)  # per landing core
        self.staging_tiles = -(-self.parts * self.slot_stride // self.tile_bytes) if staging else 1
        self.tiles = -(-self.part_bytes // self.tile_bytes)

        self.landing = cl.tile_column(mesh_device, self.landing_core, self.landing_tiles, dtype, cores=self.parts)
        self.ctrl = cl.tile_column(mesh_device, self.landing_core, 1, ttnn.uint32, cores=self.parts)
        self.staging = cl.tile_column(mesh_device, self.sender_core, self.staging_tiles, dtype)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        self._ctrl_zero = ttnn.from_torch(
            torch.zeros(1, 1, self.parts * cl.TILE, cl.TILE),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        self._landing_zero = ttnn.from_torch(
            torch.zeros(1, 1, self.parts * self.landing_tiles * cl.TILE, cl.TILE),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        ttnn.synchronize_device(mesh_device)
        self.landing_address = ttnn.get_device_tensors(self.landing)[0].buffer_address()
        self.ctrl_address = ttnn.get_device_tensors(self.ctrl)[0].buffer_address()
        self.staging_address = ttnn.get_device_tensors(self.staging)[0].buffer_address()

    def l1_bytes(self) -> int:
        """L1 this site takes on every core of every chip: landing and ctrl of one core, and the staging."""
        return (self.landing_tiles + self.staging_tiles) * self.tile_bytes + cl.TILE_BYTES[ttnn.uint32]

    def routes_from(self, coord) -> list:
        """As Site.routes_from. "mesh": four multicast branches; a branch leaves through one connection (the
        branch's cardinal direction) and its east / west teeth are delivered by the routers along the trunk."""
        if self.routes != "mesh":
            return super().routes_from(coord)
        rows, cols = self.mesh_shape
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

    def bytes_per_use(self) -> dict:
        """Fabric bytes of one use by one chip: payload bytes it injects (once per route), payload bytes that
        land on it from the other members, and packets it injects."""
        routes = len(self.routes_from(self.rings[0][0]))
        packets_per_part = -(-self.part_bytes // self.chunk_bytes)
        return {
            "injected": routes * self.payload_bytes,
            "received": (self.members - 1) * self.payload_bytes,
            "packets": routes * self.parts * packets_per_part,
            "routes": routes,
        }
