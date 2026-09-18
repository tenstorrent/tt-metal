#!/usr/bin/env python3
"""List Tenstorrent chips, their IDs, mesh coordinates, and ethernet links.

Prints:
  1. A per-chip table (chip_id, mesh coord, unique_id, arch, MMIO/PCI, board,
     EthCoord).
  2. A per-chip list of ethernet connections (chan -> remote_chip:chan) plus an
     undirected edge summary.
  3. A 2D grid arranged by SystemMesh coordinates. Draws `---` / `|` between
     grid cells that share an ethernet link.

Uses `tt_umd` for topology / ethernet info and ttnn's `SystemMeshDescriptor`
for mesh coordinates.

Note: opens the device. Do not run while another process is using the card.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import tt_umd
import ttnn


def discover():
    opts = tt_umd.TopologyDiscoveryOptions()
    opts.discover_remote_devices = True
    opts.wait_on_ethernet_link_training = True
    return tt_umd.TopologyDiscovery.create_cluster_descriptor(opts)


def build_mesh_maps():
    """Return (mesh_shape, device_to_coord, coord_to_device, is_local_map)."""
    smd = ttnn._ttnn.multi_device.SystemMeshDescriptor()
    shape = tuple(smd.shape())
    rows, cols = shape[0], shape[1]
    device_to_coord = {}
    coord_to_device = {}
    is_local_map = {}
    all_local = smd.all_local()
    for r in range(rows):
        for c in range(cols):
            coord = ttnn.MeshCoordinate(r, c)
            local = True if all_local else smd.is_local(coord)
            is_local_map[(r, c)] = local
            if not local:
                coord_to_device[(r, c)] = None
                continue
            try:
                dev_id = smd.get_device_id(coord)
            except Exception:
                dev_id = None
            coord_to_device[(r, c)] = dev_id
            if dev_id is not None:
                device_to_coord[int(dev_id)] = (r, c)
    return (rows, cols), device_to_coord, coord_to_device, is_local_map


def print_chip_table(cluster_desc, device_to_coord):
    chips = sorted(cluster_desc.get_all_chips())
    mmio_map = cluster_desc.get_chips_with_mmio()
    unique_ids = cluster_desc.get_chip_unique_ids()
    locations = cluster_desc.get_chip_locations()

    header = (
        f"{'chip':>4}  {'mesh':>7}  {'unique_id':>18}  {'arch':<10}  {'mmio':<5}  "
        f"{'pci':>3}  {'board':<14}  {'cl':>2}  {'rk':>2}  {'sh':>2}  {'x':>3}  {'y':>3}"
    )
    line = "=" * len(header)
    print(line)
    print("CHIPS")
    print(line)
    print(header)
    print("-" * len(header))
    for chip_id in chips:
        uid = unique_ids.get(chip_id)
        arch = str(cluster_desc.get_arch(chip_id)).split(".")[-1]
        is_mmio = cluster_desc.is_chip_mmio_capable(chip_id)
        pci = mmio_map.get(chip_id, "-")
        board = str(cluster_desc.get_board_type(chip_id)).split(".")[-1]
        loc = locations.get(chip_id)
        if loc is not None:
            cid, rack, shelf, x, y = loc.cluster_id, loc.rack, loc.shelf, loc.x, loc.y
        else:
            cid = rack = shelf = x = y = "-"
        mesh = device_to_coord.get(int(chip_id))
        mesh_s = f"({mesh[0]},{mesh[1]})" if mesh is not None else "-"
        uid_s = f"0x{uid:016x}" if isinstance(uid, int) else str(uid)
        print(
            f"{chip_id:>4}  {mesh_s:>7}  {uid_s:>18}  {arch:<10}  {str(is_mmio):<5}  "
            f"{str(pci):>3}  {board:<14}  {str(cid):>2}  {str(rack):>2}  {str(shelf):>2}  {str(x):>3}  {str(y):>3}"
        )
    print()


def print_connections(cluster_desc):
    eth = cluster_desc.get_ethernet_connections()
    print("=" * 60)
    print("ETHERNET CONNECTIONS  (chip:chan -> remote_chip:chan)")
    print("=" * 60)
    pair_counts = defaultdict(int)
    for chip_id in sorted(eth):
        chan_map = eth[chip_id]
        conns = []
        for chan in sorted(chan_map):
            remote_chip, remote_chan = chan_map[chan]
            conns.append(f"{chan}->{remote_chip}:{remote_chan}")
            a, b = sorted((int(chip_id), int(remote_chip)))
            pair_counts[(a, b)] += 1
        print(f"chip {chip_id}: {', '.join(conns) if conns else '(none)'}")
    print()
    print("-" * 60)
    print("Edge summary (undirected):")
    print("-" * 60)
    for (a, b), count in sorted(pair_counts.items()):
        # each physical link is reported from both endpoints
        print(f"  {a} <-> {b} : {count // 2} link(s)")
    print()
    return pair_counts


def print_mesh_grid(mesh_shape, coord_to_device, is_local_map, cluster_desc):
    rows, cols = mesh_shape
    eth = cluster_desc.get_ethernet_connections()

    linked = set()
    for chip_id, chan_map in eth.items():
        for _c, (remote_chip, _rc) in chan_map.items():
            a, b = sorted((int(chip_id), int(remote_chip)))
            linked.add((a, b))

    def has_link(c1, c2):
        if c1 is None or c2 is None:
            return False
        a, b = sorted((int(c1), int(c2)))
        return (a, b) in linked

    print("=" * 60)
    print(f"MESH GRID  (SystemMesh shape = {rows} x {cols})")
    print("=" * 60)
    print("Cells show:  chip_id @ (row,col)")
    print()

    cell_w = 16
    hspace = 5  # width of the "---" gap between cells
    for r in range(rows):
        # chip row
        row_str = ""
        for c in range(cols):
            dev = coord_to_device.get((r, c))
            local = is_local_map.get((r, c), True)
            if dev is None and not local:
                cell = "[remote]"
            elif dev is None:
                cell = "[  ?  ]"
            else:
                cell = f"[{dev:>2} @({r},{c})]"
            row_str += f"{cell:^{cell_w}}"
            if c != cols - 1:
                right = coord_to_device.get((r, c + 1))
                row_str += " --- " if has_link(dev, right) else "     "
        print(row_str)
        # vertical link row
        if r != rows - 1:
            vrow = ""
            for c in range(cols):
                top = coord_to_device.get((r, c))
                bot = coord_to_device.get((r + 1, c))
                mark = "|" if has_link(top, bot) else " "
                vrow += f"{mark:^{cell_w}}"
                if c != cols - 1:
                    vrow += " " * hspace
            print(vrow)
    print()


def main():
    print("Running topology discovery...")
    cluster_desc = discover()
    n = len(cluster_desc.get_all_chips())
    print(f"Discovered {n} chip(s).\n")

    mesh_shape, device_to_coord, coord_to_device, is_local_map = build_mesh_maps()
    print(f"SystemMesh shape: {mesh_shape[0]} x {mesh_shape[1]}\n")

    print_chip_table(cluster_desc, device_to_coord)
    print_connections(cluster_desc)
    print_mesh_grid(mesh_shape, coord_to_device, is_local_map, cluster_desc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
