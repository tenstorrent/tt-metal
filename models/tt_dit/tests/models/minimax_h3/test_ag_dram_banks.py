# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Is the ring-attention all-gather delivering the right pages to the right place on this machine?

Standalone: no SDPA, no sparse path. Dumps every chip's DRAM harvesting mask and DRAM bank -> NOC0 table, then
runs `ring_attention_all_gather_async` on K-shaped tensors whose content encodes (shard, head, row), reads each
device's persistent output buffer back and checks it tile by tile. Wrong tiles are counted per (device, slot) and
per DRAM bank (page id mod number of banks), which is the fingerprint of a page landing in the wrong bank.

    pytest models/tt_dit/tests/models/minimax_h3/test_ag_dram_banks.py -s -k 4x8
"""

from __future__ import annotations

import collections
import re

import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.tensor import from_torch
from .common import GALAXY_RING

H, D = 14, 128
ROWS = 1184


def _harvesting_masks() -> dict[int, dict[str, int]]:
    """Per-chip harvesting masks from the serialized cluster descriptor (host only, no device needed)."""
    path = ttnn.cluster.serialize_cluster_descriptor()
    text = open(path).read()
    section = text.split("harvesting:", 1)[1] if "harvesting:" in text else ""
    masks: dict[int, dict[str, int]] = {}
    chip = None
    for line in section.splitlines():
        m = re.match(r"^\s{2}(\d+):\s*$", line)
        if m:
            chip = int(m.group(1))
            masks[chip] = {}
            continue
        m = re.match(r"^\s{4}(\w+):\s*(\S+)", line)
        if m and chip is not None:
            try:
                masks[chip][m.group(1)] = int(m.group(2), 0)
            except ValueError:
                pass
        elif line and not line.startswith(" ") and chip is not None:
            break
    return masks


def _device_ids(mesh_device) -> list[int]:
    for get in (
        lambda: list(mesh_device.get_device_ids()),
        lambda: [d.id() for d in mesh_device.get_devices()],
        lambda: list(range(mesh_device.get_num_devices())),
    ):
        try:
            return get()
        except AttributeError:
            continue
    return list(ttnn.get_device_ids())


@GALAXY_RING
def test_ag_dram_banks(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    S = tuple(mesh_device.shape)[sp_axis]
    ids = _device_ids(mesh_device)

    # 1. Harvesting masks and bank tables: do the chips agree on where logical DRAM bank k lives?
    masks = _harvesting_masks()
    for chip in sorted(masks):
        m = masks[chip]
        logger.info(
            f"[chip] {chip}: tensix {m.get('harvest_mask', 0):#x} dram {m.get('dram_harvesting_mask', 0):#x} "
            f"eth {m.get('eth_harvesting_mask', 0):#x}"
        )
    tables = {}
    for did in ids:
        t = ttnn.cluster.get_dram_bank_table(did)
        tables[did] = tuple((e["bank_id"], e["noc_x"], e["noc_y"]) for e in t)
    distinct = collections.Counter(tables.values())
    logger.info(f"[banks] {len(ids)} devices, {len(distinct)} distinct DRAM bank->NOC0 tables")
    for table, n in distinct.most_common():
        devs = [d for d in ids if tables[d] == table]
        logger.info(f"[banks] {n} devices {devs}: " + " ".join(f"b{b}=({x},{y})" for b, x, y in table))
    num_banks = len(next(iter(tables.values())))

    # 2. The all-gather on K/V-shaped tensors whose value encodes (shard, head, row): shard s holds
    #    s + 0.01 * head + 1e-5 * row in every column, exactly representable enough in bf16 to identify a tile.
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    shape = (S, H, ROWS, D)
    src = torch.zeros(shape)
    for s in range(S):
        for h in range(H):
            src[s, h] = (s + 0.01 * h + 1e-5 * torch.arange(ROWS).view(ROWS, 1)).expand(ROWS, D)
    src = src.bfloat16().float()
    mesh_axes = [sp_axis, None, None, None]
    k_dev = from_torch(src, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
    v_dev = from_torch(src, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=mesh_axes, layout=ttnn.TILE_LAYOUT)
    k_shape = (1, H, ROWS, D)
    for rep in range(2):
        pk = ccl.get_ag_ping_pong_buffer(k_shape, 2, sp_axis, dtype=ttnn.bfloat16)
        pv = ccl.get_ag_ping_pong_buffer(k_shape, 2, sp_axis, dtype=ttnn.bfloat16)
        ttnn.experimental.ring_attention_all_gather_async(
            [k_dev, v_dev],
            persistent_output_buffer=[pk, pv],
            dim=2,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl.ccl_sub_device_id,
        )
        ttnn.synchronize_device(mesh_device)
        for name, buf in (("K", pk), ("V", pv)):
            got = ttnn.to_torch(
                buf,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=[1, 0] if sp_axis == 1 else [0, 1], mesh_shape=tuple(mesh_device.shape)
                ),
            )[
                :, :H
            ].float()  # [S devices, H, S*ROWS, D]
            Nt, Dt = S * ROWS // 32, D // 32
            total_bad = 0
            for d in range(S):
                per_slot = []
                by_bank = collections.Counter()
                for s in range(S):
                    if s == d:
                        per_slot.append("local")
                        continue
                    blk = got[d, :, s * ROWS : (s + 1) * ROWS]  # [H, ROWS, D]
                    ref = src[s]
                    tiles_bad = (
                        (blk.view(H, ROWS // 32, 32, Dt, 32) != ref.view(H, ROWS // 32, 32, Dt, 32)).any(-1).any(2)
                    )
                    n_bad = int(tiles_bad.sum())
                    per_slot.append(f"s{s}:{n_bad}")
                    if n_bad:
                        for h, tr, tc in tiles_bad.nonzero().tolist():
                            page = h * Nt * Dt + (s * ROWS // 32 + tr) * Dt + tc
                            by_bank[page % num_banks] += 1
                    total_bad += n_bad
                tiles_per_slot = H * (ROWS // 32) * Dt
                logger.info(
                    f"[ag {name} rep {rep}] device {d}: bad tiles per slot (of {tiles_per_slot}) "
                    + " ".join(per_slot)
                    + (f"; by bank {dict(sorted(by_bank.items()))}" if by_bank else "")
                )
            logger.info(f"[ag {name} rep {rep}] total bad tiles {total_bad}")
            if rep == 1:
                assert total_bad == 0, f"{name}: {total_bad} gathered tiles are wrong"
