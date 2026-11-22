# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Part 1 of the cross-host KV write test (see common.py): the decode side, rank 1 on host 41.

Brings up mesh 1's fabric, allocates decode-shaped paged pools (``POOLS``: compressed KV and index keys) filled with
``SENTINEL``, and sends the sender their addresses with every chip's DRAM bank table, the way a decode server would
send the pages it allocated for a request. Each round it picks a random set of blocks (a page-table row), asks for
them to be written, and reads them back the moment the sender's "done" arrives: no device-side wait of any kind, so a
pass says the TCP reply did not overtake the fabric writes. After the last round every block of every pool is
checked (untouched blocks must still hold the sentinel), and the rounds' block lists go to ``DESCRIPTOR`` for
``receiver_verify.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import torch

import ttnn
from models.experimental.deepseek_v41_flash.tests.xhost_kv.common import (
    BLOCK,
    DESCRIPTOR,
    MESH_SHAPE,
    PORT,
    POOLS,
    RECEIVER_RANK,
    SENTINEL,
    block_data,
    close_mesh,
    connect,
    expected_pool,
    init_rank,
    log,
    open_mesh,
)
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import recv_msg, send_msg

ROLE = "receiver"


def describe(mesh, pools: dict[str, ttnn.Tensor], pairs: list[tuple[int, int]], seed: int) -> dict:
    """What the sender needs to address this mesh's pages: per chip its fabric node, ASIC id and DRAM bank table
    (TRANSLATED NOC x / y and offset a bank), per pool its address (the same on every chip) and page geometry."""
    chips = []
    for r, c in pairs:
        coord = ttnn.MeshCoordinate(r, c)
        node = mesh.get_fabric_node_id(coord)
        table = ttnn.cluster.get_dram_bank_table(mesh.get_device_id(coord))
        chips.append(
            {
                "coord": [r, c],
                "chip_id": int(node.chip_id),
                "unique_id": int(
                    ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh, int(node.mesh_id), int(node.chip_id))
                ),
                "banks": [[int(e["noc_x"]), int(e["noc_y"]), int(e["base_addr"])] for e in table],
            }
        )
    mesh_ids = {int(mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)).mesh_id) for r, c in pairs}
    assert len(mesh_ids) == 1, mesh_ids
    return {
        "op": "connect",
        "mesh_id": mesh_ids.pop(),
        "seed": seed,
        "chips": chips,
        "pools": {
            name: {
                "addr": int(t.buffer_address()),
                "page_bytes": POOLS[name][2],
                "pages_per_block": POOLS[name][3],
                "blocks": int(t.shape[0]),
                "width": POOLS[name][0],
            }
            for name, t in pools.items()
        },
    }


def bad_rows(mesh, pools, pairs, seed: int, writes: list[list[int]], whole: bool) -> dict[str, int]:
    """Rows a pool, over the paired chips, that differ from what the rounds ``writes`` should have left: only the
    last round's blocks, or (``whole``) every block of the pool."""
    C = MESH_SHAPE[1]
    bad = {}
    for name, pool in pools.items():
        width = POOLS[name][0]
        shards = ttnn.get_device_tensors(pool)
        bad[name] = 0
        for r, c in pairs:
            q = r * C + c
            got = ttnn.to_torch(shards[q])
            if whole:
                want = expected_pool(seed, q, got.shape[0], width, writes)
            else:
                blocks = writes[-1]
                got, want = got[blocks], block_data(seed, len(writes) - 1, q, len(blocks), width)
            bad[name] += int((got != want).any(dim=-1).sum())
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sender", default="10.82.97.40", help="the sender's address (host 40)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--fabric", default="FABRIC_2D_TORUS_XY", help="the same on both ranks")
    parser.add_argument("--payload", type=int, default=None, help="fabric packet payload bytes, the same on both ranks")
    parser.add_argument(
        "--chips", default=None, help="comma-separated row-major mesh indices of the chip pairs to use (default: all)"
    )
    parser.add_argument("--pool-blocks", type=int, default=256, help="blocks a pool (64 entries a block)")
    parser.add_argument("--write-blocks", type=int, default=32, help="blocks written a round")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.write_blocks <= args.pool_blocks:
        sys.exit(f"--write-blocks {args.write_blocks} must be in 1..--pool-blocks {args.pool_blocks}")
    n_chips = MESH_SHAPE[0] * MESH_SHAPE[1]
    chips = list(range(n_chips)) if args.chips is None else [int(q) for q in args.chips.split(",")]
    if not chips or not all(0 <= q < n_chips for q in chips) or len(set(chips)) != len(chips):
        sys.exit(f"--chips {args.chips}: distinct indices in 0..{n_chips - 1}")

    init_rank(RECEIVER_RANK, ROLE)
    mesh = open_mesh(args.fabric, args.payload)
    log(ROLE, f"mesh {tuple(mesh.shape)} open, {args.fabric}")
    pairs = [divmod(q, MESH_SHAPE[1]) for q in chips]
    failed = 0
    try:
        pools = {
            name: ttnn.from_torch(
                torch.full((args.pool_blocks, 1, BLOCK, width), SENTINEL, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for name, (width, layout, _, _) in POOLS.items()
        }
        desc = describe(mesh, pools, pairs, args.seed)
        log(ROLE, "pools " + ", ".join(f"{n} @ {p['addr']:#x}" for n, p in desc["pools"].items()))
        sock = connect(args.sender, args.port)
        send_msg(sock, desc)
        rng = random.Random(args.seed)
        writes: list[list[int]] = []
        for round_ in range(args.rounds):
            writes.append(rng.sample(range(args.pool_blocks), args.write_blocks))
            send_msg(sock, {"op": "write", "round": round_, "blocks": writes[-1]})
            done, _ = recv_msg(sock)
            t_done = time.perf_counter()
            assert done.get("done") == round_, done
            bad = bad_rows(mesh, pools, pairs, args.seed, writes, whole=False)
            failed += sum(bad.values())
            log(
                ROLE,
                f"round {round_}: sender {done['seconds'] * 1e3:.1f} ms; read back {time.perf_counter() - t_done:.2f} s"
                " after its done: " + ", ".join(f"{n} {v} bad row(s)" for n, v in bad.items()),
            )
        send_msg(sock, {"op": "bye"})
        sock.close()
        bad = bad_rows(mesh, pools, pairs, args.seed, writes, whole=True)
        failed += sum(bad.values())
        log(ROLE, "whole pools after the last round: " + ", ".join(f"{n} {v} bad row(s)" for n, v in bad.items()))
        desc.update(writes=writes)
        with open(DESCRIPTOR, "w") as f:
            json.dump(desc, f)
        log(ROLE, f"{'PASS' if not failed else f'FAIL: {failed} bad row(s)'}; descriptor in {DESCRIPTOR}")
        return 1 if failed else 0
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    sys.exit(main())
