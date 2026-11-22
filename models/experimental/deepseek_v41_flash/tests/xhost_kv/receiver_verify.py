# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Part 3 of the cross-host KV write test (see common.py): on host 41, after the tt-run job, opening no device.

Reads every page of both pools on every paired chip straight out of DRAM over a bare UMD cluster (no device lock, the
same path a KV manager uses next to a running server) and compares them byte for byte with what ``receiver_init.py``
recorded in ``DESCRIPTOR``: each block holds the data of the last round that wrote it, untouched blocks the sentinel.
It runs after the receiver closed its mesh, so it also shows the written pages outlive the process that allocated
them, as long as nothing reallocates that DRAM.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

import ttnn
from models.experimental.deepseek_v41_flash.tests.xhost_kv.common import (
    DESCRIPTOR,
    MESH_SHAPE,
    POOLS,
    expected_pool,
    log,
    page_bytes_of,
)

ROLE = "verify"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--descriptor", default=DESCRIPTOR)
    args = parser.parse_args()
    with open(args.descriptor) as f:
        desc = json.load(f)
    seed, writes = desc["seed"], desc["writes"]
    read_dram_umd_batch = ttnn.experimental.disaggregation.read_dram_umd_batch

    failed = 0
    for name, pool in desc["pools"].items():
        width, layout, page_bytes, ppb = POOLS[name]
        assert (pool["page_bytes"], pool["pages_per_block"], pool["width"]) == (page_bytes, ppb, width), pool
        n_pages = pool["blocks"] * ppb
        bad_chips = []
        for chip in desc["chips"]:
            banks = chip["banks"]
            # read_dram_umd addresses a bank as (DRAM view << 32) | offset and adds the view's own offset itself.
            assert all(off == 0 for _, _, off in banks), f"chip {chip['coord']}: nonzero bank offsets {banks}"
            nb = len(banks)
            per_bank = [(n_pages - b + nb - 1) // nb for b in range(nb)]
            raw = read_dram_umd_batch(
                [chip["unique_id"]] * nb,
                [(b << 32) | pool["addr"] for b in range(nb)],
                [count * page_bytes for count in per_bank],
            )
            chunks = torch.frombuffer(bytearray(raw), dtype=torch.uint8).split([c * page_bytes for c in per_bank])
            got = torch.empty(n_pages, page_bytes, dtype=torch.uint8)
            for bank, chunk in enumerate(chunks):
                got[bank::nb] = chunk.view(-1, page_bytes)
            r, c = chip["coord"]
            want = page_bytes_of(expected_pool(seed, r * MESH_SHAPE[1] + c, pool["blocks"], width, writes), layout)
            bad_pages = int((got != want).any(dim=1).sum())
            if bad_pages:
                bad_blocks = sorted({p // ppb for p in (got != want).any(dim=1).nonzero().flatten().tolist()})
                bad_chips.append((tuple(chip["coord"]), bad_pages, bad_blocks[:8]))
            failed += bad_pages
        log(
            ROLE,
            f"{name}: {len(desc['chips'])} chip(s) x {n_pages} page(s) of {page_bytes} B: "
            + ("all match" if not bad_chips else f"mismatches (coord, pages, first blocks) {bad_chips}"),
        )
    log(ROLE, "PASS" if not failed else f"FAIL: {failed} bad page(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
