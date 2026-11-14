# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage C: KV pool allocation probe at the loaded 78-layer (+MTP) model, no 1M prefill. Per chip, per mesh row pool of P pages (page = 512 tokens = 64 tokens x 8 columns):
79 layer tensors uint8 RM [P*64, 656] (SCALED_FP8 packed row; stored at the DRAM-aligned page size) + 22 index-key tensors bfp8 TILE [P*64, 128]. Reports how many pages fit
(bisection, allocate everything, read the allocator, free). Env: PAGES_LO/HI, KVROW (656|576|1152 widths), HY4_ATTN_BFP4, TAGNAME."""
import json
import os
import time

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

OUT = os.environ.get("OUT", "/out")
W = int(os.environ.get("KVROW", "656"))


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def try_pages(md, P, nl=79, nk=22):
    ts, ok, err = [], True, ""
    before = dram_view(md)
    try:
        for i in range(nl):
            ts.append(
                ttnn.empty(
                    [1, 1, P * 64, W],
                    dtype=ttnn.uint8,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        for i in range(nk):
            ts.append(
                ttnn.empty(
                    [1, 1, P * 64, 128],
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
    except Exception as e:  # noqa: BLE001
        ok, err = False, repr(e)[:300]
    after = dram_view(md)
    n_alloc = len(ts)
    for t in ts:
        ttnn.deallocate(t)
    return ok, err, before, after, n_alloc


def main():
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "600e6"))))
    try:
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=os.environ.get("HY4_CKPT", "/ckpt"))
        t0 = time.time()
        ch.add_layers(0, 78, log=lambda s: None)
        ch.add_layers(78, 79, log=lambda s: None)
        ch.build_head()
        res(event="built", sec=round(time.time() - t0, 1), **dram_view(md))
        lo, hi = int(os.environ.get("PAGES_LO", "1024")), int(os.environ.get("PAGES_HI", "6000"))
        hist = []
        # first the planning point 2048, then bisect
        for P in (2048, 4096):
            ok, err, b, a, n = try_pages(md, P)
            per_page_bank = (a["alloc_per_bank"] - b["alloc_per_bank"]) / P if ok else None
            res(
                event="probe",
                pages=P,
                ok=ok,
                err=err,
                n_tensors=n,
                bytes_per_page_per_bank=per_page_bank,
                bytes_per_page_per_chip=None if per_page_bank is None else per_page_bank * 8,
                alloc_delta_per_bank=a["alloc_per_bank"] - b["alloc_per_bank"],
                free_before_per_bank=b["free_per_bank"],
            )
        while hi - lo > 16:
            mid = (lo + hi) // 2
            ok, err, b, a, n = try_pages(md, mid)
            res(event="bisect", pages=mid, ok=ok)
            if ok:
                lo = mid
            else:
                hi = mid
        res(
            event="max_pages_fit",
            pages=lo,
            tokens_per_row=lo * 512,
            **{"free_after_weights_per_bank": dram_view(md)["free_per_bank"]},
        )
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
