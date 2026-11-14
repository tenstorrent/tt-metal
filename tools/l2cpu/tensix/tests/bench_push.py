#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Push bandwidth: ROW_MAJOR bf16 rows of 303,872 bytes (one page each, interleaved DRAM) -> L2CPU memory through the
coherent Memory Port alias or the uncached System Port alias, 1 and 32 rows, 1/2/4 cores. Each configuration is
captured in a trace and replayed N times back to back; the destination bytes are checked against the source."""
import argparse
import time

from link_setup import DATA_OFF, open_link, ops, stop_responder

ROW = 303_872


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    import torch
    import ttnn

    dev, hw, link = open_link()
    torch.manual_seed(0)
    src = torch.randn(1, 1, 32, ROW // 2).to(torch.bfloat16)
    t = ttnn.from_torch(
        src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ref = src.view(torch.int16).numpy().tobytes()
    hw.pa_write(link.base + ops.LINK_OFF_PUSH_SRC, link.push_source_table(t, ROW))
    print("| rows | alias | cores | ms per push | GB/s | bytes identical |\n|---|---|---|---|---|---|")
    for rows in (1, 32):
        for uncached in (False, True):
            for nc in (1,) if rows == 1 else (1, 2, 4):
                off = DATA_OFF + (0xC00000 if uncached else 0)  # separate zones: never mix the aliases on one line
                prog = link.push_program(rows, ROW, off, uncached=uncached, ncores=nc)
                link.run(prog)
                ttnn.synchronize_device(dev)
                rd = hw.pa_read(link.base + off - (ops.UNCACHED_DELTA if uncached else 0), rows * ROW)
                tid = ttnn.begin_trace_capture(dev, cq_id=0)
                link.run(prog)
                ttnn.end_trace_capture(dev, tid, cq_id=0)
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                ms = (time.perf_counter() - t0) / a.iters * 1e3
                ttnn.release_trace(dev, tid)
                print(
                    f"| {rows} | {'uncached' if uncached else 'coherent'} | {nc} | {ms:.3f} | {rows * ROW / ms / 1e6:.2f} | {rd == ref[: rows * ROW]} |",
                    flush=True,
                )
    stop_responder(hw, link)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
