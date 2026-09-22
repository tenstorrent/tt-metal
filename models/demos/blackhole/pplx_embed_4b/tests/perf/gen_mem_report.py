# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-op L1/DRAM memory snapshot for pplx-embed-v1-4B.

Reproduces the ttnn-visualizer "memory per operation" view by snapshotting the
device's *live buffers* after every op (via ``ttnn._ttnn.reports.get_buffers``,
the same call the report DB's ``buffers`` table is built from) during ONE eager
(non-trace) prefill.  buffer_type 0=DRAM, 1=L1, 3=L1_SMALL; ``max_size_per_bank``
is per-bank bytes, so summing L1 buffers gives the per-core L1 high-water at that op.

Writes a per-op CSV and prints the peak-L1 ops + L1 headroom (the room to pin more
tensors into L1).  Build runs with reports OFF (graph-report tracing breaks the
RoPE ``from_torch``); only the forward is hooked.

    python models/demos/blackhole/pplx_embed_4b/tests/perf/gen_mem_report.py --batch 1 --seq 512 --out /tmp/pplx4b_logs/mem_bs1.csv
"""
import argparse
import csv
import time
from collections import defaultdict

from loguru import logger

import ttnn
from models.demos.blackhole.pplx_embed_4b.demo._common import (
    apply_workload_env,
    build_single_device_model,
    generate_synthetic_inputs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", type=str, default="/tmp/pplx4b_logs/mem.csv")
    args = ap.parse_args()

    apply_workload_env(args.batch, args.seq)

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000, num_command_queues=1)
    try:
        generator, model_args, kv_caches, page_table = build_single_device_model(
            device, batch_size=args.batch, seq_len=args.seq
        )
        input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, args.batch, args.seq)

        # L1 capacity per core (Blackhole worker L1). Query from device when available.
        try:
            l1_per_core = device.l1_size_per_core()
        except Exception:
            l1_per_core = 1499136  # Blackhole worker L1 default
        try:
            n_dram_banks = device.num_dram_channels()
        except Exception:
            n_dram_banks = 8

        records = []  # (op_idx, op_name, l1_per_bank_bytes, dram_total_bytes, n_l1_bufs, n_dram_bufs)
        op_counter = [0]

        def mem_hook(operation, function_args, function_kwargs, output):
            try:
                ttnn.synchronize_device(device)
                bufs = ttnn._ttnn.reports.get_buffers([device])
            except Exception:
                return None
            l1 = 0
            dram = 0
            n_l1 = n_dram = 0
            for b in bufs:
                bt = b.buffer_type.value if hasattr(b.buffer_type, "value") else int(b.buffer_type)
                if bt == 1:  # L1
                    l1 += b.max_size_per_bank
                    n_l1 += 1
                elif bt == 0:  # DRAM
                    dram += b.max_size_per_bank
                    n_dram += 1
            records.append((op_counter[0], str(operation), l1, dram * n_dram_banks, n_l1, n_dram))
            op_counter[0] += 1
            return None

        logger.info("Running ONE eager prefill with per-op buffer snapshots...")
        t0 = time.perf_counter()
        from ttnn.decorators import register_post_operation_hook

        with register_post_operation_hook(mem_hook):
            generator.prefill_forward_text(
                input_ids,
                page_table=page_table,
                kv_cache=kv_caches,
                prompt_lens=prompt_lens,
                enable_trace=False,
                return_hidden_states=True,
                warmup_prefill=False,
            )
        logger.info(f"Captured {len(records)} op snapshots in {time.perf_counter() - t0:.1f}s")

        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op_idx", "op_name", "l1_per_bank_bytes", "dram_total_bytes", "n_l1_bufs", "n_dram_bufs"])
            w.writerows(records)

        # Summary
        peak_l1 = max((r[2] for r in records), default=0)
        peak_dram = max((r[3] for r in records), default=0)
        logger.info("=" * 70)
        logger.info(f"  L1 per-core capacity:   {l1_per_core/1024:.0f} KB  ({n_dram_banks} DRAM banks)")
        logger.info(f"  Peak L1 per core:       {peak_l1/1024:.0f} KB  ({peak_l1/l1_per_core*100:.1f}% of L1)")
        logger.info(f"  L1 headroom (peak):     {(l1_per_core-peak_l1)/1024:.0f} KB")
        logger.info(f"  Peak DRAM (total):      {peak_dram/1024/1024:.0f} MB")
        logger.info(f"  CSV: {args.out}")
        logger.info("=" * 70)

        # Per-op-type peak L1 (which op holds the most L1 live)
        by_type = defaultdict(lambda: [0, 0])  # name -> [peak_l1, peak_dram]
        for _, name, l1, dram, _, _ in records:
            short = name.split("(")[0].strip()
            by_type[short][0] = max(by_type[short][0], l1)
            by_type[short][1] = max(by_type[short][1], dram)
        logger.info("Top ops by peak live L1 (per bank):")
        for name, (l1, dram) in sorted(by_type.items(), key=lambda kv: -kv[1][0])[:15]:
            logger.info(f"  {name:42s} L1={l1/1024:8.1f}KB  DRAM={dram/1024/1024:8.1f}MB")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
