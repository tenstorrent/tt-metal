#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qualify the two-reader DRAM-sharded decode matmul for bf8 / bf4 weights on one Blackhole chip (under a hold).

For every (K, N) of ``decode_matmul.TWO_WORKER_PROJECTIONS`` and every dense dtype, the same host weight (bf16, then
packed to the dtype on the host) is uploaded in the one-reader and the two-reader bank layouts and multiplied by the
same 32-row bf16 activation with the dtype's compute config (HiFi4 / HiFi2 / LoFi, fp32 accumulation); the two
outputs must be bitwise equal.  A host fp32 reference on the packed weight (``ttnn.to_torch`` of the upload) bounds
the absolute error so a wrong-format read cannot pass as "equal garbage".  Writes a JSON record.

    python tools/qualify_dense_two_readers.py --out qualification.json [--device-id 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(20260924)
    device = ttnn.open_device(device_id=args.device_id)
    record = {
        "device_id": args.device_id,
        "rows": args.rows,
        "banks": None,
        "shapes": [],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        record["banks"] = int(device.dram_grid_size().x)
        for (k, n), cores in dm.TWO_WORKER_PROJECTIONS.items():
            host_weight = (torch.randn(1, 1, k, n) * 0.02).to(torch.bfloat16)
            activation = (torch.randn(1, 1, args.rows, k) * 0.5).to(torch.bfloat16)
            for dtype in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b):
                tag = dm.dense_dtype_tag(dtype)
                compute = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, dm.dense_math_fidelity_name(dtype)),
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
                outputs = {}
                packed_reference = None
                timings = {}
                for workers in (1, 2):
                    weight = ttnn.as_tensor(
                        host_weight,
                        dtype=dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=dm.dram_sharded_weight_memory_config(
                            device, k, n, num_workers_per_dram_bank=workers
                        ),
                    )
                    if packed_reference is None:
                        packed_reference = ttnn.to_torch(weight).float().reshape(k, n)
                    act_config, program_config = dm.dram_sharded_matmul_configs(
                        device, k, n, num_cores=cores, num_workers_per_dram_bank=workers
                    )
                    act = ttnn.from_torch(activation, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                    act_ws = ttnn.to_memory_config(act, act_config)
                    out_ws = ttnn.linear(
                        act_ws,
                        weight,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        program_config=program_config,
                        compute_kernel_config=compute,
                    )
                    ttnn.synchronize_device(device)
                    started = time.perf_counter()
                    for _ in range(5):
                        rep = ttnn.linear(
                            act_ws,
                            weight,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                            program_config=program_config,
                            compute_kernel_config=compute,
                        )
                        ttnn.deallocate(rep)
                    ttnn.synchronize_device(device)
                    timings[workers] = round((time.perf_counter() - started) / 5 * 1e6, 1)
                    out = ttnn.to_torch(ttnn.to_memory_config(out_ws, ttnn.DRAM_MEMORY_CONFIG)).reshape(args.rows, n)
                    outputs[workers] = out
                    for tensor in (out_ws, act_ws, act, weight):
                        ttnn.deallocate(tensor)
                reference = activation.float().reshape(args.rows, k) @ packed_reference
                err = {w: float((outputs[w].float() - reference).abs().max()) for w in (1, 2)}
                bitwise = bool(torch.equal(outputs[1].view(torch.int16), outputs[2].view(torch.int16)))
                row = {
                    "k": k,
                    "n": n,
                    "cores": cores,
                    "dtype": tag,
                    "fidelity": dm.dense_math_fidelity_name(dtype),
                    "bitwise_equal_1_vs_2_readers": bitwise,
                    "max_abs_err_vs_host_fp32_of_packed_weight": err,
                    "reference_abs_max": float(reference.abs().max()),
                    "eager_us_per_call_incl_dispatch": timings,
                }
                record["shapes"].append(row)
                print("QUALIFY", json.dumps(row), flush=True)
    finally:
        ttnn.close_device(device)
    by_dtype = {}
    for row in record["shapes"]:
        by_dtype.setdefault(row["dtype"], []).append(row["bitwise_equal_1_vs_2_readers"])
    record["qualified_dtypes"] = sorted(tag for tag, flags in by_dtype.items() if all(flags))
    record["unqualified_dtypes"] = sorted(tag for tag, flags in by_dtype.items() if not all(flags))
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print("QUALIFIED", record["qualified_dtypes"], "UNQUALIFIED", record["unqualified_dtypes"], flush=True)
    return 0 if not record["unqualified_dtypes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
