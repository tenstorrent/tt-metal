# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public logical prompt boundaries, with the same persistent max-context cache."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

p = argparse.ArgumentParser()
p.add_argument("--full", action="store_true")
p.add_argument("--lengths", default="1,31,32,33,4095,4096,4097,33,31,4097")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
gen = None
try:
    gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=None if a.full else [0, 3])
    gen._ensure_cache(1, gen.model.context)
    report = dict(full=a.full, context=gen.model.context, rows=[])
    for length in map(int, a.lengths.split(",")):
        previous_captures = gen.counters["trace_captures"]
        known = any(row["length"] == length for row in report["rows"])
        print("LENGTH_BEGIN", length, flush=True)
        start = time.perf_counter()
        count = min(3, gen.model.context - length + 1)
        if count:
            result = gen.generate([1596] * length, count)
            assert len(result) == count
            position = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).item()
            assert position == length + count - 1, (length, position)
        else:
            gen.reset()
            result = gen.prefill_forward(
                torch.full((1, length), 1596), page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=[length]
            )
            del result
        if known:
            assert gen.counters["trace_captures"] == previous_captures
        report["rows"].append(
            dict(length=length, seconds=time.perf_counter() - start, passed=True, decode_steps=max(0, count - 1))
        )
        a.output.write_text(json.dumps(report, indent=2) + "\n")
        print("LENGTH_PASS", length, report["rows"][-1], flush=True)
finally:
    if gen is not None:
        gen.close()
    ttnn.close_mesh_device(mesh)
