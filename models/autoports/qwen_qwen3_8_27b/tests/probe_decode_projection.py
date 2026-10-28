# SPDX-License-Identifier: Apache-2.0
"""Actual selected TP4 decode projection input/layout/config comparison.

Includes original helper's input reshards/output placement. Reduced two real
layers only; acceptance requires whole64 model measurements and quality gates.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--role", choices=["mix", "gateup", "out", "down"], default="gateup")
p.add_argument("--cores", type=int, default=8)
p.add_argument("--block", type=int, default=20)
p.add_argument("--readers", type=int)
p.add_argument("--tile-height", type=int, choices=[16, 32], default=32)
p.add_argument("--dma-retile", action="store_true")
p.add_argument("--replays", type=int, default=100)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
ttnn.CONFIG.throw_exception_on_fallback = True
root = Path(__file__).resolve().parents[1]
original = OptimizedDecoder._linear
captured = {}
result = dict(
    status="running",
    scope="Actual real layer0/3 B1 decode projection helpers; not full64 performance/quality",
    role=a.role,
    cores=a.cores,
    block=a.block,
    readers=a.readers,
    tile_height=a.tile_height,
    tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
    skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
    cases=[],
)
traces = []


def observed(self, x, w, activation=None, **kw):
    role = self.weight_roles[id(w)]
    if role == a.role and not self.policy.get("_prefill_phase"):
        captured[self.layer_idx] = (self, x, w, activation, kw)
    return original(self, x, w, activation, **kw)


def tiny_projection(owner, x, w, activation, kw, target_memory):
    assert activation is None and a.tile_height == 16
    cfg = owner.policy[a.role]
    k = x.shape[-1]
    grid = ttnn.num_cores_to_corerangeset(a.cores, mesh.compute_with_storage_grid_size(), row_wise=True)
    memory = ttnn.create_sharded_memory_config(
        [16, k // a.cores],
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    if a.dma_retile:
        from models.autoports.qwen_qwen3_8_27b.tests.b1_tile_conversion import convert

        x = convert(ttnn.reshape(x, [1, 1, 1, k]), 16, memory)
    else:
        x = ttnn.to_memory_config(ttnn.reshape(x, [1, 1, 1, k]), ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.tilize(x, tile=ttnn.Tile([16, 32]), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_memory_config(x, memory)
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=a.block,
        per_core_M=1,
        per_core_N=math.ceil(w.shape[-1] / (32 * a.cores)),
        num_workers_per_dram_bank=cfg["readers"],
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, cfg.get("fidelity", owner.policy["fidelity"])),
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    y = ttnn.linear(
        x,
        w,
        program_config=pc,
        compute_kernel_config=ckc,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        output_tile=ttnn.Tile([16, 32]),
    )
    if a.dma_retile:
        y = convert(y, 32, target_memory)
    else:
        y = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
        y = ttnn.tilize(y, tile=ttnn.Tile([32, 32]), memory_config=ttnn.L1_MEMORY_CONFIG)
    logical = owner.logical_widths[id(w)]
    if y.shape[-1] != logical:
        y = y[..., :logical]
    y = ttnn.to_memory_config(y, target_memory)
    return ttnn.reshape(y, [1, 1, logical])


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
try:
    gen = build_generator(
        root,
        mesh,
        checkpoint=a.checkpoint,
        layer_indices=[0, 3],
        batch_size=1,
        context_length=512,
        prefill_chunk_size=128,
        prepare=False,
    )
    logits = gen.prefill_forward(
        torch.full((1, 33), 100, dtype=torch.int64),
        page_table=gen.page_table,
        kv_cache=gen.kv_cache,
        prompt_lens=[33],
        return_device=True,
    )
    sampled = gen.host_logits(logits).reshape(-1).argmax().to(torch.int32).reshape(1, 1, 1, 1)
    gen.refresh(sampled, gen.tokens)
    gen.refresh(torch.tensor([33], dtype=torch.int32), gen.positions)
    gen.refresh(torch.tensor([[33]], dtype=torch.int32), gen.rope_indices)
    OptimizedDecoder._linear = observed
    try:
        gen._decode()
    finally:
        OptimizedDecoder._linear = original
    assert set(captured) == {0, 3}, captured.keys()
    calls = {}
    outputs = {}
    for layer, (owner, x, w, activation, kw) in sorted(captured.items()):
        view = copy.copy(owner)
        view.policy = copy.deepcopy(owner.policy)
        view.policy[a.role]["cores"] = a.cores
        view.policy[a.role]["block"] = a.block
        if a.readers is not None:
            view.policy[a.role]["readers"] = a.readers
        for arm, obj in [("baseline", owner), ("candidate", view)]:
            key = f"{layer}:{arm}"
            if arm == "candidate" and a.tile_height == 16:
                target = outputs[f"{layer}:baseline"].memory_config()
                calls[key] = lambda obj=obj, x=x, w=w, activation=activation, kw=kw, target=target: tiny_projection(
                    obj, x, w, activation, kw, target
                )
            else:
                calls[key] = lambda obj=obj, x=x, w=w, activation=activation, kw=kw: original(
                    obj, x, w, activation, **kw
                )
            y = calls[key]()
            outputs[key] = ttnn.zeros_like(y)
            del y
        result["cases"].append(
            dict(
                layer=layer,
                input_shape=list(x.shape),
                input_memory=str(x.memory_config()),
                weight_shape=list(w.shape),
                weight_dtype=str(w.dtype),
                baseline_config=owner.policy[a.role],
                candidate_config=view.policy[a.role],
                tile_scope="32→16→32 compatibility boundary costs included; does not reject a carried-tiny MLP topology"
                if a.tile_height == 16
                else "Selected32-row tiles",
            )
        )

    def run(key):
        ttnn.copy(calls[key](), outputs[key])

    for key in calls:
        run(key)
    ttnn.synchronize_device(mesh)

    def compare(layer):
        pairs = zip(
            ttnn.get_device_tensors(outputs[f"{layer}:baseline"]),
            ttnn.get_device_tensors(outputs[f"{layer}:candidate"]),
        )
        result = []
        for l, r in pairs:
            l, r = ttnn.to_torch(l).float(), ttnn.to_torch(r).float()
            result.append(
                dict(
                    exact=bool(torch.equal(l, r)),
                    pcc=float(torch.corrcoef(torch.stack([l.flatten(), r.flatten()]))[0, 1]),
                    max_abs=float((l - r).abs().max()),
                )
            )
        return result

    for row in result["cases"]:
        row["before_replay"] = compare(row["layer"])
        assert all(r["pcc"] > 0.999 for r in row["before_replay"]), row
    for key in calls:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        run(key)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        traces.append(trace)
    trace_map = dict(zip(calls, traces))
    times = {key: [] for key in calls}
    for order in [list(calls), list(reversed(calls)), list(calls)]:
        for key in order:
            ttnn.execute_trace(mesh, trace_map[key], cq_id=0, blocking=True)
            started = time.monotonic()
            for _ in range(a.replays):
                ttnn.execute_trace(mesh, trace_map[key], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            times[key].append((time.monotonic() - started) * 1e6 / a.replays)
    for row in result["cases"]:
        layer = row["layer"]
        row["trace_us"] = {arm: times[f"{layer}:{arm}"] for arm in ["baseline", "candidate"]}
        row["speed_ratio"] = statistics.median(row["trace_us"]["baseline"]) / statistics.median(
            row["trace_us"]["candidate"]
        )
        row["after_replay"] = compare(layer)
        assert all(r["pcc"] > 0.999 for r in row["after_replay"]), row
    result.update(status="pass", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
except BaseException as exc:
    result.update(status="fail", error=repr(exc))
    raise
finally:
    OptimizedDecoder._linear = original
    for trace in traces:
        ttnn.release_trace(mesh, trace)
    ttnn.close_mesh_device(mesh)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
