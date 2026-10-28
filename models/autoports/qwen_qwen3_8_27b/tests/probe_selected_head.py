# SPDX-License-Identifier: Apache-2.0
"""Same-selected-BFP8 LM-head geometry probe on real reduced-model hidden states.

No whole-model performance or quality claim is made by this diagnostic.
"""
import argparse
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

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--chunk", type=int, default=16384)
p.add_argument("--readers", type=int, default=1)
p.add_argument("--block", type=int, default=5)
p.add_argument("--cores", type=int, default=8)
p.add_argument("--replays", type=int, default=100)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
ttnn.CONFIG.throw_exception_on_fallback = True
root = Path(__file__).resolve().parents[1]
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
result = dict(
    status="running",
    scope="Selected BFP8/HiFi2 head only on real layers0/3 hidden; not full64 TSU or quality",
    config=vars(a) | {"output": str(a.output)},
    tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
    skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
)
traces = {}
try:
    gen = build_generator(
        root,
        mesh,
        checkpoint=a.checkpoint,
        layer_indices=[0, 3],
        batch_size=a.batch,
        context_length=512,
        prepare=False,
        prefill_chunk_size=256,
    )
    model = gen.model
    assert model.head.dtype == ttnn.bfloat8_b
    assert model.head_ckc.math_fidelity == ttnn.MathFidelity.HiFi2
    observed = []
    terminal = model.terminal

    def save(hidden, **kw):
        observed.append(hidden)
        return terminal(hidden, **kw)

    model.terminal = save
    gen.prefill_forward(
        torch.full((a.batch, 33), 100, dtype=torch.int64),
        page_table=gen.page_table,
        kv_cache=gen.kv_cache,
        prompt_lens=[33] * a.batch,
        return_device=True,
    )
    model.terminal = terminal
    hidden = observed[-1]
    if hidden.shape[-1] == 1280:
        hidden = model.layers[0]._gather(hidden, ("terminal",), compact_output=a.batch > 1)
    hidden = ttnn.reshape(hidden, [1, 1, a.batch, 5120])
    hidden = ttnn.rms_norm(
        ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG),
        weight=model.final_norm,
        epsilon=model.config.rms_norm_eps,
        compute_kernel_config=model.layers[0].ckc,
    )
    # Every candidate is built before capture. No dtype/fidelity change.
    k = 5120
    local = model.config.vocab_size // 4
    raw = model.checkpoint.tensor("lm_head.weight").T.contiguous()
    dg = mesh.dram_grid_size()
    dram = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))])
    grid = ttnn.num_cores_to_corerangeset(a.cores, mesh.compute_with_storage_grid_size(), row_wise=True)
    assert k % (32 * a.cores) == 0 and (k // (32 * a.cores)) % a.block == 0
    imem = ttnn.create_sharded_memory_config(
        [32, k // a.cores],
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    weights = []
    for offset in range(0, local, a.chunk):
        logical = min(a.chunk, local - offset)
        physical = math.ceil(logical / (32 * dg.x * a.readers)) * 32 * dg.x * a.readers
        pieces = [
            torch.nn.functional.pad(raw[:, i * local + offset : i * local + offset + logical], (0, physical - logical))
            for i in range(4)
        ]
        wm = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram, [k, physical // dg.x], ttnn.ShardOrientation.ROW_MAJOR),
        )
        weight = ttnn.from_torch(
            torch.cat(pieces, dim=-1).contiguous(),
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wm,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=a.block,
            per_core_M=1,
            per_core_N=physical // (32 * dg.x * a.readers),
            num_workers_per_dram_bank=a.readers,
        )
        weights.append((weight, pc, logical))
    del raw, pieces

    def baseline():
        return ttnn.linear(
            hidden,
            model.head,
            compute_kernel_config=model.head_ckc,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def candidate():
        x = ttnn.to_memory_config(hidden, imem)
        outputs = []
        for w, pc, n in weights:
            y = ttnn.linear(
                x,
                w,
                program_config=pc,
                compute_kernel_config=model.head_ckc,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            y = ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)
            outputs.append(y if y.shape[-1] == n else y[..., :n])
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    variants = {"baseline": baseline, "candidate": candidate}
    outputs = {name: model.put(torch.zeros(1, 1, a.batch, local)) for name in variants}

    def run(name):
        ttnn.copy(variants[name](), outputs[name])

    for name in variants:
        run(name)
    ttnn.synchronize_device(mesh)

    def compare():
        pairs = [
            (ttnn.to_torch(l).float(), ttnn.to_torch(r).float())
            for l, r in zip(ttnn.get_device_tensors(outputs["baseline"]), ttnn.get_device_tensors(outputs["candidate"]))
        ]
        return [
            dict(
                pcc=float(torch.corrcoef(torch.stack([l.flatten(), r.flatten()]))[0, 1]),
                exact=bool(torch.equal(l, r)),
                max_abs=float((l - r).abs().max()),
            )
            for l, r in pairs
        ]

    result["before_replay"] = compare()
    assert all(row["pcc"] > 0.999 for row in result["before_replay"])
    for name in variants:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        run(name)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        traces[name] = trace
    times = {name: [] for name in variants}
    for order in [("baseline", "candidate"), ("candidate", "baseline"), ("baseline", "candidate")]:
        for name in order:
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=True)
            started = time.monotonic()
            for _ in range(a.replays):
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            times[name].append((time.monotonic() - started) * 1e6 / a.replays)
    result["after_replay"] = compare()
    assert all(row["pcc"] > 0.999 for row in result["after_replay"])
    result.update(
        status="pass",
        trace_us=times,
        speed_ratio=statistics.median(times["baseline"]) / statistics.median(times["candidate"]),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
except BaseException as exc:
    result.update(status="fail", error=repr(exc))
    raise
finally:
    for trace in traces.values():
        ttnn.release_trace(mesh, trace)
    ttnn.close_mesh_device(mesh)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
