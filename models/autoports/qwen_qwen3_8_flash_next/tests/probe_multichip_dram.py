# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 local projection geometry search with actual rank-partitioned checkpoint weights.

Run serially with a bounded timeout and TT_METAL_TRACE_ALLOC_TRACKING=1.
--list-candidates is CPU-only. A failed constructor or device call saves its
error and ABORTS: retry a repaired candidate in a fresh admitted device job.
This is component evidence, not a decoder accuracy gate. Activation inputs are
normalized seeded BF16 values, not captured target-model activations; synthetic
accuracy misses alone must not reject a faster precision policy.
"""

import argparse
import csv
import gc
import itertools
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

ROLES = {
    "qsa": (3, ["self_attn." + p + "_proj.weight" for p in ("q", "k", "v")]),
    "wo": (3, ["self_attn.o_proj.weight"]),
    "gdn": (0, ["linear_attn.in_proj_" + p + ".weight" for p in ("z", "a", "b")]),
    "shared": (0, ["mlp.shared_expert." + p + "_proj.weight" for p in ("gate", "up")]),
    "gr": (0, ["attn_hyper_connection.block_inject_weight.weight"]),
}


def integers(value):
    return tuple(int(x) for x in value.split(","))


def host_operands(snapshot, role, batch, seed):
    layer, names = ROLES[role]
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    full = []
    for name in names:
        key = f"model.language_model.layers.{layer}.{name}"
        with safe_open(snapshot / index[key], framework="pt") as stream:
            full.append(stream.get_tensor(key).bfloat16())
    if role == "qsa":
        q, k, v = full
        # Four Q partitions; two KV heads, each shared by adjacent TP ranks.
        parts = [[q.chunk(4)[r], k.chunk(2)[r // 2], v.chunk(2)[r // 2]] for r in range(4)]
    elif role == "wo":
        parts = [[full[0].chunk(4, dim=1)[r]] for r in range(4)]
    else:
        parts = [[w.chunk(4, dim=0)[r] for w in full] for r in range(4)]
    assert len(parts) == 4 and all(len(rank) == len(names) for rank in parts)
    widths = [w.shape[0] for w in parts[0]]
    k = parts[0][0].shape[1]
    assert all([w.shape[0] for w in rank] == widths and all(w.shape[1] == k for w in rank) for rank in parts)
    generator = torch.Generator().manual_seed(seed)
    # WO consumes different local SDPA heads. Other projections consume the
    # replicated mixed/full-normalized input used by the actual decoder.
    x = torch.randn(4 if role == "wo" else 1, 1, batch, k, generator=generator)
    x = (x / x.square().mean(-1, keepdim=True).sqrt()).bfloat16()
    if x.shape[0] == 1:
        x = x.repeat(4, 1, 1, 1)
    native_dtype = "bfloat8_b" if role in ("qsa", "wo", "shared") else "bfloat16"
    return names, x, parts, native_dtype, role in ("qsa", "gdn")


def candidates(args, k, projections):
    packing = (False, True) if args.packing == "both" and projections > 1 else (args.packing == "packed",)
    if projections == 1:
        packing = (False,)
    for dtype, fidelity, packed, cores, activation_dtype in itertools.product(
        args.dtypes.split(","),
        args.fidelities.split(","),
        packing,
        args.cores,
        getattr(args, "activation_dtypes", "bfloat16").split(","),
    ):
        if k % (32 * cores):
            continue
        common = dict(
            dtype=dtype, fidelity=fidelity, packed=packed, cores=cores, native=False, activation_dtype=activation_dtype
        )
        if args.storage in ("all", "interleaved"):
            yield dict(common, storage="interleaved", readers=1, block=None)
        if args.storage in ("all", "dram"):
            shard_tiles = k // (32 * cores)
            k_tiles = k // 32
            divisors = (
                [n for n in range(1, k_tiles + 1) if k_tiles % n == 0]
                if args.blocks == "all"
                else integers(args.blocks)
            )
            for readers, block in itertools.product(args.readers, divisors):
                # Single-reader supports a fraction of a storage shard OR a
                # whole number of consecutive shards. Multicore pipeline also
                # supports ragged blocks (factory independently bounds K groups).
                # Do not miss large legal blocks merely because K is sharded.
                legal = block > 0 and (
                    readers >= 2 or (k_tiles % block == 0 and (shard_tiles % block == 0 or block % shard_tiles == 0))
                )
                if legal:
                    yield dict(common, storage="dram", readers=readers, block=block)


def make_group(t, mesh, x, parts, config):
    widths = [w.shape[0] for w in parts[0]]
    k, height = x.shape[-1], math.ceil(x.shape[-2] / 32) * 32
    # Native GDN packs exactly Z1536/A12/B12 with no intervening tile pad.
    # Candidate packing owns padding and trims each output at this boundary.
    padded = widths if config["native"] and config["packed"] else [math.ceil(w / 32) * 32 for w in widths]
    per_rank = []
    for rank in parts:
        matrices = [torch.nn.functional.pad(w.T.contiguous(), (0, n - w.shape[0])) for w, n in zip(rank, padded)]
        per_rank.append([torch.cat(matrices, dim=-1)] if config["packed"] else matrices)
    compute = t.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(t.MathFidelity, config["fidelity"]),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=not config["native"],
    )
    grid_size = mesh.compute_with_storage_grid_size()
    if config["cores"]:
        core_set = t.num_cores_to_corerangeset(config["cores"], grid_size, row_wise=True)

    def l1(width):
        return t.MemoryConfig(
            t.TensorMemoryLayout.WIDTH_SHARDED,
            t.BufferType.L1,
            t.ShardSpec(core_set, (height, width), t.ShardOrientation.ROW_MAJOR),
        )

    weights, specs = [], []
    for projection in range(len(per_rank[0])):
        matrices = [rank[projection] for rank in per_rank]
        n = matrices[0].shape[-1]
        spec = dict(memory_config=t.DRAM_MEMORY_CONFIG)
        memory = t.DRAM_MEMORY_CONFIG
        if config["storage"] == "dram":
            banks = mesh.dram_grid_size().x
            bank_grid = t.CoreRangeSet({t.CoreRange(t.CoreCoord(0, 0), t.CoreCoord(banks - 1, 0))})
            memory = t.MemoryConfig(
                t.TensorMemoryLayout.WIDTH_SHARDED,
                t.BufferType.DRAM,
                t.ShardSpec(bank_grid, (k, math.ceil(n / (32 * banks)) * 32), t.ShardOrientation.ROW_MAJOR),
            )
            out_width = math.ceil(n / (32 * config["cores"])) * 32
            spec = dict(
                memory_config=l1(out_width),
                program_config=t.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=config["block"],
                    per_core_M=height // 32,
                    per_core_N=out_width // 32,
                    fused_activation=None,
                    num_workers_per_dram_bank=config["readers"],
                    use_multicore_pipeline=config["readers"] >= 2,
                ),
            )
        elif not config["native"]:
            legal_x = [
                i for i in range(1, grid_size.x + 1) if config["cores"] % i == 0 and config["cores"] // i <= grid_size.y
            ]
            if not legal_x:
                raise ValueError("Selected interleaved core count cannot form a rectangular grid")
            spec["core_grid"] = t.CoreGrid(x=max(legal_x), y=config["cores"] // max(legal_x))
        # Mapping on batch dimension retains independent rank matrices even
        # when output width is not tile aligned (GDN's 1560, GR's one output).
        weights.append(
            t.from_torch(
                torch.stack(matrices).unsqueeze(1),
                device=mesh,
                dtype=getattr(t, config["dtype"]),
                layout=t.TILE_LAYOUT,
                memory_config=memory,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=0),
            )
        )
        specs.append(spec)

    def run():
        activation = t.typecast(x, t.bfloat8_b) if config.get("activation_dtype", "bfloat16") == "bfloat8_b" else x
        if config["storage"] == "dram":
            activation = t.to_memory_config(activation, l1(k // config["cores"]))
        # Tensor dtype metadata, not a host tensor read. This records the actual
        # operand passed to every matmul in the captured group.
        run.actual_activation_dtype = str(activation.dtype)
        values = []
        for weight, spec in zip(weights, specs):
            value = t.linear(activation, weight, dtype=t.bfloat16, compute_kernel_config=compute, **spec)
            if config["storage"] == "dram":
                value = t.to_memory_config(value, t.DRAM_MEMORY_CONFIG)
            values.append(value)
        if config["packed"]:
            result, offset = [], 0
            for width, physical in zip(widths, padded):
                result.append(values[0][..., offset : offset + width])
                offset += physical
            return result
        return [value[..., :width] for value, width in zip(values, widths)]

    return run


def measure(t, mesh, run, reference, repeats):
    from ttnn.tools import trace_allocation_tracker

    composer = t.ConcatMeshToTensor(mesh, dim=0)

    def read(values):
        return [t.to_torch(value, mesh_composer=composer).clone() for value in values]

    warm = run()
    eager = read(warm)
    del warm
    warm = run()
    t.synchronize_device(mesh)
    del warm
    trace, capturing = None, False
    try:
        trace = t.begin_trace_capture(mesh, cq_id=0)
        capturing = True
        outputs = run()
        t.end_trace_capture(mesh, trace, cq_id=0)
        capturing = False
        for value in outputs:
            trace_allocation_tracker.acknowledge_corruptible(value)
        times = []
        for _ in range(repeats + 2):
            start = time.perf_counter()
            t.execute_trace(mesh, trace, cq_id=0, blocking=True)
            times.append((time.perf_counter() - start) * 1000)
        first = read(outputs)
        t.execute_trace(mesh, trace, cq_id=0, blocking=True)
        actual = read(outputs)
        exact = all(torch.equal(a, e) and torch.equal(a, f) for a, e, f in zip(actual, eager, first))
        reference = actual if reference is None else reference

        def score(expected, value):
            a, b = expected.float().flatten(), value.float().flatten()
            if a.numel() < 2 or a.std() == 0 or b.std() == 0:
                return None
            result = torch.corrcoef(torch.stack((a, b)))[0, 1].item()
            return result if math.isfinite(result) else None

        per_rank = []
        for rank in range(4):
            e = torch.cat([v[rank].flatten() for v in reference])
            a = torch.cat([v[rank].flatten() for v in actual])
            per_rank.append(
                dict(
                    rank=rank,
                    whole_pcc=score(e, a),
                    max_abs=float((e.float() - a.float()).abs().max()),
                    projection_pcc=[score(e[rank], a[rank]) for e, a in zip(reference, actual)],
                )
            )
        all_expected, all_actual = torch.cat([v.flatten() for v in reference]), torch.cat([v.flatten() for v in actual])
        whole = score(all_expected, all_actual)
        finite = bool(torch.isfinite(all_actual).all())
        return (
            dict(
                status="measured",
                traced_ms=statistics.median(times[2:]),
                samples_ms=times[2:],
                whole_pcc=whole,
                per_rank=per_rank,
                replay_exact=exact,
                finite=finite,
                passes_component_bar=finite
                and exact
                and whole is not None
                and whole >= 0.995
                and all(row["whole_pcc"] is None or row["whole_pcc"] >= 0.995 for row in per_rank),
                actual_activation_dtype=run.actual_activation_dtype,
            ),
            actual,
        )
    finally:
        if trace is not None:
            if capturing:
                t.end_trace_capture(mesh, trace, cq_id=0)
            t.release_trace(mesh, trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--clean-timing", action="store_true")
    parser.add_argument("--seed", type=int, default=819)
    parser.add_argument("--dtypes", default="bfloat8_b")
    parser.add_argument(
        "--activation-dtypes",
        default="bfloat16",
        help="bfloat16,bfloat8_b; cast cost is included before activation sharding",
    )
    parser.add_argument("--fidelities", default="LoFi,HiFi2,HiFi4")
    parser.add_argument("--cores", type=integers, default=(8, 16))
    parser.add_argument("--readers", type=integers, default=(1, 2, 3))
    parser.add_argument(
        "--blocks",
        default="all",
        help="all legal tiled-K divisors, including multi-shard blocks; or comma-separated integers",
    )
    parser.add_argument("--packing", choices=("both", "packed", "split"), default="both")
    parser.add_argument("--storage", choices=("all", "dram", "interleaved"), default="dram")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=1, help="Bound each admitted device job; 0 runs all selected candidates"
    )
    parser.add_argument("--list-candidates", action="store_true", help="CPU-only candidate table, no TTNN import")
    args = parser.parse_args()
    if args.repeats < 2 or args.start_index < 0 or args.limit < 0:
        parser.error("Need repeats >= 2 and nonnegative start-index/limit")
    if any(n <= 0 for n in (*args.cores, *args.readers)):
        parser.error("cores/readers must be positive")
    if any(n not in ("bfloat16", "bfloat8_b", "bfloat4_b") for n in args.dtypes.split(",")):
        parser.error("Unsupported dtype")
    if any(n not in ("bfloat16", "bfloat8_b") for n in args.activation_dtypes.split(",")):
        parser.error("Unsupported activation dtype")
    if any(n not in ("LoFi", "HiFi2", "HiFi4") for n in args.fidelities.split(",")):
        parser.error("Unsupported fidelity")
    torch.set_num_threads(8)
    names, host_x, parts, dtype, packed = host_operands(args.snapshot, args.role, args.batch, args.seed)
    configs = [dict(index=i, **config) for i, config in enumerate(candidates(args, host_x.shape[-1], len(names)))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.list_candidates:
        args.output.write_text(json.dumps(configs, indent=2) + "\n")
        print(json.dumps(dict(candidates=len(configs), output=str(args.output))))
        return
    if not args.clean_timing and os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Set TT_METAL_TRACE_ALLOC_TRACKING=1")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "1":
        parser.error("Program-cache allocations must be tracked")
    selected = configs[args.start_index : args.start_index + args.limit] if args.limit else configs[args.start_index :]
    if not selected:
        parser.error("No selected legal candidates")
    report = dict(
        role=args.role,
        names=names,
        snapshot=str(args.snapshot.resolve()),
        mesh=[1, 4],
        batch=args.batch,
        local_weight_shapes=[[list(w.shape) for w in rank] for rank in parts],
        partition_source="tt/multichip_decoder.py: MeshQSA, MeshGDN, MeshMoe, MeshResidual",
        activation_provenance=f"seed {args.seed}, normalized BF16 random; WO rank-diverse, others replicated",
        reference="Identical per-rank native default TTNN projection; same real weights and input",
        timing="serial warmed blocking trace; includes optional activation cast, input/output reshard and unpack/slicing; no CCL",
        activation_policies=args.activation_dtypes.split(","),
        pcc_note="Whole-output and per-rank PCC; singleton GR B1 per-rank PCC undefined, global PCC reported",
        trace_alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
        candidate_count=len(configs),
        selected_indices=[c["index"] for c in selected],
        rows=[],
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        columns = [
            "index",
            "native",
            "dtype",
            "activation_dtype",
            "actual_activation_dtype",
            "fidelity",
            "packed",
            "storage",
            "cores",
            "readers",
            "block",
            "status",
            "traced_ms",
            "speedup_vs_native",
            "whole_pcc",
            "replay_exact",
            "passes_component_bar",
        ]
        with args.output.with_suffix(".csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["rows"])

    import ttnn as t

    save()
    t.set_fabric_config(t.FabricConfig.FABRIC_1D_RING)
    mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
    try:
        report.update(arch=str(mesh.arch()), dram_banks=mesh.dram_grid_size().x)
        x = t.from_torch(
            host_x,
            dtype=t.bfloat16,
            layout=t.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=t.ShardTensorToMesh(mesh, dim=0),
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        native = dict(
            index=-1,
            native=True,
            dtype=dtype,
            activation_dtype="bfloat16",
            fidelity="HiFi4",
            packed=packed,
            cores=0,
            storage="interleaved",
            readers=1,
            block=None,
        )
        baseline, native_ms = None, None
        for config in [native, *selected]:
            row = dict(config)
            run = None
            try:
                run = make_group(t, mesh, x, parts, config)
                measured, actual = measure(t, mesh, run, baseline, args.repeats)
                row.update(measured)
                if config["native"]:
                    baseline, native_ms = actual, measured["traced_ms"]
                row["speedup_vs_native"] = native_ms / row["traced_ms"]
            except Exception as error:
                row.update(
                    status="blocker_requires_fresh_job_and_investigation",
                    error=str(error),
                    traceback=traceback.format_exc(),
                )
                raise
            finally:
                run = None
                gc.collect()
                report["rows"].append(row)
                save()
            print(json.dumps({k: v for k, v in row.items() if k not in ("samples_ms", "per_rank")}), flush=True)
    finally:
        save()
        t.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
