# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight WO -> residual update -> next GR: coherent native/fused CCL probes.

Device use belongs to the serialized stage owner. Run this module under timeout.
QSA layer3 is the default native projection target. GDN layer0 is supported but
its native projection is a diagnostic replacement for the shipping exact tree.
Activations are seeded features, not recorded attention outputs; this screening
probe cannot accept or reject a precision policy on behalf of whole-model gates.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_residual_contract import (
    HiddenResidual,
    ProbeCollectives,
    compare,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshLoader, MeshResidual

DRAM = ttnn.DRAM_MEMORY_CONFIG
CANDIDATES = (
    "ordered_bf16",
    "ordered_fp32",
    "rs_fp32",
    "fused_rs_fp32",
    "ag_mm_fp32",
    "fused_ag_mm_fp32",
)


class ProjectionFamilies:
    """All producer buffers and semaphores are allocated once, before execution.

    One CQ serializes calls. Each persistent output is consumed completely before
    the same family executes again. Optional screening captures retain at most
    one live trace; it is released before another candidate executes. Captured
    inputs stay unchanged during replay, so this is not a changed-input gate.
    RS and AG own separate buffers; neither aliases residual/next-GR storage.
    """

    def __init__(self, mesh, loader, weight, args):
        self.mesh, self.args = mesh, args
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, args.math_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        weight_dtype = getattr(ttnn, args.weight_dtype)
        self.row_weight = loader.shards(
            [part.T.reshape(1, 1, 1536, 2560) for part in weight.chunk(4, dim=1)],
            dim=2,
            dtype=weight_dtype,
        )
        self.column_weight = loader.shards(
            [part.T.reshape(1, 1, 6144, 640) for part in weight.chunk(4, dim=0)],
            dim=3,
            dtype=weight_dtype,
        )
        grid = mesh.compute_with_storage_grid_size()
        if grid.x < 8 or grid.y < 10:
            raise ValueError("The initial configs require at least an8x10 worker grid")
        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        mm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        self.rs_program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=args.in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=10,
            out_block_w=args.rs_out_block_w,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            allowed_worker_cores=mm_cores,
        )
        self.ag_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(5, 4),
            in0_block_w=args.in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.rs_sems = [ttnn.create_global_semaphore(mesh, all_cores, 0) for _ in range(3)]
        self.rs_barrier = ttnn.create_global_semaphore(mesh, all_cores, 0)
        self.ag_sems = [ttnn.create_global_semaphore(mesh, all_cores, 0) for _ in range(2)]
        self.ag_barrier = ttnn.create_global_semaphore(mesh, all_cores, 0)
        self.rs_intermediate = self.buffer(2560, ttnn.float32)
        self.rs_output = self.buffer(640, ttnn.float32)
        self.ag_output = self.buffer(6144, getattr(ttnn, args.activation_dtype))
        self.ccl = ProbeCollectives(mesh, "gather_fp32")

    def buffer(self, width, dtype):
        return ttnn.from_torch(
            torch.zeros(1, 1, 32, width),
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def __call__(self, name, features):
        if name in ("ordered_bf16", "ordered_fp32", "rs_fp32"):
            partial = ttnn.matmul(
                features,
                self.row_weight,
                dtype=ttnn.bfloat16 if name == "ordered_bf16" else ttnn.float32,
                memory_config=DRAM,
                program_config=self.rs_program,
                compute_kernel_config=self.compute,
            )
            if name == "rs_fp32":
                return ttnn.reduce_scatter(
                    partial,
                    dim=3,
                    cluster_axis=1,
                    memory_config=DRAM,
                    num_links=self.args.num_links,
                    topology=ttnn.Topology.Ring,
                )
            return self.ccl.sum(partial)
        if name == "fused_rs_fp32":
            _, reduced = ttnn.experimental.matmul_reduce_scatter_async(
                features,
                self.row_weight,
                persistent_intermediate_buffer=self.rs_intermediate,
                persistent_output_buffer=self.rs_output,
                dim=3,
                multi_device_global_semaphore=self.rs_sems,
                reduce_scatter_core_grid_offset=(0, 8),
                barrier_semaphore=self.rs_barrier,
                num_links=self.args.num_links,
                memory_config_rs=DRAM,
                memory_config_mm=DRAM,
                topology=ttnn.Topology.Ring,
                dtype=ttnn.float32,
                program_config=self.rs_program,
                compute_kernel_config=self.compute,
            )
            return reduced
        gather_kwargs = dict(
            persistent_output_buffer=self.ag_output,
            dim=3,
            multi_device_global_semaphore=self.ag_sems,
            barrier_semaphore=self.ag_barrier,
            num_links=self.args.num_links,
            topology=ttnn.Topology.Ring,
        )
        if name == "ag_mm_fp32":
            gathered = ttnn.experimental.all_gather_async(features, memory_config=DRAM, **gather_kwargs)
            return ttnn.matmul(
                gathered,
                self.column_weight,
                dtype=ttnn.float32,
                memory_config=DRAM,
                program_config=self.ag_program,
                compute_kernel_config=self.compute,
            )
        if name == "fused_ag_mm_fp32":
            _, output = ttnn.experimental.all_gather_matmul_async(
                features,
                self.column_weight,
                all_gather_core_grid_offset=(0, 6),
                memory_config_ag=DRAM,
                memory_config_mm=DRAM,
                dtype=ttnn.float32,
                program_config=self.ag_program,
                compute_kernel_config=self.compute,
                **gather_kwargs,
            )
            return output
        raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--layer", type=int, choices=(0, 3), default=3)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 31, 32])
    parser.add_argument("--candidates", choices=CANDIDATES, nargs="+", default=list(CANDIDATES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--changed-inputs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2071)
    parser.add_argument("--weight-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat8_b")
    parser.add_argument("--activation-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--math-fidelity", choices=("HiFi4", "HiFi2", "LoFi"), default="HiFi4")
    parser.add_argument("--in0-block-w", type=int, default=4)
    parser.add_argument("--rs-out-block-w", type=int, choices=(1, 2, 5, 10), default=5)
    parser.add_argument("--num-links", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if any(n < 1 or n > 32 for n in args.lengths) or args.repeats < 1 or args.changed_inputs < 2:
        parser.error("Lengths must be1..32, repeats positive, changed-inputs at least2")
    if args.in0_block_w < 1 or 48 % args.in0_block_w:
        parser.error("in0-block-w must be a positive divisor of local K/32=48")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    role = "self_attn.o_proj." if args.layer == 3 else "linear_attn.out_proj."
    weights = {}
    for prefix in ("attn_hyper_connection.", "mlp_hyper_connection.", role):
        weights.update(load_weights(args.snapshot, args.layer, prefix))
    maximum = max(args.lengths)
    if args.reference:
        saved = torch.load(args.reference, map_location="cpu", weights_only=False)
        if saved.get("metadata", {}).get("layer") != args.layer:
            raise ValueError("Reference layer does not match requested layer")
        host_residual = saved["inputs"][0][:, :, :maximum, :].contiguous()
        if tuple(host_residual.shape) != (1, 1, maximum, 10240):
            raise ValueError("Reference does not have enough residual rows")
    else:
        host_residual = (torch.randn(1, 1, maximum, 10240) * 0.02).bfloat16()
    feature_sets = [(torch.randn(1, 1, maximum, 6144) * 0.02).bfloat16() for _ in range(args.changed_inputs)]
    report = dict(
        passed=False,
        argv=sys.argv,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        snapshot=str(args.snapshot.resolve()),
        layer=args.layer,
        mesh=[1, 4],
        regime="serial",
        seed=args.seed,
        scope="WO producer + residual write + entire next MLP GR read; eager warmed and optional same-input trace",
        limitations=[
            "Seeded producer activations, real target weights; not whole-decoder acceptance",
            "Optional trace component check; no whole-decoder watcher, profiler or context acceptance claimed",
            "Trace replays unchanged input with one live trace; not a changed-input or multi-trace acceptance gate",
            "GDN native producer differs from shipping exact GDN if layer0 selected",
            "Boundary host reconstruction and first GR injection excluded from timing",
            "Both row/column weight representations coexist for this one-layer experiment only",
        ],
        policy=dict(
            weight=args.weight_dtype,
            activation=args.activation_dtype,
            math_fidelity=args.math_fidelity,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            output="FP32 except ordered_bf16 control",
        ),
        configs=dict(
            rs_grid=[8, 8],
            ag_grid=[5, 4],
            rs_ccl_offset=[0, 8],
            ag_ccl_offset=[0, 6],
            in0_block_w=args.in0_block_w,
            rs_per_core_n=10,
            ag_per_core_n=1,
            rs_out_block_w=args.rs_out_block_w,
            subblock=[1, 1],
            num_links=args.num_links,
        ),
        contracts=dict(
            input="local[1,1,32,1536], logical tails padded internally with zero",
            row_weight="local[1,1,1536,2560]",
            column_weight="local[1,1,6144,640]",
            residual_control="BF16[1,1,T,2560], complete stream/rank",
            residual_hidden="BF16[1,4,T,640], four hidden-sliced streams/rank",
            next_consumer="hidden residual add, distributed per-stream norm, normalized gather for next exact GR",
            boundary="No full block gather before hidden residual write; host gather outside timing",
            memory="TILE DRAM-interleaved throughout",
        ),
        persistence=dict(
            rs_intermediate_bytes=32 * 2560 * 4,
            rs_output_bytes=32 * 640 * 4,
            ag_output_bytes=32 * 6144 * (4 if args.activation_dtype == "float32" else 2),
            rs_semaphores=3,
            rs_barriers=1,
            ag_semaphores=2,
            ag_barriers=1,
            ownership="Probe lifetime, serial CQ0, consumed before reuse; at most one live trace; no alias with next GR",
            native_rs="Framework output allocation; fused RS owns intermediate/output; AG owns gathered input",
        ),
        rows=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
        loader = MeshLoader(weights, mesh)
        ccl = ProbeCollectives(mesh, "gather_fp32")
        first = MeshResidual(loader, "attn_hyper_connection.", config, ccl)
        stream = MeshResidual(loader, "mlp_hyper_connection.", config, ccl)
        hidden = HiddenResidual(loader, "mlp_hyper_connection.", config, ccl)
        producer = ProjectionFamilies(mesh, loader, weights[role + "weight"], args)

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )

        def read(outputs, is_hidden):
            residual, mixed, injection = outputs
            residual = ttnn.to_torch(residual, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
            if is_hidden:
                residual = residual.permute(0, 2, 1, 3).reshape(1, 1, residual.shape[-2], 10240)
            mixed = ttnn.to_torch(mixed, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1]
            injection = ttnn.to_torch(injection, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0 if is_hidden else 1))
            if is_hidden:
                injection = injection[:1]
            return {"residual": residual.clone(), "mixed": mixed.clone(), "injection": injection.clone()}

        for n in args.lengths:
            residual_host = host_residual[:, :, :n, :]
            residual = upload(residual_host)
            fractured = upload(residual_host.reshape(1, n, 4, 2560).permute(0, 2, 1, 3))
            _, injection = first.read(residual)
            injection_all = ccl.gather(injection, dim=1)
            inputs = []
            for features in feature_sets:
                padded = torch.nn.functional.pad(features[:, :, :n, :], (0, 0, 0, 32 - n))
                inputs.append(upload(padded, getattr(ttnn, args.activation_dtype)))

            def run(candidate, features):
                block = ttnn.typecast(producer(candidate, features)[..., :n, :], ttnn.bfloat16)
                if candidate.startswith("ordered_"):
                    updated = stream.write(residual, block, injection)
                    mixed, next_injection = stream.read(updated)
                else:
                    updated = hidden.write(fractured, block, injection_all)
                    mixed, next_injection = hidden.read(updated)
                return updated, mixed, next_injection

            expected = [read(run("ordered_bf16", features), False) for features in inputs]
            precise = [read(run("ordered_fp32", features), False) for features in inputs]
            for candidate in dict.fromkeys(["ordered_bf16", "ordered_fp32", *args.candidates]):
                report["active_case"] = dict(length=n, candidate=candidate)
                save()
                comparisons = []
                own_eager = None
                for index, features in enumerate(inputs):
                    actual = read(run(candidate, features), not candidate.startswith("ordered_"))
                    if index == 0:
                        own_eager = actual
                    comparisons.append(
                        dict(
                            input=index,
                            vs_bf16={key: compare(expected[index][key], actual[key]) for key in actual},
                            vs_fp32={key: compare(precise[index][key], actual[key]) for key in actual},
                        )
                    )
                warm = run(candidate, inputs[0])
                ttnn.synchronize_device(mesh)
                del warm
                samples = []
                for iteration in range(args.repeats):
                    start = time.perf_counter()
                    output = run(candidate, inputs[iteration % len(inputs)])
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - start) * 1000)
                    del output
                passed = all(
                    metric["finite"] and metric["pcc"] is not None and metric["pcc"] >= 0.995
                    for comparison in comparisons
                    for metric in comparison["vs_bf16"].values()
                )
                row = dict(
                    length=n,
                    candidate=candidate,
                    comparisons=comparisons,
                    samples_ms=samples,
                    median_ms=statistics.median(samples),
                    passed=passed,
                )
                if args.trace:
                    from ttnn.tools import trace_allocation_tracker

                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    outputs = run(candidate, inputs[0])
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    # These three capture-produced outputs are read before the
                    # next candidate can overwrite them. No other trace is live.
                    for value in outputs:
                        trace_allocation_tracker.acknowledge_corruptible(value)
                    traced = []
                    try:
                        for _ in range(args.repeats + 2):
                            start = time.perf_counter()
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                            traced.append((time.perf_counter() - start) * 1000)
                        actual = read(outputs, not candidate.startswith("ordered_"))
                        parity = {key: compare(expected[0][key], actual[key]) for key in actual}
                        own_eager_exact = {key: torch.equal(own_eager[key], actual[key]) for key in actual}
                    finally:
                        ttnn.release_trace(mesh, trace)
                    row.update(
                        traced_ms=statistics.median(traced[2:]),
                        trace_samples_ms=traced[2:],
                        trace_pcc=parity,
                        trace_own_eager_exact=own_eager_exact,
                        trace_changed_input=False,
                        max_live_traces=1,
                        trace_scope="Same-input component screening; not decoder acceptance",
                        trace_tracking=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
                    )
                    row["passed"] &= all(
                        value["finite"] and value["pcc"] is not None and value["pcc"] >= 0.995
                        for value in parity.values()
                    )
                    row["passed"] &= all(own_eager_exact.values())
                    del outputs
                report["rows"].append(row)
                save()
                print(json.dumps({key: row[key] for key in ("length", "candidate", "median_ms", "passed")}), flush=True)
        report.pop("active_case", None)
        report["passed"] = all(row["passed"] for row in report["rows"])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
