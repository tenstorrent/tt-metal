# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare complete explicit and fused-tail normalization adapters, via traces.

Hardware is opt-in. Both variants include core BF16 rounding, FP32 square/sum,
unchanged SiLU lookup, every placement copy, and the current head concatenation.
Only setup/readback and the subsequent output projection are outside timing.
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_gated_norm_tail_fusion_probe as helper
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

E = helper.E
EXPECTED_HELPER_SHA256 = "47d50c370b8038bc1b10f24f86c5e58f61222e21b6cc7a1bb22d6318055961bd"


class CompleteAdapter:
    """Identical setup and output contract for both measured device paths."""

    def __init__(self, t, mesh, gamma):
        self.t, self.mesh = t, mesh
        self.mem = t.L1_MEMORY_CONFIG
        self.compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.fused = helper.GatedNormTail(t, mesh, gamma, debug=False)
        self.weight = t.from_torch(
            gamma.float().reshape(1, 1, 128),
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            device=mesh,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        table = torch.nn.functional.silu(codes.float()).reshape(1, 65536)
        self.table = t.from_torch(
            table, dtype=t.float32, layout=t.ROW_MAJOR_LAYOUT, device=mesh, memory_config=t.DRAM_MEMORY_CONFIG
        )
        self.round_chain = [
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value)),
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value)),
        ]
        self.head_shard = t.create_sharded_memory_config(
            (32, 128),
            t.CoreGrid(y=1, x=1),
            t.ShardStrategy.HEIGHT,
            t.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def __call__(self, core, z, variant):
        t, mem = self.t, self.mem
        assert tuple(core.shape) == (1, 1, 32, 128) and core.dtype == t.float32
        assert tuple(z.shape) == (1, 1, 4096) and z.dtype == t.bfloat16
        core_bf16 = t.typecast(core, t.bfloat16)
        core_rows = t.reshape(core_bf16, (1, 32, 128))
        wide = t.typecast(core_rows, t.float32, memory_config=mem)
        squares = t.multiply(wide, wide, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem)
        sums = t.sum(squares, dim=-1, keepdim=True, compute_kernel_config=self.compute, memory_config=mem)
        if variant == "explicit":
            variance = t.multiply(sums, 1 / 128, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem)
            shifted = t.add(variance, helper.EPSILON, memory_config=mem)
            inverse = t.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=mem)
            normalized_fp32 = t.multiply(
                wide, inverse, dtype=t.float32, fast_and_approximate_mode=False, memory_config=mem
            )
            normalized = t.typecast(normalized_fp32, t.bfloat16, memory_config=mem)
            weighted = t.multiply(
                normalized,
                self.weight,
                dtype=t.float32,
                fast_and_approximate_mode=False,
                activations=self.round_chain,
                memory_config=mem,
            )
            gate = helper.device_lookup(t, z, self.table)
            gated = t.multiply(weighted, gate, dtype=t.bfloat16, fast_and_approximate_mode=False, memory_config=mem)
        else:
            assert variant == "fused_tail"
            gate = helper.device_lookup(t, z, self.table)
            gated, debug = self.fused(wide, sums, gate)
            assert not debug
        # Same real decoder consumer boundary. The candidate's existing shard
        # can satisfy this placement; any actual copy in either path is timed.
        gated = t.to_memory_config(t.reshape(gated, (1, 1, 32, 128)), self.head_shard)
        merged = t.experimental.nlp_concat_heads_decode(gated, num_heads=32)
        return t.reshape(merged, (1, 1, 4096), (1, 32, 4096))


def prepare(args):
    assert verify_source(helper.__file__, EXPECTED_HELPER_SHA256)
    gamma, fixtures = helper.fixtures(args.checkpoint, args.random_cases)
    for fixture in fixtures:
        if "evidence" in fixture:
            metadata = json.loads((E / fixture["evidence"]).read_text())
            position = int(fixture["name"].removeprefix("position"))
            row = next(row for row in metadata["cases"] if row["position"] == position)
            capture = torch.load(row["capture"], weights_only=False)
            fixture["core_input"] = capture["tt_values"]["core"].float().reshape(1, 1, 32, 128).contiguous()
            assert torch.equal(fixture["core_input"].bfloat16().reshape(1, 32, 128), fixture["core"])
        else:
            fixture["core_input"] = fixture["core"].float().reshape(1, 1, 32, 128)
        fixture["z_input"] = fixture["z"].reshape(1, 1, 4096).contiguous()
        cpu, _ = helper.cpu_tail(
            fixture["core"],
            fixture["core"].float().square().sum(-1, keepdim=True),
            gamma,
            torch.nn.functional.silu(fixture["z"].float()),
        )
        assert torch.equal(cpu, fixture["expected"])
    timing = next(f for f in fixtures if f["name"] == f"position{args.position}")
    evidence_path = E / "v6_gated_norm_tail_fusion_device.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "completed_diagnostic" and evidence["exact_native_tail"] and evidence["device_closed"]
    assert evidence["script_sha256"] == EXPECTED_HELPER_SHA256
    result = dict(
        classification="Complete gated-normalization adapter microbenchmark; not full decoder performance or acceptance",
        acceptance_eligible=False,
        oracle_assisted=False,
        status="cpu_setup_pass",
        device_closed=False,
        source_sha256=helper.sha(__file__),
        helper_sha256=raw_sha256(helper.__file__),
        helper_historical_sha256=EXPECTED_HELPER_SHA256,
        helper_device_evidence=str(evidence_path),
        helper_device_evidence_sha256=helper.sha(evidence_path),
        timing_scope="FP32 recurrent core TILE[1,1,32,128] + BF16 Z TILE[1,1,4096] -> merged BF16 TILE[1,1,4096], including native sums, full SiLU lookup, all runtime placement copies, current head concat. Excludes setup/readback/output projection.",
        fused_debug=False,
        persistent_input_placement=dict(core="interleaved L1", projection_z="interleaved DRAM"),
        timing_fixture=timing["name"],
        same_input_each_replay=True,
        correctness_fixtures=[f["name"] for f in fixtures],
        trace_replays_per_sample=args.replays,
        samples_per_variant=args.samples,
        warmup=args.warmup,
        device_open_parameters=dict(
            physical_device_ids=[0], l1_small_size=32768, trace_region_size=args.trace_region_size
        ),
        correctness=[],
        timing=[],
    )
    return gamma, fixtures, timing, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-device", action="store_true")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / helper.REVISION,
    )
    p.add_argument("--position", type=int, default=3271)
    p.add_argument("--random-cases", type=int, default=8)
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trace-region-size", type=int, default=16 * 1024 * 1024)
    p.add_argument("--order", choices=["explicit-first", "fused-first"], default="explicit-first")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert args.replays >= 100 and args.samples > 0 and args.warmup > 0
    assert not args.output.exists(), "Preserve evidence"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    gamma, fixtures, timing_fixture, result = prepare(args)
    result["command"] = sys.argv
    snapshot = Path(str(args.output) + ".source.py.snapshot")
    snapshot.write_bytes(Path(__file__).read_bytes())
    result["source_snapshot"] = str(snapshot)
    result["source_snapshot_sha256"] = helper.sha(snapshot)
    input_snapshot = Path(str(args.output) + ".inputs.pt")
    torch.save(dict(gamma=gamma, fixtures=fixtures, timing_fixture=timing_fixture["name"]), input_snapshot)
    result["input_snapshot"] = str(input_snapshot)
    result["input_snapshot_sha256"] = helper.sha(input_snapshot)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_GATED_NORM_TIMING_SETUP_PASS", len(fixtures))
        return
    for variable in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(variable) in (None, "", "0"), (variable, "timing must be separate")
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        mesh = t.open_mesh_device(t.MeshShape(1, 1), **result["device_open_parameters"])
        adapter = CompleteAdapter(t, mesh, gamma)

        def upload(x, dtype, memory=None):
            return t.from_torch(
                x.contiguous(),
                dtype=dtype,
                layout=t.TILE_LAYOUT,
                device=mesh,
                memory_config=t.DRAM_MEMORY_CONFIG if memory is None else memory,
            )

        for fixture in fixtures:
            core, z = upload(fixture["core_input"], t.float32, t.L1_MEMORY_CONFIG), upload(
                fixture["z_input"], t.bfloat16
            )
            outputs = {}
            for variant in ("explicit", "fused_tail"):
                with device_only():
                    output = adapter(core, z, variant)
                outputs[variant] = t.to_torch(output).clone()
            check = helper.code_compare(outputs["explicit"], outputs["fused_tail"])
            assert check["bf16_code_differences"] == 0, (fixture["name"], check)
            result["correctness"].append(
                dict(
                    name=fixture["name"],
                    fused_vs_explicit=check,
                    explicit_vs_cpu=helper.code_compare(fixture["expected"].reshape(1, 1, 4096), outputs["explicit"]),
                )
            )
            save()
            del core, z, output

        core, z = upload(timing_fixture["core_input"], t.float32, t.L1_MEMORY_CONFIG), upload(
            timing_fixture["z_input"], t.bfloat16
        )
        with device_only():
            reference_tensor = adapter(core, z, "explicit")
        reference = t.to_torch(reference_tensor).clone()
        del reference_tensor
        variants = ["explicit", "fused_tail"] if args.order == "explicit-first" else ["fused_tail", "explicit"]
        for variant in variants:
            for _ in range(args.warmup):
                with device_only():
                    warm = adapter(core, z, variant)
                t.synchronize_device(mesh)
                del warm
            trace_id, capture_open, output = None, False, None
            try:
                trace_id = t.begin_trace_capture(mesh, cq_id=0)
                capture_open = True
                with device_only():
                    output = adapter(core, z, variant)
                t.end_trace_capture(mesh, trace_id, cq_id=0)
                capture_open = False
                # This output belongs only to this trace and is intentionally
                # overwritten on replay. Inputs and setup constants stay live.
                trace_allocation_tracker.acknowledge_corruptible(output)
                for _ in range(args.warmup):
                    t.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                first = t.to_torch(output).clone()
                first_check = helper.code_compare(reference, first)
                assert first_check["bf16_code_differences"] == 0, first_check
                samples = []
                for _ in range(args.samples):
                    t.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.replays):
                        t.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    t.synchronize_device(mesh)
                    samples.append(1000 * (time.perf_counter() - start) / args.replays)
                last = t.to_torch(output).clone()
                last_check, deterministic = helper.code_compare(reference, last), helper.code_compare(first, last)
                assert last_check["bf16_code_differences"] == deterministic["bf16_code_differences"] == 0
                row = dict(
                    variant=variant,
                    median_ms=statistics.median(samples),
                    min_ms=min(samples),
                    max_ms=max(samples),
                    samples_ms=samples,
                    replays_per_sample=args.replays,
                    total_timed_replays=args.replays * args.samples,
                    synchronization="before and after each replay group",
                    first_vs_explicit=first_check,
                    last_vs_explicit=last_check,
                    first_vs_last=deterministic,
                    trace_output_memory_config=str(output.memory_config()),
                    trace_output_logical_shape=list(output.shape),
                    trace_output_padded_shape=list(output.padded_shape),
                )
                result["timing"].append(row)
                save()
                print(variant, row["median_ms"], "ms", flush=True)
            finally:
                if trace_id is not None:
                    try:
                        if capture_open:
                            t.end_trace_capture(mesh, trace_id, cq_id=0)
                        t.synchronize_device(mesh)
                    finally:
                        t.release_trace(mesh, trace_id)
                del output
        by_name = {row["variant"]: row for row in result["timing"]}
        before, after = by_name["explicit"]["median_ms"], by_name["fused_tail"]["median_ms"]
        result.update(
            status="completed_microbenchmark",
            median_delta_ms=after - before,
            median_speedup=before / after,
            improved=after < before,
        )
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        assert verify_source(helper.__file__, EXPECTED_HELPER_SHA256)
        save()


if __name__ == "__main__":
    main()
