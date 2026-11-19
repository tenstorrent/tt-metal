# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded TP4 prefill projection-family probe on captured real activations.

No imports of torch/TTNN until main's hardware entrypoint. One role per job;
parent serializes jobs and adapts material failures. Microbench wins require a
subsequent whole-layer measurement. This script never selects model defaults.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("inherited2d", "tuned2d", "1d-in0", "1d-in1", "minimal")
ROLES = ("packed_gdn", "packed_gdn_residual", "packed_qkvg", "gateup", "down", "attention_out", "joint")


def divisor(value, ceiling):
    return max(v for v in range(1, min(value, ceiling) + 1) if value % v == 0)


def program_spec(family, rows, k, n, policy, *, grid=(8, 4), k_block=None, minimal_blocks=None):
    """CPU-only legal logical geometry; device L1 capacity still needs evidence."""
    mt, kt, nt = math.ceil(rows / 32), k // 32, n // 32
    inherited = policy.get("_projection_options", {}) if family == "inherited2d" else {}
    kb = k_block or inherited.get("in0_block_w", policy.get("prefill_k", 8))
    if k % 32 or n % 32 or kb < 1 or kt % kb:
        raise ValueError("K/N must be tiled and K block must divide K tiles")
    if family in ("inherited2d", "tuned2d"):
        gx, gy = inherited.get("grid", policy.get("prefill_grid", (11, 10))) if family == "inherited2d" else grid
        pm, pn = math.ceil(mt / gy), math.ceil(nt / gx)
        sw = inherited.get("subblock_w", max(v for v in (1, 2, 4) if pn % v == 0))
        requested_bh = inherited.get(
            "out_block_h", 1 if family == "tuned2d" or policy.get("prefill_height_block", False) else pm
        )
        bh = divisor(pm, requested_bh)
        requested_h = inherited.get("subblock_h", 4 if policy.get("prefill_subblock_h", False) else 1)
        sh = max(v for v in (1, 2, 4) if v <= min(bh, requested_h) and bh % v == 0 and v * sw <= 4)
        if pn % sw:
            raise ValueError("Inherited N subblock does not divide per-core N")
        return {
            "compute_with_storage_grid_size": (gx, gy),
            "in0_block_w": kb,
            "out_subblock_h": sh,
            "out_subblock_w": sw,
            "per_core_M": pm,
            "per_core_N": pn,
            "out_block_h": bh,
            "out_block_w": pn,
            "transpose_mcast": False,
            "fused_activation": None,
            "fuse_batch": True,
        }
    if family in ("1d-in0", "1d-in1"):
        cores = grid[0] * grid[1]
        # Native validation requires one M partition for mcast_in0 or one N
        # partition for mcast_in1. Small internal output blocks bound CB space.
        pm, pn = (mt, math.ceil(nt / cores)) if family == "1d-in0" else (math.ceil(mt / cores), nt)
        bh, bw = divisor(pm, 2), divisor(pn, 8)
        sw = max(v for v in (1, 2, 4) if bw % v == 0)
        return {
            "compute_with_storage_grid_size": grid,
            "in0_block_w": kb,
            "out_subblock_h": 1,
            "out_subblock_w": sw,
            "per_core_M": pm,
            "per_core_N": pn,
            "out_block_h": bh,
            "out_block_w": bw,
            "fuse_batch": True,
            "fused_activation": None,
            "mcast_in0": family == "1d-in0",
        }
    if family == "minimal":
        mb, mk, nb = minimal_blocks or ((2, kb, 8) if rows <= 512 else (4, kb, 16))
        if min(mb, mk, nb) < 1 or kt % mk:
            raise ValueError("Minimal K block must divide K tiles")
        sw = max(v for v in (1, 2, 4) if nb % v == 0)
        return {
            "M_block_size": mb,
            "K_block_size": mk,
            "N_block_size": nb,
            "subblock_h": 1,
            "subblock_w": sw,
            "compute_with_storage_grid_size": grid,
        }
    raise ValueError(family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=(0, 3), required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[128])
    parser.add_argument("--family", nargs="+", choices=FAMILIES, default=["inherited2d"])
    parser.add_argument("--weight-layout", nargs="+", choices=("dram_sharded", "interleaved"), default=["dram_sharded"])
    parser.add_argument("--grid", nargs=2, type=int, default=[8, 4])
    parser.add_argument("--k-block", type=int)
    parser.add_argument("--minimal-blocks", nargs=3, type=int)
    parser.add_argument("--activation-file", type=Path)
    parser.add_argument(
        "--policy-file", type=Path, help="Explicit candidate policies, recorded; never a final default measurement"
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(1 <= m <= 4096 for m in args.rows) or args.samples < 1 or args.replays < 1:
        raise ValueError("Rows must be1..4096; samples/replays positive")
    if min(args.grid) < 1:
        raise ValueError("Grid must be positive")
    watcher = os.environ.get("TT_METAL_WATCHER", "0") not in ("", "0")
    profiler = os.environ.get("TT_METAL_DEVICE_PROFILER", "0") not in ("", "0")
    if watcher and profiler:
        raise ValueError("Watcher and profiler must be separate")
    if not args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve()):
        raise ValueError("Artifacts must be inside stage5")
    if args.output.exists():
        raise FileExistsError("Preserve previous evidence; use a new output name")
    import torch
    import ttnn
    from tracy import signpost
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
    from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder

    torch.set_num_threads(8)
    report = {
        "status": "running",
        "pass": False,
        "mesh": [1, 4],
        "role": args.role,
        "layer": args.layer,
        "watcher": watcher,
        "profiler": profiler,
        "weights": "real checkpoint, BFP8",
        "reference": "same decoded weights and captured inputs, inherited2D with interleaved weight",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__).resolve(), ROOT / "tt/multichip_decoder.py", ROOT / "tt/optimized_decoder.py")
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        fields = [
            "role",
            "rows",
            "k",
            "logical_n",
            "physical_n",
            "family",
            "weight_layout",
            "output_dtype",
            "median_us",
            "us_per_row",
            "minimum_pcc",
            "first_last_trace_equal",
            "status",
            "program",
        ]
        with args.output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in report["cases"]:
                writer.writerow(
                    {k: json.dumps(row[k]) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in fields}
                )

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    try:
        device_grid = mesh.compute_with_storage_grid_size()
        if args.grid[0] > device_grid.x or args.grid[1] > device_grid.y:
            raise ValueError("Requested grid exceeds device compute grid")
        cfg = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
        policy = json.loads(args.policy_file.read_text()) if args.policy_file else {}
        if args.role == "packed_qkvg":
            policy["packed_qkvg"] = True
        layer = MultichipDecoder.from_state_dict(
            load_layer(args.layer), hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, policy=policy
        )
        if args.role == "joint" and "packed_gdn_correction_joint" not in layer.weights:
            from models.autoports.qwen_qwen3_6_27b.tt.packed_correction_candidate import PackedCorrection

            PackedCorrection(layer)  # Setup-only exact quantized packing, no installed forward hook.
        key = {
            "gateup": "mlp.gate_up.weight",
            "down": "mlp.down_proj.weight",
            "joint": "packed_gdn_correction_joint",
            "attention_out": "linear_attn.out_proj.weight" if args.layer == 0 else "self_attn.o_proj.weight",
        }.get(args.role, args.role)
        if key not in layer.weights:
            raise ValueError(
                f"Role {key} absent from selected layer/policy; select an applicable role or retained-weight policy"
            )
        weight = layer.weights[key]
        if weight.dtype != ttnn.bfloat8_b:
            raise ValueError("This optimization-stage probe requires BFP8 weights")
        k, n = tuple(weight.shape)
        logical_n = layer.projection_info.get(key, {}).get("logical_n", 6784 if args.role == "joint" else n)
        dtype = ttnn.float32 if args.role in ("packed_gdn", "packed_gdn_residual", "joint") else ttnn.bfloat16
        role_group = "mlp" if key.startswith("mlp.") else "attention"
        ckc = layer.matmul_configs[role_group]
        report.update(
            selected_policy=dict(layer.policy), output_dtype=str(dtype), weight_shape=[k, n], logical_n=logical_n
        )

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        interleaved = ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)
        assert all(
            torch.equal(a, b) for a, b in zip(read(weight), read(interleaved))
        ), "Weight layout copy changed decoded values"
        if not weight.is_sharded() and "dram_sharded" in args.weight_layout:
            raise ValueError("Requested DRAM-sharded control is not actually sharded")
        source = args.activation_file or ROOT / f"doc/optimized_decoder/text_activations/text_l{args.layer}_b32_s254.pt"
        recorded = torch.load(source, weights_only=True)["x"]
        batch = 1 if max(args.rows) <= recorded.shape[1] else min(32, recorded.shape[0])
        seq = math.ceil(max(args.rows) / batch)
        if seq > recorded.shape[1]:
            raise ValueError("Recorded activation fixture does not provide enough logical rows")
        x = recorded[:batch, :seq].contiguous()
        captured = {}
        norm, linear = layer._norm, layer._linear

        def capture_norm(owner, value, name, **kwargs):
            result = norm(value, name, **kwargs)
            wanted = "post_attention_layernorm.weight" if args.role == "gateup" else "input_layernorm.weight"
            if name == wanted and args.role not in ("down", "attention_out") and "x" not in captured:
                captured["x"] = [v.reshape(-1, v.shape[-1]).contiguous() for v in read(result)]
            return result

        def capture_linear(owner, value, name):
            if name == key and args.role in ("down", "attention_out") and "x" not in captured:
                captured["x"] = [v.reshape(-1, v.shape[-1]).contiguous() for v in read(value)]
            return linear(value, name)

        layer._norm, layer._linear = MethodType(capture_norm, layer), MethodType(capture_linear, layer)
        state = (
            layer.create_recurrent_state(batch)
            if args.layer == 0
            else layer.create_paged_state(torch.arange(batch * math.ceil(seq / 32)).reshape(batch, -1))
        )
        rope = None
        if args.layer == 3:
            rotary = Qwen3_5TextRotaryEmbedding(cfg)
            rope = tuple(layer._upload(v.unsqueeze(1)) for v in rotary(x, torch.arange(seq).expand(batch, -1)))
        captured_output = layer.prefill_forward(layer.upload_residual(x), state=state, rope=rope)
        ttnn.synchronize_device(mesh)
        del captured_output
        layer._norm, layer._linear = norm, linear
        host = captured["x"]
        assert len(host) == 4 and all(v.shape[0] >= max(args.rows) and v.shape[1] == k for v in host)
        report["activation_provenance"] = {
            "file": str(source),
            "size_bytes": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "captured_prefill_shape": list(x.shape),
            "source": "setup-only capture of actual layer prefill operand",
            "capture_boundary": (
                "post_attention_layernorm"
                if args.role == "gateup"
                else "linear_operand" if args.role in ("down", "attention_out") else "input_layernorm"
            ),
            "rank_operand_shapes": [list(v.shape) for v in host],
        }
        save()

        def check(actual, expected):
            scores = [pcc(a[..., :logical_n], b[..., :logical_n]) for a, b in zip(actual, expected)]
            assert all(torch.isfinite(a).all() for a in actual)
            assert min(scores) >= 0.995, scores
            return scores

        def call(tx, family, layout, rows, spec):
            # The accepted inherited path caps large logical row batches. Keep
            # that complete control, while candidate families test one requested
            # logical chunk with their own internal block geometry.
            cap = projection_policy["_row_cap"]
            if family == "inherited2d" and rows > cap:
                if cap < 32 or cap % 32:
                    raise ValueError("Inherited row cap must be a multiple of32")
                parts = []
                for start in range(0, rows, cap):
                    stop = min(start + cap, rows)
                    part_spec = program_spec(family, stop - start, k, n, projection_policy)
                    parts.append(call(tx[:, start:stop, :], family, layout, stop - start, part_spec))
                return ttnn.concat(parts, dim=1)
            selected_weight = weight if layout == "dram_sharded" else interleaved
            if family == "minimal":
                return ttnn.experimental.minimal_matmul(
                    tx,
                    selected_weight,
                    config=ttnn.MinimalMatmulConfig(**spec),
                    compute_kernel_config=ckc,
                    dtype=dtype,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            cls = (
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig
                if family.startswith("1d-")
                else ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
            )
            return ttnn.linear(
                tx,
                selected_weight,
                program_config=cls(**spec),
                compute_kernel_config=ckc,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        projection_policy = dict(layer.policy)
        configs = layer.policy.get("prefill_projection_configs", {})
        projection_policy["_projection_options"] = configs.get(key, configs.get(key.removesuffix("_residual"), {}))
        projection_policy["_row_cap"] = layer.policy.get("matmul_row_cap", 512)
        if args.role == "joint":
            # The installed owner-safe joint path has its own grid/output block
            # and row cap. Generic matmul_row_cap would mislabel its control.
            helper = layer._packed_correction_candidate
            joint_options = (
                helper.prefill_config
                if helper is not None
                else (layer.policy.get("packed_correction_options") or {}).get("prefill_config", {})
            )
            options = {**configs.get("packed_gdn", {}), **joint_options}
            projection_policy["_projection_options"] = options
            projection_policy["_row_cap"] = options.get("row_cap", projection_policy["_row_cap"])
            report["joint_scope"] = {
                "measured": "raw FP32 primary-plus-correction joint matmul; combine is outside this microbenchmark",
                "input": "actual current input norm operand before the installed joint projection",
                "output_logical_columns": logical_n,
                "selected_joint_hook_installed": helper is not None,
                "inherited_options": options,
                "inherited_row_cap": projection_policy["_row_cap"],
            }
        combinations = [("inherited2d", "interleaved")]
        report["source_adaptations"] = []
        for family in args.family:
            for layout in args.weight_layout:
                if family == "1d-in0" and layout == "dram_sharded":
                    report["source_adaptations"].append(
                        {
                            "family": family,
                            "requested_layout": layout,
                            "measured_layout": "interleaved",
                            "reason": "mcast_in0 factory treats any non-global-CB sharded weight as locally resident and binds its address into an L1 CB; use exact setup interleaved copy",
                        }
                    )
                    layout = "interleaved"
                if (family, layout) not in combinations:
                    combinations.append((family, layout))
        save()

        for rows in args.rows:
            values = [v[:rows].reshape(1, rows, k).to(torch.bfloat16) for v in host]
            banks = [
                layer._upload_shards(values),
                layer._upload_shards([(-0.75 * v + 0.015625).to(torch.bfloat16) for v in values]),
            ]
            tx = ttnn.empty_like(banks[0])
            ref_spec = program_spec("inherited2d", rows, k, n, projection_policy)
            references = [read(call(bank, "inherited2d", "interleaved", rows, ref_spec)) for bank in banks]
            for family, layout in combinations:
                spec = program_spec(
                    family,
                    rows,
                    k,
                    n,
                    projection_policy,
                    grid=tuple(args.grid),
                    k_block=None if family == "inherited2d" else args.k_block,
                    minimal_blocks=args.minimal_blocks,
                )
                entry = {
                    "role": args.role,
                    "rows": rows,
                    "k": k,
                    "logical_n": logical_n,
                    "physical_n": n,
                    "family": family,
                    "weight_layout": layout,
                    "output_dtype": str(dtype),
                    "program": spec,
                    "status": "running",
                }
                if family == "inherited2d" and rows > projection_policy["_row_cap"]:
                    cap = projection_policy["_row_cap"]
                    entry["program"] = {
                        "outer_row_cap": cap,
                        "kernel_rows": [min(cap, rows - start) for start in range(0, rows, cap)],
                        "kernel_programs": [
                            program_spec(family, min(cap, rows - start), k, n, projection_policy)
                            for start in range(0, rows, cap)
                        ],
                    }
                report["cases"].append(entry)
                save()
                tid = None
                try:
                    scores = []
                    for step in (0, 1):
                        ttnn.copy(banks[step], tx)
                        scores += check(read(call(tx, family, layout, rows, spec)), references[step])
                    ttnn.copy(banks[0], tx)
                    for _ in range(2):
                        warm = call(tx, family, layout, rows, spec)
                        del warm
                    ttnn.synchronize_device(mesh)
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    traced = call(tx, family, layout, rows, spec)
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    trace_checks = []
                    for step in (0, 1, 0):
                        ttnn.copy(banks[step], tx)
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                        actual = read(traced)
                        scores += check(actual, references[step])
                        if step == 0:
                            trace_checks.append(actual)
                    equal = all(torch.equal(a, b) for a, b in zip(*trace_checks))
                    assert equal, "First/last original-input replay differs"
                    timings = []
                    if not watcher:
                        signpost(
                            "PERF_PREFILL_KERNEL",
                            json.dumps({"role": args.role, "rows": rows, "family": family, "weight_layout": layout}),
                        )
                        for _ in range(args.samples):
                            ttnn.synchronize_device(mesh)
                            start = time.perf_counter_ns()
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            timings.append((time.perf_counter_ns() - start) / 1000 / args.replays)
                        signpost("PERF_PREFILL_KERNEL_END")
                        scores += check(read(traced), references[0])
                    entry.update(
                        minimum_pcc=min(scores),
                        first_last_trace_equal=equal,
                        changed_input_trace=True,
                        trace_us_samples=timings,
                        median_us=statistics.median(timings) if timings else None,
                        us_per_row=statistics.median(timings) / rows if timings else None,
                        status="pass",
                    )
                except BaseException:
                    entry["status"] = "failed"
                    entry["exception"] = traceback.format_exc()
                    raise
                finally:
                    if tid is not None:
                        ttnn.release_trace(mesh, tid)
                    save()
                print(
                    json.dumps(
                        {
                            k: entry[k]
                            for k in (
                                "role",
                                "rows",
                                "family",
                                "weight_layout",
                                "minimum_pcc",
                                "median_us",
                                "status",
                            )
                        }
                    ),
                    flush=True,
                )
            del tx, banks, references
        if profiler:
            ttnn.ReadDeviceProfiler(mesh)
        report["pass"], report["status"] = True, "pass"
        save()
    except BaseException:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        save()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
