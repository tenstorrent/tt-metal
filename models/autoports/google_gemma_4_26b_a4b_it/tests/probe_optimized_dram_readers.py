# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture real decode projections and compare DRAM readers with identical storage.

Capture uses the audited decoder harness. The isolated phase measures only linear;
resharding, padding, upload and readback are outside the trace. CPU CSV processing
imports neither TTNN nor the decoder and can run while another job owns the device.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tt/optimized_decoder.py"
ROLES = ("qkv", "output", "shared_gate_up", "shared_down")
TILE_BYTES = {"float32": 4096, "bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}
COMPUTE_FIELDS = ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def geometry(k, logical_n, block, banks, available_x, dtype):
    """One common weight/input/output layout for all three reader counts."""
    if k % 32 or block <= 0 or (k // 32) % block:
        raise ValueError("K must be tile aligned and the positive K block must divide K/32")
    legal = [c for c in (8, 6, 4, 3, 2, 1) if c <= available_x and (k // 32) % (c * block) == 0]
    if not legal:
        raise ValueError("No activation storage grid can hold a whole K block")
    cores = legal[0]
    alignment = 32 * math.lcm(banks, 2 * banks, 3 * banks, cores)
    padded_n = math.ceil(logical_n / alignment) * alignment
    per_bank = padded_n // (32 * banks)
    tile_bytes = TILE_BYTES[dtype]
    block_bytes = per_bank * block * tile_bytes
    page_bytes = (16384 // tile_bytes) * tile_bytes
    while block_bytes % page_bytes:
        page_bytes -= tile_bytes
    return dict(
        logical_MKN=[1, k, logical_n],
        physical_MKN=[32, k, padded_n],
        in0_block_w=block,
        input_storage_cores=cores,
        input_shard_shape=[32, k // cores],
        output_shard_shape=[32, padded_n // cores],
        weight_banks=banks,
        weight_shard_shape=[k, padded_n // banks],
        per_bank_N_tiles=per_bank,
        stored_tile_bytes=tile_bytes,
        blackhole_noc_burst_bytes=16384,
        blackhole_one_reader_page_bytes=page_bytes,
        blackhole_one_reader_pages_per_K_block=block_bytes // page_bytes,
        logical_weight_bytes=(k // 32) * math.ceil(logical_n / 32) * tile_bytes,
        physical_weight_bytes=(k // 32) * (padded_n // 32) * tile_bytes,
        readers={
            str(r): dict(
                compute_workers=banks * r,
                per_reader_N_tiles=per_bank // r,
                per_reader_row_bytes=per_bank // r * tile_bytes,
                in1_triple_buffer_bytes=per_bank // r * block * 3 * tile_bytes,
                row_at_least_4KiB=per_bank // r * tile_bytes >= 4096,
                blackhole_split_reader_bursts_per_row=math.ceil(per_bank // r * tile_bytes / 16384) if r > 1 else None,
            )
            for r in (1, 2, 3)
        },
        multicast_at_least_64_cores=cores >= 64,
        estimates_scope="Weight payload/header bytes only; not total CB/L1 allocation. Screening flags are heuristics.",
    )


def reader_order(round_index):
    return list(tuple(itertools.permutations((1, 2, 3)))[round_index % 6])


def dtype_name(dtype, ttnn):
    return next(name for name in TILE_BYTES if dtype == getattr(ttnn, name))


def capture_inputs(args):
    import torch

    import ttnn
    from models.autoports.google_gemma_4_26b_a4b_it.tests import run_optimized_decoder
    from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    fixture = torch.load(args.input_fixture, map_location="cpu", weights_only=True)
    if fixture["metadata"]["layer"] != args.layer:
        raise ValueError("Fixture layer does not match --layer")
    # Preserve the exact real prefill and first continuation token. The harness
    # validates model/revision, BF16 transport, finite values and both HF gates.
    selected = dict(
        metadata={**fixture["metadata"], "steps": 1},
        prefill=fixture["prefill"],
        decode=fixture["decode"][:, :1].clone(),
    )
    fixture_path = args.capture.with_suffix(".fixture.pt")
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected, fixture_path)
    harness_report = args.capture.with_suffix(".decoder.json")
    targets, retained, policy = {}, {}, {}
    factory = OptimizedDecoder.from_state_dict.__func__
    original_linear, original_close = ttnn.linear, ttnn.close_mesh_device
    payload = dict(
        metadata=dict(
            source="Actual first decode projection tensors, captured from passing optimized decoder",
            layer=args.layer,
            length=fixture["prefill"].shape[1],
            runtime_sha256=digest(RUNTIME),
            probe_sha256=digest(__file__),
            original_fixture=str(args.input_fixture),
            original_fixture_sha256=digest(args.input_fixture),
            fixture_metadata=fixture["metadata"],
            capture_fixture=str(fixture_path),
            capture_fixture_sha256=digest(fixture_path),
            harness_report=str(harness_report),
        ),
        roles={},
    )

    def build(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        attention, shared = decoder.layer.self_attn, decoder.layer.shared_mlp
        qkv = attention.source.weights.wqkv
        while hasattr(qkv, "decode_source"):
            qkv = qkv.decode_source
        if len(qkv.weights) != 1 or shared.split:
            raise ValueError("This probe requires the selected packed QKV/shared topology")
        for role, weight, width in (
            ("qkv", qkv.weights[0], qkv.weights[0].shape[-1]),
            ("output", attention.source.weights.o_proj, attention.source.weights.o_proj.shape[-1]),
            ("shared_gate_up", shared.gate_up.weight, shared.gate_up.width),
            ("shared_down", shared.down.weight, shared.down.width),
        ):
            targets[id(weight)] = (role, width)
        policy.update(decoder.precision_policy)
        return decoder

    def linear(x, weight, **kw):
        output = original_linear(x, weight, **kw)
        match = targets.get(id(weight))
        if match and x.shape[-2] == 1 and match[0] not in retained:
            role, width = match
            config = kw["compute_kernel_config"]
            retained[role] = (x, weight, output)
            payload["roles"][role] = dict(
                logical_n=width,
                input_dtype=dtype_name(x.dtype, ttnn),
                weight_dtype=dtype_name(weight.dtype, ttnn),
                output_dtype=dtype_name(output.dtype, ttnn),
                source_input_memory=str(x.memory_config()),
                source_weight_memory=str(weight.memory_config()),
                source_output_memory=str(output.memory_config()),
                source_program=str(kw.get("program_config")),
                compute={
                    field: (
                        str(getattr(config, field)).split(".")[-1]
                        if field == "math_fidelity"
                        else bool(getattr(config, field))
                    )
                    for field in COMPUTE_FIELDS
                },
            )
        return output

    def close(mesh):
        try:
            if set(retained) != set(ROLES):
                raise AssertionError(f"Missing production projection captures: {set(ROLES) - set(retained)}")
            for role, tensors in retained.items():
                record = payload["roles"][role]
                for name, tensor in zip(("input", "weight", "production_output"), tensors):
                    record[name] = ttnn.to_torch(tensor).float().contiguous()
        finally:
            original_close(mesh)

    argv = [
        sys.argv[0],
        "--defaults",
        "--real",
        "--layer",
        str(args.layer),
        "--length",
        str(selected["prefill"].shape[1]),
        "--decode",
        "--steps",
        "1",
        "--verify-program-cache",
        "--input-fixture",
        str(fixture_path),
        "--output",
        str(harness_report),
    ]
    # Keep the established runtime guards. Only Python references are retained
    # inside forward; readbacks happen in close(), after the harness assertions.
    with (
        patch.object(OptimizedDecoder, "from_state_dict", classmethod(build)),
        patch.object(ttnn, "linear", linear),
        patch.object(ttnn, "close_mesh_device", close),
        patch.object(sys, "argv", argv),
        patch.object(torch, "set_num_threads", lambda _: None),
    ):
        run_optimized_decoder.main()
    report = json.loads(harness_report.read_text())
    assert report["passed"] and report["decode"]["passed"] and report["decode"]["repeated_equal"]
    assert report["runtime_sha256"] == payload["metadata"]["runtime_sha256"] == digest(RUNTIME)
    assert payload["roles"]["output"]["input_dtype"] == "bfloat16", "Native BF16 boundary was not captured"
    payload["metadata"].update(harness_report_sha256=digest(harness_report), policy=policy, harness_argv=argv)
    torch.save(payload, args.capture)
    return payload


def pcc(a, b):
    import torch

    a, b = a.double().reshape(-1), b.double().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("Nonfinite projection result")
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / denominator) if denominator > 0 else 0.0


def summarize_csv(csv_path, report):
    """Attach complete native replay windows; host samples never stand in for them."""
    windows, active = {}, None
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            code = row["OP CODE"]
            if row["OP TYPE"] == "signpost" and code.startswith("DRAM_READER_"):
                if code.endswith("_END"):
                    if active != code.removesuffix("_END"):
                        raise ValueError(f"Unmatched signpost: {code}")
                    active = None
                else:
                    if active or code in windows:
                        raise ValueError(f"Overlapping or duplicate signpost: {code}")
                    active = code
                    windows[code] = []
            elif active and row["OP TYPE"] == "tt_dnn_device":
                windows[active].append(row)
    if active:
        raise ValueError("Unterminated DRAM reader signpost")
    expected_windows = {sample["signpost"] for case in report["cases"] for sample in case.get("samples", [])}
    if set(windows) != expected_windows:
        raise ValueError("Profiler windows do not match the complete probe report")
    for case in report["cases"]:
        if case["status"] != "measured_host_device_pending":
            continue
        grouped = {str(r): [] for r in case["legal_readers"]}
        for sample in case["samples"]:
            rows = windows[sample["signpost"]]
            if len(rows) != report["replays_per_sample"] or any(r["OP CODE"] != "MatmulDeviceOperation" for r in rows):
                raise ValueError(f"Expected exactly one native matmul per replay: {sample['signpost']}")
            reader = sample["readers"]
            for row in rows:
                attributes = row["ATTRIBUTES"]
                if not re.search(rf"num_workers_per_dram_bank[=:]\s*{reader}(?:\D|$)", attributes):
                    raise ValueError("Actual profiler reader config does not match candidate")
                if row["MATH FIDELITY"] != case["compute"]["math_fidelity"]:
                    raise ValueError("Actual profiler fidelity does not match production")
                for column, name in (
                    ("INPUT_0_DATATYPE", "input_dtype"),
                    ("INPUT_1_DATATYPE", "weight_dtype"),
                    ("OUTPUT_0_DATATYPE", "output_dtype"),
                ):
                    if row[column].upper().split("::")[-1] != case[name].upper():
                        raise ValueError(f"Actual profiler {column} does not match production")
            durations = [float(row["DEVICE KERNEL DURATION [ns]"]) / 1000 for row in rows]
            sample["device_us"] = durations
            sample["device_median_us"] = statistics.median(durations)
            sample["native_attributes"] = rows[0]["ATTRIBUTES"]
            sample["native_core_count"] = rows[0]["CORE COUNT"]
            sample["risc_median_us"] = {
                risc: statistics.median(float(row[f"DEVICE {risc} KERNEL DURATION [ns]"]) / 1000 for row in rows)
                for risc in ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")
                if all(row.get(f"DEVICE {risc} KERNEL DURATION [ns]", "").strip() not in ("", "nan") for row in rows)
            }
            grouped[str(reader)].extend(durations)
        case["device_summary"] = {}
        for reader, values in grouped.items():
            median = statistics.median(values)
            physical = case["geometry"]["physical_weight_bytes"] / median / 1000
            logical = case["geometry"]["logical_weight_bytes"] / median / 1000
            round_medians = [s["device_median_us"] for s in case["samples"] if str(s["readers"]) == reader]
            case["device_summary"][reader] = dict(
                median_us=median,
                min_us=min(values),
                max_us=max(values),
                round_medians_us=round_medians,
                round_median_spread_us=max(round_medians) - min(round_medians),
                physical_weight_GBps=physical,
                logical_weight_GBps=logical,
                physical_percent_declared_peak=100 * physical / report["declared_dram_peak_GBps"],
            )
        case["status"] = "measured_device_and_host"
    report["device_profile"] = dict(path=str(csv_path), sha256=digest(csv_path), complete_windows=len(windows))
    return report


def benchmark_case(ttnn, torch, mesh, role, data, block, args):
    k, logical_n = data["input"].shape[-1], data["logical_n"]
    available, bank_grid = mesh.compute_with_storage_grid_size(), mesh.dram_grid_size()
    geo = geometry(k, logical_n, block, bank_grid.x * bank_grid.y, available.x, data["weight_dtype"])
    case = {key: data[key] for key in ("input_dtype", "weight_dtype", "output_dtype", "compute")}
    case.update(role=role, block=block, geometry=geo, invalid={}, samples=[], legal_readers=[], accuracy={})
    n, cores = geo["physical_MKN"][-1], geo["input_storage_cores"]
    memory = lambda shape: ttnn.create_sharded_memory_config(
        shape,
        ttnn.CoreGrid(x=cores, y=1),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    weight_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(bank_grid.x - 1, bank_grid.y - 1))}),
            geo["weight_shard_shape"],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    host_weight = torch.nn.functional.pad(data["weight"][..., :logical_n], (0, n - logical_n))
    weight = ttnn.from_torch(
        host_weight,
        device=mesh,
        dtype=getattr(ttnn, data["weight_dtype"]),
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_memory,
    )
    x = ttnn.from_torch(
        data["input"],
        device=mesh,
        dtype=getattr(ttnn, data["input_dtype"]),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory(geo["input_shard_shape"]),
    )
    output_memory = memory(geo["output_shard_shape"])
    out = ttnn.from_torch(
        torch.zeros(1, 1, 1, n),
        device=mesh,
        dtype=getattr(ttnn, data["output_dtype"]),
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_memory,
    )
    repacked = ttnn.to_torch(weight).float()
    assert torch.equal(repacked, host_weight), "Whole-tile padding changed captured quantized weights"
    assert torch.equal(ttnn.to_torch(x).float(), data["input"]), "Input repack changed production values"
    compute_values = {**data["compute"], "math_fidelity": getattr(ttnn.MathFidelity, data["compute"]["math_fidelity"])}
    compute = ttnn.init_device_compute_kernel_config(mesh.arch(), **compute_values)
    golden = torch.matmul(data["input"].float(), data["weight"][..., :logical_n].float())
    production = data["production_output"][..., :logical_n]
    traces, outputs = {}, {}
    try:
        for readers in (1, 2, 3):
            if readers > 1 and mesh.arch() != ttnn.Arch.BLACKHOLE:
                case["invalid"][str(readers)] = "Multiple DRAM readers require Blackhole"
                continue
            if available.x * available.y - cores < bank_grid.x * bank_grid.y * readers:
                case["invalid"][str(readers)] = "Insufficient workers after excluding activation storage cores"
                continue
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block, per_core_M=1, per_core_N=n // 32 // cores, num_workers_per_dram_bank=readers
            )

            def forward():
                return ttnn.linear(
                    x,
                    weight,
                    dtype=getattr(ttnn, data["output_dtype"]),
                    memory_config=output_memory,
                    program_config=program,
                    compute_kernel_config=compute,
                    optional_output_tensor=out,
                )

            try:
                forward()
                ttnn.synchronize_device(mesh)
            except RuntimeError as error:
                message = str(error)
                # Only known setup/placement constraints may be skipped. A kernel
                # failure, timeout or unrelated exception aborts the experiment.
                if not any(
                    token in message
                    for token in (
                        "Statically allocated circular buffers",
                        "L1 buffer",
                        "L1 size",
                        "Not enough worker",
                        "overlap with L1",
                    )
                ):
                    raise
                case["invalid"][str(readers)] = message
                continue
            value = ttnn.to_torch(out).float()[..., :logical_n]
            metrics = dict(
                pcc_fp32_dot=pcc(golden, value),
                pcc_production=pcc(production, value),
                max_abs_diff_production=float((production - value).abs().max()),
            )
            metrics["passed"] = min(metrics["pcc_fp32_dot"], metrics["pcc_production"]) >= args.pcc
            case["accuracy"][str(readers)] = metrics
            outputs[readers] = value
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            mesh.set_program_cache_misses_allowed(False)
            try:
                forward()
            finally:
                mesh.set_program_cache_misses_allowed(True)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[readers] = trace
            case["legal_readers"].append(readers)
            # First-use trace dispatch is not a warmed timing sample.
            for _ in range(2):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(out).float()[..., :logical_n], value)
        if not traces:
            case["status"] = "all_readers_invalid"
            return case
        from tracy import signpost

        for round_index in range(args.rounds):
            for readers in reader_order(round_index):
                if readers not in traces:
                    continue
                marker = f"DRAM_READER_{role}_k{block}_r{readers}_round{round_index}"
                ttnn.synchronize_device(mesh)
                if args.profile:
                    signpost(marker)
                start = time.perf_counter_ns()
                for _ in range(args.replays):
                    ttnn.execute_trace(mesh, traces[readers], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                host_us = (time.perf_counter_ns() - start) / (1000 * args.replays)
                if args.profile:
                    signpost(marker + "_END")
                repeated = ttnn.to_torch(out).float()[..., :logical_n]
                assert torch.equal(repeated, outputs[readers]), "Repeated trace result changed"
                case["samples"].append(
                    dict(readers=readers, round=round_index, signpost=marker, warmed_host_us=host_us)
                )
        case["pairwise_pcc"] = {
            f"{a}_vs_{b}": pcc(outputs[a], outputs[b]) for a, b in itertools.combinations(outputs, 2)
        }
        case["all_legal_accuracy_passed"] = all(result["passed"] for result in case["accuracy"].values())
        case["host_median_us"] = {
            str(r): statistics.median(s["warmed_host_us"] for s in case["samples"] if s["readers"] == r) for r in traces
        }
        case["status"] = "measured_host_device_pending"
        return case
    finally:
        for trace in traces.values():
            ttnn.release_trace(mesh, trace)
        for tensor in (out, x, weight):
            tensor.deallocate(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path, help="Recorded projection payload .pt; reused only with matching runtime"
    )
    parser.add_argument("--input-fixture", type=Path, help="Required when creating a new capture")
    parser.add_argument("--layer", type=int, choices=(0, 5), default=0)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--roles", choices=ROLES, nargs="+", default=list(ROLES))
    parser.add_argument("--qkv-blocks", type=int, nargs="+", default=[1, 11])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pcc", type=float, default=0.995)
    parser.add_argument("--declared-dram-peak-gbps", type=float, default=512)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summarize-csv", type=Path)
    args = parser.parse_args()
    if args.summarize_csv:
        write_json(args.report, summarize_csv(args.summarize_csv, json.loads(args.report.read_text())))
        return
    if not args.capture or (not args.capture.exists() and not args.input_fixture):
        parser.error("Provide --capture and, for a new capture, --input-fixture")
    if args.rounds < 6 or args.rounds % 6 or args.replays < 2 or not 1 <= args.threads <= 4:
        parser.error("Use complete six-permutation rounds, at least two replays, and one to four CPU threads")
    import torch

    import ttnn

    torch.set_num_threads(args.threads)
    payload = (
        torch.load(args.capture, map_location="cpu", weights_only=True)
        if args.capture.exists()
        else capture_inputs(args)
    )
    metadata = payload["metadata"]
    if metadata["runtime_sha256"] != digest(RUNTIME) or metadata["layer"] != args.layer:
        raise ValueError("Capture runtime/layer differs from the current requested decoder")
    if args.input_fixture and digest(args.input_fixture) != metadata["original_fixture_sha256"]:
        raise ValueError("Capture fixture hash differs from --input-fixture")
    report = dict(
        command=sys.argv,
        runtime_sha256=digest(RUNTIME),
        probe_sha256=digest(__file__),
        capture=str(args.capture),
        capture_sha256=digest(args.capture),
        capture_metadata=metadata,
        rounds=args.rounds,
        replays_per_sample=args.replays,
        declared_dram_peak_GBps=args.declared_dram_peak_gbps,
        declared_peak_scope="Explicit aggregate SKU assumption; weight-only bandwidth excludes activation/output traffic",
        timing_scope="One warmed linear per trace; host wall includes dispatch/synchronization. Native CSV required for device time.",
        integration_status="No runtime policy changes; a winning isolated candidate still requires whole-layer integration",
        cases=[],
    )
    write_json(args.report, report)
    if args.capture_only:
        return
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        mesh.enable_program_cache()
        report["device"] = dict(
            arch=str(mesh.arch()),
            compute_grid=str(mesh.compute_with_storage_grid_size()),
            dram_grid=str(mesh.dram_grid_size()),
        )
        for role in args.roles:
            blocks = args.qkv_blocks if role == "qkv" else [16 if role == "output" else 11]
            for block in blocks:
                case = benchmark_case(ttnn, torch, mesh, role, payload["roles"][role], block, args)
                report["cases"].append(case)
                write_json(args.report, report)
                print(role, block, case["status"], case.get("host_median_us"), flush=True)
        assert digest(RUNTIME) == report["runtime_sha256"], "Runtime changed during probe"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
