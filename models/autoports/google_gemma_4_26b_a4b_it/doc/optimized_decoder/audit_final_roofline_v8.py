# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audit saved v8 profiler evidence using the standard library; never run hardware."""

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[5]
DOC = Path(__file__).resolve().parent
MODEL = DOC.parent.parent
RUNTIME = "5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898"


def source(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def module(name):
    spec = importlib.util.spec_from_file_location(name, MODEL / "tests" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def equal(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), (a, b)


def rows(path, delimiter=","):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def operand(row, prefix, perf):
    return {
        "padded_shape": perf.tensor_shape(row, prefix),
        "logical_shape": perf.tensor_shape(row, prefix, logical=True),
        "dtype": row.get(prefix + "_DATATYPE"),
        "memory": row.get(prefix + "_MEMORY"),
    }


def audit_prefill_projections(folder, selected, runner, layer, perf):
    attachment_path = folder / "prefill_projection_native_audit.json"
    attachment = read(attachment_path)
    assert attachment["status"] == "passed" and attachment["runtime_sha256"] == RUNTIME
    assert attachment["source_csv_sha256"] == source(folder / "ops.csv")["sha256"]
    minimal = [row for row in selected if "minimalmatmul" in row["OP CODE"].lower()]
    assert len(minimal) == attachment["native_minimal_rows"]
    expected_roles = {
        "qkv": (
            (1, 1, 2816, 8192 if layer == 0 else 9216),
            8 if layer == 0 else 16,
            "HiFi4" if layer == 0 else "HiFi2",
        ),
        "output": ((1, 1, 4096 if layer == 0 else 8192, 2816), 16 if layer == 0 else 8, "LoFi"),
    }
    role_records = {}
    projection_flops = 0
    for role, (weight, k_block, fidelity) in expected_roles.items():
        rows_for_role = [
            row
            for row in selected
            if "matmul" in row["OP CODE"].lower() and perf.tensor_shape(row, "INPUT_1", True) == weight
        ]
        assert rows_for_role and sum(perf.tensor_shape(row, "INPUT_0", True)[-2] for row in rows_for_role) == 4096
        is_minimal = role == "qkv" or layer == 5
        records = []
        for row in rows_for_role:
            assert ("minimalmatmul" in row["OP CODE"].lower()) == is_minimal
            assert row["INPUT_1_DATATYPE"] == "BFLOAT8_B" and row["OUTPUT_0_DATATYPE"] == "FLOAT32"
            assert row["MATH FIDELITY"] == fidelity
            if is_minimal:
                config = {
                    key: perf.attribute_integer(row, key)
                    for key in ("M_block_size", "K_block_size", "N_block_size", "subblock_h", "subblock_w")
                }
                assert config == dict(
                    M_block_size=2 if role == "qkv" and layer == 5 else 4,
                    K_block_size=k_block,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=4,
                )
                saved_rows = attachment["roles"][role]["rows"]
                saved = [item for item in saved_rows if item["global_call_count"] == row["GLOBAL CALL COUNT"]]
                assert len(saved) == 1 and saved[0]["config"] == config and saved[0]["attributes"] == row["ATTRIBUTES"]
            else:
                assert perf.attribute_integer(row, "in0_block_w") == 16
            flops = 2 * math.prod(perf.tensor_shape(row, "INPUT_0", True)) * weight[-1]
            projection_flops += flops
            records.append(
                dict(
                    global_call_count=row["GLOBAL CALL COUNT"],
                    operation=row["OP CODE"],
                    useful_projection_flops=flops,
                    attributes=row["ATTRIBUTES"],
                    math_fidelity=fidelity,
                    operands={prefix: operand(row, prefix, perf) for prefix in ("INPUT_0", "INPUT_1", "OUTPUT_0")},
                    kernel_sources={key: value for key, value in row.items() if "KERNEL SOURCE" in key and value},
                )
            )
        role_records[role] = dict(native_rows=len(records), minimal=is_minimal, rows=records)
    assert projection_flops == attachment["useful_projection_flops_from_native_logical_shapes"]
    assert projection_flops == perf.useful_prefill_flops(runner["layer_type"])["qkv_and_output_projections"]
    policy = runner["precision_policy"]["prefill_qkv_projection"]
    assert policy["backend"] == "minimal_matmul" and policy["k_block"] == (8 if layer == 0 else 16)
    return dict(
        status="passed",
        attachment=source(attachment_path),
        native_minimal_rows=len(minimal),
        useful_projection_flops=projection_flops,
        roles=role_records,
        report_config_limitation=attachment["installed_report_config_limitation"],
    )


def audit_layer(layer, perf, reconciliation):
    kind = "sliding_attention" if layer == 0 else "full_attention"
    folder = DOC / "tracy" / f"actual_optimized_v8_layer{layer}"
    summary_path = folder / "whole_layer.json"
    runner_path = DOC / f"profile_actual_optimized_v8_layer{layer}.json"
    journal = DOC / "actual_optimized_v8_profile_commands.json"
    summary, runner = read(summary_path), read(runner_path)
    assert runner["runtime_sha256"] == RUNTIME
    assert runner["candidate"] == {"defaults": True, "overrides": {}}
    assert runner["passed"] and runner["decode"]["passed"] and runner["decode"]["repeated_equal"]
    assert runner["program_cache_miss_guard"] and runner["runtime_prefill_audit"] == "clean"
    assert runner["decode"]["runtime_decode_audit"] == "clean"
    assert source(ROOT / runner["input_fixture"])["sha256"] == runner["input_fixture_sha256"]
    assert runner["decode"]["steps"] == 128 and runner["decode"]["positions"] == [4096, 4223]
    assert summary["workload"] == dict(
        input_tokens=4096, output_tokens=128, batch=1, concurrency=1, decode_start_position=4096
    )
    assert summary["layer_type"] == kind
    command_log = read(folder / "summary_commands.json")
    command = command_log["summary_command"]
    assert command[command.index("--peak-fidelity-cycles") + 1] == "1"
    assert command[command.index("--native-sdpa-read-chunk-size") + 1] == "128"
    assert len(command_log["report_commands"]) == 4
    for entry in command_log["report_commands"]:
        assert entry["returncode"] == 0 and entry["advice_enabled"]
        assert (ROOT / entry["stdout"]).stat().st_size > 0
        assert "--no-advice" not in entry["argv"]
    raw_csv = ROOT / command_log["profile_source"]
    csv_path = ROOT / summary["source"]
    assert source(raw_csv)["sha256"] == source(csv_path)["sha256"]
    windows, endpoints = {"prefill": [], "decode": []}, {}
    phase = None
    for row in rows(csv_path):
        code = row["OP CODE"]
        if code in ("PERF_PREFILL", "PERF_PREFILL_END", "PERF_DECODE", "PERF_DECODE_END"):
            assert code not in endpoints
            endpoints[code] = int(row["HOST START TS"])
            phase = None if code.endswith("_END") else code.removeprefix("PERF_").lower()
        elif phase and row["OP TYPE"] == "tt_dnn_device":
            windows[phase].append(row)
    assert len(endpoints) == 4
    raw_messages = folder / "raw" / ".logs" / "tracy_ops_data.csv"
    messages = {}
    for row in rows(raw_messages, ";"):
        name = row["MessageName"].strip("`").removeprefix("TT_SIGNPOST: ")
        if name in endpoints:
            assert name not in messages
            messages[name] = int(row["total_ns"])
    assert messages == endpoints
    projection_audit = audit_prefill_projections(folder, windows["prefill"], runner, layer, perf)
    phases, sparse_records, kv_records, replays = {}, [], [], []
    traffic_by_op = defaultdict(float)
    representative = []
    for phase, selected in windows.items():
        assert len({r["DEVICE ID"] for r in selected}) == 1
        scale = statistics.median(
            float(r["DEVICE FW DURATION [ns]"]) / (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"]))
            for r in selected
            if float(r["DEVICE FW END CYCLE"]) > float(r["DEVICE FW START CYCLE"])
        )
        groups = defaultdict(list)
        for row in selected:
            groups[row["METAL TRACE REPLAY SESSION ID"] if phase == "decode" else "prefill"].append(row)
        assert len(groups) == (128 if phase == "decode" else 1)
        assert len({len(group) for group in groups.values()}) == 1
        durations, transfers = [], []
        ordered = sorted(groups.items(), key=lambda item: min(float(r["DEVICE FW START CYCLE"]) for r in item[1]))
        for index, (session, group) in enumerate(ordered):
            first = min(float(r["DEVICE FW START CYCLE"]) for r in group)
            last = max(float(r["DEVICE FW END CYCLE"]) for r in group)
            duration = (last - first) * scale / 1000
            durations.append(duration)
            if phase != "decode":
                continue
            assert session not in ("", "-", "nan")
            position, transfer = 4096 + index, 0
            sdpa_count, sparse_count = 0, 0
            for row in group:
                counts = None
                if perf.is_native_sdpa_decode(row):
                    sdpa_count += 1
                    counts, basis = perf.native_sdpa_cache_reads(row, position, kind, 128)
                    heads, width = (8, 256) if layer == 0 else (2, 512)
                    logical_start = max(0, position + 1 - 1024) if layer == 0 else 0
                    expected_tokens = ((position + 128) // 128) * 128 - (logical_start // 128) * 128
                    assert basis["read_tokens"] == expected_tokens and basis["chunk_tokens"] == 128
                    equal(basis["kv_dram_bytes"], 2 * expected_tokens * heads * width * 1088 / 1024)
                    assert row["INPUT_1_DATATYPE"] == row["INPUT_2_DATATYPE"] == "BFLOAT8_B"
                    assert row["OUTPUT_0_DATATYPE"] == "BFLOAT16"
                    kv_records.append(dict(session=session, **basis))
                if "sparsematmul" in row["OP CODE"].lower():
                    sparse_count += 1
                    basis = perf.sparse_weight_groups(row)
                    assert basis["indexed"] and basis["active"] == 8 and basis["groups"] == 128
                    assert perf.attribute_integer(row, "nnz") is None
                    assert row["INPUT_3_DATATYPE"] == "UINT16"
                    assert perf.tensor_shape(row, "INPUT_3", True) == (1, 1, 1, 8)
                    assert perf.tensor_shape(row, "OUTPUT_0", True)[1] == 8
                    assert row["MATH FIDELITY"] == "LoFi"
                    expected_dtype = "BFLOAT8_B" if layer == 0 and sparse_count == 1 else "BFLOAT4_B"
                    assert row["INPUT_1_DATATYPE"] == expected_dtype
                    sparse_records.append(dict(global_call_count=row.get("GLOBAL CALL COUNT"), **basis))
                    # All sparse operands other than the weight bank are in L1.
                    dram = [key for key, value in row.items() if key.endswith("_MEMORY") and "DRAM" in value]
                    assert dram == ["INPUT_1_MEMORY"]
                    equal(
                        perf.traffic(row),
                        math.prod(perf.tensor_shape(row, "INPUT_1")) * 8 / 128 * perf.element_bytes(expected_dtype),
                    )
                amount = perf.traffic(row, counts)
                transfer += amount
                traffic_by_op[row["OP CODE"]] += amount / 128
                if index == 0 and ("matmul" in row["OP CODE"].lower() or perf.is_native_sdpa_decode(row)):
                    representative.append(
                        dict(
                            operation=row["OP CODE"],
                            fidelity=row["MATH FIDELITY"],
                            operands={
                                prefix: operand(row, prefix, perf)
                                for prefix in ("INPUT_0", "INPUT_1", "INPUT_2", "INPUT_3", "OUTPUT_0")
                                if perf.tensor_shape(row, prefix)
                            },
                            estimated_dram_bytes=amount,
                        )
                    )
            assert sdpa_count == 1 and sparse_count == 2
            transfers.append(transfer)
            replays.append(
                dict(
                    session=session,
                    first_cycle=first,
                    last_cycle=last,
                    device_us=duration,
                    native_ops=len(group),
                    estimated_dram_bytes=transfer,
                    decode_position=position,
                )
            )
        result = dict(
            device_us=statistics.mean(durations),
            min_device_us=min(durations),
            max_device_us=max(durations),
            ns_per_cycle=scale,
            samples=len(groups),
            native_ops=len(selected),
            summed_kernel_device_us=sum(perf.number(r, "DEVICE KERNEL DURATION [ns]") or 0 for r in selected)
            / 1000
            / len(groups),
        )
        if transfers:
            result["estimated_dram_bytes"] = statistics.mean(transfers)
        for key, value in result.items():
            equal(value, summary["whole_layer_windows"][phase][key])
        phases[phase] = result
    assert len(sparse_records) == 256 and sparse_records == summary["sparse_weight_reads"]
    assert len(kv_records) == 128 and kv_records == summary["native_sdpa_cache_reads"]["per_replay"]
    equal(summary["prefill_device_us"], phases["prefill"]["device_us"])
    equal(summary["decode_device_us"], phases["decode"]["device_us"])
    equal(summary["decode_dram_bytes"], phases["decode"]["estimated_dram_bytes"])
    equal(
        summary["native_sdpa_cache_reads"]["mean_kv_dram_bytes"],
        statistics.mean(row["kv_dram_bytes"] for row in kv_records),
    )
    equal(
        summary["native_sdpa_cache_reads"]["mean_read_tokens"],
        statistics.mean(row["read_tokens"] for row in kv_records),
    )
    saved_replays = rows(summary_path.with_suffix(".replays.csv"))
    assert len(saved_replays) == len(replays)
    for actual, saved in zip(replays, saved_replays):
        assert actual["session"] == saved["session"]
        for key, value in actual.items():
            if key != "session":
                equal(value, float(saved[key]))
    terms = dict(
        qkv_and_output_projections=2
        * 4096
        * 2816
        * (2 * 16 * (256 if layer == 0 else 512) + (4096 if layer == 0 else 1024)),
        shared_mlp=6 * 4096 * 2816 * 2112,
        active_experts_top8=6 * 4096 * 2816 * 704 * 8,
        router_projection=2 * 4096 * 2816 * 128,
        causal_attention_qk_and_pv=4
        * 16
        * (256 if layer == 0 else 512)
        * (4096 * 1024 - 1024 * 1023 // 2 if layer == 0 else 4096 * 4097 // 2),
    )
    assert terms == perf.useful_prefill_flops(kind) == summary["useful_flops_terms"]
    assert sum(terms.values()) == summary["prefill_useful_flops"]
    assert summary["peak_flops_per_s"] == 120 * 4096 * 1350000000
    assert summary["peak_dram_bytes_per_s"] == 512000000000
    equal(
        summary["prefill_flops_pct"],
        100 * sum(terms.values()) / (summary["peak_flops_per_s"] * phases["prefill"]["device_us"] / 1e6),
    )
    equal(
        summary["decode_dram_pct"],
        100 * phases["decode"]["estimated_dram_bytes"] / (512e9 * phases["decode"]["device_us"] / 1e6),
    )
    reconciliation_path = folder / "timing_reconciliation.json"
    rec = reconciliation.reconcile(
        summary_path.relative_to(ROOT),
        runner_path.relative_to(ROOT),
        journal.relative_to(ROOT),
        (DOC / f"validated_v8_headline_layer{layer}.json").relative_to(ROOT),
    )
    assert rec == read(reconciliation_path)
    equal(rec["decode_profile_loop_host_mean_us"], (messages["PERF_DECODE_END"] - messages["PERF_DECODE"]) / 128000)
    equal(rec["decode_fixed_position_host_median_us"], statistics.median(runner["traced_decode_host_us"]))
    return dict(
        layer=layer,
        layer_type=kind,
        status="passed",
        runtime_sha256=runner["runtime_sha256"],
        sources=[
            source(path)
            for path in (
                summary_path,
                runner_path,
                csv_path,
                raw_csv,
                raw_messages,
                reconciliation_path,
                folder / "summary_commands.json",
            )
        ],
        windows=phases,
        native_ops_per_decode=phases["decode"]["native_ops"] // 128,
        decode_op_histogram=dict(
            Counter(row["OP CODE"] for row in windows["decode"][: phases["decode"]["native_ops"] // 128])
        ),
        useful_flops_terms=terms,
        useful_prefill_flops=sum(terms.values()),
        prefill_flops_pct=summary["prefill_flops_pct"],
        decode_dram_pct=summary["decode_dram_pct"],
        estimated_decode_dram_bytes_by_op=dict(traffic_by_op),
        compact_sparse_rows_checked=len(sparse_records),
        native_sdpa_rows_checked=len(kv_records),
        native_sdpa_mean_read_tokens=statistics.mean(row["read_tokens"] for row in kv_records),
        native_sdpa_mean_kv_bytes=statistics.mean(row["kv_dram_bytes"] for row in kv_records),
        decode_matmul_and_sdpa_operand_metadata=representative,
        measured_precision_policy={
            k: runner["precision_policy"][k]
            for k in (
                "kv_cache",
                "native_sdpa_fidelity",
                "prefill_attention_fidelity",
                "expert_gate",
                "decode_expert_down",
                "prefill_expert_gate",
                "prefill_expert_down",
                "decode_expert_activation",
                "decode_expert_fidelity",
                "decode_expert_fused_gelu",
                "decode_expert_backend",
                "decode_expert_slots",
                "shared_decode_weights",
                "qkv_fidelity",
            )
        },
        same_run_reconciliation=rec,
        prefill_projection_native_audit=projection_audit,
        checks=[
            "Runtime and recorded fixture hashes",
            "Successful matching command and raw CSV hash",
            "Four advice-enabled reports",
            "All complete native windows and replay counts",
            "All 256 indexed sparse rows have eight IDs/compact groups",
            "All 128 source-derived rounded KV reads",
            "Every saved replay duration and estimated traffic",
            "Useful FLOP terms and declared peak ratios",
            "Raw Tracy total_ns matches CSV host signposts",
            "Exact same-run reconciliation including the final command-journal hash",
            "Selected native minimal QKV/output configs and logical projection FLOPs",
        ],
    )


def main():
    assert Path.cwd().resolve() == ROOT, "Run from the repository root"
    assert source(MODEL / "tt" / "optimized_decoder.py")["sha256"] == RUNTIME
    journal = read(DOC / "actual_optimized_v8_profile_commands.json")
    assert len(journal) == 2 and all(entry["returncode"] == 0 for entry in journal)
    perf, reconciliation = module("summarize_perf"), module("reconcile_perf")
    layers = [audit_layer(layer, perf, reconciliation) for layer in (0, 5)]
    old = read(DOC / "roofline_audit_e810_historical.json")
    paths = [ROOT / item["path"] for item in old["sources"]]
    paths += [
        Path(__file__).resolve(),
        DOC / "actual_optimized_v8_profile_commands.json",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp",
        ROOT
        / "ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.hpp",
        ROOT / "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp",
        DOC / "run_profiles_v8.py",
    ]
    result = dict(
        schema_version=2,
        status="passed",
        runtime_sha256=RUNTIME,
        scope=dict(
            model="google/gemma-4-26B-A4B-it",
            mode="CPU source and saved-artifact audit",
            input_tokens=4096,
            decode_positions_inclusive=[4096, 4223],
            decode_replays=128,
            batch=1,
            devices=1,
            hardware_run_by_auditor=False,
            final_profile_totals_verified=True,
        ),
        conclusion="Both final v8 summaries and same-run reconciliations reproduce from their raw profiler evidence; no numerical correction required.",
        sources=[source(path) for path in paths],
        layers=layers,
        peak_basis=dict(
            cores_per_asic=120,
            clock_hz=1350000000,
            lofi_flops_per_core_cycle=4096,
            common_fidelity_cycles=1,
            peak_flops_per_s=663552000000000,
            peak_dram_bytes_per_s=512000000000,
            official_topology_source="https://docs.tenstorrent.com/aibs/blackhole/p300.html",
            interpretation="Common theoretical LoFi peak for mixed-fidelity code; percentages are neither measured FPU utilization nor controller bandwidth.",
        ),
        limitations=[
            "Estimated DRAM operands include actual padded metadata shapes and BFP exponent storage; extra core rereads, NoC/reduction and profiler writes are excluded.",
            "Useful FLOPs omit padding, masked attention, extra union-expert prefill work, scalar normalization and transcendental work; their device time remains included.",
            "Decode denominator includes every operation and intra-layer gap in each replay, excluding inter-replay host/input refresh gaps.",
            "Fixed-position host and successive-position device runs have different execution regimes; arithmetic gaps, including negative gaps, do not isolate causes.",
            "Separate unprofiled observations are excluded from same-run gaps.",
            "The profiler checks selected HF outputs; complete correctness and strict actual-input long-row gates remain separately evidenced in validated_v8_validation_summary.json.",
            "This accounting audit does not itself establish allocation-free trace capture or tight-cache bounds. Those repaired contracts have separate tracker and native-call bounds evidence in the v8 validation manifest and its explicitly inherited v6 checks.",
        ],
        historical=dict(
            path="roofline_audit_e810_historical.json",
            runtime_sha256=old["runtime_sha256"],
            scope="Earlier source-only audit; HiFi4 peak 165.888 TFLOP/s differs by 4x from current LoFi peak. No historical metric is current v8 evidence.",
            completed_v4_audit="roofline_audit_v4_historical.json",
            completed_v4_runtime_sha256="3d51014f98128dfb21bb484fcece50993524b967754f68f6dfe825ae7f472ba9",
        ),
        cpu_verification=dict(
            command=f"python {Path(__file__).resolve().relative_to(ROOT)}",
            status="passed",
            method="Independent window/peak/FLOP/KV arithmetic plus reviewed summarizer traffic helpers and exact reconciliation regeneration; standard library only.",
        ),
    )
    (DOC / "final_roofline_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "layers": [
                    {k: layer[k] for k in ("layer", "native_ops_per_decode", "prefill_flops_pct", "decode_dram_pct")}
                    for layer in layers
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
