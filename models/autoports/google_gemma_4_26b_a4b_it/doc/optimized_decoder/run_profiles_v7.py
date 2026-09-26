# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the two v7 profiles serially, then reports and same-run reconciliation.

Prepared without hardware execution. --plan prints commands without executing
them; the default mode is for the parent hardware owner after v7 validation.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DOC = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder")
RUNTIME_PATH = DOC.parent.parent / "tt/optimized_decoder.py"
RUNTIME = "daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b"
JOURNAL = DOC / "actual_optimized_v7_profile_commands.json"
POLICY = (
    "Selected v7 defaults, runtime " + RUNTIME + ". Prefill QKV uses minimal matmul 11x8, "
    "K8 sliding/K16 full, HiFi4 sliding/HiFi2 full, FP32 accumulation/output with BFP8 weights. Full prefill output "
    "uses minimal matmul K8 LoFi; sliding retains 2D K16 LoFi. Prefill expert gate BFP8 sliding/"
    "BFP4 full, down BFP4. Indexed compact top8 decode with actual native index/output metadata. "
    "Single-ASIC theoretical LoFi peak; useful work excludes padded/inactive experts."
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def renamed(command):
    return [part.replace("v4", "v7") for part in command]


def option(command, name):
    return command[command.index(name) + 1]


def prepared_commands():
    previous = read(DOC / "actual_optimized_v4_profile_commands.json")
    assert len(previous) == 2 and all(entry["returncode"] == 0 for entry in previous)
    commands = []
    for entry in previous:
        profile = renamed(entry["command"])
        layer = int(option(profile, "--layer"))
        prior = read(DOC / "tracy" / f"actual_optimized_v4_layer{layer}" / "summary_commands.json")
        summary = renamed(prior["summary_command"])
        summary[summary.index("--precision-policy") + 1] = POLICY
        reports = [
            {"argv": renamed(record["argv"]), "stdout": record["stdout"].replace("v4", "v7")}
            for record in prior["report_commands"]
        ]
        commands.append(
            dict(
                layer=layer,
                profile=profile,
                summary=summary,
                reports=reports,
                reconciliation=renamed(prior["reconciliation_command"]),
            )
        )
    assert [item["layer"] for item in commands] == [0, 5]
    return commands


def run(command, log, env=None):
    assert digest(RUNTIME_PATH) == RUNTIME, "Runtime differs from the frozen v7 campaign"
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=1800)
    assert digest(RUNTIME_PATH) == RUNTIME, "Runtime changed during command"
    return result.returncode


def one_replay(folder):
    with (folder / "ops.csv").open() as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = reader.fieldnames, list(reader)
    start = next(i for i, row in enumerate(rows) if row["OP CODE"] == "PERF_DECODE")
    end = next(i for i, row in enumerate(rows) if row["OP CODE"] == "PERF_DECODE_END")
    native = [row for row in rows[start + 1 : end] if row["OP TYPE"] == "tt_dnn_device"]
    sessions = {row["METAL TRACE REPLAY SESSION ID"] for row in native}
    assert len(sessions) == 128 and not sessions.intersection({"", "-", "nan"})
    first = min(native, key=lambda row: float(row["DEVICE FW START CYCLE"]))["METAL TRACE REPLAY SESSION ID"]
    selected = [row for row in native if row["METAL TRACE REPLAY SESSION ID"] == first]
    assert len(native) == 128 * len(selected)
    with (folder / "decode_one_replay.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([rows[start], *selected, rows[end]])


def shape(row, prefix):
    return [int(re.findall(r"\d+", row[f"{prefix}_{axis}_PAD[LOGICAL]"])[-1]) for axis in "WZYX"]


def integer_attribute(row, name):
    pattern = rf"\b{re.escape(name)}(?:['\"]\s*:\s*['\"]?|\s*=\s*)(\d+)"
    values = {int(value) for value in re.findall(pattern, row["ATTRIBUTES"])}
    assert len(values) == 1, f"Missing/conflicting native {name}: {row['ATTRIBUTES']}"
    return values.pop()


def prefill_backend_audit(folder, runner, layer):
    with (folder / "ops.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    start = next(i for i, row in enumerate(rows) if row["OP CODE"] == "PERF_PREFILL")
    end = next(i for i, row in enumerate(rows) if row["OP CODE"] == "PERF_PREFILL_END")
    native = [row for row in rows[start + 1 : end] if row["OP TYPE"] == "tt_dnn_device"]
    minimal = [row for row in native if "minimalmatmul" in row["OP CODE"].lower()]
    assert minimal, "Selected minimal prefill backend is absent from native records"
    targets = {
        "qkv": dict(
            weight=[1, 1, 2816, 8192 if layer == 0 else 9216],
            k=8 if layer == 0 else 16,
            fidelity="HiFi4" if layer == 0 else "HiFi2",
        ),
    }
    if layer == 5:
        targets["output"] = dict(weight=[1, 1, 8192, 2816], k=8, fidelity="LoFi")
    roles = {}
    for role, expected in targets.items():
        selected = [row for row in minimal if shape(row, "INPUT_1") == expected["weight"]]
        assert selected and sum(shape(row, "INPUT_0")[-2] for row in selected) == 4096
        records = []
        for row in selected:
            config = {
                key: integer_attribute(row, key)
                for key in ("M_block_size", "K_block_size", "N_block_size", "subblock_h", "subblock_w")
            }
            assert config["K_block_size"] == expected["k"] and config["N_block_size"] == 8
            assert config["subblock_h"] == 1 and config["subblock_w"] == 4
            assert config["M_block_size"] == min(4, (shape(row, "INPUT_0")[-2] + 31) // 32)
            assert row["MATH FIDELITY"] == expected["fidelity"]
            assert row["INPUT_1_DATATYPE"] == "BFLOAT8_B" and row["OUTPUT_0_DATATYPE"] == "FLOAT32"
            records.append(
                dict(
                    operation=row["OP CODE"],
                    global_call_count=row["GLOBAL CALL COUNT"],
                    config=config,
                    attributes=row["ATTRIBUTES"],
                    math_fidelity=row["MATH FIDELITY"],
                    tensor_metadata={
                        key: value for key, value in row.items() if key.startswith(("INPUT_", "OUTPUT_")) and value
                    },
                    kernel_sources={key: value for key, value in row.items() if "KERNEL SOURCE" in key and value},
                    device_kernel_ns=float(row["DEVICE KERNEL DURATION [ns]"]),
                )
            )
        roles[role] = dict(native_rows=len(records), logical_token_rows=4096, rows=records)
    assert sum(role["native_rows"] for role in roles.values()) == len(minimal), "Unclassified minimal prefill operation"
    output_weight = [1, 1, 4096 if layer == 0 else 8192, 2816]
    output_rows = [
        row for row in native if "matmul" in row["OP CODE"].lower() and shape(row, "INPUT_1") == output_weight
    ]
    assert output_rows and sum(shape(row, "INPUT_0")[-2] for row in output_rows) == 4096
    assert all(("minimalmatmul" in row["OP CODE"].lower()) == (layer == 5) for row in output_rows)
    qkv_rows = [row for row in minimal if shape(row, "INPUT_1") == targets["qkv"]["weight"]]
    useful_projection_flops = sum(
        2 * math.prod(shape(row, "INPUT_0")) * shape(row, "INPUT_1")[-1] for row in [*qkv_rows, *output_rows]
    )
    assert useful_projection_flops == (283467841536 if layer == 0 else 401579442176)
    result = dict(
        status="passed",
        runtime_sha256=RUNTIME,
        layer=layer,
        source_csv=str(folder / "ops.csv"),
        source_csv_sha256=digest(folder / "ops.csv"),
        native_minimal_rows=len(minimal),
        roles=roles,
        output_native_operation_names=sorted({row["OP CODE"] for row in output_rows}),
        useful_projection_flops_from_native_logical_shapes=useful_projection_flops,
        useful_projection_scope="QKV plus output; tied full K/V counted once. Internal K-block tail work and block padding are excluded.",
        runner_prefill_qkv=runner["precision_policy"]["prefill_qkv_projection"],
        runner_prefill_output=runner["precision_policy"]["prefill_output_projection"],
        accounting="Every native operation remains in whole-layer windows. Useful FLOPs are model logical work, independent of native matmul naming or block padding. Operand traffic uses actual CSV shapes/dtypes.",
        installed_report_config_limitation="tt-perf-report 1.3.0 recognizes Matmul for shape/FLOP accounting, but extracts block/subblock advice only from program_config/in0_block_w/out_subblock_h/w. MinimalMatmulConfig uses config/K_block_size/subblock_h/w; missing-config advice does not prove a missing config. Raw attributes and parsed native fields above are authoritative.",
    )
    write(folder / "prefill_projection_native_audit.json", result)


def postprocess(item):
    layer = item["layer"]
    folder = DOC / "tracy" / f"actual_optimized_v7_layer{layer}"
    runner_path = Path(option(item["profile"], "--output"))
    runner = read(runner_path)
    assert runner["runtime_sha256"] == RUNTIME and runner["candidate"] == {"defaults": True, "overrides": {}}
    assert runner["passed"] and runner["decode"]["passed"] and runner["decode"]["repeated_equal"]
    raw = list((folder / "raw").rglob("ops_perf_results*.csv"))
    assert len(raw) == 1, raw
    shutil.copyfile(raw[0], folder / "ops.csv")
    prefill_backend_audit(folder, runner, layer)
    command = list(item["summary"])
    command[command.index("--precision-policy") + 1] = (
        POLICY + " Runtime policy: " + json.dumps(runner["precision_policy"])
    )
    records = dict(profile_source=str(raw[0]), summary_command=command, report_commands=[])
    assert run(command, folder / "summary.log") == 0, "Summary failed"
    summary = read(folder / "whole_layer.json")
    sparse = summary.get("sparse_weight_reads", [])
    assert len(sparse) == 256 and all(row["indexed"] and row["active"] == 8 and row["groups"] == 128 for row in sparse)
    one_replay(folder)
    write(folder / "summary_commands.json", records)
    for report in item["reports"]:
        code = run(report["argv"], Path(report["stdout"]))
        records["report_commands"].append(dict(**report, returncode=code, advice_enabled=True))
        write(folder / "summary_commands.json", records)
        assert code == 0, "tt-perf-report failed"
    # The profile journal is already final for both layers; this avoids binding
    # layer 0's reconciliation to an intermediate one-command journal hash.
    code = run(item["reconciliation"], folder / "reconciliation.log")
    records["reconciliation_command"] = item["reconciliation"]
    records["reconciliation_returncode"] = code
    write(folder / "summary_commands.json", records)
    assert code == 0, "Same-run reconciliation failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", action="store_true", help="Print exact commands only; no device access or files written"
    )
    parser.add_argument(
        "--postprocess-only", action="store_true", help="Use the completed two-command v7 profile journal"
    )
    args = parser.parse_args()
    commands = prepared_commands()
    if args.plan:
        print(json.dumps(commands, indent=2))
        return
    assert digest(RUNTIME_PATH) == RUNTIME
    validation = read(DOC / "validated_v7_validation_summary.json")
    assert (
        validation["runtime_sha256"] == RUNTIME
        and validation["status"] == "current_targeted_and_inherited_gates_passed"
    )
    if not args.postprocess_only:
        assert (
            not JOURNAL.exists()
        ), "Existing profile journal: inspect it, or use --postprocess-only after both profiles finish"
        # Reuse the flag-name whitelist, then capture the current environment.
        keys = read(DOC / "validated_v5_environment_snapshot.json")["environment"]
        assert not any(os.environ.get(key) for key in keys), "Inherited profiler/watcher flags must be clear"
        env = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        env.pop("TT_METAL_WATCHER", None)
        write(
            DOC / "actual_optimized_v7_profile_environment.json",
            dict(
                environment={key: env.get(key) for key in keys},
                captured_at_utc=datetime.now(timezone.utc).isoformat(),
                driver=str(Path(__file__).resolve()),
                driver_sha256=digest(Path(__file__)),
                scope="Profile subprocess inherited environment before Tracy enables profiling; Watcher disabled",
            ),
        )
        journal = []
        for item in commands:
            profile = item["profile"]
            runner_path, raw_dir = Path(option(profile, "--output")), Path(option(profile, "-o"))
            assert (
                not runner_path.exists() and not raw_dir.parent.exists()
            ), "Do not overwrite previous profile evidence"
            raw_dir.parent.mkdir(parents=True)
            code = run(profile, runner_path.with_suffix(".log"), env)
            journal.append(dict(command=profile, returncode=code, runtime_sha256=RUNTIME))
            write(JOURNAL, journal)
            assert code == 0, f"Profile failed: layer {item['layer']}"
    journal = read(JOURNAL)
    assert len(journal) == 2 and all(entry["returncode"] == 0 for entry in journal)
    assert [entry["command"] for entry in journal] == [item["profile"] for item in commands]
    for item in commands:
        postprocess(item)
    print("Completed both v7 profiles, advice-enabled reports, and final-journal same-run reconciliations")


if __name__ == "__main__":
    main()
