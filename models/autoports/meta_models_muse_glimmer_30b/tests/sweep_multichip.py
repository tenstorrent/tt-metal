# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize candidate processes, preserving live triage before timed-out jobs stop."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main(args):
    root = Path(__file__).resolve().parents[1]
    destination = args.destination or root / "doc/multichip_decoder"
    destination.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(args.candidates.read_text())
    records = []
    for name, entry in candidates.items():
        path = destination / (name + ".json")
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.meta_models_muse_glimmer_30b.tests.run_multichip",
            "--mode",
            entry.get("mode", "replicated"),
            "--layer",
            str(entry.get("layer", 0)),
            "--length",
            str(entry.get("length", 1024)),
            "--batch",
            str(entry.get("batch", 1)),
            "--stress",
            str(entry.get("stress", 2)),
            "--continuation",
            str(entry.get("continuation", 0)),
            "--policy",
            json.dumps(entry.get("policy", {})),
            "--output",
            str(path),
        ]
        if entry.get("stack", False):
            cmd.append("--stack")
        if entry.get("microbench", False):
            cmd.append("--microbench")
        if entry.get("validate", False):
            cmd.append("--validate")
        if entry.get("perf", True):
            cmd.append("--perf")
        if entry.get("decode_perf", False):
            cmd.append("--decode-perf")
        if entry.get("pages") is not None:
            cmd += ["--pages", str(entry["pages"])]
        if entry.get("prefill_repeats") is not None:
            cmd += ["--prefill-repeats", str(entry["prefill_repeats"])]
        if entry.get("module"):
            cmd = [sys.executable, "-m", entry["module"], *entry.get("argv", [])]
        env = dict(os.environ, **entry.get("env", {}))
        for variable in entry.get("unset_env", []):
            env.pop(variable, None)
        print("START", name, flush=True)
        with path.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                code = process.wait(timeout=entry.get("timeout", 180))
            except subprocess.TimeoutExpired:
                triage = destination / (name + "_triage")
                triage.mkdir(exist_ok=True)
                triage_cmd = [
                    "timeout",
                    "120",
                    "tools/tt-triage.py",
                    "--llm-output",
                    "--llm-output-path",
                    str(triage / "tt-triage.txt"),
                    "--triage-summary-path",
                    str(triage / "triage-summary.txt"),
                ]
                with (triage / "collection.log").open("w") as out:
                    subprocess.run(triage_cmd, stdout=out, stderr=subprocess.STDOUT)
                process.terminate()
                try:
                    code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
        record = dict(name=name, command=cmd, returncode=code)
        log_text = path.with_suffix(".log").read_text() if code else ""
        clean_pcc_exit = (
            code == 1
            and entry.get("no_reset_on_clean_pcc_failure", False)
            and "AssertionError" in log_text
            and "pcc" in log_text
            and "Closing devices in cluster completed." in log_text
            and "TT_FATAL" not in log_text
            and "Timeout" not in log_text
        )
        if clean_pcc_exit:
            record["clean_pcc_rejection_without_reset"] = True
            print("PCC_REJECT", name, code, flush=True)
        elif code:
            print("FAILED", name, code, flush=True)
            with path.with_suffix(".reset.log").open("w") as log:
                rc = subprocess.run(["timeout", "180", "tt-smi", "-r"], stdout=log, stderr=subprocess.STDOUT).returncode
                ls = subprocess.run(
                    ["timeout", "60", "tt-smi", "-ls", "--local"], stdout=log, stderr=subprocess.STDOUT
                ).returncode
                smoke = None
                if not rc and not ls:
                    smoke = subprocess.run(
                        [
                            "timeout",
                            "60",
                            sys.executable,
                            "-c",
                            "import ttnn; ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING); "
                            "m=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0); "
                            'ttnn.close_mesh_device(m); print("MESH_SMOKE_OK")',
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ).returncode
            record.update(reset_returncode=rc, list_returncode=ls, mesh_smoke_returncode=smoke)
            if rc or ls or smoke:
                records.append(record)
                args.output.write_text(json.dumps(records, indent=2) + "\n")
                raise RuntimeError("Hardware recovery requires inspection")
        else:
            result = json.loads((destination / entry.get("result", name + ".json")).read_text())
            if not isinstance(result, dict):
                result = {"result_rows": len(result)}
            record.update(
                prefill_ms=result.get("prefill_ms"),
                decode_ms=result.get("decode_ms"),
                prefill_pcc=result.get("prefill_pcc"),
                decode_pcc=result.get("decode_pcc"),
            )
            print("PASS", name, record.get("prefill_ms"), record.get("decode_ms"), flush=True)
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        if code and args.stop_on_failure:
            raise RuntimeError(f"Required case failed: {name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--destination", type=Path)
    p.add_argument("--stop-on-failure", action="store_true")
    main(p.parse_args())
