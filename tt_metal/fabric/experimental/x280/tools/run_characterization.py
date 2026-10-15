#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run explicitly selected fixed configurations in an already prepared device epoch.

Each benchmark boots resident idle and prepares its own inputs; device resets,
ownership verification and recovery are the operator's.
A failure stops the sequence; each invocation retains a fresh attempt directory.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from string import Template
import xml.etree.ElementTree as ET

from tabulate_suites import ParseError, read_attempt


def sha256(path):
    """Hash a retained file without changing it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_attempt(path, record):
    """Atomically update this runner's attempt record; retain its last state on failure."""
    pending = path.with_suffix(".tmp")
    with pending.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    pending.replace(path)


def command_for(manifest, config, values):
    """Resolve argv/environment as data and clear inherited suite selection knobs and dispatch mode.

    A null environment value unsets that variable for the run.
    """
    prefixes = ("X280_", "PACKET_INJECTION_", "PIPELINE_STAGE_", "P2P_LATENCY_", "TRANSPORT_BW_")
    environment = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    environment.pop("TT_METAL_WATCHER", None)
    environment.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
    selected = dict(manifest["common_environment"], **config["environment"])
    for key, value in selected.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = Template(value).substitute(values)
    return [Template(a).substitute(values) for a in config["argv"]], environment


def validate_test_report(path, expected):
    """Reject zero tests, skips, failures and a different filtered test."""
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if len(cases) != 1:
        raise ValueError(f"expected one filtered test, got {len(cases)}")
    case = cases[0]
    if case.get("classname") + "." + case.get("name") != expected:
        raise ValueError("gtest identity differs from requested filter")
    if case.get("status") != "run" or case.get("result") != "completed" or list(case):
        raise ValueError("gtest failed, skipped or did not complete")


def record_host_label(campaign, label):
    """Record the campaign's host label on first use. `label` None collects unlabelled: nothing is
    written or checked, even on a campaign that already has a host.json. Raise ValueError naming
    both labels when they differ, and when an existing host.json is not a JSON object holding a
    string host_label."""
    if label is None:
        return
    path = campaign / "host.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if not isinstance(recorded, dict) or not isinstance(recorded.get("host_label"), str):
            raise ValueError(f"{path} does not record a host_label")
        recorded = recorded["host_label"]
        if recorded != label:
            raise ValueError(f"campaign was collected on host {recorded!r}, not {label!r}")
        return
    with path.open("x") as output:
        output.write(json.dumps({"host_label": label}, indent=2) + "\n")


def run_one(manifest, config, values, campaign, engine, host_label=None):
    """Retain one attempt and validate its exact record set; never reset hardware. The attempt
    record carries `host_label` only when one is given."""
    attempt = Path(tempfile.mkdtemp(prefix=config["id"] + ".", dir=campaign / "attempts"))
    values = dict(values, CAMPAIGN=str(attempt))
    record = {"configuration": config["id"]}
    if host_label is not None:
        record["host_label"] = host_label
    record.update(status="running", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_attempt(attempt / "attempt.json", record)
    try:
        argv, environment = command_for(manifest, config, values)
        filters = [a.split("=", 1)[1] for a in argv if a.startswith("--gtest_filter=")]
        if len(filters) != 1:
            raise ValueError("one exact gtest filter is required")
        argv.append("--gtest_output=xml:" + str(attempt / "gtest.xml"))
        record.update(
            argv=argv,
            environment={
                k: environment.get(k) for k in set(manifest["common_environment"]) | set(config["environment"])
            },
        )
        record["producer_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=engine, text=True).strip()
        record["binary_sha256"] = sha256(argv[0])
        record["firmware_sha256"] = {k: sha256(v) for k, v in environment.items() if k.endswith("_FW_IMAGE")}
        write_attempt(attempt / "attempt.json", record)
        with (attempt / "run.log").open("x") as log:
            result = subprocess.run(
                argv, env=environment, cwd=engine, stdout=log, stderr=subprocess.STDOUT, timeout=180
            )
        record["returncode"] = result.returncode
        if result.returncode:
            raise ValueError(f"benchmark exit {result.returncode}; inspect run.log before device reuse")
        validate_test_report(attempt / "gtest.xml", filters[0])
        read_attempt(config, sorted((attempt / "raw").glob("*.jsonl")))
        record["status"] = "passed"
    except (ValueError, KeyError, OSError, ET.ParseError, ParseError, subprocess.SubprocessError) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        print(f"{config['id']}: {error}", flush=True)
    record["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    artifacts = [attempt / "run.log", attempt / "gtest.xml", *sorted((attempt / "raw").glob("*.jsonl"))]
    record["artifacts_sha256"] = {str(path.relative_to(attempt)): sha256(path) for path in artifacts if path.is_file()}
    write_attempt(attempt / "attempt.json", record)
    print(f"{record['status']}: {config['id']} ({attempt})", flush=True)
    return record["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "characterization_manifest.json"
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--host-build", type=Path, required=True)
    parser.add_argument("--fw-build", type=Path, required=True)
    parser.add_argument("--metal-src", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--id", action="append", required=True)
    parser.add_argument(
        "--host-label",
        help="names the collecting host in the campaign record; a campaign refuses a second label",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if set(manifest) != {"common_environment", "configurations"}:
        parser.error("manifest requires only common_environment and configurations")
    for config in manifest["configurations"]:
        if set(config) != {"id", "member", "argv", "environment"}:
            parser.error("each configuration requires only id, member, argv and environment")
    engine = Path(__file__).resolve().parents[5]
    values = {
        "HOST_BUILD": str(args.host_build.resolve()),
        "FW_BUILD": str(args.fw_build.resolve()),
        "METAL_SRC": str(args.metal_src.resolve()),
        "DEVICE": args.device,
        "CAMPAIGN": str(args.campaign.resolve()),
    }
    configs = {c["id"]: c for c in manifest["configurations"]}
    if not configs or len(configs) != len(manifest["configurations"]):
        parser.error("configurations must be nonempty and have unique IDs")
    selected = [configs[key] for key in args.id]
    if len(set(args.id)) != len(args.id):
        parser.error("duplicate configuration selection")
    if args.dry_run:
        for config in selected:
            argv, environment = command_for(manifest, config, values)
            print(
                json.dumps(
                    {
                        "configuration": config["id"],
                        "argv": argv,
                        "environment": {
                            k: environment.get(k)
                            for k in set(manifest["common_environment"]) | set(config["environment"])
                        },
                    },
                    indent=2,
                )
            )
        return 0
    if args.host_label == "":
        parser.error("--host-label must not be empty")
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=engine).returncode:
        parser.error("commit producer changes before collection")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "tt_metal/fabric/experimental/x280"],
        cwd=engine,
        text=True,
    )
    if untracked:
        parser.error("untracked X280 inputs must be committed before collection")
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    frozen = campaign / "manifest.json"
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        parser.error("campaign manifest differs; choose a new campaign directory")
    if not frozen.exists():
        with frozen.open("x") as output:
            output.write(json.dumps(manifest, indent=2) + "\n")
    try:
        record_host_label(campaign, args.host_label)
    except ValueError as error:
        parser.error(f"{error}; choose a new campaign directory")
    (campaign / "attempts").mkdir(exist_ok=True)
    for config in selected:
        if not run_one(manifest, config, values, campaign, engine, args.host_label):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
