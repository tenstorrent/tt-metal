# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only acceptance of separate, matched diagnostics-off stage-five timing."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/optimized_multichip_decoder"


def load_timing_map():
    return json.loads((D / "review_timing_map.json").read_text())


def command_metadata(path):
    return json.loads((D / (path.removesuffix(".json") + "_command.json")).read_text())


def validate_diagnostics(data, metadata):
    assert data["tracking"] == "0", "timing requires allocation tracking disabled"
    assert not data["profile"] and data["benchmark"], "timing requires an unprofiled benchmark"
    env = metadata["env"]
    for name in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS"):
        assert env.get(name) == "0", f"missing explicit disabled setting: {name}"
    command = metadata["command"]
    for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_SLOW_DISPATCH_MODE"):
        assert any(command[i : i + 2] == ["-u", name] for i in range(len(command) - 1)), name
    assert "--profile" not in command and "tracy" not in command
    assert metadata["regime"] == "serial" and metadata["devices"] == [0, 1, 2, 3]


def option(metadata, name):
    command = metadata["command"]
    return command[command.index(name) + 1] if name in command else None


def validate_pair(before_path, after_path):
    before, after = [json.loads((D / p).read_text()) for p in (before_path, after_path)]
    old_meta, new_meta = [command_metadata(p) for p in (before_path, after_path)]
    for data, meta in ((before, old_meta), (after, new_meta)):
        validate_diagnostics(data, meta)
    for key in (
        "batch",
        "prefix_length",
        "capacity",
        "steps",
        "layout",
        "io_contract",
        "starts",
        "state_checks",
        "sample_prefill",
        "layer",
        "prefill_warmup",
        "prefill_repeats",
        "trace",
    ):
        assert before.get(key) == after.get(key), (before_path, after_path, key)
    for flag in ("--lengths", "--reference"):
        assert option(old_meta, flag) == option(new_meta, flag), (before_path, after_path, flag)
    # The frozen decoder is selected by a wrapper/flag; timing and RNG code must match.
    for source in ("tests/multichip_probe.py", "tests/multichip_packed_stack.py"):
        assert old_meta["source_sha256"][source] == new_meta["source_sha256"][source], source


def pcc_values(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if "pcc" in key and isinstance(child, (float, int)):
                yield child
            else:
                yield from pcc_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from pcc_values(child)


def audit_timing(mapping):
    ledger = [json.loads(line) for line in (D / "device_jobs.jsonl").read_text().splitlines()]
    manifest = json.loads((D / "final_manifest.json").read_text())
    production = manifest["source_sha256"]
    artifacts = []
    for original, path in mapping.items():
        data = json.loads((D / path).read_text())
        meta = command_metadata(path)
        validate_diagnostics(data, meta)
        completed = [row for row in ledger if row["label"] == meta["label"]]
        assert len(completed) == 1 and completed[0]["exit_code"] == 0, path
        assert completed[0]["command"] == meta["command"]
        # Both roles run in the same checkout; the before role explicitly chooses frozen source.
        for source, digest in production.items():
            assert meta["source_sha256"][source] == digest, (path, source)
        if original.startswith("final_v2_"):
            assert data["source_sha256"] == production["tt/multichip_decoder.py"]
            assert not data.get("stage4", False) and not data.get("baseline", False)
        else:
            assert data.get("stage4") or data.get("measurement_role") == "before stage-five optimization, TP4"
            frozen = ROOT / "tests/multichip_stage4_reference.py"
            assert (
                meta["source_sha256"]["tests/multichip_stage4_reference.py"]
                == hashlib.sha256(frozen.read_bytes()).hexdigest()
            )
        values = list(pcc_values(data["records"]))
        assert values and all(math.isfinite(v) and v >= 0.995 for v in values), path
        artifacts.append(
            {
                "original": original,
                "path": path,
                "sha256": hashlib.sha256((D / path).read_bytes()).hexdigest(),
                "minimum_pcc": min(values),
                "pcc_values": len(values),
                "job": meta["label"],
            }
        )
    result = {
        "status": "passed",
        "diagnostics": {
            "tracking": "0",
            "tracebacks": "0",
            "watcher": "unset",
            "device_profiler": "unset",
            "tracy": False,
        },
        "regime": "serial",
        "devices": [0, 1, 2, 3],
        "artifacts": artifacts,
    }
    return result
