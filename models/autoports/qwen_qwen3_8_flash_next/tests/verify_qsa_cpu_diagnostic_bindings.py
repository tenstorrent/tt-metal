# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bind completed two-row CPU diagnostics to exact fixtures and source text.

No device access or diagnostic rerun. Preserve the historical diagnostic script
bytes while independently checking every input identity they consumed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from .probe_qsa import load_reference


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ttuser/qwen38-flash"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    stage = Path(__file__).parents[1]
    doc = stage / "doc/functional_decoder"
    source_path = args.root / "qsa_cpu_prefill_262144.pt"
    device_path = args.root / "qsa_final_device_262144.pt"
    source = torch.load(source_path, weights_only=True, mmap=True, map_location="cpu")
    device = torch.load(device_path, weights_only=True, mmap=True, map_location="cpu")
    source_key = source["key"]
    assert source_key["snapshot"] == str(args.snapshot.resolve())
    assert source_key["length"] == source_key["capacity"] == device["length"] == 262144
    assert source_key["decode_steps"] == 0 and source_key["seed"] == 1979
    metadata = device["reference_fixture"]
    assert metadata["source_key"] == source_key
    assert metadata["path"] == str(source_path)
    assert metadata["key"] == metadata["requested_key"] == {**source_key, "length": 262143, "decode_steps": 1}
    assert metadata["boundary_split_reuse"] is True
    assert tuple(device["actual"].shape) == tuple(source["expected"].shape) == (1, 262144, 2560)
    assert source["hidden"].shape == source["expected"].shape
    assert source["hidden"].dtype == source["expected"].dtype == device["actual"].dtype == torch.bfloat16
    _, _, _, _, provenance = load_reference(args.snapshot)
    for key in ("reference_sha256", "config_sha256", "transformers"):
        assert source_key[key] == provenance[key]
    report = {
        "scope": __doc__,
        "source_key": source_key,
        "device_fixture_metadata": metadata,
        "source_fixture_sha256": digest(source_path),
        "device_output_sha256": digest(device_path),
        "reference_provenance": provenance,
        "windows": {},
        "diagnostics": {},
        "verifier_sha256": digest(Path(__file__)),
        "device_executed": False,
    }
    for prefix in (105568, 58528):
        path = args.root / f"qsa_score_window_{prefix}.pt"
        fixture = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        assert fixture["source_key"] == source_key
        assert fixture["source_path"] == str(source_path)
        assert fixture["provenance"] == provenance
        assert (fixture["prefix"], fixture["rows"], fixture["capacity"]) == (prefix, 32, 262144)
        assert torch.equal(fixture["hidden"], source["hidden"][:, prefix : prefix + 32])
        assert torch.equal(fixture["saved_expected"], source["expected"][:, prefix : prefix + 32])
        report["windows"][str(prefix)] = {
            "path": str(path),
            "sha256": digest(path),
            "source_key_exact": True,
            "hidden_exact": True,
            "saved_expected_exact": True,
            "continued_expected_bitexact": torch.equal(fixture["continued_expected"], fixture["saved_expected"]),
        }
    for stem in ("qsa_two_row_precision_cpu", "qsa_two_row_midpoint_cpu"):
        script = stage / "tests/diagnostic_sources" / (stem + ".py.txt")
        artifact = doc / (stem + ".json")
        value = json.loads(artifact.read_text())
        assert value["source_sha256"] == digest(script)
        assert value["provenance"] == provenance
        assert set(value["rows"]) == {"105584", "58531"}
        report["diagnostics"][stem] = {
            "artifact": str(artifact),
            "artifact_sha256": digest(artifact),
            "preserved_script": str(script),
            "preserved_script_sha256": digest(script),
            "recorded_script_hash_exact": True,
        }
    assert "ttnn" not in sys.modules
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "windows": report["windows"]}), flush=True)


if __name__ == "__main__":
    main()
