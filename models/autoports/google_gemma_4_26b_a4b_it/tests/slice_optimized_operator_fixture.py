# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Slice recorded 4096/128 inputs into one-replay operator-audit fixtures on CPU.

These fixtures retain the complete prefill and first actual decode input. They
are operator-audit evidence only, never headline or telemetry measurements.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

SCOPE = "one_replay_operator_audit_only"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slice_fixture(source, output_dir, layer):
    source_manifest = source.with_suffix(".json")
    source_digest = sha256(source)
    manifest = json.loads(source_manifest.read_text())
    if manifest["fixture_sha256"] != source_digest:
        raise ValueError(f"Source fixture hash disagrees with its manifest: {source}")
    fixture = torch.load(source, map_location="cpu", weights_only=True)
    metadata = fixture["metadata"]
    if (metadata["layer"], metadata["length"], metadata["steps"]) != (layer, 4096, 128):
        raise ValueError("Operator audit requires the matching layer's original 4096/128 fixture")
    for prefix, tokens in (("prefill", 4096), ("decode", 128)):
        for key in (prefix, f"raw_{prefix}"):
            tensor = fixture[key]
            if tensor.shape != (1, tokens, 2816) or tensor.dtype != torch.float32:
                raise ValueError(f"Unexpected source shape/dtype: {key}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite source tensor: {key}")
        if not torch.equal(fixture[prefix], fixture[f"raw_{prefix}"].bfloat16().float()):
            raise ValueError(f"Source {prefix} is not the recorded BF16 transport of its raw inputs")
    if fixture["token_ids"].shape != (4224,):
        raise ValueError("Expected 4224 source token IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"operator_audit_layer{layer}_4096_1.pt"
    if path.resolve() == source.resolve():
        raise ValueError("Source and derived fixture paths must differ")
    derived = {
        "source_fixture": str(source),
        "source_fixture_sha256": source_digest,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "source_length": 4096,
        "source_steps": 128,
        "decode_slice": [0, 1],
        "token_slice": [0, 4097],
        "transform": "Exact CPU tensor copies; no activation recomputation or dtype conversion",
        "helper_sha256": sha256(__file__),
    }
    metadata = copy.deepcopy(metadata)
    metadata.update(
        steps=1,
        evaluation_scope=SCOPE,
        headline_eligible=False,
        telemetry_eligible=False,
        derived_from=derived,
    )
    sliced = {
        "metadata": metadata,
        "prefill": fixture["prefill"].clone(),
        "decode": fixture["decode"][:, :1].clone(),
        "raw_prefill": fixture["raw_prefill"].clone(),
        "raw_decode": fixture["raw_decode"][:, :1].clone(),
        "token_ids": fixture["token_ids"][:4097].clone(),
    }
    torch.save(sliced, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("prefill", "decode", "raw_prefill", "raw_decode", "token_ids"):
        if not torch.equal(loaded[key], sliced[key]):
            raise ValueError(f"Saved fixture differs from the exact source slice: {key}")
    if sha256(source) != source_digest:
        raise ValueError("Source fixture changed while slicing")
    result = {
        **metadata,
        "fixture": str(path),
        "fixture_sha256": sha256(path),
        "prefill_shape": list(loaded["prefill"].shape),
        "decode_shape": list(loaded["decode"].shape),
        "token_count": 4097,
        "decode_first_token_ids": loaded["token_ids"][4096:].tolist(),
        "finite": True,
        "saved_tensor_equality_verified": True,
    }
    path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fixture-template", required=True, help="Original fixture path containing {layer}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", choices=(0, 5), default=[0, 5])
    args = parser.parse_args()
    if "{layer}" not in args.input_fixture_template:
        parser.error("Input template must contain {layer}")
    torch.set_num_threads(2)
    results = [
        slice_fixture(Path(args.input_fixture_template.format(layer=layer)), args.output_dir, layer)
        for layer in sorted(set(args.layers))
    ]
    manifest = {
        "command": [sys.executable, *sys.argv],
        "evaluation_scope": SCOPE,
        "headline_eligible": False,
        "telemetry_eligible": False,
        "fixtures": results,
    }
    output = args.output_dir / "operator_audit_fixture_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "fixtures": [result["fixture"] for result in results]}, indent=2))


if __name__ == "__main__":
    main()
