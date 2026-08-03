# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed optimized-decoder latency and Tracy-signpost harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.functional_decoder_perf import _decode, _prefill
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _real_layer_one_state,
    _synthetic_state,
)
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_optimized_decoder import _real_dense_layer_zero_state
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import FunctionalDecoder
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import MODEL_ID, OptimizedDecoder


def _git(repo_root, *args):
    return subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_provenance(implementation):
    repo_root = Path(__file__).resolve().parents[4]
    source_paths = [
        Path("models/autoports/coherelabs_north_mini_code_1_0/tt/optimized_decoder.py"),
        Path("models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py"),
    ]
    if implementation == "functional":
        source_paths.append(Path("models/autoports/coherelabs_north_mini_code_1_0/tt/functional_decoder.py"))
    sources = {}
    for relative_path in source_paths:
        absolute_path = repo_root / relative_path
        computed_blob_oid = _git(repo_root, "hash-object", str(relative_path))
        object_available = (
            subprocess.run(
                ("git", "cat-file", "-e", computed_blob_oid),
                cwd=repo_root,
                capture_output=True,
            ).returncode
            == 0
        )
        head_lookup = subprocess.run(
            ("git", "rev-parse", f"HEAD:{relative_path}"),
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        sources[str(relative_path)] = {
            "sha256": hashlib.sha256(absolute_path.read_bytes()).hexdigest(),
            # hash-object computes an object ID but does not store an untracked
            # source.  Keep that distinction explicit so provenance never
            # implies an unavailable historical source can be recovered.
            "computed_git_blob_oid": computed_blob_oid,
            "git_object_available": object_available,
            "head_git_blob_oid": head_lookup.stdout.strip() if head_lookup.returncode == 0 else None,
            "git_status": _git(repo_root, "status", "--short", "--", str(relative_path)),
        }
    return {
        "git_head": _git(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git(repo_root, "branch", "--show-current"),
        "sources": sources,
    }


def _correctness_binding(args):
    """Validate and content-bind manually supplied correctness evidence."""

    required = (
        args.pcc is not None,
        args.pcc_evidence is not None,
        args.pcc_scope is not None,
        args.pcc_status is not None,
        args.pcc_threshold is not None,
    )
    any_metadata = any(required) or args.pcc_note is not None
    if any_metadata and not all(required):
        raise ValueError(
            "--pcc, --pcc-evidence, --pcc-scope, --pcc-status, and --pcc-threshold must be supplied together"
        )
    if not any_metadata:
        return None
    if not -1.0 <= args.pcc <= 1.0 or not -1.0 <= args.pcc_threshold <= 1.0:
        raise ValueError("PCC value and threshold must be in [-1, 1]")
    if args.pcc_status == "pass" and args.pcc < args.pcc_threshold:
        raise ValueError("PCC status says pass but the value is below the threshold")
    if args.pcc_status == "fail" and args.pcc >= args.pcc_threshold:
        raise ValueError("PCC status says fail but the value meets the threshold")

    repo_root = Path(__file__).resolve().parents[4]
    evidence = args.pcc_evidence.expanduser()
    if not evidence.is_absolute():
        evidence = (Path.cwd() / evidence).resolve()
    else:
        evidence = evidence.resolve()
    try:
        evidence_relative = evidence.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"PCC evidence must be inside the repository: {evidence}") from error
    if not evidence.is_file():
        raise ValueError(f"PCC evidence does not exist or is not a file: {evidence_relative}")
    return {
        "value": args.pcc,
        "scope": args.pcc_scope,
        "status": args.pcc_status,
        "threshold": args.pcc_threshold,
        "evidence": {
            "repo_relative_path": str(evidence_relative),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        },
        "note": args.pcc_note,
        # For exact_workload this is an explicit assertion that the evidence
        # exercised these same timing parameters.  Other scopes deliberately
        # remain labelled controls instead of masquerading as exact PCC.
        "timing_workload": {
            "implementation": args.implementation,
            "candidate": args.candidate if args.implementation == "optimized" else None,
            "mode": args.mode,
            "batch": args.batch,
            "sequence": args.sequence,
            "layer": args.layer,
            "real_weights": args.real_weights,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=1, choices=(0, 1, 4))
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--candidate", default="default")
    parser.add_argument("--implementation", choices=("functional", "optimized"), default="optimized")
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pcc", type=float)
    parser.add_argument("--pcc-evidence", type=Path)
    parser.add_argument(
        "--pcc-scope",
        choices=("exact_workload", "same_candidate_control", "cross_workload_control"),
    )
    parser.add_argument("--pcc-status", choices=("pass", "fail", "informational"))
    parser.add_argument("--pcc-threshold", type=float)
    parser.add_argument("--pcc-note")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.real_weights and args.layer not in (0, 1):
        raise ValueError(
            "the repo-local partial checkpoint contains real weights only for representative layers 0 and 1"
        )
    correctness_binding = _correctness_binding(args)
    provenance = _source_provenance(args.implementation)
    argv = list(sys.argv)

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION)
    if args.real_weights:
        state = _real_dense_layer_zero_state() if args.layer == 0 else _real_layer_one_state()
    else:
        state = _synthetic_state(config, args.layer, sparse_weights=config.mlp_layer_types[args.layer] == "sparse")
    max_cache_len = args.sequence if args.mode == "prefill" else 32
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128 * 1024 * 1024)
    effective_configuration = None
    try:
        decoder_cls = FunctionalDecoder if args.implementation == "functional" else OptimizedDecoder
        decoder_kwargs = dict(
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=max_cache_len,
        )
        if decoder_cls is OptimizedDecoder:
            decoder_kwargs["candidate"] = args.candidate
        decoder = decoder_cls.from_state_dict(state, **decoder_kwargs)
        if decoder_cls is OptimizedDecoder:
            effective_configuration = decoder.effective_configuration()
        if args.mode == "prefill":
            result = _prefill(
                decoder,
                mesh_device,
                config,
                sequence=args.sequence,
                warmups=args.warmups,
                iterations=args.iterations,
            )
        else:
            result = _decode(
                decoder,
                mesh_device,
                config,
                warmups=args.warmups,
                iterations=args.iterations,
            )
    finally:
        ttnn.close_mesh_device(mesh_device)

    result.update(
        {
            "decoder": args.implementation,
            "candidate": args.candidate if args.implementation == "optimized" else None,
            "real_weights": args.real_weights,
            "mode": args.mode,
            "batch": args.batch,
            "sequence": args.sequence,
            "layer": args.layer,
            "model_revision": REAL_REVISION,
            "effective_configuration": effective_configuration,
            "provenance": provenance,
            "argv": argv,
            "correctness_binding": correctness_binding,
            # Compatibility fields are derived from the validated binding;
            # new readers should consume correctness_binding.
            "pcc": args.pcc,
            "pcc_evidence": (
                correctness_binding["evidence"]["repo_relative_path"] if correctness_binding is not None else None
            ),
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
