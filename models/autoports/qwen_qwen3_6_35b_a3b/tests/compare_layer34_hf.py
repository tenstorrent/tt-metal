# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only matched comparison against an existing canonical HF layer34 oracle.

No model execution or TTNN import. The historical oracle runner is verified
against its exact Git source, HF/reference/cache sources against their recorded
hashes, and saved inputs bitwise against every compared candidate. This emits
aggregate accuracy and per-step diagnostics, not stage acceptance or a waiver.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DOC = ROOT / "doc/optimized_multichip_decoder"
ORACLE_RESULT = ROOT / "doc/fused_decoder/endurance/v6_ordered_precision_inline_selector_v2_layer34_seed154.json"
HISTORICAL_RUNNER_COMMIT = "e3127d3f"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    # Existing oracle metadata stores the installed TorchVersion string subclass.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, weights_only=True, map_location="cpu")


def correlations(a, b):
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    a, b = a.double().flatten(1), b.double().flatten(1)
    same = (a == b).all(dim=1)
    a, b = a - a.mean(dim=1, keepdim=True), b - b.mean(dim=1, keepdim=True)
    result = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1))
    result[same] = 1.0
    assert torch.isfinite(result).all()
    return result.tolist()


def metrics(reference, actual, start):
    values = correlations(reference, actual)
    bad = [{"step": i, "position": start + i, "pcc": v} for i, v in enumerate(values) if v < 0.995]
    anomalies = [row for row in bad if row["pcc"] < 0.99]
    aggregate = correlations(reference.reshape(1, -1), actual.reshape(1, -1))[0]
    return dict(
        aggregate_pcc=aggregate,
        aggregate_meets_0_995=aggregate >= 0.995,
        per_step_diagnostic_meets_0_99=not anomalies,
        below_0_99_diagnostic=anomalies,
        first_below_0_99_diagnostic=anomalies[0] if anomalies else None,
        per_step_whole_output_pcc=values,
        minimum_pcc=min(values),
        minimum_step=values.index(min(values)),
        below_0_995=bad,
        first_below_0_995=bad[0] if bad else None,
        bit_equal=torch.equal(reference, actual),
    )


def run(args):
    begin = time.monotonic()
    torch.set_num_threads(args.threads)
    record = read(args.oracle_result)
    key = record["cpu_oracle_key"]
    assert key["revision"] == REVISION and str(key["torch_version"]) == str(torch.__version__)
    parameters = key["parameters"]
    assert parameters["layer"] == 34 and parameters["seed"] == 154 and parameters["prompt_length"] == 128
    cache_path = Path(record["cpu_oracle_cache"])
    if not cache_path.is_absolute():
        cache_path = REPO / cache_path
    oracle = load(cache_path)
    assert oracle["key"] == key
    historical_commit = subprocess.check_output(
        ["git", "rev-parse", HISTORICAL_RUNNER_COMMIT], cwd=REPO, text=True
    ).strip()
    source_checks = []
    for name, expected in key["source_hashes"].items():
        path = Path(name)
        if path.name == "stress_fused_decoder.py":
            relative = str(path.relative_to(REPO))
            historical = subprocess.check_output(["git", "show", f"{historical_commit}:{relative}"], cwd=REPO)
            actual = hashlib.sha256(historical).hexdigest()
            assert actual == expected, "Historical HF oracle runner source changed"
            source_checks.append(
                dict(
                    path=name,
                    sha256=actual,
                    identity="exact historical Git source",
                    commit=historical_commit,
                    current_sha256=digest(path),
                )
            )
        else:
            actual = digest(path)
            assert actual == expected, f"Canonical HF/reference/cache source changed: {name}"
            source_checks.append(dict(path=name, sha256=actual, identity="exact current source"))
    assert {Path(row["path"]).name for row in source_checks} == {
        "stress_fused_decoder.py",
        "reference.py",
        "modeling_qwen3_5_moe.py",
        "cache_utils.py",
    }
    artifacts = args.artifact or sorted(args.doc.glob("repair_*l34.json"))
    assert artifacts, "No layer34 artifacts requested"
    rows = []
    baseline_outputs = None
    for artifact in artifacts:
        result = read(artifact)
        provenance_path = Path(str(artifact) + ".provenance.json")
        provenance = read(provenance_path)
        tensor_path = Path(result["tensor_artifact"])
        if not tensor_path.is_absolute():
            tensor_path = REPO / tensor_path
        saved = load(tensor_path)
        signature = saved["signature"]
        for field in ("layer", "seed", "length", "decode_steps", "batch", "stack_depth"):
            assert signature[field] == result[field], f"{artifact}: mismatched tensor signature"
        assert (
            signature["layer"],
            signature["seed"],
            signature["length"],
            signature["batch"],
            signature["stack_depth"],
        ) == (34, 154, 128, 1, 1)
        count = signature["decode_steps"]
        assert 0 < count <= len(oracle["refs"])
        assert torch.equal(saved["inputs"]["x"], oracle["prefix"]), f"{artifact}: HF prefix mismatch"
        expected_tokens = oracle["tokens"][:, :count].transpose(0, 1).unsqueeze(2)
        assert torch.equal(saved["inputs"]["xd"], expected_tokens), f"{artifact}: HF decode input mismatch"
        hf = torch.stack(oracle["refs"][:count])
        baseline, mesh = saved["baseline"]["decode"], saved["multichip"]["decode"]
        if baseline_outputs is None:
            baseline_outputs = baseline.clone()
        assert torch.equal(baseline_outputs, baseline), "Compared runs have different single-chip controls"
        cross = metrics(baseline, mesh, 128)
        assert len(result["decode_pcc"]) == count
        assert all(abs(a - b) < 1e-10 for a, b in zip(result["decode_pcc"], cross["per_step_whole_output_pcc"]))
        row = dict(
            artifact=str(Path(artifact).resolve()),
            tensor_artifact=str(tensor_path.resolve()),
            provenance=str(provenance_path.resolve()),
            command=provenance["command"],
            measured_mesh_source_sha256=provenance["source_sha256"]["tt/multichip_decoder.py"],
            signature=signature,
            hf_inputs_bit_equal=True,
            same_singlechip_control=True,
            baseline_vs_hf=metrics(hf, baseline, 128),
            mesh_vs_hf=metrics(hf, mesh, 128),
            mesh_vs_baseline=cross,
            prefill_pcc_vs_hf={
                side: correlations(oracle["ref_p"], saved[side]["prefill"])[0] for side in ("baseline", "multichip")
            },
            saved_state_pcc_vs_singlechip=result["state_checks"],
            state_limit="Only terminal TT states are saved; this oracle has no per-step intermediate or state snapshots",
        )
        rows.append(row)
        print(
            Path(artifact).stem,
            "HF aggregate",
            f"{row['mesh_vs_hf']['aggregate_pcc']:.9f}",
            "diagnostic min",
            f"{row['mesh_vs_hf']['minimum_pcc']:.9f}",
            "first diagnostic <0.99",
            row["mesh_vs_hf"]["first_below_0_99_diagnostic"],
            flush=True,
        )
        del saved
    assert not any(name == "ttnn" or name.startswith("ttnn.") for name in sys.modules)
    report = dict(
        execution="CPU-only saved-tensor comparison; no HF computation or TTNN import",
        checkpoint_revision=REVISION,
        oracle_result=str(args.oracle_result.resolve()),
        oracle_cache=str(cache_path.resolve()),
        oracle_key=key,
        source_checks=source_checks,
        comparison_script=str(Path(__file__).resolve()),
        comparison_script_sha256=digest(__file__),
        command=sys.argv,
        torch_version=str(torch.__version__),
        elapsed_seconds=time.monotonic() - begin,
        rows=rows,
        contract=dict(
            scope="Additional layer34 S128/D192 aggregate sensitivity stress; existing mandatory fixtures unchanged",
            aggregate_pcc_threshold=0.995,
            per_step_diagnostic_pcc_floor=0.99,
            per_step_below_0_995="Informational only for this additional aggregate stress, not a new failure gate",
            top1_token_agreement="Not applicable: standalone hidden-state channels are not vocabulary logits",
        ),
        interpretation="Aggregate PCC >=0.995 and per-step diagnostic minimum >=0.99 satisfy this additional "
        "stress check. Default/selector satisfy both; no HF waiver is needed. Native diagnostic anomalies below "
        "0.99 require repair or rejection even when aggregate PCC passes. Existing stage4 mandatory per-forward "
        "fixtures/checker and stage5 gates are unchanged. Historical HF oracle references precede any TT device execution; "
        "this comparison alone does not establish stage acceptance.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("MATCHED_HF_COMPARISON_WRITTEN", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--oracle-result", type=Path, default=ORACLE_RESULT)
    parser.add_argument("--artifact", type=Path, action="append")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DOC / "layer34_hf_comparison.json")
    run(parser.parse_args())
