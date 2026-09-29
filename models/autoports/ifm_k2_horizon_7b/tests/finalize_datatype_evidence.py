"""CPU-only final selection/provenance checks and concise stage report."""

import ast
import hashlib
import json
from pathlib import Path

from .report_datatype_sweep import main as report

DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/datatype_sweep")


def read(name):
    return json.loads((DOC / name).read_text())


def verify_source_normalization():
    """Verify exact hook-only transforms without rewriting measured provenance."""
    receipt = read("post_measurement_source_normalization.json")
    for row in receipt["sources"]:
        current = Path(row["path"]).read_text()
        assert hashlib.sha256(current.encode()).hexdigest() == row["current_sha256"]
        measured = current
        for replacement in reversed(row["exact_replacements"]):
            assert measured.count(replacement["after"]) == 1
            measured = measured.replace(replacement["after"], replacement["before"])
        assert hashlib.sha256(measured.encode()).hexdigest() == row["measured_sha256"]
        old_tree, new_tree = ast.parse(measured), ast.parse(current)
        if row["path"].endswith("/tt/model.py"):
            assert not any(
                isinstance(node, ast.Name) and node.id == "optimized_full_model_policy" for node in ast.walk(old_tree)
            )
        for tree in (old_tree, new_tree):
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if row["path"].endswith("/tt/model.py") and node.module == "optimized_full_model_policy":
                        node.names = [name for name in node.names if name.name != "optimized_full_model_policy"]
                    node.names.sort(key=lambda name: (name.name, name.asname or ""))
        assert ast.dump(old_tree) == ast.dump(new_tree)
    return receipt


def main():
    selected = read("selected_precision_config.json")
    final = read("final_selected.json")
    candidate = read(f"qualifications/{selected['config_id']}_qualification.json")
    baseline = read("qualifications/baseline_mixed_hifi2_lofi_qualification.json")
    token = read("post_selection_token_out.json")
    context = read("context_selected.json")
    head_geometry = read("head_cores_full36.json")
    assert head_geometry["pass"]
    assert head_geometry["precision_config"] == selected
    assert head_geometry["medians"]["16"] >= head_geometry["medians"]["8"]
    assert final["precision_config"] == "default selected artifact"
    assert final["dtype_policy"] == final["runtime_summary"]["config"] == selected
    assert final["runtime_policies"] == candidate["runtime_policies"]
    assert final["runtime_policies"] == token["runtime_policies"] == context["runtime_policies"]
    assert token["precision_config"] == context["precision_config"] == selected
    assert final["accuracy_pass"] and final["capability_pass"] and final["trace_verified"]
    assert baseline["accuracy_pass"] and baseline["capability_pass"] and baseline["trace_verified"]
    assert final["layers"] == token["layers"] == context["layers"] == 36
    assert context["pass"] and context["capacity"]["context"] == selected["max_context"] == 524288
    assert context["optimized_buffers"]["max_history_and_full_cache_coexist"]
    assert token["pass"] and token["token_out_no_readback"]["final_token_matches_generation"]
    normalization = verify_source_normalization()
    normalized_sources = {row["path"]: row for row in normalization["sources"]}
    source_hashes_current = True
    for source, digest in final["source_sha256"].items():
        current_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if current_digest != digest:
            source_hashes_current = False
            assert normalized_sources[source]["measured_sha256"] == digest, source
            assert normalized_sources[source]["current_sha256"] == current_digest, source
    assert final["source_sha256"] == token["source_sha256"]
    parity = []
    for kind, new_path, old_path in [
        ("selected", DOC / "final_selected_quality.json", Path(candidate["quality_artifact"])),
        ("baseline", Path(baseline["quality_artifact"]), DOC.parent / "optimized_full_model/qualitative_selected.json"),
    ]:
        new = json.loads(new_path.read_text())
        old = json.loads(old_path.read_text())
        assert len(new) == len(old) == 6
        for a, b in zip(new, old):
            for key in ["id", "prompt_token_ids", "tt_token_ids", "tt_text_through_eos", "first_eos_index"]:
                assert a[key] == b[key], (kind, a["id"], key)
        parity.append(
            {
                "kind": kind,
                "new": str(new_path),
                "control": str(old_path),
                "all_six_prompt_token_text_eos_records_identical": True,
                "new_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest(),
                "control_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            }
        )
    (DOC / "final_quality_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    verdicts = read("quality_verdicts.json")
    verdicts["baseline_mixed_hifi2_lofi"] = {
        "pass": True,
        "reason": "All six final records exactly reproduce the independently reviewed optimized-full-model control.",
        "artifact": baseline["quality_artifact"],
        "comparison": str(DOC / "final_quality_parity.json"),
        "control_review": str(DOC.parent / "optimized_full_model/QUALITY_REVIEW.md"),
    }
    verdicts[selected["config_id"]]["default_artifact"] = str(DOC / "final_selected_quality.json")
    verdicts[selected["config_id"]]["default_parity"] = str(DOC / "final_quality_parity.json")
    (DOC / "quality_verdicts.json").write_text(json.dumps(verdicts, indent=2) + "\n")
    report()
    assert (
        hashlib.sha256(
            json.dumps(read("sweep_results.json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == normalization["sweep_json"]["canonical_content_sha256"]
    )
    rows = read("sweep_results.json")["results"]
    speed = final["decode_tokens_per_second_per_user"]
    faster = [
        r for r in rows if r["decode_tokens_per_second_per_user"] > speed and r["config_id"] != selected["config_id"]
    ]
    assert all(r["status"] in {"continuation_fail", "quality_fail", "accuracy_fail"} for r in faster)
    eligible = [r for r in rows if r["selection_eligible"]]
    assert max(eligible, key=lambda r: r["decode_tokens_per_second_per_user"])["config_id"] == selected["config_id"]
    validation = {
        "selected_config_id": selected["config_id"],
        "default_construction_verified": True,
        "selected_config_sha256": hashlib.sha256((DOC / "selected_precision_config.json").read_bytes()).hexdigest(),
        "runtime_policies_equal_qualified_candidate": True,
        "source_hashes_current": source_hashes_current,
        "measured_runtime_matches_current_after_verified_import_cleanup": True,
        "source_normalization_artifact": str(DOC / "post_measurement_source_normalization.json"),
        "selected_token_out_context_policies_identical": True,
        "context_preserved": 524288,
        "teacher_regime": {
            key: final[key]
            for key in ["prompt_len", "generation_len", "batch_size", "cache_capacity_tokens", "trace_verified"]
        },
        "faster_rejections": [
            {key: row[key] for key in ["config_id", "status", "artifact", "decode_tokens_per_second_per_user"]}
            for row in faster
        ],
        "fastest_evaluated_eligible_policy": True,
        "evaluated_full_model_configs": len(rows),
        "quality_parity": parity,
        "teacher_tps": speed,
        "post_selection_token_out_tps": token["token_out_no_readback"]["tokens_per_second_per_user"],
        "head_geometry_control": {
            "artifact": str(DOC / "head_cores_full36.json"),
            "medians_tps": head_geometry["medians"],
        },
    }
    (DOC / "selection_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    table = []
    for row in rows:
        name = row["config_id"]
        status = "selected" if name == selected["config_id"] else row["status"]
        if status == "pass":
            status = "baseline control" if name == "baseline_mixed_hifi2_lofi" else "slower; not selected"
        path = Path(row["artifact"]).relative_to(DOC)
        table.append(
            f"| [{name}]({path}) | {row['top1']*100:.0f}/{row['top5']*100:.0f}/{row['top100']*100:.0f} | {row['decode_tokens_per_second_per_user']:.3f} | {len(row['teacher_perf'])} | {status} |"
        )
    prefill = final["prefill"][0]
    bf = baseline["decode_tokens_per_second_per_user"]
    content = f"""# K2-Horizon-7B datatype sweep — stage 8

Selected **`{selected['config_id']}`**, consumed by default through the required
[precision artifact](selected_precision_config.json). On the main AIME24 chat
reference (156 prompt tokens, 100 generated tokens, B1), the final default gives
**{final['top1']*100:.0f}% top-1 / {final['top5']*100:.0f}% top-5 / {final['top100']*100:.0f}% top-100** in traced teacher forcing;
prefill agreement is **{prefill['top1']*100:.0f}% / {prefill['top5']*100:.0f}% / {prefill['top100']*100:.0f}%**.
Acceptance requires top-1 ≥90%, top-5 ≥98%, top-100 =100% in both modes,
valid non-aligned prompts, the inherited continuation check, controlled shared-suite
quality, and the unchanged **524288-token context**.

| Measurement | Baseline | Final selected default |
| --- | ---: | ---: |
| Traced teacher-forcing decode, 156/100/B1, cache 256, median of 9 | {bf:.3f} t/s/u | **{speed:.3f} t/s/u** |
| Same-regime warmed generator TTFT | {baseline['ttft_ms']:.3f} ms | **{final['ttft_ms']:.3f} ms** |
| Token-out no-readback decode, 128/128/B1, cache 256 | 94.514 t/s/u | **{token['token_out_no_readback']['tokens_per_second_per_user']:.3f} t/s/u** |
| Token-out warmed generator TTFT, median of 3 requests | 20.550 ms | **{token['selected']['ttft_seconds']*1000:.3f} ms** |

Teacher forcing selects the policy ({(speed/bf-1)*100:.2f}% faster than the refreshed
matching baseline). **Later performance reports and vLLM comparisons must use the
post-selection token-out number when comparing autoregressive decode.** TTFT is
the generator's warmed prefill/first-sample interval; input/trace preparation is
reported separately in the raw benchmark. These are not external serving latencies.
No vLLM integration or serving measurement is part of this stage.

## Selected numerical policy

Layer ranges are inclusive and zero-based. The model has 36 layers.

| Group | Selected dtype / compute fidelity |
| --- | --- |
| QKV and attention output | BFP8 / HiFi2, all layers |
| MLP gate/up | BFP8 / HiFi2 in layers 0, 3–10; BFP4 / LoFi in layers 1–2, 11–35 |
| MLP down | BFP4 / LoFi in layers 24–34; BFP8 / HiFi2 elsewhere |
| Vocabulary head | BFP8 / LoFi; unchanged 16-core input, K4, split8192, one reader/bank |
| Decode activations, residual, CCL, matmul outputs | BF16 |
| Fused prefill sources | BFP8 QKV/MLP; layer 1 QKV retains BF16 |
| KV cache | BFP8 |
| Embedding/norm/RoPE/logits/sampling values | BF16; token IDs uint32 |
| Accumulation/attention | FP32 projection/head accumulation; norms HiFi4; stock SDPA HiFi2; intrinsic accurate long-context attention HiFi4/FP32 |

Every selected numerical field reaches actual construction. See
[propagation paths](POLICY_PROPAGATION.md), [live default exports](final_selected.json),
and [selection/provenance checks](selection_validation.json). Redundant legacy
attention aliases were removed from the final artifact; resolved layer policies
exactly equal the qualified candidate. `precision_config=<path>` provides an
explicit override; [safe baseline](configs/baseline_mixed_hifi2_lofi.json) remains
available. The mandatory shared loader is the future adapter contract; an adapter
does not exist yet, so no vLLM propagation claim is made.

## Search, fidelity comparisons and rejected configurations

All {len(rows)} rows below are full 36-layer model evaluations on 4 Blackhole P300c chips,
MeshShape(1,4), physical ring 1–0–3–2. Coarse runs use the median of 3 warmed repetitions;
qualified finalists use the median of 9. Each has 99 model and sampler trace replays,
99 forced-token refreshes and 99 prediction reads. Eager and one-layer smoke
numbers are excluded from selection and plots. Raw rows include prefill scores,
TTFT, exact dtype/fidelity policy, command, hardware, source hashes and trace counters.

| Config / exact evidence | Teacher top1/top5/top100 (%) | Traced t/s/u | Repeats | Decision |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(table)}

The canonical numerical mappings are all-BFP8/HiFi2 accuracy and all-MLP
BFP4/LoFi performance, adapted to K2's FP32/norm/attention contract and reviewed
geometry. No native canonical K2 implementation was found. Aggressive all-BFP4
is a separate stress policy, not the canonical mapping. Every material BFP4 group
(QKV, O, gate/up, down, head) has LoFi and HiFi2 comparisons. Dominant BFP8 groups
also compare LoFi with HiFi2. BFP8 CCL and BFP8 decoder inputs were tested alone
and together, including direct BFP8 gather consumption; their extra conversions
or reduced precision did not yield a faster accepted configuration.

Numerically faster rows are rejected by the inherited real-model whole/split 257-token
continuation gate (PCC≥.995 at cuts 31/32 plus the same HF token), or by concrete
shared-suite regressions. [Quality decisions](quality_verdicts.json) and
[actual-output review](QUALITY_REVIEW.md) identify prompts and controls.
All tested policies pass the AIME accuracy thresholds; that alone does not waive
capability or quality failures. Slower rows need no additional qualification to
establish that they cannot beat the selected passing result.

The [down geometry check](geometry_down4_summary.json) tests BFP4/LoFi/FP32 with
real checkpoint weights and recorded activations: 60/63 configurations run;
8 cores/K4/2 readers is fastest. The three K48/one-reader failures exceed physical
L1; other legal core/K/reader combinations, including larger/non-power-of-two K,
are measured. 64 input cores cannot divide K3072 into whole tiles.
The [head check](geometry_head8_summary.json) tests the selected BFP8/LoFi/FP32
under working-core and K variants. Corrected coherent comparisons find 8 cores
2.8 µs faster in isolation. The [full-stack ABBA comparison](head_cores_full36.json)
gives 95.804 t/s/u for 16 cores versus 95.682 for 8 with both accuracy gates passing;
the reviewed 16-core default is retained. K16/K32 exceed
physical L1. Component numbers do not replace full-model accuracy or ranking.

## Pareto interpretation

These pyplot charts show every evaluated full-model config. The blue numerical
frontier may include capability/quality failures (brown X). The red selected point
is the fastest evaluated policy that preserves the full acceptance contract.
Dotted vertical lines mark the minimum top-1/top-5 accuracy. All teacher top-5
values are 100%, so its numerical frontier is a single marked point. Numbered
labels map to [CSV](sweep_results.csv); [JSON](sweep_results.json) retains policies
and exact evidence paths.

![Top-1 Pareto](top1_perf_pareto.png)
![Top-5 Pareto](top5_perf_pareto.png)

## Context, quality and limits

[Default context validation](context_selected.json) repeats public traced
generation at 1, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097, 4353 tokens and verifies
non-aligned continuation. All 36 maximum-context caches coexist with optimized
token history and prepared-prefill buffers. Late prefill/decode reaches positions
524252/524287 with finite output. The updated [context contract](../context_contract.json)
and [candidate memory ledgers](candidate_context_ledgers.json) preserve 524288.
The late-context prefix is initialized-zero structural coverage, not a full-context
HF quality comparison. No advertised capability was reduced.

The six default outputs exactly reproduce the qualified selected policy;
the refreshed baseline exactly reproduces the reviewed optimized baseline
([parity](final_quality_parity.json)). Haiku still loops in syllable counting:
pinned BF16 HF reproduces that exact-prefix trajectory. Story drafting is covered
by an eligible exact-prefix control. Neither is claimed as a complete answer.
Physics and parts of the code response also exhaust the fixed 512-token budget.
Mechanical checker success does not replace this direct text review.

## Reproduction and handoff

Use `source python_env/bin/activate`, then
`source /home/vkovacevic/k2-horizon/lane-env.sh`; device runs set
`HF_HUB_OFFLINE=1 TT_METAL_TRACE_ALLOC_TRACKING=1`. All TT commands are serialized.
The pinned revision/reference and chat template are recorded in
[reference provenance](reference_provenance.json) and
[prompt metadata](qualitative_prompt_format.json).

- [Final exact commands and exit status](final_validation_execution.json)
- [Default accuracy/trace/runtime evidence](final_selected.json)
- [Separate post-selection token-out benchmark](post_selection_token_out.json)
- [Work log, setup, commands and commit records](work_log.md)
- [Classified anomalies](ANOMALIES.md)
- [Lossless raw-log archive manifest](evidence_archives/manifest.json)
- [Verified post-measurement import cleanup](post_measurement_source_normalization.json)
- [Independent stage review](STAGE_REVIEW.md)

Raw `.log` files remain local and are ignored by Git. The committed XZ archives
retain their exact bytes; the manifest maps each original path to its archive,
SHA-256, and size. Recover any raw log with `xz -dc <archive_path> > <raw_path>`.
Measured source hashes are preserved. Repository hooks subsequently removed one
unused import binding and sorted import names; the normalization receipt and CPU
finalizer verify the exact transformations and equivalent Python syntax trees.

Rebuild summaries/plots with `python -m models.autoports.ifm_k2_horizon_7b.tests.finalize_datatype_evidence`
after context finalization. No push is authorized or performed.
"""
    (DOC / "README.md").write_text(content)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
