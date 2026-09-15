# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only final current-source smoke gate. No historical source fallback.

Standard library only. Never imports producers, runs commands, loads tensors,
or changes records/sources. Missing outputs are PENDING, not a partial PASS.
"""

import argparse
import ast
import hashlib
import json
import shlex
from pathlib import Path

import validate_research_checkpoint as V

HERE, ROOT, PREFIX = V.HERE, V.ROOT, V.V2
PLAN = HERE / "final-smoke-plan.json"


def specifications():
    rows = [
        ("final-captured-v1", "captured_fullchip.py", "captured_interface", "captured-interface-smoke-v2.jsonl"),
        ("final-hi2-native-v1", "hifi2_native_fullchip.py", "hifi2_late", "hi2-native-rne-smoke-v1.json"),
        ("final-hi2-lut-v1", "hifi2_lut_fullchip.py", "hifi2_late", "hi2-lut-on-smoke-v1.json"),
        ("final-hi2-bf16-lut-v1", "hifi2_bf16_lut_fullchip.py", "hifi2_late", "hi2-bf16-lut-on-1024-v1.json"),
        (
            "final-native-storage-v1",
            "native_storage_fullchip.py",
            "native_storage",
            "native-storage-lofi_fp32-1024-v1.json",
        ),
        ("final-vaxis-D-v1", "Vtransposed_fullchip.py", "v_transpose", "vt-smoke-full-D-v1.jsonl"),
        ("final-vaxis-N-v1", "Vtransposed_fullchip.py", "v_transpose", "vt-smoke-full-N-v1.jsonl"),
        ("final-recipe-v1", "combined_recipe_fullchip.py", "combined_recipe", "recipe-smoke-h16-vN-pm-v1.jsonl"),
    ]
    for fmt in ("b8_b4", "b8_b8"):
        for distribution in ("normal", "uniform"):
            for mode in ("none", "mean_error"):
                rows.append(
                    (
                        f"final-mean-{fmt}-{distribution}-{mode}-v1",
                        "value_mean_error_fullchip.py",
                        "mean_error",
                        f"mean-error-smoke-{fmt}-{distribution}-{mode}-v1.json",
                    )
                )
    for recip in (0, 1):
        for fidelity in (0, 1):
            rows.append(
                (
                    f"final-recip-r{recip}-f{fidelity}-v1",
                    "recip8_fullchip.py",
                    "recip_final",
                    f"final-scale-r{recip}-f{fidelity}-smoke-v1.jsonl",
                )
            )
    for distribution in ("normal", "thresholds", "wide", "zeros"):
        rows.append((f"final-b4-{distribution}-v1", "bfp4_round.py", "b4_final", None))
    return {
        label: dict(
            label=label,
            driver=driver,
            family=family,
            baseline=baseline,
            output=PREFIX
            + ("bfp4_round/" if family == "b4_final" else "")
            + label
            + (".jsonl" if baseline and baseline.endswith(".jsonl") else ".json"),
        )
        for label, driver, family, baseline in rows
    }


SPECS = specifications()


def required_sources(spec, manifest_record):
    family = spec["family"]
    if family == "b4_final":
        required = {
            PREFIX + "bfp4_round.py",
            *(PREFIX + "bfp4_round/" + f for f in ("reader.cpp", "compute.cpp", "writer.cpp")),
        }
    elif family == "recip_final":
        fast = "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/"
        tail = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/"
        required = {
            V.REPRO,
            V.CHAIN,
            fast + tail + "compute_common.hpp",
            fast + tail + "compute_streaming.hpp",
            fast + "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
            *(
                PREFIX + f
                for f in (
                    "recip8_fullchip.py",
                    "recip8_streaming/compute.cpp",
                    "recip8_streaming/recip_override.hpp",
                    "recip8_streaming/final_scale.hpp",
                    "fast_correction.hpp",
                    "safe_rescale.hpp",
                    "fullchip/reader_chain.cpp",
                    "fullchip/writer.cpp",
                    "preprocess.py",
                    "preprocess/reader.cpp",
                    "preprocess/compute.cpp",
                    "preprocess/writer.cpp",
                )
            ),
        }
    else:
        required = V.required_manifest(family, manifest_record)
    return required | {PREFIX + spec["driver"]}


class CurrentAudit(V.Audit):
    """Current pins are hard gates; snapshots are deliberately never consulted."""

    def current_provenance(self, e, record, spec):
        manifest = record.get("source_sha256")
        e.require(isinstance(manifest, dict) and bool(manifest), "Missing current-source manifest")
        if not isinstance(manifest, dict):
            return
        missing = sorted(required_sources(spec, record) - set(manifest))
        e.require(not missing, "Missing principal source pins: " + repr(missing))
        drift = []
        for name, expected in manifest.items():
            e.digest(expected, "Recorded source " + name)
            actual = self.current_hash(name)
            if actual != expected or actual is None:
                drift.append(name)
        e.require(not drift, "Sources do not match CURRENT bytes: " + repr(drift))
        e.provenance = dict(
            status="PASS" if not missing and not drift else "FAIL",
            principal_omissions=missing,
            current_source_mismatches=drift,
            manifest_entries=len(manifest),
            historical_fallback=False,
        )

    def assertion_witness(self, e, record, driver, expression, node_type=ast.Assert):
        name = PREFIX + driver
        expected = record.get("source_sha256", {}).get(name)
        actual = self.current_hash(name)
        if actual is None or actual != expected:
            e.failures.append("CURRENT producer unavailable/mismatched for witness: " + name)
            return False
        data = (self.root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            e.failures.append("Producer changed while reading witness: " + name)
            return False
        tree = ast.parse(data.decode(), filename=name)
        expressions = [
            ast.unparse(n.test if isinstance(n, ast.Assert) else n) for n in ast.walk(tree) if isinstance(n, node_type)
        ]
        found = any(expression in text for text in expressions)
        e.require(found, "CURRENT producer lacks witness: " + expression)
        e.summary.setdefault("current_producer_witnesses", []).append(
            dict(source=name, sha256=expected, expression=expression, node_type=node_type.__name__, matched=found)
        )
        return found


def read_record(path):
    data = path.read_bytes()
    value = (
        [V.decode(line) for line in data.decode().splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else V.decode(data.decode())
    )
    return value, data


def result_rows(e, data, family):
    if not isinstance(data, list):
        e.require(isinstance(data, dict), "Record is not an object")
        return data, [data]
    expected_kind = "captured_evaluation" if family == "captured_interface" else "result"
    e.require(bool(data) and data[0].get("kind") == "provenance", "Missing first provenance record")
    complete = [r for r in data if r.get("kind") == "complete"]
    if not complete:
        e.pending.append("No completion footer; final smoke is not complete")
    else:
        e.equal(len(complete), 1, "Completion footer count")
        e.equal(data[-1].get("kind"), "complete", "Completion must be last")
        e.equal(complete[0].get("sources_unchanged"), True, "During-run source stability")
    e.equal(sum(r.get("kind") == "provenance" for r in data), 1, "Provenance count")
    e.require(
        all(r.get("kind") in ("provenance", expected_kind, "complete") for r in data), "Unexpected record/failure kind"
    )
    return data[0], [r for r in data if r.get("kind") == expected_kind]


def no_timing(e, r):
    if "iters" in r:
        e.equal(r["iters"], 0, "No performance iterations in final smoke")
    if "timings" in r:
        e.equal(r["timings"], {}, "Captured smoke has no timed measurements")
    else:
        for name in ("attention", "preprocessing", "combined"):
            e.require(name in r, "Missing smoke timing scope: " + name)
    objects = {
        k: r[k] for k in ("attention", "preprocessing", "combined", "epilogue", "attention_with_epilogue") if k in r
    }
    objects.update(r.get("timings", {}))
    objects.update(r.get("preprocessing_stages", {}))
    for name, t in objects.items():
        V.timing(e, t, "No timing: " + name, 0, zero=name == "epilogue")
    for name in ("attention_tflops", "combined_tflops", "read_write_GBps"):
        if name in r:
            e.equal(r[name], None, "No final-smoke throughput: " + name)


def exact_preprocessing(e, checks):
    e.require(isinstance(checks, list) and bool(checks), "Missing explicit exact-preprocessing results")
    if not isinstance(checks, list):
        return
    names = [x.get("input") for x in checks]
    e.require(all(names.count(x) == 1 for x in ("Q", "K", "V")), "Missing/duplicate Q/K/V preprocessing checks")
    e.equal(len(names), len(set(names)), "Duplicate preprocessing checks")
    for x in checks:
        e.equal(x.get("mismatch"), 0, "Exact preprocessing: " + str(x.get("input")))
        if x.get("adaptive_statistics"):
            for key in ("induced_exponent_mismatches", "native_grid_roundtrip_mismatches"):
                e.equal(x["adaptive_statistics"].get(key), 0, "Adaptive exactness: " + key)


def b4_primitive(audit, e, record, spec):
    distribution = spec["label"][len("final-b4-") : -len("-v1")]
    for key, value in dict(
        length=1024,
        cores=4,
        actual_cores=4,
        batch=4,
        fp32_dst=False,
        output_format="b4",
        distribution=distribution,
        seed=1240,
        host_only=False,
        iters=0,
        mismatch=0,
        oracle_mismatch=0,
        numel=1024 * 128,
        median_ms=None,
        replay_ms=[],
        read_write_GBps=None,
    ).items():
        e.equal(record.get(key), value, "B4 primitive " + key)
    e.number(record.get("quantization_l2_pct"), "B4 representation L2")
    audit.assertion_witness(e, record, "bfp4_round.py", "mismatch == 0")
    audit.assertion_witness(e, record, "bfp4_round.py", "oracle_mismatch == 0")
    audit.assertion_witness(e, record, "bfp4_round.py", "bool(torch.isfinite(values).all())")
    e.summary.update(
        exact_oracle=True,
        finite_basis="Exact decoded match to finite validated oracle",
        complete_output_hash_comparison="NOT_AVAILABLE",
        trace_replay="NOT_RUN: iters=0",
        during_run_source_stability="NOT_RECORDED; manifest is collected after computation",
        attention_all_query_scope="NOT_APPLICABLE: quantizer primitive",
    )
    e.notes.append(
        "B4 has no output hash/replay/source-stability boolean; no such evidence is fabricated. Older controls use N4096/one core, so are not hash-matched baselines."
    )


SETTINGS = (
    "variant",
    "length",
    "heads",
    "cores",
    "q_chunk",
    "seed",
    "distribution",
    "sample_rows",
    "q_prescale",
    "b8_rne",
    "bfp8_pack_precise",
    "exp_degree",
    "native_exp",
    "center_k",
    "mean_mode",
    "fix_correction",
    "reader_chain",
    "reader_split",
    "reader_linear_k",
    "read_barrier_tiles",
    "input_slots",
    "fidelity",
    "fp32_dst",
    "destination",
    "denom_only",
    "kv_formats",
    "correction_mode",
    "h16",
    "adaptive_v",
    "v_transposed",
    "grid7_exp",
    "recip8",
    "final_scale_hifi4",
)
KERNEL_SETTINGS = (
    "defines",
    "input_slots",
    "fidelity",
    "fp32_dst",
    "q_chunk",
    "k_chunk",
    "head_dim",
    "k_format",
    "v_format",
    "pv_transpose_in1",
    "v_group_axis",
    "score_scale",
    "h16",
    "adaptive_v",
)


def compare_baseline(e, actual, baseline):
    key = lambda r: r.get("variant", r.get("distribution", "single"))
    old = {key(r): r for r in baseline}
    keys = [key(r) for r in actual]
    V.coverage(e, keys, set(old))
    e.equal(len(old), len(baseline), "Baseline row keys unique")
    comparisons = []
    for r in actual:
        if key(r) not in old:
            continue
        b = old[key(r)]
        for field in SETTINGS:
            if field in r or field in b:
                e.equal(r.get(field), b.get(field), "Settings-matched baseline: " + field)
        for field in (
            "sampled_query_rows",
            "original_input_sha256",
            "config",
            "reference_scope",
            "accuracy_scope",
            "timing_scope",
        ):
            if field in r or field in b:
                e.equal(r.get(field), b.get(field), "Baseline input/scope: " + field)
        for field in KERNEL_SETTINGS:
            rk, bk = r.get("kernel", {}), b.get("kernel", {})
            if field in rk or field in bk:
                e.equal(rk.get(field), bk.get(field), "Baseline kernel: " + field)
        e.digest(r.get("output_sha256"), "Final complete output hash")
        e.digest(b.get("output_sha256"), "Baseline complete output hash")
        e.equal(r.get("output_sha256"), b.get("output_sha256"), "Complete-output bit identity versus pre-format smoke")
        comparisons.append(
            dict(
                case=key(r),
                baseline_output_sha256=b.get("output_sha256"),
                final_output_sha256=r.get("output_sha256"),
                identical=r.get("output_sha256") == b.get("output_sha256"),
            )
        )
    e.summary["baseline_comparisons"] = comparisons


def validate_case(audit, e, spec, data, baseline=None):
    manifest_record, rows = result_rows(e, data, spec["family"])
    audit.current_provenance(e, manifest_record, spec)
    args = manifest_record.get("args", manifest_record)
    if "label" in args:
        e.equal(args["label"], spec["label"], "Final label")
    e.equal(args.get("iters"), 0, "Final smoke only")
    if spec["family"] == "b4_final":
        b4_primitive(audit, e, manifest_record, spec)
        return
    if spec["family"] == "captured_interface":
        capture = manifest_record["capture"]
        e.equal(capture.get("shape"), [1, 2, 1024, 128], "Synthetic capture shape")
        e.equal(capture.get("dtype"), "bfloat16", "Original capture dtype")
        e.equal(
            capture["metadata"]["provenance"].get("source_kind"), "synthetic", "Interface smoke is not model validation"
        )
        e.equal(capture["metadata"].get("causal"), False, "Noncausal capture")
        e.equal(capture["metadata"].get("mask"), None, "Unmasked capture")
        if baseline:
            e.equal(
                capture.get("input_sha256"),
                baseline[0]["capture"].get("input_sha256"),
                "Same synthetic artifact inputs",
            )
        e.notes.append(
            "Artifact generator metadata describes the historical capture, not runtime code; runtime source pins must all be current."
        )
    else:
        for field, value in dict(length=1024, heads=2, cores=4, check_preprocess=True).items():
            e.equal(args.get(field), value, "Final SDPA scope " + field)
    for r in rows:
        no_timing(e, r)
        e.equal(r.get("trace_equal"), True, "Actual correctness trace replay")
        e.equal(r.get("all_output_finite", r.get("finite")), True, "Complete output finite")
        if "sources_unchanged" in r:
            e.equal(r["sources_unchanged"], True, "During-run source stability")
        elif not isinstance(data, list):
            e.failures.append("Missing single-record source-stability gate")
        e.equal(r.get("sampled_query_rows"), list(range(1024)), "Every N1024 query referenced")
        if "all_query_rows_referenced" in r:
            e.equal(r["all_query_rows_referenced"], True, "All-query scope boolean")
        metrics = r.get("metrics", r.get("accuracy"))
        V.metric(e, metrics, "Original-input FP64 accuracy (no numerical cutoff)")
        e.equal(
            r.get("useful_attention_flops", r.get("useful_flops")),
            4 * 2 * 1024 * 1024 * 128,
            "Useful attention FLOPs exclude preprocessing and extra matmuls",
        )
        scope = r.get("reference_scope", r.get("accuracy_scope"))
        e.require(isinstance(scope, str) and bool(scope), "Missing explicit reference scope")
        for field in ("original_input_sha256",):
            hashes = r.get(field)
            if hashes is not None:
                for h in hashes.values() if isinstance(hashes, dict) else hashes:
                    e.digest(h, "Original input hash")
        family = spec["family"]
        if family == "captured_interface":
            e.equal(r.get("immutable_inputs_verified"), True, "Capture immutable inputs")
            e.equal(
                r.get("preprocessing_exact_checked"),
                r["variant"].startswith("lofi_"),
                "Applicable capture preprocessing gate",
            )
            if r["variant"] == "fast":
                e.equal(r.get("fast_private_correction_reset"), True, "Private FAST integration fix")
        elif family in ("hifi2_late", "native_storage"):
            if "preprocessing_checks" in r:
                exact_preprocessing(e, r["preprocessing_checks"])
                for x in r["preprocessing_checks"]:
                    if x["input"] in ("K", "V"):
                        e.equal(x.get("identity_bits_preserved"), True, "BF16 K/V identity preparation")
            else:
                audit.assertion_witness(e, manifest_record, spec["driver"], "mismatch == 0")
                audit.assertion_witness(e, manifest_record, spec["driver"], "args.check_preprocess", ast.If)
                e.notes.append(
                    "Exact preparation uses current pinned builder assertions plus enabled flag; mismatch fields were omitted."
                )
            audit.assertion_witness(e, manifest_record, spec["driver"], "args.length // 512", ast.Assign)
            audit.assertion_witness(
                e,
                manifest_record,
                spec["driver"],
                "reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])",
                ast.Assign,
            )
            if spec["driver"] == "hifi2_bf16_lut_fullchip.py":
                for field in ("cpu_inputs_unchanged", "device_inputs_unchanged"):
                    e.equal(r.get(field), True, "BF16 control original inputs: " + field)
                e.equal(r.get("correctness_trace_replays"), 2, "BF16 control replay count")
                e.equal(r.get("replay_output_sha256"), [r["output_sha256"]] * 2, "BF16 control replay hashes")
        else:
            exact_preprocessing(e, r.get("kernel", r).get("preprocessing_checks"))
        if family in ("v_transpose", "combined_recipe"):
            for expression in (
                "[tensor_hash(x) for x in inputs] == original_hashes",
                "all((bf16_bitwise_equal(ttnn.to_torch(t).bfloat16(), x) for t, x in zip(originals, inputs)))",
            ):
                audit.assertion_witness(e, manifest_record, spec["driver"], expression)
            e.notes.append(
                "Original-input immutability is a CURRENT producer assertion witness, not an explicit record boolean."
            )
        if family in ("mean_error", "recip_final"):
            for field in ("cpu_inputs_unchanged", "device_inputs_unchanged"):
                e.equal(r.get(field), True, "Original inputs unchanged: " + field)
            e.equal(r.get("correctness_trace_replays"), 2, "Two correctness trace replays")
            trace = r.get("trace_check", r.get("correctness_trace_qualification", {}))
            hashes = trace.get("output_sha256", trace.get("replay_output_sha256"))
            e.equal(hashes, [r["output_sha256"]] * 2, "Both replay hashes match complete output")
        if family == "mean_error" and r.get("correction_mode") == "mean_error":
            e.equal(r["epilogue_check"].get("mismatch"), 0, "Device BF16 epilogue exactness")
            e.equal(r["value_mean_error_correction"]["check"].get("bias_mismatch"), 0, "Device BF16 mean-error delta")
    if baseline is None:
        e.pending.append("Required matching pre-format smoke is absent")
    else:
        be = V.Evidence("historical-comparison-only", spec["family"])
        _, baseline_rows = result_rows(be, baseline, spec["family"])
        e.require(not be.failures and not be.pending, "Baseline incomplete/malformed")
        compare_baseline(e, rows, baseline_rows)
    e.summary.update(
        result_rows=len(rows),
        all_query_reference=True,
        sequence_length=1024,
        performance_measured=False,
        numerical_acceptance_cutoff=None,
        model_quality_claim=False,
    )


def plan_checks(root, plan):
    failures = []
    cases = plan.get("cases", [])
    labels = [r.get("label") for r in cases]
    if len(labels) != 24 or len(set(labels)) != 24 or set(labels) != set(SPECS):
        failures.append("Plan must contain the explicit 24 unique final-smoke cases")
    for entry in cases:
        if entry.get("label") not in SPECS:
            continue
        spec = SPECS[entry["label"]]
        if entry.get("driver") != spec["driver"] or entry.get("output") != spec["output"]:
            failures.append(entry["label"] + ": unexpected producer/output path")
        source = root / (PREFIX + spec["driver"])
        tree = ast.parse(source.read_text())
        flags = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                flags.update(x.value for x in node.args if isinstance(x, ast.Constant) and isinstance(x.value, str))
        argv = shlex.split(entry.get("args", ""))
        unknown = [s.split("=", 1)[0] for s in argv if s.startswith("--") and s.split("=", 1)[0] not in flags]
        if unknown:
            failures.append(entry["label"] + ": undeclared CLI flags " + repr(unknown))
        if "--iters" not in argv or argv.index("--iters") + 1 >= len(argv) or argv[argv.index("--iters") + 1] != "0":
            failures.append(entry["label"] + ": missing explicit --iters 0")
        command = shlex.split(entry.get("command", ""))
        prefix = ["python_env/bin/python", "-B", PREFIX + spec["driver"], *argv]
        if command[: len(prefix)] != prefix:
            failures.append(entry["label"] + ": command disagrees with driver/args")
    return failures


def run(plan_path=PLAN, root=ROOT, directory=HERE):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    plan_path = Path(plan_path)
    plan_bytes = plan_path.read_bytes()
    plan = V.decode(plan_bytes.decode())
    failures = plan_checks(root, plan)
    audit = CurrentAudit(root, directory)
    snapshots = {plan_path: plan_bytes}
    for entry in plan.get("cases", []):
        if entry.get("label") not in SPECS:
            continue
        spec = SPECS[entry["label"]]
        e = V.Evidence(spec["output"], spec["family"])
        path = directory / Path(spec["output"]).relative_to(PREFIX)
        if not path.is_file():
            e.pending.append("Expected final-smoke output is absent")
        else:
            try:
                data, raw = read_record(path)
                snapshots[path] = raw
                e.summary["record_sha256"] = hashlib.sha256(raw).hexdigest()
                baseline = None
                if spec["baseline"]:
                    bp = directory / spec["baseline"]
                    e.summary["baseline_file"] = spec["baseline"]
                    if bp.is_file():
                        baseline, br = read_record(bp)
                        snapshots[bp] = br
                        e.summary["baseline_record_sha256"] = hashlib.sha256(br).hexdigest()
                validate_case(audit, e, spec, data, baseline)
            except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, SyntaxError) as error:
                e.failures.append(type(error).__name__ + ": " + str(error))
        audit.items.append(e.export())
    for path, raw in snapshots.items():
        if not path.is_file() or path.read_bytes() != raw:
            failures.append("Plan/evidence changed during audit: " + str(path))
    result = audit.result()
    failed = failures or result["evidence_status"] == "FAIL" or result["sources_changed_during_audit"]
    result.update(
        schema="final-current-source-smokes-v1",
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        plan_failures=failures,
        final_status="FAIL" if failed else result["evidence_status"],
        historical_source_fallback=False,
        limitations=[
            "Read-only record/source audit, not device execution or tensor recomputation",
            "No global L2 cutoff, performance qualification or model-quality claim",
            "B4 primitive has exact-oracle evidence but no output hash, replay, or during-run source-stability field",
            "Captured fixture provenance may be historical; all runtime manifest sources must match current bytes",
        ],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON to stdout only")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=HERE)
    args = parser.parse_args()
    result = run(args.plan, args.repo_root, args.evidence_dir)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print("Final current-source smoke:", result["final_status"])
        for error in result["plan_failures"]:
            print("PLAN FAIL:", error)
        for e in result["evidence"]:
            print(e["status"], Path(e["file"]).name, "; ".join(e["failures"] + e["pending"]))
        print("No historical fallback, producer execution, device calls, or file writes.")
    return 0 if result["final_status"] == "PASS" else 2 if result["final_status"] == "PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
