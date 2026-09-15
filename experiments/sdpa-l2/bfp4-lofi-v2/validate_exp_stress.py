# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only current-source exp stress audit; standard library, no producer execution.

Audits 80 planned configurations and their separately requested -v2 repetitions.
Missing results remain PENDING. PASS means recorded gates, not an L2 threshold.
"""

import argparse
import ast
import hashlib
import itertools
import json
import shlex
import re
from pathlib import Path

import validate_final_smokes as F

V, HERE, ROOT, PREFIX = F.V, F.HERE, F.ROOT, F.PREFIX
DISTRIBUTIONS = (
    "normal",
    "outliers",
    "scaled_qk",
    "common_q",
    "common_k",
    "common_v",
    "constant_v",
    "uniform",
    "uniform_constant_v",
    "biased_v",
)
SEEDS = (1240, 1241)
ROUTES = ("lofi_native", "lofi_lut", "hi2_native", "hi2_lut")
PLAN = HERE / "exp-stress-plan.json"
LONG_DISTRIBUTIONS = ("normal", "outliers", "scaled_qk", "common_q", "common_k", "common_v")
CONTEXT256K_DISTRIBUTIONS = ("normal", "outliers", "common_k")


def specifications():
    result = {}
    for route, distribution, seed, repeat in itertools.product(ROUTES, DISTRIBUTIONS, SEEDS, (1, 2)):
        hi2 = route.startswith("hi2")
        label = f"exp-stress-{route}-{distribution}-s{seed}-v{repeat}"
        result[label] = dict(
            label=label,
            route=route,
            distribution=distribution,
            seed=seed,
            repeat=repeat,
            length=1024,
            heads=2,
            cores=4,
            sample_rows=1024,
            driver="hifi2_bf16_lut_fullchip.py" if hi2 else "exp_lut_macro_streaming.py",
            family="hifi2_late" if hi2 else "lut_macro",
            output=PREFIX + label + ".json",
        )
    return result


SPECS = specifications()


def options(argv):
    result = {}
    for token in argv:
        if token.startswith("--"):
            if token in result:
                raise ValueError("Duplicate option: " + token)
            result[token] = []
            key = token
        elif not result:
            raise ValueError("Unexpected positional argument")
        else:
            result[key].append(token)
    return result


def expected_options(spec):
    result = {
        "--" + k.replace("_", "-"): [str(v)]
        for k, v in dict(
            label=spec["label"],
            length=spec["length"],
            heads=spec["heads"],
            cores=spec["cores"],
            seed=spec["seed"],
            distribution=spec["distribution"],
            sample_rows=spec["sample_rows"],
            read_barrier_tiles=2,
            warmup=0,
            iters=0,
            max_l2=1000000000,
        ).items()
    }
    result["--check-preprocess"] = []
    if spec["route"].endswith("lut"):
        result["--lut-exp"] = []
    if spec["route"].startswith("hi2"):
        result.update({"--variant": ["hi2_fp32_bf16"], "--native-exp": [], "--reader-chain": []})
    return result


def plan_specs(plan, repeat_suffix="-v2"):
    if not re.fullmatch(r"-[A-Za-z0-9_-]+", repeat_suffix) or repeat_suffix == "-v1":
        raise ValueError("Repeat suffix must be a safe distinct suffix, e.g. -v2")
    result, coverage, scopes = {}, set(), set()
    for row in plan.get("cases", []):
        opt = options(shlex.split(row["args"]))
        value = lambda name: opt["--" + name][0]
        label = value("label")
        if not re.fullmatch(r"[A-Za-z0-9_-]+-v1", label) or row["label"] != label:
            raise ValueError("Expected safe unique -v1 labels")
        if row["driver"] not in ("hifi2_bf16_lut_fullchip.py", "exp_lut_macro_streaming.py"):
            raise ValueError("Unexpected numerical producer")
        hi2 = row["driver"].startswith("hifi2")
        route = ("hi2" if hi2 else "lofi") + ("_lut" if "--lut-exp" in opt else "_native")
        spec = dict(
            label=label,
            route=route,
            distribution=value("distribution"),
            seed=int(value("seed")),
            repeat=1,
            driver=row["driver"],
            family="hifi2_late" if hi2 else "lut_macro",
            output=PREFIX + label + ".json",
            length=int(value("length")),
            heads=int(value("heads")),
            cores=int(value("cores")),
            sample_rows=int(value("sample-rows")),
        )
        if (
            spec["length"] not in (1024, 32768, 262144)
            or spec["heads"] != 2
            or not 2 <= spec["cores"] <= 110
            or spec["cores"] % 2
        ):
            raise ValueError("Unsupported length/head/core scope")
        if spec["sample_rows"] != (1024 if spec["length"] == 1024 else 128):
            raise ValueError("N1024 needs all Q; N32768/N262144 need Q128")
        key = (route, spec["distribution"], spec["seed"])
        if key in coverage or label in result:
            raise ValueError("Duplicate configuration/label")
        coverage.add(key)
        scopes.add(tuple(spec[k] for k in ("length", "heads", "cores", "sample_rows")))
        result[label] = spec
        repeated = dict(spec, label=label[:-3] + repeat_suffix, repeat=2)
        repeated["output"] = PREFIX + repeated["label"] + ".json"
        result[repeated["label"]] = repeated
    if len(scopes) != 1:
        raise ValueError("Plan must have one shared shape/core/reference scope")
    length = next(iter(scopes))[0]
    dists = {1024: DISTRIBUTIONS, 32768: LONG_DISTRIBUTIONS, 262144: CONTEXT256K_DISTRIBUTIONS}[length]
    if coverage != set(itertools.product(ROUTES, dists, SEEDS)):
        raise ValueError("Plan does not have the exact route/distribution/two-seed Cartesian coverage")
    return result


def check_plan(plan, specs=None):
    failures = []
    specs = plan_specs(plan) if specs is None else specs
    cases = plan.get("cases", [])
    expected = {name for name, s in specs.items() if s["repeat"] == 1}
    labels = [r.get("label") for r in cases]
    if len(labels) != len(expected) or set(labels) != expected:
        failures.append("Plan must contain exactly the unique expected v1 configurations")
    for row in cases:
        spec = specs.get(row.get("label"))
        if not spec:
            continue
        try:
            argv = shlex.split(row["args"])
            if options(argv) != expected_options(spec):
                failures.append(spec["label"] + ": CLI numerical/scope options differ from contract")
            if row["driver"] != spec["driver"] or row["output"] != spec["output"]:
                failures.append(spec["label"] + ": wrong producer/output")
            if shlex.split(row["command"]) != ["python_env/bin/python", "-B", PREFIX + spec["driver"], *argv]:
                failures.append(spec["label"] + ": command differs from declared args")
        except (KeyError, ValueError, TypeError) as error:
            failures.append(str(error))
    return failures


def validate_case(audit, e, spec, r):
    audit.current_provenance(e, r, spec)
    hi2, lut = spec["route"].startswith("hi2"), spec["route"].endswith("lut")
    expected = dict(
        label=spec["label"],
        distribution=spec["distribution"],
        seed=spec["seed"],
        length=spec["length"],
        heads=spec["heads"],
        cores=spec["cores"],
        actual_cores=spec["cores"],
        sample_rows=spec["sample_rows"],
        check_preprocess=True,
        read_barrier_tiles=2,
        warmup=0,
        iters=0,
        trace_repeats=1,
        lut_exp=lut,
        q_chunk=256,
        input_slots=1,
        fp32_dst=True,
        native_exp=True,
        bfp8_pack_precise=False,
        fix_correction=False,
        useful_flops=4 * spec["heads"] * spec["length"] ** 2 * 128,
    )
    for key, value in expected.items():
        e.equal(r.get(key), value, key)
    e.require(r.get("max_l2") == 1e9, "Abort safety limit differs; this is not an acceptance criterion")
    sampled = r.get("sampled_query_rows")
    if spec["length"] == 1024:
        e.equal(sampled, list(range(1024)), "All 1024 Q rows referenced")
    else:
        e.require(
            isinstance(sampled, list)
            and len(sampled) == 128
            and len(set(sampled)) == 128
            and sampled == sorted(sampled)
            and sampled[0] == 0
            and sampled[-1] == spec["length"] - 1
            and all(type(x) is int and 0 <= x < spec["length"] for x in sampled),
            "128 explicit sorted query rows",
        )
    e.require(isinstance(r.get("accuracy_scope"), str) and bool(r["accuracy_scope"]), "Missing reference scope")
    V.metric(e, r.get("accuracy"), "Original-input FP64 accuracy, no L2 cutoff")
    e.digest(r.get("output_sha256"), "Complete BF16 output hash")
    F.no_timing(e, r)
    checks = r.get("preprocessing_checks")
    F.exact_preprocessing(e, checks)
    defines = r.get("defines", {})
    for flag in ("SDPA_FP32_STREAMING", "SDPA_FP32_STATE", "SDPA_LOFI_NATIVE_EXP"):
        e.equal(defines.get(flag), "1", "Kernel flag " + flag)
    for flag in ("SDPA_LOFI_LUT_EXP", "SDPA_LOFI_LUT_MACRO"):
        e.equal(defines.get(flag), "1" if lut else None, "Paired exp flag " + flag)
    audit.assertion_witness(e, r, spec["driver"], "mismatch == 0")
    audit.assertion_witness(e, r, spec["driver"], "torch.isfinite(actual).all()")
    audit.assertion_witness(
        e, r, spec["driver"], "reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])", ast.Assign
    )
    audit.assertion_witness(
        e,
        r,
        spec["driver"],
        "inputs = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, args.distribution)",
        ast.Assign,
    )
    audit.assertion_witness(e, r, spec["driver"], "args.length // 512", ast.Assign)
    if hi2:
        for key, value in dict(
            variant="hi2_fp32_bf16",
            reader_chain=True,
            reader_split=False,
            reader_linear_k=False,
            q_prescale=1.0,
            center_k=False,
            b8_rne=False,
            finite=True,
            sources_unchanged=True,
            trace_equal=True,
            correctness_trace_replays=2,
        ).items():
            e.equal(r.get(key), value, "HiFi2 " + key)
        e.equal(r.get("replay_output_sha256"), [r["output_sha256"]] * 2, "Two complete trace replay hashes")
        original = r.get("original_input_sha256")
        e.require(isinstance(original, list) and len(original) == 3, "Three original input hashes")
        for digest in original or []:
            e.digest(digest, "Original BF16 input")
        for field in ("cpu_inputs_unchanged", "device_inputs_unchanged"):
            e.equal(r.get(field), True, field)
        for stage in ("initial_input_check", "replay_input_check", "final_input_check"):
            obj = r.get(stage, {})
            for field in ("cpu_inputs_unchanged", "device_inputs_unchanged"):
                e.equal(obj.get(field), True, stage + ": " + field)
            for field in ("cpu_input_sha256", "device_input_sha256"):
                e.equal(obj.get(field), original, stage + ": " + field)
        for c in checks or []:
            e.digest(c.get("output_sha256"), "Prepared " + str(c.get("input")))
            e.equal(c.get("storage"), "BF16", "Prepared storage")
            e.equal(c.get("bits"), 7 if c.get("input") == "Q" else 8, "Prepared width")
            if c.get("input") in ("K", "V"):
                e.equal(c.get("identity_bits_preserved"), True, "BF16 identity preparation")
        e.summary.update(
            trace_replay="Two complete-output bit-identical traces", input_integrity="Explicit CPU/device hash gates"
        )
    else:
        for key, value in dict(
            destination="fp32",
            kv_formats="b8_b8",
            k_format="b8",
            v_format="b8",
            k_chunk=512,
            head_dim=128,
            fidelity="LoFi",
            raw_lut=False,
            denominator_matches_pv=True,
            p_sfpu_prerounding=False,
            exp_lut_refinement=lut,
            trace_equal=None,
        ).items():
            e.equal(r.get(key), value, "LoFi " + key)
        for c in checks or []:
            e.equal(c.get("format"), "bf16" if c.get("input") == "Q" else "b8", "Prepared format")
        audit.assertion_witness(
            e, r, spec["driver"], "hashlib.sha256(p.read_bytes()).hexdigest() == provenance[str(p.relative_to(ROOT))]"
        )
        e.summary.update(
            trace_replay="NOT_RUN: iters=0 returns before trace capture",
            input_integrity="NOT_RECORDED: no original/prepared tensor hashes or immutability gate",
            during_run_source_stability="CURRENT producer assertion witness",
            finite_output="CURRENT producer assertion witness",
        )
        e.notes.append(
            "Exact preparation is value equality, not a signed-zero bit-identity claim. Separate-process output repetition is not trace replay."
        )
    e.summary.update(
        route=spec["route"],
        distribution=spec["distribution"],
        seed=spec["seed"],
        repeat=spec["repeat"],
        l2_pct=r.get("accuracy", {}).get("l2_pct"),
        pcc=r.get("accuracy", {}).get("pcc"),
        original_reference="Original BF16 Q/K/V, FP64, all heads/KV and explicit Q rows",
        reference_query_rows=spec["sample_rows"],
        all_query_reference=spec["sample_rows"] == spec["length"],
        timing=False,
        numerical_cutoff=None,
        abort_safety_limit=1e9,
    )


PAIR_SETTINGS = (
    "length",
    "heads",
    "cores",
    "q_chunk",
    "k_chunk",
    "head_dim",
    "input_slots",
    "seed",
    "distribution",
    "fidelity",
    "fp32_dst",
    "destination",
    "kv_formats",
    "q_prescale",
    "center_k",
    "b8_rne",
    "bfp8_pack_precise",
    "q_preprocessing",
    "k_preprocessing",
    "v_preprocessing",
    "k_format",
    "v_format",
    "preprocessing_checks",
    "sampled_query_rows",
    "reader_chain",
    "reader_split",
    "reader_linear_k",
    "read_barrier_tiles",
    "fix_correction",
    "cb_bytes_per_core",
)


def compare_pair(e, native, lut, hi2):
    for key in PAIR_SETTINGS:
        if key in native or key in lut:
            e.equal(native.get(key), lut.get(key), "Exp-only paired contract: " + key)
    e.equal(native.get("source_sha256"), lut.get("source_sha256"), "Same producer/dependency bytes across exp pair")
    remove = {"SDPA_LOFI_LUT_EXP", "SDPA_LOFI_LUT_MACRO"}
    e.equal(
        {k: v for k, v in native.get("defines", {}).items() if k not in remove},
        {k: v for k, v in lut.get("defines", {}).items() if k not in remove},
        "Only exp-refiner defines change",
    )
    if hi2:
        e.equal(
            native.get("original_input_sha256"), lut.get("original_input_sha256"), "Original hashes across exp pair"
        )
        e.summary["paired_input_basis"] = "Recorded original and prepared tensor hashes identical"
    else:
        e.summary["paired_input_basis"] = (
            "Same seed/config and pinned deterministic generator plus exact preparation oracle gates; tensor hashes NOT_RECORDED"
        )
    e.summary.update(
        native_l2_pct=native["accuracy"]["l2_pct"],
        lut_l2_pct=lut["accuracy"]["l2_pct"],
        l2_change_pct_points=lut["accuracy"]["l2_pct"] - native["accuracy"]["l2_pct"],
    )


def compare_repeat(e, first, second):
    # A label is the sole intended difference between separate-process runs.
    e.equal(
        {k: v for k, v in first.items() if k != "label"},
        {k: v for k, v in second.items() if k != "label"},
        "Full recorded result repeats exactly",
    )
    e.equal(first["output_sha256"], second["output_sha256"], "Separate-process complete-output hash equality")
    e.summary.update(output_sha256=first["output_sha256"], trace_replay_claim=False)


def run(plan_path=PLAN, root=ROOT, directory=HERE, repeat_suffix="-v2"):
    root, directory, plan_path = Path(root).resolve(), Path(directory).resolve(), Path(plan_path)
    plan_bytes = plan_path.read_bytes()
    plan = V.decode(plan_bytes.decode())
    specs = plan_specs(plan, repeat_suffix)
    failures = check_plan(plan, specs)
    audit = F.CurrentAudit(root, directory)
    records, snapshots = {}, {plan_path: plan_bytes}
    for label, spec in specs.items():
        e = V.Evidence(label + ".json", "exp_stress_case")
        path = directory / (label + ".json")
        if not path.is_file():
            e.pending.append("Expected stress result absent")
        else:
            try:
                r, raw = F.read_record(path)
                snapshots[path] = raw
                e.summary["record_sha256"] = hashlib.sha256(raw).hexdigest()
                validate_case(audit, e, spec, r)
                if not e.failures:
                    records[label] = r
            except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, SyntaxError) as error:
                e.failures.append(type(error).__name__ + ": " + str(error))
        audit.items.append(e.export())
    index = {(s["route"], s["distribution"], s["seed"], s["repeat"]): name for name, s in specs.items()}
    dists = sorted({s["distribution"] for s in specs.values()})
    for precision, distribution, seed, repeat in itertools.product(("lofi", "hi2"), dists, SEEDS, (1, 2)):
        e = V.Evidence(f"pair:{precision}:{distribution}:{seed}:v{repeat}", "exp_pair")
        names = [index[(precision + "_" + exp, distribution, seed, repeat)] for exp in ("native", "lut")]
        if all(n in records for n in names):
            compare_pair(e, *(records[n] for n in names), hi2=precision == "hi2")
        else:
            e.pending.append("Exp pair incomplete")
        audit.items.append(e.export())
    for route, distribution, seed in itertools.product(ROUTES, dists, SEEDS):
        e = V.Evidence(f"repeat:{route}:{distribution}:{seed}", "process_repeat")
        names = [index[(route, distribution, seed, i)] for i in (1, 2)]
        if all(n in records for n in names):
            first, second = (records[n] for n in names)
            compare_repeat(e, first, second)
        else:
            e.pending.append("Separate-process repeat incomplete")
        audit.items.append(e.export())
    for path, raw in snapshots.items():
        if not path.is_file() or path.read_bytes() != raw:
            failures.append("Plan/record changed during audit: " + str(path))
    result = audit.result()
    result.update(
        schema="exp-stress-current-v1",
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        plan_failures=failures,
        status="FAIL" if failures or result["sources_changed_during_audit"] else result["evidence_status"],
        expected_configs=len(specs) // 2,
        expected_results=len(specs),
        expected_exp_pairs=len(specs) // 2,
        expected_process_repeats=len(specs) // 2,
        repeated_label_contract="Planned -v1 commands repeated unchanged except fresh " + repeat_suffix + " labels",
        limitations=[
            "Separate from the earlier 85-output final qualification; no timing or model-quality acceptance",
            "No common L2 cutoff; max_l2=1e9 is only producer abort safety",
            "LoFi iters=0: no trace replay, original/prepared tensor hashes, or input immutability gate",
            "LoFi paired inputs rely on same seeded producer plus exact value oracles, not measured bit hashes",
            "No historical-source witness fallback; principal dependencies are not a full compiler/firmware closure",
        ],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Audit JSON to stdout only")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=HERE)
    parser.add_argument("--repeat-suffix", default="-v2", help="Fresh repeat label suffix; use --repeat-suffix=-v2")
    args = parser.parse_args()
    try:
        r = run(args.plan, args.repo_root, args.evidence_dir, args.repeat_suffix)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        r = dict(status="FAIL", plan_failures=[type(error).__name__ + ": " + str(error)], families={}, evidence=[])
    if args.json:
        print(json.dumps(r, indent=2, allow_nan=False))
    else:
        print(
            "Exp stress:",
            r["status"],
            "|",
            r.get("expected_configs"),
            "configurations,",
            r.get("expected_results"),
            "required results",
        )
        for family, summary in r["families"].items():
            print(family, summary)
        for error in r["plan_failures"]:
            print("PLAN FAIL", error)
        for e in r["evidence"]:
            if e["status"] == "FAIL":
                print("FAIL", e["file"], e["failures"])
    return 0 if r["status"] == "PASS" else 2 if r["status"] == "PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
