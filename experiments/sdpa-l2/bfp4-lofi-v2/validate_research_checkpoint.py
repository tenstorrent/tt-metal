# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standard-library, read-only checkpoint evidence audit; --json writes stdout only.

No Torch/TTNN imports, source execution, subprocesses, devices, or file writes.
Coverage integrity is not a new SKU accuracy threshold or hardware retest.
Historical run stability, current source drift, and missing evidence are separate.
"""
import argparse
import ast
import collections
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = "experiments/sdpa-l2/bfp4-lofi-v2/"
SHA = re.compile(r"^[0-9a-f]{64}$")
DISTRIBUTIONS = ("normal", "outliers", "scaled_qk", "scaled_down", "biased_v", "common_q", "common_k",
                 "common_v", "constant_v", "uniform", "uniform_constant_v", "channel_k", "channel_v")
CHAIN = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp"
REPRO = "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
FROZEN = "experiments/sdpa-l2/hybrid-mixed-v1/candidate/"
COMMON = FROZEN + "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
SFPU = FROZEN + "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h"


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(x):
    raise ValueError("Nonfinite JSON constant: " + x)


def strict_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Overflowed JSON float: " + text)
    return value


def decode(text):
    return json.loads(text, object_pairs_hook=strict_object, parse_constant=reject_constant, parse_float=strict_float)


class Evidence:
    def __init__(self, name, family, optional=False):
        self.name, self.family, self.optional = name, family, optional
        self.failures, self.pending, self.notes = [], [], []
        self.summary, self.provenance = {}, {}

    def require(self, condition, message):
        if not condition:
            self.failures.append(message)

    def equal(self, actual, expected, label):
        self.require(type(actual) is type(expected) and actual == expected,
                     f"{label}: expected {expected!r}, observed {actual!r}")

    def number(self, x, label):
        self.require(finite(x) and x >= 0, label + ": invalid nonnegative finite number")

    def close(self, actual, expected, label):
        self.require(finite(actual) and finite(expected) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
                     f"{label}: arithmetic mismatch ({actual!r} vs {expected!r})")

    def digest(self, x, label):
        self.require(isinstance(x, str) and SHA.fullmatch(x) is not None, label + ": invalid SHA256")

    def export(self):
        return dict(file=self.name, family=self.family, optional=self.optional,
                    status="FAIL" if self.failures else "PENDING" if self.pending else "PASS",
                    failures=self.failures, pending=self.pending, notes=self.notes,
                    summary=self.summary, provenance=self.provenance)


def metric(e, value, label):
    e.require(isinstance(value, dict), label + ": missing metric object")
    if not isinstance(value, dict):
        return
    e.number(value.get("l2_pct"), label + ".l2_pct")
    pcc = value.get("pcc")
    e.require(pcc is None or (finite(pcc) and -1.000000000001 <= pcc <= 1.000000000001), label + ": invalid PCC")


def timing(e, obj, label, iterations, zero=False):
    e.require(isinstance(obj, dict), label + ": missing timing object")
    if not isinstance(obj, dict):
        return None
    samples, median = obj.get("replay_ms"), obj.get("median_ms")
    e.require(isinstance(samples, list), label + ": replay_ms must be a list")
    if not isinstance(samples, list):
        return None
    if zero and median == 0 and samples == []:
        return 0.0
    if iterations == 0:
        e.equal(samples, [], label + ": smoke samples")
        e.equal(median, None, label + ": smoke median")
        return None
    e.equal(len(samples), iterations, label + ": sample count")
    e.require(bool(samples) and all(finite(x) and x > 0 for x in samples), label + ": invalid sample")
    if samples and all(finite(x) for x in samples):
        e.close(median, statistics.median(samples), label + ": median")
    e.require(finite(median) and median > 0, label + ": invalid median")
    return median if finite(median) and median > 0 else None


def throughput(e, r, flops, name, ms):
    if ms is None:
        e.equal(r.get(name + "_tflops"), None, name + ": absent smoke TFLOPs")
    else:
        e.close(r.get(name + "_tflops"), flops / (ms * 1e9), name + ": TFLOP arithmetic")


def rows(e, values, length, count, label):
    e.require(isinstance(values, list), label + ": absent query rows")
    if not isinstance(values, list):
        return
    e.equal(len(values), min(count, length), label + ": row count")
    e.require(all(type(x) is int and 0 <= x < length for x in values), label + ": invalid row")
    e.require(values == sorted(set(values)), label + ": duplicate/unsorted rows")
    if len(values) > 1:
        e.require(values[0] == 0 and values[-1] == length - 1, label + ": missing endpoints")


def required_manifest(family, r):
    base = {REPRO, V2 + "numerics.py", V2 + "preprocess.py", V2 + "preprocess/compute.cpp"}
    if family == "adaptive_primitive":
        return {V2 + x for x in ("adaptive_bfp4_round.py", "adaptive_bfp4_round/compute.cpp", "bfp4_round.py",
                                "bfp4_round/reader.cpp", "bfp4_round/writer.cpp")}
    if family.startswith("identity4"):
        resident = r.get("resident")
        files = ("compute_resident.cpp", "reader_resident.cpp", "writer_resident.cpp") if resident else (
            "compute.cpp", "reader_chain.cpp", "writer.cpp")
        return base | {CHAIN, COMMON, SFPU, V2 + "identity4_streaming.py", V2 + "exp_native.hpp",
                       V2 + "exp_refiner.hpp", V2 + "identity4_streaming/compute_streaming.hpp",
                       V2 + ("identity4_resident.py" if resident else "identity4_streaming.py"),
                       *(V2 + "identity4_streaming/" + x for x in files)}
    base |= {CHAIN, V2 + "fullchip/compute.cpp", V2 + "fullchip/reader_chain.cpp", V2 + "fullchip/writer.cpp"}
    if family in ("native_suite", "accurate_kcenter"):
        base |= {COMMON, SFPU, V2 + "fullchip.py", V2 + "native_exp_qualification.py",
                 V2 + "streaming/compute_streaming.hpp", V2 + "exp_native.hpp", V2 + "exp_refiner.hpp",
                 V2 + "center_mean.py", V2 + "center_preprocess.py", V2 + "center_preprocess/compute.cpp",
                 V2 + "center_preprocess/round.hpp", "experiments/sdpa-l2/bfp4-lofi-v1/probe.py",
                 "experiments/sdpa-l2/bfp4-lofi-v1/numerics.py", "experiments/sdpa-l2/frontier-accuracy-v1/run.py"}
        if family == "accurate_kcenter":
            base.add(V2 + "accurate_kcenter_diagnostic.py")
    elif family == "value_centering":
        fast = "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/"
        base |= {fast + "compute_common.hpp", fast + "compute_streaming.hpp", V2 + "fast_correction.hpp",
                 V2 + "safe_rescale.hpp", V2 + "exp_refiner.hpp", V2 + "center_mean.py",
                 V2 + "center_preprocess.py", V2 + "center_preprocess/compute.cpp", V2 + "center_preprocess/round.hpp",
                 V2 + "value_centered_fullchip.py"}
        if r.get("v_format") == "b8":
            base.discard(V2 + "value_centered_fullchip.py")
            base |= {V2 + "value_centered_b8_fullchip.py", V2 + "effective_v_preprocess.py",
                     V2 + "effective_v_preprocess/compute.cpp"}
    return base


class Audit:
    def __init__(self, root, directory):
        self.root, self.directory = root.resolve(), directory.resolve()
        self.hashes, self.items = {}, []

    def current_hash(self, relative):
        if relative in self.hashes:
            return self.hashes[relative]
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            return None
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_file():
            return None
        self.hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self.hashes[relative]

    def provenance(self, e, r, manifest=None):
        manifest = r.get("source_sha256") if manifest is None else manifest
        malformed, drift, missing = [], [], []
        if not isinstance(manifest, dict) or not manifest:
            malformed.append("Absent/empty source_sha256 mapping")
            manifest = {}
        omissions = sorted(required_manifest(e.family, r) - set(manifest))
        for name, expected in sorted(manifest.items()):
            if not isinstance(expected, str) or SHA.fullmatch(expected) is None:
                malformed.append(name + ": invalid SHA256")
                continue
            observed = self.current_hash(name)
            if observed is None:
                missing.append(name)
            elif observed != expected:
                drift.append(dict(source=name, recorded_sha256=expected, current_sha256=observed))
        e.provenance = dict(
            status="WARN" if omissions or drift or missing or malformed else "PASS", manifest_entries=len(manifest),
            manifest_omissions=omissions, current_source_drift=drift, unavailable_current_sources=missing,
            malformed_manifest=malformed, policy="Explicit critical sources, not complete compiler/firmware/transitive closure")

    def assertion_witness(self, e, r, driver, expression, node_type=ast.Assert):
        name = V2 + driver
        if self.current_hash(name) != r.get("source_sha256", {}).get(name) or self.current_hash(name) is None:
            e.pending.append("Unpinned/drifted producer prevents assertion witness: " + expression)
            return False
        tree = ast.parse((self.root / name).read_text())
        expressions = [ast.unparse(n.test if isinstance(n, ast.Assert) else n)
                       for n in ast.walk(tree) if isinstance(n, node_type)]
        found = any(expression in x for x in expressions)
        e.require(found, "Pinned producer lacks assertion: " + expression)
        return found

    def load(self, name, family, callback, optional=False, **kwargs):
        e = Evidence(name, family, optional)
        path = self.directory / name
        if not path.is_file():
            e.pending.append("Expected evidence is absent; not a successful measurement")
        else:
            try:
                data = path.read_bytes()
                e.summary["record_sha256"] = hashlib.sha256(data).hexdigest()
                value = [decode(x) for x in data.decode().splitlines() if x.strip()] if name.endswith(".jsonl") else decode(data.decode())
                callback(self, e, value, **kwargs)
                e.require(path.read_bytes() == data, "Record changed during audit")
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, SyntaxError) as error:
                e.failures.append(type(error).__name__ + ": " + str(error))
        self.items.append(e.export())

    def result(self):
        families = {}
        for name in sorted({x["family"] for x in self.items}):
            subset = [x for x in self.items if x["family"] == name]
            counts = dict(collections.Counter(x["status"] for x in subset))
            families[name] = dict(status="FAIL" if counts.get("FAIL") else "PENDING" if counts.get("PENDING") else "PASS",
                                  files=len(subset), counts=counts)
        failed = any(x["status"] == "FAIL" for x in self.items)
        pending = any(x["status"] == "PENDING" for x in self.items)
        warnings = [x["file"] for x in self.items if x["provenance"].get("status") == "WARN"]
        changed = [p for p, h in self.hashes.items() if not (self.root / p).is_file()
                   or hashlib.sha256((self.root / p).read_bytes()).hexdigest() != h]
        return dict(
            schema_version=1, evidence_status="FAIL" if failed else "PENDING" if pending else "PASS",
            checkpoint_status="FAIL" if failed or changed else "PENDING" if pending else "PROVENANCE_WARNING" if warnings else "PASS",
            provenance_status="FAIL" if changed else "WARN" if warnings else "PASS",
            provenance_warning_files=warnings, sources_changed_during_audit=changed,
            repository_root=str(self.root), evidence_directory=str(self.directory), families=families, evidence=self.items,
            limitations=["Record validation, not device execution or raw-tensor recomputation",
                         "Finite/replay coverage is not universal L2/PCC acceptance",
                         "Missing optional planned measurements stay PENDING",
                         "Recorded run stability differs from present source drift",
                         "No grid7/LUT-macro or unlisted historical experiments are promoted"])


def envelope(audit, e, records, kind):
    e.require(isinstance(records, list) and bool(records), "Empty/non-list JSONL")
    if not records:
        return {}, []
    provenance = [r for r in records if r.get("kind") == "provenance"]
    complete = [r for r in records if r.get("kind") == "complete"]
    e.equal(len(provenance), 1, "Provenance count")
    e.require(records[0].get("kind") == "provenance", "Provenance is not first")
    if not complete:
        e.pending.append("No completion footer; stream may still be running")
    else:
        e.equal(len(complete), 1, "Completion count")
        e.require(records[-1].get("kind") == "complete", "Completion is not last")
        e.equal(complete[0].get("sources_unchanged"), True, "Recorded during-run source stability")
    e.require(all(r.get("kind") in ("provenance", "complete", kind) for r in records), "Unexpected record kind")
    p = provenance[0] if provenance else {}
    audit.provenance(e, p)
    e.summary["during_run_sources_unchanged"] = complete[0].get("sources_unchanged") if complete else None
    return p, [r for r in records if r.get("kind") == kind]


def coverage(e, observed, expected):
    e.equal(len(observed), len(set(observed)), "Duplicate case keys")
    missing, extra = sorted(expected - set(observed)), sorted(set(observed) - expected)
    if missing and e.pending:
        e.pending.append("Unfinished cases: " + str(len(missing)))
    else:
        e.require(not missing, "Missing expected case keys: " + repr(missing))
    e.require(not extra, "Unexpected case keys: " + repr(extra))
    e.summary.update(cases=len(observed), expected_cases=len(expected), missing_case_keys=missing, extra_case_keys=extra)


def native_suite(audit, e, records, length):
    p, cases = envelope(audit, e, records, "qualification")
    args = p["args"]
    variants = ("lofi_fp32_b8", "lofi_fp32_b4", "accurate")
    for key, value in (("lengths", [length]), ("seeds", [1240, 1241])):
        e.equal(args.get(key), value, "Suite " + key)
    e.equal(set(args.get("distributions", [])), set(DISTRIBUTIONS), "Suite distributions")
    e.equal(set(args.get("variants", [])), set(variants), "Suite variants")
    expected = {(length, s, d, v, c) for s, d, v in itertools.product((1240, 1241), DISTRIBUTIONS, variants)
                for c in ((False, True) if d == "common_k" and v != "accurate" else (False,))}
    keys, inputs, bands = [], {}, collections.defaultdict(list)
    for i, r in enumerate(cases):
        label = f"case[{i}]"
        key = (r["length"], r["seed"], r["distribution"], r["variant"], r["center_k"])
        keys.append(key)
        e.require(type(r["center_k"]) is bool, label + ": center_k must be boolean")
        for name, value in (("finite", True), ("nonfinite_count", 0), ("trace_equal", True), ("status", "FINITE_REPLAY_IDENTICAL")):
            e.equal(r.get(name), value, label + "." + name)
        e.digest(r.get("output_sha256"), label + ": output SHA")
        hashes = r.get("original_input_sha256", [])
        e.equal(len(hashes), 3, label + ": Q/K/V hashes")
        for h in hashes:
            e.digest(h, label + ": input SHA")
        e.require(key[:3] not in inputs or inputs[key[:3]] == hashes, label + ": changed original inputs across variants")
        inputs[key[:3]] = hashes
        config, kernel = r["config"], r["kernel"]
        for name, value in (("q_chunk", 256), ("length", length), ("variant", r["variant"]),
                            ("center_k", r["center_k"]), ("native_exp", r["variant"] != "accurate")):
            e.equal(config.get(name), value, label + ".config." + name)
        # ACCURATE deliberately consumes original BF16 inputs without a LoFi
        # quantizer; false here means no preprocessing, not CPU quantization.
        for name, value in (("input_slots", 1), ("fp32_dst", True),
                            ("device_preprocessing", r["variant"] != "accurate")):
            e.equal(kernel.get(name), value, label + ".kernel." + name)
        rows(e, r.get("sampled_query_rows"), length, args["sample_rows"], label)
        metric(e, r["metrics"].get("original"), label + ": original")
        metric(e, r["metrics"].get("bf16_output_rounding_floor"), label + ": rounding floor")
        residual, undefined = r["metrics"].get("residual_l2_pct"), r["metrics"].get("residual_relative_undefined")
        e.require((undefined is True and residual is None) or (undefined is False and finite(residual) and residual >= 0),
                  label + ": undefined residual-relative contract")
        bands[r["variant"]].append(r["metrics"]["original"]["l2_pct"])
    coverage(e, keys, expected)
    e.summary["l2_ranges_pct"] = {k: [min(v), max(v)] for k, v in bands.items() if v and all(finite(x) for x in v)}
    e.notes.append("82 finite/replay cases is coverage integrity, not 82 universal accuracy passes")
    if args.get("check_preprocess") is False:
        e.notes.append("Exact preprocessor-output option was disabled in this suite")


def adaptive_primitive(audit, e, r, search, fmt, distribution):
    audit.provenance(e, r)
    for k, v in (("search", search), ("output_format", fmt), ("distribution", distribution), ("host_only", False),
                 ("fp32_dst", True), ("mismatch", 0), ("decoded_bit_mismatch", 0), ("dst_tiles_reserved", 3), ("batch", 1)):
        e.equal(r.get(k), v, k)
    e.equal(r.get("numel"), r["length"] * 128, "Input shape arithmetic")
    for k in ("actual_sha256", "expected_sha256"):
        e.digest(r.get(k), k)
    e.equal(r.get("actual_sha256"), r.get("expected_sha256"), "Decoded actual/oracle hashes")
    o = r["oracle"]
    for k in ("induced_exponent_mismatches", "native_grid_roundtrip_mismatches"):
        e.equal(o.get(k), 0, "Oracle " + k)
    e.equal(o.get("groups"), r["numel"] // 16, "Group count")
    counts = o.get("selected_counts", {})
    e.equal(set(counts), {"-1", "0", "1"}, "Selection keys")
    e.require(all(type(x) is int and x >= 0 for x in counts.values()), "Invalid selection counts")
    e.equal(sum(counts.values()), o["groups"], "Selection sum")
    if search == "baseline" or distribution == "zeros":
        e.equal(counts.get("0"), o["groups"], "Baseline/zero tie preference")
    if search == "minus":
        e.equal(counts.get("1"), 0, "Forbidden E+1 in minus search")
    e.number(r.get("quantization_l2_pct"), "Representation L2")
    timing(e, r, "primitive", r["iters"])
    e.summary.update(quantization_l2_pct=r["quantization_l2_pct"], selected_counts=counts,
                     fp64_selection_differences=o.get("selection_differs_fp64_groups"), hash_contract=r.get("hash_contract"))
    e.notes.append("Exact FP32-tree oracle; universal FP64-MSE tie agreement is not required; decoded signed zeros canonicalized")


def identity4(audit, e, r, resident, length=None):
    audit.provenance(e, r)
    for k, v in (("resident", resident), ("identity4_mode", "both"), ("qualification_pass", True),
                 ("bitwise_compared", True), ("off_on_bit_mismatches", 0), ("sources_unchanged", True)):
        e.equal(r.get(k), v, k)
    flops = 4 * 256 * 512 * 128 * r["q_repeats"] * r["k_chunks"] if resident else 4 * r["heads"] * r["length"]**2 * 128
    e.equal(r.get("useful_flops"), flops, "Useful FLOPs")
    if length is not None:
        e.equal(r.get("length"), length, "Full-chip length")
    if not resident:
        e.equal(r.get("q_chunk"), 256, "Top-level Q256")
        audit.assertion_witness(e, r, "identity4_streaming.py", "args.length // 512", node_type=ast.Assign)
        e.notes.append("Full-chip K512 uses matching pinned builder assignment; case records omit direct K-chunk metadata")
    cases = r.get("cases", [])
    e.equal(len(cases), 2, "Off/on count")
    e.equal([c.get("identity4") for c in cases], [False, True], "Off/on ordering")
    hashes, summaries = [], []
    for i, c in enumerate(cases):
        label = f"case[{i}]"
        for k in ("finite", "trace_bitwise_equal"):
            e.equal(c.get(k), True, label + "." + k)
        e.digest(c.get("output_sha256"), label + ": output SHA")
        hashes.append(c.get("output_sha256"))
        metric(e, c.get("accuracy"), label + ": accuracy")
        e.require(c["accuracy"]["l2_pct"] < r["max_l2"], label + ": recorded L2 gate")
        pcc = c["accuracy"].get("pcc")
        e.require(pcc is None or pcc >= r["min_pcc"], label + ": recorded PCC gate")
        if resident:
            e.equal(c.get("q_chunk"), 256, label + ": Q256")
            e.equal(c.get("k_chunk"), 512, label + ": K512")
        e.equal(c.get("input_slots"), {"q": 2, "k": 1, "v": 1} if resident else 1, label + ": slots")
        defines = dict(c["defines"])
        e.equal("SDPA_IDENTITY4" in defines, c["identity4"], label + ": optimization flag")
        defines.pop("SDPA_IDENTITY4", None)
        if i:
            off = dict(cases[0]["defines"])
            off.pop("SDPA_IDENTITY4", None)
            e.equal(defines, off, "Only identity4 changes defines")
            e.equal(c["cb_bytes_per_core"], cases[0]["cb_bytes_per_core"], "Off/on CB bytes")
        for name in ("attention", "combined"):
            ms = timing(e, c.get(name), label + "." + name, r["iters"])
            throughput(e, c, flops, name, ms)
        if c.get("preprocessing") is not None:
            timing(e, c["preprocessing"], label + ".preprocessing", r["iters"])
        summaries.append(dict(identity4=c["identity4"], l2_pct=c["accuracy"]["l2_pct"],
                              attention_ms=c["attention"]["median_ms"], attention_tflops=c["attention_tflops"]))
    e.require(len(hashes) == 2 and hashes[0] == hashes[1], "Off/on output hashes differ")
    e.summary.update(useful_flops=flops, cases=summaries, during_run_sources_unchanged=r.get("sources_unchanged"))
    e.notes += ["Repeated-KV resident does not qualify changing maxima; distinct full-chip is separately required",
                "No branch counters; equality does not prove every branch executed"]


def value_centering(audit, e, r, length, vfmt, mode, distribution):
    audit.provenance(e, r)
    for k, v in (("length", length), ("v_format", vfmt), ("k_format", "b8"), ("center_mode", mode),
                 ("distribution", distribution), ("destination", "fast_bf16"), ("denom_only", True),
                 ("fp32_dst", False), ("input_slots", 2), ("q_chunk", 256), ("k_chunk", 512),
                 ("device_preprocessing", True), ("fix_correction", True), ("trace_equal", True)):
        e.equal(r.get(k), v, k)
    driver = "value_centered_b8_fullchip.py" if vfmt == "b8" else "value_centered_fullchip.py"
    fw = audit.assertion_witness(e, r, driver, "torch.isfinite(actual).all()")
    sw = audit.assertion_witness(e, r, driver, "source_hashes[str(p.relative_to(ROOT))]")
    e.notes.append("Finite/run-stability witnesses are matching pinned producer assertions, not explicit booleans")
    metric(e, r.get("accuracy"), "Original-reference accuracy")
    c = r["centered_output_accuracy"]
    e.equal(c.get("gain_alignment"), False, "No gain fitting")
    e.require("Same FP64 mean(original V" in c.get("centering", ""), "Centered reference must share original-V mean")
    constant = distribution == "constant_v"
    e.equal(c.get("constant_v"), constant, "Constant-V classification")
    if constant:
        e.equal(c.get("l2_pct"), None, "Constant-V relative residual is undefined")
        e.require(bool(c.get("relative_error_undefined_reason")), "Missing undefined-relative explanation")
        e.equal(c.get("centered_reference_rms"), 0.0, "Analytical zero residual")
    else:
        e.number(c.get("l2_pct"), "Centered L2")
    for k in ("absolute_error_rms", "absolute_error_max"):
        e.number(c.get(k), "Centered " + k)
    if mode != "none":
        e.equal(r.get("epilogue_check", {}).get("mismatch"), 0, "BF16 epilogue oracle")
        e.equal(r.get("epilogue_materialized"), True, "Materialized epilogue")
    if mode == "matched_mean":
        e.equal(r["value_centering"].get("represented_v_exact_in_lofi_right_operand"), True, "Matched represented-V operand")
        if vfmt == "b8":
            e.equal(r["value_centering"].get("extra_full_v_truncation_in_preprocessing"), True, "V8 trunc5 pass")
    rows(e, r.get("sampled_query_rows"), length, r["sample_rows"], "Sample rows")
    e.digest(r.get("output_sha256"), "Output SHA")
    flops = 4 * r["heads"] * length**2 * 128
    e.equal(r.get("useful_flops"), flops, "Useful FLOPs")
    for name in ("attention", "preprocessing", "combined", "epilogue", "attention_with_epilogue"):
        ms = timing(e, r.get(name), name, r["iters"], zero=name == "epilogue" and mode == "none")
        if name in ("attention", "combined"):
            throughput(e, r, flops, name, ms)
    e.summary.update(l2_pct=r["accuracy"]["l2_pct"], pcc=r["accuracy"].get("pcc"), centered_l2_pct=c["l2_pct"],
                     attention_tflops=r["attention_tflops"], combined_tflops=r["combined_tflops"],
                     finite_producer_assertion=fw, sources_unchanged_producer_assertion=sw)
    e.notes.append("Attention-only TFLOPs exclude the measured epilogue; combined includes it")
    if r.get("check_preprocess") is False:
        e.notes.append("Full exact preprocessing checks disabled; epilogue oracle is separate")


def accurate_kcenter(audit, e, records, length):
    p, cases = envelope(audit, e, records, "diagnostic")
    args = p["args"]
    e.equal(args.get("lengths"), [length], "Diagnostic lengths")
    expected = {(length, 1240, d, c) for d, c in itertools.product(("normal", "common_k"), (False, True))}
    keys, summaries = [], []
    for i, r in enumerate(cases):
        label = f"case[{i}]"
        keys.append((r["length"], r["seed"], r["distribution"], r["center_k"]))
        for k in ("all_output_finite", "trace_equal", "immutable_inputs_verified"):
            e.equal(r.get(k), True, label + "." + k)
        e.equal(r["config"].get("variant"), "accurate", label + ": accurate control")
        e.equal(r["preparation"].get("shifted_bf16_mismatches"), 0, label + ": BF16 shift oracle")
        e.equal(r["preparation"].get("immutable_original_k_verified"), True, label + ": immutable K")
        for k in ("original_reference", "kernel_vs_actual_prepared_bf16_reference", "bf16_shift_attention_drift",
                  "exact_shift_attention_invariance", "original_bf16_output_rounding_floor"):
            metric(e, r["metrics"].get(k), label + ".metrics." + k)
        invariant = r["metrics"]["exact_shift_attention_invariance"]["l2_pct"]
        e.require(invariant < 1e-7, label + ": exact FP64 K-shift invariance exceeds 1e-7 percent")
        e.digest(r.get("output_sha256"), label + ": output SHA")
        e.equal(r.get("useful_attention_flops"), 4 * args["heads"] * length**2 * 128, label + ": useful FLOPs")
        if args["iters"]:
            names = {"attention", "combined"} | ({"mean", "subtract", "preprocessing"} if r["center_k"] else set())
            e.equal(set(r["timings"]), names, label + ": timing stages")
            for name, t in r["timings"].items():
                timing(e, t, label + ".timings." + name, args["iters"])
        else:
            e.equal(r.get("timings"), {}, label + ": no fabricated smoke timing")
        summaries.append(dict(distribution=r["distribution"], center_k=r["center_k"],
                              l2_pct=r["metrics"]["original_reference"]["l2_pct"]))
    coverage(e, keys, expected)
    e.summary["accuracy"] = summaries
    e.notes.append("Diagnostic attribution, not promotion of K centering into the frozen accurate SKU")


def run_audit(root=ROOT, directory=HERE):
    a = Audit(root, directory)
    for suffix, n in (("32k-v1", 32768), ("256k-v2", 262144)):
        a.load("native-suite-" + suffix + ".jsonl", "native_suite", native_suite, length=n)
    primitive = [(s, "bf16", "normal") for s in ("baseline", "minus", "pm")]
    primitive += [("pm", "b4", d) for d in ("normal", "wide", "ties", "zeros", "group_outliers")]
    for search, fmt, dist in primitive:
        a.load(f"adaptive_bfp4_round/adaptive-{search}-{fmt}-{dist}-v1.json", "adaptive_primitive", adaptive_primitive,
               search=search, fmt=fmt, distribution=dist)
    for label, resident, n in (("resident-smoke", True, None), ("resident-perf", True, None),
                               ("distinct-smoke", False, 4096), ("fullchip-32768", False, 32768),
                               ("fullchip-262144", False, 262144)):
        a.load("identity4_streaming/identity4-" + label + "-v1.json",
               "identity4_resident" if resident else "identity4_fullchip", identity4, resident=resident, length=n)
    for vfmt, n, mode, dist in itertools.product(("b4", "b8"), (32768, 262144),
                                                ("none", "original_mean", "matched_mean"),
                                                ("normal", "common_v", "constant_v")):
        prefix = "valuecenter-b8-" if vfmt == "b8" else "valuecenter-"
        a.load(f"{prefix}{n}-{mode}-{dist}-v1.json", "value_centering", value_centering, optional=True,
               length=n, vfmt=vfmt, mode=mode, distribution=dist)
    for n, suffix in ((1024, "1024"), (32768, "32k")):
        a.load(f"accurate-kcenter-{suffix}-v1.jsonl", "accurate_kcenter", accurate_kcenter, optional=True, length=n)
    return a.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON on stdout only; no output-file option")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=HERE)
    args = parser.parse_args()
    r = run_audit(args.repo_root, args.evidence_dir)
    if args.json:
        print(json.dumps(r, indent=2, allow_nan=False))
    else:
        print("Checkpoint:", r["checkpoint_status"], "| evidence:", r["evidence_status"], "| provenance:", r["provenance_status"])
        for family, row in r["families"].items():
            print(f"  {family}: {row['status']} {row['counts']}")
        for item in r["evidence"]:
            if item["status"] != "PASS":
                print(" ", item["status"], item["file"], "; ".join(item["failures"] + item["pending"]))
        print("Provenance warning files:", len(r["provenance_warning_files"]))
        print("Use --json for omissions/drift/gates. No files were written.")
    return 1 if r["checkpoint_status"] == "FAIL" else 2 if r["checkpoint_status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
