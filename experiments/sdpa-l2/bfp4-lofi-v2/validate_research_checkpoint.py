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
WITNESS_SOURCES = V2 + "witness_sources/"
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
    if family in ("v8_axis", "combined_recipe", "paired_vaxis"):
        extra = {"v8_axis": (), "combined_recipe": ("combined_recipe_fullchip.py", "hadamard_preprocess.py",
                 "adaptive_bfp4_round.py", "adaptive_bfp4_round/compute.cpp"),
                 "paired_vaxis": ("paired_vaxis_timing.py",)}[family]
        return required_manifest("v_transpose", r) | {V2 + f for f in extra}
    if family == "mean_error":
        return required_manifest("value_centering", r) | {V2 + f for f in (
            "value_mean_error_fullchip.py", "effective_v_preprocess.py", "effective_v_preprocess/compute.cpp")}
    if family in ("native_storage", "hifi2_late"):
        driver = "fullchip.py" if family == "native_storage" else ("hifi2_bf16_lut_fullchip.py"
            if r.get("variant") == "hi2_fp32_bf16" else "hifi2_lut_fullchip.py" if "lut_exp" in r else "hifi2_native_fullchip.py")
        extra = {V2 + driver, CHAIN, COMMON, SFPU, *(V2 + f for f in (
            "fullchip/compute.cpp", "fullchip/reader_chain.cpp", "fullchip/writer.cpp", "preprocess/reader.cpp",
            "preprocess/writer.cpp", "streaming/compute_streaming.hpp", "exp_native.hpp", "exp_refiner.hpp"))}
        if r.get("b8_rne"):
            extra |= {V2 + "bfp8_round.py", V2 + "bfp8_round/compute.cpp"}
        if "lut_exp" in r:
            extra |= {V2 + "exp_lut.hpp", V2 + "exp_lut_macro.hpp"}
        return base | extra
    if family == "v_transpose":
        fast = "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/"
        tail = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/"
        return base | {CHAIN, fast + tail + "compute_common.hpp", fast + tail + "compute_streaming.hpp",
                       fast + "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h",
                       *(V2 + f for f in ("Vtransposed_fullchip.py", "bfp4_residual_preprocess.py",
                          "bfp4_round.py", "bfp4_round/compute.cpp", "bfp4_round/reader.cpp", "bfp4_round/writer.cpp",
                          "preprocess/reader.cpp", "preprocess/writer.cpp", "fast_correction.hpp", "safe_rescale.hpp",
                          "exp_refiner.hpp", "fullchip/compute.cpp", "fullchip/writer.cpp", "vtransposed/compute.cpp",
                          "vtransposed/reader_chain.cpp", "vtransposed/pv_transpose.hpp", "vtransposed/transpose_writer.cpp")),
                       "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp",
                       "tt_metal/hw/inc/api/compute/transpose.h",
                       "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_transpose_dest_api.h",
                       "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h"}
    if family == "captured_interface":
        prefixes = ("experiments/sdpa-l2/single-core-resident-v1/main/",
                    "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/")
        tail = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/"
        return base | {CHAIN, COMMON, SFPU,
                       *(p + tail + f for p in prefixes for f in ("compute_common.hpp", "compute_streaming.hpp")),
                       *(p + "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
                         for p in prefixes),
                       *(V2 + f for f in ("captured_inputs.py", "captured_fullchip.py", "fullchip.py",
                          "fullchip/compute.cpp", "fullchip/reader_chain.cpp", "fullchip/writer.cpp",
                          "fast_correction.hpp", "streaming/compute_streaming.hpp", "exp_native.hpp",
                          "exp_refiner.hpp", "bfp4_round.py", "bfp4_round/compute.cpp"))}
    if family == "codec_cpu":
        return {REPRO, V2 + "codec_limits_models.py", V2 + "adaptive_bfp4_round.py", V2 + "bfp4_round.py",
                "experiments/sdpa-l2/bfp4-lofi-v1/probe.py"}
    if family == "lut_macro":
        resident = "q_repeats" in r
        files = ("compute_resident.cpp", "reader_resident.cpp", "writer_resident.cpp") if resident else (
            "compute.cpp", "reader_chain.cpp", "writer.cpp")
        return base | {COMMON, SFPU, V2 + "exp_lut_macro_streaming.py", V2 + "exp_native.hpp",
                       V2 + "exp_lut.hpp", V2 + "exp_lut_macro.hpp", V2 + "exp_lut_models.py",
                       V2 + "streaming/compute_streaming.hpp",
                       *(V2 + "exp_lut_macro_streaming/" + f for f in files)} | (
                           {V2 + "exp_lut_macro_resident.py", V2 + "resident/reader.cpp", V2 + "resident/writer.cpp"}
                           if resident else {CHAIN})
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
    if family == "adaptive_fullchip":
        fast = "experiments/sdpa-l2/bf16-denom-pair-v3/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/"
        return base | {fast + "compute_common.hpp", fast + "compute_streaming.hpp",
                       *(V2 + f for f in ("adaptive_fullchip.py", "adaptive_bfp4_round.py",
                         "adaptive_bfp4_round/compute.cpp", "bfp4_round.py", "bfp4_round/compute.cpp",
                         "bfp4_round/reader.cpp", "bfp4_round/writer.cpp", "fast_correction.hpp", "safe_rescale.hpp"))}
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
        expected = r.get("source_sha256", {}).get(name)
        if not isinstance(expected, str) or SHA.fullmatch(expected) is None:
            e.pending.append("Missing/invalid recorded producer hash prevents assertion witness: " + expression)
            return False
        current = self.current_hash(name)
        source = name if current == expected else WITNESS_SOURCES + expected + ".source"
        if self.current_hash(source) is None:
            e.pending.append("No hash-matched current producer or historical source snapshot for witness: " + expression)
            return False
        data = (self.root / source).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            e.failures.append("Producer witness source SHA mismatch: " + source)
            return False
        # Parse only after matching the hash recorded at measurement time.
        # This snapshot is a source witness, never imported or executed, and
        # never replaces the original manifest or hides current-source drift.
        tree = ast.parse(data.decode("utf-8"), filename=source)
        expressions = [ast.unparse(n.test if isinstance(n, ast.Assert) else n)
                       for n in ast.walk(tree) if isinstance(n, node_type)]
        found = any(expression in x for x in expressions)
        e.require(found, "Pinned producer lacks assertion: " + expression)
        e.summary.setdefault("producer_witnesses", []).append(dict(
            producer=name, recorded_sha256=expected, inspected_source=source,
            basis="current_source" if source == name else "historical_sha256_snapshot",
            node_type=node_type.__name__, expression=expression, matched=found))
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

    def load_pair(self, names, family, callback, **kwargs):
        """One comparison entry, two independently hashed records; never silently use a partial pair."""
        e = Evidence(" + ".join(names), family)
        records, snapshots = [], {}
        for name in names:
            path = self.directory / name
            if not path.is_file():
                e.pending.append("Required paired evidence is absent: " + name)
                continue
            try:
                snapshots[name] = path.read_bytes()
                records.append(decode(snapshots[name].decode()))
            except (ValueError, OSError) as error:
                e.failures.append(name + ": " + str(error))
        e.summary["record_sha256"] = {n: hashlib.sha256(v).hexdigest() for n, v in snapshots.items()}
        if len(records) == len(names):
            try:
                callback(self, e, records, **kwargs)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, OSError, SyntaxError) as error:
                e.failures.append(type(error).__name__ + ": " + str(error))
        for name, data in snapshots.items():
            e.require((self.directory / name).is_file() and (self.directory / name).read_bytes() == data,
                      "Paired record changed during audit: " + name)
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
                         "LUT-macro pairs qualify only tested raw/macro equivalence, not a universal accuracy band",
                         "Codec comparisons are CPU-only representation experiments, not hardware qualification",
                         "No grid7 or unlisted historical experiments are promoted"])


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


def chain_layout(e, r, length, heads, cores, slots):
    for key, expected in dict(length=length, heads=heads, actual_cores=cores, q_chunk=256, k_chunk=512,
                              head_dim=128, input_slots=slots, q_jobs=heads * length // 256).items():
        e.equal(r.get(key), expected, key)
    chain, jobs = cores // heads, length // 256
    e.equal(r.get("chain_length"), chain, "Chain length")
    expected = []
    for head, rank in itertools.product(range(heads), range(chain)):
        expected.append(dict(head=head, rank=rank, jobs=jobs // chain + (rank < jobs % chain),
                             first_flat_q_job=head * jobs + rank * (jobs // chain) + min(rank, jobs % chain)))
    e.equal(r.get("assignments"), expected, "Per-head query assignment")
    e.equal(r.get("jobs_per_core"), [x["jobs"] for x in expected], "Query counts")
    cbs = r["cb_audit"]
    e.equal(len(cbs), len({x["cb"] for x in cbs}), "Unique CB indices")
    for cb in cbs:
        e.equal(cb["total_bytes"], cb["tiles"] * cb["page_bytes"], "CB allocation arithmetic")
    by_index = {c["cb"]: c for c in cbs}
    formats = {"bf16": (2048, "DataType.BFLOAT16"), "b8": (1088, "DataType.BFLOAT8_B"),
               "b4": (576, "DataType.BFLOAT4_B")}
    for index, fmt, count in ((0, "bf16", 64), (1, r["k_format"], 64 * slots), (2, r["v_format"], 64 * slots)):
        page_bytes, dtype = formats[fmt]
        e.equal(by_index[index]["tiles"], count, "Unchanged Q/K/V CB capacity")
        e.equal(by_index[index]["page_bytes"], page_bytes, "Q/K/V page format")
        e.equal(by_index[index]["dtype"], dtype, "Q/K/V CB data format")
    e.equal(r["cb_bytes_per_core"], sum(x["total_bytes"] for x in cbs), "Total CB allocation")
    e.equal(r["raw_l1_headroom_bytes"], 1536 * 1024 - r["cb_bytes_per_core"], "Raw L1 headroom")
    rows(e, r.get("sampled_query_rows"), length, r["sample_rows"], "Query sample")


def adaptive_fullchip(audit, e, r, length, fmt, search, distribution):
    audit.provenance(e, r)
    for key, expected in dict(mode="denom_bf16", destination="fast_bf16", denom_only=True, b4_search=search,
                              kv_formats=fmt, distribution=distribution, seed=1240, fp32_dst=False,
                              fidelity="LoFi", device_preprocessing=True, cpu_input_transform=False,
                              fix_correction=True, safe_rescale=True, executed_matmul_factor=1).items():
        e.equal(r.get(key), expected, key)
    chain_layout(e, r, length, 10, 110, 2)
    e.require("SDPA_STREAMING_NUMERATOR_COMPENSATION" not in r["defines"], "Denominator-only control changed")
    kfmt, vfmt = fmt.split("_")
    e.equal(r.get("k_format"), kfmt, "K format")
    e.equal(r.get("v_format"), vfmt, "V format")
    checks = r["preprocessing_checks"]
    e.require(type(r.get("check_preprocess")) is bool, "Missing preprocessing-check setting")
    if r["check_preprocess"]:
        e.equal([c["input"] for c in checks], ["Q", "K", "V"], "Exact preprocessing coverage")
        for c, expected_format in zip(checks, ("bf16", kfmt, vfmt)):
            e.equal(c["format"], expected_format, "Preprocessing format")
            e.equal(c["decoded_bit_mismatches"], 0, "Preprocessing decoded bit mismatches")
            e.equal(c["values"], 10 * length * 128, "Preprocessing value count")
            e.digest(c["actual_sha256"], "Decoded actual hash")
            e.digest(c["expected_sha256"], "Decoded oracle hash")
            e.equal(c["actual_sha256"], c["expected_sha256"], "Exact preprocessing hash equality")
            counts = c.get("selected_counts")
            if expected_format == "b4" and search != "native":
                e.require(isinstance(counts, dict) and set(counts) == {"-1", "0", "1"}, "Selection histogram keys")
                e.require(all(type(n) is int and n >= 0 for n in counts.values()), "Selection counts")
                e.equal(sum(counts.values()), c["values"] // 16, "Selection group coverage")
                if search == "minus":
                    e.equal(counts["1"], 0, "Minus search cannot select E+1")
                n = c.get("selection_differs_fp64_groups")
                e.require(type(n) is int and 0 <= n <= c["values"] // 16, "FP32/FP64 selection difference count")
    else:
        e.equal(checks, [], "Disabled preprocessing checks must not claim exact results")
        e.notes.append("This run did not repeat full preprocessing exactness checks")
    audit.assertion_witness(e, r, "adaptive_fullchip.py", "not torch.isfinite(actual).all()", ast.If)
    audit.assertion_witness(e, r, "adaptive_fullchip.py", "after_pins == pins")
    e.equal(r.get("trace_equal"), True, "Replay equality")
    e.digest(r.get("output_sha256"), "Output hash")
    metric(e, r.get("accuracy"), "Original-input accuracy")
    centered = r["centered_output_accuracy"]
    e.equal(centered.get("gain_alignment"), False, "No fitted gain alignment")
    e.number(centered.get("l2_pct"), "Centered L2")
    e.equal(centered.get("constant_v"), False, "Expected nonconstant stress inputs")
    flops = 4 * 10 * length**2 * 128
    e.equal(r.get("useful_flops"), flops, "Useful FLOPs")
    for stage in ("attention", "preprocessing", "combined"):
        ms = timing(e, r.get(stage), stage, r["iters"])
        if stage != "preprocessing":
            throughput(e, r, flops, stage, ms)
    e.summary.update(length=length, kv_formats=fmt, search=search, distribution=distribution,
                     l2_pct=r["accuracy"]["l2_pct"], pcc=r["accuracy"]["pcc"],
                     attention_tflops=r["attention_tflops"], combined_tflops=r["combined_tflops"],
                     preprocessing_exact_checked=r["check_preprocess"])
    e.notes.append("Recorded precision/performance comparison; no universal adaptive-improvement gate")


def lut_macro_record(audit, e, r, scope, raw):
    audit.provenance(e, r)
    resident = scope == "resident"
    driver = "exp_lut_macro_resident.py" if resident else "exp_lut_macro_streaming.py"
    for key, expected in dict(raw_lut=raw, lut_exp=True, native_exp=True, exp_lut_refinement=True,
                              exp_refiner="raw10" if raw else "macro8", q_chunk=256, k_chunk=512,
                              head_dim=128, probability_pack_width=4, denominator_matches_pv=True,
                              p_sfpu_prerounding=False, seed=1240, distribution="normal").items():
        e.equal(r.get(key), expected, key)
    e.equal(r.get("exp_refiner_issued_instructions_per_two_vectors"), 10 if raw else 8, "Refiner issue count")
    e.equal(r["defines"].get("SDPA_LOFI_LUT_MACRO"), None if raw else "1", "Macro compile setting")
    finite_expr = "bool(torch.isfinite(actual).all())" if resident else "torch.isfinite(actual).all()"
    audit.assertion_witness(e, r, driver, finite_expr)
    audit.assertion_witness(e, r, driver, "provenance[str(p.relative_to(ROOT))]")
    metric(e, r.get("accuracy"), "LUT accuracy")
    e.require(r["accuracy"]["l2_pct"] < r["max_l2"], "Recorded LUT L2 gate")
    e.digest(r.get("output_sha256"), "LUT output hash")
    if resident:
        e.equal(r.get("cores"), 1, "Resident core count")
        e.equal(r.get("q_repeats"), 16, "Resident Q repeats")
        e.equal(r.get("k_chunks"), 512, "Resident K chunks")
        e.equal(r.get("input_slots"), {"q": 2, "k": 1, "v": 1}, "Resident buffering")
        e.equal(r.get("preprocessing_in_timing"), False, "Resident preprocessing excluded")
        e.equal(r.get("recurring_input_dm"), False, "Resident recurring DM excluded")
        e.equal(r.get("preprocess_mismatches"), [0, 0, 0], "Resident preprocessing exactness")
        e.equal(len(r["prepared_input_sha256"]), 3, "Resident prepared hashes")
        for digest in r["prepared_input_sha256"]:
            e.digest(digest, "Prepared input SHA")
        flops = 4 * 256 * 512 * 128 * 16 * 512
        ms = timing(e, r, "resident", r["iters"])
        e.close(r.get("tflops_per_core"), flops / (ms * 1e9), "Resident TFLOPs")
        peak = 4096 * r["clock_mhz"] * 1e-6
        e.close(r.get("nominal_lofi_peak_tflops_per_core"), peak, "Nominal LoFi peak arithmetic")
        e.close(r.get("nominal_lofi_utilization_pct"), 100 * r["tflops_per_core"] / peak, "Nominal utilization arithmetic")
        e.equal(r.get("trace_equal"), True, "Resident replay equality")
    else:
        length = 1024 if scope == "smoke" else int(scope)
        heads, cores = (2, 4) if scope == "smoke" else (10, 110)
        chain_layout(e, r, length, heads, cores, 1)
        e.equal(r.get("fp32_dst"), True, "LUT FP32 destination")
        e.equal(r.get("kv_formats"), "b8_b8", "LUT K/V formats")
        flops = 4 * heads * length**2 * 128
        for stage in ("attention", "preprocessing", "combined"):
            ms = timing(e, r.get(stage), stage, r["iters"])
            if stage != "preprocessing":
                throughput(e, r, flops, stage, ms)
        e.equal(r.get("trace_equal"), True if r["iters"] else None, "LUT timed replay gate")
        e.equal(r.get("check_preprocess"), scope == "smoke", "Expected preprocessing check scope")
        if r["check_preprocess"]:
            e.equal(r["preprocessing_checks"], [dict(input=k, format=f, mismatch=0)
                    for k, f in zip(("Q", "K", "V"), ("bf16", "b8", "b8"))], "LUT preprocessing exactness")
        else:
            e.equal(r["preprocessing_checks"], [], "No fabricated preprocessing exactness")
        if not r["iters"]:
            e.notes.append("Smoke pair compares output bit hashes; no trace replay or timing was performed")
    e.equal(r.get("useful_flops"), flops, "LUT useful FLOPs")
    e.summary.update(l2_pct=r["accuracy"]["l2_pct"], output_sha256=r["output_sha256"])


def lut_macro(audit, e, records, scope):
    raw, macro = records
    components = []
    for mode, record in ((True, raw), (False, macro)):
        child = Evidence("raw" if mode else "macro", "lut_macro")
        lut_macro_record(audit, child, record, scope, mode)
        for name in ("failures", "pending", "notes"):
            getattr(e, name).extend(child.name + ": " + x for x in getattr(child, name))
        components.append(child.export())
    e.provenance = dict(status="WARN" if any(c["provenance"]["status"] == "WARN" for c in components) else "PASS",
                        component_records={c["file"]: c["provenance"] for c in components})
    # Same output bits are required, not merely similar L2 or PCC.
    e.equal(raw.get("output_sha256"), macro.get("output_sha256"), "Raw/macro output bit identity")
    common = ("seed", "distribution", "q_chunk", "k_chunk", "head_dim", "input_slots", "cb_bytes_per_core",
              "exp_coefficients", "useful_flops", "source_sha256", "accuracy")
    common += ("q_repeats", "k_chunks", "prepared_input_sha256", "clock_mhz") if scope == "resident" else (
        "length", "heads", "actual_cores", "assignments", "sampled_query_rows", "kv_formats")
    for key in common:
        e.equal(raw.get(key), macro.get(key), "Paired setting " + key)
    raw_defines = dict(raw["defines"])
    macro_defines = dict(macro["defines"])
    macro_defines.pop("SDPA_LOFI_LUT_MACRO", None)
    e.equal(raw_defines, macro_defines, "Only macro compile setting changes")
    times = [r["median_ms"] if scope == "resident" else r["attention"]["median_ms"] for r in records]
    e.summary.update(scope=scope, output_bit_identical=raw["output_sha256"] == macro["output_sha256"],
                     raw_attention_ms=times[0], macro_attention_ms=times[1],
                     attention_speedup=times[0] / times[1] if all(times) else None,
                     l2_pct=macro["accuracy"]["l2_pct"])
    e.notes.append("Raw/macro LUT equivalence only; does not assert equivalence to native unrefined exp")


def codec_cpu(audit, e, records, suite):
    allowed = {"provenance", "inputs", "attention", "complete"}
    e.require(all(r.get("kind") in allowed for r in records), "Unexpected CPU codec record kind")
    p, cases = envelope(audit, e, [r for r in records if r.get("kind") != "inputs"], "attention")
    args = p["args"]
    vaxis = suite == "vaxis"
    distributions = ["normal", "outliers", "channel_v", "common_v"] if vaxis else ["normal", "outliers"]
    codecs = ["tt_rne", "tt_adaptive_pm", "nvfp4_e4m3"] if vaxis else [
        "tt_native", "tt_rne", "tt_adaptive_pm", "uniform7_continuous", "e2m1_continuous", "e2m1_power2_g16", "nvfp4_e4m3"]
    qmodes, axes, qcount = (["bf16"], ["D", "N"], 64) if vaxis else (["bf16", "q7"], ["D"], 128)
    for key, expected in dict(lengths=[4096], seeds=[1240], q_rows=qcount, distributions=distributions,
                              codecs=codecs, q_modes=qmodes, v_axes=axes, nv_v_axis_n=False).items():
        e.equal(args.get(key), expected, "CPU suite " + key)
    e.equal(p.get("execution"), "CPU-only, no TTNN import or device calls", "CPU execution scope")
    e.equal(p.get("output_rounding"), "reported both FP64 and BF16", "Output precision reporting")
    e.require(type(args.get("threads")) is int and 0 < args["threads"] <= 4, "CPU thread budget")
    for test in ("codebook_roundtrip", "all_midpoint_ties_even", "finite_both_axes", "uniform_integer_exact"):
        e.equal(p.get("self_tests", {}).get(test), True, "Codec self-test " + test)
    expected = set()
    for dist, qm in itertools.product(distributions, qmodes):
        expected.add((4096, qcount, 1240, dist, qm, "unquantized", "D", "control"))
        expected.update((4096, qcount, 1240, dist, qm, codec, axis, scope)
                        for codec, axis, scope in itertools.product(codecs, axes, ("K_only", "V_only", "KV")))
    fields = ("length", "q_rows", "seed", "distribution", "q_mode", "codec", "v_axis", "scope")
    keys = [tuple(r.get(k) for k in fields) for r in cases]
    if e.pending:
        e.equal(len(keys), len(set(keys)), "Duplicate CPU cases in unfinished stream")
        e.require(set(keys) <= expected, "Unexpected CPU cases in unfinished stream")
        e.summary["cases"] = len(keys)
    else:
        coverage(e, keys, expected)
    inputs = [r for r in records if r.get("kind") == "inputs"]
    input_keys = [(r["length"], r["q_rows"], r["seed"], r["distribution"]) for r in inputs]
    if not e.pending:
        e.equal(set(input_keys), {(4096, qcount, 1240, d) for d in distributions}, "CPU input coverage")
    e.equal(len(input_keys), len(set(input_keys)), "CPU input uniqueness")
    for r in inputs:
        e.equal(set(r.get("sha256", {})), {"Q", "K", "V"}, "CPU input hashes")
        for digest in r["sha256"].values():
            e.digest(digest, "CPU original input SHA")
    for r in cases:
        e.require(not any("tflop" in k or k in ("trace_equal", "device_time_ms") for k in r),
                  "CPU representation record must not claim hardware throughput/replay")
        for name in ("fp64_output", "bf16_output"):
            metric(e, r.get(name), "CPU " + name)
        if r["scope"] == "control":
            if r["q_mode"] == "bf16":
                e.require(r["fp64_output"]["l2_pct"] < 1e-8, "Unquantized BF16-input FP64 control agreement")
            continue
        for tensor, axis in (("K", "D"), ("V", r["v_axis"])):
            rep = r[tensor]
            e.equal(rep.get("codec"), r["codec"], "Codec representation setting")
            e.equal(rep.get("group_axis"), axis, "Codec grouping axis")
            e.equal(rep.get("group_size"), 16, "Codec group size")
            e.number(rep.get("reconstruction_l2_pct"), "Representation L2")
            e.require(finite(rep.get("relative_mean_error")), "Representation mean error")
            e.require(finite(rep.get("zero_fraction")) and 0 <= rep["zero_fraction"] <= 1, "Representation zero fraction")
        if r["distribution"] == "channel_v":
            metric(e, r.get("quiet_channel_fp64_output"), "Quiet-channel accuracy")
            e.number(r.get("quiet_V_reconstruction_l2_pct"), "Quiet V reconstruction L2")
        if r["distribution"] == "common_v":
            for name in ("centered_fp64_output", "centered_bf16_output"):
                metric(e, r.get(name), "CPU " + name)
    e.summary.update(execution="CPU_ONLY", suite=suite, input_sets=len(inputs),
                     representation_only=True, device_qualification=False, hardware_performance=False)
    e.notes.append("FP64 attention/state with unquantized P isolates representation; not Sage3 or hardware emulation")


def captured_interface(audit, e, records, historical=False):
    p, cases = envelope(audit, e, records, "captured_evaluation")
    capture, args = p["capture"], p["args"]
    variants = ["main", "fast", "balanced", "accurate", "lofi_fp32_b8", "lofi_fp32_b4"]
    for key, expected in dict(schema="sdpa-captured-inputs-v1", shape=[1, 2, 1024, 128],
                              dtype="bfloat16", layout="BHND-contiguous", useful_attention_flops=1073741824).items():
        e.equal(capture.get(key), expected, "Capture " + key)
    meta = capture["metadata"]
    for key, expected in dict(causal=False, mask=None, scale=1 / math.sqrt(128)).items():
        e.equal(meta.get(key), expected, "Capture semantics " + key)
    e.equal(meta["provenance"].get("source_kind"), "synthetic", "Synthetic interface evidence only")
    e.equal(meta["provenance"].get("capture_stage"), "interface-smoke-only", "Capture stage")
    e.equal(meta["provenance"].get("seed"), 1240, "Synthetic seed")
    canonical = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    e.equal(capture.get("metadata_sha256"), hashlib.sha256(canonical).hexdigest(), "Canonical metadata hash")
    e.digest(capture.get("artifact_sha256"), "Serialized artifact hash")
    e.require(type(capture.get("artifact_bytes")) is int and capture["artifact_bytes"] > 0, "Artifact byte count")
    e.equal(set(capture["input_sha256"]), {"q", "k", "v"}, "Original capture hash coverage")
    for digest in capture["input_sha256"].values():
        e.digest(digest, "Original capture tensor hash")
    e.equal(capture.get("loading"), "torch.load(weights_only=True,map_location=cpu); no unsafe fallback", "Loading contract")
    e.equal(args.get("variants"), variants, "Requested six variants")
    e.equal(args.get("sample_rows"), 1024, "All-query reference request")
    e.equal(args.get("iters"), 0, "Interface smoke, no performance sampling")
    e.equal(args.get("check_preprocess"), True, "Requested exact preprocessing")
    observed = [r["variant"] for r in cases]
    if e.pending:
        e.equal(len(observed), len(set(observed)), "No duplicate variants in partial stream")
        e.require(set(observed) <= set(variants), "Unexpected partial captured variant")
        e.summary["cases"] = len(observed)
    else:
        coverage(e, observed, set(variants))
        footer = records[-1]
        e.equal(footer.get("original_inputs_unchanged"), True, "Final immutable inputs")
        e.equal(footer.get("evaluated_variants"), variants, "Completed variants")
    cb_bytes = dict(main=1169408, fast=1333248, balanced=1335296, accurate=1335296,
                    lofi_fp32_b8=1212416, lofi_fp32_b4=1146880)
    accuracy = []
    for r in cases:
        name, config, kernel = r["variant"], r["config"], r["kernel"]
        lofi, fp32 = name.startswith("lofi_"), name not in ("main", "fast")
        for key in ("all_output_finite", "trace_equal", "immutable_inputs_verified"):
            e.equal(r.get(key), True, name + ": " + key)
        e.equal(r.get("original_input_sha256"), capture["input_sha256"], name + ": original input identity")
        e.equal(r.get("sampled_query_rows"), list(range(1024)), name + ": every Q row referenced")
        e.digest(r.get("output_sha256"), name + ": output bit hash")
        e.equal(r.get("timings"), {}, name + ": no smoke timing")
        e.require(not any("tflop" in k for k in r), name + ": no fabricated smoke throughput")
        e.equal(r.get("useful_attention_flops"), 1073741824, name + ": useful FLOPs")
        e.equal(r.get("preprocessing_exact_checked"), lofi, name + ": preprocessing applies only to LoFi")
        e.equal(r.get("fast_private_correction_reset"), name == "fast", name + ": FAST private correction fix")
        for key, expected in dict(variant=name, q_chunk=256, length=1024, heads=2, cores=4, q_prescale=1.0,
                                  center_k=False, b8_rne=False, bfp8_pack_precise=False, check_preprocess=True,
                                  fix_correction=name == "fast", exp_degree=3, native_exp=lofi,
                                  reader_chain=True, read_barrier_tiles=2).items():
            e.equal(config.get(key), expected, name + ": config " + key)
        for key, expected in dict(actual_cores=4, q_jobs=8, jobs_per_core=[2, 2, 2, 2],
                                  input_slots=1 if fp32 else 2, fp32_dst=fp32, device_preprocessing=lofi,
                                  cb_bytes_per_core=cb_bytes[name]).items():
            e.equal(kernel.get(key), expected, name + ": kernel " + key)
        fidelity = "MathFidelity.LoFi" if lofi else "MathFidelity.HiFi4" if name == "accurate" else "MathFidelity.HiFi2"
        e.equal(kernel.get("fidelity"), fidelity, name + ": fidelity")
        defines = kernel["defines"]
        e.equal(defines.get("SDPA_QK4"), "1" if name == "balanced" else None, name + ": mixed fidelity")
        e.equal(defines.get("SDPA_LOFI_NATIVE_EXP"), "1" if lofi else None, name + ": native exp")
        if name == "fast":
            for key in ("SDPA_LOFI_FIX_CORRECTION", "SDPA_STREAMING_NUMERATOR_COMPENSATION"):
                e.equal(defines.get(key), "1", name + ": " + key)
        if name == "accurate":
            for key, expected in dict(SDPA_DIAG_EXP_MODE="4", SDPA_FP32_L1_SUB="1", SDPA_DENOM_PHASES="2").items():
                e.equal(defines.get(key), expected, name + ": " + key)
        m = r["metrics"]
        metric(e, m, name + ": captured accuracy")
        e.equal(m.get("relative_l2_undefined"), False, name + ": normal reference norm")
        e.equal(m.get("residual_relative_undefined"), False, name + ": normal centered reference norm")
        e.number(m.get("residual_l2_pct"), name + ": residual L2")
        e.number(m.get("bf16_rounding_floor_l2_pct"), name + ": output rounding floor")
        accuracy.append(dict(variant=name, l2_pct=m["l2_pct"], pcc=m["pcc"]))
    # The fixture source is pinned inside capture metadata rather than the driver's source manifest.
    fixture = V2 + "capture_smoke_fixture.py"
    expected = meta["provenance"].get("fixture_generator_sha256")
    e.digest(expected, "Fixture generator metadata pin")
    current = audit.current_hash(fixture)
    if current is None:
        e.provenance["unavailable_current_sources"].append(fixture)
        e.provenance["status"] = "WARN"
    elif current != expected:
        e.provenance["current_source_drift"].append(dict(source=fixture, recorded_sha256=expected, current_sha256=current))
        e.provenance["status"] = "WARN"
    e.summary.update(accuracy=accuracy, source_kind="synthetic", model_activation_qualification=False,
                     all_query_rows=1024, performance_measured=False,
                     evidence_role="historical" if historical else "current-source qualification",
                     fixture_pin_location="capture.metadata.provenance.fixture_generator_sha256")
    e.notes.append("Record-only audit: no unsafe capture reload or raw tensor recomputation; no actual model data supplied")


def v_transpose(audit, e, records, length, seed, axis, vfmt="b4", recipe=None):
    p, cases = envelope(audit, e, records, "result")
    smoke = length == 1024
    heads, cores = (2, 4) if smoke else (10, 110)
    distributions = ["normal", "channel_v"]
    if smoke:
        distributions += ["constant_v"]
    elif vfmt == "b4" and length == 32768 and seed == 1240:
        distributions += ["outliers", "common_v"]
    if recipe is not None:
        distributions = ["normal", "constant_v", "k_outliers_channel_v"] if smoke else [
            "normal", "outliers", "k_outliers_channel_v", "common_k", "common_q"]
    iterations = 0 if smoke else 3 if recipe is not None else 5 if vfmt == "b8" else 7
    kfmt = "b4" if recipe is not None else "b8"
    driver = "combined_recipe_fullchip.py" if recipe is not None else "Vtransposed_fullchip.py"
    transposed = axis == "N"
    expected = dict(destination="fast_bf16", denom_only=False, v_transposed=transposed,
                    grid7_exp=False, kv_formats=kfmt + "_" + vfmt, length=length, heads=heads, cores=cores,
                    seed=seed, distributions=distributions, sample_rows=128, check_preprocess=True,
                    read_barrier_tiles=2, max_l2=None, warmup=5 if smoke else 2 if recipe is not None else 3,
                    iters=iterations, trace_repeats=1)
    if recipe is not None:
        expected.update(h16=recipe[0], adaptive_v=recipe[1])
    for key, value in expected.items():
        e.equal(p["args"].get(key), value, "V-axis suite " + key)
    coverage(e, [r.get("distribution") for r in cases], set(distributions))
    witnesses = [audit.assertion_witness(e, p, driver, expression) for expression in (
        "[tensor_hash(x) for x in inputs] == original_hashes",
        "all((bf16_bitwise_equal(ttnn.to_torch(t).bfloat16(), x) for t, x in zip(originals, inputs)))")]
    for r in cases:
        for key, value in expected.items():
            e.equal(r.get(key), value, "V-axis result " + key)
        for key in ("all_output_finite", "trace_equal", "sources_unchanged"):
            e.equal(r.get(key), True, "V-axis " + key)
        # Existing records lack this boolean; require it if later explicitly provided.
        if "immutable_inputs_verified" in r:
            e.equal(r["immutable_inputs_verified"], True, "V-axis input immutability")
        e.equal(r.get("accuracy_scope"), "Original BF16 Q/K/V FP64 reference, all KV and explicit Q rows; no gain/reference shifting", "V-axis reference scope")
        timing_scope = "Combined includes both real H16 matmuls and real V transpose when enabled, plus Q/K/V quantization and attention; stage timings are a disjoint decomposition of preprocessing, never added twice. Uploads excluded; useful FLOPs exclude preprocessing" if recipe is not None else "Combined includes real transpose (when enabled) plus Q/K/V quantization and attention; stage timings are a disjoint decomposition of preprocessing, never added twice. Uploads excluded; useful FLOPs exclude preprocessing"
        e.equal(r.get("timing_scope"), timing_scope, "V-axis timing scope")
        rows(e, r.get("sampled_query_rows"), length, 1024 if smoke else 128, "V-axis sampled Q")
        e.equal(r.get("all_query_rows_referenced"), smoke, "All-query smoke versus sampled long reference")
        e.equal(len(r["original_input_sha256"]), 3, "Original Q/K/V hash count")
        for digest in r["original_input_sha256"]:
            e.digest(digest, "V-axis original input")
        e.digest(r.get("output_sha256"), "V-axis output")
        kernel = r["kernel"]
        for key, value in dict(actual_cores=cores, chain_length=cores // heads, input_slots=2, fidelity="LoFi", fp32_dst=False,
                               q_chunk=256, k_chunk=512, head_dim=128, k_format=kfmt, v_format=vfmt,
                               device_preprocessing=True, grid7_exp=False, fix_correction=True, safe_rescale=True,
                               pv_transpose_in1=transposed, executed_matmul_factor=1, output_dtype="BF16").items():
            e.equal(kernel.get(key), value, "V-axis kernel " + key)
        e.equal(kernel.get("v_group_axis"), "N: 16 consecutive tokens within one channel" if transposed else
                "D: 16 consecutive channels within one token", "V shared-group axis")
        e.equal(kernel.get("v_physical_shape"), [1, heads, 128, length] if transposed else [1, heads, length, 128], "V physical layout")
        e.equal(kernel.get("v_cb_tile_order"), "N-major k_tile*4+d_tile in both controls", "V CB tile order")
        defines = kernel["defines"]
        for key in ("SDPA_STREAMING_ACCURACY", "SDPA_STREAMING_NUMERATOR_COMPENSATION", "SDPA_LOFI_FIX_CORRECTION", "SDPA_LOFI_SAFE_RESCALE"):
            e.equal(defines.get(key), "1", "V-axis compensated kernel " + key)
        e.equal(defines.get("SDPA_V_TRANSPOSED"), "1" if transposed else None, "PV transpose compile flag")
        e.equal(kernel.get("q_jobs"), heads * length // 256, "All-query job count")
        jobs = kernel["jobs_per_core"]
        e.equal(len(jobs), cores, "Per-core assignment count")
        e.require(all(type(x) is int and x > 0 for x in jobs), "Invalid per-core jobs")
        e.equal(sum(jobs), kernel["q_jobs"], "Complete query assignment")
        cb = {x["cb"]: x for x in kernel["cb_audit"]}
        for index, tiles, page in ((0, 64, 2048), (1, 128, 1088 if kfmt == "b8" else 576), (2, 128, 1088 if vfmt == "b8" else 576)):
            e.equal(cb[index]["tiles"], tiles, "Unchanged input CB capacity")
            e.equal(cb[index]["page_bytes"], page, "Input CB format bytes")
        for item in cb.values():
            e.equal(item["total_bytes"], item["tiles"] * item["page_bytes"], "CB footprint arithmetic")
        e.equal(sum(x["total_bytes"] for x in cb.values()), kernel["cb_bytes_per_core"], "Total CB footprint")
        checks = kernel["preprocessing_checks"]
        names = [x["input"] for x in checks]
        e.equal(names, (["V transpose"] if transposed else []) + ["Q", "K", "V"], "Exact preprocessing coverage")
        for check in checks:
            e.equal(check.get("mismatch"), 0, "Preprocessing exactness " + check["input"])
            if check["input"] == "V transpose":
                e.equal(check.get("comparison"), "BF16 uint16 storage bits, including signed zero", "Transpose exact-bit oracle")
            else:
                e.equal(check.get("format"), {"Q": "bf16", "K": kfmt, "V": vfmt}[check["input"]], "Preprocessing format")
                e.equal(check.get("comparison"), "Exact decoded numeric values; signed zero treated as equal", "Quantization oracle scope")
        vcheck = checks[-1]
        metric(e, vcheck.get("packed_v_vs_original_bf16"), "Packed V reconstruction")
        e.equal(vcheck.get("packed_v_metric_scope"), "Decoded storage only, before any further LoFi operand truncation", "V representation scope")
        if transposed:
            t = kernel["v_transpose"]
            e.equal(t.get("input_shape"), [1, heads, length, 128], "Transpose input shape")
            e.equal(t.get("output_shape"), [1, heads, 128, length], "Transpose output shape")
            e.equal(t.get("bytes_read_and_written"), 4 * heads * length * 128, "Transpose BF16 bytes")
        else:
            e.equal(kernel.get("v_transpose"), None, "No transpose in D-axis control")
        for name in ("accuracy", "bf16_output_rounding_floor"):
            metric(e, r.get(name), "V-axis " + name)
        if r["distribution"] == "constant_v":
            e.equal(r["centered_output_accuracy"].get("l2_pct"), None, "Constant-V residual undefined")
            e.equal(r["centered_output_accuracy"].get("constant_v"), True, "Constant-V classification")
        else:
            metric(e, r.get("centered_output_accuracy"), "V-axis centered accuracy")
        e.equal(r["centered_output_accuracy"].get("gain_alignment"), False, "No residual gain fit")
        e.equal(r["centered_output_accuracy"].get("scope"), "All heads and explicitly sampled Q rows; original BF16 V, all KV rows", "Residual reference scope")
        if r["distribution"] in ("channel_v", "k_outliers_channel_v"):
            metric(e, r.get("quiet_value_channel_accuracy"), "Quiet-channel accuracy")
        else:
            e.equal(r.get("quiet_value_channel_accuracy"), None, "Quiet-channel metric not applicable")
        flops = 4 * heads * length**2 * 128
        e.equal(r.get("useful_flops"), flops, "Useful FLOP problem size")
        for name in ("attention", "preprocessing", "combined"):
            ms = timing(e, r.get(name), "V-axis " + name, iterations)
            if name != "preprocessing":
                throughput(e, r, flops, name, ms)
        stages = r["preprocessing_stages"]
        expected_stages = {"q_quantization", "k_quantization", "v_quantization"} if recipe is not None else {"quantization"}
        if transposed:
            expected_stages.add("v_transpose")
        if recipe is not None and recipe[0]:
            expected_stages |= {"q_rotation", "k_rotation"}
        e.equal(set(stages), expected_stages, "Real preprocessing stage coverage")
        for name, stage in stages.items():
            timing(e, stage, "V-axis stage " + name, iterations)
        if recipe is not None:
            h16, adaptive = recipe
            for key, value in dict(h16=h16, adaptive_v=adaptive, adaptive_k=False,
                                   score_scale=1 / math.sqrt(128) / (16 if h16 else 1)).items():
                e.equal(kernel.get(key), value, "Combined recipe " + key)
            e.equal(len(kernel["qk_rotation_metadata"]), 2 if h16 else 0, "Q/K rotation metadata")
            e.equal(len(kernel["qk_rotation_checks"]), 2 if h16 else 0, "Measured BF16 transform checks")
            for rotation in kernel["qk_rotation_metadata"]:
                for key, value in dict(block_size=16, score_scale_divisor=16, sign_seed=20260915,
                                       centering=False, device_input_transform=True).items():
                    e.equal(rotation.get(key), value, "Actual H16 " + key)
                e.digest(rotation.get("matrix_sha256"), "H16 matrix")
            if h16:
                e.equal(kernel["qk_rotation_metadata"][0]["matrix_sha256"], kernel["qk_rotation_metadata"][1]["matrix_sha256"], "Shared Q/K transform")
            for check in checks:
                if check["input"] != "V transpose":
                    transformed = h16 if check["input"] in ("Q", "K") else transposed
                    e.equal(check.get("oracle_source"), "Actual device BF16 transformed input" if transformed else "Original BF16 input", "Actual-spill quantizer oracle")
            search = checks[-1].get("adaptive_statistics")
            if adaptive == "pm":
                e.equal(search.get("induced_exponent_mismatches"), 0, "Adaptive selected exponent")
                e.equal(search.get("native_grid_roundtrip_mismatches"), 0, "Adaptive native grid")
                e.equal(sum(search["selected_counts"].values()), search["groups"], "Adaptive coverage")
            else:
                e.equal(search, None, "No adaptive V in control")
    e.summary.update(length=length, seed=seed, axis=axis, preprocessing_exact_checked=True,
                     input_immutability_producer_assertions=all(witnesses), all_query_accuracy=smoke,
                     accuracy=[dict(distribution=r["distribution"], l2_pct=r["accuracy"]["l2_pct"],
                                    combined_tflops=r["combined_tflops"]) for r in cases])
    e.notes.append("Input immutability is a pinned producer-assertion witness, not an explicit recorded boolean or a tensor recheck")
    e.notes.append("Stage and combined timings are independent medians; no additivity or numerical-improvement gate is imposed")


v8_axis = v_transpose
combined_recipe = v_transpose


def mean_error(audit, e, r, length, fmt, mode, distribution):
    audit.provenance(e, r)
    smoke = length == 1024
    heads, cores, iterations = (2, 4, 0) if smoke else (10, 110, 3)
    for key, value in dict(destination="fast_bf16", denom_only=False, kv_formats=fmt,
                           correction_mode=mode, mean_mode="bf16_fpu", seed=1240, distribution=distribution,
                           check_preprocess=True, fidelity="LoFi", fp32_dst=False, fix_correction=True,
                           safe_rescale=True, device_preprocessing=True, trace_equal=True,
                           all_output_finite=True, sources_unchanged=True, cpu_inputs_unchanged=True,
                           device_inputs_unchanged=True, iters=iterations,
                           sample_rows=1024 if smoke else 128, executed_matmul_factor=1).items():
        e.equal(r.get(key), value, "Mean-error " + key)
    chain_layout(e, r, length, heads, cores, 2)
    e.equal(r.get("accuracy_scope"), "Original BF16 Q/K/V FP64 reference; sampled Q rows, all KV and heads", "Mean-error reference")
    e.equal(r["defines"].get("SDPA_STREAMING_NUMERATOR_COMPENSATION"), "1", "Full numerator compensation")
    checks = r["preprocessing_checks"]
    e.equal([x["input"] for x in checks], ["Q", "K", "V"], "Mean-error exact prep coverage")
    for x, fmt_expected in zip(checks, ["bf16", *fmt.split("_")]):
        e.equal(x.get("mismatch"), 0, "Exact original-input preprocessing")
        e.equal(x.get("format"), fmt_expected, "Preprocessing format")
    e.equal(r["value_centering"].get("mode"), "none", "No V precentering")
    e.digest(r.get("output_sha256"), "Mean-error output hash")
    hashes = r["original_input_sha256"]
    e.equal(len(hashes), 3, "Original input hash count")
    for h in hashes:
        e.digest(h, "Original input")
    qualification = r["correctness_trace_qualification"]
    e.equal(r.get("correctness_trace_replays"), 2, "Mandatory correctness replay count")
    e.equal(qualification.get("explicit_combined_trace_replays"), 2, "Explicit replay count")
    e.equal(qualification.get("replay_output_sha256"), [r["output_sha256"]] * 2, "Trace output bit identity")
    for x in [qualification, qualification["before"], qualification["after"], r["final_input_immutability"]]:
        for key in ("cpu_inputs_unchanged", "device_inputs_unchanged"):
            e.equal(x.get(key), True, "Mean-error " + key)
        if "cpu_input_sha256" in x:
            e.equal(x["cpu_input_sha256"], hashes, "CPU input hashes preserved")
            e.equal(x.get("device_input_sha256"), hashes, "Device input hashes preserved")
    e.equal(qualification.get("output_bitwise_equal"), True, "Correctness output bits")
    if mode == "mean_error":
        correction = r["value_mean_error_correction"]
        e.equal(correction.get("precenter_v"), False, "Quantize original V")
        e.equal(correction.get("formula"), "mean(original V) - mean(actual LoFi-consumed quantized ORIGINAL V)", "Delta formula")
        e.equal(correction["check"].get("bias_mismatch"), 0, "Actual device BF16 delta oracle")
        e.equal(r["epilogue_check"].get("mismatch"), 0, "BF16 epilogue oracle")
        effective = correction["check"].get("effective_value_check")
        if fmt.endswith("b8"):
            e.require(isinstance(effective, dict), "V8 effective operand check missing")
            e.equal(effective.get("mismatch"), 0, "Actual LoFi V8 operand oracle")
    else:
        e.equal(r.get("epilogue_check"), None, "No correction epilogue oracle")
    metric(e, r.get("accuracy"), "Mean-error accuracy")
    residual = r["centered_output_accuracy"]
    e.equal(residual.get("gain_alignment"), False, "No residual gain fitting")
    if distribution in ("constant_v", "uniform"):
        e.equal(residual.get("l2_pct"), None, "Analytically zero residual undefined")
        e.require(bool(residual.get("relative_error_undefined_reason")), "Undefined residual explanation")
    else:
        e.number(residual.get("l2_pct"), "Residual L2")
    flops = 4 * heads * length**2 * 128
    e.equal(r.get("useful_flops"), flops, "Useful attention work")
    for stage in ("attention", "preprocessing", "combined", "epilogue", "attention_with_epilogue"):
        ms = timing(e, r.get(stage), "Mean-error " + stage, iterations, zero=stage == "epilogue" and mode == "none")
        if stage in ("attention", "combined"):
            throughput(e, r, flops, stage, ms)
    e.summary.update(length=length, mode=mode, kv_formats=fmt, distribution=distribution,
                     preprocessing_exact_checked=True, l2_pct=r["accuracy"]["l2_pct"],
                     combined_tflops=r["combined_tflops"], all_query_accuracy=smoke)


def hifi2_late(audit, e, r, length, route, option):
    audit.provenance(e, r)
    smoke, bf16 = length == 1024, route == "bf16"
    storage = route == "storage"
    driver = "fullchip.py" if storage else "hifi2_bf16_lut_fullchip.py" if bf16 else "hifi2_lut_fullchip.py" if route == "lut" else "hifi2_native_fullchip.py"
    variant = option if storage else "hi2_fp32_bf16" if bf16 else "hi2_fp32_b8"
    heads, cores, iterations = (2, 4, 0) if smoke else (10, 110, 5 if bf16 else 7)
    exact = route != "native" or smoke
    for key, value in dict(length=length, variant=variant, heads=heads, actual_cores=cores, cores=cores,
                           q_chunk=256, seed=1240, distribution="normal", iters=iterations,
                           sample_rows=1024 if smoke else 128, check_preprocess=exact,
                           fp32_dst=True, input_slots=1, native_exp=True, device_preprocessing=True,
                           center_k=False, q_prescale=1.0, reader_chain=True,
                           trace_equal=True, finite=True, sources_unchanged=True).items():
        e.equal(r.get(key), value, "FP32 late " + key)
    audit.assertion_witness(e, r, driver, "args.length // 512", ast.Assign)
    if exact and not bf16:
        audit.assertion_witness(e, r, driver, "mismatch == 0")
        audit.assertion_witness(e, r, driver, "args.check_preprocess", ast.If)
        e.notes.append("Exact prep uses enabled flag plus hash-matched build assertion; per-input mismatch counts were not recorded")
    elif not exact:
        e.notes.append("This native long-context run disabled exact preprocessing checks; associated 1024 controls are separately required")
    if bf16:
        checks = r["preprocessing_checks"]
        e.equal([x["input"] for x in checks], ["Q", "K", "V"], "BF16 prep coverage")
        for x, bits in zip(checks, (7, 8, 8)):
            e.equal(x.get("mismatch"), 0, "BF16 prep oracle")
            e.equal(x.get("bits"), bits, "BF16 prep width")
            e.equal(x.get("storage"), "BF16", "BF16 input storage")
            if x["input"] != "Q":
                e.equal(x.get("identity_bits_preserved"), True, "Identity K/V preparation")
        for key in ("cpu_inputs_unchanged", "device_inputs_unchanged"):
            e.equal(r.get(key), True, "BF16 original input immutability")
        e.equal(r.get("correctness_trace_replays"), 2, "BF16 correctness replays")
        e.equal(r.get("replay_output_sha256"), [r["output_sha256"]] * 2, "BF16 bitwise replay")
    expected_rne = route == "lut" or (route == "native" and option == "rne")
    e.equal(r.get("b8_rne"), expected_rne, "B8 RNE route")
    e.equal(r.get("bfp8_pack_precise"), route == "native" and option == "single", "Single-RNA route")
    lut = route in ("lut", "bf16") and option == "on"
    if route in ("lut", "bf16"):
        e.equal(r.get("lut_exp"), lut, "LUT option")
    e.equal(r["defines"].get("SDPA_LOFI_LUT_EXP"), "1" if lut else None, "LUT compile flag")
    e.equal(r["defines"].get("SDPA_LOFI_LUT_MACRO"), "1" if lut else None, "Macro compile flag")
    e.equal(r.get("fidelity"), "MathFidelity.LoFi" if storage else "MathFidelity.HiFi2", "Compute fidelity")
    e.equal(r.get("input_storage"), "BF16 Q/K/V" if bf16 or variant == "lofi_fp32" else "BF16 Q; BFP8 K/V", "Input storage")
    e.equal(r.get("accuracy_scope"), "All heads and KV; only explicit sampled Q rows; all output values checked finite", "Sampled original reference scope")
    rows(e, r.get("sampled_query_rows"), length, 1024 if smoke else 128, "FP32 late query rows")
    metric(e, r.get("accuracy"), "FP32 late accuracy")
    e.digest(r.get("output_sha256"), "FP32 late output hash")
    e.equal(r.get("q_jobs"), heads * length // 256, "All query jobs")
    e.equal(len(r["jobs_per_core"]), cores, "Core job count")
    e.equal(sum(r["jobs_per_core"]), r["q_jobs"], "Query job coverage")
    flops = 4 * heads * length**2 * 128
    e.equal(r.get("useful_flops"), flops, "FP32 late useful FLOPs")
    for stage in ("attention", "preprocessing", "combined"):
        ms = timing(e, r.get(stage), "FP32 late " + stage, iterations)
        if stage != "preprocessing":
            throughput(e, r, flops, stage, ms)
    e.summary.update(length=length, route=route, option=option, preprocessing_exact_checked=exact,
                     l2_pct=r["accuracy"]["l2_pct"], combined_tflops=r["combined_tflops"])


native_storage = hifi2_late


def paired_vaxis(audit, e, records, length):
    kinds = {"provenance", "initial_memory", "qualified", "trace_qualification", "round_order",
             "round_samples", "summary", "trace_cleanup", "complete"}
    e.require(all(x.get("kind") in kinds for x in records), "Unexpected paired-timing record/failure")
    p, qualified = envelope(audit, e, [x for x in records if x.get("kind") in ("provenance", "qualified", "complete")], "qualified")
    smoke = length == 1024
    heads, cores, iterations, warmup = (2, 4, 2, 0) if smoke else (10, 110, 7, 3)
    for key, value in dict(length=length, heads=heads, cores=cores, barriers=[2, 8, 16], seed=1240,
                           distribution="normal", sample_rows=128, iters=iterations, warmup=warmup, trace_repeats=1).items():
        e.equal(p["args"].get(key), value, "Paired timing " + key)
    e.equal(p.get("timing_scope"), "Combined real device Q/K/V quantization, optional V transpose and attention; host uploads/oracles excluded; host wall-clock blocking trace latency per invocation", "Paired timing scope")
    audit.assertion_witness(e, p, "paired_vaxis_timing.py", "BASE.REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])", ast.Assign)
    rows(e, p.get("sampled_query_rows"), length, length if smoke else 128, "Paired reference rows")
    e.equal(len(p["input_sha256"]), 3, "Paired original input hash count")
    for h in p["input_sha256"]:
        e.digest(h, "Paired original inputs")
    ids = [f"{axis}-b{barrier}" for barrier in (2, 8, 16) for axis in ("D", "N")]
    coverage(e, [x.get("id") for x in qualified], set(ids))
    by_id, axis_hashes = {}, {}
    for r in qualified:
        name, axis, barrier = r["id"], r["axis"], r["barrier"]
        e.equal(name, f"{axis}-b{barrier}", "Candidate ID settings")
        for key in ("all_output_finite", "exact_preprocessing", "original_inputs_immutable", "barrier_output_bits_equal"):
            e.equal(r.get(key), True, "Paired qualified " + key)
        e.digest(r.get("output_sha256"), "Paired output")
        metric(e, r.get("accuracy"), "Paired original-reference accuracy")
        kernel = r["kernel"]
        for key, value in dict(q_chunk=256, k_chunk=512, head_dim=128, input_slots=2, k_format="b8", v_format="b4",
                               fidelity="LoFi", fp32_dst=False, fix_correction=True, safe_rescale=True,
                               pv_transpose_in1=axis == "N").items():
            e.equal(kernel.get(key), value, "Paired kernel " + key)
        checks = kernel["preprocessing_checks"]
        e.equal([x["input"] for x in checks], (["V transpose"] if axis == "N" else []) + ["Q", "K", "V"], "Paired exact prep coverage")
        for x in checks:
            e.equal(x.get("mismatch"), 0, "Paired exact prep mismatch")
        if axis in axis_hashes:
            e.equal(r["output_sha256"], axis_hashes[axis], "Barrier changes preserve output bits")
        axis_hashes[axis] = r["output_sha256"]
        by_id[name] = r
    singleton = {}
    for kind in ("trace_qualification", "summary", "trace_cleanup"):
        values = [r for r in records if r.get("kind") == kind]
        if not values and e.pending:
            e.pending.append("Paired stream awaiting " + kind)
            return
        e.equal(len(values), 1, "Paired " + kind + " count")
        if not values:
            return
        singleton[kind] = values[0]
    trace = singleton["trace_qualification"]
    e.equal(trace.get("each_candidate_replay_bitwise_equal"), True, "Paired trace bit identity")
    e.equal(trace.get("traces"), 6, "All six captures")
    orders = [r for r in records if r.get("kind") == "round_order"]
    samples = [r for r in records if r.get("kind") == "round_samples"]
    e.equal(len(orders), iterations + warmup, "Complete interleaved rounds")
    e.equal(len(samples), len(orders), "Samples for each round")
    measured = {name: [] for name in ids}
    for ordinal, (order, row) in enumerate(zip(orders, samples)):
        indices = [(ordinal // 2 + offset) % 6 for offset in range(6)]
        if ordinal % 2:
            indices.reverse()
        expected_ids = [ids[i] for i in indices]
        e.equal(order.get("ordinal"), ordinal, "Round order ordinal")
        e.equal(row.get("ordinal"), ordinal, "Round sample ordinal")
        e.equal(order.get("candidates"), expected_ids, "Counterbalanced round order")
        e.equal(row.get("measured"), ordinal >= warmup, "Warmup versus measurement")
        e.equal(order.get("phase"), "measure" if ordinal >= warmup else "warmup", "Round phase")
        e.equal([x["id"] for x in row["samples"]], expected_ids, "All candidates sampled in declared order")
        for position, sample in enumerate(row["samples"]):
            e.equal(sample.get("position"), position, "Position within round")
            e.require(finite(sample.get("combined_ms")) and sample["combined_ms"] > 0, "Positive combined timing")
            e.number(sample.get("start_since_timing_origin_s"), "Timing origin")
            if ordinal >= warmup:
                measured[sample["id"]].append(sample["combined_ms"])
    summary = singleton["summary"]
    for key in ("sources_unchanged", "each_candidate_replay_bitwise_equal", "barrier_output_bits_equal", "original_inputs_immutable"):
        e.equal(summary.get(key), True, "Paired final " + key)
    e.equal({x["id"] for x in summary["candidates"]}, set(ids), "Final candidate coverage")
    for r in summary["candidates"]:
        e.equal(r.get("samples_ms"), measured[r["id"]], "Summary matches measured rounds")
        e.close(r.get("median_combined_ms"), statistics.median(measured[r["id"]]), "Paired median arithmetic")
        e.equal(r.get("output_sha256"), by_id[r["id"]]["output_sha256"], "Final output preserved")
    e.equal(singleton["trace_cleanup"].get("errors"), [], "No trace cleanup errors")
    e.equal(singleton["trace_cleanup"].get("trace_count"), 6, "All traces released")
    if records[-1].get("kind") == "complete":
        e.equal(records[-1].get("all_traces_released"), True, "Completion cleanup gate")
        e.equal(records[-1].get("retained_candidate_count"), 6, "Retained candidate count")
    e.summary.update(length=length, candidates=6, measured_rounds=iterations, preprocessing_exact_checked=True,
                     useful_flops=4 * heads * length**2 * 128, reference_scope="Original BF16; hash-pinned producer reference assignment")
    e.notes.append("Combined blocking-trace latency only; no claimed FPU activity or clock-normalized timing")


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
    for length, suffix in ((32768, "32k"), (262144, "256k")):
        for fmt, search in itertools.product(("b8_b4", "b4_b4"), ("native", "minus", "pm")):
            distributions = ["normal"]
            if length == 32768 and search != "minus":
                distributions += ["outliers", "scaled_qk", "channel_outlier_k", "channel_outlier_v"]
            for dist in distributions:
                a.load(f"adaptive{suffix}-{fmt}-{search}-{dist}-v1.json", "adaptive_fullchip", adaptive_fullchip,
                       length=length, fmt=fmt, search=search, distribution=dist)
    for scope in ("smoke", "resident", "32768", "262144"):
        a.load_pair([f"lut-macro-{mode}-{scope}-v1.json" for mode in ("raw", "macro")],
                    "lut_macro", lut_macro, scope=scope)
    for suite in ("vaxis", "formats"):
        a.load(f"codec-{suite}-v1.jsonl", "codec_cpu", codec_cpu, suite=suite)
    a.load("captured-interface-smoke-v1.jsonl", "captured_interface", captured_interface, optional=True, historical=True)
    a.load("captured-interface-smoke-v2.jsonl", "captured_interface", captured_interface)
    for length, seed in ((32768, 1240), (262144, 1240), (32768, 1241)):
        for axis in ("D", "N"):
            suffix = "-seed1241" if seed == 1241 else ""
            a.load(f"vt-full-{length}-{axis}{suffix}-v1.jsonl", "v_transpose", v_transpose,
                   length=length, seed=seed, axis=axis)
    for length, fmt, mode in itertools.product((1024, 32768, 262144), ("b8_b8", "b8_b4"), ("none", "mean_error")):
        distributions = ("normal", "uniform", "constant_v", "common_v") if length == 1024 else ("normal", "uniform")
        for distribution in distributions:
            suffix = "smoke" if length == 1024 else str(length)
            a.load(f"mean-error-{suffix}-{fmt}-{distribution}-{mode}-v1.json", "mean_error", mean_error,
                   length=length, fmt=fmt, mode=mode, distribution=distribution)
    for length in (1024, 32768, 262144):
        for route, options in (("native", ("rne", "single")), ("lut", ("off", "on")), ("bf16", ("off", "on"))):
            suffix = "smoke" if length == 1024 and route != "bf16" else str(length)
            prefix = "hi2-bf16-lut" if route == "bf16" else "hi2-" + route
            for option in options:
                a.load(f"{prefix}-{option}-{suffix}-v1.json", "hifi2_late", hifi2_late, length=length, route=route, option=option)
        for variant in ("lofi_fp32", "lofi_fp32_b8"):
            a.load(f"native-storage-{variant}-{length}-v1.json", "native_storage", native_storage,
                   length=length, route="storage", option=variant)
        for axis in ("D", "N"):
            suffix = "smoke" if length == 1024 else "full-" + str(length)
            a.load(f"vt-b8-{suffix}-{axis}-v1.jsonl", "v8_axis", v8_axis, length=length, seed=1240, axis=axis, vfmt="b8")
    for length in (1024, 32768):
        for h16, axis, adaptive in ((False, "D", "none"), (False, "N", "none"), (True, "D", "none"), (True, "N", "none"), (True, "N", "pm")):
            suffix = "smoke" if length == 1024 else str(length)
            option = "-pm" if adaptive == "pm" else "" if length == 1024 else "-none"
            a.load(f"recipe-{suffix}-h{16 if h16 else 0}-v{axis}{option}-v1.jsonl", "combined_recipe", combined_recipe,
                   length=length, seed=1240, axis=axis, recipe=(h16, adaptive))
        a.load("vt-paired-" + ("smoke" if length == 1024 else "32k") + "-v1.jsonl", "paired_vaxis", paired_vaxis, length=length)
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
