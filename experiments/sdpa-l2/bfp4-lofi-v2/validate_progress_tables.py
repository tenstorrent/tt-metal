# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only standard-library audit of selected recent PROGRESS.md evidence.

No Torch/TTNN imports, device access, or writes. Numeric/scope failures are
fatal. Incomplete historical source manifests and differences from today's
checkout are separate warnings, not evidence that a historical run is invalid.
This deliberately audits a bounded set, not all v2 records or model claims.
"""

import argparse
import ast
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = "experiments/sdpa-l2/bfp4-lofi-v2/"
REF = "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
FROZEN_SFPU = "experiments/sdpa-l2/hybrid-mixed-v1/candidate/tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h"


def field(value, dotted):
    for part in dotted.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class Audit:
    def __init__(self):
        self.progress = (HERE / "PROGRESS.md").read_text()
        self.records = {}
        self.failures = []
        self.checks = Counter()
        self.hash_cache = {}

    def require(self, condition, description, kind="scope"):
        self.checks[kind] += 1
        if not condition:
            self.failures.append(description)

    def load(self, name, **flags):
        if name not in self.records:
            path = HERE / name
            value = json.loads(path.read_text())
            self.records[name] = value
        value = self.records[name]
        for key, expected in flags.items():
            self.require(field(value, key) == expected, f"{name}: {key} != {expected!r}")
        return value

    def section(self, heading):
        marker = "## Update: " + heading
        self.require(marker in self.progress, f"Missing PROGRESS section {heading}", "markdown")
        return self.progress.split(marker, 1)[1].split("\n## ", 1)[0]

    def row(self, section, label):
        matches = [
            line for line in section.splitlines() if line.startswith("|") and line.split("|")[1].strip() == label
        ]
        self.require(len(matches) == 1, f"Expected one table row: {label}", "markdown")
        return [cell.strip() for cell in matches[0].split("|")[2:-1]]

    def number(self, actual, displayed, description):
        text = str(displayed)
        expected = Decimal(text)
        tolerance = Decimal(1).scaleb(expected.as_tuple().exponent) / 2
        self.require(
            abs(Decimal(str(actual)) - expected) <= tolerance,
            f"{description}: raw {actual}, displayed {text}",
            "numeric",
        )

    def prose(self, section, value, token, description):
        self.require(token in section, f"PROGRESS lacks {description}: {token}", "markdown")
        self.number(value, token, description)

    def fullchip(self, name, length, **flags):
        d = self.load(name, length=length, heads=10, cores=110, actual_cores=110, sample_rows=128, seed=1240, **flags)
        self.require(len(d["sampled_query_rows"]) == 128, f"{name}: not128 sampled Q rows")
        self.require(d["useful_flops"] == 4 * 10 * length**2 * 128, f"{name}: wrong useful FLOP scope")
        for kind in ("attention", "combined"):
            if d[kind]["median_ms"] is not None:
                expected = d["useful_flops"] / (d[kind]["median_ms"] * 1e9)
                self.require(abs(expected - d[kind + "_tflops"]) < 1e-9, f"{name}: {kind} TF arithmetic", "arithmetic")
        return d

    def manifests(self):
        warnings, changed, missing_current, pairs = [], [], [], 0
        for name, d in self.records.items():
            hashes = d.get("source_sha256")
            companion = (HERE / name).with_suffix(".provenance.json")
            if hashes is None and companion.exists():
                hashes = json.loads(companion.read_text()).get("source_sha256")
            if not hashes:
                warnings.append(dict(record=name, warning="No source manifest in record or companion"))
                continue
            required = []
            if "sampled_query_rows" in d or name.startswith("qcenter-"):
                required.append(REF)
            if REL + "numerics.py" in hashes:
                required += [
                    "experiments/sdpa-l2/bfp4-lofi-v1/probe.py",
                    "experiments/sdpa-l2/bfp4-lofi-v1/numerics.py",
                    "experiments/sdpa-l2/frontier-accuracy-v1/run.py",
                ]
            if REL + "center_preprocess.py" in hashes:
                required.append(REL + "center_preprocess/round.hpp")
            if d.get("fp32_dst") or d.get("destination") == "fp32":
                required.append(FROZEN_SFPU)
            for source in required:
                if source not in hashes:
                    warnings.append(
                        dict(record=name, warning="Critical source absent from historical manifest", source=source)
                    )
            differences, missing = [], []
            for source, digest in hashes.items():
                pairs += 1
                path = ROOT / source
                if path not in self.hash_cache:
                    self.hash_cache[path] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                now = self.hash_cache[path]
                if now is None:
                    missing.append(source)
                elif now != digest:
                    differences.append(source)
            if differences:
                changed.append(dict(record=name, sources=differences))
            if missing:
                missing_current.append(dict(record=name, sources=missing))
        return dict(
            completeness_scope="Selected critical reference/oracle/SFPU pins; not exhaustive dependency closure",
            completeness_warnings=warnings,
            current_checkout_comparison=dict(
                record_source_pairs=pairs,
                unique_current_paths=len(self.hash_cache),
                records_with_changed_sources=changed,
                records_with_missing_sources=missing_current,
                interpretation="Historical hashes need not match today's checkout; these are not metric/scope failures",
            ),
        )


def audit_tables(a):
    native = a.section("native FP32, long-context stress, and preprocessing")
    for length, prefix, values in (
        (32768, "native32k", (("b8", "2.8244", "153.83"), ("b4", "16.8573", "155.04"))),
        (262144, "native256k", (("b8", "2.7984", "161.19"), ("b4", "16.9369", "166.38"))),
    ):
        for fmt, l2, tf in values:
            name = f"{prefix}-lofi_fp32_{fmt}-v1.json"
            d = a.fullchip(
                name,
                length,
                distribution="normal",
                native_exp=True,
                fp32_dst=True,
                input_slots=1,
                device_preprocessing=True,
                center_k=False,
                reader_chain=True,
            )
            a.prose(native, d["accuracy"]["l2_pct"], l2, name + " L2")
            a.prose(native, d["combined_tflops"], tf, name + " combined TF")
            a.require("SDPA_LOFI_NATIVE_EXP" in d["defines"], name + " native flag")
    d = a.fullchip("quadratic32k-b8-v1.json", 32768, distribution="normal", native_exp=False, exp_degree=2)
    a.prose(native, d["accuracy"]["l2_pct"], "2.1584", "quadratic L2")
    a.prose(native, d["combined_tflops"], "110.19", "quadratic combined TF")

    for distribution, label in (("constant_v", "Constant V=1"), ("common_v", "V with common offset32")):
        cells = a.row(native, label)
        for index, mode in enumerate(("main", "full", "denom")):
            name = f"denomstress256k-{distribution}-{mode}-v1.json"
            d = a.fullchip(
                name,
                262144,
                distribution=distribution,
                kv_formats="b8_b4",
                destination="main_bf16" if mode == "main" else "fast_bf16",
                denom_only=mode == "denom",
                input_slots=2,
                iters=0,
            )
            a.number(d["accuracy"]["l2_pct"], cells[index], name)
            a.require(d["combined_tflops"] is None, name + " stress must not claim timing")

    for mode, label in (
        ("none", "None"),
        ("original_mean", "Center + original BF16 mean"),
        ("matched_mean", "Center + matched represented mean"),
    ):
        cells = a.row(native, label)
        for distribution in ("normal", "common_v", "constant_v"):
            name = f"valuecenter-32768-{mode}-{distribution}-v1.json"
            d = a.fullchip(
                name,
                32768,
                distribution=distribution,
                kv_formats="b8_b4",
                destination="fast_bf16",
                denom_only=True,
                input_slots=2,
                center_mode=mode,
            )
            if distribution == "normal":
                a.number(d["accuracy"]["l2_pct"], cells[0], name + " L2")
                a.number(d["combined_tflops"], cells[2], name + " combined TF")
            elif distribution == "common_v":
                a.number(d["accuracy"]["l2_pct"], cells[1], name + " L2")
                if mode == "matched_mean":
                    m = d["centered_output_accuracy"]
                    a.prose(native, m["l2_pct"], "125.23", "common-V residual L2")
                    a.require(m["gain_alignment"] is False, "Residual metric must not fit gain")
            elif mode != "none":
                m = d["centered_output_accuracy"]
                a.require(m["constant_v"] and m["l2_pct"] is None, name + " zero-residual handling")
                a.require(d["accuracy"]["max_abs"] < 1e-12, name + " constant V exact modulo FP64 noise")
            if mode != "none":
                a.require(
                    d["epilogue_check"]["mismatch"] == 0 and d["epilogue_dtype"] == "BF16", name + " BF16 epilogue"
                )

    for fmt, label in (("b4_b8", "K4/V8"), ("b4_b4", "K4/V4"), ("b8_b4", "K8/V4")):
        cells = a.row(native, label)
        unrotated, rotated = cells[0].split("/"), cells[1].split("/")
        for index, seed in enumerate((1240, 1241)):
            for prefix, expected in (("unrotated32k", unrotated[index]), ("rotated16-32k", rotated[index])):
                name = f"{prefix}-outliers-{fmt}-{seed}-v1.json"
                d = a.load(
                    name,
                    length=32768,
                    heads=10,
                    actual_cores=110,
                    distribution="outliers",
                    seed=seed,
                    kv_formats=fmt,
                    destination="fast_bf16",
                    denom_only=True,
                    input_slots=2,
                )
                a.number(d["accuracy"]["l2_pct"], expected.strip(), name)
                if prefix.startswith("rotated"):
                    a.require(d["hadamard_size"] == 16, name + " rotation size")
            normal = a.load(
                f"rotated16-32k-normal-{fmt}-{seed}-v1.json",
                length=32768,
                heads=10,
                actual_cores=110,
                distribution="normal",
                seed=seed,
                hadamard_size=16,
                denom_only=True,
            )
            # The prose is approximate: preserve its broad207..215 range with
            # rounding-to-integer tolerance, not a claim of a precise bound215.
            a.require(206.5 <= normal["combined_tflops"] < 215.5, "H16 approximate throughput range")

    for mode, l2 in (("skip", "2.1930"), ("zero", "2.1930"), ("normal", "2.1697"), ("commonq", "1.5385")):
        d = a.load(
            f"qcenter-halfpost-{mode}-v1.json",
            length=1024,
            q_chunk=128,
            k_chunk=512,
            half_sync=True,
            fp32_dst=True,
            dst_full_sync_en=False,
            attention_cores=1,
            iters=0,
        )
        a.prose(native, d["accuracy"]["l2_pct"], l2, "Qcenter " + mode)
        expected_add = "bypassed diagnostic" if mode == "skip" else "2score+2correction"
        a.require("all128" in d["reference"] and expected_add in d["correction_add"], "Qcenter " + mode + " scope")
    tiny = a.load("tiny-mean-perf-262144-v1.json", length=262144)
    for d, expected, calls in zip(tiny["cases_results"], ("1.6688", "0.2302", "0.3244"), (8, 1, 1)):
        a.prose(native, d["median_ms"], expected, "tiny " + d["case"])
        a.require(d["matmul_calls_per_invocation"] == calls and d["distinct_mean_rows"] == 8, "Tiny mean batch scope")
        a.require(
            d["producer_and_retiling_included"] is False and d["partial_tile_input_output_conversion_exact"],
            "Tiny primitive exclusions",
        )
        if d["versus_first_case"] is not None:
            a.require(d["versus_first_case"]["max_abs"] == 0, "Tiny outputs not identical")

    compressed = a.section("compressed P remains a negative optimization")
    for fmt, blocked, label in (
        ("fp32", False, "FP32, scalar"),
        ("fp32", True, "FP32, standard4"),
        ("bf16", False, "BF16 padded alias, scalar"),
        ("bf16", True, "BF16 padded alias, custom4"),
    ):
        cells = a.row(compressed, label)
        for index, exp in enumerate(("cubic", "native")):
            prefix = "padded-p16-blocked-resident" if blocked else "padded-p16-resident"
            name = f"{prefix}-{fmt}-{exp}-v1.json"
            d = a.load(
                name,
                p_format=fmt,
                p_pack_width=4 if blocked else 1,
                native_exp=exp == "native",
                q_chunk=256,
                k_chunk=512,
                q_repeats=16,
                k_chunks=512,
                cores=1,
                recurring_input_dm=False,
                preprocessing_in_timing=False,
            )
            a.number(d["tflops_per_core"], cells[index], name)
            if fmt == "bf16":
                a.require(d["p_page_bytes"] == 4096 and d["p_fifo_counters_independent"], name + " padded alias scope")
            if blocked:
                control = a.load(f"padded-p16-resident-{fmt}-{exp}-v1.json")
                a.require(d["output_sha256"] == control["output_sha256"], name + " scalar/blocked equality")
    for dst in ("bf16", "fp32"):
        d = a.load(f"padded_pack_probe/padded-pack-smoke-{dst}-v1.json", tiles=128, fp32_dst=dst == "fp32")
        a.require(
            d["correctness_pass"] and all(v == 0 for v in d["padded_vs_compact_mismatches"].values()),
            "Padded pack exactness",
        )
    for fmt in ("bf16", "fp32"):
        for exp in ("cubic", "native"):
            d = a.load(
                f"padded-p16-blocked-smoke-{fmt}-{exp}-v1.json",
                length=1024,
                heads=2,
                sample_rows=1024,
                p_format=fmt,
                native_exp=exp == "native",
                p_pack_width=4,
                check_preprocess=True,
            )
            a.require(
                len(d["sampled_query_rows"]) == 1024 and all(c["mismatch"] == 0 for c in d["preprocessing_checks"]),
                "PaddedP smoke all-output scope",
            )

    old = a.section("cheaper FP32 exp and asymmetric K/V")
    for label, name, degree in (
        ("Cubic", "exp-resident-fp32-degree3-v2.json", 3),
        ("Quadratic", "exp-resident-fp32-degree2-v2.json", 2),
        ("Linear", "exp-resident-fp32-degree1-v2.json", 1),
        ("Native, no refiner or repeated grid init", "native-exp-resident-v1.json", 3),
    ):
        cells = a.row(old, label)
        d = a.load(
            name,
            qkv_route="host",
            q_chunk=256,
            k_chunk=512,
            q_repeats=16,
            k_chunks=512,
            cores=1,
            destination="fp32",
            distinct_kv=False,
            exp_degree=degree,
        )
        a.number(d["tflops_per_core"], cells[0], name + " TF")
        a.number(d["l2_pct"], cells[1], name + " L2")


def qualification_source_audit():
    """Evaluate only the existing read-only source_hashes function, no imports."""
    path = HERE / "native_exp_qualification.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_hashes")
    namespace = dict(Path=Path, hashlib=hashlib, HERE=HERE, ROOT=ROOT, __file__=str(path))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    hashes = namespace["source_hashes"]()
    required = [
        REF,
        FROZEN_SFPU,
        REL + "center_preprocess/round.hpp",
        "experiments/sdpa-l2/bfp4-lofi-v1/probe.py",
        "experiments/sdpa-l2/bfp4-lofi-v1/numerics.py",
        "experiments/sdpa-l2/frontier-accuracy-v1/run.py",
        "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl",
        "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl",
    ]
    return dict(
        script=str(path.relative_to(ROOT)),
        listed_sources=len(hashes),
        reference_is_pinned=REF in hashes,
        missing_selected_dependencies=[p for p in required if p not in hashes],
        note="Current script audit, not a retroactive repair of any historical manifest",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Full evidence index and separate provenance warnings to stdout"
    )
    args = parser.parse_args()
    audit = Audit()
    audit_tables(audit)
    provenance = audit.manifests()
    qualification = qualification_source_audit()
    result = dict(
        status="FAIL" if audit.failures else "PASS",
        records=len(audit.records),
        checks=dict(audit.checks),
        failures=audit.failures,
        scope="Selected recent nativeFP32/denomstress/Vcenter32K/H16/Qcenter/tiny/P16 evidence, not all PROGRESS",
        evidence=[
            dict(file=name, sha256=hashlib.sha256((HERE / name).read_bytes()).hexdigest())
            for name in sorted(audit.records)
        ],
        provenance=provenance,
        native_qualification_sources=qualification,
        unverified_claims="Model-level conclusions, pending jobs, locked-variant validation counts and allocation failure logs are outside this JSON audit",
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"{result['status']}: {result['records']} records; {sum(audit.checks.values())} numeric/scope/markdown checks"
        )
        for failure in audit.failures:
            print("FAIL:", failure)
        print(f"WARN source-pin completeness: {len(provenance['completeness_warnings'])} selected dependency omissions")
        current = provenance["current_checkout_comparison"]
        print(
            f"INFO current checkout (not historical validity): {len(current['records_with_changed_sources'])} records differ; "
            f"{len(current['records_with_missing_sources'])} have missing paths"
        )
        print(
            f"WARN native_exp_qualification.py: reference pinned={qualification['reference_is_pinned']}; "
            f"{len(qualification['missing_selected_dependencies'])} selected dependency omissions"
        )
    raise SystemExit(bool(audit.failures))


if __name__ == "__main__":
    main()
