#!/usr/bin/env python3
"""Inventory and execute the complete tt-llk SFPU header corpus."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
LLK = ROOT / "tt_metal/tt-llk"
MANIFEST = HERE / "sfpu_corpus_v2.tsv"
DEVICE_BASELINE = HERE / "sfpu_device_baseline_v1.tsv"
# Scoped silicon baselines are keyed by chip class: absolute cycles re-baseline
# across device classes, so a measurement may only ever be compared against a
# baseline taken on the same class.  sfpu_device_baseline_v1.tsv is the
# immutable p100a-era migration source; p150 rows live in their own file.
DEVICE_BASELINES = {
    "p100a": DEVICE_BASELINE,
    "p150": HERE / "sfpu_device_baseline_p150_v1.tsv",
}
PYTEST_PLUGIN = "sfpu_corpus_pytest_plugin"
EXPECTED = {
    "logical": 164,
    "bh": 152,
    "wh": 138,
    "qsr": 42,
    "physical_paths": 332,
    "basename_union": 143,
    "legacy_bh": 41,
    "legacy_wh": 32,
    "legacy_qsr": 14,
    "raw": 51,
    "typed": 151,
    "replay": 13,
    "mop": 3,
}
DISCOVERY_FIELDS = [
    "id",
    "surface",
    "arches",
    "header_bh",
    "header_wh",
    "header_qsr",
    "raw_tti",
    "typed_sfpi",
    "replay",
    "mop",
    "functional_modules",
    "perf_modules",
    "mapping_state",
    "notes",
]
AUDIT_FIELDS = [
    "semantic_cpp_class",
    "semantic_cpp_blocker",
    "paired_selector_status",
    "test_status",
    "perf_status",
    "correctness_metric",
    "correctness_threshold",
    "correctness_source",
    "silicon_status",
    "silicon_result",
    "silicon_source",
]
FIELDS = ["version", *DISCOVERY_FIELDS, *AUDIT_FIELDS]
SEMANTIC_CPP_CLASSES = {
    "ready",
    "typed_wrapper_needed",
    "macro_dependent",
    "multithread_boundary",
    "unmapped",
}
PAIR_STATUSES = {"absent", "blocked", "implemented"}
GATE_STATUSES = {"not_run", "blocked", "pass", "fail"}
PERF_STATUSES = {"not_run", "blocked", "measured"}
SILICON_STATUSES = {"not_run", "blocked", "win", "parity", "loss"}
CORRECTNESS_METRICS = {"none", "pcc", "exact", "tolerance"}

# Capabilities are attached to complete corpus IDs, never inferred from a test
# filename.  A missing capability blocks only rows that declare it.
ROW_REQUIRED_CAPABILITIES = {
    "legacy__ckernel_sfpu_topk": ("indexed_topk",),
}
COMPILER_CAPABILITY_PROBES = {
    "indexed_topk": r"""
using vec_t = __xtt_vector;
void probe(vec_t a, vec_t b, vec_t c, vec_t d,
           vec_t e, vec_t f, vec_t g, vec_t h) {
  auto swap = __builtin_rvtt_sfpswap_indexed(a, b, c, d, 1);
  auto transpose = __builtin_rvtt_sfptransp8(a, b, c, d, e, f, g, h);
  (void)swap;
  (void)transpose;
}
""",
}

# Every exception is keyed by the complete stable corpus ID.  There are no
# basename fragments, substring matches, or inferred semantic classifications.
AUDITED_SEEDS = {
    "metal__ckernel_sfpu_exp": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="Typed 21-bit BF16 Exp is functionally valid; competitive lowering needs loop-invariant SFPU constant placement and replay formation for counted typed loops.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="tolerance",
        correctness_threshold="Float16_b rtol=0.05 atol=0.05 plus PCC > 0.99; existing Exp functional domain and edge_spec",
        correctness_source="test_sfpu_unary.py::test_exp_fresh_cpp",
        silicon_status="loss",
        silicon_result="BH MATH_ISOLATE: fresh semantic C++ 989.75 cycles vs production 579.7421875 (+70.72%), three fresh samples each.",
        silicon_source="EXP_SEMANTIC_SILICON_AB.md; sfpu_device_baseline_v1.tsv; audited BH device archive",
    ),
    "metal__ckernel_sfpu_sigmoid_appx": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="Fresh typed cubic is functionally valid; competitive lowering needs loop-invariant SFPU constant materialization/hoisting, special-register allocation, and counted-loop replay formation.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="tolerance",
        correctness_threshold="Float16_b rtol=0.05 atol=0.13 plus PCC > 0.99",
        correctness_source="test_sfpu_unary.py::test_sigmoid_appx_fresh_cpp",
        silicon_status="loss",
        silicon_result="BH MATH_ISOLATE: fresh semantic C++ 446.8515625 cycles vs production 222.8515625 (+100.52%), three fresh samples each.",
        silicon_source="FRESH_CPP_SILICON_ATTACK.md; sfpu_device_baseline_v1.tsv; audited BH device archive",
    ),
    "metal__ckernel_sfpu_signbit": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="Fresh typed load/shift/cast/store body is complete; compiler-owned descriptor materialization and delayed SFPLOADMACRO scheduling are formed only under the opt-in structural proof.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="pcc",
        correctness_threshold="Float32 rtol=0.05 atol=0.05 plus PCC > 0.99; signed finite domain excludes zero",
        correctness_source="test_sfpu_unary.py::test_eltwise_unary_sfpu_signbit; helpers/utils.py:548-785; SIGNBIT_LOADMACRO_SILICON_AB.md",
        silicon_status="win",
        silicon_result="BH TILE_LOOP MATH_ISOLATE: generated typed+SFPLOADMACRO 21508 cycles vs production 23246 (-7.4766%), three identical fresh processes each.",
        silicon_source="SIGNBIT_LOADMACRO_SILICON_AB.md; sfpu_device_baseline_v1.tsv; audited BH device archive",
    ),
    "metal__ckernel_sfpu_lerp": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="The production Lerp body is already clean typed SFPI C++. The opt-in generic latency, adjacent-Dst fusion, invariant-immediate, and replay-hoist passes fuse two rows per replay without changing the source or numerical expression.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="pcc",
        correctness_threshold="Float16_b rtol=0.05 atol=0.05 plus PCC > 0.99 on the existing seeded ternary domain",
        correctness_source="test_sfpu_ternary.py::test_sfpu_ternary[formats:Float16_b->Float16_b-dest_acc:No-mathop:SfpuLerp]; helpers/utils.py:548-785; LERP_COMPILER_SILICON_AB.md",
        silicon_status="win",
        silicon_result="BH TILE_LOOP mean(MATH_ISOLATE): compiler-pass ON 564.984375 cycles/tile vs OFF 580.984375 (-2.7539%), three identical fresh processes each.",
        silicon_source="LERP_COMPILER_SILICON_AB.md; sfpu_device_baseline_v1.tsv; audited BH device archive",
    ),
    "legacy__ckernel_sfpu_welfords": dict(
        semantic_cpp_class="typed_wrapper_needed",
        semantic_cpp_blocker="Generated vFloat body exists; raw LREG live-in/live-out ABI remains an explicit typed-boundary requirement.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="tolerance",
        correctness_threshold="mean rtol=0.02 atol=0.02; m2 rtol=0.03 atol=0.03",
        correctness_source="test_sfpu_welford_prefix_snapshot.py:77-78",
        silicon_status="win",
        silicon_result="BH WELFORD_BODY: generated 323 cycles vs handwritten 326 (-0.92%); scoped body metric.",
        silicon_source="sfpu_device_baseline_v1.tsv; SFPI_COMPILER_UPGRADE.md section 13.8",
    ),
    "legacy__ckernel_sfpu_reduce_custom": dict(
        semantic_cpp_class="typed_wrapper_needed",
        semantic_cpp_blocker="Arithmetic is semantic SFPI; destination loads, TTINCRWC barrier, L8 discard load, and replay ownership require typed architectural boundaries.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="pcc",
        correctness_threshold="PCC > 0.99 and Float16_b element tolerance rtol=0.05 atol=0.05",
        correctness_source="test_sfpu_reduce_sdpa.py:135; helpers/utils.py:548-785",
        silicon_status="win",
        silicon_result="BH REDUCE_SDPA_BODY: D1 generated 834 cycles vs handwritten 840 (-0.714%); three fresh samples.",
        silicon_source="REDUCE_SDPA_SILICON_AB.md; sfpu_device_baseline_v1.tsv",
    ),
    "legacy__ckernel_sfpu_binary_bcast": dict(
        semantic_cpp_class="typed_wrapper_needed",
        semantic_cpp_blocker="Arithmetic island is semantic vFloat; broadcast addressing, address modifiers, fixed-LREG endpoints, and replay remain explicit architectural boundaries.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="pcc",
        correctness_threshold="PCC > 0.99 and Float16_b element tolerance rtol=0.05 atol=0.05",
        correctness_source="test_sfpu_binary.py:1748-1749; helpers/utils.py:548-785",
        silicon_status="parity",
        silicon_result="BH BINARY_BCAST_BODY: generated 608 cycles vs handwritten 608 (exact cycle parity), three fresh samples.",
        silicon_source="BINARY_BCAST_SILICON_AB.md; sfpu_device_baseline_v1.tsv",
    ),
    "legacy__ckernel_sfpu_where": dict(
        semantic_cpp_class="macro_dependent",
        semantic_cpp_blocker="Canonical v_if selector is correct, but competitive lowering requires general SFPLOADMACRO formation; generated replay payload is seven slots versus three handwritten slots.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="exact",
        correctness_threshold="bit-exact selected Float16_b payload with NaNs equal; no tolerance",
        correctness_source="test_sfpu_ternary.py:298-321",
        silicon_status="loss",
        silicon_result="BH TTNN_WHERE_BODY: generated 312.50 cycles vs handwritten 159.25 (+96.23%), three fresh samples.",
        silicon_source="TTNN_WHERE_COMPILER_AB.md; sfpu_device_baseline_v1.tsv",
    ),
    "legacy__ckernel_sfpu_mul_int": dict(
        semantic_cpp_class="macro_dependent",
        semantic_cpp_blocker="Fresh integer arithmetic selector is correct, but competitive lowering requires general SFPLOADMACRO formation and typed mul24/shift/saturation scheduling.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="exact",
        correctness_threshold="Int32 element tolerance rtol=0 atol=0 plus PCC > 0.99 when signal is nonzero",
        correctness_source="test_sfpu_binary.py:902-921; helpers/utils.py:548-785",
        silicon_status="loss",
        silicon_result="BH MATH_ISOLATE: generated 562.625 cycles vs handwritten 283.9296875 (+98.16%).",
        silicon_source="sfpu_device_baseline_v1.tsv; audited BH device archive",
    ),
    "metal__ckernel_sfpu_mul_int32": dict(
        semantic_cpp_class="macro_dependent",
        semantic_cpp_blocker="Production Metal implementation owns SFPLOADMACRO scheduling; paired test selector is mapped through the legacy test surface pending a direct row mapping.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="exact",
        correctness_threshold="Int32 element tolerance rtol=0 atol=0 plus PCC > 0.99 when signal is nonzero",
        correctness_source="test_sfpu_binary.py:902-921; helpers/utils.py:548-785",
        silicon_status="loss",
        silicon_result="BH MATH_ISOLATE: generated 562.625 cycles vs handwritten 283.9296875 (+98.16%).",
        silicon_source="sfpu_device_baseline_v1.tsv; audited BH device archive",
    ),
    "metal__ckernel_sfpu_recip": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="A fresh typed value-space body is implemented. BH's canonical accurate sfpu_reciprocal<false> expansion still trips rvtt_expand SSA verification; the branch-free cubic Newton correction is the audited semantic route, while architectural replay selection remains compiler-owned.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="pcc",
        correctness_threshold="PCC > 0.99 and per-format element tolerance; measured accurate Float16_b lane uses rtol=0.05 atol=0.05",
        correctness_source="test_sfpu_unary.py:964-1021; helpers/utils.py:548-785; RECIPROCAL_SEMANTIC_SILICON_AB.md",
        silicon_status="win",
        silicon_result="BH RECIPROCAL_BODY accurate Float16_b: semantic 459 cycles vs production 467 (-1.713%), three fresh samples.",
        silicon_source="RECIPROCAL_SEMANTIC_SILICON_AB.md; sfpu_device_baseline_v1.tsv",
    ),
    "legacy__ckernel_sfpu_topk": dict(
        semantic_cpp_class="typed_wrapper_needed",
        semantic_cpp_blocker="Needs sound multi-result indexed SFPSWAP and eight-value SFPTRANSP modeling plus explicit RWC/DST/config/replay boundaries before a full selector is accepted.",
        paired_selector_status="blocked",
        test_status="blocked",
        perf_status="blocked",
        correctness_metric="exact",
        correctness_threshold="exact value/index association; exact stable indices, tie-equivalent indices only for explicitly unstable cases",
        correctness_source="test_topk.py:134-240; TOPK_TYPED_CONVERSION_BLOCKER.md",
        silicon_status="blocked",
        silicon_result="No semantically complete paired selector or isolated silicon result.",
        silicon_source="TOPK_TYPED_CONVERSION_BLOCKER.md; sfpu_device_baseline_v1.tsv",
    ),
    "legacy__ckernel_sfpu_generic_moe_gate_topk": dict(
        semantic_cpp_class="typed_wrapper_needed",
        semantic_cpp_blocker="Semantic scoring can be fresh C++; TopK paired state, destination/RWC movement, transpose, and MOP/replay ownership need typed architectural models.",
        paired_selector_status="absent",
        test_status="not_run",
        perf_status="blocked",
        correctness_metric="tolerance",
        correctness_threshold="values must pass the test's format tolerance/PCC gate and indices must preserve value/index association",
        correctness_source="test_sfpu_generic_moe_gate_topk.py",
        silicon_status="blocked",
        silicon_result="No paired isolated profiler fixture.",
        silicon_source="f1_candidates.tsv rank 7",
    ),
    "legacy__ckernel_sfpu_sdpa_exp_unclamped": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="No architectural conversion blocker identified at the vFloat leaf boundary; numerical-domain and FP32 destination parity must be characterized before promotion.",
        paired_selector_status="absent",
        test_status="not_run",
        perf_status="blocked",
        correctness_metric="pcc",
        correctness_threshold="PCC > 0.99 plus per-format element tolerance unless the paired test establishes a stricter contract",
        correctness_source="test_sfpu_sdpa_exp_unclamped.py; helpers/utils.py:548-785",
        silicon_status="blocked",
        silicon_result="No isolated paired profiler module.",
        silicon_source="f1_candidates.tsv rank 10",
    ),
    "legacy__ckernel_sfpu_softmax_k": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="No architectural conversion blocker identified at the vFloat leaf boundary; transcendental approximation and precision parity must be characterized before promotion.",
        paired_selector_status="absent",
        test_status="not_run",
        perf_status="blocked",
        correctness_metric="pcc",
        correctness_threshold="PCC > 0.99 plus per-format element tolerance unless the paired test establishes a stricter contract",
        correctness_source="test_sfpu_softmax_k.py; helpers/utils.py:548-785",
        silicon_status="blocked",
        silicon_result="No isolated paired profiler module.",
        silicon_source="f1_candidates.tsv rank 9",
    ),
    "metal__ckernel_sfpu_addcmul": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="Fresh typed ternary addcmul body is complete; the OFF-to-ON delta is carried by the generic dst-autoincr/latency passes (the macro planner refuses: no macro region), landing at exact-class cycle parity with the handwritten kernel.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="measured",
        correctness_metric="pcc",
        correctness_threshold="Float16_b element tolerance rtol=0.05 atol=0.05 plus PCC > 0.99",
        correctness_source="test_sfpu_ternary.py::test_fresh_cpp_addcmul",
        silicon_status="parity",
        silicon_result="BH p150 TILE_LOOP mean(MATH_ISOLATE)/tile: semantic ON 36.624 vs handwritten 36.615 (+0.02% parity; causal OFF 44.629 -> ON -17.94%), three fresh processes per leg.",
        silicon_source="sfpu_device_baseline_p150_v1.tsv; sweep-2x2 evidence-20260816 SCOREBOARD.md",
    ),
    "metal__ckernel_sfpu_binary_max_min": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="Fresh semantic max/min body (fresh_cpp_operations.h) is complete; the planner derives the SFPLOADMACRO emission generically (Min/Max exact calendar deleted at WP7, byte-parity oracles green at the WP8 tip).",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="not_run",
        correctness_metric="pcc",
        correctness_threshold="Float16_b element tolerance rtol=0.05 atol=0.05 plus PCC > 0.99; CRAQ matrix {BH,WH}x{Min,Max}x{OFF,ON} through the generic simulator path",
        correctness_source="test_sfpu_binary.py::test_fresh_cpp_binary_max_min",
        silicon_status="not_run",
        silicon_result="p150 silicon perf pending: the perf kernel failed to compile at 55ce75be (unqualified call_binary_max_min_fresh_cpp) — fixed on this branch; measurement lands in sfpu_device_baseline_p150_v1.tsv.",
        silicon_source="sweep-2x2 evidence-20260816 SCOREBOARD.md blocked-list item 1",
    ),
    "legacy__ckernel_sfpu_typecast_fp16b_uint16": dict(
        semantic_cpp_class="ready",
        semantic_cpp_blocker="Fresh semantic UInt16-to-Float16_b body (eltwise_unary_typecast_test.cpp) is complete and the planner fires on the corr shape; WP8 step 4 four-region descriptor sharing targets the remaining gap.",
        paired_selector_status="implemented",
        test_status="pass",
        perf_status="not_run",
        correctness_metric="exact",
        correctness_threshold="exact converted payload per the typecast golden",
        correctness_source="test_eltwise_unary_typecast.py::test_eltwise_unary_typecast",
        silicon_status="not_run",
        silicon_result="p150 silicon perf pending: prior p100a record +17.2% macro / +19.6% semantic loss; the impl-parameterized profiler node is test_typecast_device_profile (reconciled: Lane D's absent-node note pointed at perf_eltwise_typecast.py).",
        silicon_source="TYPECAST_LOADMACRO_SILICON_AB.md; sweep-2x2 evidence-20260816 SCOREBOARD.md blocked-list item 2",
    ),
    "legacy__ckernel_sfpu_topk_xl": dict(
        semantic_cpp_class="typed_wrapper_needed",
        semantic_cpp_blocker="Requires typed paired value/index state, eight-value transpose, replay-range allocation, alternating-direction legality, and general MOP formation.",
        paired_selector_status="absent",
        test_status="not_run",
        perf_status="blocked",
        correctness_metric="exact",
        correctness_threshold="exact values and exact companion indices for deterministic stimuli",
        correctness_source="test_topk_xl.py",
        silicon_status="blocked",
        silicon_result="No paired isolated profiler fixture.",
        silicon_source="f1_candidates.tsv rank 9",
    ),
}

AUDITED_MAPPINGS = {
    "metal__ckernel_sfpu_exp": dict(
        functional_modules="test_sfpu_unary.py::test_exp_fresh_cpp",
        perf_modules="perf_eltwise_unary_sfpu.py::test_perf_exp_fresh_cpp",
        notes="Audited paired production/fresh typed 21-bit BF16 Exp selector with functional, edge, and isolated profiler fixtures.",
    ),
    "metal__ckernel_sfpu_sigmoid_appx": dict(
        functional_modules="test_sfpu_unary.py::test_sigmoid_appx_fresh_cpp",
        perf_modules="perf_eltwise_unary_sfpu.py::test_perf_sigmoid_appx_fresh_cpp",
        notes="Audited paired production/fresh semantic-C++ SigmoidAppx selector and isolated profiler fixture.",
    ),
    "metal__ckernel_sfpu_signbit": dict(
        functional_modules="test_sfpu_unary.py::test_eltwise_unary_sfpu_signbit",
        perf_modules="perf_eltwise_unary_sfpu.py::test_perf_signbit_fresh_cpp",
        notes="Audited production/fresh typed Signbit selector; compiler opt-in forms one dominating descriptor configuration and a delayed macro launch per row.",
    ),
    "metal__ckernel_sfpu_addcmul": dict(
        functional_modules="test_sfpu_ternary.py::test_fresh_cpp_addcmul",
        perf_modules="perf_sfpu_ternary.py::test_perf_fresh_cpp_addcmul",
        notes="Audited paired production/fresh typed ternary addcmul selector with functional and isolated profiler fixtures.",
    ),
    "metal__ckernel_sfpu_binary_max_min": dict(
        functional_modules="test_sfpu_binary.py::test_fresh_cpp_binary_max_min",
        perf_modules="perf_eltwise_binary_sfpu.py::test_perf_fresh_cpp_binary_max_min",
        notes="Audited paired production/fresh semantic max/min selector; the macro planner derives the SFPLOADMACRO emission generically on the fresh body.",
    ),
    "legacy__ckernel_sfpu_typecast_fp16b_uint16": dict(
        functional_modules="test_eltwise_unary_typecast.py::test_eltwise_unary_typecast",
        perf_modules="test_eltwise_unary_typecast.py::test_typecast_device_profile",
        notes="Audited paired handwritten/semantic UInt16->Float16_b selector; the impl-parameterized profiler node lives in the functional module, not perf_eltwise_typecast.py.",
    ),
    "metal__ckernel_sfpu_recip": dict(
        functional_modules="test_sfpu_unary.py::test_reciprocal_semantic,test_sfpu_unary.py::test_reciprocal_semantic_edges",
        perf_modules="test_sfpu_unary.py::test_reciprocal_device_profile",
        notes="audited test-only production/semantic selector; accurate and approximate BF16/FP32 plus registered-domain edges; scoped accurate-BF16 profiler",
    ),
    "metal__ckernel_sfpu_lerp": dict(
        functional_modules="test_sfpu_ternary.py::test_sfpu_ternary[formats:Float16_b->Float16_b-dest_acc:No-mathop:SfpuLerp]",
        perf_modules="perf_sfpu_ternary.py::test_perf_sfpu_lerp",
        notes="Audited identical-source compiler-flag A/B on the production typed Lerp body; exact measured lane is Float16_b without destination accumulation.",
    ),
}

DEFAULT_AUDIT = dict(
    semantic_cpp_class="unmapped",
    semantic_cpp_blocker="No row-specific semantic-C++ conversion audit or paired selector has been completed.",
    paired_selector_status="absent",
    test_status="not_run",
    perf_status="not_run",
    correctness_metric="none",
    correctness_threshold="not established",
    correctness_source="none",
    silicon_status="not_run",
    silicon_result="No paired silicon measurement.",
    silicon_source="none",
)


def headers(arch):
    chip = {"bh": "blackhole", "wh": "wormhole_b0", "qsr": "quasar"}[arch]
    roots = {
        "legacy": LLK / f"tt_llk_{chip}/common/inc/sfpu",
        "metal": ROOT / f"tt_metal/hw/ckernels/{chip}",
    }
    out = {
        f"{surface}:{p.name}": p
        for surface, base in roots.items()
        for p in base.rglob("ckernel_sfpu*.h")
    }
    if arch == "qsr":
        base = LLK / "tt_llk_quasar/common/inc"
        for p in base.rglob("ckernel_sfpu*.h"):
            out[f"legacy:{p.name}"] = p
    return out


def modules(prefix):
    p = LLK / "tests/python_tests"
    return sorted(
        x.relative_to(p).as_posix()
        for x in p.rglob(f"{prefix}*.py")
        if "__pycache__" not in x.parts
    )


def classify(text):
    return (
        bool(re.search(r"\bTTI_[A-Z0-9_]+", text)),
        bool(re.search(r"\bsfpi::|__builtin_rvtt_|using namespace sfpi", text)),
        bool(re.search(r"lltt::(?:record|replay)", text)),
        bool(re.search(r"\bTTI_MOP|\bMOP\b", text)),
    )


def seed_maps():
    out = {}
    with (HERE / "f1_candidates.tsv").open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or row[0].startswith("#"):
                continue
            for path in row[4].split(","):
                name = pathlib.Path(
                    path.replace("{blackhole,wormhole_b0}", "blackhole")
                ).name
                out.setdefault(name, (row[6], row[8], row[9] + "; " + row[10]))
    return out


def inventory():
    bh, wh, qsr, seed = headers("bh"), headers("wh"), headers("qsr"), seed_maps()
    tests, perfs = modules("test_"), modules("perf_")
    rows = []
    for rel in sorted(set(bh) | set(wh) | set(qsr)):
        p = bh.get(rel) or wh.get(rel) or qsr[rel]
        raw, typed, replay, mop = classify(p.read_text(errors="replace"))
        stem = p.stem.removeprefix("ckernel_sfpu_")
        surface, shortrel = rel.split(":", 1)
        mapped = seed.get(p.name) if surface == "legacy" else None
        # Mapping is evidence, not name similarity.  Only the reviewed override
        # seed may map a header; every other header remains explicitly unmapped.
        functional = mapped[0] if mapped else ""
        perf = mapped[1] if mapped else ""
        state = "mapped" if functional else "unmapped"
        arches = ",".join(
            a for a, d in (("bh", bh), ("wh", wh), ("qsr", qsr)) if rel in d
        )
        row = dict(
            version="2",
            id=(surface + "__" + shortrel.removesuffix(".h").replace("/", "__")),
            surface=surface,
            arches=arches,
            header_bh=bh[rel].relative_to(ROOT).as_posix() if rel in bh else "",
            header_wh=wh[rel].relative_to(ROOT).as_posix() if rel in wh else "",
            header_qsr=qsr[rel].relative_to(ROOT).as_posix() if rel in qsr else "",
            raw_tti=str(int(raw)),
            typed_sfpi=str(int(typed)),
            replay=str(int(replay)),
            mop=str(int(mop)),
            functional_modules=functional,
            perf_modules=perf,
            mapping_state=state,
            notes=(
                mapped[2]
                if mapped
                else "explicitly unmapped: no audited functional module"
            ),
        )
        if row["id"] in AUDITED_MAPPINGS:
            row.update(AUDITED_MAPPINGS[row["id"]])
            row["mapping_state"] = "mapped"
        row.update(DEFAULT_AUDIT)
        row.update(AUDITED_SEEDS.get(row["id"], {}))
        rows.append(row)
    return rows


def read_manifest():
    with MANIFEST.open() as f:
        return list(
            csv.DictReader((x for x in f if not x.startswith("#")), delimiter="\t")
        )


def write_manifest(rows):
    with MANIFEST.open("w", newline="") as f:
        f.write("# sfpu-corpus-manifest-version\t2\n")
        w = csv.DictWriter(f, FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def validate(rows):
    inv = inventory()
    errors = []
    by_id = {r.get("id"): r for r in rows}
    inv_by_id = {r["id"]: r for r in inv}
    if set(by_id) != set(inv_by_id):
        errors.append(
            "manifest ID set differs from discovered inventory; audit additions/removals before --update"
        )
    for row_id in sorted(set(by_id) & set(inv_by_id)):
        for field in ["version", *DISCOVERY_FIELDS]:
            if by_id[row_id].get(field) != inv_by_id[row_id].get(field):
                errors.append(f"manifest discovery drift: {row_id}.{field}")
        row = by_id[row_id]
        if row.get("semantic_cpp_class") not in SEMANTIC_CPP_CLASSES:
            errors.append(f"bad semantic_cpp_class: {row_id}")
        if row.get("paired_selector_status") not in PAIR_STATUSES:
            errors.append(f"bad paired selector status: {row_id}")
        if row.get("test_status") not in GATE_STATUSES:
            errors.append(f"bad test status: {row_id}")
        if row.get("perf_status") not in PERF_STATUSES:
            errors.append(f"bad perf status: {row_id}")
        if row.get("correctness_metric") not in CORRECTNESS_METRICS:
            errors.append(f"bad correctness metric: {row_id}")
        if row.get("silicon_status") not in SILICON_STATUSES:
            errors.append(f"bad silicon status: {row_id}")
        if not row.get("semantic_cpp_blocker"):
            errors.append(f"missing exact blocker/readiness statement: {row_id}")
        if (
            row.get("mapping_state") == "mapped"
            and row.get("semantic_cpp_class") == "unmapped"
        ):
            errors.append(f"mapped row lacks semantic-C++ audit: {row_id}")
        if row.get("silicon_status") in {"win", "parity", "loss"}:
            if (
                row.get("test_status") != "pass"
                or row.get("correctness_metric") == "none"
            ):
                errors.append(f"ungated silicon result: {row_id}")
            if row.get("perf_status") != "measured":
                errors.append(f"silicon result without measured perf: {row_id}")
    counts = {"logical": len(inv)}
    for a in ("bh", "wh", "qsr"):
        counts[a] = sum(a in r["arches"].split(",") for r in inv)
    for a in ("bh", "wh", "qsr"):
        counts[f"legacy_{a}"] = sum(
            r["surface"] == "legacy" and a in r["arches"].split(",") for r in inv
        )
    counts["physical_paths"] = counts["bh"] + counts["wh"] + counts["qsr"]
    counts["basename_union"] = len(
        {
            pathlib.Path(
                next(r[x] for x in ("header_bh", "header_wh", "header_qsr") if r[x])
            ).name
            for r in inv
        }
    )
    for key in ("raw", "typed", "replay", "mop"):
        col = {
            "raw": "raw_tti",
            "typed": "typed_sfpi",
            "replay": "replay",
            "mop": "mop",
        }[key]
        counts[key] = sum(r[col] == "1" for r in inv)
    for k, v in EXPECTED.items():
        if counts[k] != v:
            errors.append(f"inventory drift: {k} expected {v}, found {counts[k]}")
    wh = set(headers("wh"))
    bh = set(headers("bh"))
    if not wh <= bh:
        errors.append("Wormhole headers are not a subset of Blackhole")
    for r in rows:
        if r["mapping_state"] not in ("mapped", "unmapped"):
            errors.append(f"bad mapping state: {r['id']}")
    return errors, counts


def sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def artifact_manifest(root, suffixes=(".elf",)):
    """Return stable content hashes keyed relative to one isolated build root."""
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    }


def elf_text_manifest(root, objcopy, output, run=subprocess.run):
    """Hash executable .text, excluding build-root-dependent debug/provenance data."""
    manifest = {}
    errors = {}
    for elf in sorted(root.rglob("*.elf")) if root.is_dir() else ():
        rel = elf.relative_to(root)
        text_bin = output / rel.parent / (rel.name + ".text.bin")
        text_bin.parent.mkdir(parents=True, exist_ok=True)
        result = run(
            [
                str(objcopy),
                "-O",
                "binary",
                "--only-section=.text",
                str(elf),
                str(text_bin),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode or not text_bin.is_file():
            errors[str(rel)] = {
                "returncode": result.returncode,
                "output": result.stdout or "",
            }
        else:
            manifest[str(rel)] = sha(text_bin)
    return manifest, errors


def classify_artifact_pair(off, on):
    paths = sorted(set(off) | set(on))
    changed = [path for path in paths if off.get(path) != on.get(path)]
    return {
        "status": "CHANGED_BINARY" if changed else "BYTE_IDENTICAL",
        "changed_artifacts": changed,
        "off_artifact_count": len(off),
        "on_artifact_count": len(on),
    }


def split_selectors(value):
    return [x for x in value.split(",") if x and " " not in x]


def row_selectors(row, mode):
    functional = split_selectors(row["functional_modules"])
    if mode != "silicon":
        return functional
    return sorted(set(functional + split_selectors(row["perf_modules"])))


def expected_sfpi_version(path=None):
    path = path or ROOT / "tt_metal/sfpi-version"
    if not path.is_file():
        return ""
    match = re.search(r"^sfpi_version=['\"]([^'\"]+)", path.read_text(), re.MULTILINE)
    return match.group(1) if match else ""


def compiler_preflight(
    compiler, arch, capabilities, run, installed_version=None, expected_version=None
):
    installed_version = installed_version or LLK / "tests/sfpi/sfpi.version"
    expected_version = expected_version or ROOT / "tt_metal/sfpi-version"
    expected = expected_sfpi_version(expected_version)
    installed = (
        installed_version.read_text().strip() if installed_version.is_file() else ""
    )
    result = {
        "compiler": str(compiler),
        "compiler_realpath": "",
        "compiler_sha256": "",
        "compiler_version": "",
        "compiler_version_returncode": None,
        "expected_sfpi_version": expected,
        "installed_sfpi_version": installed,
        "pin_match": bool(expected and installed and expected == installed),
        "capabilities": {},
    }
    if not compiler.is_file():
        result["status"] = "MISSING_COMPILER"
        return result
    result["compiler_realpath"] = str(compiler.resolve())
    result["compiler_sha256"] = sha(compiler)
    version = run(
        [str(compiler), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result["compiler_version_returncode"] = version.returncode
    result["compiler_version"] = (
        (version.stdout or "").splitlines()[0] if version.stdout else ""
    )
    if version.returncode:
        result["status"] = "COMPILER_ERROR"
        return result
    mcpu = {"bh": "tt-bh-tensix", "wh": "tt-wh-tensix", "qsr": "tt-qsr32-tensix"}[arch]
    for capability in sorted(capabilities):
        source = COMPILER_CAPABILITY_PROBES[capability]
        probe = run(
            [
                str(compiler),
                f"-mcpu={mcpu}",
                "-std=c++17",
                "-fsyntax-only",
                "-x",
                "c++",
                "-",
            ],
            input=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        result["capabilities"][capability] = {
            "available": probe.returncode == 0,
            "returncode": probe.returncode,
            "output": probe.stdout or "",
        }
    result["status"] = "PASS" if result["pin_match"] else "PIN_MISMATCH"
    return result


def missing_row_capabilities(row_id, preflight):
    return [
        cap
        for cap in ROW_REQUIRED_CAPABILITIES.get(row_id, ())
        if not preflight.get("capabilities", {}).get(cap, {}).get("available", False)
    ]


def pytest_report_command(python, selectors, extra, report, collect_only=False):
    cmd = [
        str(python),
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-p",
        PYTEST_PLUGIN,
        *selectors,
        *extra,
        "-q",
    ]
    if collect_only:
        cmd.append("--collect-only")
    return cmd


def invoke_pytest_report(
    python,
    cwd,
    selectors,
    extra,
    report,
    log,
    env,
    collect_only=False,
    run=subprocess.run,
):
    child_env = env.copy()
    child_env["SFPU_CORPUS_PYTEST_REPORT"] = str(report)
    prior = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = str(HERE) + (os.pathsep + prior if prior else "")
    cmd = pytest_report_command(python, selectors, extra, report, collect_only)
    with log.open("w") as f:
        rc = run(
            cmd, cwd=cwd, env=child_env, stdout=f, stderr=subprocess.STDOUT
        ).returncode
    payload = (
        json.loads(report.read_text())
        if report.is_file()
        else {
            "schema": 1,
            "exitstatus": rc,
            "collected": [],
            "collection_errors": [],
            "reports": {},
        }
    )
    return rc, payload


def terminal_outcome(phases):
    failed = [when for when, data in phases.items() if data.get("outcome") == "failed"]
    if failed:
        return "FAIL" if failed == ["call"] else "ERROR"
    call = phases.get("call", {})
    if call.get("outcome") == "passed":
        return "PASS"
    if call.get("outcome") == "skipped" or any(
        x.get("outcome") == "skipped" for x in phases.values()
    ):
        return "SKIP"
    return "NOT_RUN"


def attribute_pytest_row(rec, nodeids, reports, reason, artifact):
    outcomes = {
        nodeid: terminal_outcome(reports.get(nodeid, {})) for nodeid in sorted(nodeids)
    }
    counts = {
        status: sum(value == status for value in outcomes.values())
        for status in ("PASS", "FAIL", "ERROR", "SKIP", "NOT_RUN")
    }
    failing = [
        nodeid for nodeid, status in outcomes.items() if status in {"FAIL", "ERROR"}
    ]
    missing = [nodeid for nodeid, status in outcomes.items() if status == "NOT_RUN"]
    rec.update(
        nodeids=sorted(nodeids),
        outcome_counts=counts,
        failing_nodeids=failing,
        missing_nodeids=missing,
        artifact=str(artifact),
        reason=reason,
    )
    if not nodeids:
        rec["status"] = "ERROR_NO_COLLECTED_TESTS"
    elif failing:
        rec["status"] = "FAIL"
    elif missing:
        rec["status"] = "ERROR_NOT_RUN"
    elif counts["PASS"]:
        rec["status"] = "PASS"
    else:
        rec["status"] = "SKIP_ALL_TESTS"


def emit_summary(run, records, provenance):
    (run / "results.json").write_text(
        json.dumps({"provenance": provenance, "results": records}, indent=2) + "\n"
    )
    with (run / "results.tsv").open("w", newline="") as f:
        keys = [
            "id",
            "arch",
            "mode",
            "status",
            "reason",
            "artifact",
            "selectors",
            "nodeids",
            "outcome_counts",
            "failing_nodeids",
            "missing_nodeids",
            *AUDIT_FIELDS,
        ]
        keys += [
            "compiler_ab_status",
            "changed_artifacts",
            "off_artifact_count",
            "on_artifact_count",
            "compiler_ab_comparison_scope",
            "compiler_ab_artifact",
        ]
        w = csv.DictWriter(
            f, keys, delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(records)
    lines = [
        "# SFPU corpus run",
        "",
        f"- mode: `{provenance['mode']}`",
        f"- revision: `{provenance['tt_metal_head']}`",
        "",
        "| id | arch | status | semantic C++ | correctness gate | silicon | reason |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['id']} | {r['arch']} | {r['status']} | {r.get('semantic_cpp_class','')} | {r.get('correctness_metric','')}: {r.get('correctness_threshold','')} | {r.get('silicon_status','')}: {r.get('silicon_result','')} | {r['reason']} |"
        for r in records
    ]
    (run / "summary.md").write_text("\n".join(lines) + "\n")


def load_baseline(path):
    if path.suffix == ".json":
        return json.loads(path.read_text()).get("results", [])
    with path.open() as f:
        return list(
            csv.DictReader(
                (line for line in f if not line.startswith("#")), delimiter="\t"
            )
        )


def compare_baseline(records, baseline, threshold):
    old = load_baseline(baseline)
    key = lambda r: (
        r.get("id"),
        r.get("arch"),
        r.get("metric"),
        r.get("scope"),
        r.get("selector"),
    )
    samples = {}
    for row in old:
        try:
            cycles = float(row.get("cycles", ""))
        except (TypeError, ValueError):
            continue
        samples.setdefault(key(row), []).append(cycles)
    index = {k: min(v) for k, v in samples.items()}
    compared = []
    for r in records:
        before = index.get(key(r))
        now = r.get("cycles")
        if (
            not isinstance(now, (int, float))
            or not isinstance(before, (int, float))
            or before == 0
        ):
            compared.append(
                {
                    "id": r["id"],
                    "status": "SKIP_NO_DEVICE_CYCLES",
                    "reason": "both runs need numeric device cycles",
                }
            )
            continue
        delta = 100.0 * (now - before) / before
        compared.append(
            {
                "id": r["id"],
                "status": "REGRESSION" if delta > threshold else "PASS",
                "delta_pct": delta,
            }
        )
    return compared


def emit_plan(rows, arch, fmt):
    keys = [
        "id",
        "arches",
        "mapping_state",
        "functional_modules",
        "perf_modules",
        *AUDIT_FIELDS,
    ]
    if fmt == "json":
        print(
            json.dumps(
                {
                    "schema": 2,
                    "arch": arch,
                    "rows": [{k: r.get(k, "") for k in keys} for r in rows],
                },
                indent=2,
            )
        )
    elif fmt == "markdown":
        print(
            "| id | arches | semantic C++ | selector | test | perf | correctness | silicon |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(
                f"| {r['id']} | {r['arches']} | {r['semantic_cpp_class']} | {r['paired_selector_status']} | {r['test_status']} | {r['perf_status']} | {r['correctness_metric']}: {r['correctness_threshold']} | {r['silicon_status']}: {r['silicon_result']} |"
            )
    else:
        for r in rows:
            print("\t".join(r.get(k, "") for k in keys))


def record(row, arch, mode, status, reason, artifact=""):
    return {
        "id": row["id"],
        "arch": arch,
        "mode": mode,
        "status": status,
        "reason": reason,
        "artifact": artifact,
        **{k: row.get(k, "") for k in AUDIT_FIELDS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mode", choices=("compile", "craq", "silicon"))
    ap.add_argument("--arch", choices=("bh", "wh", "qsr"), default="bh")
    ap.add_argument(
        "--row", action="append", default=[], help="exact corpus row id (repeatable)"
    )
    ap.add_argument("--plan-format", choices=("tsv", "json", "markdown"), default="tsv")
    ap.add_argument("--run-root", type=pathlib.Path)
    ap.add_argument("--simulator", type=pathlib.Path)
    ap.add_argument("--baseline", type=pathlib.Path)
    ap.add_argument(
        "--chip-class",
        choices=sorted(DEVICE_BASELINES),
        help="select the checked-in device baseline for this chip class when --baseline is not given explicitly",
    )
    ap.add_argument("--max-regression-pct", type=float, default=0.0)
    ap.add_argument(
        "--execute",
        action="store_true",
        help="execute the selected mode (otherwise emit a plan)",
    )
    ap.add_argument(
        "--require-executed-mapped",
        action="store_true",
        help="fail unless at least one mapped row executed and every mapped row passed",
    )
    ap.add_argument(
        "--require-compiler-pin",
        action="store_true",
        help="block execution when the installed SFPI version differs from tt_metal/sfpi-version",
    )
    ap.add_argument(
        "--compiler-ab-off-options",
        help="compile-mode A/B: options for the control build",
    )
    ap.add_argument(
        "--compiler-ab-on-options",
        help="compile-mode A/B: options for the candidate build",
    )
    ap.add_argument(
        "--require-changed-binary",
        action="store_true",
        help="fail compiler A/B rows whose ELF sets are byte-identical",
    )
    ap.add_argument("--allow-hardware", action="store_true")
    ap.add_argument(
        "--hardware-lock",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/tt-llk-sfpu-silicon.lock"),
    )
    ap.add_argument(
        "--measurements",
        type=pathlib.Path,
        help="silicon TSV: id,arch,metric,scope,selector,cycles",
    )
    ap.add_argument(
        "--compare-results",
        type=pathlib.Path,
        help="compare an existing results.json to --baseline",
    )
    a = ap.parse_args()
    rows = inventory()
    if a.baseline is None and a.chip_class:
        a.baseline = DEVICE_BASELINES[a.chip_class]
    compiler_ab = (
        a.compiler_ab_off_options is not None or a.compiler_ab_on_options is not None
    )
    if compiler_ab and (a.mode != "compile" or not a.execute):
        ap.error("compiler A/B requires --mode compile --execute")
    if compiler_ab and (
        a.compiler_ab_off_options is None or a.compiler_ab_on_options is None
    ):
        ap.error(
            "compiler A/B requires both --compiler-ab-off-options and --compiler-ab-on-options"
        )
    if a.require_changed_binary and not compiler_ab:
        ap.error("--require-changed-binary requires compiler A/B options")
    if a.compare_results:
        if not a.baseline:
            ap.error("--compare-results requires --baseline")
        results = json.loads(a.compare_results.read_text()).get("results", [])
        out = compare_baseline(results, a.baseline, a.max_regression_pct)
        print(json.dumps(out, indent=2))
        return int(any(x["status"] == "REGRESSION" for x in out))
    if a.update:
        # Discovery refreshes paths/features/mappings, but reviewed semantic
        # audit fields are durable data and must survive regeneration.
        if MANIFEST.exists():
            reviewed = {r["id"]: r for r in read_manifest()}
            for row in rows:
                if row["id"] in reviewed:
                    row.update(
                        {k: reviewed[row["id"]].get(k, "") for k in AUDIT_FIELDS}
                    )
                row.update(AUDITED_SEEDS.get(row["id"], {}))
        write_manifest(rows)
    current = read_manifest() if MANIFEST.exists() else []
    errors, counts = validate(current)
    if a.validate or a.update:
        print(json.dumps({"counts": counts, "errors": errors}, sort_keys=True))
    if errors and (a.validate or a.mode):
        return 1
    requested = set(a.row)
    unknown = requested - {r["id"] for r in current}
    if unknown:
        ap.error("unknown corpus row(s): " + ",".join(sorted(unknown)))
    selected = [
        r
        for r in current
        if a.arch in r["arches"].split(",") and (not requested or r["id"] in requested)
    ]
    if a.list:
        emit_plan(selected, a.arch, a.plan_format)
    if not a.mode:
        return 0
    run = a.run_root or HERE / "runs" / (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{a.arch}-{a.mode}"
    )
    run.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    prov = {
        "schema": 2,
        "mode": a.mode,
        "arch": a.arch,
        "tt_metal_head": head,
        "manifest_sha256": sha(MANIFEST),
        "simulator": str(a.simulator or ""),
        "threshold_pct": a.max_regression_pct,
        "hardware_lock": str(a.hardware_lock),
    }
    if compiler_ab:
        prov["compiler_ab"] = {
            "off_options": a.compiler_ab_off_options,
            "on_options": a.compiler_ab_on_options,
            "require_changed_binary": a.require_changed_binary,
        }
    records = []
    for r in selected:
        selectors = row_selectors(r, a.mode)
        if not selectors:
            records.append(
                record(r, a.arch, a.mode, "SKIP_UNMAPPED", "no audited module mapping")
            )
            continue
        if a.mode == "silicon" and (
            r["paired_selector_status"] != "implemented"
            or r["test_status"] != "pass"
            or r["correctness_metric"] == "none"
        ):
            records.append(
                record(
                    r,
                    a.arch,
                    a.mode,
                    "SKIP_CORRECTNESS_NOT_GATED",
                    "silicon requires an implemented selector and an explicit passing correctness metric",
                )
            )
            continue
        if a.mode == "craq" and (not a.simulator or not a.simulator.is_file()):
            records.append(
                record(r, a.arch, a.mode, "SKIP_NO_SIMULATOR", "--simulator required")
            )
            continue
        if a.mode == "silicon" and (not a.execute or not a.allow_hardware):
            records.append(
                record(
                    r,
                    a.arch,
                    a.mode,
                    "SKIP_HARDWARE_NOT_AUTHORIZED",
                    "requires --execute --allow-hardware",
                )
            )
            continue
        if not a.execute:
            status = (
                "SKIP_HARDWARE_NOT_AUTHORIZED" if a.mode == "silicon" else "PLAN_ONLY"
            )
            records.append(record(r, a.arch, a.mode, status, ",".join(selectors)))
            continue
        rec = record(r, a.arch, a.mode, "QUEUED", "pytest collection pending")
        rec["selectors"] = selectors
        records.append(rec)
    queued = [r for r in records if r["status"] == "QUEUED"]
    if queued:
        pydir = LLK / "tests/python_tests"
        python = pydir / ".venv/bin/python"
        log = run / f"{a.mode}.log"
        env = os.environ.copy()
        env.update(
            {
                "TT_METAL_HOME": str(ROOT),
                "SHORT_ARCH": a.arch,
                "SIM_ARCH": {"bh": "blackhole", "wh": "wormhole", "qsr": "quasar"}[
                    a.arch
                ],
            }
        )
        if not python.is_file():
            prov["pytest_returncode"] = None
            for rec in queued:
                rec["status"] = "SKIP_MISSING_ENV"
                rec["reason"] = "missing tt-llk .venv"
        else:
            compiler = LLK / "tests/sfpi/compiler/bin/riscv-tt-elf-g++"
            required = {
                cap
                for rec in queued
                for cap in ROW_REQUIRED_CAPABILITIES.get(rec["id"], ())
            }
            preflight = compiler_preflight(compiler, a.arch, required, subprocess.run)
            preflight_path = run / "compiler-preflight.json"
            preflight_path.write_text(
                json.dumps(preflight, indent=2, sort_keys=True) + "\n"
            )
            prov["compiler_preflight"] = {
                k: v for k, v in preflight.items() if k != "capabilities"
            }
            prov["compiler_preflight"]["artifact"] = str(preflight_path)
            prov["compiler_preflight"]["capabilities"] = {
                name: {"available": data["available"], "returncode": data["returncode"]}
                for name, data in preflight["capabilities"].items()
            }
            for rec in queued:
                missing = missing_row_capabilities(rec["id"], preflight)
                if preflight["status"] in {"MISSING_COMPILER", "COMPILER_ERROR"}:
                    rec["status"] = "BLOCKED_COMPILER_PREFLIGHT"
                    rec["reason"] = "SFPI compiler preflight failed"
                elif a.require_compiler_pin and not preflight["pin_match"]:
                    rec["status"] = "BLOCKED_COMPILER_PIN"
                    rec["reason"] = (
                        "installed SFPI version does not match tt_metal/sfpi-version"
                    )
                elif missing:
                    rec["status"] = "BLOCKED_COMPILER_CAPABILITY"
                    rec["reason"] = "missing compiler capabilities: " + ",".join(
                        missing
                    )
                if rec["status"] != "QUEUED":
                    rec["artifact"] = str(preflight_path)

            collection_rcs = {}
            row_nodes = {}
            collect_extra = ["--compile-producer"] if a.mode == "compile" else []
            for rec in [x for x in queued if x["status"] == "QUEUED"]:
                stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", rec["id"])
                report = run / f"collect-{stem}.json"
                collect_log = run / f"collect-{stem}.log"
                rc, payload = invoke_pytest_report(
                    python,
                    pydir,
                    rec["selectors"],
                    collect_extra,
                    report,
                    collect_log,
                    env,
                    collect_only=True,
                )
                collection_rcs[rec["id"]] = rc
                nodeids = set(payload.get("collected", []))
                row_nodes[rec["id"]] = nodeids
                rec["nodeids"] = sorted(nodeids)
                if rc != 0 or payload.get("collection_errors") or not nodeids:
                    rec["status"] = "ERROR_COLLECTION"
                    rec["reason"] = "row selector collection failed"
                    rec["artifact"] = str(collect_log)
            prov["collection_returncodes"] = collection_rcs

            runnable = [x for x in queued if x["status"] == "QUEUED"]
            mods = sorted(
                {selector for rec in runnable for selector in rec["selectors"]}
            )
            if runnable and compiler_ab:
                ab_returncodes = {}
                for rec in runnable:
                    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", rec["id"])
                    variants = {}
                    for label, options in (
                        ("off", a.compiler_ab_off_options),
                        ("on", a.compiler_ab_on_options),
                    ):
                        variant_root = run / "compiler-ab" / stem / label
                        variant_root.mkdir(parents=True)
                        variant_env = env.copy()
                        variant_env["RUNNER_TEMP"] = str(variant_root)
                        variant_env["TT_LLK_EXTRA_COMPILER_OPTIONS"] = options
                        report = variant_root / "pytest.json"
                        variant_log = variant_root / "pytest.log"
                        rc, payload = invoke_pytest_report(
                            python,
                            pydir,
                            rec["selectors"],
                            ["--compile-producer"],
                            report,
                            variant_log,
                            variant_env,
                        )
                        outcome = {"status": "QUEUED"}
                        attribute_pytest_row(
                            outcome,
                            row_nodes[rec["id"]],
                            payload.get("reports", {}),
                            f"compiler A/B {label} compile",
                            variant_log,
                        )
                        build_root = variant_root / "tt-llk-build"
                        objcopy = compiler.with_name("riscv-tt-elf-objcopy")
                        manifest, manifest_errors = elf_text_manifest(
                            build_root, objcopy, variant_root / "text"
                        )
                        (variant_root / "elf-manifest.json").write_text(
                            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                        )
                        (variant_root / "elf-manifest-errors.json").write_text(
                            json.dumps(manifest_errors, indent=2, sort_keys=True) + "\n"
                        )
                        variants[label] = {
                            "returncode": rc,
                            "outcome": outcome["status"],
                            "report": str(report),
                            "log": str(variant_log),
                            "manifest": manifest,
                            "manifest_errors": manifest_errors,
                        }
                    ab_returncodes[rec["id"]] = {
                        k: v["returncode"] for k, v in variants.items()
                    }
                    pair = classify_artifact_pair(
                        variants["off"]["manifest"], variants["on"]["manifest"]
                    )
                    rec.update(
                        compiler_ab_status=pair["status"],
                        changed_artifacts=pair["changed_artifacts"],
                        off_artifact_count=pair["off_artifact_count"],
                        on_artifact_count=pair["on_artifact_count"],
                        compiler_ab_comparison_scope="ELF .text",
                        compiler_ab_artifact=str(run / "compiler-ab" / stem),
                    )
                    if (
                        variants["off"]["outcome"] != "PASS"
                        or variants["on"]["outcome"] != "PASS"
                    ):
                        rec.update(
                            status="FAIL",
                            reason="compiler A/B compile/correctness gate failed",
                            artifact=str(run / "compiler-ab" / stem),
                        )
                    elif (
                        variants["off"]["manifest_errors"]
                        or variants["on"]["manifest_errors"]
                        or not variants["off"]["manifest"]
                        or not variants["on"]["manifest"]
                    ):
                        rec.update(
                            status="ERROR_NOT_RUN",
                            reason="compiler A/B could not extract ELF .text",
                            artifact=str(run / "compiler-ab" / stem),
                        )
                    else:
                        rec.update(
                            status="PASS",
                            reason="compiler A/B " + pair["status"],
                            artifact=str(run / "compiler-ab" / stem),
                        )
                        if (
                            a.require_changed_binary
                            and pair["status"] != "CHANGED_BINARY"
                        ):
                            rec.update(
                                status="FAIL",
                                reason="compiler A/B required a changed ELF",
                            )
                prov["compiler_ab_returncodes"] = ab_returncodes
            elif runnable and a.mode in {"compile", "silicon"}:
                extra = ["--compile-producer"] if a.mode == "compile" else []
                report = run / f"{a.mode}-pytest.json"
                if a.mode == "silicon":
                    import fcntl

                    with a.hardware_lock.open("w") as lk:
                        fcntl.flock(lk, fcntl.LOCK_EX)
                        rc, payload = invoke_pytest_report(
                            python, pydir, mods, extra, report, log, env
                        )
                    why = "serialized correctness-plus-silicon gate"
                else:
                    rc, payload = invoke_pytest_report(
                        python, pydir, mods, extra, report, log, env
                    )
                    why = "compile gate"
                prov["pytest_returncode"] = rc
                prov["pytest_report"] = str(report)
                for rec in runnable:
                    attribute_pytest_row(
                        rec, row_nodes[rec["id"]], payload.get("reports", {}), why, log
                    )
            elif runnable:
                runner = (
                    pathlib.Path(
                        os.environ.get("CRAQ_SIM_ROOT", "/localdev/nkapre/craq-sim")
                    )
                    / "scripts/perf/llk-sim-perf.sh"
                )
                cmd = [
                    str(runner),
                    "--sample",
                    "1",
                    "--run-root",
                    str(run / "craq"),
                ] + sum((["--module", m] for m in mods), [])
                env["SIMULATOR"] = str(a.simulator)
                with log.open("w") as f:
                    rc = subprocess.run(
                        cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT
                    ).returncode
                prov["pytest_returncode"] = rc
                metric = run / "craq/llk_sim.tsv"
                measured = []
                if metric.is_file():
                    with metric.open() as f:
                        measured = list(csv.DictReader(f, delimiter="\t"))
                for rec in runnable:
                    hits = [
                        x
                        for x in measured
                        if x.get("nodeid") in row_nodes[rec["id"]]
                        and x.get("simulated_cycles")
                    ]
                    if hits:
                        rec.update(
                            status="PASS",
                            reason="CRAQ modeled-cycle/functional gate",
                            artifact=str(log),
                            metric="simulated_cycles",
                            scope="craq_program",
                            selector="default",
                            cycles=max(float(x["simulated_cycles"]) for x in hits),
                            measured_nodeids=sorted({x["nodeid"] for x in hits}),
                        )
                    else:
                        rec.update(
                            status="ERROR_NOT_RUN",
                            reason="mapped CRAQ row produced no exact-nodeid modeled-cycle metric",
                            artifact=str(log),
                        )

            if a.mode == "silicon":
                measured = []
                if a.measurements and a.measurements.is_file():
                    with a.measurements.open() as f:
                        measured = list(csv.DictReader(f, delimiter="\t"))
                for rec in runnable:
                    if rec["status"] != "PASS":
                        continue
                    hits = [
                        x
                        for x in measured
                        if x.get("id") == rec["id"]
                        and x.get("arch") == a.arch
                        and x.get("cycles")
                    ]
                    if hits:
                        x = hits[-1]
                        rec.update(
                            metric=x["metric"],
                            scope=x["scope"],
                            selector=x["selector"],
                            cycles=float(x["cycles"]),
                        )
                    else:
                        rec["status"] = "FAIL"
                        rec["reason"] = (
                            "mapped silicon row produced no scoped device-cycle metric"
                        )
    failed = False
    if a.require_executed_mapped:
        mapped = [
            r
            for r in records
            if next(x for x in selected if x["id"] == r["id"])["functional_modules"]
        ]
        if not mapped or any(r["status"] != "PASS" for r in mapped):
            failed = True
            prov["executed_mapped_gate"] = "FAIL"
        else:
            prov["executed_mapped_gate"] = "PASS"
    if compiler_ab and a.require_changed_binary:
        failed = failed or any(
            r.get("compiler_ab_status") != "CHANGED_BINARY"
            for r in records
            if r.get("selectors")
        )
    if a.baseline:
        comparisons = compare_baseline(records, a.baseline, a.max_regression_pct)
        (run / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
        prov["baseline"] = str(a.baseline)
        failed = failed or any(x["status"] == "REGRESSION" for x in comparisons)
    emit_summary(run, records, prov)
    print(run)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
