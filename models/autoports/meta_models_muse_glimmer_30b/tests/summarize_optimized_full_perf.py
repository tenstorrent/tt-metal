# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage7 timing and capacity arithmetic; never opens devices."""
import csv
import json
import math
import statistics
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL / "doc/optimized_full_model"
full = json.loads((ROOT / "final.json").read_text())
before = json.loads((ROOT / "baseline_extended.json").read_text())
profile = json.loads((ROOT / "tracy_reduced/perf_run.json").read_text())
iterations = profile["iterations"]


def numeric(row, key):
    value = row[key]
    return float(value) if value and value != "nan" else 0.0


def total(rows):
    return dict(
        classification="measured reduced kernel sum",
        kernel_ms=sum(numeric(r, "Device Time") for r in rows) / iterations / 1000,
        gap_ms=sum(numeric(r, "Op-to-Op Gap") for r in rows) / iterations / 1000,
    )


def medians(run, field):
    return {
        mode: {
            k: statistics.median(r[k] for r in run[field] if r["mode"] == mode)
            for k in ("ttft_ms", "decode_ms", "decode_tps")
        }
        for mode in ("sample", "argmax")
    }


phases = {}
for phase in ("prefill", "decode", "sample", "argmax", "token_out", "token_out_no_readback", "reset"):
    rows = list(csv.DictReader((ROOT / f"tracy_reduced/{phase}_perf_report.csv").open()))
    phases[phase] = dict(total(rows), host_ms=profile[phase + "_wall_ms"])
    if phase == "decode":
        decode = rows
assert len(decode) == iterations * 96, len(decode)
segments = {}
for name, lo, hi in (
    ("embedding_rope", 0, 8),
    ("sliding", 8, 49),
    ("full", 49, 87),
    ("terminal", 87, 96),
    ("lm_head", 88, 89),
):
    rows = [r for i in range(iterations) for r in decode[i * 96 + lo : i * 96 + hi]]
    segments[name] = total(rows)

head_bytes = 208 * 198 * 8 * 1088
layer_bytes = 69636096
kv_bytes = statistics.mean(math.ceil(n / 32) * 2 * 4 * 1088 * 52 for n in range(129, 256))
norm_bytes = 52 * 4 * 6656 * 2
byte_count = 52 * layer_bytes + head_bytes + kv_bytes + norm_bytes + 6656 * 2 / 4
scaled = (
    segments["sliding"]["kernel_ms"] * 39
    + segments["full"]["kernel_ms"] * 13
    + segments["terminal"]["kernel_ms"]
    + segments["embedding_rope"]["kernel_ms"]
    + phases["sample"]["kernel_ms"]
)
primary = medians(full, "token_out_no_readback")
result = dict(
    workload=dict(profile="single_user_decode", prompt_len=128, gen_len=128, batch=1),
    ttft_ms=primary["sample"]["ttft_ms"],
    decode_ms_per_token_e2e=primary["sample"]["decode_ms"],
    decode_ms_per_token_device=None,
    roofline_ms_per_token_estimate=byte_count / 512e9 * 1000,
    named_limitations=[
        "Full-stack profiler is prohibited; device rows are measured on real layers0/3 and extrapolated explicitly.",
        "Device op dispatch and the 52-layer selected numerical policy remain material.",
        "LM-head K1 wins legal whole-terminal candidates; larger legal splits add weight traffic/launches.",
    ],
    before=dict(output_returning=medians(before, "perf"), no_readback=medians(before, "token_out_no_readback")),
    after=dict(output_returning=medians(full, "perf"), no_readback=primary),
    readiness_teacher_forcing_sampling_inclusive=full["teacher"][0],
    traced_logits_only=full["logits_only"],
    reduced_phases=phases,
    reduced_segments=segments,
    modeled_full_kernel_ms=scaled,
    modeled_roofline=dict(
        bytes_per_chip=byte_count,
        bandwidth_bytes_per_second_per_chip=512e9,
        head_weight_bytes_per_chip=head_bytes,
        cache_read_bytes_per_chip=kv_bytes,
    ),
    layer_stack_bound=dict(
        ms=39 * 0.320355 + 13 * 0.309949,
        classification="modeled sum of prior measured standalone layer traces",
        caveat="Standalone contexts33/1024; current full model128..254 in one trace, so standalone replay gaps are not additive.",
    ),
    terminal_policy=full["terminal_policy"],
    clear_cache_on_generate=full.get("clear_cache_on_generate", True),
    profile_graph_provenance="profile_graph_provenance.json",
    source="final.json; baseline_extended.json; tracy_reduced/",
)
(ROOT / "perf_summary.json").write_text(json.dumps(result, indent=2) + "\n")
capacity = json.loads((MODEL / "doc/full_model/memory_capacity_plan.json").read_text())
terms = capacity["per_device_terms"]
old = terms.pop("lm_head_dram_sharded_bf16")
terms["lm_head_dram_sharded_bfp8"] = head_bytes
# Three one-tile uint32 DRAM buffers; reserve one 4KiB page on all eight banks.
admission_reservation = 3 * 4096 * 8
terms["admission_snapshots_dram_upper_bound"] = admission_reservation
for field, sign in (("per_device_default_total_bytes", 1), ("per_device_default_margin_bytes", -1)):
    capacity[field] += sign * (head_bytes - old + admission_reservation)
capacity["measured_allocator"] = dict(
    classification="measured", evidence="optimized_full_model/final.json", views=full["allocator"]
)
capacity["terminal_policy"] = full["terminal_policy"]
capacity["trace_count"] = len(full["traces"])
capacity["modeled_max_pages_with_reservations"] = (capacity["per_device_dram_bytes"] - sum(terms.values())) // capacity[
    "per_device_kv_bytes_per_page_all_layers"
]
capacity["modeled_full_context_concurrency"] = capacity["modeled_max_pages_with_reservations"] // 4096
capacity["context_evidence"] = "optimized_full_model/context.json"
capacity["batch_evidence"] = "optimized_full_model/batch32.json"
capacity[
    "change_scope"
] = "LM-head weight storage shrinks; KV/context/activation/residual policy unchanged. Three admission uint32 tiles reserve at most96KiB per chip (modeled bank-granularity upper bound). Reset and two admission traces fit within existing1GiB reservation."
(ROOT / "memory_capacity_plan.json").write_text(json.dumps(capacity, indent=2) + "\n")
path = MODEL / "doc/context_contract.json"
contract = json.loads(path.read_text())
contract["optimized_full_model_stage_context_preservation"] = capacity
path.write_text(json.dumps(contract, indent=2) + "\n")
print(
    json.dumps(
        {"before": result["before"], "after": result["after"], "roofline": result["roofline_ms_per_token_estimate"]},
        indent=2,
    )
)
