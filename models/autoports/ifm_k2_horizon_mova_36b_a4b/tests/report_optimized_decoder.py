# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build final CPU-only acceptance summaries from measured artifacts; reject stale rows."""

import csv
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_decoder"
SHA = hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (DOC / name).write_text(json.dumps(value, indent=2) + "\n")


def stored_bytes(contract, copies=1):
    k, n = contract["shape"][-2:]
    tile = {"DataType.BFLOAT4_B": 576, "DataType.BFLOAT8_B": 1088, "DataType.BFLOAT16": 2048, "DataType.FLOAT32": 4096}[
        contract["dtype"]
    ]
    return copies * math.ceil(k / 32) * math.ceil(n / 32) * tile


def main():
    wall = []
    for layer in (0, 3):
        for seq in (32, 256, 2048, 2049):
            a = read(DOC / "candidates" / f"final_baseline_s{seq}_layer{layer}.json")
            b = read(DOC / "candidates" / f"final_default_s{seq}_layer{layer}.json")
            assert b["source_sha256"] == SHA
            for tag in ("baseline", "default"):
                execution = read(DOC / f"final_{tag}_s{seq}_l{layer}_execution.json")
                assert execution["exit_code"] == 0
            assert a["activation_source"] == b["activation_source"] == "checkpoint-HF"
            assert a["decode_position"] == b["decode_position"] == seq
            assert b["implementation"].endswith("tt.optimized_decoder")
            wall.append(
                dict(
                    layer=layer,
                    seq=seq,
                    optimized_physical_chunk=2048,
                    optimized_chunks=math.ceil(seq / 2048),
                    baseline=a,
                    optimized=b,
                )
            )
    correctness = []
    for path in sorted(DOC.glob("validation_layer*_*.json")):
        d = read(path)
        if d.get("decoder_sha256") != SHA:
            continue
        if not any(path.stem.endswith("_" + t) for t in ("final", "watcher", "bf16cache", "pytest")):
            continue
        assert d["trace_alloc_tracking"] == "1" and d["program_cache_tracking"]
        assert d["captures"] == 1 and d["request_time_releases"] == 0 and d["teardown_releases"] == 1
        assert d["cache_ownership_and_padding_unchanged"] and all(r["deterministic"] for r in d["rows"])
        row = d["rows"][0]
        correctness.append(
            dict(
                artifact=path.name,
                layer=row["layer"],
                batch=row["batch"],
                page=row["page_size"],
                prefill_min=min(r["prefill_pcc"] for r in d["rows"]),
                decode_min=min(r["traced_decode_pcc"] for r in d["rows"]),
                cases=len(d["rows"]),
                cache_dtype=d["cache_dtype"],
                watcher=d["watcher"],
                stress_replays=d["stress_replays"],
                trace_tracking=d["trace_alloc_tracking"],
                program_cache_tracking=d["program_cache_tracking"],
            )
        )
    assert read(DOC / "pytest_execution.json")["exit_code"] == 0
    assert len(correctness) >= 20, f"Missing final suites: {len(correctness)}"
    assert min(r["prefill_min"] for r in correctness) >= 0.995
    assert min(r["decode_min"] for r in correctness) >= 0.995
    write("correctness_summary.json", dict(decoder_sha256=SHA, runs=correctness))
    watchers = []
    for layer in (0, 3):
        for batch in (1, 32):
            tag = f"watcher_b{batch}_layer{layer}"
            raw = DOC / "watcher" / tag / "generated/watcher/watcher.log"
            compact = DOC / "watcher" / (tag + ".log")
            shutil.copy2(raw, compact)
            text = compact.read_text()
            bad = [
                line
                for line in text.splitlines()
                if re.search(r"assert.*fail|watcher.*error|NOC.*error|corrupt|sanitize.*fail", line, re.I)
            ]
            assert not bad, bad
            assert re.search(r"Dump #\d+ completed", text) and "detach" in text.lower()
            validation = read(DOC / f"validation_layer{layer}_b{batch}_page32_watcher.json")
            assert validation["decoder_sha256"] == SHA and validation["stress_seconds"] >= 12
            assert validation["stress_replays"] > 0
            watchers.append(
                dict(
                    run=tag,
                    path=str(compact.relative_to(ROOT)),
                    raw_path=str(raw.relative_to(ROOT)),
                    sha256=hashlib.sha256(compact.read_bytes()).hexdigest(),
                    completed_dumps=len(re.findall(r"Dump #\d+ completed", text)),
                    detached="detach" in text.lower(),
                    anomalous_lines=bad,
                    validation=f"doc/optimized_decoder/validation_layer{layer}_b{batch}_page32_watcher.json",
                )
            )
    write("watcher_clean.json", watchers)
    context = []
    for layer in (0, 3):
        d = read(DOC / f"long_context_layer{layer}.json")
        assert d["decoder_sha256"] == SHA and d["lengths"] == [524275, 524288]
        assert min(r["pcc"] for r in d["rows"]) >= 0.995 and d["traced_decode_pcc"] >= 0.995
        context.append(d)
        for cap in (16352, 16384, 65504, 65536, 262112, 262144, 524256):
            d = read(DOC / f"decode_capacity_layer{layer}_{cap}.json")
            assert d["decoder_sha256"] == SHA and min(d["traced_decode_pcc"]) >= 0.995
            if cap == 65504:
                assert d["long_prefix_prefill_pcc"] >= 0.995
        d = read(DOC / f"decode_capacity_layer{layer}_524288_final_bf16.json")
        assert d["decoder_sha256"] == SHA and d["long_prefix_prefill_pcc"] >= 0.995
    profiles = []
    for layer, kind in ((0, "dense"), (3, "mova_moe")):
        d = read(DOC / "candidates" / f"profile_final_layer{layer}.json")
        assert d["source_sha256"] == SHA
        provenance = read(DOC / "tracy/final" / kind / "provenance.json")
        names = ["attention_packed.weight", "self_attn.o_proj.weight"]
        names += (
            ["mlp.gate_up.weight", "mlp.down_proj.weight"]
            if layer == 0
            else ["mlp.shared_router.weight", "mlp.shared_experts.down_proj.weight"]
        )
        traffic = {n: stored_bytes(d["decode_weight_contract"][n]) for n in names}
        if layer == 3:
            for name, active in (("value_experts", 4), ("experts.gate_up", 8), ("experts.down_proj", 8)):
                traffic[name] = stored_bytes(d["weight_contract"][name], active)
        traffic["norm_gamma"] = sum(
            stored_bytes(d["weight_contract"][name])
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight")
        )
        traffic["KV_read_384_tokens_BFP8"] = 2 * 8 * 384 * 128 // 1024 * 1088
        traffic["KV_update_read_modify_write_32row_tiles"] = 2 * 2 * 8 * 4 * 1088
        if layer == 3:
            traffic["router_bias"] = sum(
                stored_bytes(d["weight_contract"][name]) for name in ("mlp.gate.bias", "self_attn.v_router.bias")
            )
        total = sum(traffic.values())
        win = provenance["windows"]["decode"]
        host = d["traced_warmed_decode_wall_ms"]
        device = win["device_kernel_sum_ms"]
        assert 0 < device < host * 1.15, "Profiler units or host/device window mismatch"
        with (DOC / "tracy/final" / kind / "decode_perf_report.csv").open() as f:
            rows = list(csv.DictReader(f))
        matmuls = [r for r in rows if "matmul" in r["OP Code"].lower()]
        role_names = (
            ["packed_attention", "output", "packed_gate_up", "down"]
            if layer == 0
            else [
                "packed_attention",
                "value_experts",
                "output",
                "shared_router",
                "shared_down",
                "expert_gate_up",
                "expert_down",
            ]
        )
        assert len(matmuls) == len(role_names), "Recheck actual measured projection topology"
        for role, row in zip(role_names, matmuls):
            row["role"] = role
            row["percent_of_decode_kernel_time"] = float(row["Device Time [μs]"]) / (device * 1000) * 100

        profiles.append(
            dict(
                layer=layer,
                run=f"candidates/profile_final_layer{layer}.json",
                trace_host_ms=host,
                device_kernel_sum_ms=device,
                device_op_gaps_ms=win["op_gap_sum_ms"],
                host_minus_kernels_and_gaps_ms=host - device - win["op_gap_sum_ms"],
                roofline_ideal_dram_GB_s=512,
                roofline_traffic_bytes=traffic,
                roofline_total_bytes=total,
                roofline_lower_bound_ms=total / 512e6,
                roofline_fraction_of_host=total / 512e6 / host,
                note="Ideal one-read active weights plus padded KV and tile read-modify-write cache update; excludes duplicated per-core reads, NoC and intermediate traffic. Fixed position256, B1. Profile-run host includes one replay and final synchronization; matched headline run amortizes these over300 replays.",
                matmul_rows=matmuls,
            )
        )
    searches = []
    for profile in profiles:
        layer = profile["layer"]
        for path in sorted((DOC / "candidates").glob(f"*_layer{layer}.json")):
            candidate = read(path)
            if candidate.get("prefill_length") != 256 or candidate.get("input_variant", 0) != 0:
                continue
            final_family = SHA in candidate.get("implementation_sources", {}).values()
            geometry = any(
                word in path.stem
                for word in (
                    "closure_geometry",
                    "closure_cores",
                    "final_split",
                    "final_separate",
                    "block",
                    "grid",
                    "selected_packed",
                )
            )
            if not final_family and not geometry:
                continue
            if "traced_warmed_decode_wall_ms" not in candidate:
                continue
            for row in profile["matmul_rows"]:
                searches.append(
                    dict(
                        layer=layer,
                        role=row["role"],
                        final_role_percent=row["percent_of_decode_kernel_time"],
                        candidate=path.stem,
                        regime="final-policy control" if final_family else "historical precision-locked geometry",
                        options=json.dumps(
                            candidate.get("effective_options", candidate.get("options", {})), sort_keys=True
                        ),
                        prefill_ms=candidate["warmed_prefill_wall_ms"],
                        traced_decode_ms=candidate["traced_warmed_decode_wall_ms"],
                        prefill_pcc=candidate["prefill_pcc"],
                        decode_pcc=candidate["traced_decode_pcc"],
                        decision=(
                            "selected default"
                            if path.stem.startswith("final_default_s256")
                            else "comparison control; see optimization_decisions.md"
                        ),
                        artifact=str(path.relative_to(DOC)),
                    )
                )
    with (DOC / "matmul_search.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(searches[0]))
        writer.writeheader()
        writer.writerows(searches)
    report = dict(
        decoder_sha256=SHA,
        pcc_bar=0.995,
        wall=wall,
        correctness=correctness,
        context=context,
        profiles=profiles,
        watcher=watchers,
    )
    write("results.json", report)
    lines = [
        "# Final measured results",
        "",
        f"Decoder SHA256: `{SHA}`. PCC acceptance bar:0.995. All timing values below are milliseconds.",
        "",
        "Recorded checkpoint/HF inputs, batch1, identical before/after requests. Fused baseline uses its existing physical chunk256; optimized uses2048. Decode is traced, warmed, nonblocking, with one synchronization after the measured loop. Neither timing includes input preparation or compilation.",
        "",
        "| Layer | Logical prefill | Fused prefill | Optimized prefill | Prefill speedup | Fused decode | Optimized decode | Decode speedup | Optimized prefill/decode PCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in wall:
        a, b = r["baseline"], r["optimized"]
        ap = a["warmed_prefill_wall_ms"]
        bp = b["warmed_prefill_wall_ms"]
        ad = a["traced_warmed_decode_wall_ms"]
        bd = b["traced_warmed_decode_wall_ms"]
        lines.append(
            f'| {r["layer"]} | {r["seq"]} | {ap:.6f} | {bp:.6f} | {ap/bp:.2f}x | {ad:.6f} | {bd:.6f} | {ad/bd:.2f}x | {b["prefill_pcc"]:.6f} / {b["traced_decode_pcc"]:.6f} |'
        )
    lines += [
        "",
        "Logical2049 runs two optimized physical chunks (2048 plus a padded one-token tail) versus nine fused256 chunks, with context4096 on both paths. Baseline and optimized PCC, options, iteration counts, source hashes and weight contracts are retained in results.json and the linked candidate artifacts. Lower-precision weights, normalized activations and attention account for the PCC delta; every acceptance case remains above the unchanged bar.",
        "",
        "## Cumulative correctness",
        "",
        "| Layer | Minimum prefill PCC | Minimum decode PCC | Cases | Watcher replays |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer in (0, 3):
        rr = [r for r in correctness if r["layer"] == layer]
        lines.append(
            f'| {layer} | {min(r["prefill_min"] for r in rr):.6f} | {min(r["decode_min"] for r in rr):.6f} | {sum(r["cases"] for r in rr)} | {sum(r["stress_replays"] for r in rr)} |'
        )
    lines += [
        "",
        "These include batches1/2/9/13/32, page sizes32/64/128/256, BF16/BFP8 caches, non-aligned lengths through2049, continuation and independent user positions. Watcher and allocation tracking pass separately from profiling. See correctness_summary.json and watcher_clean.json.",
        "",
        "## Same-run profiling and bandwidth lower bound",
        "",
        "| Layer | Ideal DRAM lower bound | Device kernel sum | Device op gaps | Profile-run host | Host minus device+gaps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in profiles:
        lines.append(
            f'| {r["layer"]} | {r["roofline_lower_bound_ms"]:.6f} | {r["device_kernel_sum_ms"]:.6f} | {r["device_op_gaps_ms"]:.6f} | {r["trace_host_ms"]:.6f} | {r["host_minus_kernels_and_gaps_ms"]:.6f} |'
        )
    lines += [
        "",
        "The lower bound uses the report tool's Blackhole peak512GB/s and stored tile bytes (BFP4:576, BFP8:1088, BF16:2048). Routed weight traffic counts four value and eight MLP experts, not the full allocated expert bank. Cache update includes full32-row tile read/modify/write, as implemented by the native paged-cache reader and writer. Norm gamma and router bias count their stored padded tiles. It is an ideal bandwidth floor, not a measured utilization claim. Profiler matmul dtype/fidelity rows are retained in results.json. Single-replay profiler host timing includes launch/synchronization and profiler instrumentation; the matched300-replay benchmark amortizes launch and final synchronization, with zero per-token host reads or refreshes.",
        "",
    ]
    (DOC / "results.md").write_text("\n".join(lines))
    contract = read(ROOT / "doc/context_contract.json")
    o = contract["optimized_decoder"]
    o.update(
        status=(
            "validated"
            if (DOC / "stage_review.md").exists()
            and "Verdict: clean-pass" in (DOC / "stage_review.md").read_text()
            and SHA in (DOC / "stage_review.md").read_text()
            else "validated_pending_independent_review"
        ),
        decoder_sha256=SHA,
        bf16_cache_compatibility="Short-context paged/continuation B2 and independent-HF-KV full-capacity decode plus long-prefix prefill probe, both layer kinds",
        attention_policy="Prefill HiFi2/BF16 accumulation,q32,k1024;8x8 below65536 prefix end,11x10 from65536. Decode LoFi/BF16 k128 below16384 capacity; HiFi2/BF16 k256 below262144; HiFi2/FP32 k1024 beyond (BF16 cache k512). B1 uses full11x10 grid from65536 capacity; other batches preserve8-wide producer/reducer ownership. Q moves to DRAM from262144 capacity.",
    )
    o["bf16_full_capacity_evidence"] = {
        str(l): f"doc/optimized_decoder/decode_capacity_layer{l}_524288_final_bf16.json" for l in (0, 3)
    }
    o["capacity_boundary_evidence"] = {
        str(l): [
            f"doc/optimized_decoder/decode_capacity_layer{l}_{c}.json"
            for c in (16352, 16384, 65504, 65536, 262112, 262144, 524256)
        ]
        for l in (0, 3)
    }
    (ROOT / "doc/context_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print("FINAL_REPORT_PASS", SHA, len(correctness))


if __name__ == "__main__":
    main()
