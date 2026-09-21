# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize the complete paired suite without dropping unfavorable samples."""

import argparse
import json
from pathlib import Path
import statistics

ORDER = ("stock", "D", "C", "B", "A", "E", "F", "G")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--real-attention", type=Path)
    parser.add_argument("--block-sweep", type=Path)
    parser.add_argument("--extra-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--repeatability", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--reset-telemetry", type=Path)
    parser.add_argument("--post-reset-repeatability", type=Path)
    parser.add_argument(
        "--hardware-label", help="New hardware cohort; do not attribute historical failures to this run"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report failed qualifications explicitly; never score their images",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh report path")
    scores = json.loads(args.scores.read_text())
    comparison = json.loads(args.comparison.read_text())
    manifests = {v: json.loads((args.suite / v / "manifest.json").read_text()) for v in ORDER}
    initial_manifests = dict(manifests)
    for path in args.extra_manifests:
        replacement = json.loads(path.read_text())
        assert replacement["evaluation_mode"] == "exploratory_repeatability_not_required"
        assert manifests[replacement["variant"]]["status"] == "failed"
        manifests[replacement["variant"]] = replacement
    completed = [v for v in ORDER if manifests[v]["status"] == "completed"]
    assert args.allow_incomplete or len(completed) == len(ORDER)
    assert set(scores["means"]) == set(completed)
    for variant, manifest in manifests.items():
        if variant not in completed:
            assert manifest["status"] == "failed" and manifest.get("error")
            continue
        assert manifest["status"] == "completed" and manifest["variant"] == variant
        assert manifest["model_repair"] == "main_fused" and manifest["conditioning"] == "stock"
        assert (
            manifest["two_step_trace_latents_bitwise_equal"]
            or manifest.get("evaluation_mode") == "exploratory_repeatability_not_required"
        )
        assert len(manifest["block_bench"]) == 6
        assert len(manifest["results"]) == 6
        assert {(r["prompt_id"], r["seed"]) for r in manifest["results"]} == {(p, s) for p in range(3) for s in (0, 42)}
    hardware_note = (
        "**Hardware interruption:** the first sustained sweep was interrupted by a PCIe 0xffffffff "
        "device-read failure after elevated-temperature telemetry. Its incomplete results and original "
        "failed qualifications are retained. See [repeatability and hardware evidence](../REPEATABILITY.md)."
    )
    if args.reset_telemetry:
        recovered = json.loads(args.reset_telemetry.read_text())["device_info"]
        assert len(recovered) == 8
        temperatures = [float(d["telemetry"]["asic_temperature"]) for d in recovered]
        hardware_note += (
            " All eight reserved boards were subsequently reset with the user's approval, rediscovered "
            f"at {min(temperatures):.1f}–{max(temperatures):.1f}°C, and passed the all-device matmul smoke test. "
            "The A/B/E/G exploratory image reruns use this recovered session. Reset recovery does not "
            "retroactively qualify old timings or establish the cause of numerical variability."
        )
    else:
        hardware_note += " No reset is represented in this report."
    repeatability_note = (
        "**Exploratory results, not a clean attention-quality ranking:** the stock control itself "
        "fails repeated-run equality (37.7–43.1% final-latent L2 between untraced 50-step runs). "
        "The shared repeatability issue remains unresolved. Initial failures are preserved; "
        "the optional exploratory mode records equality failures without aborting image generation. "
        "Finite-value, shape and no-fallback checks remain mandatory."
    )
    if args.hardware_label:
        hardware_note = (
            f"**Hardware cohort:** {args.hardware_label}. All image variants in this report use this allocation. "
            "Earlier bh-51 results, reset attempts and thermal/PCIe failures are retained separately in STATUS.md. "
            "Converted-weight caching is enabled with the pinned-host-memory-cache workaround; "
            "this setting is common to all choices and does not change attention arithmetic."
        )
        repeatability_note = (
            "**Exploratory evaluation:** exact replay is measured, but is not an admission requirement for these images. "
            "Finite-value, shape and no-fallback checks remain mandatory. Historical stock runs were non-repeatable; "
            "the fresh same-input stock control below determines whether that issue persists in this cohort. "
            "Cross-variant latent L2 measures end-to-end change relative to D, not per-call SDPA error "
            "or error against a ground-truth reference model."
        )
    text = [
        "# FLUX.2 attention frontier evaluation",
        "",
        f"{6 * len(completed)} generated images out of 48 planned: seven frozen attention recipes plus the separate stock ring control. "
        "Three prompts × seeds 0/42; 50 steps, 1024×1024, guidance 4, no prompt upsampling. "
        "Checkpoint, prompt embeddings, noise seeds, scheduler, encoder and VAE settings match.",
        "",
        repeatability_note,
        "",
        hardware_note,
        "",
        "Eight Blackhole chips, 2×4 mesh, SP2/TP4, 48 global heads, head width 128. "
        "4096 image + 512 text tokens; frontier Q256/K512. The missing residual and per-head "
        "normalization are repaired identically using the main-style fused implementation. "
        "Stock conditioning is held fixed; the additional guidance/phase correction is not enabled. "
        "Thus this is not a full-reference FLUX.2 numerical qualification; known conditioning differences "
        "from Diffusers are deliberately common to all choices.",
        "",
        "## Integration qualification",
        "",
        "Failed choices are not substituted or represented by another choice's images. "
        "'Initial gates passed' records only the original qualification calls, not proof of general deterministic execution. "
        "See the repeatability investigation for the limits of interpreting paired image differences.",
        "",
        "| Choice | Result | Images |",
        "|---|---|---:|",
    ]
    if args.hardware_label and all(manifests[v].get("converted_weight_cache") for v in completed):
        loads = [r for v in completed for r in manifests[v]["weight_loads"] if not r["already_loaded"]]
        setups = [manifests[v]["pipeline_setup_seconds"] for v in completed]
        hits = sum(not r["torch_state_dict_requested"] for r in loads)
        index = text.index("## Integration qualification")
        text[index:index] = [
            "## Weight caching",
            "",
            f"All {hits}/{len(loads)} initial component loads hit the converted-weight cache. "
            f"Pipeline setup was {min(setups):.2f}–{max(setups):.2f} s across the eight choices "
            f"(median {statistics.median(setups):.2f} s). Unexpected conversion fallback was forbidden.",
            "",
            "See [the separate cold/warm qualification](../WEIGHT_CACHE.md) for startup timings, "
            "exact-weight checks and the pinned-host-memory-cache workaround. "
            "Caching changes startup, not the measured attention arithmetic.",
            "",
        ]
    for variant in ORDER:
        manifest = manifests[variant]
        original = initial_manifests[variant]
        status = "Initial gates passed" if original["status"] == "completed" else original["error"].splitlines()[0]
        if manifest.get("evaluation_mode") == "exploratory_repeatability_not_required":
            mismatched_blocks = sum(not r["first_device_output_bitwise_equal"] for r in manifest.get("block_bench", []))
            replay_status = "exact" if manifest.get("two_step_trace_latents_bitwise_equal") else "different"
            detail = f"Exploratory: model replay {replay_status}; {mismatched_blocks}/6 blocks differ"
            status = detail if original["status"] == "completed" else status + "; " + detail
        text.append(f"| {variant} | {status} | {len(manifest['results'])} |")
    if args.repeatability:
        repeat = json.loads(args.repeatability.read_text())
        assert repeat["status"] == "completed" and repeat["variant"] == "stock"
        text += [
            "",
            "## Stock repeatability control",
            "",
            "One process, identical cached prompt embeddings and noise seed, stock attention, "
            "50 denoising steps, three untraced calls followed by three steady traced calls. "
            "The first trace-capture call is intentionally discarded. See the measured equality results below; "
            "this control uses stock attention, not a frontier replacement.",
            "",
            "| Call | Mode | Final-latent L2 vs first | Exact vs first |",
            "|---|---|---:|---|",
        ]
        for row in repeat["rows"]:
            if row["mode"] != "capture":
                text.append(
                    f"| {row['index']} | {row['mode']} | {row['l2_vs_first_pct']:.3f}% | {row['exact_vs_first']} |"
                )
        mismatches = [r for r in repeat["rows"] if r["mode"] != "capture" and not r["exact_vs_first"]]
        text += [
            "",
            (
                "Stock still exhibits a full-model repeatability limitation; its root cause is not established. "
                "Do not interpret cross-variant final-latent L2 as isolated attention error."
                if mismatches
                else "All measured stock calls were bitwise identical. This finite control does not prove general determinism "
                "or identify which changed hardware/runtime condition explains the historical failures."
            ),
            "",
        ]
    if args.post_reset_repeatability:
        repeated = json.loads(args.post_reset_repeatability.read_text())
        assert repeated["status"] == "completed" and repeated["variant"] == "stock"
        text += [
            "",
            "### Stock repeatability after reset",
            "",
            "Same diagnostic repeated after hardware recovery; the capture call is excluded.",
            "",
            "| Call | Mode | Final-latent L2 vs first | Exact vs first |",
            "|---|---|---:|---|",
        ]
        for row in repeated["rows"]:
            if row["mode"] != "capture":
                text.append(
                    f"| {row['index']} | {row['mode']} | {row['l2_vs_first_pct']:.3f}% | {row['exact_vs_first']} |"
                )
        text.append("")
    text += [
        "",
        "## CLIP and output differences",
        "",
        "Raw OpenAI CLIP ViT-B/32 normalized text/image cosine (not ×100). "
        "Six paired images per choice are exploratory, not a statistically powered model evaluation. "
        "CLIP measures text alignment, not fidelity to a reference image. D is a comparator, not ground truth.",
        "",
        "| Choice | Mean CLIP | Min–max CLIP | Mean paired Δ vs D | Mean final-latent L2 vs D |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ORDER:
        if variant not in completed:
            text.append(f"| {variant} | — | — | — | — |")
            continue
        rows = [r for r in scores["rows"] if r["variant"] == variant]
        assert len(rows) == 6
        values = [r["clip_cosine"] for r in rows]
        latent = [r["latent_l2_vs_D_pct"] for r in comparison["latent_comparisons"] if r["variant"] == variant]
        text.append(
            f"| {variant} | {statistics.mean(values):.5f} | {min(values):.5f}–{max(values):.5f} | "
            f"{statistics.mean(r['paired_delta_vs_D'] for r in rows):+.5f} | {statistics.mean(latent):.2f}% |"
        )
    text += [
        "",
        "## Performance",
        "",
        (
            "Image-run block timings below are warmed but still subject to clock, temperature and run-order variation. "
            "These are measured full-model block costs, not SDPA-only FLOP utilization."
            if args.hardware_label
            else "The initial image-run block times below are **not a steady-state ranking**. D and F's "
            "early dual-block values drifted substantially; prefer the later sustained-workload table "
            "for block comparisons, subject to its thermal caveat."
        ),
        "",
        "Block times are isolated **full-block blocking mesh trace replays**, including preprocessing "
        "and communication, after 250 warmups, with 50 timed samples. They exclude capture, compilation "
        "and weight conversion. These are host-observed accelerator trace latencies, not hardware-counter "
        "FPU utilization. Denoising-step times are separate traced-pipeline host measurements.",
        "",
        "Stock uses joint ring SDPA; all seven frontier variants use local Q plus prepared-format KV "
        "all-gather. Stock-versus-frontier speed differences therefore include the communication "
        "schedule, not just numerical arithmetic. The frontier adapter is not overlapped ring attention.",
        "",
        "| Choice | Median step, ms | Dual 0 | Dual 3 | Dual 7 | Single 0 | Single 23 | Single 47 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ORDER:
        if variant not in completed:
            text.append(f"| {variant} | — | — | — | — | — | — | — |")
            continue
        perf = comparison["performance"][variant]
        blocks = {r["block"]: r for r in perf["block_bench"]}
        times = [
            blocks[name]["median_ms"] for name in ("dual.0", "dual.3", "dual.7", "single.0", "single.23", "single.47")
        ]
        text.append(
            f"| {variant} | {perf['median_denoising_step_ms']:.2f} | " + " | ".join(f"{t:.3f}" for t in times) + " |"
        )
    text += [
        "",
        "All block columns are milliseconds. Raw replay samples, min/max and exact-replay checks "
        "are retained in each manifest. Values from this 4608-token model run do not characterize "
        "the very-long-context precision loss observed in the standalone SDPA experiments.",
        "",
    ]
    if args.telemetry:
        telemetry = json.loads(args.telemetry.read_text())
        devices = telemetry["device_info"]
        temps = [float(d["telemetry"]["asic_temperature"]) for d in devices]
        clocks = [float(d["telemetry"]["aiclk"]) for d in devices]
        text += [
            "**Telemetry snapshot:** the recorded sample "
            f"showed {min(temps):.1f}–{max(temps):.1f}°C and {min(clocks):.0f}–{max(clocks):.0f} MHz. "
            "These are observed timings on this allocation, not qualified peak performance. "
            "Thermal/clock variation can bias both run-order comparisons and speedups. "
            "This observation does not establish the cause of numerical nondeterminism.",
            "",
        ]
    if args.block_sweep:
        sweep = json.loads(args.block_sweep.read_text())
        complete_sweep = sweep["status"] == "completed" and len(sweep["rows"]) == 96
        assert complete_sweep or (args.allow_incomplete and len(sweep["rows"]) >= 48)
        text += [
            "## Sustained-workload block check",
            "",
            "To reduce warmup and run-order bias, this follow-up "
            "uses the same D-captured inputs for every choice, 5000 initial dual-block replays, "
            "then 750 warmups and 50 timed samples per block. The intended schedule is one forward "
            "and one reverse pass. Each cell pools the available completed rounds (50 samples per round). "
            "The rounds column makes incomplete coverage explicit. Replay differences are recorded; a timing measurement is not a "
            "correctness pass. Prefer these warmed block numbers over the initial image-run block measurements.",
            "",
            "| Choice | Dual 0 | Dual 3 | Dual 7 | Single 0 | Single 23 | Single 47 | Rounds | Max round-median difference |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for variant in ORDER:
            cells, drift, counts = [], [], []
            for name in ("dual.0", "dual.3", "dual.7", "single.0", "single.23", "single.47"):
                rows = [r for r in sweep["rows"] if r["variant"] == variant and r["block"] == name]
                assert 1 <= len(rows) <= 2
                counts.append(len(rows))
                cells.append(statistics.median(t for r in rows for t in r["replay_ms"]))
                medians = [r["median_ms"] for r in rows]
                if len(rows) == 2:
                    drift.append(100 * (max(medians) / min(medians) - 1))
            drift_text = f"{max(drift):.2f}%" if len(drift) == 6 else "incomplete"
            text.append(
                f"| {variant} | " + " | ".join(f"{t:.3f}" for t in cells) + f" | {min(counts)} | {drift_text} |"
            )
        text += ["", "All block times are milliseconds; raw samples are in `block-sweep.json`.", ""]
        if not complete_sweep:
            text += [
                f"**Hardware-interrupted:** {len(sweep['rows'])}/96 block measurements were durably recorded "
                "in the JSON report. The first pass covers all eight choices. During the reverse pass, "
                "device discovery failed with a PCIe 0xffffffff read. Active and queued tests were stopped "
                "before the later recovery work. The raw log contains additional completed blocks from the interrupted "
                "variant, which are not included in the table because its full six-block round was not saved.",
                "",
            ]
        failures = [r for r in sweep["rows"] if not r["first_device_output_bitwise_equal"]]
        text.append(
            f"Replay mismatches in the saved diagnostic sweep: {len(failures)}/{len(sweep['rows'])} block measurements."
        )
        text.append("")
        for r in failures:
            checks = [c for c in r["replay_checks"] if not c["exact"]]
            text.append(
                f"- Round {r['round']}, {r['variant']} {r['block']}: max replay L2 {max(c['l2_pct'] for c in checks):.6f}%."
            )
        text.append("")
    text += ["## Paired images", ""]
    for prompt in range(3):
        for seed in (0, 42):
            text.append(f"- [Prompt {prompt}, seed {seed}](comparison/prompt{prompt}-seed{seed}.png)")
    text += [
        "",
        "Full-resolution originals and final latents are in each variant directory; "
        "all six samples per variant are included, without selecting favorable seeds.",
    ]
    if args.real_attention:
        real = json.loads(args.real_attention.read_text())
        assert real["status"] == "completed" and len(real["rows"]) == 28
        text += [
            "",
            "## Identical real-QKV operator checks",
            "",
            "All seven recipes evaluated on the same D captures from four blocks, four selected "
            "global heads, 512 selected query rows and all recorded KV tokens. Reference: original "
            "BF16 Q/K/V evaluated in FP64. This is an operator check, not final image error.",
            "",
            "| Choice | Min–max L2 across captured blocks | Min PCC |",
            "|---|---:|---:|",
        ]
        for variant in ORDER[1:]:
            rows = [r for r in real["rows"] if r["variant"] == variant]
            text.append(
                f"| {variant} | {min(r['l2_pct'] for r in rows):.3f}%–{max(r['l2_pct'] for r in rows):.3f}% | "
                f"{min(r['pcc'] for r in rows):.6f} |"
            )
    text += [
        "",
        "## Reproducibility",
        "",
        "Each manifest records the pinned checkpoint, source hashes, numeric defines, observed KV "
        "transport types, image/latent hashes, software versions and trace verification. "
        "Unsupported inputs fail; no attention-mode fallback is allowed. Exploratory mode records "
        "replay mismatches instead of asserting exact equality; it does not change numerical recipes. Earlier noise-producing "
        "bring-up runs are excluded from this suite.",
        "",
    ]
    args.output.write_text("\n".join(text))


if __name__ == "__main__":
    main()
