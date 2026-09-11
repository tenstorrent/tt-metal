# B1: per-layer MoE variance - evidence review, 2026-09-11

**Historical PP4 captures locate 85.2% of a same-stage MoE cost gap in Dispatch/Combine. New local controlled replays establish repeatable placement sensitivity: balanced placement reduces isolated Combine time by 38.36% for L18 routing and 25.82% for L23 routing. Destination imbalance alone, the cause of the historical full-model gap, and an end-to-end optimization remain unproven.**

Task B1 belongs to Sonnet in `mistral4_bringup/CURRENT_PLAN-9-8.md`. This updated finding and its local issue draft live on the separate `ssalice/mistral4-b1-investigation` branch. The historical analysis was performed against `b48bf4095de`; the new experiments used the compatible recovered native build with the followups checkout at `96125e3b5f7` and an archived diagnostic worker. See [local provenance](b1-controlled-routing/LOCAL_RUN_PROVENANCE.json). No production optimization has been implemented or posted.

## Evidence and provenance

The existing **untracked** draft `/data/ssalice/temp/tt-metal/models/demos/deepseek_v3_d_p/docs/MISTRAL4_PER_LAYER_MOE.md` identifies these as September 8, eager, 36-layer captures on `bh-glx-120-b03u02`, reference branch `kmabee/akhan/mistral4-prefill-followups` at `11a16d60b493e3bc39fea3682d9e454d3a41d8b0`. The commit exists; the surviving runner log confirms `PREFILL_USE_TRACE=False`. The capture itself does not independently attest a Git revision, so that revision remains documented provenance, not a cryptographic build identification.

Full PP reports survive under `/data/ssalice/temp/tt-metal/mistral4_perf_profile/pp4_deep36/rank{0,1,2,3}/reports/`. Exact filenames and SHA-256 hashes are in [b1_evidence.json](b1_evidence.json). These are stronger evidence for B1 than the committed single-layer captures: the latter confound stage and layer and cannot reconstruct a nine-layer spread. The plan's original 5.70→8.77 ms comparison used a different historical capture and metric; this report does not silently equate it with the numbers below.

## Recomputed findings

The metric is the **sum of each MoE operation's maximum device kernel duration, averaged over nine retained passes**. It is an operation budget, not measured layer wall time. Each PP rank has eight devices. There are ten signposted passes per layer; the first is discarded. The script asserts eight distinct devices and consecutive call counts for every relevant logical operation, nine layers per rank, ten passes per layer, and complete MoE operation categories in retained passes.

| PP layer | MoE budget |
|---|---:|
| L2, minimum across 36 layers | 6.129 ms |
| L18, maximum across 36 layers | 10.377 ms |
| Spread | **69.31%** |

A stronger comparison holds the stage and physical devices fixed: rank 2, global layers 18 and 23.

| Operation | L18 | L23 | Difference |
|---|---:|---:|---:|
| Combine | 4.434 ms | 2.017 ms | 2.416 ms |
| Dispatch | 2.874 ms | 1.877 ms | 0.997 ms |
| Expert FFN | 2.855 ms | 2.264 ms | 0.591 ms |
| Remaining MoE operations combined | approximately equal | approximately equal | <0.001 ms |

**Dispatch + Combine explain 3.413 ms, or 85.2%, of the 4.005 ms MoE-budget difference.** The existing MLA category remains approximately 3.55–3.56 ms across this stage. That category excludes projection matmuls, so it should not be described as the complete attention block.

The surviving TP raw pair (`mistral4_perf_profile/1rank_deep36/rank0/.logs/cpp_device_perf_report.csv` and `tracy_ops_data.csv`) also reproduces a 2.29→3.84 ms MoE spread with the existing untracked analyzer: L6 cheapest, L24 dearest, four retained passes after discarding the first of five. An independent read-only check verified 350,400 device rows / 10,950 logical operations: each block contains 32 distinct devices, consecutive call counts, and one host operation name. This supports the historical TP result, but is not included in the PP-only compact reproduction script. Absolute PP and TP budgets are not matched comparisons: different contexts, topology, and expert layouts.

## What this establishes - and what remains open

The data locates the variation principally in the routing communication operations. It does **not** establish that cross-chip traffic volume is the only changing variable. Routing skew, source/destination placement, congestion, synchronization, memory locality, and implementation sensitivity can interact. Identical shapes or static attributes do not rule out program improvements. Nor do differing TP/PP hot-layer rankings prove that a single expert remapping cannot help both.

The previous draft's “30.4 sigma,” split-half correlations, placement causality, and “not fixable with a program-config change” claims are not adopted here: their statistical procedure or causal controls have not been independently reproduced. Kernel-budget differences are not measured end-to-end speedups.

## New local evidence: controlled routing replay

On September 11, `bh-glx-120-b10u14` ran twelve successful captures on the eight-chip Galaxy column selected by `TT_VISIBLE_DEVICES=0,1,2,3,11,10,9,8`. Each case ran in two fresh processes, with the second pass in reverse order. Each process executed ten eager iterations and discarded the first. All 1,920 target operation rows validated; 1,728 remain after warm-up removal.

The replay uses SP8/TP1, all 128 experts, 16 per chip, 5,120 tokens, top-4 routing, embedding width 4,096 and Mistral's 4,096-byte fabric payload. The original shared replay represented one TP4 column with 32 experts and could not be reused unchanged. The successful diagnostic omits the subdevice manager while retaining the same first-row dispatch cores. It substitutes layout conversion for FFN and omits shared-expert concurrency and downstream weighted reduction.

These selections were recovered from a separate September 10 TP execution, not the September 8 PP run. They are actual captured selections replayed under a counterfactual PP placement. The routing cases and their new isolated timings are paired; the historical full-model timings and selections are not.

Each table entry is the mean of two process-level means. Within a process, the metric is the mean of the maximum device kernel duration for each retained iteration. It is not model latency. Exact process ranges and all per-device samples are linked below.

| Layer's routing | Case | Dispatch (ms) | Combine (ms) |
|---|---|---:|---:|
| L18 | Captured | 1.659080 | 1.723374 |
| L18 | Balanced placement | 1.624431 | 1.062319 |
| L18 | Source shuffled | 1.660601 | 1.653402 |
| L23 | Captured | 1.641102 | 1.477407 |
| L23 | Balanced placement | 1.606123 | 1.095940 |
| L23 | Source shuffled | 1.626815 | 1.478913 |

Balanced placement shortens Combine by **38.36% / 25.82%** and Dispatch by **2.09% / 2.13%** for L18/L23 respectively. The effect is larger than the observed differences between the two successful process repetitions. Two repetitions are limited evidence, not a broad statistical characterization.

Placement preserves each original expert's assignment count through a bijective remapping, but changes device load, padded expert-region workload, expert ordering, fanout and locality together. For L18, raw destination max/mean falls from 1.7402 to 1.0176, while mean destinations per token rises from 3.2334 to 3.4375. This cannot isolate imbalance alone as the mechanism. [Host metrics](b1-controlled-routing/host_comparison.json) also include 32-token-rounded expert-region loads.

Source shuffling preserves expert counts, destination totals and the fanout histogram, but changes source flow and token ordering/batching. L18 Combine falls **4.06%** despite unchanged destination totals; a single aggregate destination-load score therefore misses relevant information. L23 changes only **+0.10%**, smaller than its observed process-to-process range. Assignment counts and unique destinations are not measured network bytes.

All successful captures used the same archived worker. Strict extraction validates signpost boundaries, eight devices, operation order, finite positive durations, and device-offset call IDs. Import interruptions, a stalled first iteration and firmware/fabric initialization failures are excluded. Targeted resets occurred between some successful runs. The earlier stall's cause remains unresolved: manager removal and recovery were not isolated from one another. No numerical output check was performed, and these are Galaxy-column measurements, not LoudBox calibration.

Evidence: [local findings and run ranges](b1-controlled-routing/LOCAL_FINDINGS.md), [capture matrix](b1-controlled-routing/MATRIX_RESULTS.md), [machine-readable timings](b1-controlled-routing/matrix_timings.json), [validation/recovery history](b1-controlled-routing/VALIDATION.md).

## B1 status and prioritized next investigation

The expensive operations are identified, routing distributions are available, and controlled replay has demonstrated placement sensitivity. A reviewable narrowed finding is ready. The historical full-model mechanism and production benefit remain open.

1. **Highest priority: pair actual PP routing with timings in a new full-model run.** Use the same request/chunk and rank for L18/L23; record valid tokens, expert mapping and physical device IDs. Retain existing device indices and export after the measured request, avoiding in-forward host reads. Compare capture enabled/disabled to quantify perturbation. See the [instrumentation design](b1-controlled-routing/PAIRED_CAPTURE_PLAN.md). This tests whether the isolated effect is relevant under real FFN/shared-expert execution.
2. **Separate placement mechanisms.** First permute expert slots only within each destination device: this preserves destination counts, source-to-destination assignments, fanout, and per-device padded totals while changing expert ordering. Then compare placements selected using raw versus tile-rounded loads, reporting fanout/locality for each. If order alone changes Combine, a balance-only explanation is inadequate. These are proposed experiments, not measured results.
3. **Separate source flow from ordering.** Permute complete source shards while preserving token order within each shard, and separately permute token order within each source shard while keeping its expert-assignment counts fixed. The latter preserves source-to-destination totals; together these controls narrow the source-shuffle effect. Repeat with multiple fixed seeds and randomized case order before generalizing the smaller effects.
4. **Only after mechanism and correctness checks, test production benefit.** A real expert remap must relocate weights consistently and preserve numerical outputs. Run matched traced end-to-end throughput tests independently for TP and PP4. Do not apply the isolated Combine percentages to full-model throughput.

Full-model pairing is more valuable next than collecting many more variants of this same proxy. The intermittent startup/stall behavior should be tracked separately if it recurs; it does not explain the successful timing differences by itself. [B1_ISSUE_DRAFT.md](B1_ISSUE_DRAFT.md) records the current finding and follow-up scope. Nothing has been posted or sent.

## Reproduction

```bash
python3 analyze_b1.py \
  --source-root /data/ssalice/temp/tt-metal \
  --output-dir /tmp/mistral4-b1-reproduction
```

The command above reproduces only the historical offline analysis; it needs no build. Reproduce the new local timing summaries with `python3 b1-controlled-routing/aggregate_timings.py` from this report directory. The local worker ran on real hardware with the existing native build; no C++ build changes were made. The command completed successfully on all four surviving PP reports; [b1_per_layer.csv](b1_per_layer.csv) contains all 36 layers. These September 8 measurements must remain separate from September 10's traced TP/PP throughput comparison.
