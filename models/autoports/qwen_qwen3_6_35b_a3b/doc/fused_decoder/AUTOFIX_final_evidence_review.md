# Final fused evidence review

Source/CPU-only review; no devices were opened and no production implementation was edited. This is an orchestration review, not an independent stage-pass verdict.

## Latest validation alignment: L1 projection candidates

This subsection supersedes the earlier seed-coverage limitations below. The queue and checker now require fresh-HF three-repeat8192+33 evidence for linear seeds913/42/154 and full-attention seed913. All final stress families validate seed/layer/length identity, prefill PCC, finite output/contribution PCC, unique complete positions per repeat and continuation, tracking including program cache, and current production/harness/reference provenance. Cached-oracle diagnostic records cannot satisfy these final checks.

The final length256/context512 runs now explicitly use10 prefill and60 decode samples, matching the snapshot comparator harness. The existing checker enforces equal sample counts and workload/environment identity in the best-correct manifest. No final-stress command supplies `--oracle-cache`; the separate full-context oracle contract remains unchanged.

The currently inspected source is `f7b052549c01fa1a2cf6c6c074d65d0589fe71ecd308eaa9b5e32ed0994cc193`, with singleton ordered projection products and reduction output in L1, checkpoint weights in DRAM, and multirow prefill products in DRAM. `projection_l1_product` has single-repeat8192+33 passes for154/42/913; `projection_l1_chain` has154/42 records at this inspection, with913 still being run by the device owner. Qualification and final source-frozen acceptance remain distinct.

Best-correct selection must include these latest qualified candidates: product-only decode1.636934ms and chain1.635427ms, atlength128/context512 with10/60 samples. The chain's0.001507ms difference is not a robust final speedup claim. Norm-L1 decode1.666140ms and weight-staging1.718130ms are rejected regressions against chain1.635427ms; the pattern assessment now records both. The final manifest must remeasure its eligible historical comparator atlength256/context512 with10/60 samples. It cannot compare those128-token prefill timings to final256-token timings, nor nominate the identical current source as its own historical control. The user contract requires strictly improved decode and a before/after prefill report. The checker records prefill before/after, absolute and percentage delta, and ratio without demanding a prefill speedup: a decode-only rewrite can leave prefill unchanged, and timing noise is not a valid extra rejection gate. Accuracy, determinism, matching workload/sampling, and strict decode improvement remain required.

At this inspection all30 fixed final JSON/provenance pairs are incomplete for this source:24 exist with stale or absent source binding, and6 are absent (the two default endurance records, two extra-seed endurance records, and two final Tracy correctness records). `profiler_completeness.json` and `best_correct_comparisons.json` are also absent. Candidate profiling under `profile_projection_l1` is useful assessment evidence but does not automatically satisfy the final `tracy/layer_{0,3}` paths. Run the final queue, final profiler/report generation, matched comparator measurements, and checker after freezing the chosen source; then reconcile stop-rule evidence and obtain independent review. No final closure is claimed.

## Findings

The final stage needs a source-frozen rerun of correctness, clean latency, and profiler evidence. `validate_fused.sh` runs the correctness/latency matrix and source/watcher audits, but does not run `collect_fused_perf.sh`, the final checker, independent stage review, or the documented stop-rule accounting. Passing that shell queue alone does not close the stage.

The old checker compared traced decode only against historical functional benchmarks. Those benchmarks are useful direct graph controls, but the linear functional implementation failed the longer recurrent tests. They cannot establish a speedup against the best known correct implementation. Corrected candidates also have different prefill lengths and sample counts: the README's early corrected timings use length128/context512 and 3/12 samples; later measurements use 10/60. The final benchmark matrix includes length256/context512 and lengths127/128/129/257/4097 at context8192. Remeasure matched workloads and sampling; do not compare their prefill medians directly.

At the initial inventory, final_0/3, stress_0, stress_extended_0, and long-context evidence existed but did not identify the then-current decoder source; stress_endurance_0/3, final Tracy correctness records, and profiler_completeness were absent. The source is still changing, so that inventory is not a final-source claim.

Seed identity must come from JSON. `sequential_prefill_state_8192_0_results.json` and `last_only_prefill_8192_3_results.json` use seed913, not seed42. Seed42 is the short benchmark default. The queue currently exercises the stress default913 only. Seed154 exposed a shared failure at position5991 and belongs in the final regression matrix. The later `output_norm_boundary_seed154_results.json` records a passing repair (8192+33, one replay), while the prior failures remain preserved; that repair does not replace the required three-replay final evidence. Any subsequently passing seed42 or913 result must retain its own exact source identity.

The graph-fusing skill requires numerical equivalence at the prior acceptance bar, no material regression, and improved measured performance for each kept rewrite. The repository README additionally leaves final profiles, stop-rule evidence, independent review, and commits open. `work_log.md` correctly distinguishes elapsed wall time from supported device-session time; no elapsed-time substitution should close the stop rule.

## Checker change

`tests/check_fused_evidence.py` now requires an explicit best-correct comparison manifest. It verifies:

- Candidate benchmark provenance identifies the current decoder; comparator benchmark and qualifying correctness records identify the supplied frozen comparator source. Snapshot-harness `measured_decoder_source_sha256` takes precedence over the live-file fingerprint.
- Both benchmarks pass PCC, determinism and fallback checks, have positive finite timing samples, use matching layer/kind/length/context, device, package versions, and sample counts, and were measured serially without watcher/profiler/allocation tracking.
- Comparator eligibility includes source-bound 8192-token/128-prompt endurance for every explicitly requested seed, complete output and contribution rows, and 33 continuation rows. A short benchmark alone cannot qualify it.
- Traced decode must improve against the declared comparator. Prefill before/after timings and their absolute/percentage delta are reported, without a strict speedup assertion. At least the corrected linear layer0 requires a comparison; layer3 is supported too, without replacing its existing functional gates.
- Existing final and matched-length benchmark checks now include their recorded prefill and direct-functional contribution PCCs.
- If any later check fails after the inherited checker has written an intermediate PASS, the final artifact is replaced with FAIL. An inherited partial PASS can no longer masquerade as a completed fused-stage result.

The checker cannot establish that an omitted candidate is slower or incorrect. The manifest rationale and stage reviewer must justify the selected best known correct control. A comparator that fails seed154 cannot be described as correct across the known regression set. Candidate seed154 acceptance is still a separate final obligation: the existing fixed stress filenames do not enforce that seed, and this change does not silently reinterpret them.

Manifest schema (illustrative filenames only; this is **not** a passing manifest):

```json
{
  "selection_rationale": "Explain why this source is the fastest qualifying earlier control for the matched workload, including rejected faster controls.",
  "comparisons": [
    {
      "candidate": "final_0.json",
      "comparator": "best_correct_l0_s256.json",
      "comparator_source": "best_correct.py.snapshot",
      "comparator_correctness": [
        "best_correct_seed913_results.json",
        "best_correct_seed154_results.json"
      ],
      "required_endurance_seeds": [913, 154]
    }
  ]
}
```

Paths are relative to the manifest directory unless absolute. Each benchmark/correctness JSON needs its `.provenance.json` sidecar. `benchmark_fused_snapshot.py` can remeasure a historical source with the standard harness and records the source actually executed. If its existing correctness records lack source identity, rerun or reconstruct provenance only from independently preserved evidence; do not relabel current-source records as historical results.

Final checker command, after all real artifacts and the manifest exist:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_fused_evidence \
  --best-correct-manifest models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/best_correct_comparisons.json
```

CPU fixture validation accepted a valid synthetic comparator and rejected mismatched workloads, contribution failure, mismatched sample counts, and failed comparator eligibility. A subsequent contract correction removed the slower-prefill rejection; a CPU fixture confirms slower prefill is reported while tied/slower decode remains rejected. A wrapper fixture confirmed a later failure overwrites an intermediate inherited PASS. These are checker tests only, not model validation.

## Remaining final orchestration

1. Freeze production, reference/test sources, checkpoint revision and selected comparator snapshots. Preserve all rejected candidate artifacts. Resolve seed154 with the unmodified acceptance oracle and explicitly run the repaired final source for known seeds, including913 and154.
2. Run the serial validation queue on that frozen source: both layer kinds; 128/1024/8192 decode and33 continuation; three deterministic tracked replays; battery/batch/partition/resume, watcher trace reuse, full-context262144 and tail rows, page ownership, padded trace, and the synthetic pytest contract. Source changes invalidate earlier candidate evidence.
3. Run clean latency benchmarks and matched best-correct controls with equal workloads and sampling. Functional comparison remains useful but does not substitute for canonical HF acceptance or corrected-control performance.
4. Run `collect_fused_perf.sh` separately, retaining raw CSVs, signposts, human-readable reports, correctness sidecars, and complete program coverage. Its profiler capacity default20000 supports the long exact state scan. Regenerate `summarize_perf` and `summarize_fused_topology` from the final source. Keep device-kernel timings distinct from host blocking timings. The validation queue uses a fixed artifact directory, whereas profiling supports `QWEN_FUSED_ARTIFACT_DIR`; avoid unintentionally splitting the evidence tree.
5. Run the final checker with the explicit manifest, reconcile README/pattern assessments/topology and the supported stop-rule ledger, then request independent stage review. Do not infer later model/serving stages from this decoder work.

## Residual checker limitations for review

The targeted change does not fully validate seed identity, repeat-position uniqueness, prefill PCC, layer identity, or capture count in every stress family; extended stress also lacks its own total-row-count assertion. The final-source checks bind production F for many artifacts but do not comprehensively bind derived watcher audits, pytest XML, profiler summaries, all reference/test hashes, and native build identity. Profiling summaries carry provenance links and CSV hashes, but the final checker does not recompute all of them. Final regeneration and independent review remain necessary. No gate or oracle was relaxed by this review.
