# Initial P1 result audit

CPU-only inspection on 2026-11-16 found the preserved initial P1 result internally consistent. All four runner checks passed and the process exited 0. This audit did not import TTNN, access devices, rerun HF, or alter canonical results, runtime code or caches. It does not choose the final precision policy.

## Evidence and recomputation

Paths below are relative to this datatype-sweep directory unless stated otherwise:

- `p1_initial_two_v1.json`: preserved runner checks, exact command, thresholds and aggregate results.
- `p1_initial_two_v1.runtime.json`: pinned HF loader/snapshot, EOS trimming and four teacher-forcing replay-counter records.
- `p1_initial_two_v1.provenance.json` and `logs/p1_initial_two_v1.log`: command/environment, source identity, successful exit and runner summary. Run interval: 2026-11-16 02:40:38.405883 to 03:23:39.781605 UTC, 2581.375 seconds; this elapsed duration is not a decode benchmark.
- `p1_candidates_two.json`, `configs/e8_two_hifi4.json`, `configs/inherited_e4_lofi.json`: exact builds; both retain 262,144-token capacity and batch 1.
- `../eval_readiness/precision_reasoning/text.jsonl`, its `scores/e8_two_hifi4.pt` and `scores/inherited_e4_lofi.pt`: original sampled tokens and per-position TT scores. `p1_texts_decoded.json` preserves their readable provenance.
- `p1_hf_loader/loader.json`: original HF model, 48 layers, exact checkpoint dtypes and original BF16 PLE rows; pinned revision `812d96e10a0dd82b69d749e68693698cf6c36daa`.
- `p1_decisions_initial.json`: compact audited decisions and scope limits.

The audit extracted only host statistical functions from the installed `runtime/readiness_check/run_precision_reasoning.py` and recomputed every saved candidate statistic and decode statistic from the CPU score caches. All match the runner's five-decimal outputs. Text identity, score targets versus answer tokens, tensor finiteness, score-cache roles, build kwargs and current code/config identities also match. The decoded artifact contains exactly the cached IDs and sampling metadata.

The two texts were sampled by `e8_two_hifi4`, seeds 1000 and 1001, with temperature 1, top-p 0.95, top-k 20 and chat thinking enabled. Prompt lengths are 111 and 113; answer lengths are 1213 and 2222, totaling **3435 scored positions**, above the unchanged minimum of 1000. The fixed-count generator produced 6144 raw tokens per request; the adapter retained each answer through its first EOS, excluding later output. Full prefills score the retained answer positions; decode checks cover each answer's final 256 positions.

## Decisions supported by this run

| Policy | Evidence | Initial P1 conclusion |
|---|---|---|
| `inherited_e4_lofi` | ΔNLL −0.00080549 ±0.00077607 nats/token; bound max(2SE, 0.004)=0.004. KL×1000=1.257088 ≤5.0. Reference argmax agreement 0.9892285. | Actual selected-config runner gate passed against `e8_two_hifi4`. |
| `e8_two_hifi4` | Original-HF comparison: KL(reference∥HF)×1000=3.62242 ≤4.0; argmax agreement 0.98719; ΔNLL(HF−reference)=+0.00069 ±0.00160. | Qualifies as the reference via the measured original-HF bound. This is not a self-comparison or a separately emitted selected-config gate for this policy. |

The installed `reference_selected_check` uses the measured HF KL bound when the selected config is the reference and no higher-precision candidate is provided. The existing HF result satisfies that rule, but this preserved run explicitly selected `inherited_e4_lofi`. If `e8_two_hifi4` ships, rerun the canonical runner with `--selected e8_two_hifi4`; do not relabel this result. Whichever policy ultimately ships must be the selected ID in the final canonical `doc/eval_readiness/precision_reasoning.json`.

The runner's `within_noise` implementation also permits nonpositive ΔNLL despite a larger KL; that exception is unnecessary here because both the ordinary NLL and KL bounds pass independently. No threshold was relaxed.

## Trace and HF scope

Each policy has two recorded teacher-forcing calls. Every call produced 256 sampling replays, **255 model replays**, 256 token refreshes and 256 readbacks/synchronizations. The first prediction comes from prefill, explaining 255 model replays for 256 predictions. The adapter explicitly resets to greedy sampling for `next_input` calls and asserts positive model replays. Both policies match 506/512 target tokens (0.98828125); their 512 predictions agree exactly. The decode top-1 difference is zero, above the runner's −0.01 bound.

These counters establish traced execution for the P1 decode check. They do **not** provide TTFT or decode t/s/u: there is a host readback/synchronization for every predicted token and no corresponding warmed timing measurement. Keep P1 correctness evidence separate from teacher-forcing performance ranking and the post-selection token-out benchmark.

`--hf-positions 2048` is a stopping minimum across whole texts in `score_hf`, not an exact row cap. After 1213 positions from the first answer, the runner scores all 2222 from the second; **3435 HF positions is expected**, not a mismatch. The original HF aggregate is corroborated by the successful runner/log and exact-loader provenance. The runner does not persist its per-position HF score rows, so this audit cannot independently recompute the HF KL without rerunning the original model; no HF execution was performed here.

Two reasoning prompts meet the configured gate but remain a small prompt sample. SE uses 128-position blocks, not 3435 independent prompts. KL uses the reference top-64 tokens plus one tail bin and is a lower bound on full-vocabulary KL. This initial pass does not replace the main/natural-text accuracy gates, shared blind qualitative suite, context checks, remaining candidate P1 work, or final traced performance selection.
