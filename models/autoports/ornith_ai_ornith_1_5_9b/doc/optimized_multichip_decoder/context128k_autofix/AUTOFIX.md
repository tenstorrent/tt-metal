# AutoFix: long-context batched SDPA accuracy

Status: repair passes all 82 final device jobs, profiling and consolidated accuracy/provenance audit. Round 1 found only a performance-instrumentation mismatch, now remediated by 24 separate diagnostics-off timing runs; independent follow-up review round 2 returns clean-pass (`../stage_review_round2.md`).

The unchanged acceptance contract is output/state PCC >= 0.995, replicated-rank PCC >= 0.99999, real TP4, changing-input trace replay with allocation tracking including program-cache allocations, and valid nonaligned lengths. No tolerance or context limit was reduced.

## Diagnosis and discriminating evidence

See `AUTODEBUG.md` for the independent source investigation. The failing default (`80ad61a`) used K64 for every decode context. TTNN SDPA stores running statistics and intermediate output in BF16 even with FP32 destination accumulation. At B32/128K, its three workers per request process about 683 chunks each, compared with 32–33 in the single-chip reference. This exposes long sequences of BF16 rounding. No page/position truncation or state-restore contradiction was found.

The exact error contribution of each internal arithmetic stage was not measured on identical captured Q/K/V. The causal attribution to the SDPA key-chunk policy is supported by controlled whole-stack interventions with the same seed, capacity, page permutations, weights and unchanged other policies:

| Control | Result |
|---|---|
| Original K64/HiFi2 | Fails step 6, row 11: 0.99142712 |
| Restore outer prefill chunks to 1024 | Fails step 1, row 16: 0.99483514 |
| K64/HiFi4 only | Fails step 1, row 16: 0.99452394 |
| K128/HiFi4, first eight steps | Pass; insufficient alone to select a repair |
| K128/HiFi2, 1,024 steps | Fails step 1,013, row 12: 0.99468619 |
| K256/HiFi2, 1,024 steps + states | Pass, minimum 0.99571836; 6.746858 ms |
| K512/HiFi2, 1,024 steps + states | Pass, minimum 0.99641383; 6.763172 ms |
| Completed stage 4, matching 1,024 steps + states | Pass, minimum 0.99865484; 28.897929 ms |
| Repaired default `b1d5016`, matching 1,024 steps + states | Pass, minimum 0.99571836; 6.734447 ms |

Full commands, immutable snapshots and results are in `context128k_accuracy*_command.json`, `context128k_stage4_answer1024_32*`, and `final_v2_context128k_answer1024_32*`. The original reference is preserved at `generated/multichip/context128k_answer1024_state_stack_32.pt`.

## Minimal production repair

`tt/multichip_decoder.py` now chooses the smallest power-of-two K >= 64 that bounds local key chunks to 200 per worker for the request's allocated page-table capacity. The SDPA program explicitly caps workers per head at 16, matching the selector. The chosen K stays fixed during trace replay; no position readback or host fallback is introduced.

Scalar decode uses 64 available cores and native batch uses 110. At B32, capacities around 4K, 64K, 128K and 262K select K64, K128, K256 and K512. Capacity can exceed the current position, so conservative allocation may select a larger K before it is needed. The bound is an empirical numerical-risk policy, not a mathematical PCC guarantee.

Attention tables round to a multiple of K/32 pages while preserving existing minimum alignment. Padding contains local physical page zero before pool offsets are applied. Cache-update tables remain unchanged. KV/recurrent storage and supported logical context remain unchanged. The maximum extra native table storage at B32 is 1,536 bytes per chip; temporary SDPA circular buffers grow with K.

## Validation and harness repairs

- Host-only exhaustive selector audit: 270,336 checks across every B1–32 and page count through 262,144 tokens; bound, smallest legal K and table alignment pass (`final_sdpa_policy_host.json`).
- Final default original failure: all 1,024 evolving steps and terminal conv/recurrent/KV checks pass with tracking on and skip-program-cache off.
- The complete final queue is rerun from the integrated default, with new watcher coverage at K transitions, heterogeneous pools, and the 128K workload. See `final_v2_all_jobs.json` and its resume list; all 82 jobs and the consolidated audit pass.
- An initial frozen-stage-4 control incorrectly received the new BF8 collective workspace. Its 32,768-byte BF16 CB exceeded the 17,408-byte BF8 allocation. The harness now chooses the frozen workspace with `--stage4`; the corrected full baseline passes. This was an invalid diagnostic fixture, not a decoder failure.
- An initial virtual-tail padding test exceeded TTNN's legal table-width/cache-block contract. It is replaced with physically matching cache capacities. Additional capacity tests explicitly sample state pages; ordinary pool tests retain complete-state comparisons. No production contract was altered to accommodate this fixture.

No resets were required. Failed jobs closed cleanly and subsequent mesh smokes passed. No production changes were made while device jobs ran. No full-model or vLLM work was performed.

Final diagnostics-off matched 128K timings are 28.889897815 ms before and 6.731745554 ms after, with unchanged PCC gates. The control-table timings above are retained historical tracked-run evidence; final reported performance comes from `../final_performance.csv` and `../final_timing_audit.json`.
