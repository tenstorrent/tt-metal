# bfp4 vs bfp8 routed experts, end to end (DeepSeek-V4.1-Flash, BH Galaxy 4x8, B=16, 40 layers, Engram ON)

Host .45 (all runs, same host, same session conditions, each config run twice, fp32 accumulation on, exact router, bfp8 shared expert/attention/head in both).
Test: tests/test_e2e_prefill_decode.py (ISL 128, B=16, traced chunk prefill -> paged pool -> traced device-loop decode, closed loop 16 steps,
teacher-forced vs the CPU-reference prefill dumps dsv4-prefill-s128). Overlay /mnt/tt-data/ssinghal/wt/h43b (no code changes: package copied from main at 5a15bdd2ccd; main has since moved (bf140483f4b + prefill notes/test edits), so the numbers are for 5a15bdd2ccd, not re-run on bf140483f4b).
Logs: /mnt/tt-data/ssinghal/dsv4-logs/h43b_{pd8_1,pd8_2,pd4_1,pd4_2,gs4,gs8}.log. Selector: MOE_COMPUTE_BFP8_WEIGHTS=1 (bfp8) / 0 (bfp4).

## Results (VERIFIED = measured by me on device)
| weights | ms/token (run1 / run2) | tok/s/user | first-token PCC (argmax) | teacher-forced PCC, steps 0/1/2 | TF argmax (of 16) | GSM8K 64q |
|---|---|---|---|---|---|---|
| bfp8 | 46.7 / 47.1 | 21.4 / 21.2 | 0.972 (14/16) | 0.983 / 0.975 / 0.980 | 15 / 15 / 13 | 81.2% (52/64) device vs 82.8% (53/64) reference: earlier run base_a, NOT re-run (see anomaly) |
| bfp4 | 44.1 / 44.0 | 22.7 / 22.7 | 0.919 (12/16) | 0.906 / 0.906 / 0.937 | 13 / 11 / 9 | NOT MEASURED (harness broken, see below) |

- Delta: bfp4 saves ~2.7-3.1 ms/token (6-7%), i.e. ~+1.3 tok/s/user, for -0.05 to -0.08 logits PCC and TF argmax 13/11/9 vs 15/15/13. Run-to-run spread 0.4 ms (bfp8) / 0.1 ms (bfp4). Results are bit-identical in PCC between repeats.
- Consistent with the cost model (~42 us/expert saved on the busiest device, ~-160 us/layer ~ -6.4 ms/token predicted): INFERRED prediction was 2x too optimistic; measured saving is ~3 ms.
- bfp4 cache was already warm on NFS (/mnt/tt-data/ssinghal/dsv4-weight-cache/layer_*/moe_*_bfp4.*, keyed by tag), so no cold build was needed; build 1144 s vs 1565 s (bfp8) on first read.
- Per-layer moe_compute kernel us (item 4): NOT measured separately (stopped on request); only the end-to-end delta above.
- B=64 demo numbers (gsm8k_b64 via text_demo, bfp8 vs bfp4): NOT obtained. Job dg8 was started and killed at the user's wrap-up request before it produced results. Earlier bfp8 reference from INTEGRATION_NOTES G4 (not mine this session): b64 60/64 correct, 68.7 ms/token.

## Unresolved anomaly: tests/test_e2e_gsm8k_device.py gives 0/64 (bfp4)
- bfp4 run (out dir dsv4-e2e-dev/bfp4_a): first tokens 16/16 match reference in all 4 batches (device prefill fine), but teacher-forced decode logits PCC ~0.07 already at step 0, closed loop diverges at token 1 for all 64, 0 users reach EOS, GSM8K 0/64.
- Control: the SAME harness with bfp8 (out dir dsv4-e2e-dev/bfp8_h43b, same overlay/session) is ALSO broken: step-0 PCC 0.055, divergence at token 1 for 64/64. So this is NOT a bfp4 effect. The traced e2e test with the same weights/caches gives sane decode (above), and the earlier harness run base_a (bfp8, older tree) gave step-0 PCC 0.998 / 81.2%.
- Conclusion: the harness has regressed against current main (decode state/trace path changed after base_a). Cause NOT isolated (INFERRED: its private decode/restore/trace setup via DSV41DecodeChain + DSV41Decoder no longer matches main's decoder; candidates: step-state/paged-pool changes, snapshot/restore of state tensors). Not investigated further.
- Therefore no valid bfp4 GSM8K accuracy exists; bfp4 accuracy evidence is first-token/teacher-forced only. Older (not mine, bfp4 chained layers 0-9) data showed layer-9 PCC 0.969 vs 0.990.

## Capacity (INFERRED, not measured)
bfp4 expert tiles are 576/1088 of bfp8: ~0.44 GB/chip/layer (bfp8) -> ~0.23 GB, i.e. ~8.3-8.5 GB/chip over 40 layers, which could fund more users. Decision (user): stay bfp8 default (bf140483f4b); MOE_COMPUTE_BFP8_WEIGHTS=0 still overrides.

## Ops notes
- .43 was occupied by another job (text_demo isl4k/8k session, flock holder). .42 (idle) hung on the first device op again (mesh opens, no progress, 15 min silent): hangwatch triage /mnt/tt-data/ssinghal/dsv4-logs/triage/hang_42_1090158_0951.txt (+.console), one tt-smi -glx_reset cycle done (rc=0), nothing more run there. All measurements moved to .45 per lead.
- Runner scripts (not part of any diff): wt/h43b/{run.sh,chain.sh,hangwatch_wrap.sh}. All jobs stopped; nothing left running by me.

---
# Addendum: larger same-prompt GSM8K comparison through the demo path (main bf140483f4b)

Setup (VERIFIED): host .44, overlay /mnt/tt-data/ssinghal/wt/h43c (= main bf140483f4b + scenarios gsm8k_b64_o{0,64,...,448}: GSM8K test questions 0..511 in 8 batches of 64,
instruct chat template, greedy, stop at EOS, max_seq_len 512, max 352 new tokens; 352 (not 384) so that prompt+gen stays under the dense-context limit 512 while the longest
questions (~146 chat-template tokens) still fit; an attempt with seq 640 hit the indexer path and failed in topk_large_indices, nothing scored from it). One process per format, one model build each,
all 8 batches in-session. bfp8 = MOE_COMPUTE_BFP8_WEIGHTS=1 (default), bfp4 = 0. Logs dsv4-logs/h43c_g8.log, h43c_g4.log. Scorer: wt/h43c/score_gsm_big.py (score_gsm extraction; answer
= final \boxed / #### / last number, vs gold in datasets/gsm8k_test.jsonl). Patch for the scenarios + DONEFLAGS log line: wt/h43c/gsm_offsets.patch (git apply --check OK on main bf140483f4b, 9 files, no deletions).

| weights | correct /512 | accuracy (Wilson 95%) | unfinished (no EOS in 352 tok) | correct among finished | decode ms/token (per batch of 64) | tok/s/user |
|---|---|---|---|---|---|---|
| bfp8 | 492 | 96.1% (94.0-97.5) | 18 | 487/494 | 68.4-69.2 (mean ~69.0) | ~14.5 |
| bfp4 | 488 | 95.3% (93.1-96.8) | 20 | 482/492 | 63.8-64.6 (mean ~64.4) | ~15.5 |

Paired (same 512 prompts): both right 483, only bfp8 right 9, only bfp4 right 5, both wrong 15. Exact McNemar two-sided p = 0.42. bfp8 - bfp4 = +0.78 pp, paired 95% CI -0.65..+2.21 pp.
Earlier 64-question demo run (first 64 questions, seq 512/gen 384, h43b_dg8/dg4 logs, scored by lead): 64/64 vs 62/64, consistent with this.

Reading: at B=64 bfp4 saves ~4.6 ms/token (6.7%) and the accuracy difference is NOT statistically distinguishable at n=512 (0.8 pp, CI includes 0), though the point estimate favours bfp8
(9 vs 5 discordant). With 14 discordant pairs this test cannot resolve a ~1 pp effect; ~2000+ questions would be needed to tell a 1 pp difference. "Correct" for an unfinished generation can be a
coincidence of the last number in the truncated text (bfp8: 5 such, bfp4: 6); the unfinished counts are reported separately for that reason. This does not contradict the lower teacher-forced PCC
(0.91 vs 0.98 on ISL-128 prompts): task accuracy on GSM8K is saturated (~96%) and insensitive to the logits-level drift measured there. Whether harder tasks show a larger bfp4 loss is NOT measured.
Note: the 0/64 anomaly of tests/test_e2e_gsm8k_device.py (above) is unrelated: it is a stale-harness problem on both formats.
Build times (NFS-bound, informational): bfp8 2403 s (cold), bfp4 3712 s (also cold, both ran after other jobs evicted the page cache).
