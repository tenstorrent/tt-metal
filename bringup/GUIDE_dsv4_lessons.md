# Lessons from the DeepSeek-V4.1-Flash bring-up (evidence guide)

Source: models/demos/blackhole/deepseek_v41_flash, branch `ssinghal/dsv4p1` (tip c29a56370c8 when this was written), 3 weeks, ~12 parallel agents on up to 15 hosts.
This file is EVIDENCE, not procedure: the skills in `.claude/skills/tt-*` hold procedures and cite the ids below (L-xx).

How to read it. Every row has a commit and a log or notes path. Numbers are those of that commit and host, NOT of any other model or commit:
for a new model they are priors to compare against, never to quote. "notes" = `models/demos/blackhole/deepseek_v41_flash/<file>` at c29a56370c8.
Logs are under `/mnt/tt-data/ssinghal/dsv4-logs/`. Where a log path was not recorded the row says "no log path" and counts as a hypothesis.
Status (section 1 rows carry it; rows in sections 2-3 are measured by an agent unless the evidence says "no log path" or "est."; treat those as inferred): M = measured by an agent on device/CPU, R = retracted/corrected claim, I = inferred/not measured.

## 1. Retracted or corrected claims (the most valuable rows)

| id | claim that was made | what is true | st | commit | evidence |
|---|---|---|---|---|---|
| L-gap | "about 40% of prefill wall time is non-kernel gaps (kernel 4.6k tok/s vs 2.7k observed)" (from a research agent) | gaps are about 8% at 4k and 3% at 30k; the cost is in kernels | R | d1035b283c8, fc75b84d520 (branch ssinghal/dsv4p1-pf-prof2) | `pf_pf_prof2_a4k.log`, `pf_pf_prof2_a4k.json`; study `/mnt/tt-data/ssinghal/prefill_study/own_breakdown.md` |
| L-batch | "batch is free in the expert matmul / decode is at the roofline" | decode 44.6 -> 83.4 ms/token B=16 -> 128; expert weight streaming explains ~10-15% of growth; mHC grew 113 -> 504 us/layer (4.5x for 8x tokens), ~335 us/layer unattributed (real-routing imbalance suspected) | R | b8468e1c5b2 | notes `BATCH_SCALING_NOTES.md`; `bs_A128_h44/`, `bs_*_h44` |
| L-tile | "small activation tiles (M=4) make the expert matmul cheaper (tile padding wastes the matrix unit)" | at LoFi the cost is 17.05 cycles per weight tile for 1, 4, 8, 16, 32-row tiles and bf16/bfp8/bfp4 weights alike; the 6x microbenchmark was an artifact | R | no commit (transcript 2026-10-03, MoE agent) | no log path recorded: treat as hypothesis, re-measure with the op under test |
| L-bfp4-claim | "bfp4 costs ~0.02-0.03 logits PCC" | 0.05-0.08 logits PCC on a real ISL-128 prompt (TF argmax 13/11/9 of 16 vs 15/15/13) | R | bf140483f4b (default stays bfp8) | notes `BFP4_NOTES.md`; `h43b_pd8_1.log`, `h43b_pd4_1.log` |
| L-spec012 | "spec decode accepts only 0.12 drafts per round" | seeding bug of the drafter state; after the fix k=3 gave ~2.3 accepted/round on GSM8K and 1.9-2.0x plain decode | R | 2b71af27ee1, 43127e4bf6c | notes `SPEC_NOTES.md` |
| L-seed | spec exact after "tail replay seeding" | needs ring_rows >= 255 + k; with 160 rows the replayed K/V were wrong (row-0 PCC 0.852 vs 0.993) and spec diverged at token 1 after a real prefill | R | 0d3f7b18028 | notes `SPEC_NOTES.md` section "Bug found and fixed"; `spec8_tail_2k.log` |
| L-attn-gap | "the attention gap vs reference explains the 40-layer PCC 0.88-0.90" and "device indexer is wrong (PCC 0.97 at 4k)" | 0.97 was the noise floor of a synthetic-cache test (two reference variants differ by the same); the chain gap was reference noise + router flips on random/9-token prompts | R | 8168c7f53bf (chain-error tools) | `reference/chain_analyze.py`; dirs `/mnt/tt-data/ssinghal/dsv4-chain-m`, `dsv4-chain-f32`; transcript 2026-10-03 |
| L-works | "prefill works" after tests at one shape | failed at: single chunk with S_pad == C >= 1024 (zero-length slice corrupted state of half the users), n8 == 1 (U=2, C=128) segfault, U=8 grouped MoE hang, U=8 L1 circular-buffer clash, a second ISL in the same process (DRAM leak), 64k B>=32 OOM after 50 min | R | b8b86bf3ac6, 7cc5fea1d85, a7fca4013f0, b46ea9bea29, 9411b8398a4, cf5e0a54cc5, 5a15bdd2ccd | commit messages; notes `INTEGRATION_NOTES.md` (G4 ladder, "second ISL OOMs") |
| L-host | "unified MoE gives odd/inconsistent first tokens" (tokens 0, 2, 90755) | host .31 is faulty (differences only in mesh rows 0 and 1; same configs on .41 and .48 are consistent, 6 determinism runs clean) | M | d1035b283c8, fc75b84d520 | `pf_pf_prof2_a4k46.log`, `pf_pf_prof2_a4k48.log`, `pf_pf_prof2_a4k.log` |
| L-host31 | (same) | rule: faulty host = wrong first tokens in some mesh rows, differs between identical runs; discard its numbers | M | same | same |
| L-x-host | single-run gains of a few % | baseline timing differs up to ~10% between hosts; only paired in-process A/B counts (scenario list env, `DSV41_PFA_AB`, `PF_ASYNC_LIST`) | M | 4244f9833c6, 85fb5d759c0, 9ee562a7a1a, 33798f79086 | demo/text_demo.py session modes; `tools/integ_e2e.sh`; no single log (transcript 2026-10-05) |
| L-first | step-0 / first-call timing in averages | first call compiles: 82 ms argmax readback at step 0 vs 4 ms later; first TTFT 91.8 s vs 86.8 s warm (B=128, 4k); reconfigured B=4 TTFT 2.8 s vs 5.9 s first build | M | test_decode_closed_loop warm-up change (uncommitted at time, 2026-10-02); 845f3d4cf66 | notes `RECONFIGURE_NOTES.md` section 3; B=128 numbers: no log path |
| L-decms | decode ms/token differs by session type, size and sign vary by batch, cause NOT isolated (candidate: grid ran with DSV41_MEMLOG=1; short scenario inside a model built for long ISL). Dedicated single-scenario GSM8K (grid G0): B=4 41.9, B=8 51.5, B=16 53.0, B=32 65.1, B=64 68.7, B=128 87.7. Multi-scenario G1 sessions, gsm8k cells: B=4 48.2, B=8 54.8, B=16 54.9, B=32 65.9, B=64 81.2, B=128 107.8; 4k cells same sessions: B=16 46.9, B=64 67.5 (so dedicated B=16 GSM8K 53.0 is SLOWER than 4k 46.9). Earlier dedicated B=16 runs: 47.4, 47.1 (6 ms below the G0 run). | I (open) | 21752eb0ab9 (G0 group added) | `grid_b{4,8,16,32,64,128}_G0.log`, `grid_b*_G1*.log` (G1 session logs, exact names not rechecked), `final_plain_b16_gsm_ref.log`, `h47i_demo_gsm_b16.log` |
| L-hw8 | hangwatch default (8-10 min silence) is a safe hang detector | killed 2 healthy builds: weight load + host tables are silent for 9-15 min; grid used 45 min for builds | M | db68dc7e754 (hangwatch.sh, default now 10), 5eac9d73ba7 (backlog) | notes `PREFILL_BACKLOG.md` "Infra"; `tools/hangwatch.sh`; RECONFIGURE_NOTES (15 min Engram load) |
| L-ghost | agent killed a "duplicate" process holding a build slot | other agent's gate job; no duplicate | R | none | transcript 2026-10-05 00:17: never kill foreign jobs without asking |

## 2. MoE correctness findings

| id | finding | commit | evidence |
|---|---|---|---|
| L-router | correction bias 10-14 in a bf16-only gate op: device matched the reference expert set for 12-39% of tokens; fp32-emulated hi/lo bias split gave 92-93% (torch) and exact-router device runs follow | ca2f838e365 (first commit of tt/router.py) | transcript 2026-10-01; test_router_exact.py, test_router_v2.py; no log path |
| L-clamp | moe_compute has no swiglu clamp; the model clamps at 10; unified op has it | 175548bbe0f | notes `PREFILL_UNIFIED_MOE_NOTES.md` ("applies the swiglu clamp") |
| L-cf | dispatch capacity factor 2 overflowed under skew (chip load 33,408 rows vs capacity 33,120) and lost tokens: first-token PCC 0.953 / 13 of 16 vs baseline 0.9655 / 16 of 16; CF=4 gives 0.9569 / 16 of 16; a loud capacity check was added | 8428b85d668 | notes `PREFILL_UNIFIED_MOE_NOTES.md`; `pf_moe_*` logs (no single path recorded) |
| L-skew | with REAL tokens one expert gets ~63x the mean; real-token dispatch/experts/combine 3.9/6.1/3.1 ms vs 1.5/2.8/1.0 ms in the random-token run (random UNDERSTATED MoE time there), while the profiler agent's random-token profile showed extreme skew (many chips at 0 ms of experts). Opposite signs: random-token MoE profiles are unreliable in either direction. | M (real-token numbers) / I (random-token comparison) | 5eac9d73ba7 | notes `PREFILL_BACKLOG.md`, `PREFILL_UNIFIED_MOE_NOTES.md`; `pf_moeoverlap_real4096.log`, `pf_moeoverlap_r512rand.log` |
| L-rs | `reduce_scatter_minimal_async` over the token dim corrupts random 32-row blocks (nondeterministic); `ttnn.reduce_scatter(dim=hidden)` + `all_to_all_async_generic` is exact at N=256..4096 | b3e630512e6 | notes `PREFILL_UNIFIED_MOE_NOTES.md` Findings; `tests/test_unified_moe.py` with `DSV41_UM_DEBUG=1` |
| L-pad | packed mHC mixes kernel silently returned wrong mixes for T=12 and 20 (verify rows PCC 0.83-0.95); padding T to a multiple of 8 gives 0.9995+; B=4 drafter T=5 asserts | 450e1196f7c | notes `SPEC_NOTES.md` |
| L-single-chunk | zero-length slice in single-chunk prefill (S_pad == C >= 1024, sparse) corrupted first half of users | b8b86bf3ac6 | commit message |
| L-u8 | grouped T=256 moe_compute + column split hangs intermittently at U=8 in traced chunk replay; unified op avoids it | a7fca4013f0, 083cce794ef | commit messages; `prefill2_u8_smoke3.log` |
| L-oob | finished users kept in the batch with positions past the cache end corrupt routing and hang dispatch/reduce_scatter (5 hangs); clamp positions | 450e1196f7c | notes `SPEC_NOTES.md` "Hangs" |
| L-chain | real 128-token prompt: chained 40-layer logits PCC 0.98 at step 0, 13-14/16 argmax; random 9-token prompt: 0.74-0.87 (router flips 88-100% at layer 39; real prompts 0.2-0.5 per layer); bf16 reference vs fp32 reference logits PCC 0.90-0.92 on that chain; "massive channels" hypothesis excluded | 8168c7f53bf | `reference/chain_analyze.py`; `dsv4-chain-m`, `dsv4-chain-f32` |
| L-chain-early | HISTORICAL early measurement, early state, NOT the final implementation (bfp4 routed weights, no fp32-accumulate fix, slow mHC): 40-layer chain, each layer fed its own device output: final-stream PCC 0.758, logits 0.752, 8/16 tokens match; a layer alone 0.996-0.999; ~0.003-0.005 PCC lost per layer. Final state (bfp8 + fp32 accumulate) is in L-chain / section 1 numbers. | I (source: plan file `/home/ttuser/.claude/plans/lets-create-a-new-zippy-flurry.md` lines 5-6, not a log; no repo log path) | none | plan file only |
| L-nearTie | PCC 0.986 at U=1 ISL 4096 / chunk 2048 was a top-512 near-tie flip from changed bf16 rounding (0.99595 / 0.99961 with sparse on) - explained, not a bug | 06ef0434b43 era; notes only | notes `PREFILL_PERF2_NOTES.md`; `prefill2_e2e_cs_2kb.log` |
| L-open | unexplained at the end: spec divergence at position ~245 (2.1-logit gap), ragged-length prompts at ISL 0.9-1.6k (14/32 users diverge, gaps 2.3-3.8) | fa39f8e7e94 | notes `SPEC_NOTES.md` |
| L-bfp4 | bfp4 vs bfp8, B=64, 512 GSM8K: 95.3% vs 96.1% (McNemar p=0.42), 63.8-64.6 vs 68.4-69.2 ms/token; teacher-forced PCC 0.91 vs 0.98; accuracy test saturated; user decision: default bfp8 | cd8b7c7dd48, bf140483f4b | notes `BFP4_NOTES.md`; `h43c_g8.log`, `h43c_g4.log` |
| L-stale | tests/test_e2e_gsm8k_device.py gave 0/64 on both bfp4 AND bfp8: harness stale vs main (control run) | bf140483f4b | notes `BFP4_NOTES.md` anomaly; `/mnt/tt-data/ssinghal/dsv4-e2e-dev/bfp4_a`, `bfp8_h43b` |
| L-gptoss | batch-128 divergence in the GPT-OSS MoE was first blamed on moe_compute; root cause was an SDPA decode core-grid mismatch | ff47a65a6f5, 42767d3d204 | git log of ssinghal/dsv4p1 (gpt_oss commits 3288124a91f, ec413bad4f8): bisect by same-prompt x N and idx % 8 patterns |

## 3. Architecture and perf facts that shaped decisions (priors only)

| id | fact | commit | evidence |
|---|---|---|---|
| L-onecopy | decode moe_compute copy (497 MB/chip/layer) + unified prefill copy (451 MB) do not fit 40 layers (2371+2274 vs ~3880 MiB/bank); unified op reading the decode ring layout in place fits; PCC ring vs copy 0.99985-0.99994 | bd3c0c7bc5d, c6a8151f662 | notes `PREFILL_UNIFIED_MOE_NOTES.md`; `pf_onecopy_ring4096.log`, `pf_onecopy_mainverify.log` |
| L-uni | unified prefill MoE, 40 layers: isl4k_b16 2951 -> 4965 tok/s, isl8k_b4 2708 -> 4547, isl32k_b8 3114 -> 5364 (same-session grid baseline) | 175548bbe0f, e3eb9124a8d | notes `PREFILL_UNIFIED_MOE_NOTES.md`; `pf_onecopy_scen40.log`, grid logs `grid_b*_G4k.log` |
| L-linear | replay time is linear in tokens (1.07-1.24 ms per row-token for C=128..4096); batch cannot amortise prefill | 882ac97c53c | `/mnt/tt-data/ssinghal/prefill_study/own_breakdown.md` |
| L-pf-break | prefill chunk (U=4, C=1024) breakdown: sparse_sdpa 18% (gather-bound, ~280 GB/s), mHC 14%, Engram 16%, MoE 28%; host 5-16% | d1035b283c8 | `pf_pf_prof2_a4k.json` |
| L-reconf | one model build serves many batch sizes; 40-layer reconfigure 67-98 s vs 60-90 min build; outputs equal fresh runs; leak found and fixed (per-T weight-row cache) | 147a84738a3, 95f909595ea, b8149b25ef3, 845f3d4cf66 | notes `RECONFIGURE_NOTES.md`; `pf_reconf_s40a.log`, `pf_reconf_s40b.log` |
| L-slots | 12 hosts building at once saturate NFS (65+ min builds, 5 MB/s per host); build slots (cap 5, later 10) + start stagger | 140739a9d9a, 0068441f680 | `tt/build_slots.py` docstring |
| L-chunk | unified chunk budget 2048 tokens per mesh row for every batch size (8192 at U=4 ran out of DRAM at 64k; dispatch buffers ~0.7 GB + 0.36 GB/chip at 4096 tokens/row with CF=4) | 0068441f680 | commit message; notes `PREFILL_UNIFIED_MOE_NOTES.md` |
| L-paged | paged-KV 64k decode B=16..128: 61.5 / 73.7 / 89.7 / 124.8 ms/token; chunked stage_commit fixed batch-128 pool OOM (second pool-sized buffer) | db0a20b75e0 | notes `PAGED_NOTES.md`; `h46k_b128.log` |
| L-spec | spec decode (k=3) 1.9-2.0x plain at B=16 on GSM8K (acceptance matches CPU); not a win at long ISL / book text (0.6-1.5 accepted per round); B=128 impossible with <=32 verify rows per call | 8fd5ac546a0, 2b71af27ee1 | notes `SPEC_NOTES.md` |
| L-fusion | fused collective+matmul ops: est. 2-4% only, weeks of C++: skipped; bfp4 ~1% for prefill; fp8 sparse-sdpa key table fails accuracy gate | 8abba965d0f | notes `PREFILL_BACKLOG.md` |

## 3b. Reference policy for the next model (user decision, not a DSV4 measurement)
DSV4 ran a full 40-layer CPU chain (S=128, 16 users ~1 min/layer, hours of CPU) and its dumps (`dsv4-prefill-s*`, `dsv4-chain-*`). Policy for MiMo: no full-model CPU run;
per-layer teacher-forced checks + a short CPU chain + separate prefill/decode checks + required GSM8K/determinism/first-token/hosted-reference gates.
Rationale rows: L-chain (random/short prompt chains misled; reference itself noisy), L-attn-gap. Hosts have 566 GB RAM (527 GB available), 64 cores; MiMo full model ~618 GB bf16 does not fit, ~13 GB experts per layer. Status: I (policy, unvalidated).

## 4. Process lessons (from the 2026-10-04 and 2026-10-06 retrospectives)

* Integrate early: features verified in isolation (indexer, spec, traced prefill) failed at full scale; a 5-minute two-ISL, 40-layer run would have caught the DRAM leak before 50-minute jobs died (commits cf5e0a54cc5, 5a15bdd2ccd).
* Gate long runs on a short check; time-box agents; cap agents by hosts (6 agents on 4 hosts queued measurement agents for hours).
* Re-check docs and measured data before making a claim; label inferred statements; each retraction (L-tile, L-batch, L-bfp4-claim, L-spec012, L-gap) sent agents down a wrong path.
* Agents work in worktrees; overlay copies of old trees later delete newer work: always check `git diff --stat` and deletions at merge (many merges applied `git apply --check` first).
* Stopped agents may not be resumable (refused on 2026-10-03); keep branches, logs and a handoff note so a fresh agent can continue.
* Hangs: signature `FDMeshCommandQueue::wait_for_outstanding_reads` at the first op on .42 (`triage/hang_42_1085121_0923.txt`, `triage/hang_42_1090158_0951.txt`), resolved by `tt-smi -glx_reset`; a watcher attached to a lock waiter caused two spurious resets (h46k).
