# Spec decode matrix (branch ssinghal/dsv4p1-spec: spec-adapt + spec-rows merged on ssinghal/dsv4p1 c29a56370c8)

Scope (coordinator decisions): GSM8K prompts only (short context, 40 layers, greedy, Engram on), B = 4, 64 measured here; B = 8/16/32 re-measured on the merged branch;
B = 128 measured with fixed k and then **dropped: spec decode is OFF by default at B >= 128** (`DSV41_SPEC_B128=1` forces it). One scenario and ONE spec pass per process.
Speedup = spec tok/s/user / plain tok/s/user of the SAME process (the plain decode of the demo, before the spec pass). Plain decode ms/token varies 10-25% from run to run on the
same scenario (e.g. B=64: 69.7 / 78.9 / 88.6 ms in three processes): compare round ms too. `cycle:adapt+kA+kB:4` = policies interleaved in windows of 4 rounds on the same stream.

## Results
| B | policy (T = U*(1+k) rows/mesh row) | acc drafts/round | round ms | tok/s/user spec vs plain | speedup | users identical to plain | log |
|---|---|---|---|---|---|---|---|
| 4 | k=3 fixed (T=4) | 2.16 | 62.6 | 50.2 vs 23.6 | 2.13x | 2/4 | pf_spec_int_b4_k3 |
| 4 | cycle: adapt{3,5} / k3 / k5 | 1.92 / 2.08 / 3.82 | 62 / 63 / 103 | 46.6 / 49.2 / 45.9 vs 23.6 | 1.98x / 2.09x / 1.95x | 2/4 | b4_adapt35 |
| 4 | cycle: adapt{0,3,5} / k3 / k5 / k0 | 1.67 / 1.94 / 3.71 / 0 | 62 / 62 / 103 / 50 | 43.3 / 46.4 / 45.8 / 19.9 vs 23.6 | 1.84x / 1.97x / 1.94x / 0.84x | 2/4 | b4_adapt035b |
| 8 | cycle: adapt{1,3,5} / k1 / k3 / k5 | 2.72 / 0.87 / 2.48 / 3.35 | 78.6 / 61.5 / 71.8 / 91.4 | 47.2 / 30.3 / 48.3 / 47.6 vs 21.3 | 2.22x / 1.42x / 2.27x / 2.24x | 3/8 | b8_adapt2 |
| 16 | cycle: adapt{1,3,5} / k1 / k3 / k5 | 3.20 / 0.86 / 2.36 / 3.26 | 102 / 78.9 / 93.2 / 109 | 41.1 / 23.6 / 36.0 / 38.8 vs 17.9 | 2.29x / 1.31x / 2.01x / 2.17x | 2/16 | b16_adapt |
| 32 | cycle: adapt{1,3} / k1 / k3 | 2.28 / 0.90 / 2.36 | 119 / 100 / 123 | 27.3 / 19.0 / 27.2 vs 15.2 | 1.80x / 1.25x / 1.79x | 9/32 | b32_adapt |
| 64 | k=1 fixed (T=32) | 0.907 | 140.8 | 13.5 vs 12.7 | 1.07x | 16/64 | b64_k1 |
| 64 | k=3 fixed (T=64, chunked 32-row verify, DSV41_SPEC_ROWS=1) | 2.318 | 222.0 | 14.9 vs 11.3 | 1.32x (plain was 88.6 ms) | 13/64 | b64_k3 |
| 64 | cycle: adapt{0,1,3} / k1 / k3 / k0 | 1.77 / 0.85 / 2.34 / 0 | 163 / 131 / 206 / 75 | 16.9 / 14.1 / 16.2 / 13.3 vs 14.3 | 1.18x / 0.98x / 1.13x / 0.93x | 15/64 | b64_adapt013c |
| 128 | k=1 (T=64) | 0.906 | 243 | 7.8 vs 11.3 | 0.69x | 26/128 | b128_k1 |
| 128 | k=3 (T=128) | 2.310 | 400 | 8.2 vs 9.7 | 0.85x | 25/128 | b128_k3 |

First generated token spec == plain for every user in every cell. Self-consistency (identical prompts give identical streams) holds. DRAM free after the spec runner(s): 610-720 MiB/bank
(B=4 716, B=8 695, B=16 686, B=32 687, B=64 647-655, B=128 612-617 at the GSM8K context; the bf16 pool is used everywhere).

Exactness: the spec stream is the plain greedy stream up to the first divergence, which is a numerically different argmax between a T-row verify block and a 1-row plain step
(bf16 / bfp8 batch-shape numerics), at generated token 2..176. About half of the divergences have a plain top1-top2 logit gap > 0.1 (max ~0.5; bf16 logits of magnitude 16-32 have 0.125 steps),
the rest are < 0.1; the demo prints the per-user gaps (`SPEC divergence near-tie evidence`, `SUSPECTED REAL DIVERGENCES`). This is the same character as the adaptive/rows agents' results; a
per-token logit comparison against a CPU reference was not done. Identical-count is therefore low at larger B (longer streams, more chances to hit a tie).

## What changed vs the three input branches
* Merged spec-adapt (resident runner per k, shared drafter, confidence scheduler, `prepare all then capture all`) and spec-rows (chunked 32-row verify, T=5..7 pad in mHC, per-row RoPE for spec verify only; plain decode keeps the fused rows RoPE). spec-k superseded, nothing taken.
* B=4 acceptance 0.05 (spec-rows branch) is not reproduced on the merged branch: 2.16 accepted/round. Cause not isolated; a unit test of the drafter at U=1 (n=1,2, `tests/test_spec_mtp.py`, `DSV41_NUSERS=4`) passes (PCC > 0.999, agreement 1.0 except one near-tie step). Most likely cause: lazily created tensors after a trace capture (fixed by prepare-all-then-capture).
* `default_ks(U)`: B=4 {3,5} (T=2/3 unsupported by the mHC kernels), B=8/16 {1,3,5}, B=32 {1,3}, B=64 {1,3} with `DSV41_SPEC_ROWS=1`. Add 0 to `DSV41_SPEC_SET` for the plain option.
* k=0 (plain / no-spec) policy in `AdaptiveSpec`: a no-draft n=1 runner (verify 1 row/user, `write_main` keeps the drafter rings current, no drafting) plus a probe runner (n=1 with drafting) every `DSV41_SPEC_PROBE` (16) rounds
  (or at once when the policy is a fixed k > 0, or at a window change) to refresh drafts + confidence; the scheduler scores k=0 as 1/round_ms. Measured cost: the k=0 round is 8-18% slower than the true plain decode
  (B=4: 50 vs 42 ms, B=64: 75 vs 70 ms) because the round still goes through the spec trace path and host feed, so at low acceptance k=0 recovers most but not all of plain speed; the probe round costs 53 ms (B=4) / 100 ms (B=64).
  At B=64 GSM8K the adaptive scheduler (1.18x) lands between k3 (1.13x) and plain; it picked k=3 31 times and k=0 20 times (about half of those are the window-start probes of the cycle test).
* B >= 128: `DSV41_SPEC` is ignored (plain decode) unless `DSV41_SPEC_B128=1`.
* Prefill bug fixed: `gsm8k_b8` (U=2, 128-token chunk, one 8-chunk group) failed with "Tensor is not allocated" in `prefill_layer.forward_cols`: `_moe_unified` sliced a full-extent chunk, the slice aliased `own`, and `own` was then deallocated. Fix: return `own` itself when n8 == 1. B=8 GSM8K now runs (spec 2.2x above).
* Diagnostics: `DSV41_SPEC_DBG=N` logs the first N rounds (block, argmaxes, accepted, new drafts, confidence), `SPEC seed draft check`, `SPEC self-consistency`.

## Limits / not done
* B=128: spec loses (0.69x k=1, 0.85x k=3): rounds of 243 / 400 ms vs ~90-100 ms plain; break-even needs ~3.4 accepted drafts. Default is plain.
* B=64 gain is modest (1.1-1.3x; plain-decode noise of +-12% between runs) and needs `DSV41_SPEC_ROWS=1`; k=1 is ~break-even.
* Only GSM8K (high acceptance) was run for B=4/64; isl4k / 64k cells were dropped by request. Earlier measurements (spec-adapt notes) show spec loses at B>=32 on isl4k; the k=0 option is the remedy but is not validated on those workloads here.
* Second prefill after a spec pass hangs the next spec compile: one scenario and one spec pass per process (unchanged). `-k gsm8k_b64` also matches `gsm8k_b64_o*` and `gsm8k_b32` has a duplicate id (`-k 'gsm8k_b64 and not gsm8k_b64_o'`, `'gsm8k_b32 and not gsm8k_b32_1'`).
* Hosts .46 (and once .48) misbehaved for 40-layer builds (a 70 min first-layer load / an abort in the Engram init): use other hosts. A new worktree path needs one JIT compile pass (~30-60 min) per host.

## Reproduce
```
W=/mnt/tt-data/ssinghal/wt/spec_int ; $W/lt.sh <host> <logname> "<ENV>" -o junit_suite_name=x models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k <expr>
ENV (all cells): DSV41_LAYERS=0-39 DSV41_ENGRAM_RAM=1 DSV41_MEMLOG=1 DSV41_TRACE_REGION=1900000000 DSV41_BUILD_STAGGER_S=480 DSV41_BUILD_SLOTS=10
B=4 fixed:   DSV41_SPEC=3 -k gsm8k_b4              B=4 adaptive: DSV41_SPEC=5 DSV41_SPEC_ADAPT=1 DSV41_SPEC_SET=0,3,5 DSV41_SPEC_POLICIES=cycle:adapt+k3+k5+k0:4
B=8/16:      DSV41_SPEC=5 DSV41_SPEC_ADAPT=1 DSV41_SPEC_SET=1,3,5 DSV41_SPEC_POLICIES=cycle:adapt+k1+k3+k5:4 -k gsm8k_b8 | gsm8k_b16
B=32:        ... DSV41_SPEC_SET=1,3 ... -k 'gsm8k_b32 and not gsm8k_b32_1'
B=64:        DSV41_SPEC_ROWS=1 DSV41_SPEC=3 [DSV41_SPEC_ADAPT=1 DSV41_SPEC_SET=0,1,3 DSV41_SPEC_POLICIES=cycle:adapt+k1+k3+k0:4] -k 'gsm8k_b64 and not gsm8k_b64_o'
B=128 (spec forced): DSV41_SPEC_B128=1 DSV41_SPEC_ROWS=1 DSV41_SPEC=3 -k gsm8k_b128
```
Logs: /mnt/tt-data/ssinghal/dsv4-logs/pf_spec_int_<name>.log

## Plain-decode regression check (spec off, gsm8k_b16, 40 layers)
main c29a56370c8 (worktree wt/spec_base, host .35): decode 48.0 ms/token (20.82 tok/s/user); this branch (host .47): 48.1 ms/token (20.79). All 16 users' output texts identical between the two logs (pf_spec_base_reg_base_b16 vs pf_spec_int_reg_new_b16e).
Two other attempts on host .35 hung (first decode step, then a prefill chunk; triage saved in dsv4-logs/triage/hang_35_regnew_2330.* and hang_35_regnew_d_0010.*) while the same build passed on .47: treated as a host .35 problem, not a branch regression (not isolated further).
