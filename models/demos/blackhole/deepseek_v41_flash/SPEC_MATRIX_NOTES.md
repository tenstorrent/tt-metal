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
* B >= 128: `DSV41_SPEC` is ignored (plain decode) unless `DSV41_SPEC_B128=1`. (Since the Defaults change below: B >= 64 is plain unless opted in.)
* Prefill bug fixed: `gsm8k_b8` (U=2, 128-token chunk, one 8-chunk group) failed with "Tensor is not allocated" in `prefill_layer.forward_cols`: `_moe_unified` sliced a full-extent chunk, the slice aliased `own`, and `own` was then deallocated. Fix: return `own` itself when n8 == 1. B=8 GSM8K now runs (spec 2.2x above).
* Diagnostics: `DSV41_SPEC_DBG=N` logs the first N rounds (block, argmaxes, accepted, new drafts, confidence), `SPEC seed draft check`, `SPEC self-consistency`.

## Limits / not done
* B=128: spec loses (0.69x k=1, 0.85x k=3): rounds of 243 / 400 ms vs ~90-100 ms plain; break-even needs ~3.4 accepted drafts. Default is plain.
* B=64 gain is modest (1.1-1.3x; plain-decode noise of +-12% between runs) and needs `DSV41_SPEC_ROWS=1`; k=1 is ~break-even.
* Only GSM8K (high acceptance) was run for B=4/64; isl4k / 64k cells were dropped by request. Earlier measurements (spec-adapt notes) show spec loses at B>=32 on isl4k; the k=0 option is the remedy but is not validated on those workloads here.
* Second prefill after a spec pass hangs the next spec compile: one scenario and one spec pass per process (unchanged). `-k gsm8k_b64` also matches `gsm8k_b64_o*` and `gsm8k_b32` has a duplicate id (`-k 'gsm8k_b64 and not gsm8k_b64_o'`, `'gsm8k_b32 and not gsm8k_b32_1'`).
* Hosts .46 (and once .48) misbehaved for 40-layer builds (a 70 min first-layer load / an abort in the Engram init): use other hosts. A new worktree path needs one JIT compile pass (~30-60 min) per host.

## Defaults (branch ssinghal/dsv4p1-specdef): adaptive spec decode is the default decode mode for B = 4..32

One resolution function, `tt/spec_policy.py::resolve(batch, mesh_rows, max_seq_len)` (pure python, no ttnn, `tests/test_spec_policy.py`), is used by `tt/common.py::_env_setup` (build and reconfigure: `DSV41_RING_ROWS`),
`demo/text_demo.py` (which passes run) and `Generator.enable_spec` (which runners are built), so build and decode always agree. The demo logs one line per scenario, e.g.
`spec decode: adaptive {1,3,5} (default for B=16) [batch 16, padded 16, max_seq_len 512]`.

| batch (U = users/mesh row) | default mode (no `DSV41_SPEC*` set) | max_seq_len with spec by default |
|---|---|---|
| 4 (U=1) | adaptive {3,5} | 70000 (fits; measured only for B=8/16 at 60k, B=4 has the smaller pool; indexer cap 131072) |
| 8 (U=2), 16 (U=4) | adaptive {1,3,5} | 70000 (measured: ISL 60453, 239 / 159 MiB/bank left after runners + traces) |
| 32 (U=8) | adaptive {0,1,3} (0 = plain rounds: spec loses at low acceptance, see isl32k below) | 40000 (measured ISL 30059: 527 free before, 247 after; ISL 60k OOMs the runner build) |
| 64, 128 | plain (`plain (B>=64 default; opt in ...)` / `plain (B>=128)`), no spec runner, `DSV41_RING_ROWS` untouched | opt-in only |
| any, `DSV41_SPEC=0` | plain (`plain (explicit DSV41_SPEC=0)`), no runner, ring rows untouched | - |

Why these sets: they are the candidate sets of the matrix above. B=32 gets 0 because at B=32 on a long document the best spec policy was below plain (isl4k 0.9x, isl32k below 0.98x with 0.42 accepted drafts/round);
the k=0 round costs 8-18% more than a true plain step and a probe round 50-100 ms, so at low acceptance adaptive recovers most but not all of plain (isl32k B=32: 0.98x, k0 35 of 46 rounds).
B=4/8/16 do not get 0: their k=1/3 rounds beat plain even at low acceptance in the measured cells (isl4k B=16 1.02-1.16x, isl64k B=8/16 1.06-1.30x) and a probe round costs more than it saves. The policy knobs
(`DSV41_SPEC_PROBE`, `DSV41_SPEC_HYST`, `DSV41_SPEC_POLICIES`, `DSV41_SPEC_TIMES`, ...) are unchanged. B=64 was measured at 1.1-1.3x with large noise and needs the chunked verify, so the user decision is plain by default.

Overrides (explicit always wins): `DSV41_SPEC=<k>` fixed k (+ `DSV41_SPEC_ADAPT=1` adaptive with `DSV41_SPEC_SET`, default set of the batch), `DSV41_SPEC_SET`, `DSV41_SPEC_ADAPT=0` (fixed k=3 of the default set),
`DSV41_SPEC_B128=1` (B >= 128), and at B >= 64 any of `DSV41_SPEC=<k>`, `DSV41_SPEC_ADAPT=1`, `DSV41_SPEC_SET`, `DSV41_SPEC_ROWS=1` opts in (chunked verify `DSV41_SPEC_ROWS=1` is switched on automatically at U=16/32,
default set {0,1,3} at B=64). Explicit settings skip the context / DRAM fallback (trusted, as before). The old B=64 recipe (`DSV41_SPEC_ROWS=1 DSV41_SPEC=3 ...`) works unchanged.
`tools/grid.py` passes `DSV41_SPEC=0` for its plain cells (they set no DSV41_SPEC before; without it B <= 32 cells would now run spec): the plain grid cells are launched exactly as before otherwise.

### Build time (RING_ROWS) and reconfigure
`DSV41_RING_ROWS=288` (the pool's window ring; read when the pool is created) is applied by `_env_setup` only when the resolved mode is spec; a user-set `DSV41_RING_ROWS` wins. `create_tt_model` AND `reconfigure_tt_model`
resolve for their own batch size (a reconfigure rebuilds the pool), and values this module applied itself are not mistaken for user settings (a build for B=16 then a reconfigure to plain B=64 drops the 288 again). The demo
checks `model.pool.ring_rows` and falls back to plain with a warning if a spec mode was resolved for a model built without spec.
Measured effect of ring 288 versus 128 (4 layers, `DSV41_SPEC=0`, B=16 / B=32, host .33, logs `specdef_ringab_*`): plain decode 11.8 / 11.9-12.1 ms and 13.1 / 13.1 ms/token (no change); weights+pool allocated at
build +0.3 MiB/bank (B=16) / +0.6 (B=32) at 4 layers, i.e. about +3-6 MiB/bank of the ~3 GiB at 40 layers (notes of the earlier work: +46 MiB/chip). DRAM headroom is dominated by the runners (below).

### Memory and long context
Cost measured with MEMLOG (40 layers, bf16 pool): first runner (views + drafter) +111 MiB/bank at B=16, +157 at B=32; extra runners +0.7 (B=16) / +34.5 (B=32, 3 runners); trace capture +19..31. At B=32 the default set costs
280 MiB/bank in total (527 -> 247 free at ISL 30k). Fits by default: `max_seq_len` <= 70000 for B <= 16 (the grid's 64k scenarios use 70000) and <= 40000 for B=32 (linear projection in max_seq_len, 11.8 MiB/1k tokens:
the B=32 requirement is met up to ~57k, not measured). Beyond: the scenario runs plain with `!!! spec decode: plain (context too long for spec: max_seq_len 70000 > 40000 at B=32)` (measured: isl64k_b32 default env falls back,
TTFT 329 s, 65.7 ms/token = the grid's plain cell). Two guards: (1) static `SPEC_MAX_CTX` per U and the 131072 cap of the matmul indexer backend (above it `spec verify needs the matmul indexer backend` asserts);
(2) a runtime check right before the runners are built (free DRAM per bank >= 190 MiB for B <= 16, 320 for B=32, largest block >= 40), skipping the spec pass with a loud warning (`DSV41_SPEC_NEED_FREE_MIB` overrides, used to test the
skip path). Not tested for isl128k/256k (above the 131072 cap or far beyond DRAM): plain. Hang rules of the notes still apply (one scenario / one spec pass per process; all runners are compiled before any trace is captured).

### Evidence (commit 0521f72a70b + later; all with DSV41_* unset except as listed; 40 layers = GSM8K, DSV41_ENGRAM_RAM=1 DSV41_MEMLOG=1 DSV41_TRACE_REGION=1900000000)
Logs in /mnt/tt-data/ssinghal/dsv4-logs/. Unit tests: `python -m pytest models/demos/blackhole/deepseek_v41_flash/tests/test_spec_policy.py` (18 pass, CPU).
| cell (default env) | mode chosen | result | log |
|---|---|---|---|
| B=4 GSM8K | adaptive {3,5} | 2.12x vs plain, 2.16 accepted/round, round 67.6 ms (plain 45.6 ms), first token == plain for 4/4 users | pf_spec_int_specdef_gsm_b4.log |
| B=8 GSM8K | adaptive {1,3,5} | 2.50x, 2.88 accepted/round, round 82.5 ms (plain 53.4 ms), 8/8 first tokens | pf_spec_int_specdef_gsm_b8.log |
| B=16 GSM8K | adaptive {1,3,5} | 2.14x, 3.15 accepted/round, round 93.7 ms (plain 48.5 ms), 16/16 | pf_spec_int_specdef_gsm_b16.log |
| B=32 GSM8K | adaptive {0,1,3} | 1.83x, 2.31 accepted/round, round 118.2 ms (plain 65.7 ms), 32/32 | pf_spec_int_specdef_gsm_b32.log |
| B=16 then reconfigure to B=4, one build | adaptive, then adaptive | B=16 2.16x (3.16 acc, 93.5 ms); B=4 after reconfigure 2.11x (63.2 ms) | pf_spec_int_specdef_gsm_b16_b4_session.log |
| B=32 ISL 30059 | adaptive {0,1,3}, fits (247 MiB/bank free after runners) | 0.98x, 0.42 accepted/round | pf_spec_int_specdef_isl32k_b32.log |
| B=32 ISL 60453 | plain (context too long for spec: 70000 > 40000) | TTFT 329 s (see the TTFT note below), 65.7 ms/token | pf_spec_int_specdef_isl64k_b32.log |
| B=64 GSM8K | plain (B>=64 default), no runner | 67.7 ms/token | pf_spec_int_specdef_gsm_b64.log |
| B=128 GSM8K | plain (B>=128) | 87.4 ms/token | pf_spec_int_specdef_gsm_b128.log |
| B=16 4k, explicit DSV41_SPEC=0 (grid env) | plain (explicit) | TTFT 9.81 s (.40), 9.48 s (.44), decode 44.0 / 42.7 ms: equal to main (9.91 / 9.57 s) | specdef_gridlike_branch_h40.log, specdef_gridlike_branch_h44.log |

TTFT note: runs with `DSV41_MEMLOG=1` (all spec-matrix runs and the evidence above except the last row) have a LARGER TTFT than the grid runs: the same code (main, 4k B=16 plain) gave 9.6-9.9 s with the grid environment
and 11.1-14.1 s with MEMLOG=1 + `DSV41_TRACE_REGION` (logs `pf_spec_int_specdef_pairbase_G4k_h33.log`, `..._h40.log`), because MEMLOG wraps the prefill forwards with DRAM-usage queries. Do not compare TTFT between MEMLOG and grid runs;
the speedups above are measured inside one process against plain, so they are not affected. Host .33 was also slow and noisy for timing (11.9 / 18.3 / 9.47 s for the same run). No prefill code differs between the branch and main
(`git diff 43677d4a764 ssinghal/dsv4p1-specdef` touches `tt/common.py`, `tt/generator.py`, `tt/spec_model.py` (formatting), `tt/spec_policy.py` (new), `demo/text_demo.py`, `tools/grid.py`).
Unit tests: 18 pass (`tests/test_spec_policy.py`); 4-layer smokes of every default and override path passed on .33.

## Reproduce
```
# from your checkout root (D=models/demos/blackhole/deepseek_v41_flash): $D/tools/spec_lt.sh <host> <logname> "<ENV>" -o junit_suite_name=x models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k <expr>
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
