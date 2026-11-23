# Llama-3.3-70B as one TP32 model on a Blackhole Galaxy

Scripts for the vLLM SLA runs and prefill profiling of `models/demos/llama3_70b_galaxy` on one BH Galaxy (8x4 mesh).
No paths are baked in: set these for your host first.

```bash
export TT_METAL_DIR=/path/to/tt-metal          # this repo, built, python_env with vLLM installed
export HF_MODEL=/path/to/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/path/to/weight-cache       # created on first run if empty
export PLUGIN_DIR=/path/to/vllm-tt-plugin
export OUT=/path/to/results/run1
source sla_scripts/bh_tp32/env.sh                 # best known model switches
```

`env.sh` sets `TT_CFG_EXTRA='"max_new_prefills_per_step": 1'`, a vllm-tt-plugin setting that only exists with
`vllm_tt_plugin_max_new_prefills.patch` applied (made against plugin 042a948):
`git -C $PLUGIN_DIR apply $TT_METAL_DIR/sla_scripts/bh_tp32/vllm_tt_plugin_max_new_prefills.patch`.
Without the patch, `export TT_CFG_EXTRA=` after sourcing `env.sh` (that costs the 32-at-once TTFT mean, about 4.0 -> 5.1 s).

| Script | What it does |
|---|---|
| `run_sla.sh` | Starts the server, runs every SLA line, prints the table, stops the server with SIGINT. |
| `run_server.sh` / `bench.sh` / `summary.py $OUT` | The same steps, one at a time. |
| `sanity_check.py [url]` | Known-answer check (short questions one at a time and batched, a ~9K-token continuation); `bench.sh` runs it first and stops if it fails. |
| `prefill_profile.sh "tag:ENV=1" ...` | Per-op device time of one prefill chunk of a 1-layer model, per variant. |

Tests in `models/demos/llama3_70b_galaxy/tests/` (run with `MESH_DEVICE=TG pytest <file> -s`):
- `test_prefill_switch_pcc.py`: full 80-layer model; last-token logits with the prefill switches vs all off.
- `test_rs_tune.py` / `test_mm_tune.py`: isolated reduce-scatter and minimal-matmul sweeps (`MM_M=2048|4096`).
- `test_prefill_layer_prof.py`: the 1-layer prefill that `prefill_profile.sh` profiles. `test_mesh_health.py`: quick all-gather check.
- `test_agmm_tune.py`: fused all-gather + minimal_matmul (`all_gather_minimal_matmul_async`) vs the unfused pair at the
  FF2 (and FF1/FF3+SwiGLU) prefill shapes on the 8x4 mesh.
- `test_cpf_port_pcc.py`: full 80-layer model, real text; the ported Wormhole prefill levers (pad 1024, seq-split,
  QKV seq-split, CPF ring + replica) and the BH switches (norm bf8, fused WO) cold and warm, vs levers off.
- `test_mmrs_prefill_tune.py`: fused `matmul_reduce_scatter_prefill` vs matmul + `reduce_scatter` at the FF1/FF3, WO and
  FF2 shapes (`MMRS_FABRIC`, `MMRS_PAYLOAD`, `MMRS_LINKS`). `test_mmrs_tune.py`: the strided fused variant at FF1/FF3.
- `test_overlap_prefill.py`: matmul beside a reduce-scatter on two sub-devices (the deepseek_v3_d_p pattern). The overlap
  variant does not run yet: `ttnn.reduce_scatter` on a sub-device that does not start at (0, 0) applies the offset twice
  and finds 0 cores.
- `test_pp_stage_bench.py` (single device): one full Llama-70B layer on one chip at 512-4096 tokens, the per-stage cost of
  a 32-stage pipeline-parallel prefill.
- `test_sdpa_decode_cores.py` (single device): paged decode SDPA at 48 / 64 / 88 cores.

`ttft_tail.py <result.json>`: where a TTFT tail comes from (slowest requests, burst size, early vs steady state); needs
`bench.sh`'s `--save-detailed` output. `LLAMA_SWITCH_TIMING=1`: logs the prefill<->decode switch cost every 100 switches.
`run_server.sh` takes `MAX_BATCHED_TOKENS` and `TT_CFG_EXTRA` (more `additional_config` "tt" keys).

Reset the Galaxy (`tt-smi -glx_reset_auto`) before an accuracy or SLA run whenever the previous process hung, was
hard-killed, failed to start, or ran an experimental op: the chips can keep producing garbage in the next process otherwise.

Decode timing: `LLAMA_DECODE_TIMING=1` logs host time per decode step every 200 steps (`DECODE_TIMING` lines in the server log).

The env switches are described in `env.sh` and at their use sites. All of them are off by default.

## Status (Nov 23 2026, tt-galaxy-43, runs/sla_nov23 + fsab2: `env.sh` + 8 API servers, see "Nov 23: closed-loop scheduling")

| SLA line | Target | Levers off | Before DP4 (sla_cap1) | Nov 20 (sla_dp4_full) | Nov 22 (sla_nov22b) | Now (sla_nov23) |
|---|---|---|---|---|---|---|
| 1 user 4K: TTFT / decode | 1 s / 25 | 0.32 s / 51.4 | 0.27 s / 51.4 | 0.27 s / 51.4 | 0.42 s / 51.4 (150 ms idle gather) | 0.42 s / 51.4 |
| Cold 16K / 32K single | 5 / 10 s | 1.97 / 5.66 s | 1.58 / 5.03 s | 1.21 / 4.74 s | 1.35 / 4.86 s (first 32K request 6.4 s: one-time JIT compile after a rebuild) | 1.35 / 4.88 s |
| 60 RPM: TTFT mean / P99 | 1 / 5 s | 0.62 / 1.14 s | 0.43 / 0.76 s | 0.43 / 0.83 s | 0.36 / 0.68 s | 0.42 / 0.77 s |
| 60 RPM: decode mean / P99 | 25 / 15 | 29.4 / 22.9 | 35.3 / 29.8 | 35.4 / 29.8 | 35.6 / 30.2 | 35.8 / 31.1 |
| 60 RPM: e2e mean | 20 s | 17.1 s | 14.2 s | 14.1 s | 14.0 s | 14.0 s |
| 60 RPM + 40% prefix: TTFT mean / P99 | 0.5 / 1.5 s | 0.77 / 2.93 s (fail) | 0.46 / 1.04 s | 0.46 / 1.03 s | 0.39 / 1.02 s | 0.45 / 1.07 s |
| 60 RPM + 40% prefix: decode / e2e mean | 25 / 20 s | 26.7 / 20.0 s | 33.9 / 15.6 s | 34.0 / 15.6 s | 34.1 / 15.5 s | 34.4 / 15.4 s |
| 240 RPM: TTFT mean / P99 | 1 / 5 s | 1.27 / 9.00 s (fail) | 0.58 / 3.00 s | 0.51 / 2.08 s | 0.40 / 0.93 s | 0.42 / 1.11 s |
| 240 RPM: decode mean / P99 | 25 / 15 | 21.5 / 16.4 (fail) | 25.7 / 21.6 | 25.8 / 22.1 | 26.1 / 22.5 | 26.4 / 23.1 |
| 32 x 4K at once: TTFT mean / P99 | 1 / 5 s | 4.81 / 9.16 s | 3.99 / 7.60 s (fail) | 1.61 / 3.20 s | **1.09-1.16** / 3.15 s (mean fails; 5 runs; sla_final 1.15 s) | **0.96** / 3.53 s (pass; 0.93-0.99 over 10 runs with the final scheduler defaults) |
| 32 x cold 16K at once: TTFT mean / P99 | 5 s | not run | 28.7 / 49 s (slide) | 7.20 / 12.65 s | **6.78** / 12.0 s (fail; 4.2x; sla_final 3 seeds; best runs 6.73 s) | **6.78** / 12.0 s (fail; 6.75-6.82 over 3 runs) |
| Concurrent users | 128 | 32 | 32 (fail) | 32 (fail) | 32 (fail) | 32 (fail; 4-Galaxy balancer in `../bh_quad`) |
| Quick MMLU / needle | 80% | 81.4% / 0 misses | 81.9% / 0 misses | 82.0% / 0 misses | 81.93% / 0 misses | 81.84% / 0 misses |

Full MMLU with every prompt forced onto DP4 (14042 questions): 82.92% on the final shipped config (full_final: SDPA LoFi,
QKV bf8, WO all-to-all fp32, bfp4 W2, decode after every prefill step; 246 lost / 236 gained), 82.99% before.
The 32 x 16K mean is above 5 s because the prefill is energy-bound at the 130 W chip limit (see "Nov 21 evening" and
"Nov 22 (later)"). 32 x 4K is 3.7x the 3.99 s baseline.

## DP4 x TP8 prefill (Nov 20 2026, `LLAMA_BH_PREFILL_DP4=1`, on in `env.sh`)

Why: the TP32 prefill splits one prompt over all 32 chips and is communication-bound. Per 4K layer it spends ~1.7 ms in
all-gather / reduce-scatter against ~0.95 ms of matmul (prefill profile), so even perfect overlap caps TP32 at ~1.7x.
The all-gather on a column runs at ~74 GB/s per chip with 2 links (`test_tp8_ccl_norm_sweep.py`), the fabric ceiling here.

What: `models/demos/llama3_70b_galaxy/tt/prefill_dp4tp8.py`. Each mesh column (8 chips) prefills its own lane's prompt
with TP8 (sequence-parallel residual, all-gather rows, per chip 8 q heads + kv head r, local causal SDPA, WO and FF2 as
`matmul_reduce_scatter_prefill`, FF1|FF3 as one `minimal_matmul(fuse_swiglu=True)`). vLLM already hands the model one
prompt per lane per prefill step (users 0-7 live on column 0, ...), so four prompts run at once on disjoint links. The KV
is written with `paged_fill_cache` into exactly the TP32 decode layout (row r = kv head r, column c = lane c): decode is
unchanged. The generator (`_dp4_prefill`) takes cold prompts >= 2048 tokens when >= 2 columns have one; single prompts
and prefix hits stay on TP32. A second weight copy (~6.7 GB per chip, cache `dp4tp8_*`) is built on first start.

Switches (all read at model build / call time):
- `LLAMA_BH_PREFILL_DP4=1` build and use it; `LLAMA_BH_PREFILL_DP4_OFF=1` keep it built but unused
- `LLAMA_BH_PREFILL_DP4_MIN` (2048) shortest padded prompt; `LLAMA_BH_PREFILL_DP4_MIN_USERS` (2) columns per step
- `LLAMA_BH_PREFILL_DP4_WARM` lengths compiled at warmup (2048..8192, 16384)
- `LLAMA_BH_PREFILL_DP4_LOFI=ff2,wo,qkv` opt-in precision change (LoFi for those bf8 matmuls)
- `LLAMA_BH_PREFILL_DP4_SDPA_FP32` / `_SDPA_APPROX` (1 / 1), `LLAMA_BH_PREFILL_DP4_FP32` (qkv,ff13 fp32 accumulation)
- `LLAMA_BH_PREFILL_DP4_PROFILE=1` synced per-part times of layer 1; `LLAMA_BH_PREFILL_DP4_TRACE` opt-in trace (no gain:
  the path is device-bound, and a trace hands its intermediates back to later allocations)

Results (vLLM, tt-galaxy-43, bench.sh settings):

| Line | TP32 (sla_cap1) | DP4 x TP8 (dp4_v4) | + LoFi ff2,wo,qkv (dp4_v5_lofi) |
|---|---|---|---|
| 32 x 4K at once (96 prompts, c32): TTFT mean / P99 | 3.99 / 7.60 s | 2.53 / 4.92 s | 2.45 / 4.54 s |
| 32 x 16K at once: TTFT mean / P99 | 28.7 / 49 s (slide) | 12.8 / 21.1 s | 12.0 / 19.7 s |
| Quick MMLU (DP4 forced onto >= 1K prompts) / needle | 81.9% / 0 | 81.84% / 0 | 81.93% / 0 |

Checks: `tests/test_dp4tp8_pcc.py` (80 layers, 4 prompts, vs TP32: top-1 match, KV PCC 0.996-0.9998),
`tests/test_dp4_pieces.py`, `tests/test_dp4_sdpa_sweep.py`, `tests/test_dp4_tp8_*` (layer microbenches).

Serving fixes for bursts (plugin side). With async scheduling a prefill step's first tokens were published only after
the next step, so when two DP4 prefill steps ran back to back half the users waited one extra step (2.4 s at 16K); and
the first request of a burst prefilled alone on TP32 while the rest waited. Fix: Gio's plugin branch
(gtobarTT/vllm-tt-plugin `gtobar/serving-config` @ 6d24eff, same base 042a948) with `vllm_tt_plugin_gio_dp4.patch` on
top (my per-lane `max_new_prefills_per_step`, and `TT_PLUGIN_IDLE_GATHER_MS`: when nothing runs, hold a partial set of
lanes' first prefills up to that long so a trickling burst starts with one prompt per lane). Run it from a worktree with
`PLUGIN_DIR=<worktree> PLUGIN_SRC=<worktree>/src` (run_server.sh puts PLUGIN_SRC ahead of the installed plugin) and
`TT_CFG_EXTRA='"max_new_prefills_per_step": 1, "flush_prefill_output": true'`. Note Gio's own
`max_prefill_requests_per_step` caps new prompts over all lanes together, which would starve DP4: keep it unset.

| Line (vLLM, Nov 20) | TP32 before | DP4 | + flush + gather 150 ms + LoFi + SDPA fp32 off (dp4_fg_sdpa16) |
|---|---|---|---|
| 32 x 4K at once: TTFT mean / P99 | 3.99 / 7.60 s | 2.53 / 4.92 s | 2.24 / 4.18 s |
| 32 x 16K at once: TTFT mean / P99 | 28.7 / 49 s | 12.8 / 21.1 s | 10.06 / 17.73 s |
| 1 user 4K TTFT | 0.27 s | 0.27 s | 0.42 s (pays the 150 ms gather when alone) |
| 240 RPM TTFT / decode (flush + gather + LoFi) | 0.58 / 3.00 s, 25.7 / 21.6 | 0.51 / 2.08 s | 0.45 / 1.32 s, 25.7 / 22.0 |
| Quick MMLU / needle | 81.9% / 0 | 81.84% / 0 | 82.02% / 0 |

Kernel change (C++, needs a rebuild): `matmul_reduce_scatter_prefill` staged its partials and output in bf16, twice the
bytes of the bf8 all-gathers. `TT_MRS_BF8_TRANSPORT=1` (read in the program factory and the output spec) stages them in
bfp8; the hold still accumulates in fp32 and the output becomes bfp8. Column axis, 8 chips: WO 4K 0.75 -> 0.47 ms, 16K
2.93 -> 1.65 ms; FF2 (LoFi) 4K 0.90 -> 0.81 ms, 16K 3.83 -> 3.59 ms; PCC vs torch >= 0.9985 (`tests/test_mrs_transport.py`).
Install after `ninja -C build _ttnncpp.so _ttnn.so`: copy `build_Release/ttnn/_ttnncpp.so` and `_ttnn.so` to
`build_Release/lib/` and `_ttnn.so` to `ttnn/ttnn/` (Python loads those copies). On .43 the Nov 16 copies are kept as
`*.bak_1116`. Do not build with the uncommitted `ccl/llama_common.cpp` edit of Nov 18 (custom decode CCL cores): it was
never tested and drops quick MMLU to ~70% (saved as `/mnt/nvme0/jerrywang/llama_common_uncommitted_1118.patch`).

With it (runs/dp4_mrs8_fixed, everything above + `TT_MRS_BF8_TRANSPORT=1`): 32 x 4K 2.12 / 3.96 s, 32 x 16K 9.47 / 16.51 s,
1 user 4K 0.42 s, quick MMLU 82.11%, needle 0 misses.

Nov 21 (runs/dp4_opt1, everything above plus the changes below, now env.sh defaults): **32 x 4K 1.80 / 3.59 s, 32 x 16K
8.22 / 14.35 s**, 1 user 4K 0.415 s, quick MMLU 82.11%, needle 0 misses.

A device profile of the forward (tracy, `DP4PCC_DEVPROF=1` in `tests/test_dp4tp8_pcc.py`) showed the 4K layer runs back
to back on device: 4.64 ms per layer, 0.5 us between ops, and a captured trace is no faster. The time beyond the
layers was host-side tail, about 85 ms per 4K step:
- **Last hidden row:** `last_hidden` read each column's whole [M/8, 8192] residual to the host to keep one row. It now
  slices the token's 32-row tile on device first, with runtime bounds so no position compiles a new program. Forward
  plus read: 426-443 -> 405-411 ms per 4 x 4K step.
- **Logits upload:** logits uploaded one 128-row tensor per user. Now one upload per step, with user k at row 32 k.
  Logits time: 42-51 -> 28-31 ms in the PCC test.
- **rms_norm:** `LLAMA_BH_PREFILL_DP4_NORM=rms`, see above.
- **Matmul blocking:** QKV / FF13 K block 16 and bf16 accumulation (`LLAMA_BH_PREFILL_DP4_FP32=none`). One chip
  (`tests/test_dp4_mm_acc.py`): FF13 4K 1.18 -> 1.05 ms, 16K 4.70 -> 3.99 ms. Same PCC; MMLU and needle unchanged.

Rejected: bfp4 norm output for the all-gathers. Forward -3%, but logits PCC 0.89-0.95.

Note: the PCC test builds its prompts from this repo's `.md` / `tt/*.py` files, so editing those changes the prompts
(and one user's logits PCC moved between 0.90 and 0.99 with the prompt, not with the code).

4K layer on device now (bf8 RS):

| Op | Time |
|---|---|
| FF13 | ~1.05 ms |
| FF2+RS | 0.81 ms |
| WO+RS | 0.47 ms |
| Two all-gathers | 0.8 ms |
| SDPA | 0.35 ms |
| QKV | 0.24 ms |
| Glue | ~0.55 ms |

Nov 21 later (runs/dp4_opt2, code defaults):

| Line | Result |
|---|---|
| 32 x 4K TTFT mean / P99 | 1.76 / 3.51 s |
| 32 x 16K TTFT mean / P99 | 8.06 / 14.04 s (3.56x the 28.7 s baseline) |
| Quick MMLU | 82.11% |
| Needle misses | 0 |

New defaults since dp4_opt1:
- **All-gather workers:** all_gather_async with 8 workers per link (`LLAMA_BH_PREFILL_DP4_AG_WPL`; `tests/test_ag_tune.py`:
  16K rows 79.8 -> 85.5 GB/s, about the link ceiling; buffers per channel and chunks per sync do not help).
- **SDPA query:** bf8 from 8K tokens (`LLAMA_BH_PREFILL_DP4_SDPA_Q8_MIN`). In the model at 16K: all-gathers 1.89 -> 1.78 ms
  each, SDPA incl. the typecast 4.37 -> 4.10 ms.

Found, not fixable here: the fused WO / FF2 matmul + reduce-scatter runs much slower in the model than alone (16K, bf8
transport):

| Op | Alone (`tests/test_mrs_breakdown.py`) | In the model |
|---|---|---|
| WO | 1.64 ms | 2.0-2.2 ms |
| FF2 | 2.74 ms with `l1_headroom=0`, 3.56 ms default | 3.8-4.1 ms |

The planner keeps below every live L1 tensor. In the model the TP32 decode path's persistent CCL buffers and ~150
semaphores occupy the top ~180 KB of each core's L1 (from 1,393,536 B; `LLAMA_BH_PREFILL_DP4_MRS_PLAN=1` logs the
plans), so it is capped at ~1.04-1.08 MB plans. Those buffers back the decode traces, so they stay.

Planner override (C++, `matmul_reduce_scatter_prefill_planner.cpp`, needs the rebuild):

- **What it does:** `TT_MRS_FORCE_CHUNKS="Mt,Kt,Nt,n_m,n_n;..."` forces the chunk split for a matching shape when it fits
  L1.
- **Why:** with the model's tight budget the planner's cost model (which its own comment says misranks row-scatter
  splits) picked slow splits.
- **Sweep** (`tests/test_mrs_chunks.py`, `l1_headroom=262144` reproduces the in-model plans):

  | Shape | Auto | Forced |
  |---|---|---|
  | FF2 16K | 3.74 ms | 2.87 ms (4,6) |
  | WO 16K | 1.80 ms | 1.60 ms (3,4) |
  | FF2 8K | 1.93 ms | 1.49 ms (2,6) |
  | WO 8K | 0.94 ms | 0.83 ms (2,3) |
  | 4K | best on auto | |

- **Defaults:** env.sh sets these splits. 80-layer 4 x 16K forward: 1646 -> 1572 ms.

runs/dp4_opt3 (Nov 21, env.sh + code defaults): **32 x 4K 1.75 / 3.47 s, 32 x 16K 7.62 / 13.30 s** (3.77x the 28.7 s
baseline), 1 user 4K 0.419 s, quick MMLU 82.11%, needle 0 misses.

Later Nov 21: three more C++ / layer changes.
- **rms_norm:** gained an output `dtype` (the layernorm op had the attribute; the interleaved path ignored it). The
  DP4 norm writes bf8 directly. One chip, incl. the add: 4K 0.19 -> 0.165 ms, 16K 0.62 -> 0.44 ms.
- **rotary_embedding_llama:** gained an output `dtype`, so q / k come out of RoPE in bf8 (`tests/test_rope_bf8.py`:
  16K q 0.44 -> 0.29 ms, PCC 0.9996 vs bf16 + typecast).
- **SDPA query:** bf8 at every length, so SDPA emits bf8 and the head concat needs no typecast.
- **SDPA chunks:** k chunk 1024 from 8K tokens is tried first (16K 3.70 -> 3.61 ms alone), but in the model it does
  not fit L1 and falls back to k512.

runs/dp4_opt5:

| Line | Result |
|---|---|
| 32 x 4K TTFT mean / P99 | 1.73 / 3.41 s |
| 32 x 16K TTFT mean / P99 | 7.52 / 13.11 s (3.82x) |
| Quick MMLU | 82.11% |
| Needle misses | 0 |

Where the 16K mean now goes. The 32 TTFTs come in 8 groups of 4: 1.92, 3.49, 5.13, 6.71, 8.34, 9.91, 11.54, 13.11 s.
- **Per step:** each step is forward ~1.53 s + logits ~11 ms + ~60 ms of vLLM work. The ~60 ms is mostly the decode
  steps the scheduler interleaves (2 prefill steps per decode step) once the first users have their first token.
- **Start:** a ~0.3 s start offset (requests trickle in through the API server).
- **Decode interleave:** `decode_interleave_prefill_steps` 4 or 8 helps 16K (7.46 s) but hurts the 4K line's later
  waves (1.91-1.92 s vs 1.73 s), so it stays at the default 2.
- **Matmuls:** QKV / FF13 block configs are swept out (best FF13 variant 1% faster, `tests/test_dp4_mm_acc.py`).

Tried and removed (Nov 21, after dp4_opt5):
- **Burst-prefill rule** in the plugin scheduler: allow up to 8 consecutive prefill steps while every lane has a long
  ready prompt. It skipped the inserted decode steps as intended, but 16K moved 7.50 -> 7.45-7.47 s (noise level).
- **Deferred mode switch:** on-device prefill sampling switches the model to decode every prefill step (~25 ms of host
  time, `GEN_HOST_TIMING`-style timers). Deferring the decode-CCL semaphore reset and layer re-pointing to the next
  decode forward kept accuracy but did not shorten the step: the host there is waiting on queued device work.
- **One batched LM-head call per DP4 step** (one norm instead of four): no change.

Run-to-run noise:
- 32 x 4K mean TTFT varies 1.72-1.93 s between runs of the same config.
- A long run sequence once left the board in a bad state (MMLU ~1%, garbage on the TP32 path) until a reset; check
  accuracy after a reset before trusting a run.

`/mnt/nvme0/jerrywang/scripts/needle_conc.py` (SLA test `conc`): 8 concurrent long prompts with distinct passphrases,
which exercises the DP4 multi-user path.

runs/dp4_final (reset board, code defaults):

| Line | Result |
|---|---|
| 32 x 4K TTFT mean / P99 | 1.73 / 3.47 s |
| 32 x 16K TTFT mean / P99 | 7.50 / 13.11 s |
| 1 user 4K | 0.411 s |
| Concurrent needle | 0 misses |
| Needle | 0 misses |

The 16K line is not beyond one Galaxy at today's kernel speeds: matmuls + SDPA alone are ~11.4 ms per 16K layer, a
FIFO mean of ~4.1 s. The gap is ~5.9 ms of collectives (two all-gathers plus exposed reduce-scatter) and ~2.8 ms of
glue per ~22 ms layer, so hiding the collectives is the 16K lever.

Where a 4K DP4 layer goes now (~5.6 ms with LoFi): matmuls ~2.2-2.9, two all-gathers ~0.9 (fabric bound), SDPA ~0.57,
norm / heads / RoPE / concat / adds ~1.5. Floors: the 32 x 4K FLOPs (18.6 PFLOP) at today's per-chip matmul rate give a
FIFO mean of ~1.0-1.3 s with zero communication, so the 1 s mean is at the one-Galaxy compute floor; 32 x 16K (83 PFLOP)
needs >= 8.6 s on one Galaxy, so its 5 s line needs a Quad (DP4 x TP8 on each Galaxy of a Quad: ~3-4 s).

### Nov 21 evening: the DP4 prefill is power-bound

**Finding.** During a DP4 x TP8 forward, every chip runs at its TDP limit and clocks down.

| | DP4 x TP8 forward | TP32 forward |
|---|---|---|
| AICLK, all 32 chips | 1093-1250 MHz | 1275-1350 MHz |
| VDD | 0.69-0.78 V (vdd_min 0.70) | up to 0.81 V |
| Mean chip power | ~130 W (= `tdp_limit` 130) | below the limit |

Temperature stays under 70 C, so this is the power limit, not a thermal one. The readings come from `tt-smi -s --snapshot_no_tty`, which takes 0.5 s and works while the chips are open. `/dev/shm/smi_poll.sh` with `smi_parse.py` logs one line per snapshot.

What this explains:
- **The device profile hides it.** It converts cycles at the nominal 1350 MHz, so it shows a back-to-back 16.3 ms 16K layer (0.5 us gaps).
- **The wall clock shows it.** The 80- vs 8-layer forward slope is 18.97 ms per layer, 14% longer.
- **4K is affected too**, by about 17%.
- **Collective wins under-delivered.** Under an average-power cap, a layer costs joules, not milliseconds: about 2.5 J per chip per 16K layer. Shortening low-power phases (all-gathers, reduce-scatter waits) barely moves TTFT. That is why the collective-hiding work gave less end to end than its op-level gains.
- **Only removing work or precision helps.** That means fewer FPU passes or fewer bytes moved.

The TDP limit is a firmware / board setting and was left alone.

**Accuracy method fix.** Until now quick MMLU never ran on DP4, for three reasons:
- DP4 takes only prompts of `LLAMA_BH_PREFILL_DP4_MIN` = 2048 tokens or more, and only when 2 or more such prompts land in different columns at once.
- vLLM prefix caching gives every 5-shot prompt a cache hit.
- lm_eval sends a `seed`, and seeded prefills skip DP4.

So every "82.11%" above measured the TP32 path. To put every MMLU prompt on DP4, run with all of these:
- `SERVER_ARGS=--no-enable-prefix-caching`
- `LLAMA_BH_PREFILL_DP4_MIN=512`
- `LLAMA_BH_PREFILL_DP4_MIN_USERS=1`
- `LLAMA_BH_PREFILL_DP4_GREEDY_SEED=1` (new: seeded but all-greedy requests may take DP4)
- `LLAMA_BH_PREFILL_DP4_COUNT=1` (new: logs DP4 coverage)

The result is deterministic: a repeat run gives the same answers, 0 flips. The current DP4 stack scores **82.37%**, versus 82.11% for TP32, so the earlier precision changes were safe.

Precision trades measured this way (32 x 16K mean; MMLU with every prompt on DP4; flips vs the 82.37% run):

| Change | 32 x 16K | MMLU | Flips (lost / gained) | Status |
|---|---|---|---|---|
| None (dp4_ship) | 7.20 s | 82.37% | | default |
| QKV matmul output bf8 (`LLAMA_BH_PREFILL_DP4_QKV8=1`) | 7.18 s | 81.05% | 19 / 4 | opt-in |
| SDPA LoFi (`LLAMA_BH_PREFILL_DP4_SDPA_FID=LoFi`) | 7.07 s | 81.05% | 28 / 13 | opt-in |
| Both | 6.98 s | 81.05% | 28 / 13 | |

QKV bf8 needed a C++ change: `rotary_embedding_llama` now accepts a bf8 input in interleaved prefill, and its rotated intermediate stays bf16.

Smaller results:
- **FF13 K block 32** is the code default (fallback 16): 7.25 vs 7.29 s, neutral.
- **A 50 ms idle gather** costs 0.7 s at 16K (8.00 s). The first step then starts with fewer than 4 prompts. Keep 150 ms.
- **SDPA k chunk 768 / 1024 does not fit** next to the model's resident L1 at 8K / 16K, so q256/k512 is what runs.

runs/dp4_ship (shipped defaults: QKV bf16, SDPA HiFi2, device-side LM-head input, cleaned plugin):

| Line | Result |
|---|---|
| 32 x 4K TTFT mean / P99 | 1.61 / 3.20 s |
| 32 x 16K TTFT mean / P99 | 7.20 / 12.65 s (3.99x the 28.7 s baseline) |
| 1 user 4K | 0.418 s |
| Needle | 0 misses |
| Concurrent needle | 0 misses |

Two follow-ups, both negative:
- **LoFi on softmax @ V only** (a C++ define on the streaming SDPA kernel, QK^T kept at HiFi2): the PCC test passed, but
  the server hung the device on its first DP4 SDPA call in warmup (M=2048). The patch was removed and the board reset.
- **Overlap does not survive the power cap.** `test_subdev_concurrency.py` (M=16384, all-gather on a CCL column next to an
  11-column matmul, versus today's full-grid all-gather then matmul):

| Loop | Full grid: AG / matmul / serial | Sub-devices: AG / matmul / both | Gain |
|---|---|---|---|
| 10 iterations (cool chips) | 1.45 / 3.91 / 5.90 ms | 4.73 / 4.18 / 5.86 ms | 0.7% |
| 600 iterations (throttled) | 1.43 / 5.60 / 6.47 ms | 4.68 / 5.75 / 6.39 ms | 1.3% |

Under sustained load the full-grid matmul slows from 3.9 to 5.6 ms, and it then takes the same time on 11 columns as on
the full grid: cores are not the scarce resource, joules are. On a CCL column the all-gather is 3x slower, which eats the
hidden time. The layer-level overlap rewrite (sub-core-grid support for rms_norm, rotary, create/concat heads and
fill_cache) is not worth building.

**Where the time goes, and the floor.** Device profiles of 3 DP4 layers with the shipped settings (kernel durations;
kernels run back to back, gaps under 1 us; times at the nominal clock, wall clock is ~14-17% longer from the throttle):

| Op, per chip | 4K layer | 16K layer |
|---|---|---|
| FF13 matmul (+SwiGLU) | 0.98 ms (490 TF/s) | 3.56 ms (540 TF/s) |
| FF2 matmul + RS | 0.78 ms | 2.65 ms |
| WO matmul + RS | 0.40 ms | 1.73 ms |
| QKV matmul | 0.23 ms | 0.68 ms |
| 2 all-gathers (at the 2-channel link ceiling, ~85 GB/s) | 0.75 ms | 2.86 ms |
| SDPA (LoFi saves only 8%: softmax-bound, not matmul-bound) | 0.33 ms | 3.47 ms |
| Norms, adds, rope, heads, concat, fill | 0.48 ms | 1.3 ms |
| Total | 3.94 ms | 16.24 ms |

(Firmware durations in these reports overlap the previous op and can show a 2-3 ms "add"; use kernel durations.)

The matmuls are already at the chip's energy limit. The FF13-shaped 16K matmul looped 600 times at the power cap runs
at ~343 TF/s per chip (130 W, ~0.38 pJ/FLOP, the same as the Blackhole spec 774 TF/s at 300 W). A 4K layer has ~876
GFLOP of matmul per chip, so the matmuls alone take >= 2.55 ms of wall time per layer. With all-gathers, attention, norms
and rope at zero cost, a 4K step (80 layers + ~12 ms) is >= 216 ms and the 32 x 4K mean is >= 0.115 s + 4.5 steps =
**1.09 s**. The 1 s target (4x) is below that floor on this Galaxy at its 130 W limit. Likewise the 16K matmuls alone
take >= 10.2 ms per layer, and adding attention and all-gathers at their energy cost puts the 32 x 16K mean above 5 s.
The schedule can't make up the gap either: an ideal schedule beats 4 prompts per step by at most 8%.

Measured directly (`tests/test_energy_floor.py`, the FF13 shape [16384, 8192] x [8192, 7168] LoFi on all 32 chips; bf16
activations do not fit L1 at K block 32):

| Weights / activations | 10 back-to-back (full clock) | 2000 back-to-back (at the power cap) |
|---|---|---|
| bf8 / bf8 | 4.09 ms, 470 TF/s | 5.88 ms, 327 TF/s |
| bfp4 / bf8 (the model's FF13 format) | 3.85 ms, 500 TF/s | 5.46 ms, 353 TF/s |

Even with every matmul at the best sustained rate (bfp4 weights everywhere, which the model does not use for QKV / WO /
FF2) and everything else free, a 4K layer is >= 2.48 ms, a 4K step >= 210 ms, and the 32 x 4K mean >= 1.06 s.

Built and measured (`tests/test_matmul_floor.py`): only the model's per-chip matmuls (QKV, WO, FF1|FF3 + SwiGLU, FF2;
the model's weight formats, LoFi, block configs) for a whole 32-prompt batch, 8 steps x 80 layers back to back on all 32
chips. No all-gathers, reduce-scatters, attention, norms, rope or LM head, so any real forward is slower:

| Batch | Matmul-only layer | Rate | 32-prompt TTFT mean, 4 per step | Ideal one-at-a-time schedule | Target |
|---|---|---|---|---|---|
| 32 x 4K | 2.99 ms | 293 TF/s per chip | 1.185 s | 1.102 s | 1.0 s (4x) |
| 32 x 16K | 11.64 ms | 301 TF/s per chip | 4.47 s | 4.14 s | 5 s SLA |

The 4K 4x target is below what the matmuls alone take on this Galaxy at 130 W. At 16K the matmuls fit under 5 s, but
attention adds ~3.5 ms per layer (~+1.4 s on the mean even at its current speed), so the 16K SLA is out of reach too.

### Nov 22: 32 x 16K past 4x

Two changes, both without precision changes:

- **Final-layer trim** (`prefill_dp4tp8.py`, default on from 8K tokens; `LLAMA_BH_PREFILL_DP4_TRIM_LAST=0` turns it off,
  `LLAMA_BH_PREFILL_DP4_TRIM_MIN` moves the threshold). In layer 80 only each prompt's last token feeds the LM head, so
  after QKV, rope and the KV-cache fill (all rows, needed by decode) the layer runs attention, WO, the residual add, the
  norm, FF13 and FF2 for that one row. On-device one-hot selectors (an fp32 position ramp compared with the uploaded
  last-token indices) pick the row; partial sums are added over the mesh column with a small all-gather. Logits match the
  full layer to 4 decimals in `test_dp4tp8_pcc.py` (unequal lengths per column, padding masked). ~13 ms less per 16K step.
  The ramps are freed after each forward: kept resident, they shifted decode timing slightly, and the closed-loop 32 x 4K
  line read 1.70 s instead of 1.61 s because its later rounds arrived bunched (round 0 itself was faster).
- **Parallel API servers** (`run_server.sh`: `SERVE_CLI=1` starts through `vllm serve`, `SERVER_ARGS=--api-server-count=4`).
  Tokenizing one 16K prompt takes ~27 ms, and the first step waits until each of the 4 lanes has a prompt, so a single
  frontend added ~110 ms before the first 16K step. The plugin's `server_example_tt.py` ignores `--api-server-count`
  (one process). With 4 frontends the first 16K group finishes at 1.67-1.70 s instead of 1.77-1.83 s. Needed a plugin
  fix: `platform.py` set `structured_outputs_config.disable_any_whitespace` on the config object shared with the engine
  args, and each extra API server process re-validated it and failed (backend "auto"); it now sets it on a copy.

| Run (32 x 16K, 3 fresh seeds each) | Mean TTFT per batch | Average | vs 28.7 s |
|---|---|---|---|
| Untrimmed (7 batches over 5 server starts) | 7.20-7.26 s | 7.221 s | 3.97x |
| Trim (9 batches) | 7.13-7.23 s | 7.165 s | 4.01x |
| Trim + 4 API servers | 7.106 / 7.091 / 7.101 s | **7.099 s** | **4.04x** |

32 x 4K with both: 1.525 s (was 1.61 s).

Confirmation (dp4_final2, trim + 4 API servers): 32 x 16K 7.154 / 7.084 s (5 batches with this config: mean 7.107 s,
4.04x, all under 7.18 s), 32 x 4K 1.510 s, 1 user 4K 0.413 s, needle 0 misses, quick MMLU 81.93%.

**DP8 x TP4 layout benchmark** (`tests/test_dp8tp4_bench.py`): the DP4 layer's op sequence and formats with random
weights, TP8 on mesh columns (axis 0, today) vs TP4 on mesh rows (axis 1, ring), all 32 chips busy, 80-layer forwards
alternating, steady-state rounds:

| Batch | Layout | ms / layer | ms per prompt-layer | Modeled 32-prompt mean (all at once) |
|---|---|---|---|---|
| 4K | DP4 x TP8 | 5.01 | 1.252 | 2.05 s |
| 4K | DP8 x TP4 | 8.01 | 1.001 (-20%) | 1.79 s (-13%) |
| 16K | DP4 x TP8 | 21.35 | 5.343 | 8.05 s |
| 16K | DP8 x TP4 | 37.95 | 4.743 (-11%) | 7.88 s (-2%) |

TP4 does each prompt's work faster (collectives move 3/4 instead of 7/8 of the rows on a 4-chip ring, wider matmuls,
half as many small ops per prompt); finishing 8 prompts per wave instead of 4 gives most of it back. Integrating it needs
TP4 weight copies (~12 GB per chip), moving each prompt's KV cache into the decode layout (one kv head per chip on the
lane's column; a row group spans all 4 columns), and a scheduler that admits 2 prompts per lane per step.

Follow-ups on DP8 x TP4:
- **TP4 on half-columns** (rows 0-3 / 4-7 of a column, KV stays in the lane's column) needs 4x4 submeshes: the linear
  all-gather on a submesh hung at its first call (killed, board reset). Submeshes also have their own allocator, so
  sharing the parent mesh's KV cache with them needs new C++ (a 4x4 variant of `unit_mesh::aggregate`).
- **Moving KV with `ttnn.point_to_point`** (`tests/test_kv_move_bench.py`): 2.7 GB/s per transfer (168 MB in 62 ms;
  transfers on different rows do run in parallel), no block-float support; a 4K step's staged move would take 849 ms.
- **Per-layer KV redistribution with the fast collectives** (benchmark config `tp4kv`): all-gather the row's kv heads
  (axis 1), zero them except on the prompt's lane column, slot the lane's 2 prompts (rows c and c + 4), reduce-scatter
  over heads down the column (axis 0); chip k of lane column c then holds head k of both prompts.

| Per prompt-layer | 4K | 16K |
|---|---|---|
| DP4 x TP8 | 1.209 ms | 5.367 ms |
| DP8 x TP4 without the KV move | 0.999 ms | 4.765 ms |
| DP8 x TP4 with the KV move | 1.054 ms (-13%) | 4.964 ms (-7.5%) |
| Modeled 32-prompt mean with the KV move vs DP4 x TP8 | 1.876 vs 1.991 s (-5.8%) | 8.238 vs 8.083 s (+1.9%) |

Net: ~6% on the 32 x 4K mean (~1.51 -> ~1.42 s), nothing at 16K.

**Built anyway, then removed** (was `tt/prefill_dp8tp4.py`, opt-in `LLAMA_BH_PREFILL_DP8=1`, prompts under `LLAMA_BH_PREFILL_DP8_MAX` =
8192; serving also needs 2 new prompts per lane per step, `NEWPF=2` in the run script, and
`VLLM_ENGINE_READY_TIMEOUT_S=2400` for the longer startup). TP4 weight copy on mesh rows (~12 GB per chip, cached as
`dp8tp4_*`), the per-layer KV move above, an 8-row LM-head input, generator steps of up to 8 prompts when more than 4
short ones wait (lane c on rows c and c + 4), warmup. `test_dp4tp8_pcc.py` with 8 prompts, 2 per lane
(`DP4PCC_USERS=0,1,8,9,16,17,24,25`): logits top-1 match on all 8, pcc 0.94-0.995 (same as DP4 on the shared
prompts); KV on decode's chips pcc 0.990-0.9998 at layer 0, 0.965-0.999 at layer 79 (DP4's range). 8 prompts in
0.682 s vs 0.383 s for 4 on DP4: 11% less time per prompt.

| 32 x 4K | DP4 only (dp4_final2) | DP8 on (dp8_b) |
|---|---|---|
| Round 0 (all 32 at once) | 1.794 s | 1.790 s |
| Rounds 1 / 2 (closed loop) | 1.365 / 1.369 s | 1.679 / 1.798 s |
| Mean / P99 | 1.51 / 3.13 s | 1.755 / 2.91 s |

The throughput gain is real (a DP8 step is 0.67-0.69 s for 8 prompts vs 0.76-0.78 s for two DP4 steps), but a DP8 step
is 1.75x a DP4 step, which is about what finishing in waves of 8 instead of 4 costs, so the all-at-once mean does not
move. Later arrivals wait behind the longer steps, and with 2 admissions per lane a step that is not full enough for DP8
runs two DP4 groups back to back. 32 x 16K unchanged (7.15 / 7.08 s). The module and its generator wiring were
removed since it did not help; the benchmarks (`tests/test_dp8tp4_bench.py`, `tests/test_kv_move_bench.py`) stay.

Tried and dropped on the way: `--api-server-count` through `server_example_tt.py` (ignored), a length-aware decode
interleave (removing decode insertions makes prefill steps throttle harder: 1.55 s each instead of 1.52 / 1.60).

Also tried and removed: a length-aware decode interleave in the plugin (4 or 8 prefill steps between decode insertions
when a step schedules >= 32K tokens, else 2): 32 x 16K 7.20 / 7.23 s vs 7.20 s, 32 x 4K unchanged.

### Nov 22 (later): energy per op, WO all-to-all, PV-only LoFi

**Energy per op** (`tests/test_op_energy.py`). The test loops each op of the shipped DP4 layer alone on all 32 chips for
~8 s while a thread polls `tt-smi`. It reports power, clock, time per op, and dynamic energy = (power - idle) x time.
Idle with the devices open is 58-62 W per chip at 1350 MHz, so only ~70 W of the 130 W cap does work. Per chip:

| Op | 16K: ms (sustained) / dyn mJ | 4K: ms / dyn mJ | pJ/FLOP (16K) |
|---|---|---|---|
| FF13 + SwiGLU | 5.64-5.87 / 394 (32%) | 1.49 / 95 (37%) | 0.21 |
| SDPA | 4.50 / 317-325 (26%) | 0.38 / 26 (10%) | 0.59 |
| FF2 MMRS | 3.70 / 258-265 (21%) | 1.00 / 68 (26%) | 0.28 |
| WO MMRS | 1.68 / 118 (10%) | 0.47 / 32 (12%) | 0.43 |
| QKV | 1.12 / 71 (6%) | 0.33 / 22 (9%) | 0.21 |
| 2 all-gathers | 2 x 1.42 / 41 (3%) | 2 x 0.37 / 9 (4%) | |
| norms, heads, rope, concat, adds | 32 (3%) | 6 (2%) | |

The model's matmul shapes alone have no slack left: they sit at the measured energy floor (0.20 pJ/FLOP dynamic for bfp4 x
bf8 LoFi). Two items are well above it:
- **WO** pays to write and reduce [M, 8192] partials from a K = 1024 matmul.
- **SDPA**: LoFi saves only 55 of 317 mJ, so about two thirds of SDPA's energy is softmax and data movement, not the
  matmuls. The exp is already a 2.5 cycle / 8 element bit trick.

Ruled out by this measurement (16K):
- **SDPA chunk sizes**: q256/k1024 saves 3%. q128 or q64 (the K/V chain modes) cost 25-75% more.
- **FF13 block shapes**: none beat today's M8/K32/N8.
- **Splitting MMRS into matmul + reduce-scatter**: equal for FF2, worse for WO.
- **bfp4 FF2 / WO / QKV weights**: about 1% each per layer at 16K, and bfp4 FF2 is 16% slower at 4K. Not worth the
  accuracy risk.

Full-layer precision trades at 16K, ms per layer at the cap (base 19.24-19.72):
- SDPA LoFi: -3%
- QKV bf8 output: -1%
- all bfp4 weights: -3%
- all five together: -6.5%

**WO via all-to-all** (opt-in, `LLAMA_BH_PREFILL_DP4_WO_A2A_MIN=<tokens>`, 0 = off). An all-to-all over the
column gives chip r every head of its own rows, [M/8, 8192] (~15 MB moved at 16K instead of ~124 MB of partials). Chip
r then applies the full WO (~5.7 GB per chip, cached as `dp4tp8_L*.wo_full`).
- **Same FLOPs and output layout.**
- **Closer to torch than the bf8-transport MMRS** (`tests/test_wo_a2a.py`: pcc 1.0000 vs 0.9988 at 16K).
- **Correctness:** `all_to_all_async_generic(x, 3, 2)` is the right argument order for this layout. The old
  `test_dp4_a2a_bench.py` check used (2, 3) and failed.
- **Time alone:** 16K 1.21 vs 1.57 ms. At 4K it is 0.52 vs 0.45 ms (slower), hence the 8K threshold.
- **Served:** 32 x 16K 7.086 / 7.044 / 7.042 s (3 seeds, 7.057 s) vs 7.107 s.
- **80-layer PCC:** equal to before.

**PV-only LoFi in SDPA** (C++ + kernel, was `TT_SDPA_PV_FIDELITY=0`; removed again once full SDPA LoFi became the
default, which covers it). Only softmax @ V runs at LoFi; QK^T stays HiFi2.
- **Why the Nov 21 attempt hung:** PV recorded a LoFi replay image but still executed the default-fidelity loop.
- **Fix:** PV setups now record at the PV fidelity, PV execute calls (`sdpa_pv_matmul_block_no_mop`) run it, and QK^T
  setups re-record their image.
- **One SDPA (8 heads):** 16K 3.69 -> 3.41 ms, 4K 0.365 -> 0.331 ms. pcc vs torch 0.99976 -> 0.99968.
- **80-layer PCC:** same logits as before (0.977-0.995, top-1 all match).
- **Served together with WO all-to-all (pv2_a):**
  - 32 x 16K 6.954 / 6.920 / 6.887 s (**6.920 s, 4.15x**)
  - 32 x 4K 1.496 s
  - 1 user 4K 0.418 s
- **Warmup:** no hang.

**Accuracy: both are opt-in.** Quick MMLU with every prompt on DP4 (`LLAMA_BH_PREFILL_DP4_MIN=512`, WO all-to-all
forced on from 512 tokens). The runs are deterministic: the control on today's code repeats the Nov 21 82.37% with 0 flips.

| Run | Quick MMLU | Lost / gained vs control |
|---|---|---|
| Control (today's code, defaults) | 82.37% | 0 / 0 |
| WO all-to-all | 81.05% | 18 / 3 |
| WO all-to-all, fp32 accumulation (`LLAMA_BH_PREFILL_DP4_WO_A2A_FP32=1`) | 81.67% | 15 / 7 |
| PV-only LoFi | 81.40% | 18 / 7 |
| Both (bf16 accumulation) | 81.32% | 15 / 3 |

The all-to-all is bit-exact across 60 back-to-back runs, synced or not (`tests/test_a2a_determinism.py`). Its pcc vs
torch (random data) and vs TP32 (80 layers, 700-4096 tokens) is as good as MMRS or better. Still, every numeric change
tried here loses ~0.7-1.3 points on the 1140-question quick set, whose stderr is 1.06 points. Full MMLU (~14K
questions) decides it:

| Full MMLU (14042 questions, every prompt on DP4) | Accuracy | Lost / gained vs control |
|---|---|---|
| Control | 82.99% +- 0.30 | |
| SDPA LoFi + QKV bf8 output + WO all-to-all with fp32 accumulation | 83.13% +- 0.30 | 150 / 170 |
| The above + bfp4 W2 (`LLAMA_BH_PREFILL_DP4_W2B4_MIN`) | 82.92% +- 0.30 | 245 / 236 |
| The above + bfp4 QKV and WO-full weights (`LLAMA_BH_PREFILL_DP4_ATTN_B4_MIN`) | 82.72% +- 0.31 | 341 / 303 |

On the full set the precision trades are neutral: the flips are symmetric. The quick-set losses were regression from
a lucky 1140-question draw, and every numeric change re-rolls ~20 borderline answers. Quick MMLU is too small to rank
precision changes; use `TESTS=accfull` (`/dev/shm/jw_dp4_sla4.sh`, ~19 min) and `scripts/mmlu_flips.py`.

So SDPA LoFi, QKV bf8 output and WO all-to-all (fp32 accumulation, from 8K tokens) are now the code defaults. bfp4 W2
for long prompts is opt-in until it is measured served.

| Served (Nov 22, 3 seeds for 32 x 16K) | 32 x 16K | 32 x 4K | 1 user 4K |
|---|---|---|---|
| dp4_final2 (start of today) | 7.107 s | 1.510 s | 0.413 s |
| + WO all-to-all (woa2a_a) | 7.057 s | 1.558 s | 0.418 s |
| + PV-only LoFi (pv2_a) | 6.920 s | 1.496 s | 0.418 s |
| SDPA LoFi + QKV bf8 + WO all-to-all fp32 (aggr_a) | 6.801 s | 1.518 s | 0.412 s |
| the same, 8 API servers (api8_a) | **6.798 s (4.22x)** | **1.472 s** | 0.420 s |
| the same, 16 API servers (api16_a) | 6.813 s | 1.487 s | 0.415 s |

8 API servers bring the first group in sooner than 4 (16K 1.58-1.61 vs 1.64 s, 4K 0.417 vs 0.444 s); 16 add nothing.
`scripts/start_probe.py` (4 x 16K, 1 output token):
- With token-id prompts, the first DP4 step starts 44 ms after the requests are sent and the token reaches the client
  23 ms after the forward.
- Text prompts add 50-140 ms of tokenization.

Idle power with the model loaded (between tests) is 62 W per chip, the same as with no model: no background kernels eat
the budget.

bfp4 W2 from 8K tokens (now the default, `LLAMA_BH_PREFILL_DP4_W2B4_MIN=8192`): served 32 x 16K 6.779 / 6.734 / 6.725 s
(6.746 s) vs 6.816 / 6.794 / 6.784 s without, same seeds.

bfp4 QKV + WO-full weights from 8K tokens (opt-in `LLAMA_BH_PREFILL_DP4_ATTN_B4_MIN=8192`): 32 x 16K 6.743 / 6.683 /
6.680 s (6.702 s, -0.65%), but full MMLU drifts down (82.72%, 341 lost / 303 gained vs control). Not worth it; opt-in.

MMRS chunk splits ranked by energy (`tests/test_mrs_energy.py`, in-model L1 headroom, sustained at the cap):

| Shape | Split: dyn mJ / ms sustained |
|---|---|
| FF2 16K, bfp4 W2 | (4,6) 233 / 3.32 (old forced), (8,3) 224 / 3.23, (6,4) 224 / 3.24, (4,4) 225 / 3.26, (2,6) 241, (4,3) 349 |
| FF2 4K | auto (1,6) 67 / 0.97; every forced split is worse |
| WO 4K | auto (1,3) 30 / 0.43; (1,4) 31.5, others worse |

So `env.sh` forces (8,3) for FF2 16K. Served 32 x 16K (chunk83_a, 3 seeds): 6.760 / 6.716 / 6.722 s (6.733 s) vs 6.746 s.

bf16 instead of fp32 DEST inside MMRS (a factory switch, tried and removed): a plain FF2-shaped matmul costs +5-16% energy
in fp32 DEST (`tests/test_dest_energy.py`), but inside MMRS bf16 DEST saves only 1-2% per op (FF2 16K 3.23 -> 3.19 ms,
4K 0.969 -> 0.946 ms, WO 4K 0.429 -> 0.424 ms sustained). The kernel keeps 8-tile subblocks either way. About 0.5% per
layer, not worth it.

Final-layer trim from 2K tokens (`LLAMA_BH_PREFILL_DP4_TRIM_MIN=2048`, trim2k_b): 32 x 4K 1.493 s, inside the 1.47-1.56 s
spread of the default runs. No measurable gain; the default stays 8K. Pitfall: running `c32` twice in one server start
replays the same seed, and the prefix cache then serves those prompts (the second run read 1.37 s with 12% shorter
steps). Never repeat a seed within one server start.

bfp4 activations for FF13 (the norm output before the second all-gather, tried and removed). The FF13 matmul at the cap
gets 8% faster (`tests/test_energy_floor.py` w4a4 4.88 vs w4a8 5.31 ms) and the 4 x 4K forward 0.376 -> 0.361 s. But
logits pcc vs TP32 drops 0.957 / 0.995 / 0.986 / 0.987 -> 0.885 / 0.982 / 0.955 / 0.946, and layer-79 V reaches 0.95.
Far worse than any weight-precision change.

Overlapping the all-gathers at 4K. A 4K layer is only lightly throttled (4.22 ms cool vs 4.50 sustained), so the
~0.75 ms of all-gathers per layer costs real time. 4 x 4K forward (`test_dp4tp8_pcc.py`), base 0.363 s:
- **Fused row-gather matmul for QKV** (`LLAMA_BH_PREFILL_DP4_RGMM=qkv`, `RGMM_MIN=2048`): 0.398 s.
- **Fused row-gather matmul for QKV + FF13:** 0.511 s.
- **Hidden-sharded layout** (`LLAMA_BH_PREFILL_DP4_LAYOUT=hidden`, fused all-gather matmuls): 0.459 s, and its logits
  are broken on the current code.

The fused ops give up matmul cores to the gather and lose more than the hidden gather saves. Row-gather now needs
M/256 % 8 == 0; other lengths fall back to the plain path (it threw at M = 3072).

Matmul block shapes for core balance. QKV is only 40 output tiles wide, so 8x8-tile blocks give 80 blocks for 120
cores. No smaller block beats the current configs at the cap (`tests/test_op_energy.py`, `OE_MM_VARIANTS`):
- QKV at 4K: 0.331 ms (M8 K16 N8) vs 0.331-0.499 ms for M4/N4/N5/M2.
- QKV at 16K: 1.121 ms vs 1.126-1.800 ms.
- FF13: N4 or M4 cost 4-8% more.

Inside the streaming SDPA kernel (8K, one chip, LoFi). Method: device-profiler zones (temporarily switching the
kernel's `std::false_type{}` profiling tag to true at the `call_step` site) plus `TT_METAL_DEVICE_PROFILER=1`; parsed
with `/dev/shm/zones.py`. The buffer held the first ~15% of each core's work. In that window the math thread is almost
always busy:

| Stage (math thread, per core) | Time |
|---|---|
| QK^T matmul + pack | 55 us |
| softmax @ V matmul + pack | 31 us |
| subtract-max | 8 us |
| max reduce + normalize | ~2 us |

The exp (8.7 us) and its pack (7.1 us) run on the pack thread in parallel. QK^T has the same FLOPs as PV but takes 1.8x
longer: its inner dimension is only head_dim / 32 = 4 tiles, so every 4 tile-MACs end in a DEST handoff and pack.

Tried: a full-sync DEST, so QK^T subblocks could be 2 x 8 tiles (a factory switch sizing for 16 tiles, tried and
reverted). It is slower: 4K 0.304 -> 0.38-0.40 ms, 16K 3.37 -> 3.83 ms. Losing math/pack double buffering costs more
than the bigger subblocks save.

Upper bounds on softmax-side rewrites. These are timing experiments that break correctness, kernel restored after each:

| Removed (one chip, LoFi, q256/k512) | 4K | 16K |
|---|---|---|
| nothing | 0.304 ms | 3.370 ms |
| the row-sum pack (each P tile packed again with L1 accumulation) | 0.312 ms | 3.377 ms |
| the exp entirely | 0.287 ms (-5.6%) | 3.272 ms (-2.9%) |
| the QK^T score (S) pack | 0.325 ms | 3.364 ms |
| the QK^T unpack transpose of K (as if K were stored pre-transposed) | 0.335 ms | 3.403 ms |

So neither the planned rewrite (row sum from the PV matmul via a ones column in V) nor a DEST-resident softmax (never
packing S) would gain anything, and even a free exp saves
under 1% of a layer. SDPA time is the two matmuls and their S / P packs, fixed by head_dim 128. That closes the SDPA
work.

**Residual-add fusion**, two attempts:
- **WO all-to-all matmul:** `ttnn.experimental.dit_minimal_matmul_addcmul_fused` computes x + (g @ WO) * ones_row, but
  it requires the residual and weight tile formats to match (`ternary_a_tile_size must be equal to in1_tile_size`). Our
  residual is bf16 and WO is bf8, so the fusion would cost precision or bf16 weights. Reverted.
- **RMSNorm with `residual_input_tensor`** (norm(x + r) in one pass, which still lacks the x + r output the next layer
  needs). At the cap it costs about as much as the separate ops: 4K 0.146 ms vs norm 0.097 + add 0.064, 16K 0.367 ms vs
  0.211 + 0.197. A real fused norm + sum kernel would save under 1% of a layer, so it is not worth writing.

**Independent review follow-ups** (Nov 23):
- **Leakage vs temperature.** In the per-chip log, clock and temperature are uncorrelated across chips (r = -0.07).
  The slow chips are not the hot ones (chip 18 at 66 C, while chip 17 at 67.9 C is among the fastest). Clock does not
  drop as a chip warms (median slope +4 MHz/C), and temperatures stay at 60-68 C. No cooling lever.
- **SDPA at 4K fills 64 of 120 cores** (128 q chunks make 64 causal pairs). One chip:

  | q heads | 4K q256 | 16K q256 |
  |---|---|---|
  | 8 (model) | 0.334 ms | 3.40 ms |
  | 15 | 0.497 ms | 6.42 ms |
  | 16 | 0.576 ms | 6.94 ms |

  With all cores busy, contention takes most of the gain. A balanced 8-head SDPA would save ~0.05 ms per 4K layer
  (~1%). Not built.

**Chip-to-chip clock spread under load** (Nov 23, after the power cycle). `test_op_energy.py` layer loops at 16K and 4K
with `OE_PERCHIP_LOG`, 178 loaded samples, summarized with `perchip_summary.py`:
- Per-chip clocks span 1135-1229 MHz (mean 1193). Power spans 117-142 W.
- Chips 18 and 7 (~1136 MHz) are the slowest in 57 and 55 of the samples, then chips 21 and 24. They draw full power at a
  lower clock: leakier silicon.
- Trays average 1188-1204 MHz.
- Every column's collectives wait for its slowest chip, and the step waits for the slowest column, so the mesh runs near
  ~1140 MHz instead of the 1193 MHz average: a ~4-5% tax.
- Software cannot recover it: ttnn runs one SPMD program with equal shard shapes per chip. Giving slow chips less work
  needs per-device programs, and zero padding computes the padded work anyway.

**Never run a second tt-smi reader during a device test.** On Nov 22 an external per-chip poll loop ran next to
`test_op_energy.py`'s own poller.
- **What happened:** every snapshot failed. At 19:53:12, with no load, chips 8-15 (one tray) stopped reporting at once.
  The test hung, a chip then read 0xffffffff over PCIe, and the first `tt-smi -glx_reset_auto` failed. The board needs a
  reset (or a power cycle of that tray) before more runs.
- **Safeguard since then:** the test's `Poller` is the only reader. It records per-chip clock, power and temperature
  (`OE_PERCHIP_LOG=<path>`; summarize with `sla_scripts/bh_tp32/perchip_summary.py`). It fails the test when chips drop
  off or 20 reads in a row fail.
- **Recovery (Nov 23):** UBB1's ASIC power rails were off. In the BMC: Power_UBB1 10 W, chip temps 0 C, and SEL showed
  "FRB3 processor startup/init failure" at 19:53:05. Every `tt-smi` reset failed at device 8 for that reason. Fixed by
  `sudo ipmitool chassis power cycle` (the host reboots in ~2 min), then `tt-smi -glx_reset_auto`, remounting
  `/mnt/nvme0`, and restoring `/dev/shm` scripts. `test_mesh_health.py` passes.

Ruled out at layer level: FF2 / WO as a plain matmul + `ttnn.reduce_scatter` instead of matmul_reduce_scatter_prefill.
Per op at 4K the split looked ~9 mJ cheaper, but the whole layer is slower: 4K 4.50 -> 5.00 ms, 16K 19.57 -> 21.50 ms.

### Nov 22 (evening): decode after every prefill step, 32 x 4K 1.47 -> 1.15 s

The closed-loop 32 x 4K line is 96 requests at concurrency 32. Its later rounds (64 of the 96) arrive when earlier users
finish their 500 output tokens. With the plugin's default decode interleave (one decode step per 2 prefill steps),
almost no decode ran during round 0's prefills, so all users decoded in lockstep afterwards. They finished together,
and the next requests arrived in bursts of 16 (rounds 1/2: 1.34 s mean TTFT).

`"decode_interleave_prefill_steps": 1` (now in `env.sh`'s `TT_CFG_EXTRA`) runs a decode step after every prefill step.
Finish times spread out, and the later rounds stop bunching:

| Interleave (vLLM, c32_4k_500) | Round 0 | Round 1 | Round 2 | Mean / P99 | 32 x 16K | 240 RPM TTFT | Decode TPOT mean / P99 |
|---|---|---|---|---|---|---|---|
| every 2 prefill steps (default, api8_a) | 1.741 | 1.333 | 1.342 | 1.472 / 3.06 s | 6.80 s | 0.40 / 0.91 s | 24.8 / 26.9 ms |
| every 8 (il8_a) | 1.719 | | | 1.699 / 3.01 s | 6.74 s | 0.39 / 0.90 s | 24.3 ms |
| every step (il1_a) | 1.808 | 0.839 | 0.754 | **1.134** / 3.18 s | 6.78 s | 0.39 / 0.95 s | 25.9 / 27.4 ms |
| every step, repeat (il1_b) | 1.789 | 0.937 | 0.764 | **1.163** / 3.15 s | 6.80 s | | 26.0 / 27.6 ms |
| every step, 2 decode steps (il1d2_a) | 1.866 | 0.677 | 0.985 | 1.176 / 3.72 s | 6.87 s | 0.40 / 1.00 s | 26.5 / 28.2 ms |

60 RPM with it (il1_b): TTFT 0.36 / 0.68 s, decode 35.6 / 30.3 tok/s (unchanged). Decode in the 32 x 4K run stays
near 38 tok/s per user (25 target). Official bench with it as the default (sla_nov22b): 32 x 4K 1.09 / 3.16 s.

What is left in the later rounds: they now arrive as groups of 4 every ~0.39 s, the prefill cadence. Each group waits
for the previous group's step and then runs its own (TTFT 0.78 s = 2 steps). In a closed loop the arrivals are paced
by the completions, so the arrival period equals the service period, and the one-step backlog never drains. From here
only a shorter step helps:
- 32 x 4K mean ~ (32 x (0.04 + 4.5 step) + 64 x 2 step) / 96.
- The mean reaches 1.0 s at a ~0.33 s step (today ~0.38 s), which the 130 W energy budget does not allow.

Intake, measured: a temporary plugin log of the time from vLLM's `arrival_time` to the request's first scheduled prefill
step (intake_a). Round 0's first DP4 step is scheduled 9-28 ms after arrival. Later groups wait 370-400 ms (one step)
or ~760 ms (two steps), behind earlier groups. With the ~23 ms output path (`scripts/start_probe.py`), non-compute
overhead per request is under 50 ms (2-4% of the 32 x 4K mean), most of it vLLM's own request and stream handling.

Two attempts to clear that backlog, both worse:

| Run | Round 0 | Rounds 1 / 2 | 32 x 4K mean |
|---|---|---|---|
| sla_nov22b (default now) | 1.790 | 0.737 / 0.739 | 1.088 s |
| 2 new prefills per lane per step (`NEWPF=2`) | 1.871 | 1.338 / 1.281 | 1.496 s |
| plugin patch, tried and reverted: admit 2 when exactly 2 wait and decodes run | 1.843 | 0.907 / 0.908 | 1.220 s |

bfp4 W2 from 2K tokens (now the default, `LLAMA_BH_PREFILL_DP4_W2B4_MIN=2048`):
- **4K MMRS plan:** under the in-model L1 budget the planner picks a good plan, (1,4): 0.836 vs 0.969 ms sustained for
  bf8, -13% FF2 energy (`tests/test_mrs_energy.py`). The earlier 4K slowdown came from tests without that headroom.
- **Forward:** 4 x 4K 0.376 -> 0.363 s.
- **Served (w2b4_2k):** 32 x 4K round-0 spacing 0.382 s vs 0.388-0.392 s, round 0 1.761 s vs 1.79-1.81 s, mean 1.108 s
  (the 1.09-1.16 s band); 32 x 16K 6.784 s.

`NEWPF=2` gives round 0 waves of 8, which removes the spreading. With the shallow-queue rule, the doubled steps delay
both groups, and the one-step backlog forms again right after.

Fused row all-gather + matmul (Nov 20, C++ + kernels, opt-in `LLAMA_BH_PREFILL_DP4_RGMM=qkv`, prompts >= 16K):
`strided_all_gather_minimal_matmul_async(dim=2)` runs all_gather_async's minimal-default transport (fabric mux, 4
workers per direction per link) with a `ROW_PROGRESS` hook that counts landed tiles per direction on every matmul in0
injector. Each matmul core row takes a 1/8 sub-band of every chip's slice and walks the slices in ring-arrival order
(own, +1, -1, +2, ...), so later slices land while earlier ones compute (`RowGatherMap`,
`fabric_bound_matmul_dataflow_common.hpp`). Correct (gathered tensor bit-exact, PCC 0.9997-0.99999 vs separate,
`tests/test_row_gather_mm.py`). One column:

| Shape | Separate (all-gather + full-grid matmul) | Fused |
|---|---|---|
| QKV 8K | 1.27 ms | 1.23 ms |
| QKV 16K | 2.45 ms | 2.13 ms |
| QKV 32K | 4.75 ms | 4.19 ms |
| FF13 16K | 6.24 ms | 6.77 ms |

FF13 loses because the matmul gives up 2 of 10 core rows to the 20 transport cores. In the 80-layer model it does not
pay: 4 x 16K forward 2.10 s vs 2.00-2.06 s without. One call costs 0.34 ms of host dispatch, and the per-layer gain
(0.1-0.3 ms) is below run-to-run noise. Keep it off.

What decided the design (measured, `tests/test_ag_variants.py`):
- Fabric throughput scales with worker cores: 1 direct worker per direction 27 GB/s, 2 workers 50, 4 through a mux
  78 GB/s.
- Two direct workers on one link hang, so more workers need a mux.
- A first multicast transport with 2 cores reached ~16 GB/s.
- One band per source chip cannot overlap with a ring gather: the far chip's rows arrive last, so that band ends at
  gather + matmul.

RMSNorm (`LLAMA_BH_PREFILL_DP4_NORM=rms`, opt-in): `ttnn.rms_norm` (its streaming large-row kernel) + typecast to bf8
instead of the 5-op ones-column norm.

| Measurement | 5-op norm | rms_norm |
|---|---|---|
| One chip, 4K rows, incl. residual add (`tests/test_dp4_norm_fast.py`) | 0.35 ms | 0.20 ms |
| One chip, 16K rows, incl. residual add | 0.80 ms | 0.63 ms |
| Profiled 4K layer, synced | 7.57 ms | 6.9 ms |
| 80-layer forward, 4 x 4K, unsynced | 432-453 ms | 426-443 ms |
| 80-layer forward, 4 x 16K, unsynced | 2.00-2.06 s | 2.07 s |

The 16K difference is within noise. PCC and top-1 match the 5-op norm.

The DP4 forward does not speed up by what the per-op savings predict. Next is to find where the unsynced forward's
time goes (device trace of a whole layer) before more op work. SDPA at 16K is at its tuned best (3.75 ms, q bf8,
q256/k512, ~147 TFLOPs): beating it needs a new causal kernel.

Next:
- Hide the two all-gathers: tried, see above (QKV only, ~1% at 16K, below noise end to end).
- Fuse heads + RoPE + typecast and the norm (~0.5 ms of ~5.6 per 4K layer).
- SDPA is still ~10% of peak: a better causal kernel is the main lever at 16K.

Older notes:
- (superseded by DP4 x TP8) One full layer on one chip (`test_pp_stage_bench.py`) gives a 32-stage pipeline 58-70K tok/s,
  but each prompt then crosses all 80 layers on one chip in sequence (~1.9 s TTFT for a single 4K prompt).
- 128 users: decode batch 128 (Wormhole TP8/DP4 ran 128 users at ~15.5 tok/s/user at 4K).
- Ruled out: 3-4 prefill links (the fabric has 2 eth channels per neighbor), decode interleave off, the prefill<->decode
  switch (~3% at 240 RPM), fused MM+RS for FF1/FF3/FF2 in-model, SDPA grid 11.

### Nov 23: closed-loop scheduling, 32 x 4K 1.09-1.16 -> 0.93-0.99 s

32 x 4K (`c32_4k_500`) is a closed loop: 96 prompts with 32 in flight, so each user sends a new prompt as soon as the
previous answer ends. Per round (`rounds.py <result.json>`): round 0 (the 32-prompt burst) took 1.75-1.79 s and rounds
1-2 took 0.74-0.82 s, two prefill steps instead of one.

Why: users prefilled back to back finish decoding one token apart. Each re-arrives about 25 ms after its last token
(stream, client, tokenize, engine), by when the next group has also finished, so every group waited behind the previous
group's ~0.4 s prefill step. The backlog lasts the whole run.

Fix (Gio plugin, `lane_scheduler.py`; `vllm_tt_plugin_gio_dp4.patch` regenerated; defaults in `env.sh`):
- `"decode_interleave_decode_steps": 4`: four decode steps between burst prefill steps, so finishes are spaced by more
  than the re-arrival time.
- `TT_PLUGIN_BUSY_GATHER_MS=8`: while decoding, hold a prefill step until no request has reached the engine for 8 ms
  (at most 50 ms) when some lane still has a free slot and nothing waiting, so a group of four is prefilled in one step.
  Uses engine-side receipt time: the request's `arrival_time` is the frontend's, 10+ ms stale.
- `TT_PLUGIN_LONG_PROMPT_TOKENS=8192`: one decode step per prefill step while the next prompt is longer than 8K
  (without it 32 x 16K went 6.78 -> 7.05 s).
- `TT_PLUGIN_FRESH_SKIP=1`: right after a prefill step, if at most one prompt per lane waits and all arrived during that
  step, prefill them at once instead of first interleaving (heals a collision instead of letting it chain).

Results: rounds 1-2 0.42-0.46 s, round 0 1.96 s (the extra decode steps cost the burst ~0.2 s). Means 0.955, 0.961,
0.992, 0.992, 0.956, 0.975 s with all four on (fsab2); other lines unchanged (sla_nov23, before FRESH_SKIP).
32 x 16K 6.75-6.82 s. A strict one-round burst of 32 x 4K is still ~1.75-1.97 s.

Tried and removed: skipping decode steps for later groups to space their finishes (`decode_gap`, also the earlier
stagger): 1.11-1.39 s. FRESH_SKIP without the one-prompt-per-lane limit fired in round 0 and broke its spacing (1.05 s).
`TT_PLUGIN_POLICY_FILE` (a JSON file re-read when it changes: dsteps, busy_gather_ms, fresh_skip, trace) A/Bs these on
one server start; `"trace": 1` logs lane states per prefill step.

Speed of light for 32 x 16K at once (83 PFLOP): ~4.3 s mean in waves of 4 if every FLOP ran at the measured matmul
rate at the 130 W cap (~340 TF/s per chip), ~2.9 s at full clock; ~5.9 s with attention at its measured energy cost.
