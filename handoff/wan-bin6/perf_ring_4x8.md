# Wan2.2 perf on a Bin6 BH Galaxy: 4x8 ring

Branch: `jameslee/test-pr60190-dram-endpoints` @ `adfc799f3a1` (PR #60190 DRAM endpoints + DRAM-harvest VAE chunking).
Machine: `bh-glx-120-b10u20` (single BH Galaxy, 32 Blackhole chips, firmware bundle 19.13.2).
Run 2026-10-20 by a Claude Code session for jameslee@tenstorrent.com.

## TL;DR

- **The 4x8 ring runs on this machine.** Unlike `b10u08` (see `README.md`), `b10u20` is cabled as a full 4x8 torus.
- **All 8 runs passed, none hung:** T2V and I2V at 480p and 720p, each at the default power limit and at `TT_METAL_TDP_LIMIT_WATTS=130`. That includes 720p with the chunked VAE, which hung 17 of 17 times on `b10u08` without the 130 W setting.
- **Only VAE decode misses its CI target, by 17–29%, and the chunking is why.** Denoise and total are 24–26% under target at the default limit. On 7-bank DRAM the branch decodes in chunks (`vae_t_chunk_size=7`). In an A/B on b10u20 at 480p, chunking costs 0.18–0.19 s (0.49–0.52 s against 0.30–0.35 s full-T), about 0.3–1% of the total. Full-T meets the 0.42 s target on this machine. With PR #61784's readback fix, CI meets both VAE targets full-T on 8 banks.
- **Against an 8-bank CI Galaxy at the same 130 W, b10u20 is 3.6–4.9% slower in denoise and 3.7–5.3% slower in total.** The reference is PR #61784 (`main` plus the VAE readback fix, #60670) on `d07` hosts: T2V 39.08 / 123.86 s denoise at 480p / 720p and I2V 720p 124.20 s, against 40.99 / 128.41 / 128.63 s on b10u20. Forcing 130 W on a `d07` host changed nothing, so power isn't the cause. The DRAM bank and the machine itself are still mixed together; unharvesting b10u20 (`gddr/README.md`) would separate them.
- **With CI's readback fixed, the VAE gap is all chunking.** CI's full-T at 480p (0.278 s) matches b10u20's full-T (0.299 s at 130 W). b10u20's chunked decode is 0.21 s slower at 480p and 0.41 s at 720p, where 7 banks can't run full-T (#59504). This branch on an 8-bank CI Galaxy matches `main` on the same host within ±0.5%, so PR #60190 itself costs nothing there.
- **75 W is a `bh-mod` override; the stock limit is 130 W.** Every chip on b10u20 has `chip_limits.tdp_limit` = 75 against a firmware default of 130 (`bh-mod get --delta`), set together with the Bin6 soft harvest. `TT_METAL_TDP_LIMIT_WATTS=0` returns to the override (75 W), not to 130 W. So the "130 W" runs are at stock power, and the 75 W runs are at a deliberately reduced limit; going to 130 W makes every config 17–26% faster. This probably reverses the README's finding 5 for `b10u08`: setting 130 W restored stock power rather than capping it.

## Machine checks

- **Ring cabling.** After `tt-smi -glx_reset`, every chip has ports 0–7 trained (8–9 are unconnected) and 2 links to each of exactly 4 neighbours. A graph check (`networkx`) of the `check_links.py --map` output says the link graph is isomorphic to a 4x8 torus:

  ```
  T1L1 T2L1 T2L2 T2L3 T2L4 T1L4 T1L3 T1L2
  T3L1 T4L1 T4L2 T4L3 T4L4 T3L4 T3L3 T3L2
  T3L5 T4L5 T4L6 T4L7 T4L8 T3L8 T3L7 T3L6
  T1L5 T2L5 T2L6 T2L7 T2L8 T1L8 T1L7 T1L6
  ```

  Before the first reset most intra-tray links were down. Reset and run `check_links.py` before using the machine.
- **Fabric.** With `single_bh_galaxy_torus_xy_graph_descriptor.textproto` (the CI descriptor), the placement step commits `4x8_Mesh_flat_torus_xy (TORUSXY)` and the fabric initializes as `FABRIC_1D_RING` on all 32 chips. A 2-link `ttnn.all_gather` with `Topology.Ring` gave correct data on both cluster axes (ring sizes 4 and 8).
- **DRAM.** Bin6: `dram_grid_size()` = 7x1. Harvest masks are mixed: each of `1, 2, …, 128` appears on 4 chips.
- **Power and harvest are `bh-mod` overrides.** `chip_limits.tdp_limit` = 75 W (firmware default 130) and `dram_table.soft_harvest_dram_mask` = one channel per chip, on all 32 chips. All 8 channels train; the harvest is soft and can be toggled with the scripts in `gddr/` (see `gddr/README.md`).

## Results

`models/tt_dit/tests/models/wan2_2/test_performance_wan.py::test_pipeline_performance[blackhole-<t2v|i2v>-resolution_<480p|720p>-ring_bh_4x8_sp1tp0]`: SP=8 on axis 1, TP=4 on axis 0, 2 links, `is_fsdp=False`, 40 steps, 81 frames, one timed run after a 2-step warmup. Environment as in the BH single-Galaxy CI job (`tests/pipeline_reorg/models_e2e_tests.yaml`). Seconds:

| Config | TDP | Text enc | Image enc | Denoise | VAE decode | Total |
|---|---|---|---|---|---|---|
| T2V 480p | 75 W (bh-mod override) | 0.061 | – | 55.75 | 0.523 | 56.34 |
| T2V 480p | 130 W | 0.069 | – | 40.99 | 0.492 | 41.56 |
| T2V 720p | 75 W (bh-mod override) | 0.061 | – | 154.97 | 1.096 | 156.14 |
| T2V 720p | 130 W | 0.062 | – | 128.41 | 1.006 | 129.49 |
| I2V 480p | 75 W (bh-mod override) | 0.067 | 0.921 | 55.97 | 0.514 | 57.48 |
| I2V 480p | 130 W | 0.067 | 0.786 | 41.40 | 0.489 | 42.75 |
| I2V 720p | 75 W (bh-mod override) | 0.061 | 1.068 | 154.78 | 1.079 | 157.00 |
| I2V 720p | 130 W | 0.062 | 0.954 | 128.63 | 1.009 | 130.66 |

Per denoise step: 1.39 s (480p) and 3.87 s (720p) at 75 W; 1.02 s and 3.21 s at 130 W.

**Against the CI targets** (`t2v_metrics()` in the test, BH 4x8 ring; I2V uses the same targets):

| Res | Stage | CI target | b10u20, 75 W | b10u20, 130 W |
|---|---|---|---|---|
| 480p | Denoise | 75.0 | 55.75 | 40.99 |
| 480p | VAE decode | 0.42 | **0.52** | **0.49** |
| 480p | Total | 76.0 | 56.34 | 41.56 |
| 720p | Denoise | 205.0 | 154.97 | 128.41 |
| 720p | VAE decode | 0.85 | **1.10** | **1.01** |
| 720p | Total | 206.0 | 156.14 | 129.49 |

Every frame range is 0–255 (not the flat grey of `main` on this hardware). These perf runs do not score quality (no CLIP / VBench), and frames were not inspected.

## Comparison with CI on non-Bin6 BH Galaxies

**Reference runs: PR #61784** (`jameslee/fix-60670-reassemble-threads` @ `0e7d662f2a1`, `main` plus the fix for #60670). Every nightly before this lost about 0.4 s per VAE pass on the BH Galaxy CI hosts. `_reassemble_2d`, the host step of the VAE readback, runs 32 small OpenMP copies with 64 torch threads. Each copy waited about 13 ms for threads that share a core with the Weka client's two polling threads, which are pinned at 100%. PR #61784 caps torch at one thread per physical core for that step. So these are the first CI numbers whose VAE time can be compared with b10u20's.

The PR runs are `main` plus the fix, not this branch: 8 banks, full-T VAE, no PR #60190. Both landed on `d07` hosts. Forcing 130 W on `d07u02` (run 39523036841) didn't change its speed, so those hosts run at 130 W by default. That was measured on `d07u02` only; `d07u08` matches it, so it is assumed to be the same.

Seconds, lower is better.

**T2V** ([run 39527173942](https://github.com/tenstorrent/tt-metal/actions/runs/39527173942), VAE targets 0.42 s at 480p and 0.85 s at 720p):

| Run | Host | DRAM banks | VAE mode | Power limit | 480p denoise | 480p VAE | 480p total | 720p denoise | 720p VAE | 720p total |
|---|---|---|---|---|---|---|---|---|---|---|
| PR #61784 on CI | bh-glx-120-d07u08 | 8 | full-T | 130 W (host default) | 39.08 | 0.278 | 39.46 | 123.86 | 0.591 | 124.57 |
| this branch | bh-glx-120-b10u20 | 7 | chunk 7 | 130 W | 40.99 | 0.492 | 41.56 | 128.41 | 1.006 | 129.49 |
| **b10u20 minus CI** | | | | | **+1.91 (+4.9%)** | **+0.214** | **+2.10 (+5.3%)** | **+4.55 (+3.7%)** | **+0.415** | **+4.92 (+3.9%)** |
| this branch | bh-glx-120-b10u20 | 7 | chunk 7 | 75 W (bh-mod override) | 55.75 | 0.523 | 56.34 | 154.97 | 1.096 | 156.14 |

**I2V 720p** ([run 39527176210](https://github.com/tenstorrent/tt-metal/actions/runs/39527176210); the bh_sc1 I2V job runs 720p only; VAE target 0.85 s):

| Run | Host | DRAM banks | VAE mode | Power limit | Text enc | Image enc | Denoise | VAE | Total |
|---|---|---|---|---|---|---|---|---|---|
| PR #61784 on CI | bh-glx-120-d07u02 | 8 | full-T | 130 W (host default) | 0.080 | 1.085 | 124.20 | 0.602 | 125.97 |
| this branch | bh-glx-120-b10u20 | 7 | chunk 7 | 130 W | 0.062 | 0.954 | 128.63 | 1.009 | 130.66 |
| **b10u20 minus CI** | | | | | −0.018 | −0.131 | **+4.43 (+3.6%)** | **+0.407** | **+4.69 (+3.7%)** |
| this branch | bh-glx-120-b10u20 | 7 | chunk 7 | 75 W (bh-mod override) | 0.061 | 1.068 | 154.78 | 1.079 | 157.00 |

What this shows:

1. **Denoise: b10u20 is 3.6–4.9% slower than an 8-bank CI Galaxy at the same 130 W,** consistently across T2V 480p, T2V 720p and I2V 720p. Power is ruled out. What's left is the missing DRAM bank, PR #60190's DRAM endpoints (which cost nothing on 8 banks, see below), or the machines themselves. Unharvesting b10u20 (`gddr/README.md`) would show which.
2. **VAE: with the readback fixed, full-T is the same on both machines, and the chunking is the whole gap.** At 480p, CI's full-T (0.278 s) is within 0.02 s of b10u20's full-T at 130 W (0.299 s in the A/B below). b10u20's chunked decode is 0.21 s slower at 480p and 0.41 s slower at 720p. The 720p figure compares chunked on 7 banks with full-T on 8, because full-T runs out of memory on 7 banks (#59504). With the fix, CI meets both VAE targets. b10u20 misses both, only because it chunks.
3. **Totals: b10u20 is 3.7–5.3% slower,** about 90% of it denoise.
4. **Local b10u20 runs don't show #60670 even though b10u20 runs the same Weka pollers** (two `wekanode` threads at 100%). Full-T took 0.30–0.35 s here without the fix. Why is not known; one unverified guess is that CI's container and `mpirun` leave the job different CPUs. The b10u20 numbers in this file don't include PR #61784.

### Older CI runs, before the VAE readback fix

Kept for denoise and for the 75 W comparison: no PR #61784 run landed on a `d04` host. The VAE column carries the extra ~0.4 s from #60670.

Source: the nightly `TT-DiT Wan2.2-T2V-A14B single galaxy e2e tests [bh_sc1]` job (`models-t1-e2e-tests.yaml` on `main`), which runs this same `ring_bh_4x8_sp1tp0` T2V test with the same environment. The CI hosts have 8 DRAM banks: their 720p runs decode the full 21 latent frames in one pass, which runs out of memory on 7 banks (#59504), and UMD logs no "inconsistent DRAM harvesting" warning.

| Run | Host | DRAM banks | VAE mode | 480p denoise | 480p VAE | 480p total | 720p denoise | 720p VAE | 720p total |
|---|---|---|---|---|---|---|---|---|---|
| `main` nightly 10-13 (`3cdbe9ca17`) | bh-glx-120-d04u20 | 8 | full-T | 56.65 | 0.736 | 57.47 | 157.38 | 1.030 | 158.51 |
| `main` nightly 10-14 (`3de19707aa`) | bh-glx-120-d04u14 | 8 | full-T | 56.28 | 0.699 | 57.07 | 156.15 | 1.107 | 157.35 |
| `main` nightly 10-20 (`909748641c`) | bh-glx-120-d04u08 | 8 | full-T | 56.34 | 0.758 | 57.20 | – | – | – (hung) |
| this branch, b10u20, 75 W (bh-mod override) | bh-glx-120-b10u20 | 7 | chunk 7 | 55.75 | 0.523 | 56.34 | 154.97 | 1.096 | 156.14 |
| | | | | | | | | | |
| `main` nightly 10-15 (`d1f9992507`) | bh-glx-120-d07u02 | 8 | full-T | 39.59 | 0.713 | 40.39 | 124.47 | 1.028 | 125.59 |
| `main` nightly 10-16 (`426b1f10d4`) | bh-glx-120-d06u08 | 8 | full-T | 38.79 | 0.739 | 39.63 | 122.47 | 0.977 | 123.55 |
| `main` nightly 10-17 (`628dbed7cc`) | bh-glx-120-d07u08 | 8 | full-T | 39.13 | 0.747 | 39.96 | 123.80 | 1.031 | 124.93 |
| `main` nightly 10-18 (`9f98dee9c5`) | bh-glx-120-d07u08 | 8 | full-T | 39.12 | 0.736 | 39.94 | 123.96 | 1.024 | 125.07 |
| this branch on CI ([run 39492727180](https://github.com/tenstorrent/tt-metal/actions/runs/39492727180), 10-20) | bh-glx-120-d06u08 | 8 | full-T | 38.99 | 0.745 | 39.82 | 122.36 | 1.024 | 123.49 |
| this branch on CI, forced 130 W ([run 39523036841](https://github.com/tenstorrent/tt-metal/actions/runs/39523036841), 10-21) | bh-glx-120-d07u02 | 8 | full-T | 39.19 | 0.722 | 40.00 | 123.89 | 0.994 | 124.98 |
| this branch, b10u20, 130 W | bh-glx-120-b10u20 | 7 | chunk 7 | 40.99 | 0.492 | 41.56 | 128.41 | 1.006 | 129.49 |

(10-19 got no perf numbers.)

- **The CI hosts fall into two speed groups.** The `d04` hosts match b10u20 at its 75 W override within 1–2% (b10u20 is slightly faster). That suggests the `d04` hosts carry the same 75 W override, but it isn't confirmed. The `d06`/`d07` hosts run at 130 W.
- **PR #60190 changes nothing on 8-bank hardware.** This branch on CI landed on `d06u08`, the same host as the 10-16 `main` nightly, and matches it within ±0.5% (denoise 38.99 against 38.79 s at 480p, 122.36 against 122.47 s at 720p).

I2V 480p has no run with the fix. The one CI number ([run 39489301673](https://github.com/tenstorrent/tt-metal/actions/runs/39489301673), `main` @ `f56b5484466` plus a 3600 s timeout and 480p added, VAE carries #60670):

| Host | Code | DRAM banks | VAE mode | Power limit | Text enc | Image enc | Denoise | VAE | Total |
|---|---|---|---|---|---|---|---|---|---|
| bh-glx-120-d04u20 (CI) | `main` | 8 | full-T | unknown | 0.081 | 1.150 | 57.06 | 0.750 | 59.05 |
| bh-glx-120-b10u20 | this branch | 7 | chunk 7 | 75 W (bh-mod override) | 0.067 | 0.921 | 55.97 | 0.514 | 57.48 |
| bh-glx-120-b10u20 | this branch | 7 | chunk 7 | 130 W | 0.067 | 0.786 | 41.40 | 0.489 | 42.75 |

### VAE chunking A/B (b10u20, T2V 480p, 75 W, same build)

`vae_t_chunk_size` forced by a pytest plugin that logs the value it actually applied. Each row is one full test run, listed in the order they ran (alternating settings).

| Run order | `vae_t_chunk_size` | VAE decode | Denoise | Total |
|---|---|---|---|---|
| 1 | `None` (full-T, same as `main`) | 0.349 s | 55.63 s | 56.06 s |
| 2 | `7` (this branch on 7-bank DRAM) | 0.531 s | 55.66 s | 56.27 s |
| 3 | `None` (full-T, same as `main`) | 0.343 s | 55.74 s | 56.16 s |
| 4 | `7` (this branch on 7-bank DRAM) | 0.520 s | 55.69 s | 56.29 s |
| **Mean, full-T** | `None` | **0.346 s** | 55.69 s | 56.11 s |
| **Mean, chunked** | `7` | **0.526 s** | 55.68 s | 56.28 s |
| **Cost of chunking** | | **+0.180 s (+52%)** | ~0 | +0.17 s (+0.3%) |

The same A/B at `TT_METAL_TDP_LIMIT_WATTS=130` (all 32 chips confirmed 130 W in every run):

| Run order | `vae_t_chunk_size` | VAE decode | Denoise | Total |
|---|---|---|---|---|
| 1 | `None` (full-T, same as `main`) | 0.295 s | 40.48 s | 40.86 s |
| 2 | `7` (this branch on 7-bank DRAM) | 0.489 s | 40.89 s | 41.46 s |
| 3 | `None` (full-T, same as `main`) | 0.303 s | 41.10 s | 41.49 s |
| 4 | `7` (this branch on 7-bank DRAM) | 0.497 s | 41.22 s | 41.79 s |
| **Mean, full-T** | `None` | **0.299 s** | 40.79 s | 41.17 s |
| **Mean, chunked** | `7` | **0.493 s** | 41.06 s | 41.62 s |
| **Cost of chunking** | | **+0.194 s (+65%)** | +0.27 s (within run-to-run noise) | +0.45 s (+1.1%) |

The cost of chunking is about the same in seconds at both power levels (+0.18 s at 75 W, +0.19 s at 130 W). Full-T gets faster with the higher limit (0.346 → 0.299 s), but the chunking cost doesn't, so it is probably fixed per-chunk host and dispatch overhead rather than device compute (not profiled).

Chunking adds about 0.18 s (+52%) to the VAE and nothing to denoise. The work is about the same (the same causal convolutions, run on 21 latent frames in pieces of 1, 7, 7 and 6 with a cache passed between them), so the extra time is probably per-chunk op and dispatch overhead plus the `concat` that joins the chunks. The branch chunks at every resolution on 7-bank DRAM. Chunking only at 720p, where full-T runs out of memory, would get the 480p time back.

## Notes

- The perf test asserts on its targets after printing the results table. For these runs the assert was changed locally to a warning so that all 8 configs would report (not committed).
- First run on a machine: converting each transformer's weights takes about 16 min. Use a pytest timeout above the default 1800 s or 2800 s for that run.
- In run 39523036841, the job's `tdp 0` helper (meant to print the host's default limit before the run and restore it after) printed nothing. The bare `python3` it calls probably can't import `ttnn` outside the job's `run` wrapper, and the error was filtered out. So the run neither read `d07u02`'s default nor confirmed a restore. That host may stay at 130 W until its next reset. Its unforced speed suggests that is its default anyway.
- The TDP limit outlives the process. Run `tt-smi -glx_reset` afterwards to return the board to 75 W.

## Reproduce

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD NO_PROMPT=1 HF_HOME=/mnt/models/huggingface
export TT_DIT_CACHE_DIR=$HOME/.cache/tt-dit TT_MESH_ID=0 TT_MESH_HOST_RANK=0 TT_METAL_SHM_TRACKING_DISABLED=1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
tt-smi -glx_reset && python -I handoff/wan-bin6/check_links.py   # no output = all links trained
# optional: export TT_METAL_TDP_LIMIT_WATTS=130   (0 returns to the 75 W bh-mod override)
pytest --timeout 5000 models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "ring_bh_4x8_sp1tp0"
```
