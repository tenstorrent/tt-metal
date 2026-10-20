# Wan2.2 perf on a Bin6 BH Galaxy: 4x8 ring

Branch: `jameslee/test-pr60190-dram-endpoints` @ `adfc799f3a1` (PR #60190 DRAM endpoints + DRAM-harvest VAE chunking).
Machine: `bh-glx-120-b10u20` (single BH Galaxy, 32 Blackhole chips, firmware bundle 19.13.2).
Run 2026-10-20 by a Claude Code session for jameslee@tenstorrent.com.

## TL;DR

- **The 4x8 ring runs on this machine.** Unlike `b10u08` (see `README.md`), `b10u20` is cabled as a full 4x8 torus.
- **All 8 runs passed, none hung:** T2V and I2V at 480p and 720p, each at the default power limit and at `TT_METAL_TDP_LIMIT_WATTS=130`. That includes 720p with the chunked VAE, which hung 17 of 17 times on `b10u08` without the 130 W setting.
- **Only VAE decode misses its CI target, by 17–29%, and the chunking is why.** Denoise and total are 24–26% under target at the default limit. On 7-bank DRAM the branch decodes in chunks (`vae_t_chunk_size=7`). In an A/B on b10u20 at 480p, chunking costs about 52% in VAE time (0.52 s against 0.35 s full-T), or 0.18 s per video, about 0.3% of the total. Full-T meets the 0.42 s target on this machine.
- **Against CI on 8-bank BH Galaxies, Bin6 costs little or nothing in denoise.** b10u20 at 75 W matches the slower CI hosts (`d04*`, about 56.5 s at 480p) to within 1–2%, and at 130 W it comes within 4–5% of the faster ones (`d06*`/`d07*`, about 39 s). CI misses the VAE target too, by more (0.70–0.76 s at 480p, full-T), and that is why the nightly job is red. CI's VAE is about 2x slower than full-T on b10u20 for reasons not yet known (see the comparison below).
- **This branch on an 8-bank CI Galaxy matches `main` on the same host within ±0.5%,** so PR #60190 costs nothing there. The same CI run decodes the VAE full-T in 0.745 s, against 0.35 s for the same code on b10u20, so CI's slow VAE comes from the CI environment.
- **130 W is not a cap here: the board default is 75 W.** With every chip at 75 W out of the box (read back with `TT_METAL_TDP_LIMIT_WATTS=0`), setting 130 W *raises* the limit, and every config runs 17–26% faster. This probably reverses the README's finding 5 for `b10u08`: raising the limit, not capping it, is what stopped the hang.

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
- **Power.** The default firmware TDP limit is 75 W on all 32 chips.

## Results

`models/tt_dit/tests/models/wan2_2/test_performance_wan.py::test_pipeline_performance[blackhole-<t2v|i2v>-resolution_<480p|720p>-ring_bh_4x8_sp1tp0]`: SP=8 on axis 1, TP=4 on axis 0, 2 links, `is_fsdp=False`, 40 steps, 81 frames, one timed run after a 2-step warmup. Environment as in the BH single-Galaxy CI job (`tests/pipeline_reorg/models_e2e_tests.yaml`). Seconds:

| Config | TDP | Text enc | Image enc | Denoise | VAE decode | Total |
|---|---|---|---|---|---|---|
| T2V 480p | 75 W (default) | 0.061 | – | 55.75 | 0.523 | 56.34 |
| T2V 480p | 130 W | 0.069 | – | 40.99 | 0.492 | 41.56 |
| T2V 720p | 75 W (default) | 0.061 | – | 154.97 | 1.096 | 156.14 |
| T2V 720p | 130 W | 0.062 | – | 128.41 | 1.006 | 129.49 |
| I2V 480p | 75 W (default) | 0.067 | 0.921 | 55.97 | 0.514 | 57.48 |
| I2V 480p | 130 W | 0.067 | 0.786 | 41.40 | 0.489 | 42.75 |
| I2V 720p | 75 W (default) | 0.061 | 1.068 | 154.78 | 1.079 | 157.00 |
| I2V 720p | 130 W | 0.062 | 0.954 | 128.63 | 1.009 | 130.66 |

Per denoise step: 1.39 s (480p) and 3.87 s (720p) at 75 W; 1.02 s and 3.21 s at 130 W.

**Against the CI targets** (`t2v_metrics()` in the test, BH 4x8 ring; I2V uses the same targets):

| Res | Target denoise / VAE / total | 75 W | 130 W |
|---|---|---|---|
| 480p | 75.0 / 0.42 / 76.0 | 55.75 / **0.52** / 56.34 | 40.99 / **0.49** / 41.56 |
| 720p | 205.0 / 0.85 / 206.0 | 154.97 / **1.10** / 156.14 | 128.41 / **1.01** / 129.49 |

Every frame range is 0–255 (not the flat grey of `main` on this hardware). These perf runs do not score quality (no CLIP / VBench), and frames were not inspected.

## Comparison with CI on non-Bin6 BH Galaxies

Source: the nightly `TT-DiT Wan2.2-T2V-A14B single galaxy e2e tests [bh_sc1]` job (`models-t1-e2e-tests.yaml` on `main`), which runs this same `ring_bh_4x8_sp1tp0` T2V test with the same environment. Numbers come from the job logs. The CI hosts have 8 DRAM banks: their 720p runs decode the full 21 latent frames in one pass, which runs out of memory on 7 banks (#59504), and UMD logs no "inconsistent DRAM harvesting" warning.

| Nightly (main SHA) | Host | 480p denoise / VAE / total | 720p denoise / VAE / total |
|---|---|---|---|
| 10-13 (`3cdbe9ca17`) | bh-glx-120-d04u20 | 56.65 / 0.736 / 57.47 | 157.38 / 1.030 / 158.51 |
| 10-14 (`3de19707aa`) | bh-glx-120-d04u14 | 56.28 / 0.699 / 57.07 | 156.15 / 1.107 / 157.35 |
| 10-15 (`d1f9992507`) | bh-glx-120-d07u02 | 39.59 / 0.713 / 40.39 | 124.47 / 1.028 / 125.59 |
| 10-16 (`426b1f10d4`) | bh-glx-120-d06u08 | 38.79 / 0.739 / 39.63 | 122.47 / 0.977 / 123.55 |
| 10-17 (`628dbed7cc`) | bh-glx-120-d07u08 | 39.13 / 0.747 / 39.96 | 123.80 / 1.031 / 124.93 |
| 10-18 (`9f98dee9c5`) | bh-glx-120-d07u08 | 39.12 / 0.736 / 39.94 | 123.96 / 1.024 / 125.07 |
| 10-20 (`909748641c`) | bh-glx-120-d04u08 | 56.34 / 0.758 / 57.20 | – (hung: device timeout in fetch queue wait) |
| **this branch on CI** ([run 39492727180](https://github.com/tenstorrent/tt-metal/actions/runs/39492727180), 10-20, `6482b03632f`) | bh-glx-120-d06u08 | 38.99 / 0.745 / 39.82 | 122.36 / 1.024 / 123.49 |
| **this branch, b10u20, 75 W** | bh-glx-120-b10u20 | 55.75 / 0.523 / 56.34 | 154.97 / 1.096 / 156.14 |
| **this branch, b10u20, 130 W** | bh-glx-120-b10u20 | 40.99 / 0.492 / 41.56 | 128.41 / 1.006 / 129.49 |

(10-19 got no perf numbers.)

**I2V.** The nightly I2V jobs never reached a timed run: the CI cache starts empty, and converting the weights (17–20 min) used up the perf test's 1200 s timeout every night. The I2V numbers below come from a one-off dispatch of the same bh_sc1 job ([run 39489301673](https://github.com/tenstorrent/tt-metal/actions/runs/39489301673), branch `jameslee/wan-i2v-bh-sc1-perf-timeout` = `main` @ `f56b5484466` plus a 3600 s timeout and 480p added). Both tests completed; the job failed only on the VAE target.

| Machine | Res | Text enc | Image enc | Denoise | VAE decode (target) | Total |
|---|---|---|---|---|---|---|
| CI bh-glx-120-d04u20 (`main`, 8 banks, full-T VAE) | 480p | 0.081 | 1.150 | 57.06 | 0.750 (0.42) | 59.05 |
| b10u20, this branch, 75 W (7 banks, chunked VAE) | 480p | 0.067 | 0.921 | 55.97 | 0.514 | 57.48 |
| b10u20, this branch, 130 W | 480p | 0.067 | 0.786 | 41.40 | 0.489 | 42.75 |
| CI bh-glx-120-d04u20 (`main`, 8 banks, full-T VAE) | 720p | 0.089 | 1.329 | 157.47 | 1.101 (0.85) | 160.00 |
| b10u20, this branch, 75 W (7 banks, chunked VAE) | 720p | 0.061 | 1.068 | 154.78 | 1.079 | 157.00 |
| b10u20, this branch, 130 W | 720p | 0.062 | 0.954 | 128.63 | 1.009 | 130.66 |

I2V tells the same story as T2V. Against the `d04` host, b10u20 at 75 W denoises 1.7–1.9% *faster* (55.97 against 57.06 s, 154.78 against 157.47 s). Every stage that involves host work is slower on CI: text encoding 0.08 against 0.06 s, image encoding 1.15 against 0.92 s, and the VAE (0.75 s full-T against 0.35 s full-T on b10u20 from the A/B). That fits the guess that CI's slow VAE comes from the host side, not the device.

What this shows:

1. **The CI hosts split into two speed groups, and b10u20 lands on one or the other depending on the power limit.** `d04*` hosts denoise 480p in about 56.5 s and 720p in about 157 s, the same as b10u20 at its 75 W default (55.75 / 154.97). `d06*` / `d07*` hosts take about 39 s and 123.5 s, close to b10u20 at 130 W (40.99 / 128.41, 4–5% slower). The CI logs don't show TDP, so it is a guess that the fast hosts simply run with a higher power limit, but the match is close. Worth checking with `TT_METAL_TDP_LIMIT_WATTS=0` on a `d04` and a `d07` host.
2. **PR #60190 changes nothing on 8-bank hardware.** This branch on CI landed on `d06u08`, the same host as the 10-16 `main` nightly, and matches it within ±0.5% (denoise 38.99 against 38.79 s at 480p, 122.36 against 122.47 s at 720p).
3. **Bin6 plus PR #60190 costs little or nothing in denoise, if the CI hosts run at the same power.** At the same power level b10u20 is within 1–2% of the `d04` group, and within 4–5% of the `d06`/`d07` group when running at 130 W.
4. **CI's VAE decode is about 2x slower than b10u20's, and the cause is the environment, not the code.** This branch decodes full-T on CI's 8-bank host in 0.745 s; the same code with the same full-T setting takes 0.35 s on b10u20. CI misses the target too: 0.70–0.76 s at 480p against 0.42, and that assert is why every one of these nightlies is red. b10u20 with the same full-T decode takes 0.35 s (A/B below). So the target is fine; the CI runs are slow. Code age is ruled out: this branch's base (`7086ed4c7a0`, 10-14) is newer than the 10-12/10-13 CI runs that already showed 0.70–0.74 s. One guess, unverified: the `vae` section includes reading 81 frames back from 32 chips and converting them in numpy, so host speed counts, and CI runs inside a container under `mpirun`. At 720p, b10u20's chunked decode (1.01–1.10 s) matches CI's full-T (0.98–1.11 s). Full-T can't run at 720p on 7 banks (#59504), so there is no like-for-like 720p number.

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
- The TDP limit outlives the process. Run `tt-smi -glx_reset` afterwards to return the board to 75 W.

## Reproduce

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD NO_PROMPT=1 HF_HOME=/mnt/models/huggingface
export TT_DIT_CACHE_DIR=$HOME/.cache/tt-dit TT_MESH_ID=0 TT_MESH_HOST_RANK=0 TT_METAL_SHM_TRACKING_DISABLED=1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
tt-smi -glx_reset && python -I handoff/wan-bin6/check_links.py   # no output = all links trained
# optional: export TT_METAL_TDP_LIMIT_WATTS=130   (0 restores the 75 W default)
pytest --timeout 5000 models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "ring_bh_4x8_sp1tp0"
```
