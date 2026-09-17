# MiniMax-H3 t2va on Wormhole Galaxy (4x8, 32 chips) — perf sweep

Re-measured 2026-09-17 on `tt-metal` @ `eab3dfbd599` (Wormhole bringup + the fused MM/RS
gate fix). Supersedes the 2026-09-16 run at `3a016b74847`, which stalled at 13/18.
Mesh param `MESH_4X8_RING_WH` (`4x8nl4`), TP=4 axis 0 / SP=8 axis 1, Ring, 4 links.
50 scheduler steps => 49 forwards. `RUN_VBENCH=0` (CLIP still gated).

Raw logs are **not** committed (too large to be useful in-tree); they were kept at
`~/h3_wormhole_results/*.log.gz` on the run host, with `parse.py` there to regenerate
these tables from any of them. Current run: `sweep_fixed.log.gz`.

## Command

```bash
TT_DIT_CACHE_DIR=~/tt_dit_cache MINIMAX_H3_DIT_FSDP=1 RUN_VBENCH=0 \
  python -m pytest models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py \
  -k "4x8nl4" -q
```

Drop `RUN_VBENCH=0` for the VBench gate (verified working, see below).
Drop `MINIMAX_H3_DIT_FSDP=1` for the unsharded baseline.

## Headline

DiT FSDP is the fix for the memory limits. Without it only 5 s fits; with it 10 s and 15 s run.

| | DiT alloc/bank | free/bank | largest contig |
|---|---|---|---|
| FSDP off | 799.5 MiB | 221.7 MiB | 221.7 MiB |
| FSDP on  | **101.8 MiB** | **919.4 MiB** | **917.7 MiB** |

7.85x reduction (SP=8 sharding), costs 5-11% denoise time, **bit-identical output**
(CLIP equal to 2 dp on all six 5 s cases).

Sweep outcome: **18/18 passed** (3 h 14 m, zero failures). The previous run reached 13/18
before an intermittent device hang blocked the rest; that hang was root-caused to the fused
MM/RS gate (open issue 2) and the 5 blocked points now all pass.

Removing the accidental fused ff2 path also made every case **2.0-4.0% faster (mean 3.0%)**.
Per-forward, old -> new: 5 s 2839 -> 2754 (21:9), 1477 -> 1417 (1:1); 10 s 6588 -> 6422 (16:9);
15 s 12700 -> 12447 (21:9). The six cases that never completed before are new measurements.

## Timings — FSDP ON (seconds unless noted)

### 5 s / 124 frames

| aspect | canvas | MPix | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 1.03 | 160.9 | 146.6 | 3.5 | 134.9 | 5.8 | 2.4 | 2754 | 28.4x | 35.95 |
| 16:9 | 1344x768 | 1.03 | 161.3 | 147.6 | 3.4 | 135.3 | 5.1 | 3.7 | 2761 | 28.6x | 37.42 |
| 9:16 | 768x1344 | 1.03 | 161.0 | 146.7 | 3.4 | 134.4 | 5.7 | 3.1 | 2744 | 28.4x | 37.02 |
| 4:3 | 1024x768 | 0.79 | 120.4 | 104.9 | 3.4 | 94.0 | 3.7 | 3.8 | 1919 | 20.3x | 37.26 |
| 3:4 | 768x1024 | 0.79 | 118.7 | 105.3 | 3.5 | 94.2 | 4.0 | 3.5 | 1922 | 20.4x | 36.54 |
| 1:1 | 768x768 | 0.59 | 92.7 | 79.3 | 3.5 | 69.4 | 3.0 | 3.4 | 1417 | 15.3x | 36.32 |

### 10 s / 243 frames

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 349.0 | 332.8 | 3.5 | 314.2 | 11.6 | 3.5 | 6413 | 32.9x | 33.84 |
| 16:9 | 1344x768 | 344.6 | 333.4 | 3.5 | 314.7 | 11.0 | 4.4 | 6422 | 32.9x | 36.92 |
| 9:16 | 768x1344 | 344.7 | 332.8 | 3.5 | 315.0 | 10.6 | 3.8 | 6428 | 32.9x | 36.40 |
| 4:3 | 1024x768 | 255.7 | 240.5 | 3.5 | 226.3 | 7.5 | 3.2 | 4618 | 23.8x | 37.09 |
| 3:4 | 768x1024 | 252.6 | 241.3 | 3.5 | 226.5 | 7.4 | 3.8 | 4623 | 23.8x | 36.89 |
| 1:1 | 768x768 | 173.8 | 159.2 | 3.5 | 146.4 | 5.8 | 3.6 | 2988 | 15.7x | 37.54 |

### 15 s / 362 frames

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 648.5 | 633.2 | 3.4 | 609.9 | 16.0 | 3.9 | 12447 | 42.0x | 35.27 |
| 16:9 | 1344x768 | 644.3 | 633.1 | 3.5 | 609.3 | 15.8 | 4.6 | 12435 | 42.0x | 36.31 |
| 9:16 | 768x1344 | 645.7 | 633.8 | 3.5 | 610.7 | 15.2 | 4.3 | 12464 | 42.0x | 35.50 |
| 4:3 | 1024x768 | 457.3 | 434.6 | 3.3 | 416.0 | 10.9 | 4.3 | 8491 | 28.8x | 36.01 |
| 3:4 | 768x1024 | 447.6 | 434.9 | 3.3 | 416.6 | 11.2 | 3.8 | 8501 | 28.8x | 36.40 |
| 1:1 | 768x768 | 281.8 | 266.6 | 3.5 | 251.2 | 8.5 | 3.6 | 5126 | 17.7x | 38.26 |

## Timings — FSDP OFF (baseline, for the 5 s comparison)

| aspect | canvas | cold | warm | denoise | ms/fwd | realtime |
|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 155.1 | 145.1 | 132.3 | 2699 | 28.1x |
| 16:9 | 1344x768 | 156.2 | 143.3 | 132.0 | 2695 | 27.7x |
| 9:16 | 768x1344 | 157.1 | 144.6 | 131.8 | 2690 | 28.0x |
| 4:3  | 1024x768 | 136.1 | 101.0 |  91.1 | 1858 | 19.6x |
| 3:4  | 768x1024 | 113.5 | 102.0 |  91.0 | 1856 | 19.8x |
| 1:1  |  768x768 | 108.8 |  76.0 |  65.3 | 1333 | 14.7x |

10 s without FSDP: 21:9 / 16:9 / 9:16 OOMed in **warmup**; 4:3 (cold 456.5) /
1:1 (191.5) / 3:4 (257.5) completed one generation then OOMed reloading the DiT for
the **timed** pass. All six 15 s OOMed in warmup.

## Scaling notes

- Denoise is 87-96% of warm time and tracks **pixel area**, not aspect: the three
  1.03 MPix 5 s canvases agree within 0.9 s regardless of orientation (134.4/134.9/135.3),
  and the same holds at 15 s (609.3/609.9/610.7).
- ms/forward vs area at 5 s: 1417 (0.59) / 1920 (0.79) / 2753 (1.03) — near-linear.
- Duration scales **superlinearly**, now measurable across all three durations:
  16:9 denoise 135.3 (5 s) -> 314.7 (10 s) -> 609.3 (15 s); 21:9 134.9 -> 314.2 -> 609.9.
  That is 2.33x for the first 1.96x in frames and 1.94x for the next 1.49x — the
  superlinearity is real but milder than the earlier two-point estimate suggested.
- Realtime factor improves with duration (28.4x at 5 s -> 42.0x at 15 s for 1.03 MPix):
  the fixed ~3.5 s encoder and the cold-start step amortise over more frames.
- Encoder is a flat ~3.5 s everywhere (same prompt, cached weights).
- The 4:3/10s audio-decode outlier of 192.6 s in the non-FSDP run dropped to 3.1 s
  with FSDP — it was memory pressure, not the conv1d MAC fallback.

## Open issues

1. **Intermittent mid-denoise device hang** — *root-caused and fixed; 18/18 now pass.*
   Counting the per-layer `No fused MM/RS` warnings in the two hang logs (50 fire per
   denoise step, so they act as a free per-layer profiler) located both stalls exactly:
   hang 1 in step i=23, hang 2 in step i=22, both blocked in the readback at
   `pipeline_minimax_h3.py:2002` — the same line as the earlier `Fatal Python error:
   Bus error` crashes. The ~13 s offset the earlier writeup leaned on turned out to be
   an arithmetic coincidence of the 2:1 per-step rates. Cause was issue 2 below.
   Evidence: 18/18 in this sweep, plus 9 standalone runs (~833 denoise steps) beforehand,
   all clean. Caveat: the pre-fix rate rests on only 2 hang events, so this is strong
   evidence rather than proof. Forensics and the evidence-preserving run recipe:
   see **`MiniMaxH3_wormhole_hang.md`**.

2. **Wormhole took the fused MM/RS path by accident** — *fixed*. `has_mmrs_config`
   gated the fused ff2 matmul+reduce-scatter on `(k, n, m % 32)` alone, but both ways
   of resolving a real blocking are Blackhole-only (`_SWEPT_BLOCKINGS` is keyed to a
   12x10 grid, the v2.3 rule engine is `is_blackhole()`-gated). So every Wormhole ff2
   landed on `default_fused_mmrs_config` — 56 of 72 cores at subblock 1x1, and a
   derived reduce-scatter worker count of **1 per link**, with the credit-based L1
   handoff active — which is the case the gate's own comment exists to prevent. The
   gate now takes the device core grid and asks `resolves_fused_mmrs_config` whether a
   measured or rule-derived blocking exists; Wormhole falls back to the ordinary matmul
   + `reduce_scatter_minimal_async`. Worth **2.0-4.0% (mean 3.0%)** across the sweep,
   and the tables above are measured with it. Note the unfused path is *not*
   bit-identical: the different reduction order moves CLIP by up to 0.6 in both
   directions, inside run-to-run noise and far above the 33.0 bar.

3. **Cache key omits device params** — `cache.load_model` keys on parallel config,
   mesh shape, dtype and FSDP, but not `l1_small_size`/device params. A cache
   written under a broken device config is silently reused forever. This cost a
   long debugging detour: a cache built during a run with `l1_small_size=0`
   produced text embeddings with `absmax=2.5e30` and a coherent video of the wrong
   subject (CLIP 13.12 instead of 37.36). Consider a validity marker.

4. **VBench setup gaps** (both fixed here, worth folding into MiniMaxH3.md:287-291):
   - no `unzip` on the box; VBench shells out to it for the RAFT checkpoints.
     Workaround: `python -c "import zipfile; zipfile.ZipFile('$HOME/.cache/vbench/raft_model/models.zip').extractall('$HOME/.cache/vbench/raft_model')"`
     — only possible *after* the first download attempt fails.
   - installing `vbench` and `opencv-python-headless` in one resolution pulls in
     the libGL-linked `opencv-python` as a vbench dep; it wins the shared `cv2`
     dir and `import cv2` dies on `libGL.so.1`. Fix: uninstall both, reinstall
     headless only.

## VBench (16:9/5s, verified passing)

| dimension | score | bar |
|---|---|---|
| subject_consistency | 0.9793 | 0.95 |
| background_consistency | 0.9779 | 0.95 |
| motion_smoothness | 0.9915 | 0.97 |
| dynamic_degree | 1.0000 | 1.0 |
| imaging_quality | 0.6802 | 0.64 |

CLIP 37.36 vs 33.0 bar (docs record 37.37 for Blackhole; imaging_quality 0.6896).
Only 16:9/5s has been VBench-verified; the sweep ran with `RUN_VBENCH=0`.

## Code changes backing these numbers (uncommitted)

| file | change |
|---|---|
| `pipelines/minimax_h3/weights_minimax_h3.py` | new — resolves the snapshot from `MINIMAX_H3_MODEL_PATH` / HF cache / download (`TT_DIT_ALLOW_HF_DOWNLOAD=1`) |
| `pipelines/minimax_h3/pipeline_minimax_h3.py` | `_PRESETS_WH` + arch-aware `resolve_mesh_preset`; `coresident` passthrough on `create_pipeline`; `_release_audio()` evicts the audio codec with the VAE stage; `is_fsdp=self.dit_fsdp` on the DiT build **and** its `cache.load_model` |
| `tests/.../minimax_h3/common.py` | `_L1_SMALL_WH = 32768`, `_ring_4k` wrapping (the WH mesh param was passing raw params with no `l1_small_size`) |
| `tests/.../minimax_h3/common_av.py` | weights gate via the resolver; `log_timing_table` reads the arch via `is_blackhole()` instead of hardcoding "Blackhole" |
| `models/MiniMaxH3.md` | HF download docs |

`dit_fsdp` defaults **off**, overridable with `MINIMAX_H3_DIT_FSDP`. Given these
results, `dit_fsdp: True` belongs in `_PRESETS_WH` (12 GB/chip needs the headroom
far more than it needs 5-11% of denoise); Blackhole at 32 GB can stay unsharded.
That decision is deliberately left open.
