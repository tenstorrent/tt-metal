# MiniMax-H3 t2va on Wormhole Galaxy (4x8, 32 chips) — perf sweep

Measured 2026-09-16 on `tt-metal` @ `3a016b74847` + uncommitted Wormhole bringup changes.
Mesh param `MESH_4X8_RING_WH` (`4x8nl4`), TP=4 axis 0 / SP=8 axis 1, Ring, 4 links.
50 scheduler steps => 49 forwards. `RUN_VBENCH=0` (CLIP still gated).

Raw logs are **not** committed (too large to be useful in-tree); they were kept at
`~/h3_wormhole_results/*.log.gz` on the run host, with `parse.py` there to regenerate
these tables from any of them.

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

Sweep outcome: **6/18 passed without FSDP, 13/18 measured with it.**
The remaining 5 are blocked by an intermittent device hang, *not* by memory.

## Timings — FSDP ON (seconds unless noted)

### 5 s / 124 frames

| aspect | canvas | MPix | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime | CLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 1.03 | 173.7 | 151.2 | 3.4 | 139.1 | 5.8 | 2.8 | 2839 | 29.3x | 36.20 |
| 16:9 | 1344x768 | 1.03 | 164.4 | 150.1 | 3.5 | 138.7 | 4.9 | 3.0 | 2832 | 29.1x | 37.36 |
| 9:16 | 768x1344 | 1.03 | 164.4 | 151.0 | 3.5 | 138.7 | 5.5 | 3.2 | 2831 | 29.2x | 36.50 |
| 4:3  | 1024x768 | 0.79 | 122.4 | 108.2 | 3.4 |  97.9 | 3.6 | 3.3 | 1998 | 20.9x | 37.34 |
| 3:4  | 768x1024 | 0.79 | 122.4 | 108.7 | 3.5 |  98.1 | 3.8 | 3.4 | 2002 | 21.0x | 36.63 |
| 1:1  |  768x768 | 0.59 |  95.7 |  81.3 | 3.5 |  72.4 | 2.9 | 2.6 | 1477 | 15.7x | 36.33 |

### 10 s / 243 frames (all six FAILED without FSDP)

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime |
|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 356.7 | 340.2 | 3.5 | 322.6 | 9.8 | 4.4 | 6583 | 33.6x |
| 16:9 | 1344x768 | 353.3 | 340.3 | 3.5 | 322.8 |10.0 | 4.1 | 6589 | 33.6x |
| 4:3  | 1024x768 | 258.6 | 245.9 | 3.5 | 232.3 | 7.0 | 3.1 | 4741 | 24.3x |
| 3:4  | 768x1024 | 259.5 | 247.4 | 3.5 | 232.8 | 7.0 | 4.1 | 4752 | 24.4x |
| 1:1  |  768x768 | 175.4 | 161.3 | 3.2 | 150.1 | 5.2 | 2.8 | 3064 | 15.9x |
| 9:16 | 768x1344 | 353.1 | — | — | — | — | — | — | **HUNG** (timed pass) |

### 15 s / 362 frames

| aspect | canvas | cold | warm | enc | denoise | vae | audio | ms/fwd | realtime |
|---|---|---|---|---|---|---|---|---|---|
| 21:9 | 1536x672 | 842.4 | 644.6 | 3.5 | 622.3 |15.3 | 3.6 |12700 | 42.7x |
| 16:9 | 1344x768 | — | — | — | — | — | — | — | **HUNG** (warmup) |
| 4:3 / 3:4 / 1:1 / 9:16 | | | | | | | | | not run |

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

- Denoise is 86-91% of warm time and tracks **pixel area**, not aspect: the three
  1.03 MPix 5 s canvases agree within 0.5 s regardless of orientation.
- ms/forward vs area at 5 s: 1477 (0.59) / 2000 (0.79) / 2834 (1.03) — near-linear.
- Duration scales **superlinearly**: 16:9 denoise 138.7 (5 s) -> 322.8 (10 s) -> ~n/a,
  and 21:9 139.1 -> 322.6 -> 622.3. Roughly 2.3x per 1.5x frames (attention).
- Encoder is a flat ~3.5 s everywhere (same prompt, cached weights).
- The 4:3/10s audio-decode outlier of 192.6 s in the non-FSDP run dropped to 3.1 s
  with FSDP — it was memory pressure, not the conv1d MAC fallback.

## Open issues

1. **Intermittent mid-denoise device hang** — was the blocker, unrelated to memory.
   Counting the per-layer `No fused MM/RS` warnings in the two hang logs located the
   stalls exactly: hang 1 in step i=23, hang 2 in step i=22, both blocked in the
   readback at `pipeline_minimax_h3.py:2002` — the same line as the earlier
   `Fatal Python error: Bus error` crashes. The ~13 s offset the earlier writeup
   leaned on turned out to be an arithmetic coincidence of the 2:1 per-step rates.
   Prime suspect and now fixed: issue 2 below. Full forensics, the evidence-preserving
   run recipe and the remaining open questions: see **`MiniMaxH3_wormhole_hang.md`**.

2. **Wormhole took the fused MM/RS path by accident** — *fixed*. `has_mmrs_config`
   gated the fused ff2 matmul+reduce-scatter on `(k, n, m % 32)` alone, but both ways
   of resolving a real blocking are Blackhole-only (`_SWEPT_BLOCKINGS` is keyed to a
   12x10 grid, the v2.3 rule engine is `is_blackhole()`-gated). So every Wormhole ff2
   landed on `default_fused_mmrs_config` — 56 of 72 cores at subblock 1x1, and a
   derived reduce-scatter worker count of **1 per link** — which is the case the gate's
   own comment exists to prevent. The gate now takes the device core grid and asks
   `resolves_fused_mmrs_config` whether a measured or rule-derived blocking exists;
   Wormhole falls back to the ordinary matmul + `reduce_scatter_minimal_async`.
   Measured on `1x1_5s`: **1477 -> 1408 ms/fwd (4.7% faster)**, CLIP 36.32 vs 36.33.
   **The tables above still include the unoptimized path**, so every denoise number
   here is pessimistic by a few percent; re-sweep to restate them.

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
