# HunyuanImage-3.0 Performance

Measured performance of the native-TTNN HunyuanImage-3.0 text-to-image render on a Tenstorrent Blackhole Galaxy. Numbers come from `tests/e2e/test_host_glue_stage3_perf.py` (traced wall-clock). Newer TT-Metal versions may shift these.

## Configuration

| Parameter | Value |
|---|---|
| Model | `tencent/HunyuanImage-3.0` (~80B total / ~13B active MoE) |
| Task | text-to-image (on-device head-glue render) |
| Device | Blackhole 6U Galaxy — `MeshShape(8,4)`, 32 chips, FABRIC_1D |
| Parallelism | TP=8 (attn/dense) + EP=32 (MoE: 64 experts, 2/chip across all 32) + shard-shared |
| Resolution | 1024×1024 |
| Denoise | 50 steps (FlowMatch / Euler), CFG=2 |
| Precision | bf8_b / bf4_b experts, bf16 VAE (host) |

## End-to-end performance (1024×1024, 50 steps)

| Path | ms/step | Denoise (s) | VAE decode (s) | E2E (s/image) |
|---|---|---|---|---|
| Hybrid (per-step host round-trip) | 5548 | 294.7 | 56.4 (fp32) | 351.4 |
| On-device head-glue — cold | 947 | 46.4 (+133.7 compile) | 36.4 (bf16) | 216.5 |
| On-device head-glue — **warm** | ~970 | ~48 | ~36 (bf16) | **~84** |

Throughput at warm: **~43 images/hr**.

## Optimization ladder (steady ms/step; hybrid path, 32 layers / 12-step / trace)

| Config (cumulative, all now default-on) | ms/step |
|---|---|
| baseline (EP off, CCL links=1) | 7770 |
| + EP=32 + shard-shared expert | 6368 (−18.0%) |
| + CCL_LINKS=2 | 6093 |
| + MM_FULLGRID | 5736 |
| + fused attention all-reduce | 5512 (−29.1% cumulative) |

The largest single win is separate from this ladder: the **on-device head-glue** port removed a ~4600 ms/step host round-trip, cutting the per-step from **5548 → 947 ms/step**. Full dated log: [`OPTIMIZATIONS_BY_DATE.md`](OPTIMIZATIONS_BY_DATE.md).

## Targets & references

| | s/image | Note |
|---|---|---|
| **Current (warm)** | **~84** | 32-chip Galaxy, resident server |
| Near-term target | ~40-50 | on-device VAE + traced render + CFG-parallel |
| Aggressive target | ~15-20 | dense-MoE roofline @ ~40% MFU |
| Floor (not reachable) | ~5-8 | needs sparse dispatch — DEAD on TT (~13-47× slower) |
| Ref: HunyuanImage-3.0 / 3×H100 | 137 | same model, GPU (soft provenance) |
| Ref: IGN port (PR #50968) | 263 | same model, BH 4-chip 2×2 SP×TP |
| Ref: LongCat-Image (TT) | 43.6 | different / smaller 6B dense model |

Machine-readable target: [`../tests/perf_targets/t2i_1024_50step.json`](../tests/perf_targets/t2i_1024_50step.json).

## Bottleneck (latest Tracy)

CCL-bound: ReduceScatter + AllGather ≈ **30% of device-kernel time** (ahead of matmul ~23%); ≈ **55% of wall-clock** once cross-chip sync is counted. Metric note: per-op device_ms does not track render time — measure traced wall-clock s/image.

## Run

```bash
# E2E on-device render latency (emits ONDEVICE_E2E_TOTAL_LATENCY_S)
HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2 ./python_env/bin/python -m pytest -o timeout=0 \
  models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_host_glue_stage3_perf.py -s
```

## 2026-08-08 — sequence-parallel win + sweeps

**WIN — SP-only sequence-parallelism** (EP=32 → EP=8 token-sharding), COMMITTED @ `1be484e7`, gated `HUNYUAN_SP` (default off, byte-identical): **623.3 ms/step steady vs 906 baseline = −31%**, PCC 0.99999, **~67 s/image warm** (50-step 1024²) vs ~81–84 s. Steady = mean of steps 2..N (the harness all-step mean is compile-diluted — ignore it).

Ladder (traced steady ms/step):

| config | ms/step |
|---|---|
| **SP-only** (token-sharding) | **623** |
| SP_FUSED-Linear | 850 |
| baseline | 906 |
| SP-RING (ring fabric + fused RS+MM) | 996 |

The fused/ring elaborations (H-shard + AG+MM + fused Ring RS+MM + distributed RMSNorm) are numerically CORRECT (PCC 0.99999) but REGRESS on this HW → kept as gated-off scaffolds (`HUNYUAN_SP_FUSED` / `HUNYUAN_SP_RING`). Added collective traffic + Ring latency > overlap; simple token-sharding wins.

- **Lever 1 (matmul block sweep) = wash** — see [`SWEEPS.md`](SWEEPS.md).
- **Next lever: on-device VAE decode** — host ~36 s = ~54% of the warm image → target ~10–17 s on mesh; multi-day port (Mochi scaffold + novel UpsampleDCAE + conv3d blocking-fix).
