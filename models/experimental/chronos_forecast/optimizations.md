# Chronos-2 on Blackhole: optimization log

This log records every optimization in the 60-commit Chronos-2 bring-up branch (`e000cf7df6f` to `c56554d92bb`). For each one it gives what the change does, which hardware limit it works around, where the code lives, and how much single-chip time it saved. Code links are relative to this directory and point at the current tree (`27bfe6ac45b` on `avan/chronos-forecast`). The run-path files are identical at `c56554d92bb`, which exists only on `avan/chronos-deadcode-trial`.

## Summary
The benchmark is the paper shape on one Blackhole p150a:

- **Batch and horizon:** 1024 series, 2048-step context, 64-step forecast, this is the standard batch/work per forward pass
- **Tokens:** 128 context patches, 4 future patches and 1 REG token give 133 tokens per series, padded to 160 (5 tiles)
- **Model:** 12 encoder layers, d_model 768, 12 heads of 64, d_ff 3072, 21 quantiles.

| State | Single-chip time for 1024 series | Speedup vs first run |
|---|---|---|
| First paper-shape run, eager `tt.forward` (`da30c104777`) | 6289 ms | 1× |
| First trace baseline, default precision, DRAM (`27c2acd7d78`) | 2659 ms replay | 2.4× |
| A10G reference (300 series/s, [tests/perf/test_paper_forward.py](tests/perf/test_paper_forward.py#L25-L26)) | 3413 ms | 1.8× |
| Current, `performance_l1`, trace replay | **153 ms** (6,690 series/s) | **41×** |
| Current, `performance_l1`, serial end-to-end (host prepare, upload, replay, readback) | 199 ms | 32× |

## Test Measurements/configs

- **Test:** [tests/perf/test_paper_forward_trace.py](tests/perf/test_paper_forward_trace.py), paper shape, unique groups (every series its own group) unless stated.
  - **Replay:** the median of 20 trace replays.
  - **Serial end-to-end:** host prepare, upload, replay and readback run back to back.
  - **Streamed per batch:** `TtChronosTraceRunner.stream()` over 20 batches.
  - **Stages:** medians of 10 serial runs.
- **Configs:**
  - `default_dram`: bf16, HiFi2, DRAM intermediates.
  - `default_l1`: default precision with L1 chunks.
  - `performance_l1`: `TtChronosPrecision.performance()` with L1 chunks.

Reproduce a single-chip number:

```bash
pytest models/experimental/chronos_forecast/tests/perf/test_paper_forward_trace.py
```

## Ranked optimizations (single chip, paper shape)

Optimzations are ranked by how much they improve set time by, this serves to be a guide for future optimzations,
also note that the performance gains taper off as the model becomes more optimized

| Rank | Optimization | Commit | Config | Before → after (ms) | Saved (ms) | Saved | Hardware limit addressed | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | [Device-resident forward and trace, unique-group shortcut](#1-device-resident-forward-and-trace-a02811a1673) | `a02811a1673` | default DRAM; eager wall vs trace end-to-end | 6289 → 2767 | 3522 | 56% | PCIe round trips and host work | measured |
| 2 | [Fold the batch into matmul M](#2-fold-the-batch-into-matmul-m-599000d80f5) | `599000d80f5` | default DRAM | 2659.3 → 922.0 | 1737.3 | 65.3% | Tensix grid utilization | measured |
| 3 | [Block-diagonal group attention](#3-block-diagonal-group-attention-a2ed3b2d150) | `a2ed3b2d150` | groups of 4, default DRAM | 965 → 748 | 217 | 22.5% | Attention cost quadratic in batch | recorded |
| 4 | [L1-resident series chunks](#4-l1-resident-series-chunks-d5ba11da637) | `d5ba11da637` | default, DRAM → L1 | 549 → 343 | 206 | 37.5% | DRAM bandwidth, bounded by L1 size | recorded |
| 5 | [Diagonal group attention](#5-diagonal-group-attention-42932ff3743) | `42932ff3743` | default DRAM | 922.0 → 739.0 | 183.0 | 19.8% | DRAM traffic from permutes; matmul count | measured |
| 6 | [Reduced-precision preset](#6-reduced-precision-preset-b6813f71b05) | `b6813f71b05` | DRAM, default → performance | 548 → 399 | 149 | 27% | DRAM and NoC bytes; matrix-engine passes | recorded |
| 7 | [Fused RoPE with a shared cos/sin cache](#7-fused-rope-with-a-batch-shared-cache-ec79b81bd6e) | `ec79b81bd6e` | default DRAM | 739.0 → 617.0 | 122.0 | 16.5% | Op dispatch count; DRAM traffic | measured |
| 8 | [One SDPA chunk per sequence](#8-one-sdpa-chunk-per-sequence-edd357c6f60) | `edd357c6f60` | default DRAM | 617.0 → 548.4 | 68.6 | 11.1% | Per-chunk overhead | measured |
| 9 | [Model-local RoPE and bank-local residual add](#9-model-local-rope-and-bank-local-residual-add-df78db0dabd) | `df78db0dabd` | performance L1 | 248.2 → 209.0 | 39.2 | 15.8% | NoC traffic | measured |
| 10 | [LoFi attention matmuls](#10-lofi-attention-matmuls-82af12bb7b3) | `82af12bb7b3` | performance L1 | 210.1 → 190.1 | 20.0 | 9.5% | Matrix-engine passes | recorded |
| 11 | [Embedding and head inside the L1 chunks](#11-embedding-and-head-inside-the-l1-chunks-ca73ac4f303) | `ca73ac4f303` | performance L1 | 181.0 → 163.2 | 17.8 | 9.8% | DRAM round trips; host tilize | recorded |
| 12 | [Sweep-tuned L1 matmul configs, RMSNorm gamma folded](#12-sweep-tuned-l1-matmul-configs-and-folded-rmsnorm-gamma-0b755e48966) | `0b755e48966` | performance L1 | 263.0 → 248.2 | 14.8 | 5.6% | Idle grid columns; DRAM gamma reads | measured |
| 13 | [Model-local RMSNorm with bf8 output](#13-model-local-rmsnorm-with-bf8-output-109ab6e059e) | `109ab6e059e` | performance L1 | 163.2 → 151.4 | 11.8 | 7.2% | NoC bytes | recorded |
| 14 | [Overlap host prepare with replay](#14-overlap-host-prepare-with-replay-fb2dda1082e) | `fb2dda1082e` | performance L1; streamed vs serial per batch | 322 → 311 | 11 | 3.4% | Serial host CPU time | recorded |
| 15 | [Fused QKV head split and RoPE](#15-fused-qkv-head-split-and-rope-41c6856ef36) | `41c6856ef36` | performance L1 | 190.1 → 181.0 | 9.1 | 4.8% | Kernel launches; L1 round trip | recorded |
| 16 | [Two command queues for input refresh](#16-two-command-queues-for-input-refresh-5ed70864de0) | `5ed70864de0` | default DRAM, end-to-end | 2767.4 → 2764.4 | ≈0 | 0.1% | none (noise) | measured |

Before → after figures were recorded at each commit, so the chain ends at 151.4 ms (rank 13). The current tree measures a 152–155 ms median replay on a warm chip, with 3–5% thermal variance; the summary uses 153 ms, matching the README.

Data parallel over a mesh ([`4abfa9b0395`](#data-parallel-scaling)) is left out of this table because it changes chip count, not single-chip time.
Note that dataparallel will get almost a linear scale till the bottleneck becomes the datamovement to the actual chip
