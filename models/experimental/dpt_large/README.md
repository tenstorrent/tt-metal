<!--
SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
-->

# DPT-Large (MiDaS 3.0) - Experimental

This implementation follows the standard model layout:

- `tt/` - TTNN + reference pipeline modules (backbone/neck/head, configs, weights)
- `demo/` - runnable scripts
- `tests/` - unit/e2e tests

The reference checkpoint is Hugging Face `Intel/dpt-large` (Apache-2.0 licensed):
https://huggingface.co/Intel/dpt-large

The model card notes the expected inference resolution is `384x384` (the original training uses random
`384` square crops).

## Demo

Canonical bounty commands (minimal surface):

1) Stage-2 strict throughput (`384x384`, `dp=2`, per-chip batch `1`):

```sh
python -m models.experimental.dpt_large.demo.runner \
  --images-dir /tmp/dpt_eval_imgs \
  --tt-run --device wormhole_n300 \
  --image-size 384 \
  --dp 2 --batch-size 2 \
  --tt-execution-mode trace_2cq --warmup 8 --repeat 30 \
  --dump-perf /tmp/dpt_runner_perf_stage2_strict.json
```

2) Stage-3 alternative throughput (`512x512`, `dp=2`, per-chip batch `2`):

```sh
python -m models.experimental.dpt_large.demo.runner \
  --images-dir /tmp/dpt_eval_imgs \
  --tt-run --device wormhole_n300 \
  --image-size 512 \
  --dp 2 --batch-size 4 \
  --tt-execution-mode trace_2cq --warmup 8 --repeat 30 \
  --dump-perf /tmp/dpt_runner_perf_stage3_512.json
```

3) PCC + throughput check (TT vs CPU reference):

```sh
python -m models.experimental.dpt_large.demo.eval_pcc \
  --images-dir /tmp/dpt_eval_imgs \
  --tt-run --device wormhole_n300 \
  --image-size 384 \
  --dp 2 --batch-size 2 \
  --tt-execution-mode trace_2cq \
  --dump-json /tmp/dpt_eval.json
```

Re-run the same command several times and summarize steady-state `fps` / `fps_device` from perf JSON outputs.
Stage-1 dataset evidence (KITTI/NYU) is attached to PR artifacts/comments rather than generated via in-repo helper scripts.

### Perf Metrics (Device vs End-to-End)

For `--tt-execution-mode trace` / `trace_2cq`, the perf JSON reports two FPS values:

- `fps` / `fps_e2e`: Headline end-to-end FPS including readback and CPU normalization.
- `fps_device`: Device-only FPS derived from `trace_wall_ms_mean` (when available).

The device-only timing is derived from `trace_wall_ms`, which is measured using a host-visible completion
barrier (`ttnn.synchronize_device`) so it is stable across runs and does not include host jitter from
`cpu().to_torch()` and normalization.

When `--dump-perf` is enabled, the runner enables strict mode and fails if any TTNN program-config fast path
throws and falls back silently. In perf runs, `program_config_fallback_total` must be `0`.

## Memory Layout & Sharding Strategy (Stage 2)

The encoder uses a conservative layout strategy to keep the strict traced path stable on Wormhole N300.

- **Wormhole N300 (Stage-2 strict baseline)**: keep encoder tokens **interleaved** and route some large
  intermediates through **DRAM**. This avoids known `layer_norm` static circular-buffer vs L1 allocation
  clashes during trace capture for DPT token shapes (e.g., padded seq `640`, dim `1024`).

The perf JSON exposes `attn_island_interleave_*` / `attn_island_reshard_*` counters for any SDPA-specific
interleave/reshard conversions. In strict Stage-2 perf runs these must remain `0`.

## Fused Ops Used

| Subsystem | Fused op | Where |
|---|---|---|
| Encoder MLP | FF1 matmul + GELU | `tt/tt_configs.py` (FF1 `fused_activation`) |
| Encoder Attention | QKV fused projection (single linear) | `tt/tt_modules.py` (`TTAttention` fused QKV weights) |

## Stage 3 (Tuning / Limitations)

Stage 3 work focuses on reducing attention overhead, minimizing TM/layout churn, and increasing core utilization.

Note: For dp-batched runs (per-chip batch size > 1), sharded LayerNorm can hit static circular-buffer vs L1-buffer
clashes under trace capture on N300. The LayerNorm implementation may temporarily interleave activations in DRAM
and reshard back for stability; this is surfaced via `ln_island_*` perf counters and remains forbidden for the
Stage-2 strict baseline (per-chip batch size == 1).

DP batching: in `dp=2` mode, the runner supports `batch_size % dp == 0` and feeds each chip a batched tensor
(per-chip batch size is `batch_size // dp`). In the sharded-token Stage-3 path (`--tt-shard-tokens`), the
encoder core grid and attention core grid scale with per-chip batch size to keep per-core attention score
shards within practical L1 budgets.

### 512x512 (Stage-3 Alternative)

On Wormhole N300 (dp=2, `trace_2cq`) the `512x512` baseline (SDPA/interleaved tokens) path runs stably without
CPU fallbacks or silent program-config fallback (`program_config_fallback_total == 0`).

Representative numbers (device-only vs end-to-end):

- `384x384` (dp=2, batch=8, `--tt-shard-tokens`): ~`30.6` FPS device-only, ~`29.5` FPS end-to-end.
- `512x512` (dp=2, batch=4): ~`17.3` FPS device-only, ~`16.8` FPS end-to-end.

Known limitation: `--tt-shard-tokens` at `512x512` is currently unstable due to sharded/interleaved
data-movement kernels hitting static circular-buffer vs L1-buffer clashes under trace capture.

If you see large variance over longer sustained runs, check device telemetry (temperature / AI clock) with:

```sh
tt-smi -l -s --snapshot_no_tty | head
```

On Wormhole N300 with `dp=2` (per-chip batch size `1`), TTNN sharded split-heads currently constrains
some sharded attention variants (it requires `batch == grid_y`). For the sharded-token path we therefore
keep QKV routed interleaved through DRAM and reshard Q/K/V for explicit attention, while keeping the
attention output projection sharded in L1 when stable. MLP intermediates are still routed through DRAM
to avoid static circular-buffer vs L1-buffer clashes under trace capture.

Run CPU-only (no TT device) to get a baseline:

```sh
python -m models.experimental.dpt_large.demo.eval_pcc \
  --images-dir path/to/images \
  --device cpu \
  --dump-json generated/dpt_large_cpu_eval.json
```

## Tests

CPU-only smoke test:

```sh
pytest -q models/experimental/dpt_large/tests/test_e2e_fallback.py
```

Hardware PCC test (TT vs CPU reference):

```sh
pytest -q models/experimental/dpt_large/tests/test_tt_pcc.py
```

Hardware perf smoke test (writes `generated/dpt_large_tt_perf_smoke.json`):

```sh
pytest -q models/experimental/dpt_large/tests/test_tt_perf.py -s
```
