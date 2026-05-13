TT_DS_CAPTURE_COMBINE_LAYERS=all TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden/ TT_DS_COMBINE_CAPTURE_DIR=/data/nmilicevic/combine_captures_1k TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528 python -m pytest -xvs models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py -k "pretrained and smoke and e256_device_fp32 and mesh-8x4 and 61_layers and longbook_qa_eng and 1024 and iter1 and balanced and right_pad"


# Combine op: Galaxy → LoudBox ethernet measurement workflow

This document describes the workflow added under `models/demos/deepseek_v3_d_p/`
to measure how ethernet bandwidth affects the MoE prefill **combine** op on
Tenstorrent Blackhole hardware. It captures combine inputs at one or more
MoE layers on a Galaxy 8×4 machine (32 chips), then replays each Galaxy
dispatch group as a standalone 8×1 run on a separate 8-chip machine (e.g.,
LoudBox) with different `num_links` configurations.

## 1. Motivation

The combine op is an all-to-all that sends each expert's output back to the
source token's chip. On Galaxy 8×4, four dispatch groups run combine
simultaneously and share physical fabric resources. The actual per-layer
combine wall-clock includes:

1. **Per-column ethernet/fabric time** for one dispatch group's 8-chip
   all-to-all writes (the inherent kernel cost).
2. **Inter-column fabric contention** when four columns push traffic through
   shared switches at once.
3. **Ethernet bandwidth** of the underlying interconnect.

Replaying one column's traffic on a standalone 8×1 mesh isolates (1) +
(3) without (2). Comparing across machines with different ethernet
configurations quantifies the bandwidth sensitivity of combine.

## 2. Big picture

```
┌───────────────────────────────────────────────────────────────────────┐
│ Galaxy 8×4 (32 chips)                                                 │
│                                                                       │
│   test_prefill_transformer.py  ── runs full 61-layer forward          │
│     │                                                                 │
│     │  at chosen MoE layers (TT_DS_CAPTURE_COMBINE_LAYERS=...),       │
│     │  TtCombineModule._capture_inputs() dumps routing tensors        │
│     ▼                                                                 │
│   ${TT_DS_COMBINE_CAPTURE_DIR}/                                       │
│     L<NN>/col0.pt, col1.pt, col2.pt, col3.pt                          │
│     (one .pt per Galaxy dispatch group)                               │
│                                                                       │
│   ${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv    │
│     (full Galaxy per-op device timings, including all 32 chips)       │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              │ rsync .pt files
                              ▼
┌───────────────────────────────────────────────────────────────────────┐
│ LoudBox 8×1 (8 chips)                                                 │
│                                                                       │
│   tests/perf/test_combine_replay.py                                   │
│     - discovers .pt files under TT_DS_COMBINE_CAPTURE_DIR             │
│     - parametrizes over (capture_file × num_links)                    │
│     - allocates dummy zero buffer on device (size from .pt config)    │
│     - pushes captured metadata + counts + offsets to chips            │
│     - runs TtCombineModule.forward() N times (warmup + timed)         │
│                                                                       │
│   ${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv    │
│     (LB per-op device timings, one entry per combine invocation)      │
└───────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Aggregate & compare                                                   │
│   tests/perf/analyze_combine_perf.py                                  │
│     - galaxy <csv>: per-layer max/median across 32 chips              │
│     - lb <csv>:     per-chip median across timed iters                │
│     - compare:      per-layer × num_links overhead ratio              │
└───────────────────────────────────────────────────────────────────────┘
```

## 3. Files touched / created

### Modified

`models/demos/deepseek_v3_d_p/tt/moe/tt_combine.py`
:  Added `_capture_inputs(...)` method on `TtCombineModule`. Called from
   `forward(...)` when `self.layer_idx ∈ _CAPTURE_LAYERS`.
   Two env-var-controlled paths:
   - Default: captures only `dispatched_metadata`, `expert_token_counts`,
     `expert_region_offsets`, and records `buffer_shape` in config. ~MB-sized
     output files; replay generates a zero buffer.
   - With `TT_DS_CAPTURE_COMBINE_FULL_BUFFER=1`: also captures the real
     `dispatched_buffer` (multi-GB output files). Only needed for PCC
     validation; perf measurement does not require this.

   The existing signpost/`_PROFILE_OPS` block is unchanged and orthogonal.

### New

`models/demos/deepseek_v3_d_p/tests/perf/test_combine_replay.py`
:  Pytest entry point that loads `.pt` files and replays them on an 8×1
   mesh. Parametrizes over `(capture_file × num_links ∈ {1, 2})` on
   `linear-8` topology. Configurable via:
   - `TT_DS_COMBINE_CAPTURE_DIR` — directory of `.pt` files
   - `TT_DS_REPLAY_WARMUP` (default 3) — warmup iters per test
   - `TT_DS_REPLAY_TIMED`  (default 10) — timed iters per test

   When a `.pt` lacks `dispatched_buffer`, the test synthesizes a zero
   buffer at the recorded shape using `ttnn.ReplicateTensorToMesh`
   (one small per-device tensor instead of the multi-GB per-column tensor),
   so host RAM stays bounded regardless of isl.

`models/demos/deepseek_v3_d_p/tests/perf/analyze_combine_perf.py`
:  CLI for aggregating CSVs.
   - `galaxy <csv>`: per-MoE-layer max/median/min combine duration across
     32 chips (layer index inferred from `GLOBAL CALL COUNT` per device).
   - `lb <csv>`: per-chip median across timed iters in one LB run.
   - `compare --galaxy <csv> --lb-glob "<glob>"`: per-layer × num_links
     comparison with `overhead_ratio = galaxy_max_chip / lb_max_col_chip`.

`models/demos/deepseek_v3_d_p/COMBINE_GLX_LB_MEASUREMENT.md`
:  This file.

## 4. Environment variables reference

### Capture (Galaxy side)

| Variable | Purpose | Default |
|---|---|---|
| `TT_DS_CAPTURE_COMBINE_LAYERS` | Comma-separated MoE layer indices to capture. Empty = no capture. | unset (no capture) |
| `TT_DS_COMBINE_CAPTURE_DIR` | Where to write `L<NN>/col<k>.pt` files. | `$TT_METAL_HOME/generated/combine_capture` |
| `TT_DS_CAPTURE_COMBINE_FULL_BUFFER` | `1` to also save real buffer values (multi-GB). | `0` (routing-only) |
| `TT_METAL_DEVICE_PROFILER` | `1` to collect device per-op timings into `profile_log_device.csv`. | unset |
| `TT_DS_PREFILL_TTNN_CACHE` | Writable dir for cached TTNN weight tensors. | `$model_path/tensor_cache_{arch}_{N}dev/` |
| `DEEPSEEK_V3_HF_MODEL` | Path to DeepSeek-R1-0528 safetensors. | repo defaults / HF download |

### Replay (LoudBox side)

| Variable | Purpose | Default |
|---|---|---|
| `TT_DS_COMBINE_CAPTURE_DIR` | Where to read captured `.pt` files from. | `$TT_METAL_HOME/generated/combine_capture` |
| `TT_DS_REPLAY_WARMUP` | Warmup iterations before timed runs. | `3` |
| `TT_DS_REPLAY_TIMED` | Timed iterations per replay. | `10` |
| `TT_METAL_DEVICE_PROFILER` | `1` to collect device per-op timings. | unset |

## 5. End-to-end workflow

### 5.1 Capture (Galaxy 8×4)

```bash
# Choose layers to capture (e.g., shallow / mid / deep MoE layers).
# Layers 0-2 are dense; 3-60 are MoE. Pick a small set for first runs.
export TT_DS_CAPTURE_COMBINE_LAYERS=3,30,60
export TT_DS_COMBINE_CAPTURE_DIR=/data/nmilicevic/combine_captures
export TT_METAL_DEVICE_PROFILER=1
export TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
export DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528

python -m pytest -xvs \
  models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py \
  -k "pretrained and smoke and e256_device_fp32 and mesh-8x4 and 61_layers and longbook_qa_eng and 1024 and iter1 and balanced and right_pad"

# Convert device timings to the standard ops CSV (no tracy host trace).
python /data/nmilicevic/tt-metal/tools/tracy/process_ops_logs.py \
  --device-only -n "galaxy_actual"
```

After this completes you have:

- `/data/nmilicevic/combine_captures/L03/col0.pt … col3.pt` (and similar for L30, L60)
- `generated/profiler/reports/<ts>/ops_perf_results_<ts>_galaxy_actual.csv`

### 5.2 Sanity-check captures

```bash
python -c "
import torch
b = torch.load('/data/nmilicevic/combine_captures/L03/col0.pt', weights_only=False)
print('keys:', list(b.keys()))                       # NO dispatched_buffer by default
print('config:', b['config'])
print('meta shape:', tuple(b['dispatched_metadata'].shape))
print('total tokens routed in this col:',
      int(b['expert_token_counts'].sum().item()))
print('per-chip token counts:',
      b['expert_token_counts'][0].sum(dim=-1).tolist())
"
```

At 1K isl expect: per-chip token sum ≈ 256 (= `seq_len_per_chip × topk / num_dispatch_groups` = 128 × 8 / 4), modulo routing skew.

### 5.3 Transfer to LoudBox

```bash
rsync -av /data/nmilicevic/combine_captures/ user@lb:/data/captures/
```

At 1K isl this is ~30 MB total per captured layer; rsync is trivial.

### 5.4 Replay (LoudBox 8×1)

Run **one capture × num_links combo at a time** so each result CSV gets a
unique filename suffix that the analyzer can parse:

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_DS_COMBINE_CAPTURE_DIR=/data/captures

for cap in $TT_DS_COMBINE_CAPTURE_DIR/L*/col*.pt; do
    layer=$(basename $(dirname $cap))    # L03, L30, L60
    col=$(basename $cap .pt)             # col0..col3
    for links_id in linear-8-1link linear-8-2link; do
        name="${layer}_${col}_${links_id}"
        python -m pytest -v \
          models/demos/deepseek_v3_d_p/tests/perf/test_combine_replay.py \
          -k "${name}"
        python /data/nmilicevic/tt-metal/tools/tracy/process_ops_logs.py \
          --device-only -n "$name"
    done
done
```

After this:

```
generated/profiler/reports/*/ops_perf_results_*_L03_col0_linear-8-1link.csv
generated/profiler/reports/*/ops_perf_results_*_L03_col0_linear-8-2link.csv
generated/profiler/reports/*/ops_perf_results_*_L03_col1_linear-8-1link.csv
... etc.
```

### 5.5 Analyze

```bash
python models/demos/deepseek_v3_d_p/tests/perf/analyze_combine_perf.py compare \
  --galaxy /path/to/galaxy/ops_perf_results_*_galaxy_actual.csv \
  --lb-glob "generated/profiler/reports/*/ops_perf_results_*_L*_col*_linear-8-*link.csv" \
  --out combine_perf_comparison.csv
```

Resulting columns:

| Column | Meaning |
|---|---|
| `layer_idx` | MoE layer index (3..60) |
| `num_links` | `linear-8-1link` / `linear-8-2link` etc. |
| `lb_max_col_chip_ns` | LB no-contention estimate: max over 4 cols of (max chip's median timed-iter duration) |
| `lb_min_col_chip_ns` | Min of the same across cols (gauges per-column variance) |
| `n_cols` | Number of cols replayed for this layer (should be 4) |
| `galaxy_max_chip_ns` | Galaxy actual: max over 32 chips of single-invocation combine duration |
| `galaxy_med_chip_ns` | Galaxy median across 32 chips |
| `overhead_ratio` | `galaxy_max_chip_ns / lb_max_col_chip_ns` (>1 means Galaxy slower) |
| `overhead_abs_ns` | Absolute ns gap |

Comparing `1link` vs `2link` rows shows ethernet bandwidth sensitivity
directly. The `overhead_ratio` quantifies how much the 4-way concurrent
contention + any ethernet capacity gap costs vs the no-contention bound.

## 6. Capture file format

Each `L<NN>/col<k>.pt` is a `torch.save` dict:

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `dispatched_metadata` | `(1, dgs=8, max_dispatch_buffer_token_size, 5)` | int32 | Flat across experts within each chip; per-chip slot range = `max_dispatch_buffer_token_size = max_dispatched_tokens_per_expert × capacity_factor` |
| `expert_token_counts` | `(1, dgs=8, num_routed_experts_per_col=64)` | int32 | Sparse; chip `r`'s 8 local experts at positions `[r*8, r*8+8)` |
| `expert_region_offsets` | `(1, 8, 64)` | int32 | Cumsum-derived starting slot per expert |
| `config` | dict | — | `dispatch_group_size, num_dispatch_groups=1, experts_per_chip, num_routed_experts=64, num_experts_per_tok, seq_len_per_chip, emb_dim, max_dispatched_tokens_per_expert, max_dispatch_buffer_token_size, metadata_len=5, buffer_shape (4D), galaxy_column, layer_idx` |
| `dispatched_buffer` | `(1, dgs=8, max_dispatch_buffer_token_size, emb_dim=7168)` | bf16 | **Only present when `TT_DS_CAPTURE_COMBINE_FULL_BUFFER=1`** |

The buffer/metadata layout uses a flat per-chip token dimension (`max_dispatch_buffer_token_size`) sized as `max_dispatched_tokens_per_expert × dispatch_buffer_capacity_factor`. Each expert occupies a TILE-aligned region within that flat dimension; region starts are given by `expert_region_offsets`. The combine kernel reads `expert_token_counts[expert]` valid slots starting at `expert_region_offsets[expert]`.

**Storage allocation note**: Sliced tensors from `ttnn.to_torch` share storage with the full composed mesh tensor. The capture code uses `.clone()` (not `.contiguous()`) on each per-column slice so that `torch.save` only pickles 1/`num_dispatch_groups` of the original storage — otherwise file size would be ~4× larger on Galaxy 8×4.

Metadata's `expert_id` field (`[..., 3]`) still contains Galaxy *global*
expert IDs (e.g., 64..127 for column 1). The combine kernel does not consume
that field for routing (it indexes the buffer by chip + local-expert
position), so this is harmless for replay.

The metadata's `src_chip` field (`[..., 0]`) is already 0..7 (relative to
the dispatch group, per `tt/moe/README.md` §2), so it maps directly to the
8 chips of an 8×1 LB mesh.

## 7. Sizing reference

Let `B = max_dispatch_buffer_token_size = dispatch_group_size × seq_len_per_chip × dispatch_buffer_capacity_factor`.
With `dispatch_buffer_capacity_factor=8` (the test default), `B = 8 × isl_total`. Per-column file sizes:

| Tensor | Formula | 1K (B=8192) | 4K (B=32768) | 25K (B=204800) |
|---|---|---|---|---|
| `dispatched_buffer` (bf16) | `8 × B × 7168 × 2 B` | 938 MB | 3.75 GB | **23.4 GB** |
| `dispatched_metadata` (int32) | `8 × B × 5 × 4 B` | 1.25 MB | 5 MB | 32 MB |
| `expert_token_counts` (int32) | `8 × 64 × 4 B` | 2 KB | 2 KB | 2 KB |
| `expert_region_offsets` (int32) | `8 × 64 × 4 B` | 2 KB | 2 KB | 2 KB |

Per captured layer (4 columns):

| isl | Routing-only (default) | Full buffer (`TT_DS_CAPTURE_COMBINE_FULL_BUFFER=1`) |
|---|---|---|
| 1K | ~5 MB | ~3.75 GB |
| 4K | ~20 MB | ~15 GB |
| 25K | ~128 MB | ~93 GB |

## 8. Why dummy buffers are correct for perf measurement

The combine kernel:

1. Reads `expert_token_counts` to bound the per-expert slot range.
2. For each valid slot, reads `dispatched_metadata` to get
   `(src_chip, token_idx, topk_idx, weight)`.
3. Reads the bfloat8 token vector from `dispatched_buffer`.
4. Multiplies vector × `weight` (an FMA on the Tensix).
5. Writes the result to the source chip's output at `(token_idx, topk_idx)`.

Steps (1)-(4) have no data-dependent branching on the values inside
`dispatched_buffer` or `weight`. The Tensix FMA runs in fixed cycles. Step
(5)'s ethernet/NOC write transfers the same `emb_dim`-sized payload
regardless of byte values. Therefore the kernel's per-iteration cycle
count is determined by:

- The number of valid slots iterated (`expert_token_counts`)
- The destination chip of each write (`metadata[..., 0] = src_chip`)
- The output address (`metadata[..., 1:3] = token_idx, topk_idx`)

All three are captured exactly. Using zero buffers and zero weights
produces identical perf characteristics. The only thing you lose is the
ability to compute correct combine output values for PCC validation.

## 9. Troubleshooting

`python -m tracy --process-logs-only` hangs with "tracy capture out not found"
:  You ran the test without `python -m tracy` wrapping pytest, so there's
   no `tracy_profile_log_host.tracy`. Use the device-only entry point
   instead: `python tools/tracy/process_ops_logs.py --device-only -n <suffix>`.

Capture .pt files don't appear
:  Check `TT_DS_CAPTURE_COMBINE_LAYERS` is set in the test process'
   environment. The layer-set is parsed at module import. Also confirm
   `TT_DS_COMBINE_CAPTURE_DIR` is writable.

Replay test reports "No capture files found"
:  `_list_captures()` runs at pytest collection time. Ensure
   `TT_DS_COMBINE_CAPTURE_DIR` is set in the shell *before* running pytest
   (export it, don't just prefix the test command if you're inside a
   subshell that strips env vars).

Permission denied writing TTNN cache
:  `TT_DS_PREFILL_TTNN_CACHE` must point to a directory you own (or
   group-write). The default falls back to `$model_path/tensor_cache_…/`
   which on shared mounts may be read-only.

LB replay shows much faster combine than Galaxy
:  Expected — that's what you're measuring. The gap quantifies (Galaxy
   4-way contention) + (ethernet bandwidth difference). If LB is *slower*,
   something is wrong with the setup or LB has lower ethernet than Galaxy.

## 10. Potential extensions

- **Slot-trim** the metadata (only save the first `count[e]` slots per
  expert per chip). Cuts metadata size by ~8× at uniform routing. Replay
  must reconstruct full-size metadata buffer before pushing to device.
- **Remap `metadata[..., 3]`** (expert_id) to local 0..63 range for cleaner
  files. Not necessary because the field is unused, but makes the saved
  data more self-consistent.
- **Per-test on-device summary**: have the replay test read its own slice
  of the device profiler immediately after timed iters and emit a small
  per-run JSON. Removes the need for the `--name-append` filename
  convention.
- **PCC mode**: capture full buffer (`TT_DS_CAPTURE_COMBINE_FULL_BUFFER=1`)
  and on LB compare combined output against a CPU reference computed from
  the same metadata + buffer. Confirms that replay produces semantically
  identical results before relying on its timing data.
