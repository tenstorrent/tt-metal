# Qwen3.6-27B on P300x2 (4-chip tensor-parallel)

Optimization of the single-chip P150a `qwen36_27b` port to the QuietBox **P300x2 (4 Blackhole chips)**.
Goal: 8 concurrent connections at TPOT 20 tok/s.

## Results (full 64-layer, decode, TRACE)
| stage | TPOT | tok/s |
|---|---|---|
| single chip (batch=1) | 164 ms | 6.10 |
| 4-chip TP | 115.6 ms | 8.65 |
| + conv moved off the scalar dataflow RISC | 68.3 ms | 14.65 |
| + projection fusion + Ring all_reduce | **52.0 ms** | **19.25** (batch=1) |
| **batch=8 (flattened-head)** | — | **117 tok/s** (≈14.7/user) |

batch=1 TPOT ≈ 20 (TP floor ~19.7; conv-free is 50.8ms). 8-concurrent met via flattened-head batch.

## Source changes (in `models/demos/qwen36_27b/` and `ttnn/.../deltanet/`)
- **Build fix** (branch bug): `ttnn/sources.cmake` + `experimental_nanobind.cpp` were missing the
  `deltanet_decode_full` / `deltanet_prefill_full` nanobind sources → undefined symbol. Added.
- **`tt/attention.py`**: fully on-device decode (`_decode_ondevice`, `_decode_ondevice_cached`) —
  per-head norm, partial RoPE, GQA SDPA-decode, paged KV cache, gate. PCC 0.9997 vs CPU path.
- **`tt/model.py`**: `forward_from_embedding` (trace entry point).
- **`tt/deltanet.py`**: `TtDeltaNetState.trace_mode` + in-place state advance for trace replay.
- **`reader_deltanet_full.cpp`**: section-2 conv made PASS-THROUGH (the 1ms scalar conv1d+silu+l2norm
  was the kernel bottleneck; moved to vectorized host ttnn ops). 18× kernel speedup (1.06→0.06ms).

## Key findings
- 4-chip TP runs on a **1x4 line mesh** (`FABRIC_1D`); the earlier "eth blocked" was a zombie container.
- Decode is NOT compute-bound — the DeltaNet kernel bottleneck was a **scalar conv on the dataflow RISC**.
- TP matmuls amortize the weight read when **batch is on the M dim** `[1,1,B,H]`; the DeltaNet kernel
  goes flat across batch via **flattening B into the head dim** (B·nv heads, ≤9 per chip, no kernel change).
- all_reduce: 2x2 p300_x2 is a degree-2 torus → **Ring** topology valid and faster than Linear in trace.

## Scripts
Run via `run.sh <N_chips> <cmd>` (sets device passthrough + MGD + SKIP_ETH). Test image built from
`Dockerfile.test`. Correctness tests: `test_tp_{mlp,attention,deltanet}.py`, `repro_deltanet_*.py`.
Perf: `tp_model.py` (full model), `tp_profile2.py` / `tp_batch2.py` (component + batch scaling).
See `BASELINE.md` for the single-chip baseline + bottleneck analysis.
NOTE: P300 single-chip open needs the p150 MGD + `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1` +
`set_fabric_config(DISABLED)`; clean up Dead docker containers (they lock /dev/tenstorrent) + `tt-smi -r`.
