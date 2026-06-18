# Qwen3.6-27B on T3K (8-chip tensor-parallel)

Tensor-parallel port of `qwen36_27b` to a Tenstorrent **T3K / QuietBox (8× Wormhole N300)**,
run as a **1×8 line mesh** with `FabricConfig::FABRIC_1D`. Adapted from the P300×2 (4-chip
Blackhole) TP port — see `../p300x2/` for that variant and its measured baseline.

Goal: serve concurrent connections at a usable per-user TPOT while fitting the full
64-layer model across the 8 chips.

## Tensor-parallel layout (TP = 8, 1×8 line)

| component | sharding on T3K (TP=8) |
|---|---|
| MLP gate/up (col) + down (row) | intermediate 17408 → **2176/chip**, row-parallel down + all_reduce |
| DeltaNet (linear attn) | value heads 48 → **6/chip**, key heads 16 → **2/chip**; key_dim 2048→256, val_dim 6144→768, conv_dim 10240→1280 |
| full_attention (GQA) | query heads 24 → **3/chip**; KV heads 4 **< 8**, so each KV head is **replicated across 2 chips** (chip pair *j* holds KV head *j*); o_proj row-parallel + all_reduce |
| lm_head | padded vocab 248320 → **31040/chip** (col-parallel), all_gather |
| collectives | **`Topology.Linear`** (1×8 line, FABRIC_1D) for all_reduce / all_gather |

The GQA KV replication is the one structural difference from P300×2 (where TP=4 gave a clean
1 KV head/chip). On T3K the fused QKV weight carries `kv_slots = TP` (= 8) KV slots — each of
the 4 KV heads duplicated ×2 — so every chip owns exactly one KV head, matching its 3 query
heads (query head *g* belongs to KV group *g // 6*; chips 2*j* and 2*j*+1 share KV group *j*).

## Performance

**Not yet measured on T3K.** The scripts run on 8× N300; numbers below are to be filled in
from a T3K run (`docker start` the device container, then run on the 8-chip mesh).

| stage (full 64-layer, decode, TRACE) | TPOT | tok/s |
|---|---|---|
| 8-chip TP, batch=1 | _TBD_ | _TBD_ |
| 8-chip TP, batch=8 (flattened-head) | _TBD_ | _TBD_ |

> For reference (different HW, do **not** read as T3K): the P300×2 (4-chip Blackhole) variant
> reached 52.0 ms/tok (19.25 tok/s) at batch=1 and ~142 tok/s at batch=8 (TRACE). See
> `../p300x2/README.md`.

## Source changes (device-agnostic, shared with the P300×2 port)
These live in `models/demos/qwen36_27b/` and `ttnn/.../deltanet/` and are reused as-is on T3K:
- **Build fix**: `ttnn/sources.cmake` + `experimental_nanobind.cpp` missing the
  `deltanet_decode_full` / `deltanet_prefill_full` nanobind sources → undefined symbol. Added.
- **`tt/attention.py`**: fully on-device decode (`_decode_ondevice`, `_decode_ondevice_cached`) —
  per-head norm, partial RoPE, GQA SDPA-decode, paged KV cache, gate.
- **`tt/model.py`**: `forward_from_embedding` (trace entry point).
- **`tt/deltanet.py`**: `TtDeltaNetState.trace_mode` + in-place state advance for trace replay.
- **`reader_deltanet_full.cpp`**: section-2 conv made PASS-THROUGH (scalar conv1d+silu+l2norm
  moved to vectorized host ttnn ops). 18× kernel speedup (1.06→0.06ms).

## Key findings (carried over; topology adapted to T3K)
- T3K runs as a **1×8 line mesh** (`FABRIC_1D`) → all_reduce/all_gather use **`Topology.Linear`**
  (the P300×2 2×2 torus used `Ring`; that does not apply to a line).
- Decode is NOT dispatch-bound — the DeltaNet kernel bottleneck was a scalar conv on the
  dataflow RISC, now moved to host vectorized ops.
- TP matmuls amortize the weight read when **batch is on the M dim** `[1,1,B,H]`; the DeltaNet
  kernel goes flat across batch by **flattening B into the head dim** (B·nv heads/chip).
- GQA KV heads (4) < TP (8): replicate each KV head across `TP/nkv = 2` chips (see layout above).

## Scripts
Same suite as `../p300x2/`, now TP=8 / 1×8 line / Linear topology:
- Perf: `tp_model.py` (full model), `tp_profile.py` / `tp_profile2.py` (component breakdown),
  `tp_batch.py` / `tp_batch2.py` / `tp_model_batch.py` / `tp_profile_b8.py` (batch scaling).
- Correctness: `test_tp_{mlp,attention,deltanet}.py`, `test_conv1d_ttnn.py`, `repro_deltanet_*.py`.
- Comms: `comm_bench.py` (all_reduce Ring vs Linear vs sequence-parallel — kept for comparison).

Run on the 8-chip mesh, e.g. `QWEN_TP_LAYERS=64 python3 tp_model.py --steps 16`.

## Sequence-parallel: tried on P300×2, NOT used (backfires for decode)
`comm_bench.py` compares all_reduce vs reduce_scatter+all_gather on the decode tensor
`[1,1,B,H]`. On P300×2 SP was ~13× slower (it shards the small token dim into two
latency-bound collectives; it only helps large bandwidth-bound tensors). The model path uses a
single all_reduce per sublayer. Re-run `comm_bench.py` on T3K to confirm the same holds for the
1×8 line before relying on it.
