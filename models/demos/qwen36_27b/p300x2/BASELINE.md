# Qwen3.6-27B — single-chip baseline (P300, 1 chip, batch=1, fabric disabled)

Measured 2026-06-14 on tt-quietbox, one P300 Blackhole chip (~32GB), bf8 weights.
Command:
```
run.sh 1 python3 /home/ttuser/ttwork/qwen36-p300/baseline.py \
    --model-path /home/ttuser/ttwork/qwen36-weights --max-tokens 32
```

| metric | value |
|---|---|
| weights fit on 1 chip | ✅ yes (27GB / 32GB) |
| output correctness | ✅ coherent generation |
| model build (load + to-device) | 216 s |
| prefill / TTFT | 20 tok in 7.10 s = **2.8 tok/s** |
| decode TPOT (median) | **171.8 ms/tok = 5.82 tok/s** |
| decode TPOT (avg, incl. JIT) | 235 ms/tok = 4.25 tok/s |

Target: **20 tok/s TPOT @ 8 concurrent**. Baseline batch=1 single chip = 5.8 tok/s → ~3.4× short even before adding concurrency.

## Why it's slow (bottlenecks)
- Attention layers (16 of 64) use a **CPU fallback**: Q/K/V copied to host, attention + RoPE + KV-cache done in torch, result copied back — a host↔device round-trip every 4th layer.
- Per-token host synchronization; no decode trace.
- Single chip only (4-chip TP blocked: inter-board ethernet not cabled/live, ETH_LIVE_STATUS=0x0 on all chips).

## Trace result (on-device attention + DeltaNet in-place state)
Full 64-layer decode captured into a trace and replayed (`trace_decode.py`):
- **trace median 164.0 ms/tok (6.10 tok/s)** vs eager 171.8 ms (5.82) → **only ~5%**.
- ⇒ decode is **COMPUTE-bound, not dispatch-bound**. Removing host dispatch (trace) barely helps; the ~2.71 ms/DeltaNet-layer is real kernel compute. Trace infra is built & validated but is not the path to 20 tok/s.

## Batch size confirmation
- `scaled_dot_product_attention_decode` + `paged_update_cache`: **batch>1 supported natively**.
- **DeltaNet decode kernel: batch=1 only** (no batch dim; parallelizes over heads; state `[1,H,Dk,Dv]`). batch=8 needs either a kernel batch-dim extension or running the op 8× with 8 independent states.
- batch=8 raises **throughput** ~8× but **not TPOT** (per-step time stays ≥164ms) → batch alone does not meet the per-user 20 tok/s goal.

## Levers to reach 20 tok/s @ batch=8 (revised — compute-bound)
PRIMARY: multi-chip tensor-parallel across 4 chips (~4× less compute/chip) — currently HW-blocked (inter-board eth not live). Then: optimize the DeltaNet decode C++ kernel (73.7%, compute-bound), lower precision. Original (pre-trace) list below.

## (original levers list)
1. Move attention fully on-device (ttnn SDPA / paged KV / device RoPE) — removes the per-layer host round-trips.
2. batch=8 decode (model is currently batch=1 only; DeltaNet recurrent state + KV cache must be batched).
3. Decode trace capture (QWEN_USE_TRACE) once the forward path is all-device.
4. Multi-chip TP (needs eth fixed) to cut per-step latency further.
