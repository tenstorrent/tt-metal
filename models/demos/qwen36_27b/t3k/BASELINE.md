# Qwen3.6-27B — T3K baseline & target (8× Wormhole N300, 1×8 line, FABRIC_1D)

Target configuration for the T3K port: full 64-layer Qwen3.6-27B, **8-way tensor-parallel**
on a 1×8 line mesh, bf8 weights. This file is the place to record the measured T3K baseline.

## Measured on T3K (2026-06-16)

Full 64-layer Qwen3.6-27B, TP=8 on the 1×8 line mesh (FABRIC_1D, Linear all_reduce), bf8
weights, dummy weights (zeros) — decode TPOT is value-independent. Measured with
`tp_model.py` (batch=1) and `tp_model_batch.py` (batched, flattened-head). 8 chips via 4×
N300 device nodes. To run on this Wormhole T3K the deltanet decay/beta math (expf/logf) was
moved off the NCRISC reader kernel to host ttnn ops (the libm tables overflow Wormhole's small
NCRISC data region — see `Perf.md`/`reader_deltanet_full.cpp`); the model builds against a
locally-built wormhole_b0 `build_Release`.

### Decode TPOT — batch scaling (64 layers, ctx=64, TRACE)
| batch | ms/step | TPOT per user | throughput |
|------:|--------:|--------------:|-----------:|
| 1 | 74.0 | 13.5 tok/s | 13.5 tok/s |
| 2 | 75.6 | 13.2 tok/s | 26.5 tok/s |
| 4 | 76.3 | 13.1 tok/s | 52.4 tok/s |
| 8 | 78.0 | 12.8 tok/s | **102.5 tok/s** |

Per-step time is ~flat in batch (74→78 ms as B 1→8): batch on the matmul M-dim amortizes the
weight read, so throughput scales ~linearly. EAGER (no trace) is ~250–275 ms/step regardless of
batch (host-dispatch bound); TRACE removes that. batch=1 also: EAGER 231.6 ms/tok (4.32 tok/s).

### Sequence-length scaling (batch=8, TRACE)
| ctx (KV len) | ms/step | throughput |
|-------------:|--------:|-----------:|
| 64 | 78.0 | 102.5 tok/s |
| 4 096 | 80.9 | 98.9 tok/s |
| 32 768 | 99.6 | 80.3 tok/s |
| 262 144 (256K) | **OOM** | — |

### Max sequence length (KV-cache fit, 64 layers) — bounded by the B×ctx product
The full-attention KV cache scales with batch×seq_len. Measured fit/OOM:

| batch | seq_len | B×ctx | result |
|------:|--------:|------:|--------|
| 1 | 262 144 (256K) | 262 144 | ✅ **fits** — TRACE 114.3 ms/step, 8.8 tok/s |
| 2 | 262 144 (256K) | 524 288 | ✅ **fits** — TRACE 118.1 ms/step, 16.9 tok/s |
| 4 | 262 144 (256K) | 1 048 576 | ❌ OOM — 536 MB buffer, ~43 MB/bank free (alloc 1028 MB/bank) |
| 2 | 524 288 (512K) | 1 048 576 | ❌ OOM — 536 MB buffer, ~45 MB/bank free (alloc 1026 MB/bank) |
| 8 | 262 144 (256K) | 2 097 152 | ❌ OOM — 1 GB buffer, ~86 MB/bank free (alloc 985 MB/bank) |

So on this config (bf8 weights, 64 layers, TP=8) the KV-memory ceiling is a **B×ctx product in
(524 288, 1 048 576]** — **256K context is reachable at batch=1 AND batch=2** (524 288 fits), but
not batch=4 (1 048 576 OOM); 512K does not fit even at batch=2. batch=8 ran up to ctx=32 768. To
push beyond B×ctx≈524 K needs more free DRAM: **bf4 weights/experts** (frees the
weight footprint, as in the OpenClaw/coder-next work), fewer layers, or not replicating the KV
cache across the mesh. Weights themselves fit across the 8× N300 (build + run OK).

### Memory model — predict fit/OOM before sweeping
The OOM is a KV-cache DRAM limit and is predictable. The failing allocation is exactly one
KV tensor `kc`/`vc` of shape `[B, nkvp, ctx, head_dim]` bf16, and the cache is **replicated on
every chip** (`mesh_mapper=REP`). Config: 64 layers, `full_attention_interval=4` → **16
full-attention layers**; full-attn `head_dim = 256`; `num_key_value_heads = 4` → with TP=8 KV
replication **nkvp = 1 per chip**.

```
KV per chip (bytes) = n_full(16) × (kc + vc)
                    = 16 × 2 × [ B × nkvp(1) × ctx × head_dim(256) × 2 (bf16) ]
                    = B × ctx × 16384 bytes        (= B·ctx × 16 KiB)
```
Per-chip DRAM budget ≈ **12.86 GB** (12 banks × ~1.072 GB). Fit when

```
W_weights(/chip, bf8 ≈ 3.4 GB) + B·ctx × 16 KiB + overhead(~1 GB) ≤ 12.86 GB
⇒  B · ctx  ≲  ~512 K    (≈ 8.4 GB KV budget / 16 KiB)
```

Cross-check (✓ matches all measured points): B·ctx 262 144 (B1@256K) → 4.3 GB KV, fits;
**524 288 (B2@256K) → 8.6 GB KV, fits** (TRACE 118 ms, 16.9 tok/s); 1 048 576 (B4@256K, B2@512K)
→ 17.2 GB, OOM; 2 097 152 (B8@256K) → 34 GB, OOM. So the measured ceiling is **B·ctx ∈
(524 288, 1 048 576]** — the model's ~512 K estimate was slightly conservative; the true KV
budget is ≈ 8.6 GB/chip (since 8.6 GB KV at B·ctx=524 288 fits), i.e. W+overhead ≈ 4.2 GB/chip.

**Practical rule for future sweeps:** compute `B·ctx`. If ≲ 520 K → fits (skip), if ≳ 1.05 M →
OOM (skip); only measure the ~520 K–1.05 M boundary (one bisection pins it). To raise the ceiling, the levers all increase
the KV budget: bf4 weights (more free DRAM), fewer/cheaper full-attn layers, smaller head_dim,
or sharding the KV cache across the mesh instead of replicating (would cut KV/chip by up to 8×).

> Historical reference (different hardware — do **not** read as T3K): a single P300 Blackhole
> chip measured 171.8 ms/tok (5.82 tok/s) eager / 164.0 ms (6.10) trace at batch=1; the 4-chip
> P300×2 TP reached 52.0 ms/tok (19.25 tok/s) batch=1. See `../p300x2/BASELINE.md`. T3K (8× WH,
> 1×8 line) batch=1 trace is 73.8–74.0 ms — slower than P300×2's 52 ms (more all_reduce hops on
> an 8-chip line vs the 4-chip torus, and WH vs BH), but scales to 102 tok/s at batch=8.

## Device-agnostic design notes (hold on T3K)
- **Decode is compute-bound, not dispatch-bound.** Trace (removing host dispatch) gave only ~5%
  on the prior HW; the DeltaNet per-layer kernel time is real compute. The win came from moving
  the scalar conv1d off the dataflow RISC into vectorized host ttnn ops (18× kernel speedup),
  not from trace alone.
- **Batch raises throughput, not per-user TPOT.** Batch is placed on the matmul M-dim
  `[1,1,B,H]` (amortizes the weight read); attention uses batched `sdpa_decode`; DeltaNet
  flattens B into the head dim (B·nv heads/chip, one kernel call). Per-step time is ~flat in B,
  so throughput ≈ B × per-user rate.
- **DeltaNet decode kernel processes its per-chip heads**; the recurrent state is `[B, H, Dk, Dv]`.
  On T3K each chip holds nv/TP = 6 value heads, nk/TP = 2 key heads.

## T3K-specific considerations
- **Topology**: 1×8 line (FABRIC_1D) → `Topology.Linear` for all_reduce / all_gather. (The
  P300×2 2×2 torus used `Ring`; not applicable here.) Confirm Linear-vs-Ring with `comm_bench.py`
  on T3K — a line is degree-≤2 so Ring is not the native topology.
- **GQA KV replication**: 4 KV heads < 8 chips → each KV head replicated across 2 chips
  (1 KV head + 3 query heads per chip). KV cache per chip is `[B, 1, ctx, hd]`.
- **Per-chip memory headroom**: N300 has less DRAM/chip than the P300 Blackhole, but TP=8 splits
  weights into 8 (vs 4), so per-chip weight footprint is ~half the P300×2 case. Verify KV-cache
  and DeltaNet-state headroom at the target batch/context when measuring.

## Levers to tune on T3K (once baselined)
1. Confirm all_reduce topology (Linear) and per-sublayer collective cost (`tp_profile2.py`).
2. Batch scaling for throughput (`tp_model_batch.py`, `tp_profile_b8.py`); find max single-call B.
3. DeltaNet decode C++ kernel (largest compute share) and precision (bf8 → bf4 experts/weights).
4. Trace capture for the full forward path to remove residual host dispatch.
