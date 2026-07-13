# Profiling the GPT-OSS-120B head (embedding) & tail (lm_head+sampling)

Same per-op device-profiling approach used for the decoder layer, applied to the **head**
(embedding) and **tail** (lm_head + sampling) stages, run **isolated** so we can see each stage's
execution time.

## TL;DR results (Blackhole loudbox, TP=8, slow dispatch, synthetic weights)

| stage | dominant ops (busy-core median µs) | notes |
|---|---|---|
| **Embedding (head)** | `EMBEDDING` lookup **~6.5**, `CROSS_DEVICE_SEND` **~6.2**, sockets **~0.9** | lightweight; ~6–10 µs of critical work vs ~70 µs / decoder layer |
| **LM-head + sampling (tail)** | vocab `MATMUL` **~5.3**, `ARGMAX` **~10.0**, `RMSNORM`/`CCL_BROADCAST` **~22** (mostly fabric wait) | ~15–25 µs; the norm/broadcast scopes are wait-inflated |

Order-of-magnitude: **head ≈ 1/10 of a decoder layer; tail ≈ 1/4–1/3**. Same compute-vs-wait caveat
as the decoder (the `*_MCAST` / `CCL_BROADCAST` enclosing scopes conflate compute + fabric wait).

## No new test needed — existing op tests, configured like the full model

Both stages already have model-parametrized silicon op tests with a `gpt_oss_120b` entry + synthetic
fallback + a harness:
- **Embedding:** `tests/blaze/fused_ops/embedding/test_embedding_layer.py`
- **LM-head:** `tests/blaze/fused_ops/lm_head/test_lm_head_sampling_layer.py`

They drive the *same* SSOT config the full model uses (`make_lm_head_config`, embedding_dim=2880,
`embedding_core_coord`, vocab padding, sampling policy) via the generic `embedding_layer` /
`lm_head_sampling_layer` ops — so profiling them isolated matches the full-model op.

### Node IDs used
```
# head — embedding, synthetic, pipeline sockets, 100 iters
tests/blaze/fused_ops/embedding/test_embedding_layer.py::test_embedding_layer[blackhole-True-gpt_oss_120b-receiver_mesh_coord0-sender_mesh_coord0-synthetic-socket_logic_oneshot-100-1337-fabric_2d]

# tail — lm_head + argmax sampling, no_sockets (op path), 5 iters
tests/blaze/fused_ops/lm_head/test_lm_head_sampling_layer.py::test_lm_head_sampling_layer[blackhole-True-device_params0-1337-5-no_sockets-argmax_op-gpt_oss_120b]
```

## Run recipe (inside the dev container)
```bash
cd /localdev/divanovic/tt-blaze && source env.sh
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
export TT_METAL_PROFILER_ZONE_NAME_ONLY=1       # avoids the 16-bit zone-hash collision
export BLAZE_PROFILE_READ=1                      # tail only: makes the lm_head harness dump device zones
D=/localdev/divanovic/<stage>; rm -rf $D; mkdir -p $D; export TT_METAL_PROFILER_DIR=$D
python -m tracy -r -p -v -o $D -m pytest "<nodeid>"
# per-op extraction (prefix per stage):
python3 scripts/perf_report/gen_perop.py $D/.logs/profile_log_device.csv EMBEDDING_LAYER__
python3 scripts/perf_report/gen_perop.py $D/.logs/profile_log_device.csv LM_HEAD_SAMPLING_LAYER__
```

## Gotchas I hit (and the fixes — now on PR #2068)

1. **`gen_csv.py` is hard-coded to the GPTOSS decoder prefix** → 0 ops for other stages. Added
   **`scripts/perf_report/gen_perop.py`** (prefix-configurable) for `EMBEDDING_LAYER__` /
   `LM_HEAD_SAMPLING_LAYER__` / any fused stage.
2. **The lm_head harness never read the device profiler** (blaze ops bypass ttnn op-dispatch, so
   tracy's auto-read at close found nothing). Added a **gated** `ttnn.ReadDeviceProfiler(mesh)` on
   `BLAZE_PROFILE_READ=1` in `tests/blaze/fused_ops/lm_head/harness.py`.
3. **lm_head `sampling_defaults` overflows the Tensix kernel-config buffer** with full per-op zones
   (`Program size 78880 > 70656`). Two fixes: use the smaller **`argmax_op`** variant (same dominant
   vocab matmul), **or** `BLAZE_PROFILE_MINIMAL_ZONES=1` (emits only `STAGE_CHECKPOINT`).
4. **Head/tail run on few cores** (embedding_core, lm_head grid), so median-over-all-cores is
   ~zone-overhead (15 ns). Use **busy-instance stats** (median/max over instances > 0.5 µs), which
   `gen_perop.py` does.
5. **`socket`/`sockets` variants** add socket kernels that push lm_head further over the config-buffer
   limit and (for lm_head) run host-socket orchestration that doesn't dump device zones → prefer
   `no_socket_logic` / `no_sockets` for op-path device profiling; `socket_logic_oneshot` is fine for the
   (small) embedding.

## Raw per-op (busy-instance median / max µs)

**Embedding**: `EMBEDDING` 6.54 / 20.9 · `CROSS_DEVICE_SEND` 6.18 / 21.0 · `CROSS_DEVICE_SIGNAL_RECEIVE`
5.84 / 6.2 · `RECEIVER_SOCKET__H2D` 0.92 / 14.8 · `SENDER_SOCKET__D2H` 0.92 / 0.98

**LM-head**: `MATMUL` (vocab proj) 5.31 / 6.40 · `ARGMAX` 9.95 / 18.1 · `RMSNORM` 22.48 / 22.76 ·
`CCL_BROADCAST` 21.72 / 22.0 · `ACT_MCAST` 0.85 / 1.17 (RMSNORM/BROADCAST are enclosing scopes → mostly
fabric wait, not compute)

*Method identical to the decoder (see `gptoss-blaze-profiling` skill): pair START/END per
(chip,core,RISC,zone) → median per core → busy cores → per-op = max/median over busy cores. Wall
markers (`STAGE_CHECKPOINT`) are per-iteration.*
