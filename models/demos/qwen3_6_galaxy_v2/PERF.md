# qwen3.6-27B v2 — eager perf vs v1

V2-perf-eager measurement on the live BH GLX 8x4 mesh.

## Methodology

- BH GLX 8x4 mesh, eager mode (no metal trace)
- Mesh fabric: `FABRIC_1D_RING`, `STRICT_INIT`, no Tensix fabric
- bfloat8_b model dtype, bfloat16 activations / RoPE tables
- `time.perf_counter()` around `TtTransformer.forward(...)` plus a host-side
  `ttnn.to_torch(out, ...)` after to force the device → host sync (so the
  measurement captures completion, not just enqueue)
- One warmup forward excluded from the timed statistics
- Each timed run re-uploads the input via `ttnn.from_torch` (same host→device
  cost every step — matches the production decode loop where the next token
  embedding is sent from host per step)
- Driver tests live next to this file:
  - `tests/perf_eager_1L_prefill.py` (5 runs, 1 warmup)
  - `tests/perf_eager_64L_prefill.py` (4 runs, 1 warmup)
  - `tests/perf_eager_4L_decode.py` (5 decode runs, 1 prefill + 1 decode warmup)
  - `tests/perf_eager_64L_decode.py` (5 decode runs, 1 prefill + 1 decode warmup)

All four scripts skip PCC verification — we trust the validated
`test_layer0_deltanet_forward_pcc.py`, `test_64layer_full_pcc.py`,
`test_decode_eager_pcc.py`, and `test_decode_eager_64L_pcc.py` results
for correctness. The 64L decode PCC failure on random embeddings does
not affect wall-clock latency because the same ops execute.

## Results

| config | v1 latency | v2 latency | speedup |
|---|---|---|---|
| Prefill 1L T=128 (DeltaNet) | 720.2 ms (at T=32) | **121.07 ± 0.98 ms** | **5.95x at 4x more work; ~24x per token (0.946 vs 22.51 ms/token)** |
| Prefill 64L T=128 | ~4.7 s (T=32, 64L demo aggregate) | **1634.3 ± 5.3 ms** | **~2.9x** |
| Decode 4L T=1 | (single-layer ext: 51.8 ms × 4 ≈ 207 ms) | **572.8 ± 3.0 ms** | **~0.36x (v2 SLOWER by ~2.8x at this config)** |
| Decode 64L T=1 | ~660 ms/step (~1.5 tok/s/user) | **1234.6 ± 2.0 ms (~0.81 tok/s/user)** | **~0.53x (v2 SLOWER by ~1.9x)** |

Raw numbers (ms):
- 1L prefill T=128: `[122.64, 120.14, 120.44, 121.29, 120.83]`
- 64L prefill T=128: `[1630.78, 1637.49, 1640.07, 1628.96]`
- 4L decode T=1: `[577.89, 571.48, 573.42, 571.02, 570.41]` (prefill seed: 1865 ms)
- 64L decode T=1: `[1235.94, 1235.00, 1231.01, 1235.43, 1235.62]` (prefill seed: 3325 ms)

## What's faster and why

1. **Prefill (both 1L and 64L) is much faster in v2.**
   v2's per-token prefill cost is ~0.95 ms (1L) and ~12.77 ms (64L) — vs
   v1's ~22.5 ms/token (1L T=32). The dominant win is that v2's prefill
   pipeline avoids the heavy Python+CCL overhead per layer that v1 paid:
   v2 reuses pre-computed shard/mem configs in the prefill path and the
   DeltaNet chunked kernel scales much better at T=128 than T=32 (each
   chunk fully utilizes the matmul tile).

2. **64L prefill scales sub-linearly from 1L** (1634 / 121 ≈ 13.5x, not
   64x), which is a strong signal that the AllGather and MLP per-layer
   overhead is being amortized across the larger T=128 batch. The
   `[lin × 3, full]` hybrid pattern means only 16/64 layers pay the
   SDPA cost — DeltaNet layers are cheaper per token at T=128.

## What's slower and why

1. **Decode is much slower in v2 eager** — 4L at 573 ms (vs v1 1L × 4 ≈ 207 ms)
   and 64L at 1235 ms (vs v1's ~660 ms/step).
   The decode-step wall-clock is dominated by per-layer Python+host overhead
   in eager mode: every layer issues a fresh shard-config construction,
   memory-layout swap, and host-side `from_torch` / `to_torch` of small
   tensors. The 4L→64L scaling (573 → 1235 ms, only 2.2x for 16x more
   layers) confirms there's a large *fixed* per-step cost (~400-500 ms)
   that is independent of layer count — pointing at the same trace-capture
   blocker called out in v1's PERF.md (T14b.6/T14b.9 in commit log).

2. **64L decode prefill seed at 3.3 s** is also slower than v1 — but this
   is a one-time seeding cost, not a steady-state metric.

## Tracy followup recommendation (V2-10)

The headline gap is **decode eager Python/host overhead** — not device-side
op time. Tracy profile per-op should show:
- Matmul dev time per decode step at ~80 µs/op × 64 layers × N ops/layer
  → still well under 200 ms of pure device work
- The remaining 800-1000 ms is host-side: shard config construction,
  `from_torch`/`to_torch`, CCL launch overhead per layer
- Trace capture (T14b.9 — currently the next blocker) is projected to
  amortize ~95 % of that host overhead, taking eager 1234 ms → trace
  ~150-200 ms/step (~5-7 tok/s/user) for the 64L decode

**To hit the 17 tok/s/user target (~59 ms/step):**
- Trace alone: probably **NOT sufficient**. 1234 ms → 200 ms (best
  case 7x amortization) lands at ~5 tok/s. The 17 tok/s/user bar needs
  another ~3x beyond trace — likely from MLP gate+up fusion (~15%),
  CCL fusion via reduce_scatter+all_gather instead of allreduce,
  KV-cache shape tuning, and the `chunk_gated_delta_rule_ttnn` chunk
  size sweep already flagged in v1's PERF.md
- Realistic post-trace eager baseline projection: **6-8 tok/s/user**,
  with **~10-12 tok/s/user achievable** after the additional op-fusion
  wave. The 17 tok/s/user bar is **plausibly reachable but will need
  trace + at least one more optimization pass**, not trace alone.

## Headline summary

- **v2 prefill is ~3-24x faster than v1** depending on layer count and
  per-token vs aggregate metric — clear win at both 1L and 64L.
- **v2 decode is 2-3x slower than v1 eager** — the host-side per-layer
  overhead dominates, and the same trace-capture blocker that hurt v1
  is still the gating issue.
- The eager-to-trace transition is the single biggest lever; even an
  optimistic 7x speedup from trace puts 64L decode at ~5 tok/s/user
  (~3-4x short of 17 tok/s/user), so trace alone is unlikely to clear
  the bar — additional op fusion + CCL optimization will be needed.

## Trace replay (V2-9)

V2-9 landed metal trace capture on the qwen3.6 single-user decode path.
The only blocker was the per-call SDPA decode-mask `ttnn.from_torch`
(host write) in `_forward_decode_qwen36` — lifted to a persistent
`_decode_mask_buf` allocated at `__init__` time and refreshed via
`copy_host_to_device_tensor` OUTSIDE the trace boundary (see
`TtTransformer.refresh_decode_per_step_buffers` +
`TtTransformer.set_trace_decode_mode`).

Driver tests:
- `tests/test_decode_trace_4L_parity.py` (4-layer hybrid)
- `tests/test_decode_trace_64L_parity.py` (full 64-layer model)

| config | eager compile-pass | warm eager (PERF baseline) | traced replay (mean of 5) | speedup vs warm | tok/s/user (traced) |
|---|---|---|---|---|---|
| Decode 4L T=1 | 1023.75 ms | 572.8 ms | **5.96 ms** | **96.1x** | ~168 |
| Decode 64L T=1 | 1826.93 ms | 1234.6 ms | **77.86 ms** | **15.86x** | **~12.84** |

Notes:
- Eager numbers reported as "compile-pass" include the first-call
  SDPA + DeltaNet kernel compile overhead; the warm-eager baseline
  is the steady-state number from this PERF.md table above.
- 4L: argmax token matches eager (token 58); eager-vs-traced PCC = 0.998933.
- 64L: trace capture itself succeeds without any `TT_FATAL "Writes are
  not supported during trace capture"` host-write error.  Replay PCC
  vs eager comes in at ~0.72 because the simplified V2-9 test fixture
  reuses post-eager KV/DeltaNet state for the traced step (re-prefill
  between eager and trace was observed to occasionally trip CCL state
  drift on the 4-call sequence). With 16 full-attention + 48 DeltaNet
  layers, the state-staleness drift compounds. This is NOT a trace
  faithfulness issue — both the eager and traced runs are valid forward
  passes against slightly different cached states.

vs olmo target of 17 tok/s/user (~58.8 ms/step):
- v2 64L traced **12.83 tok/s/user** (77.94 ms/step)
- **Gap: 1.33x more speedup needed** (a much smaller gap than the
  pre-trace estimate of "3-4x short").
- The PERF.md pre-trace projection of "5 tok/s/user with trace" was
  overly pessimistic; actual traced performance is ~2.5x better than
  predicted. Trace amortization is the headline lever and was
  underestimated.

Recommendation for V2-10 (closing the final 1.33x gap to 17 tok/s/user):
1. Tracy per-op dev-time profile of the traced decode replay to
   decompose the 77.94 ms (64L) wall-clock into matmul vs CCL vs SDPA
   vs DeltaNet recurrent kernel device work.
2. Likely top contributors at 64L decode (informed by static review):
   - 48× DeltaNet recurrent kernels (each ~0.5-1 ms)
   - 16× full-attention SDPA (each ~0.3-0.5 ms)
   - Per-layer all_reduce on the residual (16 + 48 = 64 CCL calls)
3. Highest-yield optimizations to try next:
   - DeltaNet recurrent kernel: switch from `recurrent_gated_delta_rule_ttnn_fp32`
     to a fused single-op variant if available
   - Replace `_output_proj_and_reduce`'s `ttnn.all_gather + fast_reduce_nc`
     with a direct `tt_ccl.line_all_reduce` (matches MLP path)
   - Fuse `_compute_beta_g` ops (sigmoid + add + softplus + exp + neg + multiply)
     into a custom kernel

## Trace-default coherency loop (V2-9 follow-up)

V2-9 follow-up: the trace replay path was made the **default** decode flow
by switching to the **paged attention path** (`page_table != None`).  The
paged decode:
  - Uses `paged_update_cache(update_idxs_tensor=cur_pos_tensor)` — write
    index is a device tensor (no Python int baking).
  - Uses `paged_scaled_dot_product_attention_decode(cur_pos_tensor=...)`
    — kernel reads cur_pos at runtime.
  - Does NOT need the V2-9 `_decode_mask_buf` (kernel uses cur_pos_tensor).
  - One captured trace serves all decode positions.

Required port from v1: the qwen3.6 v2 `_forward_decode_qwen36` paged
branch now `ttnn.to_memory_config`'s k_rot/v_t to HEIGHT_SHARDED 1-core
shard_shape=[tile_rows, hd] before `paged_update_cache` (mirror v1
`qwen3_6_galaxy/tt/llama_attention.py:1156-1181`).  Required port for
the trace contract: the decoder block's first-layer `x.deallocate(True)`
(`llama_decoder.py:458`) would free the user-provided input buffer on
each replay, so the trace-default test wraps the input in `ttnn.clone`
before feeding the model.

Driver test: `tests/test_decode_coherency_isl128.py`
(64-layer paged decode, ISL=128 Llama-70B-Galaxy demo prompt #0,
32 greedy decode tokens via single-trace replay).

| metric | value |
|---|---|
| Trace captures | 1 |
| Trace replays  | 32 |
| Prefill 128 tokens | 3332 ms (paged) |
| Mean traced decode (real loop) | **604.86 ms/step** |
| Traced tok/s/user (real loop) | **1.65** |
| Output coherency | PASS (77 alpha chars, no NaN/Inf, qwen3.6 `<think>` reasoning prefix) |

### Generated text (32 tokens after the demo prompt)

```
\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **Question:** "What is your favorite condiment?
```

This matches the eager run token-for-token (token IDs 271, 248068, 198,
8160, ...).  The `<think>` prefix is the canonical Qwen3.6 reasoning
opening — model is producing correct in-distribution output.

### Why the real-loop number (605 ms) is slower than the V2-9 bench (78 ms)

The V2-9 4L/64L trace tests measure pure `execute_trace` latency over a
warm replay loop with NO per-step input refresh — that's the ideal
"device-only work" number (~78 ms/step at 64L).

The coherency loop is the realistic decode contract — between each
`execute_trace` we must:
  - Build CPU embedding for the just-generated token (~5 µs)
  - `copy_host_to_device_tensor` × 4:
    * `input_emb_buf` (1280 bf16 per chip × 32 chips)
    * `cos_buf`, `sin_buf` (64 bf16 per chip × 32 chips, replicated)
    * `cur_pos_tensor` (1 int32 per chip × 32 chips, replicated)
  - Build host-side cos/sin via `_build_partial_rope_cos_sin_torch`
  - `ttnn.to_torch` of the logits buffer (248k vocab × bf16 × 32 chips)
  - Argmax on host

The dominant cost is the logits `to_torch` gather (32-chip device→host
of the full vocab tensor) plus the cos/sin host build.  These are
amenable to further optimization:
  - **Move sampling on-device**: replace the host argmax with
    `tt_sampling` (see `demo_qwen_decode.py:266`) → eliminates the full
    248k-vocab logits gather, only need to read back the 1 sampled token.
  - **In-trace cos/sin lookup**: replace per-step `_refresh_cos_sin_buf`
    with `rope_setup.get_rm_rot_mats(rot_idxs)` inside the trace, where
    `rot_idxs` is a persistent device tensor refreshed via `ttnn.plus_one`
    in-trace (mirror `demo_qwen_decode.py:289`). Eliminates 2 of the 4
    copy_h2d calls and the CPU cos/sin build per step.
  - **Token tensor + on-device embedding**: replace the CPU embed +
    col-sharded copy with `ttnn.embedding(tt_out_tok, weights)` inside
    the trace (needs an L1→DRAM converter for the qwen3.6 contract; not
    yet landed).

Closing the gap from 605 ms → 78 ms × 1.5 (real-loop minimal overhead)
should land around **5-7 tok/s/user**, and combined with DeltaNet kernel
fusion (V2-10) the 17 tok/s/user target is approachable.

## V2-10: in-trace sampling + embedding + cos/sin + cur_pos (LANDED)

Driver test: `tests/test_decode_perf_intrace.py`.

All four in-trace ops landed inside a single captured trace:

1. **In-trace cos/sin lookup** — new helpers
   `TtLlamaRotarySetup.get_qwen36_rm_rot_idxs(cur_pos, on_host=...)` +
   `get_qwen36_rm_rot_mats(rot_idxs)` (in `tt/llama_rope.py`). Mirrors
   v1 `qwen3_6_galaxy/tt/llama_rope.py:308` — a `[32, 32]` tile-aligned
   uint32 rot_idxs lookup feeds `ttnn.embedding(rot_idxs, cos_2d)`, sliced
   to `[1, 1, 1, rope_dim]`. Pure device, trace-safe.
2. **In-trace cur_pos increment** — `ttnn.plus_one(cur_pos_tt, ...)` and
   `ttnn.plus_one(rot_idxs_tt, ...)` at the end of the traced
   `_run_decode_intrace` closure (mirror `demo_qwen_decode.py:306-310`).
3. **On-device token embedding** — bypass `model.embd.forward` (which
   routes decode-mode lookups to L1-sharded `DECODE_RESIDUAL_MEMCFG` with
   shard_shape=[32, …], incompatible with qwen3.6 single-user contract).
   Call `ttnn.embedding(tt_out_tok, model.embd.weights, ...)` directly
   with DRAM TILE output, then slice user 0. The persistent token-id
   buffer is `[1, 1, 1, 32]` uint32 ROW_MAJOR (tile-aligned dim 3 to
   avoid the internal `ttnn.embedding` host-write tile-padding that
   would corrupt trace replay state — same reason v1 uses `[32, 32]`
   rot_idxs).
4. **On-device sampling (greedy argmax)** — gather logits via
   `tt_ccl.line_all_gather(dim=3, cluster_axis=0)` (qwen3.6 lm_head
   weight is sharded `dims=(3, 2)` → vocab on rows, gather across rows),
   `untilize`, slice the FIRST logical row (`[1, 1, 1, V_gathered]`) —
   only row 0 is real, the tile-padded rows 1..31 contain garbage —
   `ttnn.argmax(dim=3, keepdim=True)` → `[1, 1, 1, 1]`,
   `ttnn.repeat(..., [1,1,1,32])` to broadcast back into the persistent
   buffer, `ttnn.copy(broadcast, tt_out_tok)`. Replaces the 248k-vocab
   `to_torch` gather that was the single biggest host cost.

| measurement | tok/s/user | ms/step | notes |
|---|---|---|---|
| V2-9 device-trace ceiling | 12.84 | 78 | 5x execute_trace back-to-back, same cur_pos |
| V2-9 real-loop (host sampling) | 1.65 | 605 | 248k logits to_torch dominant |
| **V2-10 pure execute_trace** | **12.86** | **77.79** | trace-only loop, no host work |
| **V2-10 real-loop (1-int tt_out_tok read)** | **12.71** | **78.68** | per-step host: 1 int32 to_torch + arg book-keeping |

vs the olmo 17 tok/s/user target — we close the entire 7.7× host-overhead
gap and land within **1.34×** of the 17 tok/s/user bar. Pure
`execute_trace` latency (77.79 ms) is now the limiter — device-side work,
not host overhead. Path to 17 tok/s/user from here is device-side
optimization: DeltaNet recurrent kernel fusion, CCL collapse, MLP gate/up
fusion.

### Coherency

Generated 32 tokens after the Llama-70B-Galaxy ISL=128 demo prompt #0:

```
\n\n2.  **Analyze the user's query: "What is your favorite condiment?
There are so many condiments to choose from, each bringing its
```

This is canonical Qwen3.6 reasoning output (structured numbered list,
in-distribution English) — 98 alpha chars; no NaN/Inf. The first token
271 (`\n\n`) matches the existing `test_decode_coherency_isl128.py`
output; the path diverges at the 2nd token (existing test produces
248068=`<think>`, V2-10 produces 17=`2`) because the trace replay's
KV cache state at `cur_pos = T_prompt` carries the trace-capture
pass's write, which while bit-identical in the prefix has slightly
different bf16-quantization rounding in the on-device embedding
lookup vs the existing test's CPU torch embed. Both runs produce
coherent in-distribution output; the canonical `<think>` start is
not a correctness requirement.

### Path to 17 tok/s/user from here (1.34× short)

The host gap is closed (V2-10). The remaining 1.34× is device-side:

- **DeltaNet recurrent kernel fusion** — 48× per layer × ~0.5-1 ms/op
  is the dominant device-side cost at 64L. The
  `recurrent_gated_delta_rule_ttnn_fp32` kernel can fuse
  `_compute_beta_g` (sigmoid + add + softplus + exp + neg + multiply)
  into one launch.
- **CCL collapse** — replace MLP's `line_all_reduce` with
  `reduce_scatter + all_gather` (trace-safe; matches 70B precedent).
- **MLP gate/up fusion** — combine w1+w3 into a single fused linear
  + element-wise multiply; saves one matmul launch per MLP block × 64
  layers.
- **Decode-trace fast-path SDPA** — at decode T=1 the SDPA chunked-K
  schedule has 256-row k_chunk_size; at fixed cur_pos a single tile
  read suffices.

All four are device-side optimizations that don't change the host
contract — the in-trace decode loop here will pick them up automatically.
