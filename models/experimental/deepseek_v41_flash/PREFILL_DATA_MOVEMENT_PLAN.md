# DeepSeek V4.1 prefill: data-movement removal and fusion plan

**Goal:** cut the non-compute traffic in the prefill decoder layer: `to_layout`, `reshape`-through-ROW_MAJOR,
`slice` + `concat`, typecasts. Do it the way decode did: pick layouts so that ops chain without conversion, and fuse
where an op boundary forces a copy.

**Scope:** `models/experimental/deepseek_v41_flash/tt/prefill/*`, the V4 prefill base classes they inherit
(`models/experimental/deepseek_v4_flash/tt/prefill/*`), and these ops:
- `mix_streams`
- `fused_hyperconnection`
- `fused_partial_rope`
- `deepseek_v4_prefill_sdpa`
- a new prefill engram gate

**Running anything:** every `pytest` below opens the device. Per the repo rules, the agent hands the command to the
user and waits for them to run it. Only `./build_metal.sh` and host-side work run unattended. Kernel sources are
JIT-compiled; only host-side op changes (`*.cpp` outside `kernels/`, nanobind) need `./build_metal.sh`.

---

## Why decode's fused ops are not a drop-in

Every decode fused op takes its operands sharded in L1:

| Op | Decode-only constraint |
|---|---|
| `engram_gate` | width-sharded L1, ROW_MAJOR streams `[B,1,hc,D]` |
| `fused_partial_rope` | height- or width-sharded input |
| `LinearDecode*` with fused RMSNorm | width-sharded activations |
| `mix_streams` ROW_MAJOR streams | L1 only (32 B half-row reads; DRAM needs 64 B alignment) |

Prefill tensors are DRAM-interleaved, and at a 1024-token chunk they are far larger than L1. So each fusion below
needs either a DRAM-interleaved program in the existing op, or a layout change that makes the existing interleaved
program sufficient.

## Where the bytes go (estimate, chunk C = 1024, bf16, per TP4 rank)

The cost is dominated by one fact. **A TILE tensor `[1, T, hc, D]` pads `hc = 4` up to 32 rows per token**, so:
- The streams take 335 MB physically against 42 MB logically (8×).
- `collapsed` `[1, T, 1, D]` takes 335 MB against 10.5 MB (32×).

Every op that reads or writes these tensors moves the padding.

Estimated physical DRAM traffic per hyper-connection sublayer (there are two per layer, attention and FFN):

| Step | Today | After Phase 2 |
|---|---|---|
| `flatten_streams` (untilize, reshape, tilize) | ~460 MB | 0 |
| `wide_rms_norm` over `hc*D` (fp32 typecast, square, mean, multiply) | ~420 MB | ~85 MB (sum of squares only) |
| `fused_hyperconnection` reads streams, writes `collapsed` | ~670 MB | ~10 MB (no streams) |
| `_norm`: `rows_from_tokens` + `wide_rms_norm` + gamma | ~490 MB | ~21 MB (`ttnn.rms_norm`) |
| collapse (`Σ pre_j·s_j`) | inside the row above | ~52 MB |
| `mix_streams` + `tokens_from_rows` | ~710 MB | ~95 MB |
| **per sublayer** | **~2.8 GB** | **~0.26 GB** |

Other costs, per layer unless noted:
- **Engram (Engram layers only):** fp32 elementwise on padded streams, about 5 GB more on each Engram layer.
- **RoPE** (slice nope, slice rope, rotate, concat; on q, kv, the inverse on attn out, and the indexer): about 200 MB.
- **SDPA I/O** (q untilize, out tilize, `nlp_create_qkv_heads`, `nlp_concat_heads`, reshapes): about 130 MB.
- **Traced window** (`to_layout` + `concat` + 2 `slice_write` + `slice`): about 3 MB, but 5 dispatches.
- **`sel_in` clone, MoE `x_rm` untilize:** under 11 MB each.

These are bandwidth estimates. They assume TILE kernels touch the padded rows, which they do because padding is
allocated. **Phase 0 measures them; reorder the later phases if the profile disagrees.**

---

## Phase 0: baseline profile (no code change to the model)

### Task 0: per-op profile of one prefill layer at chunk 1024

**Files:**
- Create: `models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer_profile.py`

**What:**
- Build one decoder layer and run its device half only (no torch reference) at `T = 1024`.
- Run three layer kinds: a window layer, a ratio-4 layer, and an Engram layer.
- Warm up, then run once between `signpost("start")` / `signpost("stop")`, following the existing `DECODE_*`
  profiling docs.
- Reuse the parametrization and weight setup of `test_decoder_layer.py`. Do not wrap them; import the fixtures or
  copy the setup.

**Run (user):**

```bash
python -m tracy -p -r -v -m pytest 'models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer_profile.py' -x
```

**Acceptance:**
- A table of the top 25 ops by device time from `ops_perf_results_*.csv.gz`, grouped into compute, layout/copy and
  elementwise.
- The table is committed into this file under "Baseline" and becomes the yardstick for every task below.

---

## Phase 1: model-only cleanups (no kernel work)

### Task 1: `_norm` becomes one `ttnn.rms_norm`

**Files:**
- Modify: `models/experimental/deepseek_v4_flash/tt/prefill/decoder_layer.py` (`_norm`, lines 201-207)

**What:**
- Replace `wide_rms_norm(rows) * gamma` with `ttnn.rms_norm(rows, weight=gamma, epsilon=self.eps)`. The width is
  `D = 5120`, not `hc*D`.
- **Done.** The prefill head already runs `ttnn.rms_norm` on the same `[1,1,T,D]` input with the same
  `load_norm_gamma` weight (`tt/prefill/model.py:334`), so it fits at this width.
- The `wide_rms_norm` docstring says `rms_norm` overflows L1 at wide rows. The current layernorm program
  (`layernorm_op_multi_core.cpp`) falls back to a streaming `large_tensor` kernel sized to the available L1 span, so
  that note may be outdated. If it still fails, use the fallback below.
- Gamma has to be in the layout `rms_norm` wants (`[1,1,1,D]` TILE or ROW_MAJOR `[D/32, 32]`). Store it that way at
  load time; do not convert it per call.
- `rows_from_tokens` stays until Task 7 removes it.
- **Fallback:** keep `wide_rms_norm` and fold gamma into the final multiply (`x * (rstd ⊗ gamma)` is not one op), or
  pass a `compute_kernel_config` with `fp32_dest_acc_en` for the reduction.

**Run (user):**

```bash
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer.py -x
```

**Acceptance:**
- All five ids pass (window, window_engram, ratio2, ratio2_engram_chunked, ratio1) at the existing PCC 0.95 and
  pre-mix atol 0.015.
- In the Task 0 profile, `_norm` drops from about 7 ops to 1 (plus the 3 of `rows_from_tokens`).

### Task 2: the RMS fold in the hyper-connection

**Files:**
- Modify: `models/experimental/deepseek_v4_flash/tt/prefill/hyperconnection.py` (`forward`, 131-136)

**What:**
- `fn(x · rstd(x)) = rstd(x) · fn(x)`, because the norm is unweighted.
- Compute `fused_w = self.fn(flat)` on the raw flattened streams.
- Compute `rstd` from `ttnn.sum(ttnn.square(flat), dim=-1)` in fp32-accumulate (a `compute_kernel_config`, not a
  typecast of the 42 MB input).
- Then scale `fused_w` `[1,1,T,24]` by `rstd`.
- This removes the fp32 copy of the streams and the normed copy (about 340 MB per sublayer). `flatten_streams` is
  still needed until Phase 2.
- **Done.** Reductions default to `fp32_dest_acc_en`, and `fn` is a HiFi4 `Linear`.

**Run (user):**

```bash
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_hyperconnection.py -x
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer.py -x
```

**Acceptance:**
- `test_hyperconnection` at its existing PCC 0.999.
- If bf16 `fn` on the un-normed input loses precision there, run the `fn` matmul with
  `fp32_dest_acc_en=True, math_fidelity=HiFi4` before giving up the fold.

### Task 3: traced window buffer without the per-chunk concat

**Files:**
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/attention.py`:
  - traced `_window` (461-466)
  - `reset_static`
  - the seeding path that writes `bufs.tail`
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/model.py` (export, line 1351)

**Interface:**
- `bufs.window` `[1,1,sw+C,K]` ROW_MAJOR stays exactly what `deepseek_v4_prefill_sdpa` reads. The op takes one
  window tensor, not a separate tail and chunk.
- At each chunk:
  1. `ttnn.slice` rows `[C, C+sw)` of `bufs.window` and `slice_write` them into rows `[0, sw)`. This is the
     previous chunk's last `sw` rows.
  2. Untilize `kv` and `slice_write` it into rows `[sw, sw+C)`.
- `bufs.tail` is `None`, and the window concat goes away.
- The slice in step 1 is a copy, so overlapping rows (`C < sw`) are fine. No chunk-size restriction is needed.

**Done.** The changes:
- `reset_static` zeroes rows `[C, C+sw)` from a zero tensor allocated in `new_static_buffers`.
- `seed_prefill` (`tt/model.py`) writes the seeded ring into rows `[C, C+sw)`.
- Export always uses `rows(bufs.window, real, real+sw)`. For an unpadded last chunk, `real = C` gives
  `[C, C+sw)`.
- The bounded-replay `empty_tail` path is unchanged (the op skips the tail rows).

**Run (user):**

```bash
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_attention.py -x
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_full_model_prefill_verify.py -x   # galaxy, traced, multi-chunk
```

**Acceptance:**
- Both pass.
- The traced window step goes from 5 ops to 3: the shift, `to_layout`, and the write. The `to_layout` goes away with
  Task 11.

**Payoff is small** (about 3 MB, 2 dispatches per layer). Do it because it is cheap and it simplifies Task 11.

### Task 4: drop the `sel_in` clone

**Files:**
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/model.py` (1054-1056)
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/attention.py` (`_select`)

**What:**
- Three places free the selection:
  - `_select`, before replacing it.
  - The end-of-stage cleanup.
  - The `sel_out` send, which on a pass-through stage forwards `io.sel_in` itself.
- **Done:** `PrefillStaticShared.received_idx` holds the receive buffer, and `entry_idx` starts as it (read in place,
  like `candidates_in`). None of the three frees it.

**Acceptance:** `test_full_model_prefill_verify.py` passes.

---

## Phase 2: stream-major streams (the main win; kernel work)

> **Status: implemented. On device, `test_stream_mix_op`, `test_hyperconnection`, `test_engram` and
> `test_decoder_layer` pass.** What was built differs from Tasks 5 to 8 below:
>
> - **Task 5:** a separate op, `ttnn.experimental.deepseek.stream_mix(streams, weights, post=None, sublayer_out=None)`,
>   rather than a `stream_major=` flag on `mix_streams`. It takes `comb` / `post` / `pre` in the per-token tile
>   layout `fused_hyperconnection` already returns, so there is no host reshape. The reader turns each row block's
>   32 token weights into diagonal tiles, and the compute accumulates `diag(w) @ s_j` matmuls in fp32 dest.
>   Kernels live under `mix_streams/device/kernels/*/…stream_mix.cpp`. The test is `tests/prefill/test_stream_mix_op.py`.
> - **Task 6:** `ttnn.experimental.deepseek.fused_hyperconnection_weights(...)` returns `[post, comb, pre]`. In the
>   pre/post device op, `hidden_streams` is now optional, and without it the op skips the collapse.
>   `fused_hyperconnection` itself is unchanged.
> - **Task 7:** `tt/prefill/hyperconnection.py` (`DeepSeekV41PrefillHyperConnection`):
>   - `fn` is a batched per-stream matmul, summed in fp32, under the cache name `fn.prefill_sm`.
>   - The norm feeding the sublayers is a plain `ttnn.rms_norm` on `[1, 1, T, D]`.
>   - The Engram now works on dense `[1, hc, T, D]`: the key is a concat of `kv` column slices, and the gate weight
>     is cached as `gate_weight.prefill_sm`.
> - **Task 8:** not done. The traced hand-off still goes through ROW_MAJOR, now as `[1, hc, C, D]`.

### The layout

Carry the residual streams as **`[1, hc, T, D]` TILE**: stream `j` of all tokens is a contiguous `[T, D]` block.
- `T` and `D` are both multiples of 32, so **there is no padding**: 42 MB physical at C = 1024.
- A single stream is a dim-1 slice.

Everything the layer does to the streams becomes either a plain ttnn op or a column-broadcast multiply-add:

| Quantity | Formula | Implementation |
|---|---|---|
| `fn` projection | `fused_w[t] = Σ_j s_j[t] @ W_j` | batched matmul `[1,hc,T,D] @ [1,hc,D,24]`, then `sum(dim=1)`. No `flatten_streams`. |
| RMS over `hc*D` | `ss[t] = Σ_j Σ_d s_j[t,d]²` | `sum(square(S), dim=-1)` then `sum(dim=1)`, folded as in Task 2 |
| collapse | `x[t] = Σ_j pre_j[t] · s_j[t]` | stream mix with one output |
| residual mix | `out_i[t] = post_i[t]·o[t] + Σ_j comb_{j,i}[t] · s_j[t]` | stream mix with `hc` outputs |

The per-token scalars `pre`, `post` and `comb` become **column vectors over tokens**:
- `pre` `[1,hc,T,1]`
- `post` `[1,hc,T,1]`
- `comb` `[1,hc*hc,T,1]`, index `j*hc + i`

Every weight is then a `mul_tiles_bcast_cols` operand against a dense `[32-token, 32-col]` tile of a stream.

Shared base classes are untouched: V4 (non-.1) prefill keeps its layout. The new layout lives in V4.1 prefill classes.
The kernel changes are opt-in flags whose default is today's behaviour, so decode and V4 are unaffected.

### Task 5 (kernel): `mix_streams` stream-major mode

**Files:**
- Modify: `ttnn/cpp/ttnn/operations/experimental/deepseek/mix_streams/` (device op, program factory, reader,
  compute, writer, nanobind)
- Create: `tests/ttnn/unit_tests/operations/experimental/deepseek/test_mix_streams_stream_major.py`

**Interface:**

```python
out = ttnn.experimental.deepseek.mix_streams(
    post,            # [1, n_out, T, 1] TILE, or None (collapse)
    comb,            # [1, hc * n_out, T, 1] TILE, index j * n_out + i
    sublayer_out,    # [1, 1, T, D] TILE, or None (collapse)
    streams,         # [1, hc, T, D] TILE, DRAM interleaved
    stream_major=True,
)                    # -> [1, n_out, T, D] TILE, DRAM interleaved
```

**Program:**
- One work unit is an output tile `(i, row block r, col tile c)`.
- Reader: pulls `s_j[r, c]` for each `j`, `o[r, c]`, and the `hc + 1` weight tiles of row block `r`. The weights
  are reused across all 160 col tiles of the row; order the work so they stay in a CB.
- Compute: `acc = post_i ⊙bcast_col o`, then `acc += comb_{j,i} ⊙bcast_col s_j` for each `j`.
- Compute must accumulate in fp32 dest; the old op does, and the pre-mix atol 0.015 depends on it.
- `n_out = 1` with `post = sublayer_out = None` is the collapse. Both uses share one op because it is the same
  kernel.

**Test:** torch reference on random streams and weights:
- `T ∈ {32, 128, 1024}`, `n_out ∈ {1, hc}`.
- PCC ≥ 0.9999 against an fp32 reference, and max abs error comparable to the existing `mix_streams` at the same
  shapes.

**Run (user):**

```bash
./build_metal.sh
pytest tests/ttnn/unit_tests/operations/experimental/deepseek/test_mix_streams_stream_major.py -x
```

### Task 6 (kernel): `fused_hyperconnection` without streams

**Files:**
- Modify: `ttnn/cpp/ttnn/operations/experimental/deepseek/hyperconnection/` (`fused_hyperconnection.cpp`,
  `device/fused_pre_post_device_operation.cpp`, its writer, nanobind)
- Modify: `models/experimental/deepseek_v4_flash/tests/decode/test_fused_hyperconnection_op.py` (add a stream-major
  case)

**Interface:**

```python
pre, post, comb = ttnn.experimental.deepseek.fused_hyperconnection(
    None,                  # no streams: the collapse moved to mix_streams
    fused_w=fused_w,       # [1, 1, T, (2 + hc) * hc] TILE, already rstd-scaled (Task 2)
    ...,                   # biases, scales, sinkhorn_iters, eps as today
    stream_major=True,
)   # pre/post [1, hc, T, 1], comb [1, hc*hc, T, 1] TILE
```

- The sigmoid/sinkhorn compute is unchanged. Only the writer changes: it places token `t`'s scalars into row
  `t % 32` of column 0 of tile `(stream, t // 32)`, instead of one tile per token.
- **Fallback if the writer change is hard:** keep today's outputs and `ttnn.permute` them. They are `T` tiles each,
  about 2 MB, so this is cheap but adds 3 dispatches.
- **Optional:** take `fused_w` as the unsummed `[1,hc,T,24]` plus `ss` `[1,hc,T,1]`, and do the `sum(dim=1)` and
  the rstd scale in the op. That saves 3 to 4 small dispatches per sublayer.

**Test:** the op's own test with the new flag, against the same torch reference.

### Task 7: V4.1 prefill hyper-connection and decoder layer on stream-major streams

**Files:**
- Create: `models/experimental/deepseek_v41_flash/tt/prefill/hyperconnection.py`. Its class subclasses
  `DeepSeekV4PrefillHyperConnection`, loads `fn` as `[1,hc,D,24]` (one block per stream) under a new cache name, and
  has `forward(streams [1,hc,T,D], pre_mix [1,hc,T,1]) -> (post, comb, collapsed [1,1,T,D], pre)`.
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/decoder_layer.py` (`_v41_block`, construction of
  `attn_hc` / `ffn_hc`):
  - `_norm(collapsed)` is `ttnn.rms_norm` directly on the dense `[1,1,T,D]`. Both `rows_from_tokens` and
    `tokens_from_rows` are gone.
  - `mix_streams(..., stream_major=True)`.
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/engram.py`:
  - `forward(streams [1,hc,T,D], rows) -> [1,hc,T,D]`.
  - `_split_copies` is gone. `key_j` is the tile-aligned column slice `kv[..., j*D:(j+1)*D]` and `value` the last
    one.
  - The gate math runs per stream on `[1,1,T,D]` operands (`gate_weight` reshaped to `[1,hc,1,D]`), or as one pass
    on `[1,hc,T,D]` with `key` assembled by `ttnn.concat(dim=1)`. Pick whichever profiles faster; Task 10 replaces
    both.
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/model.py`:
  - `embed`: `repeat` the `[1,1,T,D]` TILE embedding along dim 1. This drops the ROW_MAJOR round trip.
  - `head` / `hc_head`: collapse with `pre` through the same `mix_streams` call.
  - `_select_row` / `_collapse`: a one-hot matmul per stream, `[1,1,1,T] @ [1,hc,T,D]`, then the collapse.
  - `io.one_hot` becomes a `[1,hc,T,1]` column.
- Modify tests:
  - The torch references in `test_hyperconnection.py`, `test_decoder_layer.py` and `test_engram.py` convert
    `[1,T,hc,D] <-> [1,hc,T,D]` on the host at the boundary only.
  - The comparisons and thresholds stay the same.

**Run (user):**

```bash
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_hyperconnection.py -x
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_engram.py -x
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer.py -x
```

**Acceptance:**
- Existing thresholds: hyper-connection PCC 0.999, Engram PCC 0.995 / delta 0.98, layer PCC 0.95 / pre-mix atol
  0.015.
- In the Task 0 profile, no `to_layout`, `reshape`, `untilize` or `tilize` touches a stream tensor.

### Task 8: traced stage hand-off in stream-major TILE

**Files:**
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/model.py`:
  - `io.streams_in` / `io.pre_in` allocation
  - receive (1046-1050)
  - send (1092-1095)
  - `_PRE_MIX_ROW` packing

**What:**
- Send and receive the `[1,hc,C,D]` streams and the `[1,hc,C,1]` `pre` as TILE tensors directly. Delete both
  `to_layout`s on each side, plus the pre-mix reshape.
- **To verify first:** that `send_direct_async` / `recv` handle TILE pages. They copy buffer pages, and a dense TILE
  tensor is one contiguous page set.
- If they do not, the fallback is to send the stream-major tensor as ROW_MAJOR. That costs a tilize/untilize of
  42 MB instead of 335 MB.

**Run (user):**

```bash
pytest models/experimental/deepseek_v41_flash/tests/prefill/test_full_model_prefill_verify.py -x
```

**Acceptance:** full-model verify passes. The hand-off copies are gone from the stage profile.

---

## Phase 3: fused ops on the remaining hot paths (kernel work)

### Task 9 (kernel): prefill partial RoPE in place

**Problem:**
- `_rope` (V4 `attention.py` 492-503) does: slice nope, slice rope, `rotary_embedding_llama`, `concat`. It runs on
  q, kv, the inverse on attn out, and the indexer q/k.
- Making the 448 nope columns an identity rotation does not work for the main heads: `rotary_embedding_llama` caps
  `head_dim` at 256 (`rotary_embedding_llama_device_operation.cpp:70`).
- It does work for the 128-wide indexer heads: use cos = 1, sin = 0 tables on the first 64 columns.
- So the indexer part can be done model-side now (one `[T,128]` table, the last 128 columns of a `[T,512]` table).
  The main heads need the kernel.

**Files:**
- Modify: `ttnn/cpp/ttnn/operations/experimental/transformer/fused_partial_rope/` (add a DRAM-interleaved program)
- Modify: `tests/ttnn/unit_tests/operations/experimental/transformer/test_fused_partial_rope.py`

**Interface:**

```python
ttnn.experimental.fused_partial_rope(x, cos, sin, trans_mat, rope_dim=64, output_tensor=x)  # in place
```

- `x`: `[1,H,T,Dh]` TILE, DRAM interleaved, with `Dh ∈ {128, 512}`.
- `cos`/`sin`: `[1,1,T,rope_dim]` TILE.
- Only the last `rope_dim / 32 = 2` tile columns of each row block are read and written. That is 1/8 of the
  main-head data, and no slice or concat.

**Then (model):** `_rope` becomes one call. The table gather in `_step_inputs` stays.

**Acceptance:**
- The op test passes: bit-equal to the slice/rope/concat path for bf16.
- `test_attention.py` passes.

### Task 10 (kernel): prefill Engram gate

**Files:**
- Create: `ttnn/cpp/ttnn/operations/experimental/deepseek/engram_gate_prefill/` (or a DRAM-interleaved program in
  `engram_gate`)
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/engram.py`
- Create: `models/experimental/deepseek_v41_flash/tests/prefill/test_engram_gate_prefill_op.py`

**Interface:**

```python
streams_out = ttnn.experimental.deepseek.engram_gate_prefill(
    streams,      # [1, hc, T, D] bf16 TILE
    kv,           # [1, 1, T, (hc + 1) * D] TILE, straight from wkv
    gate_weight,  # [1, hc, 1, D] fp32
    eps=...,
    clamp=1e-6,
)                 # -> [1, hc, T, D] bf16 TILE
```

**Program:** per `(stream j, row block r)`, it makes two passes over the 160 col tiles:
1. Accumulate three row reductions: `h·g·k`, `h²` and `k²`.
2. Write `h + gate · v`.

All three reductions accumulate in fp32 dest; there is no fp32 copy of any operand. This replaces about 18
elementwise dispatches and all fp32 typecasts.

**Acceptance:** op test against the torch `Engram` reference, then `test_engram.py` at PCC 0.995 / delta 0.98.

### Task 11 (kernel): `deepseek_v4_prefill_sdpa` TILE / group-major I/O

**Files:**
- Modify: `ttnn/cpp/ttnn/operations/transformer/sdpa/deepseek_v4_prefill_sdpa*` (reader for TILE q; writer for a
  group-major TILE output)
- Modify: `tests/ttnn/unit_tests/operations/sdpa/test_deepseek_v4_prefill_sdpa.py`
- Modify: `models/experimental/deepseek_v41_flash/tt/prefill/attention.py` (497, 512), V4 `_q_stem` (508-528),
  `_o_proj` (957-985)

**Interface:**
- `q`: accepts TILE `[1,H,T,K]`. This drops `to_layout(q, ROW_MAJOR)` (line 497).
- Optional `output_layout=TILE`. This drops `to_layout(out, TILE)` (line 512).
- **Stretch goal:** `output_group_major=g` writes `[1,g,T,(H/g)*K]` directly, the layout the grouped `o_a` matmul
  consumes. That drops the reshape / `nlp_concat_heads` / reshape in `_o_proj`.
- The window and cache inputs stay ROW_MAJOR (Task 3 already writes them that way).
- **Prerequisite:** confirm `o_groups` and `o_lora_rank` from the checkpoint config before designing the
  group-major output.

**Acceptance:** the SDPA unit test (all existing parametrizations plus TILE variants), then `test_attention.py`.

---

## Not planned (and why)

- **MoE `x_rm` untilize (`moe.py` 234-236):** `fused_experts_prefill` gathers token rows itself and needs
  ROW_MAJOR DRAM. Gathering rows out of tiles means 32 B face-row reads against DRAM's 64 B alignment. That is about
  10 MB per layer; not worth a gather rewrite.
- **`o_proj` g-slices + concat:** about 8 MB. Revisit only if Task 11's group-major output lands and the profile
  still shows it.
- **`_window_view`:** 3 layers only.
- **Head `_collapse` / `_select_row` round trip:** once per chunk on the last stage. Task 7 removes it as a side
  effect.

## Order and dependencies

```
Task 0 ── Task 1 ── Task 2 ─────────────── Task 7 ── Task 8
                   Task 5 ──┐            ↗
                   Task 6 ──┴────────────┘
Task 3, Task 4 (independent)
Task 9, Task 10 (after Task 7), Task 11 (after Task 3)
```

- Tasks 1 to 4 can land one by one with no kernel work.
- Tasks 5 and 6 can be built in parallel with Phase 1.
- Task 7 is the switch-over. It should land in one change with its tests, because a half-converted layer has
  conversions at every boundary and is slower than today.

## Baseline

TP4 1×4 submesh, chunk 1024, Phase 1 applied. Times are device kernel time summed over one layer's ops, taking the
slowest of the 4 chips per op. Source: `generated/profiler/reports/2026_11_10_19_09_22/`.

| Layer | Ops | Total | Streams layout* | MoE (`fused_experts_prefill`) | Engram gate |
|---|---|---|---|---|---|
| 0 window | 102 | 36.8 ms | 20.4 ms (55%) | 10.9 ms | - |
| 1 window + Engram | 142 | 110.6 ms | 20.4 ms | 10.0 ms | **~73 ms** |
| 2 ratio-2 source | 148 | 37.9 ms | 20.4 ms | 10.0 ms | - |
| 20 ratio-1 source | 130 | 42.7 ms | 20.4 ms | 12.2 ms | - |

\* Per layer, the streams-layout row adds up:
- `mix_streams` ×2: 10.1 ms, about 5 ms each.
- `fused_pre_post` ×2: 3.6 ms.
- Untilize of a padded `[1,1024,32,5120]` ×4: 4.0 ms. Two come from `flatten_streams`, two from `rows_from_tokens`.
- Smaller items: the flatten tilize and reshape, the `fn` matmul, and the RMS square and mean, about 2.4 ms.

Attention is small by comparison: SDPA 0.7-1.8 ms, the indexer up to 2.6 ms, and the RoPE slice/concat about 0.6 ms.

**Engram layers (1 and 14) are the pipeline bottleneck.** They sit on stages 0 and 2, so those stages take about
75 ms more per chunk than the others. The gate runs 39 ops on padded fp32 `[1,1024,32,5120]` tensors:
- BinaryNg: about 5 ms each.
- Typecast: about 2.9 ms each.
- Each `ttnn.mean`: a Clone (3.6 ms) plus a FillPad (3.6 ms) before the reduce.
- `_split_copies` tilize-with-padding: 3.3 ms.

This moves Task 10 ahead, and it has a model-only first step: Task 7's Engram change, on dense rows.

## After Phase 2 (stream-major streams and the dense Engram)

Same setup and method. Source: `generated/profiler/reports/2026_11_10_19_43_44/`.

| Layer | Ops | Total (baseline) | Total now | MoE | Engram |
|---|---|---|---|---|---|
| 0 window | 118 | 36.8 ms | **19.1 ms** | 10.9 ms | - |
| 1 window + Engram | 151 | 110.6 ms | **26.7 ms** | 10.0 ms | ~6.7 ms (`wkv` matmul 2.5 ms) |
| 2 ratio-2 source | 164 | 37.9 ms | **20.2 ms** | 10.0 ms | - |
| 20 ratio-1 source | 146 | 42.7 ms | **24.8 ms** | 12.2 ms | - |

- Streams layout: about 20 ms per layer down to about 2.5 ms.
  - `stream_mix`: 0.16 ms per collapse and 0.32 ms per residual mix, so 1.0 ms for the four calls. The old
    `mix_streams` took 10.1 ms for two.
  - Each hyper-connection: `fn` matmul 0.3 ms, `square` 0.22 ms, reduce 0.13 ms, `fused_pre_post` 0.05 ms,
    sinkhorn 0.06 ms.
- Engram gate: about 73 ms down to about 4.2 ms.
  - It is now seven fp32 BinaryNg ops (0.5-0.65 ms each), three fp32 reduces (0.24 ms each) and four typecasts
    (0.33-0.36 ms each).
  - Task 10's fused gate would remove most of it, but this is no longer the bottleneck.
- MoE is now about half of every layer.
