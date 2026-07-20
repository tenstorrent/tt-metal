# `return_lse` output for `ttnn.transformer.scaled_dot_product_attention` + `chunked_scaled_dot_product_attention`

> **STATUS 2026-07-19: IMPLEMENTED + DEVICE-VERIFIED on QB2.** `test_return_lse.py` **6/6**:
> `return_lse=False` byte-identical (safety) + emitted LSE ≈ `torch.logsumexp` (abs-error gate,
> PCC≥0.98 sanity — PCC is misleading on the low-variance noncausal LSE clustered at ~log(#keys)).
> Two bugs found+fixed during bringup vs the original plan below: (1) `sdpa_inner_loop_step` also
> needed the `emit_lse/cb_lse_out/cb_scale_in` template params (its `normalize_row` lambda forwards
> them) + forwarding at the `sdpa_standard_v2` call site; (2) the LSE emit must **col-reduce**
> `cur_sum_cb` via `matmul(sum, col_identity)` to get the scalar `l` BEFORE `log` — reading the raw
> front tile only captures one column's partial sum and loses `log(#keys)`. Build: `ninja -C build
> ttnn` (`build`→`build_Release`), then sync `build/ttnn/_ttnn.so`→`ttnn/ttnn/_ttnn.so`; the compute
> kernel is JIT (recompiles on next device run). Original design (accurate) preserved below.

**Task:** DiffusionGemma paged-prefix task **T6**. Expose the per-row log-sum-exp (LSE) statistic from the *plain* and *chunked* SDPA compute path as an optional second output tensor, so a later host/device step can merge attention over KV partials (paged prefix + new chunk) using the standard flash-attention rescale identity.

**Hard constraint:** `return_lse=False` must be **byte-identical** to today for every existing caller (gemma4 backbone, decode, prefill). No C++ that changes the default program. This is a **design artifact only** — no build in this session.

**Core fact being exploited:** the LSE is *already computed* inside the shared flash compute. The running max `m` and running sum `l` are the softmax accumulators that live at the end of the K-loop; the ring-joint eager path already emits `LSE = scale·m + log(l)` to a stats tensor. We reuse that exact op sequence on the non-ring streaming path.

---

## 0. The LSE identity, and where it already exists

The ring-joint compute kernel produces LSE with three ops (this is the template):

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:1928-1988` (inside `sdpa_inner_loop`, `sdpa_type == RING` branch):

```cpp
// compute_common.hpp:1929-1933  (ring_iter == 0 seeds LSE; later iters merge via logsigmoid)
log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);          // cur_max_scratch = log(l)
mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);   // m := scale * m
add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);  // m := scale*m + log(l)
...
// compute_common.hpp:1986-1987  (ring_iter == 0)
copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);            // LSE tile-block -> output CB
```

So, per row:  **`LSE_row = scale · m_raw_row + log(l_row)`**

where `m_raw_row` is the **raw** (un-scaled) per-row max of `Q·Kᵀ`, and `l_row = Σ_k exp((qkᵀ − m_raw)·scale)` is the running softmax denominator. This is exactly `logsumexp_k(scale · Q·Kᵀ)`.

**Scale convention is identical on the streaming path** — confirmed:
- The max reduce uses the *identity* scaler `cb_identity_scale_in` (=1.0), so the stored max is raw: `reduce_c_row_group<in0_cb, cb_identity_scale_in, ...>` (`compute_streaming.hpp:400-462`, called at `:1470`).
- The scale is folded into the **exp**, not the max: `exp_packthread_tile_init<true, scale_fp32, InputClamping::None>()` (`compute_streaming.hpp:1284`), applied in `sub_exp_block_bcast_cols` (`:468-557`).

Therefore the streaming accumulators hold the same `m_raw` / `l` as the ring path, and the same `scale·m + log(l)` formula applies verbatim. `log_tile`/`log_tile_init` are available and already used by `log_block` (`compute_common.hpp:801-819`); `add_binary_tile` is already used inside `normalize_row_streaming` (`compute_streaming.hpp:740`).

---

## 1. `compute_streaming.hpp` — where `m` and `l` live, and how to emit `lse`

### 1a. Location of the accumulators at end of K-loop

`sdpa_standard_v2` (`compute_streaming.hpp:1919-2174`) is the plain/chunked streaming compute driver. Per Q chunk it ping-pongs two accumulator halves (`:1962-1963`):

```cpp
AccumulatorHalf prev = {cb_sum_A, cb_max_A, cb_out_im_A};
AccumulatorHalf cur  = {cb_sum_B, cb_max_B, cb_out_im_B};
```

The K-loop (`:2084-2171`) calls `sdpa_inner_loop_step` per K chunk. On the **last** K chunk (`is_last_iter`), the `normalize_row` lambda fires (`compute_streaming.hpp:1659-1677`):

```cpp
auto normalize_row = [&](uint32_t& pushed, uint32_t sbh) {
    CircularBuffer(cur.sum).push_back(sbh);            // <-- cur.sum row-group made visible
    CircularBuffer(out_cb).push_back(sbh * vDHt);
    normalize_row_streaming<...>(cur.sum, out_cb, sbh, cur.max, sink_row_offset);  // consumes cur.sum
    ...
};
```

At this point, for the row-group being normalized:
- **`cur.max`** (= `cb_max_B`/`cb_max_A`): raw per-row max, **column-0 valid**, `Sq_chunk_t` tiles, kept fronted until popped at `compute_streaming.hpp:2167` (`sdpa_cb_pop_front_out_of_line(cur.max, Sq_chunk_t)`).
- **`cur.sum`** (= `cb_sum_B`): per-row `l`, column-0 valid; **consumed (pop_front) row-by-row inside `normalize_row_streaming`** at `compute_streaming.hpp:757`.

> The reciprocal that `normalize_row_streaming` computes already uses `l` (`matmul_block(cur_sum, col_identity)` → `recip_tile`, `:730-748`). The LSE needs `log(l)` **before** that pop. So the emit must be co-located with the recip step.

### 1b. Minimal change — emit LSE inside `normalize_row_streaming`, mirroring the attention-sink plumbing

`normalize_row_streaming` (`compute_streaming.hpp:699-801`) **already** takes an optional per-row read into `cur.max` at an absolute base offset for attention-sink (`cur_max_cb_rt`, `sink_row_offset`; waits at `:711`, reads at `:734`). We reuse that exact pattern for LSE: add compile-time `emit_lse` + `cb_lse_out`, and a runtime `lse_row_offset` that advances by `sbh` per call (identical cadence to `sink_row_offset` at `:1674`).

**Diff sketch (template signature, `compute_streaming.hpp:689-704`):**
```cpp
// BEFORE
template <bool profiling_enabled, uint32_t head_dim_t_, uint32_t dst_size,
          uint32_t col_identity_cb, uint32_t scratch_cb, uint32_t normalized_out_cb,
          uint32_t scale_fp32 = 0, bool use_attention_sink = false,
          uint32_t cb_attention_sink = INVALID_CB>
static ... void normalize_row_streaming(
    uint32_t cur_sum_cb, uint32_t cur_out_cb, uint32_t sbh,
    [[maybe_unused]] uint32_t cur_max_cb_rt = 0,
    [[maybe_unused]] uint32_t sink_row_offset = 0) {

// AFTER  (two new template params + two new runtime params, all default-off)
template <bool profiling_enabled, uint32_t head_dim_t_, uint32_t dst_size,
          uint32_t col_identity_cb, uint32_t scratch_cb, uint32_t normalized_out_cb,
          uint32_t scale_fp32 = 0, bool use_attention_sink = false,
          uint32_t cb_attention_sink = INVALID_CB,
          bool emit_lse = false, uint32_t cb_lse_out = INVALID_CB, uint32_t cb_scale_in = INVALID_CB>
static ... void normalize_row_streaming(
    uint32_t cur_sum_cb, uint32_t cur_out_cb, uint32_t sbh,
    [[maybe_unused]] uint32_t cur_max_cb_rt = 0,
    [[maybe_unused]] uint32_t sink_row_offset = 0,
    [[maybe_unused]] uint32_t lse_row_offset = 0) {
    if constexpr (emit_lse) {
        // cur.max kept fronted (Sq_chunk_t) by caller; make sure this group's rows are visible.
        CircularBuffer(cur_max_cb_rt).wait_front(lse_row_offset + sbh);
    }
```

**Emit block — inside the per-row loop `for (uint32_t s = 0; s < sbh; s++)` (`compute_streaming.hpp:714`), inserted *before* the recip step at `:715-758` (which pops `cur_sum_cb`):**
```cpp
if constexpr (emit_lse) {
    // LSE_row = scale * m_raw + log(l).  cur_sum_cb front tile == l for this row (col 0);
    // cur_max_cb_rt[lse_row_offset + s] == m_raw (col 0). Both read non-destructively.
    CircularBuffer(cb_lse_out).reserve_back(1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cur_sum_cb);
    copy_tile(cur_sum_cb, 0, 0);                 // DST0 = l         (front tile, not yet popped)
    log_tile_init();
    log_tile(0);                                 // DST0 = log(l)
    copy_tile_to_dst_init_short(cur_max_cb_rt);
    copy_tile(cur_max_cb_rt, lse_row_offset + s, 1);   // DST1 = m_raw
    // DST1 *= scale  (bcast scalar, same op the ring path uses: mul_block_bcast_scalar_inplace<cb_scale_in>)
    mul_tiles_bcast_scalar_init_short(/*in0*/1_dst, cb_scale_in);   // see note on scalar mul below
    // DST0 = log(l) + scale*m
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_lse_out);                    // fp32 pack (cb_lse_out is Float32)
    tile_regs_release();
    CircularBuffer(cb_lse_out).push_back(1);
}
```
Then the existing recip block runs and pops `cur_sum_cb` row `s` as today. `cur.max` is untouched (still popped once by the caller at `compute_streaming.hpp:2167`).

**Scalar-mul-by-`scale` — two equally valid options:**
1. **Reuse the ring mechanism (recommended):** allocate a 1-tile `cb_scale_in` holding `scale` (bcast scalar), exactly like ring; multiply `m` by it. This is the literal `mul_block_bcast_scalar_inplace<cb_scale_in>` primitive at row granularity. Requires the writer to `generate_bcast_scalar(cb_scale_in, scale_packed)` (the writer already receives `scale_val` as CTA 13, `writer_interleaved.cpp:31`).
2. **Fold `scale` into the log with no new CB:** `scale·m + log(l) = log(l · e^{scale·m})` is *not* cheaper. Instead precompute `log(l)` then use a unary SFPU `binop_with_scalar`/`mul_unary_tile` with the compile-time constant `scale_fp32` on DST1. Avoids the scalar CB entirely but depends on a scalar-constant SFPU mul being available on both WH/BH; option 1 is the proven path and is preferred for a first landing.

**Wire the flag through the driver.** `sdpa_standard_v2` (`compute_streaming.hpp:1919`) gains a compile-time `bool emit_lse` + `uint32_t cb_lse_out` + `uint32_t cb_scale_in`, and the `normalize_row` lambda (`:1659`) passes them plus a `lse_row_offset` counter incremented by `sbh` after each call (clone of the `sink_row_offset` logic at `:1673-1675`). `sdpa.cpp` compute kernel (`sdpa.cpp:156-195`) forwards a new compile-time arg (see §3). **When `emit_lse==false` the added `if constexpr` blocks compile to nothing → byte-identical.**

---

## 2. `ring_joint_sdpa.cpp` — the reference we are mirroring

`ring_joint_sdpa.cpp` does not itself compute LSE; it *routes* the aliased stats CB into `sdpa_inner_loop` (the RING branch shown in §0). The relevant wiring:

- `ring_joint_sdpa.cpp:139-140`: `cb_max_in` is aliased — `constexpr uint32_t cb_lse_in = cb_max_in;  // eager norm: LSE`.
- `ring_joint_sdpa.cpp:149-150`: `cb_max_out` is aliased — `constexpr uint32_t cb_lse_out = cb_max_out;  // eager norm: LSE`.
- `ring_joint_sdpa.cpp:406-407`: `cb_lse_in` / `cb_lse_out` are passed into `sdpa_inner_loop`.

**Key architectural note:** the ring path *reuses the running-max CB as the LSE CB* because ring normalization is *eager* (LSE overwrites max in place each iteration). Our plain path is *deferred* (max is consumed only at the end), and gemma4 callers must stay byte-identical, so we **cannot alias `cb_max_*`** — an in-place LSE would corrupt the max still needed for normalization. We therefore introduce a **dedicated** `cb_lse_out` CB (see §6). The op sequence is copied from the ring branch (§0); only the destination CB differs.

The writer side of the ring template is `write_output_and_lse` (`ring_joint_writer.cpp:306-338`) — that is the DRAM-drain pattern we copy in §4.

---

## 3. `sdpa.cpp` (compute kernel) — gain the guarded LSE emit

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp` is the plain/chunked compute kernel. It reads compile-time args 0-33 (`:16-53`) then CBs at `cb_arg_offset = 34` (`:74-92`).

**Diff sketch:**
1. Add one scalar compile-time flag after arg 33:
```cpp
// sdpa.cpp:53  (after use_zigzag_balancing = get_compile_time_arg_val(33))
constexpr bool return_lse = get_compile_time_arg_val(34) == 1;   // NEW arg 34
```
   (Bump `cb_arg_offset` to 35, and add `cb_lse_out` + `cb_scale_in` to the CB-id block appended by `CBIds::compute_compile_time_args()` — see §6. Read them as `get_compile_time_arg_val(cb_arg_offset + 18)` / `+ 19`.)

2. Forward into the streaming driver (`sdpa.cpp:156-195`), adding the three template params + the CB ids:
```cpp
sdpa_standard_v2<
    ..., use_provided_mask,
    /*emit_lse=*/return_lse>(          // NEW trailing compile-time template param
    ..., lw_mask, q_num_chunks, use_zigzag_balancing,
    cb_lse_out, cb_scale_in);          // NEW runtime CB ids (or template — see §1b)
```

3. **The non-streaming fallback path (`sdpa.cpp:196-275`, `sdpa_standard`) is out of scope.** gemma4 chunked prefill uses the streaming path (`use_streaming_compute==true`). Guard with a host-side `TT_FATAL(!return_lse || use_streaming_compute, ...)` in the program factory so `return_lse` is only accepted where implemented. (Streaming is the default for the shapes gemma4 uses; extending the deferred `sdpa_standard`/`sdpa_inner_loop` non-RING branch is a follow-up.)

When `return_lse==false`, the emit `if constexpr` in §1 is discarded and the extra CBs are never allocated (§6) → **existing programs unchanged**.

---

## 4. `writer_interleaved.cpp` — drain LSE tiles to the new output tensor

`writer_interleaved.cpp` currently drains only `cb_out` (streaming: `write_block_row_grouped`, `:199-209`). Mirror `ring_joint_writer.cpp:322-337`.

**Diff sketch:**
1. New compile-time flag + a second TensorAccessor chained after the existing `out_args`/`cu_window_args` (`writer_interleaved.cpp:48-49`):
```cpp
constexpr bool return_lse = get_compile_time_arg_val(26) == 1;    // NEW (shift windowed flag to 27, or append at tail)
constexpr auto out_args      = TensorAccessorArgs<27>();
constexpr auto cu_window_args= TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
constexpr auto lse_args      = TensorAccessorArgs<cu_window_args.next_compile_time_args_offset()>();  // NEW
```
   Add `cb_lse_out` to the CB-id block (`:74-80`).

2. New runtime arg for the LSE DRAM base address (after `cu_window_seqlens_eles` at `:68`):
```cpp
const uint32_t lse_addr = get_arg_val<uint32_t>(12);              // NEW
const auto lse_writer = TensorAccessor(lse_args, lse_addr);
const auto lse_tile_shape = TensorTileShape(B, NQH, valid_Sqt, 1);   // [.,.,S,1]
constexpr uint32_t lse_tile_bytes = get_tile_size(cb_lse_out);
```

3. After the `cb_out` drain for each Q chunk (`writer_interleaved.cpp:195-221`), drain the matching `Sq_chunk_t` LSE tiles (guarded, copied from `ring_joint_writer.cpp:324-337`):
```cpp
if constexpr (return_lse) {
    CircularBuffer cb_lse(cb_lse_out);
    cb_lse.wait_front(out_row_tile_count);     // compute pushes 1 LSE tile per real Q row-tile
    uint32_t rd = cb_lse.get_read_ptr();
    for (uint32_t i = 0; i < out_row_tile_count; ++i) {
        noc.async_write(CoreLocalMem<uint32_t>(rd), lse_writer, lse_tile_bytes,
                        {}, {.page_id = lse_tile_shape.id_of(nb, nq, write_offset + out_row_start_tile + i, 0)});
        rd += lse_tile_bytes;
    }
    noc.async_writes_flushed();
    cb_lse.pop_front(out_row_tile_count);
}
```
   Handle the padded-row tail the same way `write_block_row_grouped` does (`out_row_tile_count` vs `Sq_chunk_t`): compute pushes `Sq_chunk_t` LSE tiles per group, so pop `Sq_chunk_t` but only write `out_row_tile_count`. Match the exact push/pop count the compute side commits (align with §1b `sbh` cadence).

---

## 5. `sdpa_device_operation_types.hpp` + `sdpa_device_operation.cpp` — attribute + optional 2nd output

### 5a. Attribute (`sdpa_device_operation_types.hpp:14-28`)
```cpp
struct SDPAParams {
    ...
    bool is_windowed = false;
    bool return_lse = false;   // NEW — participates in the program-cache key
};
```

### 5b. Return types — mirror JointSDPA (same `device_operation::launch` framework)

JointSDPA already returns a variable-length vector under this exact framework (`joint_sdpa_device_operation.hpp:22-23`, `joint_sdpa_device_operation_types.hpp:45` → `using JointSDPAResultSpec = std::vector<TensorSpec>`). Adopt the same for SDPA to carry an optional second output.

**Diff sketch (`sdpa_device_operation.hpp:21-22`):**
```cpp
// BEFORE
using spec_return_value_t   = TensorSpec;
using tensor_return_value_t = Tensor;
// AFTER
using spec_return_value_t   = std::vector<TensorSpec>;   // [out]  or  [out, lse]
using tensor_return_value_t = std::vector<Tensor>;
```

**`compute_output_specs` (`sdpa_device_operation.cpp:425-432`):**
```cpp
SDPAOperation::spec_return_value_t SDPAOperation::compute_output_specs(const SDPAParams& attrs, const SDPAInputs& tensors) {
    auto shape = tensors.q.logical_shape();
    if (attrs.use_mla) shape[3] = attrs.head_dim_v.value_or(shape[3]);
    std::vector<TensorSpec> specs;
    specs.emplace_back(shape, TensorLayout(tensors.q.dtype(), PageConfig(Layout::TILE), attrs.output_mem_config));  // out (unchanged dtype/shape)
    if (attrs.return_lse) {
        auto lse_shape = tensors.q.logical_shape();   // [B, NQH, Sq, DH]
        lse_shape[3] = 1;                             // -> [B, NQH, Sq, 1]
        specs.emplace_back(lse_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config));
    }
    return specs;
}
```

**`create_output_tensors` (`sdpa_device_operation.cpp:434-437`):**
```cpp
SDPAOperation::tensor_return_value_t SDPAOperation::create_output_tensors(const SDPAParams& attrs, const SDPAInputs& tensors) {
    std::vector<Tensor> outs;
    for (const auto& spec : compute_output_specs(attrs, tensors))
        outs.push_back(create_device_tensor(spec, tensors.q.device()));
    return outs;
}
```
`create_op_performance_model` (`:439-516`) takes `tensor_return_value_t& output_tensor` → change to reference the vector and use `output_tensor[0]`.

**Free function churn (important for byte-identity):** `ttnn::prim::sdpa(...)` (`sdpa_device_operation.cpp:521-565`) currently returns `Tensor`. Two moves:
- Keep the existing free function returning `Tensor` by `return launch<...>(...)[0];` and set `.return_lse = false` — **every existing internal caller (flash_mla, ring_distributed, chunked, decode) is untouched and gets exactly one output tensor from an unchanged program.**
- Add a sibling `std::vector<Tensor> ttnn::prim::sdpa_with_lse(...)` (same args + `.return_lse=true`) returning the 2-element vector, used only by the new top-level path (§7).

The program factory (§6) reads `tensor_return_value[0]` as the output and `tensor_return_value[1]` (when present) as the LSE tensor.

---

## 6. `sdpa_program_factory.cpp` — CB + output buffer + compile arg + writer wiring

Entry `SDPAProgramFactory::create_descriptor` (`sdpa_program_factory.cpp:155-1413`). `output_tensor` is `tensor_return_value` (`:166`); becomes `tensor_return_value[0]`.

**(a) LSE output buffer:** near `out0_buffer` (`:323`):
```cpp
auto* out0_buffer = tensor_return_value[0].buffer();
auto* lse_buffer  = attrs.return_lse ? tensor_return_value[1].buffer() : nullptr;   // NEW
```

**(b) New CBs.** Add ids to `CBIds` (`sdpa_interleaved_cb_ids.hpp:13-36`): `uint32_t lse_out = inactive;` and `uint32_t scale_in = inactive;`. Append both to `writer_compile_time_args()` (`:42-44`) and `lse_out`+`scale_in` to `compute_compile_time_args()` (`:46-66`) (keep order stable — append at the **end** so existing indices are unchanged).

Allocate alongside the stats CBs (`sdpa_program_factory.cpp:760-765`). `statistics_tiles == Sq_chunk_t` (`:401`):
```cpp
if (attrs.return_lse) {
    tt::DataFormat lse_df = tt::DataFormat::Float32;                 // fp32 output
    cb_ids.lse_out  = allocate_tile_cb(statistics_tiles, tt::tile_size(lse_df), lse_df);   // Sq_chunk_t fp32 tiles
    cb_ids.scale_in = allocate_tile_cb(1, scalar_tile_size, scalar_df);                    // 1-tile bcast scalar (§1b opt.1)
}
```
> Size note: matching the drain granularity, `cb_lse_out` needs at least the per-group `sbh` tiles live; sizing it to `statistics_tiles` (= `Sq_chunk_t`) is safe and small. For gemma4 chunked prefill `Sq_chunk_t` is 1 tile (q_chunk 32 → 1 tile), so `cb_lse_out` ≈ 4 KB fp32, `cb_scale_in` ≈ 2-4 KB. Negligible vs the head_dim-512 intermediates.

**(c) Compute compile-time arg** — append the `return_lse` scalar to `compute_compile_time_args` (`:591-627`, currently ends at arg 33) as **arg 34**, before the CB-id block is appended (`:774-776`):
```cpp
compute_compile_time_args.push_back(static_cast<uint32_t>(attrs.return_lse));   // arg 34
```

**(d) Writer compile-time arg + accessor + scalar.** Append `return_lse` to `writer_compile_time_args` (`:555-583`), then append the **LSE TensorAccessorArgs** after the existing out/cu_window accessors (`:587-589`):
```cpp
writer_compile_time_args.push_back(static_cast<uint32_t>(attrs.return_lse));            // new scalar flag
TensorAccessorArgs(attrs.return_lse ? lse_buffer : nullptr).append_to(writer_compile_time_args);   // lse_args
```
The writer already has `scale_val` (CTA 13) → in `writer_interleaved.cpp` add `generate_bcast_scalar(CircularBuffer(cb_scale_in), scale_val);` next to the existing `generate_bcast_col_scalar` (`:95`) under `if constexpr (return_lse)`.

**(e) Writer runtime arg.** Append `lse_buffer` to the writer RT arg list (`:1382-1395`, currently ends with `cu_window_seqlens_eles`) as the new arg 12:
```cpp
writer_desc.emplace_runtime_args(core, { out0_buffer, i, num_phases, ..., cu_window_buffer, cu_window_seqlens_eles,
                                         lse_buffer /* arg 12; nullptr when !return_lse */ });
```
Append the LSE buffer to `compute` accessor args too only if the compute kernel reads it via accessor — it does **not** (compute only packs to the CB; the writer owns DRAM), so no compute RT arg needed. The existing `TensorAccessorArgs(output_tensor.buffer()).append_to(compute_compile_time_args)` at `:776` stays as-is (`tensor_return_value[0]`).

**(f) Grid / CB budget — head_dim 512.** gemma4's denoise/prefill SDPA program config is set in `models/experimental/diffusion_gemma/tt/diffusion_attention.py:72-92`: for `head_dim >= 512` it uses `grid = CoreCoord(8, 1)`, `q_chunk = k_chunk = 32` (note: the task brief said grid `(8,4)`; the live code is `(8,1)` — reconcile before build). At head_dim 512, `DHt = vDHt = 16`, so `qk_im` and `out_im_A/B` are the dominant L1 consumers. The new `cb_lse_out` (`Sq_chunk_t` fp32 tiles) + `cb_scale_in` (1 tile) add only a few KB, but **L1 headroom on the streaming path at DHt=16 must be re-verified** — this is the main budget risk (see §8). No change to the grid or subblock geometry is required.

---

## 7. `sdpa.cpp` / `sdpa.hpp` / `sdpa_nanobind.cpp` — public API

### 7a. `sdpa.hpp` (`:16-28`) — add `return_lse` with a Python-tuple-or-Tensor return

A C++ function cannot statically return *either* `Tensor` *or* `tuple`. Standard ttnn resolution: the **C++ core returns `std::vector<Tensor>`** (size 1 or 2); the **nanobind wrapper** returns a bare `Tensor` when `return_lse=False` and a `tuple` when `True`. Keep the existing `Tensor`-returning signature as the default façade and add an internal vector-returning core.

```cpp
// sdpa.hpp — add trailing param (default false); keep return type Tensor for the common path.
ttnn::Tensor scaled_dot_product_attention(
    ..., const std::optional<ttnn::Tensor>& cu_window_seqlens = std::nullopt,
    bool return_lse = false);                       // NEW
// plus an internal core returning the pair, or reuse ttnn::prim::sdpa_with_lse directly in the nanobind wrapper.
```

### 7b. `sdpa.cpp` (`:35-96`) — thread `return_lse` into the prim call

The mask pre-scale logic (`:69-76`) is unchanged. Route to the vector prim when `return_lse`:
```cpp
if (return_lse) {
    auto outs = ttnn::prim::sdpa_with_lse(input_tensor_q, ..., /*return_lse=*/true, ...);
    // outs[0] = attention output (dtype == q), outs[1] = LSE fp32 [B,NQH,Sq,1]
    // return both to caller (via the vector-returning core / nanobind tuple).
} else {
    return ttnn::prim::sdpa(...);   // unchanged single-Tensor path (byte-identical)
}
```
Both `chunked_scaled_dot_product_attention` overloads (`:99-168`) get the same optional `return_lse` param, forwarding `.return_lse` into `ttnn::prim::sdpa`. (chunked is always `is_causal=true`, streaming path — the LSE emit applies unchanged.)

### 7c. `sdpa_nanobind.cpp` — binding

For `scaled_dot_product_attention` (`:309-325`) and `chunked_*` (`:460-474`): add `nb::arg("return_lse") = false`, and wrap the callee in a small lambda that returns `nb::object`:
```cpp
[](... , bool return_lse) -> nb::object {
    if (!return_lse) return nb::cast(scaled_dot_product_attention(..., /*return_lse=*/false));  // Tensor
    auto pr = scaled_dot_product_attention_lse(...);                                             // std::tuple<Tensor,Tensor>
    return nb::cast(pr);                                                                         // (out, lse) tuple
}
```
This satisfies the task contract exactly: **`return_lse=False` → single `Tensor`; `return_lse=True` → `(output, lse)` tuple.**

---

## LSE output tensor spec (summary)

| property | value |
|---|---|
| logical shape | `[B, NQH, Sq, 1]` (mirror of `q.logical_shape()` with `[3]=1`; ring uses the same `stats_shape[3]=1` trick, `ring_joint_sdpa_device_operation.cpp:609-610`) |
| dtype | `FLOAT32` (task requirement; ring stats are bf16 — we upgrade to fp32 via a dedicated `cb_lse_out` Float32 CB) |
| layout | `TILE` |
| `[·,1]` column-in-tile mapping | last dim = 1 is padded to a 32-wide tile; **column 0 holds the LSE value**, columns 1-31 are pad. The sequence dim `Sq` is padded to `Sqt` row-tiles. One CB tile per Q row-tile → one DRAM page at `id_of(nb, nq, seq_tile, 0)` (identical addressing to `ring_joint_writer.cpp:333`). Downstream readers take column 0 (`[..., 0]`). |

---

## Effort estimate

| area | files | effort |
|---|---|---|
| Compute emit (`normalize_row_streaming` + `sdpa_standard_v2` + `sdpa.cpp` forward) | `compute_streaming.hpp`, `sdpa.cpp` | ~1.0 d |
| Writer drain | `writer_interleaved.cpp` | ~0.5 d |
| Program factory (CBs, args, buffers, scalar gen, grid/L1 check) | `sdpa_program_factory.cpp`, `sdpa_interleaved_cb_ids.hpp` | ~1.0 d |
| Device op multi-output (types, specs, create, perf-model, prim free fns) | `sdpa_device_operation_types.hpp`, `sdpa_device_operation.{hpp,cpp}` | ~1.0 d |
| Public API + nanobind + Python tuple return | `sdpa.{hpp,cpp}`, `sdpa_nanobind.cpp` | ~0.5 d |
| Unit test + QB2 verification | new `tests/ttnn/.../sdpa/test_sdpa_lse.py` | ~1.0 d |
| **Total** | | **~5 d** (one build-capable session block) |

---

## Risks

1. **CB / L1 budget at head_dim 512 (highest).** The streaming path at `DHt = vDHt = 16` is already L1-heavy (`qk_im`, `out_im_A/B`). Adding fp32 `cb_lse_out` (+`cb_scale_in`) is small (~4-8 KB) but must be validated to fit on the gemma4 `(8,1)` grid config (`diffusion_attention.py:79-84`). Mitigation: `cb_lse_out` sized to `Sq_chunk_t` (1 tile for gemma4), fp32 pack from DST is native, no extra DST pressure.
2. **fp32 packing.** `cb_lse_out` is Float32 while the stats/im CBs are Float16_b. `pack_tile` to an fp32 CB requires a `pack_reconfig_data_format(cb_lse_out)` around the emit (and restore afterwards), analogous to the ring path's `pack_reconfig_data_format(cb_lse_out)` at `compute_common.hpp:1986`. Missing reconfig → garbage LSE and/or corrupted subsequent bf16 packs (the exact failure `normalize_row_streaming:795-800` warns about).
3. **Byte-identity for gemma4 (`return_lse=False`).** Guaranteed structurally: every new code block is `if constexpr (emit_lse/return_lse)` (compute) or `if (attrs.return_lse)` (host), no new CB allocated, compute/writer compile-arg lists append at the tail so existing indices are stable, and `ttnn::prim::sdpa` keeps returning `[0]`. **Verify** by a bit-exact diff of the `return_lse=False` output tile stream vs `main` on a gemma4 chunked-prefill shape.
4. **`cur.max` / `cur.sum` index alignment.** `cur.max` uses *absolute* row index (`lse_row_offset + s`); `cur.sum` is read at its *current front* (row-group relative) because prior groups popped it. The `lse_row_offset += sbh` cadence must exactly track the `normalize_row` call sequence including the remainder row-group (`compute_streaming.hpp:1713-1821`). An off-by-`sbh` here silently mislabels LSE rows. Clone the proven `sink_row_offset` accounting verbatim.
5. **Non-streaming fallback.** `sdpa_standard` (deferred, non-RING) is *not* covered; host `TT_FATAL` must reject `return_lse` there. Acceptable because gemma4 uses streaming, but it is a real API gap to document.

---

## Build + QB2 verification steps

**Build:** standard TTNN build (C++ changes only in the SDPA op). No new kernel file. `./build_metal.sh` (or the project's configured build) — the design touches only `.hpp`/`.cpp` under `ttnn/cpp/ttnn/operations/transformer/sdpa/`, so a targeted `ninja ttnn` rebuild suffices.

**Unit test (new `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_lse.py`), QB2:**

1. **LSE vs torch reference (correctness of the statistic).**
   - Random Q,K,V (bf16), non-causal and causal, shapes covering gemma4 (e.g. `[1, H, 512, 512]`, head_dim 128 and 512), streaming path.
   - `out, lse = ttnn.transformer.scaled_dot_product_attention(q,k,v, is_causal=..., return_lse=True)`.
   - Reference: `S = scale*(Qf@Kf.transpose(-1,-2))` (+causal mask); `lse_ref = torch.logsumexp(S, dim=-1, keepdim=True)` → `[B,H,Sq,1]`.
   - Assert `lse[...,0]` vs `lse_ref[...,0]`: bf16-input tolerance (rtol≈2e-2 / high PCC ≥ 0.999); assert `out` PCC unchanged vs `return_lse=False` run.

2. **Byte-identity gate (`return_lse=False`).** Run the same op with `return_lse=False` and assert the output tensor is **bit-identical** to the current `main` build on a gemma4 chunked-prefill shape (`chunked_scaled_dot_product_attention` with a page table). This is the gemma4-safety gate.

3. **`full-softmax(concat) == merge(partials)` equivalence (the T6 payoff).** No in-repo merge helper exists — implement the flash merge in the test:
   - Full: `out_full, lse_full = sdpa(q, K, V, return_lse=True)` over the whole KV.
   - Split K,V along seq into `[K1,V1]`,`[K2,V2]`; `o1,l1 = sdpa(q,K1,V1,return_lse=True)`, `o2,l2 = sdpa(q,K2,V2,return_lse=True)` (both **non-causal** partials).
   - Merge (host torch): `l = torch.logaddexp(l1, l2)`; `o = o1*torch.exp(l1-l) + o2*torch.exp(l2-l)`.
   - Assert `o` ≈ `out_full` and `l` ≈ `lse_full` (high PCC). This is the exact identity the paged-prefix + new-chunk merge will use.

**Device hygiene:** streaming SDPA at head_dim 512 is L1-tight — if a trace-region/CB overflow poisons the device, recover with `tt-smi -r` (per DG session notes; eth core 29-25 is the recurring offender on QB2).

---

## One-paragraph summary

The LSE is already computed on the plain streaming SDPA path — `cur.max` (raw per-row max) and `cur.sum` (softmax denominator `l`) sit fronted at the end of the K-loop, and the ring-joint kernel already emits `LSE = scale·max + log(l)` with `log_block` + `mul_block_bcast_scalar_inplace<cb_scale_in>` + `add_block_inplace` (`compute_common.hpp:1928-1988`). The plan reproduces those three ops at row granularity inside `normalize_row_streaming` (`compute_streaming.hpp:699-801`), reusing the existing attention-sink `cur.max` read plumbing, packs `scale·max + log(l)` into a **dedicated fp32 `cb_lse_out`** (not the ring's aliased `cb_max_*`, since our normalization is deferred), and drains it to a new `[B,NQH,Sq,1]` fp32 output via a `write_output_and_lse`-style loop in `writer_interleaved.cpp`. Everything is gated behind a compile-time `return_lse` flag so `return_lse=False` is byte-identical for gemma4; the device op becomes multi-output by mirroring JointSDPA's `std::vector<Tensor>` return under the same `device_operation::launch` framework, and the nanobind wrapper returns a single `Tensor` (false) or an `(output, lse)` tuple (true). **The single riskiest unknown is L1/CB budget on the head_dim-512 streaming configuration** (`DHt=vDHt=16`, gemma4 `(8,1)` grid): the added fp32 CBs are small, but the streaming path is already the tightest L1 consumer at that head dim, and the fp32-pack reconfig around the emit must be correct or it will both corrupt the LSE and poison subsequent bf16 packs — so the first build step must be an L1-fit + bit-identity check on the exact gemma4 chunked-prefill shape.
