# Plan: `sparse_sdpa` — Sparse MLA Prefill Op (Blackhole, single-chip)

Target branch: `pjosipovic/sparse_mla_prefill_ref` (torch reference committed there).

## Staging
- **Stage 1 (`H = 32` heads):** Q/scores/out occupy a full `{32,32}` tile. The end-to-end op, PCC ≥ 0.99 vs the golden.
- **Stage 2 (`H = 16`):** `{16,32}` half-tiles (§6.4) + the LLK uplift that makes the streaming primitives `{16,32}`-capable (§11).

The op reads `H` from the Q shape (Stage 1 validates `H == 32`; Stage 2 also accepts `16`). The op *structure* is shared; **Stage 2 additionally needs** host half-tile CB descriptors (`CBFormatDescriptor::tile`), a 2-face mask builder (reader), and a writer that emits only 16 rows — so reader/host/writer get `{16,32}`-specific branches, not zero change. Everything below is Stage 1 unless tagged **[Stage 2]**.

---

## 1. What the op computes

Absorbed MQA prefill over the top-k selected latents, per query token, with masking baked into the index list (contract: `models/demos/deepseek_v32/reference_cpu/sparse_sdpa_prefill.py::sparse_mla`):

```
out = softmax( (Q @ Kᵀ) · scale  masked )  @ V        # V = K[:, :512]
```

### Tensors (Stage 1, `H = 32`)

| Tensor   | Shape                | dtype | Layout (DRAM interleaved) | Meaning |
|----------|----------------------|-------|---------------------------|---------|
| Q        | `[1, 32, 640, 576]`  | bf16  | ROW-MAJOR                 | absorbed queries; 32 heads, 640 query tokens, 576 = 512 latent + 64 rope |
| KV       | `[1, 1, 56320, 576]` | bf16  | ROW-MAJOR                 | latent prefix, single kv head; K = full 576, V = `KV[..., :512]` |
| Indices  | `[1, 1, 640, 2048]`  | uint32| ROW-MAJOR                 | per query token, 2048 selected key rows into KV; `0xFFFFFFFF` = masked |
| Output   | `[1, 32, 640, 512]`  | bf16  | ROW-MAJOR                 | attention output in latent-512 space |

[Stage 2] `H = 16`: Q `[1,16,640,576]`, out `[1,16,640,512]`; KV/Indices unchanged (Indices are per-token, shared across heads).

Constants for the target case: `S=640`, `H=32`, `K_DIM=576` (`DHt=18` tiles), `V_DIM=512` (`vDHt=16` tiles), `TOPK=2048` (`Skt=64` tiles), `T=56320`, `scale = 576**-0.5`. **`S`, `T`, `TOPK` are parametric** (validation §5.4 pins only `H∈{16,32}`, `K_DIM=576`, `V_DIM=512`, dtypes/layouts, `TOPK % k_chunk_size == 0`, row alignment, and the bounds below) — so unit tests / craq-sim (§10) may use **small** `S`/`T`/`TOPK` with the same kernel. **`TOPK ≤ 2048`** (hard bound — `Skt ≤ 64`; the indices row + `kt_in`/score CB sizing assume it); `S, T, TOPK > 0`.

### Producer contract (relied on, not validated on device)
- Sentinels are a **contiguous tail**: after the first `0xFFFFFFFF` in a row, all remaining indices are sentinel (valid keys are a prefix `[0:n_valid]`).
- Every row has **≥ 1 valid key** (`n_valid ≥ 1`) — no all-masked row, no softmax NaN.
- All non-sentinel indices are **`< T`** — the op only guards the `0xFFFFFFFF` sentinel; an out-of-range non-sentinel index would be an unchecked DRAM read.

These are documented as hard preconditions in the binding; the op stays producer-coupled / experimental until the producer owns validation.

### Masking
Masked keys contribute nothing to the softmax numerator **and** denominator. The reader builds a per-chunk `-inf/0` additive mask (zero-filling K is insufficient — score `0` ≠ `-inf`, which would inflate the denominator). Under the tail contract every per-chunk mask is **contiguous** (all-`0`, vertical `-inf` from a column, or all-`-inf`). Details + the truncation perf refinement: §6.1.

---

## 2. Parallelization — per-token, fully independent cores (no mcast)
- Work unit = **one query token** (all `H` heads of it). 640 tokens total.
- Distribute across the BH compute grid (≈110 worker cores): `640/110 → ≤ 6 tokens/core` (contiguous ranges per core, like SDPA's `global_q_start`/`global_q_count`, `sdpa_program_factory.cpp:1286-1295`).
- **No core-to-core data movement, no mcast, no semaphores, no KV-forwarding.** Each core reads its own Q/Indices/KV pages from DRAM, computes, writes its own output. (Drops SDPA's non-causal chain/topology machinery, `sdpa_program_factory.cpp:776-1243`.)
- Each core loops over its ≤6 tokens sequentially.

---

## 3. Per-core pipeline (per token)
One token's latent K `[2048,576]`=2.36 MB and V `[2048,512]`=2.0 MB both far exceed BH L1 (~1.5 MB). So the 2048 keys are **streamed over the key axis (Sk) in chunks** and processed flash-style (running max/sum, rescale) — this reads each gathered latent page exactly **once** (QK uses all 576 cols, PV the first 512), the DRAM-optimal structure for the dominant KV tensor.

For each token (**reader delivers row-major; compute tilizes — tilize never happens in the reader**):
1. **Reader** gathers the 32 Q head-rows `[32,576]` (row-major); **compute** tilizes → `q_tiles` (`[1,18]` grid of `{32,32}` tiles), resident for the token.
2. Init running `max`/`sum` (ping-pong) + output accumulator.
3. **For each Sk-chunk** (fixed `n_chunks = ceil(TOPK / k_chunk_size)`, factory-passed to both kernels):
   a. **Reader** reads the chunk's `k_chunk_size` index values, issues that many indexed DRAM page reads → K rows `[k_chunk,576]` row-major; `idx==0xFFFFFFFF` → NoC zero-fill (`fill_zeros_async`, `dataflow_common.hpp:25`, OOB safety). Builds the chunk's contiguous `-inf/0` mask (§6.1).
   b. **Compute** tilizes → `k_tiles[Skt,DHt]`, then **produces Kᵀ** `cb_kt_in[DHt,Skt]` (§6.2). Kᵀ stays resident for both matmuls.
   c. **QK**: `Q @ Kᵀ → scores` (`blocked_matmul_and_pack<transpose=true>`).
   d. **Mask add**: add the chunk's mask tiles to the scores, at the QK→reduce seam (§4.2/§6.1).
   e. **Online softmax update** (REDUCE_ROW): running max → `exp((qk−max)·scale)` + L1 row-sum → cross-chunk rescale.
   f. **PV**: `probs @ V` where V is read from `cb_kt_in` via `KT_stride` (`inplace_v_matmul_pack_batched`), accumulating into the output.
4. **Normalize**: collapse the partial row-sum + reciprocal + multiply (`normalize_row_streaming`).
5. **Untilize** the output → row-major; **writer** writes the `H` head-rows to DRAM.

### Tuning knobs
- `k_chunk_size` (Sk streaming granularity, in keys). **Must divide `TOPK=2048` and be a multiple of 32** — the fixed loop reads `n_chunks·k_chunk` index slots, so a non-divisor would read past the indices row. Default 128; the host L1-budget check (§5.4) caps the viable set (≈ `{32, 64, 128}`).
- The QK inner dim is the full `DHt=18` (matmul accumulates over all 18 D-tiles before packing); the 576 dim is not sub-sliced (`[k_chunk,576]` fits L1).

---

## 4. Kernels
Three kernels per core (reader / compute / writer).

### 4.1 Reader (`device/kernels/dataflow/sparse_sdpa_reader.cpp`) — RISCV_1 / read NoC
- **Row-major page read** via `TensorAccessor`, `page_id = row_index`, `page_size = bytes_per_row` (Q/K row = 576·2 = 1152 B; indices row = `TOPK·4` ≤ 8192 B; all 32-B aligned). Pattern: `read_page_table_for_batch` (`dataflow_common.hpp:71-92`).
- **Indexed gather**: read the token's indices row into L1 once; find `n_valid` (first-sentinel position). For each of the fixed `n_chunks` over `TOPK` (parametric, ≤ 2048): `for j in chunk: idx=idx_ptr[j]; if idx==0xFFFFFFFF fill_zeros_async(...) else noc.async_read(kv_reader, dst, row_bytes, {.page_id=idx})`, with a periodic `noc.async_read_barrier()` every `barrier_threshold` (`dataflow_common.hpp:20`). Model on `read_paged_chunk_with_padding` (`dataflow_common.hpp:255-315`) with `idx_ptr[j]` replacing the block-table math.
- **Per-chunk contiguous mask** into `cb_mask_in` (added to scores by compute, §4.2): a full `{32, Sk_t}` bf16 tile per chunk, built from `n_valid` — all-`0` (chunk `< n_valid`), vertical `-inf` from column `n_valid % k_chunk` (boundary chunk), or all-`-inf` (chunk `≥ n_valid`); `-inf` replicated down all 32 rows. Vertical-mask builder (`dataflow_common.hpp:363`); 4-face (fine for Stage-1 `{32,32}`; Stage 2 needs a 2-face-aware variant, §6.4). No arbitrary-column generator needed.
- **Zero-fill** = NoC loopback read from `MEM_ZEROS_BASE` (`noc_zero_l1.inl:12-41`), covered by the trailing `async_read_barrier` on BH.
- Emits per chunk: row-major K rows + mask tiles; once per token: Q rows.

### 4.2 Compute (`device/kernels/compute/sparse_sdpa_compute.cpp`)
Both Q and K are tilized in the compute kernel. The kernel is **our own per-token loop composing the granular `compute_streaming.hpp` functions** (QK / reduce / sub-exp / PV / normalize) — we do **not** call the monolithic `sdpa_inner_loop_step` (§6.3). We own the loop, ping-pong accumulator, mask add, and CB lifecycle (reference: `sdpa_inner_loop_step`'s body). In-kernel tilize→matmul→softmax is a shipping BH pattern (`sdpa_flash_decode.cpp:209-221`; conv precedent `conv_bmm_tilize.cpp`).

- **Tilize** Q and each K chunk: `compute_kernel_lib::tilize<block_width_tiles, in_dfb, out_dfb>(num_blocks, total_input_pages)` — asymmetric page mode (`tilize_helpers.hpp:106-108`) ingests row-sized input pages. `compute_kernel_hw_startup(in_dfb, out_dfb)` once at start. After each `tilize`, re-issue `mm_*_init_short` + `reconfig_data_format_srca` (`sdpa_flash_decode.cpp:218`); back-to-back tilizes use `InitOnly/Neither/UninitOnly` (BH SrcA-override caveat, `conv_bmm_tilize.cpp:45-52`).
- **Kᵀ production (the one genuinely new compute block — exact lifecycle).** Grid-reposition `k_tiles[Skt,DHt] → cb_kt_in[DHt,Skt]`, **tile contents unchanged** (the QK `transpose=true` does the intra-tile transpose). Use the **compute `copy_tile` (CB→DST)** (`tile_move_copy.h:82`) — NOT the dataflow `copy_tile` (`dataflow_common.hpp:317`, a self-NoC L1 copy), and NOT `transpose_wh` (intra-tile — would double-transpose). Sequence with the required waits/inits:
  ```
  cb_wait_front(k_tiles, DHt*Skt)                 // tilize must have pushed it (tile_move_copy.h:87 requires the wait)
  copy_tile_to_dst_init_short(k_tiles)            // init copy from this CB
  cb_reserve_back(cb_kt_in, DHt*Skt)
  for sk in 0..Skt-1: for d in 0..DHt-1:
      tile_regs_acquire(); copy_tile(k_tiles, sk*DHt+d, 0); tile_regs_commit()
      tile_regs_wait();   pack_tile(0, cb_kt_in, d*Skt+sk); tile_regs_release()
  cb_push_back(cb_kt_in, DHt*Skt); cb_pop_front(k_tiles, DHt*Skt)
  ```
  Same wait-after-tilize-before-consume rule applies to `q_tiles` and each `k_tiles` (`sdpa_flash_decode.cpp:209`). The full per-CB waits/pushes/pops for `q_tiles`/`kt_in`/`mask_in`/`qkt_im` mirror the §6.3 skeleton + `sdpa_inner_loop_step`'s ordering. (SDPA's reader gets the grid transpose via a DRAM write-stride trick, `dataflow_common.hpp:277`; we can't, so we do it post-tilize.)
- **Per-chunk compute** (skeleton in §6.3): `blocked_matmul_and_pack<transpose=true>` QK off `cb_kt_in` (`:286`) → **mask stamp** → `reduce_c_row_group` (`:378`) → `sub_exp_block_bcast_cols<scale_fp32>` (`:447`) → SALAD rescale (`:543/:587`) → `inplace_v_matmul_pack_batched` PV reading V from `cb_kt_in` via `KT_stride` (so rope cols are never read, `:338`) → `normalize_row_streaming` on the last chunk (`:678`). `scale` is **compile-time** (in the hash); scaler + `cb_col_identity` built once in the **writer** (§4.3); the `cb_qkt_im`/`cur.sum` reserve + deferred Kᵀ pop follow `sdpa_inner_loop_step`'s ordering.
- **Mask stamp — hold-preserving, NOT `add_block_inplace`.** `cb_qkt_im` is held (`cb_push_back_hold_wr_ptr`, wr-ptr not advanced) between the QK pack and the reduce; `add_block_inplace` (`compute_common.hpp:560`) `cb_wait_front`/`cb_pop_front`/re-reserves its in0 and would break that flow. Use the **streaming L1-accumulate mask stamp** that adds `cb_mask_in` onto the held score rows in place before the hold-push (the lightweight-mask stamp pattern, `compute_streaming.hpp:1283`). Our mask is a full `{32,Sk_t}` tile (reader-built, §4.1), so it's a dense L1-accumulate add at that seam — adapt that stamp, don't introduce a CB-popping add.
- **Untilize**: `compute_kernel_lib::untilize<vDHt=16, in_dfb, out_dfb>(1)` → row-major block in the out CB.

### 4.3 Writer (`device/kernels/dataflow/sparse_sdpa_writer.cpp`) — RISCV_0 / write NoC
- **Builds the two constant CBs once at start:** reduce scaler via `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_identity_scale_in, PoolType::MAX, ReduceDim::REDUCE_ROW, /*factor*/1, /*compute_uses_reduce_tile=*/true>()` (`reduce_helpers_dataflow.hpp:94`; `compute_uses_reduce_tile=true` ⇒ row-0 fill, since compute calls the `reduce_tile` LLK directly); col-identity via `generate_bcast_col_scalar(cb_col_identity, packed_1.0f)` (`generate_bcast_scalar.hpp:12`). Mirrors `writer_interleaved.cpp:78-84`.
- **Row-major output write:** `compute_kernel_lib::untilize` emits **tile-sized** CB pages (row-major bytes inside, 16 pages per row-block — `untilize_helpers.hpp:109`, `.inl:221`), so use the **blocked row-major writer pattern**: `cb_wait_front`/`cb_pop_front` the full 16-tile block, then one `noc_async_write` per head-row from L1 byte offset `r·block_row_stride` → DRAM `page_id = head·S + token`, `512·2 = 1024 B`. Reference: `write_row_major_block_from_cb` (`layernorm_dataflow_utils.h:230`) + `untilize_all_blocks_from_cb` (`layernorm_compute_utils.h:86`). (Not `writer_interleaved.cpp`, which writes tiles.)

---

## 5. Host side (new device-op infra — `ProgramDescriptor` concept)
SDPA uses the new device-op infra (`ProgramDescriptorFactoryConcept`): all-static methods, declarative `ProgramDescriptor`, no `override_runtime_arguments`. A new op must use this (pre-commit `scripts/detect_legacy_device_op.py` rejects legacy). Mirror `sdpa`.

### 5.1 Files to create
```
ttnn/cpp/ttnn/operations/transformer/sdpa/
  sparse_sdpa.hpp / .cpp                         # host entry: ttnn::transformer::sparse_sdpa(...)
  device/sparse_sdpa_device_operation_types.hpp  # SparseSDPAParams (attrs), SparseSDPAInputs (tensors)
  device/sparse_sdpa_device_operation.hpp / .cpp # SparseSDPAOperation struct + prim:: free fn → launch<>
  device/sparse_sdpa_program_factory.hpp / .cpp  # create_descriptor(...)
  device/kernels/dataflow/sparse_sdpa_reader.cpp
  device/kernels/dataflow/sparse_sdpa_writer.cpp
  device/kernels/compute/sparse_sdpa_compute.cpp
```
### 5.2 Files to edit
- `transformer/sources.cmake` — add the 3 new `.cpp` (device op, program factory, host entry). Kernels are auto-globbed (`CMakeLists.txt:16-21`).
- `transformer/CMakeLists.txt` — add `sparse_sdpa.hpp` to the `FILE_SET api` (around `:25`); `sources.cmake` covers `.cpp` only.
- `sdpa/sdpa_nanobind.cpp` — add the `ttnn.transformer.sparse_sdpa` binding inside `bind_sdpa` (`ttnn::bind_function<"sparse_sdpa", "ttnn.transformer.">(...)`, `:306`). Document the producer preconditions (§1) in the docstring.

### 5.3 Device-op struct (mirror `sdpa_device_operation.hpp:18-42`)
```cpp
struct SparseSDPAOperation {
    using operation_attributes_t = SparseSDPAParams;     // scale (compile-time, in hash), k_chunk_size, out_mem_config, compute_kernel_config
    using tensor_args_t          = SparseSDPAInputs;     // q, kv, indices
    using spec_return_value_t    = TensorSpec;
    using tensor_return_value_t  = Tensor;
    struct ProgramFactory { static ProgramDescriptor create_descriptor(const op_attrs&, const tensor_args&, tensor_return_value_t&); };
    using program_factory_t = std::variant<ProgramFactory>;
    static void validate_on_program_cache_miss(const op_attrs&, const tensor_args&);
    static spec_return_value_t compute_output_specs(const op_attrs&, const tensor_args&);
    static tensor_return_value_t create_output_tensors(const op_attrs&, const tensor_args&);
    static ttsl::hash::hash_t compute_program_hash(const op_attrs&, const tensor_args&);    // include scale + k_chunk_size
};
// free fn: ttnn::prim::sparse_sdpa(...) → ttnn::device_operation::launch<SparseSDPAOperation>(attrs, inputs);
```
No `program_config` — the op exposes `scale`, `k_chunk_size`, `memory_config`, `compute_kernel_config`, and uses the full device grid. Output spec is explicitly **ROW_MAJOR**:
```cpp
auto shape = q.logical_shape(); shape[3] = 512;   // head_dim_v
return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), attrs.out_mem_config));
```

### 5.4 `validate_on_program_cache_miss`
- `q.device()->arch() == tt::ARCH::BLACKHOLE` (BH-only).
- `H == 32` (Stage 1; Stage 2 also accepts `16`).
- All four tensors ROW_MAJOR / DRAM / interleaved and **on the same device**; Q/KV/out bf16, Indices uint32; output `memory_config` DRAM-interleaved.
- `q=[1,H,S,576]`, `kv=[1,1,T,576]`, `indices=[1,1,S,TOPK]`; `S, T, TOPK > 0`; **`TOPK ≤ 2048`**; indices row bytes (`TOPK·4`) and K row bytes (`576·2`) both `%32==0`.
- **Row-major page = padded last-dim × elem_size** (`to_memory_config_op.cpp:229`). The reader's `page_id = row_index` math is only valid if there's no last-dim padding: assert **`padded_shape == logical_shape`** for Q/KV/Indices, and that `buffer()->page_size()` equals the expected row bytes (Q/K `1152`, Indices `TOPK·4`). Otherwise the gather silently reads wrong rows.
- `k_chunk_size ≥ 32`, multiple of 32, `TOPK % k_chunk_size == 0`. `head_dim_v (512) ≤ 576`. `scale > 0`.
- **`compute_kernel_config.fp32_dest_acc_en == false`** — reject `true` (the streaming `kt_inplace_v` / `dst_size=8` path requires it; don't silently override, or the hash/config would disagree with the compiled kernel).
- **L1 budget**: sum the per-core CB bytes (§5.6, with depth) for the chosen `k_chunk_size`; `TT_FATAL` if it exceeds usable L1 (~1.4 MB). Fail loud, no silent clamp.

### 5.5 Program factory — `create_descriptor`
- Grid: full `device->compute_with_storage_grid_size()` → dense `CoreRangeSet` from origin (`sdpa_program_factory.cpp:316-321`); kernels launch on **every** core and `num_cores = grid.x·grid.y`. The 640 tokens are split across all cores; cores past the work get **`tok_count = 0`** and early-return (SDPA's `global_q_count==0` idle-core pattern, `:1282`). Don't shrink to `min(grid, S)` — that would launch kernels on cores with no runtime args. (For small-`S` tests, most cores are idle with `tok_count=0`.)
- Per-core token range `(tok_start, tok_count)` via even split, **runtime args emitted for every core** (`:1286-1295`).
- CBs via the `allocate_cb`/`allocate_tile_cb` lambdas (`:661-678`) into `desc.cbs` (§5.6).
- 3 `KernelDescriptor`s: reader `ReaderConfigDescriptor{}`, writer `WriterConfigDescriptor{}`, compute `ComputeConfigDescriptor{ math_fidelity, fp32_dest_acc_en=false, … }` (`fp32_dest_acc_en=false` → `dst_size=8`, `:355`). `scale_fp32`, `H`, `k_chunk_size`, `n_chunks`, `DHt`, `vDHt`, `TOPK` are compile-time args → in `compute_program_hash`.
- Runtime args via `emplace_runtime_args(core, {...})`; pass `Buffer*` directly (auto-refresh on cache hit): tensor buffers + `tok_start`, `tok_count`.
- No semaphores (no mcast).

### 5.6 CBs / DataflowBuffers (per core)
Double-buffer (depth 2) **only** the CBs handed off between reader/compute/writer threads (one in-flight slot is enough); keep per-token-resident state single. The `max/sum/out_acc` ping-pong (×2) is the flash cross-chunk recurrence, a separate concern.

| CB / DFB | Contents | Depth | Pages × size (per slot) |
|---|---|---|---|
| q_rm_in | Q rows (row-major) | 2 | 32 × 1152 B |
| q_tiles | Q tiled `[1,18]` | 1 | 18 × 2 KB |
| k_rm_in | K chunk rows (row-major) | 2 | k_chunk × 1152 B |
| k_tiles | tilize output `[k_chunk_t,18]` (transient → Kᵀ) | 1–2 | k_chunk_t·18 × 2 KB |
| kt_in | **Kᵀ** `[18,k_chunk_t]` (QK in1 + PV V-source) | 2 | 18·k_chunk_t × 2 KB |
| mask_in | per-chunk full `{32,k_chunk_t}` contiguous mask | 2 | k_chunk_t × 2 KB |
| qkt_im (`cb_qkt_im`) | scores `[1,k_chunk_t]` | 1 | k_chunk_t × 2 KB |
| indices_in | one token's index row (uint32) | 2 | 1 × `TOPK·4` B (≤ 8192) |
| reduce_scaler | reduce identity scaler | 1 | 1 × 2 KB |
| col_identity | column of 1s for the sum collapse | 1 | 1 × 2 KB |
| max/sum/exp_max_diff | softmax stats | 2 (ping-pong) | 1 × 2 KB each |
| out_acc | output accumulator | 2 (ping-pong) | 16 × 2 KB |
| out_tiles | normalized out (untilize input) | 1 | 16 × 2 KB |
| out_rm | untilized row-major block (tile-sized pages) | 2 | 16 × 2 KB |

DFB indices map 1:1 to CB indices. CB byte size = `page_size · num_pages · depth`; `indices_in`/`mask_in` track `TOPK`/`Sk_t`. Stage 1 tiles are all `{32,32}` = 2 KB. The host L1 check sums this exact set (incl. both `k_tiles` and `kt_in`); the viable `k_chunk_size` is derived from that Σ (likely ≤128; first levers if tight: `k_tiles`→depth-1, free it before PV).

**[Stage 2 `H=16`]** half-tile CBs need **`CBFormatDescriptor::tile = Tile({16,32})`** set explicitly (`program_descriptors.hpp:61`) — page size alone keeps a full tile, and Stage-1's `allocate_tile_cb` would silently make `{32,32}` CBs (`sdpa_decode` sets it, `sdpa_decode_program_factory.cpp:512`). Which CBs are `{16,32}` vs full follows §6.4; `k_tiles`/`kt_in` stay `{32,32}`.

---

## 6. Key design points

### 6.1 Masking
v1 runs the **fixed `n_chunks` loop** (factory-passed to both kernels — they agree by construction, no runtime sync) and masks **every** chunk with the reader-built contiguous `-inf/0` mask (§4.1). Fully-sentinel tail chunks get an all-`-inf` mask (correct, just wasted compute). The mask is applied before reduce_max so masked keys are excluded from both the max and the sum.

**Truncation (perf refinement):** the reader passes `n_valid` via a small per-token metadata CB; compute truncates the loop to `ceil(n_valid/k_chunk_size)` (skipping fully-sentinel tail chunks — a large win for early/sparse tokens) and masks only the boundary tile. Deferred until the e2e op runs; the fixed `n_chunks` is the upper bound.

`{16,32}` (Stage 2) is the head/M axis; masking is the key/N axis — orthogonal.

### 6.2 K layout — Kᵀ required; V read via `KT_stride`
QK contracts over the head dim, and `matmul` walks `in1` contiguously (`compute_common.hpp:1276`), so QK needs K physically as the **Kᵀ grid `[DHt, Skt]`** — the matmul `transpose=true` flag handles the *intra-tile* transpose, but the **tile grid** must already be transposed. SDPA's reader produces the grid transpose via a write-stride trick on contiguous tiled DRAM (`dataflow_common.hpp:277`, `reader_interleaved.cpp:451`); our indexed row-major gather can't, so compute grid-repositions the tilized rows with a `copy_tile` loop — **tile contents unchanged, no `transpose_wh`** (§4.2).

The streaming `kt_inplace_v` path then drives **both** matmuls off the single Kᵀ buffer: QK = `blocked_matmul_and_pack<transpose=true>` (`compute_streaming.hpp:286`); PV reads V from Kᵀ via `KT_stride` (`inplace_v_matmul_pack_batched`, `:338`, `V[sk][vd] == Kᵀ[vd][sk]`), touching exactly `vDHt` rows of Kᵀ so the 2 rope columns are never read. One Kᵀ buffer feeds both matmuls (`Sq_chunk_t==1`, satisfied by one tile-row of heads); the pop is deferred to after PV (`:1377-1381`/`:1754-1758`).

### 6.3 Compute primitives
**Decision: a slim purpose-built kernel that COMPOSES the granular `compute_streaming.hpp` functions** — we do **not** call the monolithic `sdpa_inner_loop_step` (it's welded to SDPA's ring/sliding-window/attention-sink/kv-pad/chunked machinery and CB contract, with no seam for our tilize/Kᵀ stage). We own the per-token loop, the ping-pong accumulator, the mask add, and the CB lifecycle; `sdpa_inner_loop_step`'s body (`:1207-1760`) is the **reference** for the exact reserve/push/pop ordering, not a function we compile in.

```
compute_kernel_hw_startup(q_rm_dfb, q_tiles_dfb)
tilize Q rows -> q_tiles
AccumulatorHalf prev={sum_A,max_A,out_A}, cur={sum_B,max_B,out_B}   // ping-pong CB ids
n_chunks = ceil(TOPK / k_chunk_size)                           // fixed, factory args
for chunk in 0 .. n_chunks-1:
    tilize k_rm_in -> k_tiles[Skt,DHt]
    grid-reposition k_tiles -> cb_kt_in (Kᵀ [DHt,Skt])         // copy_tile CB→DST→pack to (d,sk); §4.2
    cb_reserve_back(cb_qkt_im, Sq_chunk_t*KT_stride); cb_reserve_back(cur.sum, Sq_chunk_t)   // lifecycle per :1215/:1217
    cb_wait_front(cb_kt_in, DHt*KT_stride)                     // resident for QK+PV (pop after PV)
    blocked_matmul_and_pack<transpose=true,KT_stride,KT_stride>(q_tiles, cb_kt_in, cb_qkt_im, …)   // QK :286
    add cb_mask_in to cb_qkt_im                                // mask add at the QK→reduce seam (§4.2)
    reduce_c_row_group(cur.max, prev.max, do_eltwise_max=chunk>0, …)        // :378
    sub_exp_block_bcast_cols<scale_fp32>(cb_qkt_im, cur.max, cur.sum, …)    // :447 exp(PACK thread)+L1 sum
    if chunk>0: sub_exp_first_col_blocks + salad_correct_fused              // :543/:587 cross-chunk rescale
    inplace_v_matmul_pack_batched<vDHt,dst_size,subblock_h=1>(cb_qkt_im, cb_kt_in, cur.out, …)   // PV via KT_stride :338
    cb_pop_front(cb_kt_in, …)                                  // deferred Kᵀ pop, after PV (:1754-1758)
    std::swap(prev, cur)
normalize_row_streaming(prev.sum, prev.out) -> out_tiles      // :678 sum collapse + recip + mul
untilize out_tiles -> out_rm  -> blocked row-major writer §4.3
```
(All `:NNN` are `compute_streaming.hpp`. The mask add must respect the `cb_qkt_im` hold-wr-ptr layout that `blocked_matmul_and_pack` packs into — replicate `sdpa_inner_loop_step`'s ordering; §12.)

The streaming MOPs are 32-only (custom MOPs hardcode 4 faces) — fine at `H=32`. **[Stage 2 `H=16`] is the §11 `{16,32}` LLK uplift** (keeps this kernel + overlap). *Contingency only* (if the uplift slips): the `compute_common` materialized path (explicit Kᵀ buffer + a compact `v_tiles` with rope cols dropped, standard `matmul_blocks`) gives correct `H=16` with 0 LLK work but heavier L1 and no overlap — a stopgap, not the target.

### 6.4 [Stage 2] `H = 16` via `{16,32}` half-tiles
Configure the Q / scores / stats / intermediate CBs as `Tile({16,32})` — set `CBFormatDescriptor::tile` explicitly (`program_descriptors.hpp:61`), page size alone keeps a full tile (§5.6). `sdpa_decode` runs this on BH (`sdpa_decode_program_factory.cpp:441-451`; `use_half_tile = is_causal && num_q_heads <= 16 && q_df==Float16_b`). It half-tiles the intermediates incl. the `im_tile`-class accumulators (`:590`) but keeps the **final output CB full** (`:610`). So follow that split: the final `out_rm` / DRAM write stays full and the writer emits 16 valid rows; whether `out_acc`/`out_tiles` are `{16,32}` (`im_tile`) or full follows `sdpa_decode`'s usage — **confirm during Stage 2** (§12).
- LLK matmul `{16,32}` path: `is_in0_16x32` (`tt_llk_blackhole/llk_lib/llk_math_matmul.h:52`); geometry from CB tile-dims (only the 16×16·by·16×16 combo is disallowed).
- REDUCE_ROW supports `{16,32}` (tiny-face path, `llk_math_reduce.h:175-220`, `llk_unpack_AB_reduce.h:62-66`); keep all reductions on REDUCE_ROW (REDUCE_SCALAR is full-tile only, `reduce_helpers_dataflow.inl:231`).
- The reduce scaler must come from the tile-dims-aware `calculate_and_prepare_reduce_scaler` (`reduce_helpers_dataflow.inl:221`), which sizes from CB tile-dims — **not** the legacy `wh_generate_reduce_scaler` (`generate_reduce_scaler.hpp`, hardcoded 2048 B / 4 faces, "WILL NOT WORK IN BLACKHOLE").
- **Mask builder is 4-face-hardcoded too.** `fill_vertical_tile_bf16` (`dataflow_common.hpp:372`) assumes 4 faces, so the Stage-1 `{32,32}` mask is fine but a `{16,32}` mask CB would be corrupted. Stage 2 needs a **tile-dims-aware vertical-mask builder** (2-face for `{16,32}`). (Keeping the mask full `{32,32}` is not an option: the `add` requires the mask tile-shape to match the `{16,32}` scores.)
- **`col_identity` stays FULL `{32,32}` in Stage 2.** `generate_bcast_col_scalar` is 4-face-hardcoded (`generate_bcast_scalar.hpp:12`) and `sdpa_decode` keeps `col_identity` full even in half-tile mode (`sdpa_decode_program_factory.cpp:579`). So the Stage-2 split is: **half-tile** q/scores/stats/mask, **full-tile** `col_identity` (and the matmul-reduce that consumes it).

---

## 7. Test plan
- New `tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa.py`, modeled on `mla_test_utils.py:444`. **BH-only: `@pytest.mark.skipif(not is_blackhole(), ...)`** (the op validates `arch==BLACKHOLE`, §5.4) so non-BH runs skip cleanly.
- **Correctness uses SMALL parametric shapes.** The golden gathers `sel[B,S,k,D]` (`sparse_sdpa_prefill.py:79`) ≈ `S·k·D` — at full `640·2048·576` that's ~2.8 GiB fp32, impractical for a unit test. Run PCC on small `S`/`TOPK`/`T` (e.g. `S=32`, `TOPK∈{32,64}`, `T≈256`), which exercise the same kernel (`S`/`T`/`TOPK` parametric, §1). Build random `KV[1,1,T,576]`, `Q[1,H,S,576]`, `indices[1,1,S,TOPK]` matching the producer contract (valid prefix + `0xFFFFFFFF` tail, entries `<T`, `n_valid ≥ 1`); vary `n_valid` (full; few-valid; mid-tile boundary; tile-aligned). Pass `kv[0,0]` (`[T,576]`) + `indices` reshaped to the reference signature — `sparse_mla._prep` does not accept a 4D `[1,1,T,576]`.
- Run `ttnn.transformer.sparse_sdpa(q, kv, indices, scale=K_DIM**-0.5, k_chunk_size=…, compute_kernel_config=WormholeComputeKernelConfig(fp32_dest_acc_en=False, ...), memory_config=ttnn.DRAM_MEMORY_CONFIG)`; PCC ≥ **0.99** vs golden, via `run_safe_pytest.sh` (no `--dev`, no timeout overrides).
- **Full production shape (`S=640`, `TOPK=2048`, `T=56320`)**: runs the op + a **mandatory sampled-row golden** — pick a handful of `(token,head)` rows, compute `sparse_mla` for just those rows (avoids the 2.8 GiB dense gather), assert PCC. Sweep `k_chunk_size ∈ {32, 64, 128}`.
- Edge cases (small shapes): all-valid, very-few-valid, token 0, boundary-tile `n_valid`, and **multi-chunk** (`TOPK > k_chunk_size`) including **fully-sentinel tail chunks** — not just the single-chunk `TOPK == k_chunk_size` case.
- **Layout-specific deterministic tests:** (a) QK tile-grid transpose — inputs where a grid-transpose bug gives a detectably wrong `scores`; (b) V rope-column exclusion — set KV's last 64 columns to large values and assert the output is unchanged; (c) Kᵀ-production in isolation.
- **Program-cache tests:** vary `scale` and `k_chunk_size` (both compile-time/hash inputs) and assert correct recompile.
- **[Stage 2] LLK tests (§11):** one tt-llk test per uplifted `{16,32}` MOP, asserting 32×32 regression + `{16,32}` correctness on BH and WH (`run-test` skill).

---

## 8. Phased implementation (validate each step on silicon via `run_safe_pytest.sh`)

### Stage 1 — `H=32`, full `{32,32}`, streaming `kt_inplace_v`
1. **Scaffold + identity**: all files; op validates inputs (BH gate, `H==32`, L1 budget), allocates output, writer emits a constant. Confirms registration/nanobind/cmake.
2. **Reader gather** (debug build, temporary `[…,576]` output): gather K + indices + zero-fill + mask; assert gathered rows == `KV[indices]`.
3. **Tilize/untilize round-trip** (debug build): row-major → tilize → untilize → write; assert output == input. (Phases 2–3 are throwaway bring-up kernels; the real op only emits `[1,H,S,512]`.)
4. **QK + masked softmax**: produce `probs`; compare to golden softmax (QK Kᵀ + grid-transpose, per-chunk mask, REDUCE_ROW softmax + scaler).
5. **PV + normalize**: full op; PCC ≥ 0.99 vs `sparse_mla` (`H=32`). **← Stage 1 e2e done.**
6. **Sweep `k_chunk_size`** + edge cases; perf baseline.

### Stage 2 — refinements (enablers land BEFORE H=16 integration)
7. **LLK `{16,32}` uplift** of the streaming MOPs (§11) + the dataflow enablers (2-face mask builder, tile-dims scaler) — one task each, each gated on its own LLK test. These are the prerequisites for `{16,32}` integration.
8. **`H=16` integration** (§6.4): host half-tile CB descriptors (`Tile({16,32})` on intermediates; **`col_identity` stays full** — §6.4) + `VectorMode::R` + 16-row writer emit. PCC ≥ 0.99 vs `sparse_mla` (`H=16`). (Contingency if the uplift stalls: the `compute_common` materialized path, §6.3 — correct but no overlap.)
9. **Truncation perf** (§6.1): reader `n_valid` → meta CB → compute truncates the loop.

---

## 9. Reference map (existing code to mirror)
- Host/infra: `sdpa/device/sdpa_device_operation.{hpp,cpp}`, `sdpa/device/sdpa_program_factory.cpp`, `sdpa/sdpa.cpp` (`flash_mla_prefill` :345), `sdpa/sdpa_nanobind.cpp` (:306), `sources.cmake`, `scripts/detect_legacy_device_op.py`.
- Reader: `sdpa/device/kernels/dataflow/dataflow_common.hpp` (page-table read :71, paged gather :255, zero-fill :25, barrier :20), `reader_interleaved.cpp` (transposed K read :451).
- Compute — granular `compute_streaming.hpp` functions we **call**: `blocked_matmul_and_pack` :286, `inplace_v_matmul_pack_batched` :338, `reduce_c_row_group` :378, `sub_exp_block_bcast_cols` :447, `sub_exp_first_col_blocks`/`salad_correct_fused` :543/:587, `normalize_row_streaming` :678; **mask = the hold-preserving L1-accumulate stamp `:1283` (NOT `add_block_inplace`, which pops its CB — `compute_common.hpp:560`).** **`sdpa_inner_loop_step` :1173 (`:1207-1760`) is the CB-lifecycle REFERENCE only — we do not call it.**
- Tilize/untilize: `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (asymmetric pages :106), `untilize_helpers.hpp`. Consumers: `experimental/cnn/convert_to_hwc`, `experimental/conv3d`, `transformer/sdpa_decode`.
- Reduce scaler / col-identity: `reduce_helpers_dataflow.hpp:94` / `.inl:221`, `generate_bcast_scalar.hpp:12`; writer pattern `writer_interleaved.cpp:78-84`.
- Row-major writer: `writer_unary_stick_layout_interleaved_start_id.cpp`; blocked-RM `layernorm_dataflow_utils.h:230` + `layernorm_compute_utils.h:86`.
- `{16,32}` precedent: `sdpa_flash_decode.cpp:209-221`, `sdpa_decode_program_factory.cpp:441-451`; LLK paths `llk_math_matmul.h:52`, `llk_math_reduce.h:175-220`.
- Golden reference: `models/demos/deepseek_v32/reference_cpu/sparse_sdpa_prefill.py`.

---

## 10. Debugging
- **craq-sim** (`https://github.com/tenstorrent/craq-sim`) — functional Tensix simulator for debugging the compute kernel on small synthetic cases (tiny S/TOPK/heads): step the matmul/softmax/Kᵀ/in-place-V math and reproduce mismatches deterministically before silicon. Not currently cloned — `git clone` under `/localdev/pjosipovic/refs/`.
- Silicon: `scripts/run_safe_pytest.sh` (built-in hang detection; no `--dev`, no timeout overrides). On hang: `./tools/tt-triage.py` in parallel, then `tt-smi -r`. `TT_METAL_WATCHER=1` only when chasing a hang.

---

## 11. [Stage 2] LLK `{16,32}` uplift of the streaming MOPs
Makes the 32-only `compute_streaming.hpp` custom MOPs work at `{16,32}` (2 faces) so the streaming compute path runs at `H=16`. Verification-first, one MOP per task, each gated on a tt-llk test (BH first, mirror WH). Use the `run-test`/`debug-kernel`/`arch-lookup`/`port-kernel` skills + `sage-blackhole`/`sage-wormhole`; debug small cases in craq-sim.

**Per MOP:** (1) baseline LLK test at `{32,32}`; (2) reproduce the `{16,32}` break; (3) parameterize on `num_faces` (drop the 4-face hardcode); (4) re-test `{16,32}` + `{32,32}` regression, then wire into the compute kernel and PCC the op.

MOPs to uplift:
- `reduce_block_max_row` (the running-max MOP) — `inner_loop=4`, `SETRWC CR_D,8` F2 jump, 4-face transpose replay (`tt_llk_blackhole/llk_lib/experimental/llk_math_reduce_custom.h:193,198,225,303-314`; WH mirror `llk_math_reduce_custom.h:165,170,250-261`); unpack side `llk_unpack_AB_reduce_custom.h:77,88`.
- `matmul_block_no_mop` — full-tile `LLK_ASSERT` (`llk_math_matmul_custom_no_mop.h:34-35`); add a 16×32-in0 replay mirroring std `is_in0_16x32`.
- `sub_tiles_bcast_cols_custom` — 4-face ELWSUBs + `8→32` `ADDR_MOD_6` jump (`llk_math_eltwise_binary_custom.h:82-102`); 2-face body (`num_faces` plumbed to init but ignored by exec).

Dataflow-side `{16,32}` enablers (also part of Stage 2, §6.4): a **2-face-aware vertical-mask builder** (`fill_vertical_tile_bf16` is 4-face-hardcoded, `dataflow_common.hpp:372`) and the tile-dims-aware reduce scaler (`calculate_and_prepare_reduce_scaler`, already 2-face-capable).

(`normalize_row_streaming` uses standard `matmul_block`, already `{16,32}`-safe.)

---

## 12. To verify during bring-up
- **Kᵀ-production block (§4.2/§6.2) — the riskiest new compute.** Confirm `cb_kt_in`'s convention against `sdpa_inner_loop_step` (grid-transposed tiles, original contents, QK `transpose=true`) and the exact `copy_tile`(CB→DST)→`pack_tile`(to `(d,sk)`) reposition sequence + its cost. Not `transpose_wh`, not the dataflow `copy_tile`.
- **CB lifecycle around the granular functions (§6.3):** we compose `blocked_matmul_and_pack` / `reduce_c_row_group` / `sub_exp_block_bcast_cols` / `inplace_v_matmul_pack_batched` / `normalize_row_streaming` ourselves, so replicate the exact `cb_qkt_im` reserve + hold-wr-ptr layout, `cur.sum` reserve, the **mask-add seam** (where the `-inf` add fits between QK-pack and reduce), and the deferred Kᵀ pop — using `sdpa_inner_loop_step`'s body (`:1207-1760`) as the authority. (We reference it; we don't call it.)
- **L1 budget (§5.6):** sum the actual CB set (incl. both `k_tiles` and `kt_in`) and re-derive the viable `k_chunk_size`.
- **bf16 accumulation:** confirm PCC ≥ 0.99 holds with `fp32_dest_acc_en=false` (MLA-prefill precedent suggests yes) at phase 5.
- **DRAM random-gather bandwidth:** the per-token indexed gather is the expected bottleneck; measure at phase 6 and tune `k_chunk_size` / barriers / truncation.
- **[Stage 2] half-tile CB plumbing:** confirm `CBFormatDescriptor::tile = {16,32}` on the intermediates, and `sdpa_decode`'s `im_tile` vs final-output split (which accumulators are `{16,32}` vs full).
