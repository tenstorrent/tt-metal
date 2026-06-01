# Ring MLA Implementation Plan

## Goal

Add a new `ttnn.transformer.ring_mla` binding for distributed MLA attention where K and V are represented by one tensor. The new op should not expose or internally require joint tensors. The op should gather, multicast/unicast, stage, and compute from a single KV tensor while using `head_dim_v` to describe the V hidden dimension inside that tensor.

The existing `ring_joint_scaled_dot_product_attention` path should stay available while this implementation is brought up.

## Target API

Add a new binding in `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`:

```python
ttnn.transformer.ring_mla(
    input_tensor_q,
    input_tensor_kv,
    *,
    persistent_output_buffer_kv,
    head_dim_v,
    logical_n,
    program_config,
    scale=None,
    compute_kernel_config=None,
    dim,
    multi_device_global_semaphore,
    num_links,
    cluster_axis,
    mesh_device,
    topology,
    subdevice_id=None,
    ccl_core_grid_offset,
    use_column_major_ccl=False,
    is_causal=False,
    is_balanced=False,
)
```

Return value should be `(output, stats)` unless Python compatibility requires a 3-tuple. There is no joint output because there are no joint tensors.

Assumed tensor convention:

- `input_tensor_q`: `[B, NHQ, local_q_seq, K_HEAD_DIM]`
- `input_tensor_kv`: `[B, NHKV, local_kv_seq, K_HEAD_DIM]`
- `persistent_output_buffer_kv`: `[B, NHKV, global_kv_seq, K_HEAD_DIM]`
- K uses all `K_HEAD_DIM` columns.
- V is the prefix view `input_tensor_kv[..., :head_dim_v]`.
- `head_dim_v` must satisfy `0 < head_dim_v < K_HEAD_DIM` and `head_dim_v % TILE_WIDTH == 0`.
- `NHQ % NHKV == 0`; each Q head maps to `kv_head = q_head / (NHQ / NHKV)`.

## New Host Operation

Add a dedicated operation instead of continuing to overload ring-joint latent-V behavior:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_mla_device_operation_types.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_mla_device_operation.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_mla_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_mla_program_factory.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_mla_program_factory.cpp`

Add public C++ wrappers in:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`

Initial implementation should copy only the useful pieces from `RingJointSDPADeviceOperation` and remove joint-specific state:

- Remove `joint_strategy`.
- Remove `joint_q`, `joint_k`, `joint_v`, `joint_output`, `L`, `Lt`, `num_joint_*`.
- Remove `input_v`, `gathered_v`, and `persistent_output_buffer_v`.
- Keep `logical_n`, `program_config`, causal/balanced/chunked prefill behavior, CCL attributes, and compute kernel config.
- Add `head_dim_v` as a reflected operation attribute and include it in the program hash.

Validation must reject:

- missing or zero `head_dim_v`
- `head_dim_v >= input_tensor_kv.logical_shape()[3]`
- non tile-aligned `head_dim_v`
- mismatched Q/K hidden dimensions
- non-device tensors, non-DRAM buffers, non-tile layout, unsupported dtypes
- `persistent_output_buffer_kv` shape not equal to all-gathered KV shape
- unsupported head mapping (`NHQ % NHKV != 0`)
- chunked prefill without `is_causal=True`

Output spec:

- output shape: `[B, NHQ, local_q_seq, head_dim_v]`
- stats shape should match current ring-joint stats behavior for the non-joint case: `[B, NHQ, 2 * local_q_seq_padded, 1]`

## CCL / All-Gather

The all-gather section should operate on exactly one tensor:

```cpp
all_gather_input_tensors = {input_tensor_kv};
all_gather_output_tensors = {persistent_output_buffer_kv};
```

The fused-op signaler can remain conceptually the same as ring-joint: SDPA reader/compute/writer wait for all-gather progress by ring iteration. The all-gather helper already accepts vectors of tensors, so the main change is removing the second K/V tensor path and validating that the vector size is one for `ring_mla`.

Runtime behavior:

- ring iteration 0 reads from local `input_tensor_kv`
- later ring iterations read from `persistent_output_buffer_kv`
- no joint tail is appended on the last ring iteration

## Chain Topology

Replace the current separate K-chain and V-chain model with one KV chain.

Chain scope:

- If `NHKV == 1`, use one batch-level chain per batch. This matches the MLA shared-KV case.
- If `NHKV > 1`, use one chain per `(batch, kv_head)` group. Each chain services all Q heads that map to that KV head.

Reader runtime args should carry only one chain config per core:

- participation flags
- injector/sink flags
- previous/next core
- mcast rectangle
- injector physical coordinate
- per-chain loop padding count

Multicast/unicast:

- Forward one full KV chunk of `Sk_chunk_t * K_HEAD_DIM_TILES`.
- Use the same chain for K access and V access.
- Mcast eligibility should be computed for the KV chain only. Do not build separate V mcast state.
- For padded mcast loop iterations, receivers/injectors still participate in the KV handshake, but no Q or compute-visible work is pushed.

## L1 Circular Buffers

Use one L1 CB for the KV tensor, but store two FIFO segments per processed K chunk:

- `cb_q_in`
- `cb_kv_in`
- mask/scalar/stats/output/intermediate CBs as needed

Remove separate `cb_k_in` and `cb_v_in` allocation from the new MLA kernels. The streaming helper can still receive two
logical CB parameters, but for `ring_mla` both should be the same `cb_kv_in` ID.

Sizing:

```cpp
kt_tiles = Sk_chunk_t * K_HEAD_DIM_TILES;
v_tiles = Sk_chunk_t * V_HEAD_DIM_TILES;
kv_tiles = 3 * kt_tiles;  // three fixed slots for MLA; kt_tiles is always >= v_tiles
```

Each K chunk is pushed in this order, with fixed-size FIFO entries:

```text
[K^T half: kt_tiles][V half: kt_tiles, first v_tiles valid]
```

The K segment uses the current streaming layout:

```text
kt_tile(kd, sk) = kd * Sk_chunk_t + sk
```

The V segment is a compact row-major prefix generated from the K segment:

```text
v_tile(sk, vd) = sk * V_HEAD_DIM_TILES + vd
source kt tile = vd * Sk_chunk_t + sk
0 <= vd < V_HEAD_DIM_TILES
```

The unused suffix of the V segment is don't-care. It exists only so reader and compute always reserve, push, wait, and pop
`kt_tiles`, keeping every FIFO entry the same size.

The MLA KV CB should be triple buffered. With only two `kt_tiles` entries, the reader can fetch the next K^T segment
while compute consumes the current V segment, but it cannot also materialize the next compact V segment until the current
V entry is popped. Three fixed-size entries allow the steady-state overlap to be:

```text
[current V][next K^T][next V]
```

after compute has popped the current K^T entry.

This keeps QK matmul utilization high while still using one CB and one transported KV chunk.

## Reader Kernel

Add a new reader kernel:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_mla_reader.cpp`

Start from `ring_joint_reader.cpp`, then simplify:

- Remove joint tensor accessors and joint branches.
- Remove local/gathered V accessors.
- Rename K accessors to KV accessors.
- Use `fetch_block(..., transpose=true)` for full-width KV downloads into the K^T segment.
- Build `TensorTileShape(B, NHKV, seq_tiles, K_HEAD_DIM_TILES)`.
- Fetch a full KV slice: `Slice(nb, nkv, row_start, row_end, 0, K_HEAD_DIM_TILES)`.
- Forward only the K^T segment through the one chain. Receivers generate their own V segment from the received K^T data.
- Push two FIFO segments per real compute iteration: K^T first, then compact V prefix.

Reader flow per real KV chunk:

1. Reserve `kt_tiles` in `cb_kv_in`.
2. Read or receive the full KV chunk into that reservation with K^T layout.
3. Forward/mcast the K^T segment through the single KV chain.
4. Push `kt_tiles` so streaming compute can start QK.
5. Reserve another `kt_tiles` half in the same `cb_kv_in`.
6. Locally transpose-copy only the V prefix from K^T into the beginning of that half.
7. Push `kt_tiles` so streaming compute can run QKT@V from the valid V prefix.

The V rematerialization copy can use local loopback NOC reads, following the same shape as the decode path's
`read_v<reuse_k>` helper:

```cpp
cb_reserve_back(cb_kv_in, kt_tiles);
uint32_t v_write_ptr = get_write_ptr(cb_kv_in);
for (uint32_t sk = 0; sk < Sk_chunk_t; ++sk) {
    uint64_t kt_read_ptr = get_noc_addr(kt_base_addr + sk * tile_bytes);
    for (uint32_t vd = 0; vd < V_HEAD_DIM_TILES; ++vd) {
        noc_async_read(kt_read_ptr, v_write_ptr, tile_bytes);
        v_write_ptr += tile_bytes;
        kt_read_ptr += Sk_chunk_t * tile_bytes;
    }
}
noc_async_read_barrier();
cb_push_back(cb_kv_in, kt_tiles);
```

Do not zero or fill the remaining `kt_tiles - v_tiles` pages unless a debug mode wants deterministic bytes. Compute must
never read past `v_tiles` in the V half.

For padded mcast loop iterations, participate in the K^T chain handshake only. Do not push K^T, do not generate V, and
do not expose data to compute.

## Compute Kernel

Add a new compute kernel:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_mla.cpp`

Streaming compute is the only implementation target for this phase. Start from the streaming path used by
`ring_joint_sdpa.cpp` and `compute_streaming.hpp`, then remove joint behavior and change the matmul views.

Out of scope for this phase:

- `compute_common.hpp`
- legacy `sdpa_ring`
- non-streaming/fp32-dest execution

Required compute changes:

1. QK matmul must consume the K^T segment from `cb_kv_in`.

   Logical operation:

   ```text
   Q[Sq, K_HEAD_DIM] @ K[Sk, K_HEAD_DIM]^T -> scores[Sq, Sk]
   ```

   Physical K^T tile index:

   ```text
   kt_tile_index(kd, sk) = kd * Sk_chunk_t + sk
   ```

   This preserves the existing streaming QK subblock shape and avoids the low-utilization `subblock_w=1` design.

### Streaming Q@K^T Detail

Current streaming QK path should remain the model:

- Reader downloads K with `transpose=true`.
- `cb_kt_in` physical tile layout is `[K_HEAD_DIM_TILES, Sk_chunk_t]`.
- `blocked_matmul_and_pack<true, KT_stride, KT_stride>` can process `actual_sbw` K columns at once because, for a
  fixed hidden-dim tile, the next K columns are contiguous in L1.

New MLA path:

- Reader downloads full KV with `transpose=true`, exactly like K today.
- The transported K^T segment is full width: `K_HEAD_DIM_TILES x Sk_chunk_t`.
- The local V prefix is generated as a second segment after K^T is pushed.
- Compute can pass the same CB ID for both logical inputs:

  ```cpp
  cb_kt_in = cb_kv_in;
  cb_v_in = cb_kv_in;
  ```

The QK phase then becomes:

```cpp
cb_wait_front(cb_kv_in, K_HEAD_DIM_TILES * Sk_chunk_t);
mm_no_mop_init_short(cb_q_in, cb_kv_in, true, actual_sbw, qkt_subblock_h, K_HEAD_DIM_TILES);
blocked_matmul_and_pack<true, Sk_chunk_t, Sk_chunk_t>(...);
cb_pop_front(cb_kv_in, K_HEAD_DIM_TILES * Sk_chunk_t);
```

That pop releases the K^T entry while the V entry remains at the front of the same CB. With three fixed-size entries,
the reader can reserve/fill the next chunk's K^T entry and then materialize that chunk's compact V entry while current
QKT@V work is still consuming the older V entry.

2. V matmul must use the same `cb_kv_in`.

   Logical operation:

   ```text
   scores[Sq, Sk] @ V[Sk, V_HEAD_DIM] -> output[Sq, V_HEAD_DIM]
   ```

   Physical V tile index:

   ```text
   v_tile_index(sk, vd) = sk * V_HEAD_DIM_TILES + vd
   ```

   The V segment is compact, so the existing QKT@V stride assumptions remain valid.

The streaming QK@V path can keep using the current helper shape:

```cpp
cb_wait_front(cb_kv_in, K_HEAD_DIM_TILES * Sk_chunk_t);
blocked_matmul_and_pack<false, V_HEAD_DIM_TILES, V_HEAD_DIM_TILES>(
    cb_qkt_im,
    cb_kv_in,
    out_cb,
    qktv_in0_index_offset + kt_sub * matmul_inner,
    kt_sub * matmul_inner * V_HEAD_DIM_TILES + v_index_offset,
    ...);
cb_pop_front(cb_kv_in, K_HEAD_DIM_TILES * Sk_chunk_t);
```

This avoids QKT@V matmul indexing changes beyond using `cb_kv_in` as the logical V CB. The helper still reads only the
compact V prefix; the pop releases the entire fixed-size V half.

3. CB synchronization must treat KV as one producer/consumer item.

   - wait on `K_HEAD_DIM_TILES * Sk_chunk_t`
   - run QK using the K^T segment
   - pop `K_HEAD_DIM_TILES * Sk_chunk_t`
   - wait on `K_HEAD_DIM_TILES * Sk_chunk_t`
   - run QKT@V using the compact V segment
   - pop `K_HEAD_DIM_TILES * Sk_chunk_t`

This intentionally mirrors the existing streaming sequence:

```cpp
cb_pop_front(cb_kt_in, DHt * Sk_chunk_t);
...
cb_pop_front(cb_v_in, Sk_chunk_t * V_HEAD_DIM_TILES);
```

but both logical CB IDs refer to the same physical `cb_kv_in`.

The MLA streaming wrapper should distinguish V's visible tile count from V's CB entry size:

```cpp
v_visible_tiles = Sk_chunk_t * V_HEAD_DIM_TILES;
v_cb_entry_tiles = Sk_chunk_t * K_HEAD_DIM_TILES;
```

The matmul reads only `v_visible_tiles`; CB wait/pop uses `v_cb_entry_tiles`.

4. Keep mask, causal, balanced, chunked-prefill, deferred normalization, and stats logic equivalent to the non-joint ring-joint path.

Host validation should reject compute configurations that would select the non-streaming path. In practice, this means
rejecting fp32 destination accumulation for `ring_mla` until there is an explicit follow-up to add a non-streaming
implementation.

## Writer Kernel

Add a new writer kernel or simplify the existing ring-joint writer:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_mla_writer.cpp`

Required changes:

- Remove joint output writes.
- Keep output writes for `[B, NHQ, local_q_seq, head_dim_v]`.
- Keep stats save/restore logic used by ring iterations and multi-Q-chunk deferred normalization.
- Keep lightweight mask tile generation, but remove joint-L mask generation.

## Program Factory

The new program factory should:

- derive `K_HEAD_DIM_TILES = input_tensor_kv.logical_shape()[3] / TILE_WIDTH`
- derive `V_HEAD_DIM_TILES = head_dim_v / TILE_WIDTH`
- allocate one physical `cb_kv_in`
- pass the same `cb_kv_in` ID wherever the streaming helper expects logical `cb_kt_in` and `cb_v_in`
- append one KV tensor accessor for local KV and one for gathered KV
- create one set of chain semaphores
- build one KV chain configuration
- pass one KV chain config in reader runtime args
- call all-gather helper with one tensor
- remove all joint-related compile-time args and runtime args

Compile-time args should clearly separate:

- `NHQ`
- `NHKV`
- `K_HEAD_DIM_TILES`
- `V_HEAD_DIM_TILES`
- `Sq_chunk_t`
- `Sk_chunk_t`
- `q_local_padded_Nt`
- `kv_local_padded_Nt`
- `padded_Nt`
- `logical_n`
- `logical_nt`
- local chunk counts
- ring size
- causal/balanced/chunked flags
- CB IDs

Avoid preserving old argument slots unless doing so materially reduces implementation risk. A new op can use a cleaner argument layout.

## Tests

Add or split tests under:

- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_mla.py`
- `tests/nightly/blackhole/sdpa/test_ring_mla.py`

Use `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` as the reference structure for:

- model config dataclasses and hardware-aware mesh config generation
- PCC/RMSE correctness checks
- repeated-run determinism checks
- local perf sweeps, perf table generation, and env-gated perf checks

Functional reference:

```python
K = KV
V = KV[..., :head_dim_v]
ref = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1) @ V
```

Test matrix:

- `head_dim_k=576`, `head_dim_v=512`
- `head_dim_v` tile-aligned but smaller than K
- invalid `head_dim_v == head_dim_k`
- invalid non-tile-aligned `head_dim_v`
- `NHKV=1`, many Q heads
- 2x2 mesh runnable on this host
- causal and non-causal as supported
- balanced and non-balanced as supported
- chunk sizes chosen to fit the new single `cb_kv_in` L1 footprint

Regression checks:

- compare PCC against PyTorch reference
- verify deterministic output for repeated runs with identical inputs, matching the ring-joint pattern of comparing
  each output to iteration 0 exactly
- run at least one trace/no-trace path if trace is supported
- verify all-gather only receives one input/output tensor
- verify debug logs show one KV chain mode, not separate K and V chains

Performance checks:

- Add a skipped-on-CI perf sweep equivalent to `test_ring_joint_attention_sdpa_sweep_perf_impl`.
- Add a perf table generator equivalent to `test_ring_joint_attention_create_perf_table`.
- Add an env-gated perf check equivalent to `test_ring_joint_attention_perf_check`.
- Include paired baseline runs with the same `B`, `seq_len`, `NHQ`, `NHKV`, `K_HEAD_DIM`, `V_HEAD_DIM`, dtype,
  topology, and chunk sizes:
  - specialized `ring_mla` with fused KV tensor
  - non-MLA/generic baseline with equivalent K and separate V materialized from `KV[..., :head_dim_v]`
- The specialized `ring_mla` path must be faster than the non-MLA baseline for the same input shape. The perf check
  should assert lower device kernel duration and equal-or-better effective math utilization, using the same profiler
  post-processing style as the nightly ring-joint test.

Build and test commands:

```bash
./build_metal.sh --release
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_mla.py -q
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_mla.py -q
```

Use `scripts/run_safe_pytest.sh` for test execution, including focused node runs and `-k` filters. Use `--dev` when
debugging hangs/asserts and `--run-all` when full pass/fail counts are needed instead of fail-fast behavior. Any perf
test that launches pytest in a subprocess should build the subprocess command around `scripts/run_safe_pytest.sh`, not a
bare `pytest` command.

## Bring-Up Order

1. Add the public API skeleton and host validation with no kernel changes.
2. Add `RingMLADeviceOperation` and output specs.
3. Add program factory that builds descriptors for one KV tensor, one output, one stats tensor.
4. Add reader kernel with one KV CB, transposed full-width K^T fetch, compact V-prefix rematerialization, and one chain.
5. Wire streaming compute so logical K and V inputs both use the same physical `cb_kv_in`.
6. Add writer kernel without joint output.
7. Wire CCL all-gather with one tensor and verify descriptor/runtime args.
8. Add Python correctness, determinism, performance, and invalid-argument tests.
9. Build with `./build_metal.sh --release`.
10. Run focused 2x2 `ring_mla` tests with `scripts/run_safe_pytest.sh`, then broader device-count/nightly tests where
    hardware is available.

## Main Risks

- The single `cb_kv_in` must have enough capacity for two fixed-size halves:
  `2 * Sk_chunk_t * K_HEAD_DIM_TILES` tiles.
- Because K^T and V share one physical CB, reader push order and compute pop order must remain exactly:
  reserve/push `kt_tiles` for K^T, reserve/push `kt_tiles` for V, pop `kt_tiles` for K^T, pop `kt_tiles` for V.
- Multicast loop padding must be based on the single KV chain's work count, not the old separate K/V chain counts.
- Local V rematerialization adds L1-to-L1 NOC traffic, but only for `V_HEAD_DIM_TILES` and preserves QK matmul
  utilization.
- Full KV forwarding uses `K_HEAD_DIM_TILES`, so chunk sizes still need explicit validation against available L1.
- Program cache hashing must include `head_dim_v`; otherwise different V hidden dimensions could reuse an incompatible program.

## Acceptance Criteria

- Python exposes `ttnn.transformer.ring_mla`.
- No joint tensors are required by the public API or by internal tensor args.
- CCL all-gather uses exactly one tensor.
- Reader downloads full-width KV as K^T on injector cores and rematerializes only the compact V prefix locally.
- Store-and-forward uses one mcast/unicast chain for KV.
- L1 uses one physical KV CB; compute consumes K^T and V as two FIFO segments from that same CB.
- Only streaming compute is implemented; non-streaming/fp32-dest configs fail validation clearly.
- `head_dim_v` is validated as smaller than K dim and tile aligned.
- 2x2 MLA correctness tests pass against PyTorch reference.
- Determinism tests pass for repeated runs with identical inputs.
- Performance tests show specialized `ring_mla` is faster than the non-MLA/generic baseline for the same input shape.
- `./build_metal.sh --release` passes.
- Tests are run through `scripts/run_safe_pytest.sh`.
