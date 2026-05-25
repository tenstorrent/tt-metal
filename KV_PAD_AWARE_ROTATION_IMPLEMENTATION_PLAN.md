# KV Pad-Aware Rotation Implementation Plan

## Branch and Source Context

- Current local branch: `ipotkonjak/chunked_attn_tests`
- Current HEAD: `1a7c4e108bcea6fccd8918b500a9523e9ab2a7ce`
- Referenced commit in this branch: `dd7875d64203f61a320b01854ea61f13e4ccb374`
- Commit summary: `ring_joint_sdpa handoff: KV-pad-aware rotation test + HTML visualizer`
- Visualizer: `models/demos/deepseek_v3_d_p/tests/kv_pad_rotation_visualizer.html`
- Canonical torch spec: `_kv_pad_rotation_layout` and `test_kv_pad_aware_rotation_torch_showcase` in `models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py`

The branch already contains the referenced commit plus newer ring-joint SDPA fixes, so the plan should be implemented on the branch head rather than detached at `dd7875d6420`.

## Feature Goal

Implement KV-pad-aware chunked prefill in `ttnn.transformer.ring_joint_scaled_dot_product_attention`.

The production case is:

- Prior cache contains `kv_actual_isl` valid tokens.
- The current input/chunk has fixed shape `chunk_size`, but only `new_actual_isl` tokens are valid.
- The server places new tokens into OLD-slab pad cells first, then into NEW slabs from device 0 onward.
- Q/K causal positions must be derived from the rotated per-device layout, not from the existing balanced-growing formula.

The output can remain in the same per-device Q-row order as the input Q. Tests should unrotate valid rows back to natural order before comparing to a natural causal reference.

## Current Limitation

The current chunked path assumes a monotonic balanced-growing local-to-global K mapping:

```cpp
kv_global_tile_for_local(ring_id, local_tile_idx)
```

and computes chunked Q starts from:

```cpp
q_start_idx_t = logical_nt - q_local_padded_Nt * ring_size
```

That works for regular chunked prefill, but fails when a single device's Q rows can represent two global-position regions:

- pad-fill rows that write into OLD slab holes;
- NEW-slab rows that continue after OLD pad has been consumed.

`logical_n` alone is insufficient because it only gives the total valid KV length after the iteration. The op also needs the split point: `kv_actual_isl`, the valid cache length before this iteration. Then it can derive `new_actual_isl = logical_n - kv_actual_isl`.

## Proposed Public API

Add one optional parameter to `ring_joint_scaled_dot_product_attention`:

- `kv_actual_isl`: valid global tokens already in cache before this iteration.

Derive:

- `new_actual_isl = logical_n - kv_actual_isl`

Activation rule:

- If neither is passed, preserve the existing behavior exactly.
- If passed, `kv_actual_isl` and the derived `new_actual_isl` must be tile-aligned for the first implementation.
- Rotated mode requires `is_causal=True` and chunked input (`Q.seq < K.seq`).
- `logical_n` must be `>= kv_actual_isl`.

Files to update:

- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp`

Include the new params in op attributes and the program hash so program-cache reuse cannot retain stale scalar rotation state.

## Core Mapping Design

Add a shared helper in:

```text
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp
```

The helper should operate in tile units first:

- `kv_actual_nt = kv_actual_isl / TILE_HEIGHT`
- `new_actual_nt = new_actual_isl / TILE_HEIGHT`
- `chunk_size_local_nt = q_local_padded_Nt`
- `old_capacity_nt = ring_size * chunk_size_local_nt`

It should provide:

- `pad_fill_on_chip(chip)`
- `phase2_prefix_before_chip(chip)`
- `q_global_tile_or_invalid(chip, q_tile)`
- `k_global_tile_or_invalid(chip, local_k_tile)`
- `k_chunk_has_valid_cols(chip, k_chunk_start, Sk_chunk_t)`

Expected mapping:

- OLD slab rows are valid up to `min(old_capacity_nt, kv_actual_nt + new_actual_nt)`.
- OLD pad-fill positions keep their OLD-slab global positions.
- NEW slab rows receive only the remaining tokens after OLD pad-fill.
- Q rows follow the exact destination order from `_kv_pad_rotation_layout`.
- Any Q/K row outside these valid regions is padding and must be masked.

Do not replace the existing balanced helper. Add a rotated helper and dispatch based on `kv_pad_rotation_enabled`.

## Kernel Wiring

### Program Factory

Update:

```text
ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp
```

Add compile-time args for:

- `kv_pad_rotation_enabled`
- `kv_actual_nt`
- `new_actual_nt` derived by the host as `logical_nt - kv_actual_nt`

First pass can keep these as compile-time args to match current `logical_n` behavior. A later optimization can move them to runtime args if trace/program-cache pressure matters.

Pass the new args consistently to:

- reader compile-time args;
- writer compile-time args;
- compute compile-time args.

### Reader

Update:

```text
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp
```

Replace rotated-mode K skip decisions with `k_chunk_has_valid_cols`.

The reader and compute must skip the same K chunks. If compute skips a K chunk but reader pushes it, or reader skips a chunk compute expects, the CB protocol can deadlock.

### Compute

Update:

```text
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp
```

Add a rotated lightweight mask path that stamps `-inf` when:

- the Q tile is padding;
- the K tile is padding;
- `K_global_tile > Q_global_tile`;

and stamps the existing causal diagonal tile when:

- `K_global_tile == Q_global_tile`.

The existing causal mask supports contiguous K positions plus one straddle jump. Rotated mode needs per-column K position checks because a K chunk can include invalid columns or cross OLD/NEW slab boundaries.

Apply this to both compute paths:

- non-streaming `sdpa_ring`;
- streaming `sdpa_ring_v2`.

### Writer

Update:

```text
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp
```

Keep output storage in local Q order. For rotated mode, use the rotated Q validity helper when deciding restore/save ranges and final output writes. Padding Q rows must not produce meaningful output rows.

## Model Integration

Update the DeepSeek MLA call site after the op is correct:

```text
models/demos/deepseek_v3_d_p/tt/mla/mla.py
```

Pass:

- `kv_actual_isl` from prior cache length for this request/layer;
- `logical_n` as total valid length after this iteration.

Current full-prefill call uses `logical_n=seq_len_local * sp_factor`, which intentionally treats padding as valid. That must change for small-ISL chunked prefill.

Implementation status: the op/API/kernel support and targeted tests are implemented. Direct MLA wiring is intentionally deferred because the current `mla.py` call site is the full-prefill path where `Q.seq == K.seq`; rotated mode validates the chunked shape where `K` contains one OLD slab plus one NEW slab and `Q` is the current chunk. The future chunked MLA caller should pass `kv_actual_isl` as the cache length before the iteration and `logical_n` as the valid total after the iteration.

## Test Plan

1. Preserve the torch showcase:

```bash
pytest models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py -k kv_pad_aware_rotation_torch_showcase
```

2. Add a device test next to the handoff tests that builds the same rotated Q/K/V layout and calls `ring_joint_scaled_dot_product_attention` with the new params.

Start with these cases from the visualizer/spec:

- `single_pad_dev3`: `kv_actual_isl=224`, `new_actual_isl=64`
- `single_pad_exact_fill`: `kv_actual_isl=224`, `new_actual_isl=32`
- `multi_pad_2_full`: `kv_actual_isl=128`, `new_actual_isl=128`
- `multi_pad_2_partial`: `kv_actual_isl=128`, `new_actual_isl=96`
- `cold_start`: `kv_actual_isl=0`, `new_actual_isl=128`
- `no_old_pad`: `kv_actual_isl=256`, `new_actual_isl=64`

Compare only valid Q rows, reordered to natural global-position order, against a natural-order torch causal MLA-SDPA reference.

3. Re-run existing ring-joint handoff coverage:

```bash
pytest models/demos/deepseek_v3_d_p/tests/test_ring_joint_sdpa_handoff.py -k "nd_sharded_kv_cache_as_k or index_based_kv_access or persistent_buffer or chunked_q_start_idx"
```

4. Re-run focused MLA op tests:

```bash
pytest models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py -k ring_joint
```

5. After model integration, add or update a small-ISL prefill/block test that passes `actual_isl < seq_len` and checks PCC against the host reference.

## Implementation Order

1. Add API/attribute fields, validation, bindings, and hash entries.
2. Add the shared tile-level rotated layout helper.
3. Wire compile-time args through program factory, reader, writer, and compute.
4. Implement rotated K/Q skip and mask logic for non-streaming compute.
5. Implement the same logic for streaming compute.
6. Add the device-level rotated-layout test and make it pass.
7. Update `mla.py` to pass actual cache/new lengths.
8. Run regression tests and trim unsupported cases with explicit validation rather than silent wrong output.

## Open Decisions

- Whether to support sub-tile derived `new_actual_isl` in the first implementation. The handoff test is tile-aligned; first pass should validate tile alignment and defer sub-tile masks.
- Whether rotated mode must support nonzero joint attention immediately. DeepSeek MLA currently uses dummy zero-length joint tensors, so first pass can validate `L == 0` if needed.
- Whether the model wants output in rotated Q order or natural order. The op should initially preserve Q input order; any natural-order reassembly should remain a caller/test concern unless the model pipeline requires otherwise.
