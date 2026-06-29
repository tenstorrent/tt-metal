# Metal 2.0 Op Migration — Status, Pybind Dependency & Migrability

> Auto-generated census of every ttnn op with a `device/` factory, classified by current
> framework, descriptor-pybind dependency, and migrability to Metal 2.0. Status reflects
> `main`; ✅-migrated ops currently live on branch `dgomez/metal2-bulk-migration` (unmerged).
> Migrability for un-annotated descriptor ops is *presumed* — only ~28 ops have been
> individually gap-assessed (the 2026-06-11 bulk run). Treat 🟢 as "no known blocker", not "verified".

## Why this doc exists

Metal 2.0 will **remove the `tt::tt_metal::ProgramDescriptor` type entirely**. We cannot delete
it until every op on the descriptor path either (a) migrates to Metal 2.0 (`ProgramSpec` +
`ProgramRunArgs`), or (b) is identified as **non-migrable** and carried by a relocated,
**ttnn-owned** descriptor concept. This doc identifies which is which.

## Three scenarios (current `main`)

| Scenario | Marker | Count |
|---|---|---|
| **1. Legacy factory** | `create_program` / `override_runtime_arguments` | 63 |
| **2. Descriptor** (Contract-1) | `create_descriptor` / `WorkloadDescriptor` | 149 |
| **3. Metal 2.0** | `create_program_spec` | 0 on main (14 on bulk branch) |
| Composite / dispatch-only | none of the above | 5 |
| **Total** | | 217 |

## Migrability legend

- ✅ **migrated** — done on the bulk branch (some partial per-variant)
- 🟢 **presumed migrable** — descriptor op, no known blocker, not yet individually assessed
- 🟡 **gap-blocked (#N)** — blocked by a Metal 2.0 API gap; per Audrey's triage these are *temporary*
- 🟠 **decision pending** — needs a human call (matmul shared-hash, gap #7)
- 🟣 **XL** — large/expensive port, deferred
- 🔵 **keep descriptor emitter** — op's dispatch can migrate, but it must retain a standalone `create_descriptor` builder for the Python fusion ecosystem
- 🔴 **irreducible** — descriptor *is* the op's contract; cannot become a fixed `ProgramSpec`. Defines the permanent ttnn-owned descriptor concept
- ⚪ **legacy** — not on descriptor yet; migrate legacy → Metal 2.0 directly
- ❓ **composite** — dispatch/decompose op; verify whether it owns a factory

## The 7 Metal 2.0 gaps (blockers)

1. **Conditional token** — genfiles emits `dfb::/ta::` tokens only for bound resources; `if constexpr` optional-feature kernels fail. 2. **Op-owned device tensors** — no owning slot in run args (`tensor_args` is non-owning). 3. **Compute-only / borrowed binding** — every TensorParameter needs a `TensorBinding`, but tensor accessors only JIT on data-movement kernels. 4. **No runtime/variadic/indexed tensor accessors** — `ta::name` is a per-binding compile-time symbol. 6. **Varargs have no per-vararg `enqueue_invariant`** — blocks the ++ fast path for tree/mcast ops. 7. **No-custom-hash `static_assert` is per-device-op, not per-factory** — a shared custom hash blocks all variants. (#5 was a false alarm — semaphores are supported.)

## The non-migrable residue → the ttnn-owned descriptor concept

These never become fixed `ProgramSpec` factories. Tackle them *together with* the descriptor
removal; afterward, `descriptor` lives purely as a ttnn authoring/fusion IR that lowers to
`ProgramSpec` via vararg-passthrough:

- **Vehicles** — `generic` (`generic_op`), `experimental/fusion` (`fusion_dispatch_op`)
- **C++ composer** — `ccl/mesh_partition` (consumes `SliceOp::create_descriptor`)
- **Emitters** — `matmul`, `data_movement/slice`, `normalization/layernorm` export `create_descriptor` to Python; must keep a *standalone* builder (extracted from their `program_factory_t`, since `AllFactoriesValid` forbids mixing) even after their dispatch migrates
- **Python consumers** (not ttnn ops) — DeepSeek V3 B1 (~54 ops), `models/experimental/ops/descriptors/`, `deepseek_v3`, `deepseek_v3_d_p`

## Full op census

| Op | Status | Pybind dep | Migrable to Metal 2.0? |
|---|---|---|---|
| `bernoulli` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `ccl/all_broadcast` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `ccl/all_gather` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `ccl/all_to_all_combine` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `ccl/all_to_all_dispatch` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `ccl/broadcast` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `ccl/mesh_partition` | descriptor | — | 🔴 irreducible — C++ composer: builds program from SliceOp::create_descriptor via std::visit |
| `ccl/reduce_scatter` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `ccl/reduce_to_root` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `conv/conv2d` | descriptor | — | 🟡 gap-blocked (#2) — likely op-owned tensors (unconfirmed) |
| `copy/typecast` | descriptor | — | ✅ migrated (bulk branch) |
| `data_movement/bcast` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/clone` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/concat` | descriptor | — | 🟡 gap-blocked (#3) — compute-only/borrowed binding; variadic N inputs (#4) |
| `data_movement/copy` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/fill_pad` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/fill_rm` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/fold` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/gather` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/indexed_fill` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/moe_expert_token_remap` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/moe_routing_remap` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/move` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/non_zero_indices` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/pad` | descriptor | — | ✅ migrated (bulk branch) |
| `data_movement/permute` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/repeat` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/reshape_on_device` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/reshape_view` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/reshape_view/device` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/scatter` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/sharded` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/sharded/interleaved_to_sharded` | descriptor | — | 🟡 gap-blocked (#4) — compile-time-vararg shard addrgen |
| `data_movement/sharded/reshard` | descriptor | — | 🟡 gap-blocked (#4) — runtime page-stride maps |
| `data_movement/sharded/sharded_to_interleaved` | descriptor | — | ✅ migrated (bulk branch) |
| `data_movement/sharded_partial/interleaved_to_sharded_partial` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/sharded_partial/sharded_to_interleaved_partial` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/slice` | descriptor | exports descriptor | 🟢 presumed migrable (not individually assessed) · 🔵 keep descriptor emitter for fusion |
| `data_movement/sort` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/split` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/tilize` | descriptor | — | ✅ migrated (bulk branch) |
| `data_movement/tilize_with_val_padding` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `data_movement/transpose` | descriptor | — | ✅ migrated (bulk branch) |
| `data_movement/untilize` | descriptor | — | ✅ migrated (bulk branch) |
| `data_movement/untilize_with_unpadding` | descriptor | — | ✅ migrated (bulk branch) |
| `debug` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `eltwise/binary` | composite? | — | ❓ composite/dispatch — verify it has its own factory |
| `eltwise/binary_ng` | descriptor | — | 🟡 gap-blocked (#4) — chained tensor accessors on plain interleaved path (fatal) |
| `eltwise/complex_binary` | composite? | — | ❓ composite/dispatch — verify it has its own factory |
| `eltwise/complex_unary` | composite? | — | ❓ composite/dispatch — verify it has its own factory |
| `eltwise/complex_unary_backward` | composite? | — | ❓ composite/dispatch — verify it has its own factory |
| `eltwise/ternary` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `eltwise/unary` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `eltwise/unary_backward/gelu_bw` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `eltwise/unary_backward/tanh_bw` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `embedding` | descriptor | — | 🟡 gap-blocked (#3) — tiled-sharded compute-only binding |
| `embedding_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `examples/example` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `examples/example_multiple_return` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/bcast_to` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/all_gather_async` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/ccl/all_gather_concat_heads_fused` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/all_gather_matmul_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/all_gather_minimal_matmul_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/all_reduce_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/all_to_all_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/all_to_all_async_generic` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/all_to_all_dispatch_metadata` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/deepseek_moe_reduce_scatter` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/llama_all_gather_matmul_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/llama_reduce_scatter` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/llama_reduce_scatter_create_heads` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/llama_reduce_scatter_matmul` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/matmul_reduce_scatter_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/minimal_matmul_strided_reduce_scatter_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/moe/selective_reduce_combine` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/moe_compute` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/moe_gpt` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/neighbor_pad_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/reduce_scatter_minimal_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/ring_attention_all_gather_async` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/ccl/rms_allgather` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/send_recv_async/recv_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/send_recv_async/send_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/slice_reshard_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/strided_all_gather_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/strided_all_gather_minimal_matmul_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ccl/strided_reduce_scatter_async` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/cnn/convert_to_chw` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/cnn/convert_to_hwc` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/conv3d` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/deepseek/mla/matmul_wo` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek/moe/deepseek_moe_gate` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek/moe/moe_gate_mm` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_moe_post_combine_tilize` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/combine` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/deepseek_prefill/dispatch` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/deepseek_prefill/extract` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/insert` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/masked_bincount` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/moe_grouped_topk` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/offset_cumsum` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/deepseek_prefill/per_token_cast_back` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/per_token_cast_to_fp8` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/post_combine_reduce` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/deepseek_prefill/rotary_embedding_indexed` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/deepseek_prefill/unified_routed_expert_ffn` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/deepseek_prefill/update_padded_kv_cache` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/dropout` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/fusion` | descriptor | exports descriptor | 🔴 irreducible — vehicle: runs user/codegen descriptor + dynamic fusion + patching |
| `experimental/isin` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/matmul/attn_matmul` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/matmul/group_attn_matmul` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/minimal_matmul` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/multi_scale_deformable_attn` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/padded_slice` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/paged_cache` | descriptor | — | 🟡 gap-blocked (#1) — conditional dfb/ta token in optional scalar path |
| `experimental/plusone` | descriptor | — | ✅ migrated (bulk branch) |
| `experimental/reduction/deepseek_grouped_gate` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/reduction/deepseek_moe_fast_reduce_nc` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/reduction/deepseek_moe_fast_reduce_nc_fused` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/reduction/fast_reduce_nc` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/reduction/integral_image` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/slice_write` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/ssm/hc_sum_reduce` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/ssm/prefix_scan` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/ssm/repeat_and_interleave_eltwise_mul` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/test/hang_device` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/topk_router_gpt` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/all_reduce_create_qkv_heads` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/concatenate_heads` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/create_qkv_heads` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/create_qkv_heads_from_separate_tensors` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/dit_layernorm_post_all_gather` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/dit_layernorm_pre_all_gather` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/fused_distributed_rmsnorm` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/nlp_concat_heads` | descriptor | — | ✅ migrated (bulk branch) |
| `experimental/transformer/nlp_concat_heads_boltz` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/nlp_concat_heads_decode` | descriptor | — | ✅ migrated (bulk branch) |
| `experimental/transformer/nlp_create_qkv_heads` | descriptor | — | ✅ migrated (bulk branch) |
| `experimental/transformer/nlp_create_qkv_heads_boltz` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/nlp_create_qkv_heads_decode` | descriptor | — | 🟡 gap-blocked (#1) — batch_offset optional-feature token |
| `experimental/transformer/nlp_create_qkv_heads_falcon7b` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/nlp_create_qkv_heads_segformer` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/nlp_create_qkv_heads_vit` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/nlp_kv_cache_load_slice` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/rotary_embedding` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/rotary_embedding_hf` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `experimental/transformer/rotary_embedding_llama` | descriptor | — | 🟡 gap-blocked (#3) — compute-only sharded decode binding |
| `experimental/transformer/rotary_embedding_llama_fused_qk` | descriptor | — | 🟡 gap-blocked (#3) — compute-only sharded binding |
| `experimental/transformer/rotate_half` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/transformer/split_query_key_value_and_split_heads` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `experimental/unary_backward/gelu_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `full` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `generic` | descriptor | exports descriptor | 🔴 irreducible — vehicle: runs an arbitrary user-built ProgramDescriptor |
| `index_fill` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `kv_cache` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `matmul` | descriptor | exports descriptor | 🟠 decision pending (#7) — 1 shared custom hash across 6 factories — DECISION PENDING · 🔵 keep descriptor emitter for fusion |
| `moreh/moreh_abs_pow` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_adam` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_adamw` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_arange` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step2` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_dot` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_dot_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_fold` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_getitem` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_group_norm` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_group_norm_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_layer_norm` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_layer_norm_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_linear_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_matmul` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_mean` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_mean_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_nll_loss/moreh_nll_loss_step1` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_nll_loss/moreh_nll_loss_step2` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_nll_loss_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_nll_loss_unreduced_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_norm` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_norm_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_sgd` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_softmax` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_softmax_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_sum` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `moreh/moreh_sum_backward` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `normalization/batch_norm` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `normalization/groupnorm` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `normalization/layernorm` | descriptor | exports descriptor | 🟣 XL — 23 kernels; not attempted · 🔵 keep descriptor emitter for fusion |
| `normalization/layernorm_distributed` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `normalization/rmsnorm_distributed` | composite? | — | ❓ composite/dispatch — verify it has its own factory |
| `normalization/softmax` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `point_to_point` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `pool` | descriptor | — | 🟡 gap-blocked (#2) — op-owned halo lookup + scalar config tensors |
| `pool/generic` | descriptor | — | 🟡 gap-blocked (#2) — op-owned halo lookup + scalar config tensors |
| `pool/grid_sample` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `pool/rotate` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `pool/upsample` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `prefetcher/prefetcher` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `rand` | descriptor | — | ✅ migrated (bulk branch) |
| `randn` | legacy | — | ⚪ legacy — migrate legacy→metal2 (not yet on descriptor) |
| `reduction/accumulation` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `reduction/accumulation/ema` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `reduction/argmax` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `reduction/generic` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `reduction/manual_seed` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `reduction/moe` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `reduction/prod` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `reduction/sampling` | descriptor | — | ✅ migrated (bulk branch) |
| `reduction/topk` | descriptor | — | ✅ migrated (bulk branch) |
| `sliding_window/halo` | descriptor | — | 🟡 gap-blocked (#2) — 4 op-owned config tensors |
| `transformer/sdpa` | descriptor | — | 🟡 gap-blocked (#6) — tree-reduction varargs |
| `transformer/sdpa_decode` | descriptor | — | 🟡 gap-blocked (#6) — per-vararg enqueue_invariant for cur_pos (++ fast path) |
| `transformer/sdpa_windowed` | descriptor | — | 🟢 presumed migrable (not individually assessed) |
| `uniform` | descriptor | — | 🟢 presumed migrable (not individually assessed) |

## Plan (3 tracks)

1. **Migrate** — drive the 🟢/🟡 descriptor ops to Metal 2.0 (the bulk-migration project, gated on Audrey's gap fixes). Empties scenario 2 down to the residue.
2. **Extract emitters** — pull the `create_descriptor` builders for `slice`/`matmul`/`layernorm` out of their `program_factory_t` into standalone builders; re-point `mesh_partition` at slice's builder. Now their dispatch can migrate without violating `AllFactoriesValid`.
3. **Relocate + lower** — move the descriptor structs `tt_metal` → `ttnn`; dispatch the residue via vararg-passthrough lowering (ttnn descriptor → Metal 2.0 `ProgramSpec`). Works because residue kernels use raw-address positional args — zero gap features needed. **Open risk to confirm first:** Metal 2.0 runtime must accept raw L1/DRAM addresses in runtime varargs.

Once tracks 1–2 are done and track 3 has relocated the type, `tt_metal` deletes its descriptor
concept and descriptor lives on as a ttnn-owned authoring/fusion IR.
