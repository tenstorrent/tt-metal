# Compute-kernel API ordering: reviewed corpus findings

Corpus: **758 kernel files**, 4,548 units (2 architectures × 3 threads), **0 coverage
holes**, **100%** of the discovered kernels analysed. The full report is
`compute_api_order_corpus.md`; the exact verdict map is
`compute_api_order_corpus_triage.yaml`.

Error findings: **237 reviewed, 231 true positives, 6 false positives (2.5%)**.
The 1,734 `loop-invariant-init` warnings are advisory, are reported in a
separate table, and are excluded from the error total and false-positive rate.

## True positives

### Deprecated API — 110 findings across 19 source sites

`[[deprecated]]` is read from the called declaration; no issue-specific name list
is involved. The legacy `acquire_dst` / `release_dst` pair accounts for 108
findings (18 call sites × 3 threads × 2 architectures) in:

- `tests/tt_metal/tt_metal/test_kernels/compute/dfb_c2_pipeline_2_0.cpp`
- `dfb_eltwise_copy_2_0.cpp`, `dfb_eltwise_relu_2_0.cpp`
- `dfb_intra_and_consume_all.cpp`, `dfb_intra_and_consume_all_2_0.cpp`
- `dfb_t6_consumer_2_0.cpp`

Two more findings are the deprecated non-templated `copy_dest_values` overload in
`ttnn/.../experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp`.

Migrate to `tile_regs_acquire` / `tile_regs_release` and the
`copy_dest_values<DataFormat>` overload.

### Missing required uninit — 70 findings across 19 sites

The curated uninit contract (tilize, untilize, broadcast, fast tilize) is not met
before the kernel ends or switches operation groups. Representative files:

- `models/demos/blackhole/qwen38_flash_next/.../compute_qkv.cpp`,
  `gdn_step/.../compute.cpp`, `gr_write/.../compute.cpp`,
  `ple/.../gate_compute.cpp`, `shared_expert/.../compute_shared_eltwise.cpp`
  (`unary_bcast_init` without `unary_bcast_uninit`);
- `models/demos/deepseek_v3_b1/micro_ops/tilize_8x32/.../tilize_8x32_kernel.cpp`
  (`tilize_init` without `tilize_uninit`);
- `tests/.../compute_kernel_sentinel.cpp`, `max_pool.cpp`,
  `max_pool_multi_core.cpp` (tilize/pack-untilize without their uninit);
- `tt-train/.../cross_entropy_bw`, `cross_entropy_fw`, `layernorm_bw`,
  `rmsnorm_bw`, `softmax` kernels (broadcast uninit);
- `ttnn/.../eltwise_binary_*_row_col_bcast.cpp`,
  `eltwise_where_sfpu_row_col_bcast.cpp` (broadcast uninit).

Fix by calling the matching uninit (`unary_bcast_uninit`, `tilize_uninit`,
`untilize_uninit`, `fast_tilize_uninit`) at the end of the operation group.

### Missing init — 37 true positives across 16 sites

Operations with no matching init on the path. Representative files:

- `models/demos/gpt_oss/tt/fused_decode/kernels/lm_rescore_compute.cpp`,
  `stream_*_compute.cpp`: pack-only kernels call `pack_tile` without
  `compute_kernel_hw_startup` / `pack_init`;
- `tt-train/.../sdpa_*_kernel.cpp`: `unary_bcast` without `unary_bcast_init`;
- `ttnn/.../normalization/softmax` and `topk`, `gated_delta_attn`,
  `eltwise_binary_row_bcast.cpp`, `sdpa_flash_decode.cpp`: same pack/init
  pattern on a path that skips the startup;
- `tests/.../eltwise_copy_block.cpp`: no `compute_kernel_hw_startup` or
  `copy_init` / `pack_init` before the copy and pack loop.

Each site should either call `compute_kernel_hw_startup` and the operation's init,
or document why the path is unreachable.

### Init / operation argument mismatch — 12 findings across 3 sites

`tt-train/.../sdpa_bw_kv_compute_kernel.cpp`,
`sdpa_bw_q_compute_kernel.cpp`, and `sdpa_fw_compute_kernel.cpp`.

The unpacker or datacopy init and its execute use different compile-time
arguments (e.g. `acc_to_dest` / datacopy configuration). Re-initialize the engine
for the execute's configuration or make the arguments agree. This is the static
form of the LLK_ASSERT unpack-configuration failures tracked by #39184 and
#39945.

### Stale pack init — 2 findings, 1 site

`tests/.../eltwise_copy_3m.cpp` calls `llk_pack_init` before
`llk_pack_hw_configure`. The LLK contract is that `configure_pack` owns the
format/strides and must run before the packer init; reorder the two calls.

## Triaged false positives

### `missing-init` checker limitations — 6 findings

- `tests/.../pack_block_2_0.cpp` and `reconfig_srca_2_0.cpp` (4 findings,
  Blackhole): the id-free 2.0 `experimental::copy_init` is outside the modelled
  API surface, so the legacy `copy_tile` init is not seen. LLK 2.0 execution
  APIs are explicitly out of scope.
- `tests/.../transpose_wh_dest.cpp` (2 findings): libclang emits the dependent
  `transpose_dest_init` call as an `UNEXPOSED_EXPR`, so the LLK init it performs
  is not followed; the kernel does call the init before `transpose_dest`.

### `loop-invariant-init` — 1,734 warnings, all advisory

These are warning-only. The init is intentionally re-run after a reconfiguration
or to restore per-iteration SFPU state; hoisting is not valid in the general case.
They are excluded from the error false-positive rate.

### Blackhole fast-tilize — the original 5 false positives

The five findings in `tilize_16x32_kernel.cpp` were caused by condition keys that
differed only by redundant parentheses: `full_dim == 1` and `block == 1` were
treated as unrelated paths even though both call sites pass the same compile-time
value. Canonicalisation now strips redundant grouping around primary expressions,
so the conditions correlate and the five findings are gone. The regression is
covered by `test_blackhole_fast_tilize_fallback_has_no_false_positives`.

### Groupnorm `wrong-init` — the 8 false positives are fixed

`compute_kernel_lib::reduce` derives a helper-local
`constexpr bool is_sfpu` from `is_sfpu_reduce_path<...>()`. The checker now folds
the constexpr function body (including the `unpack_src_format` array and the
nested `if constexpr` returns), selects the FPU `reduce_init` branch, and reports
no `wrong-init` for `groupnorm_sharded_v2.cpp`.
