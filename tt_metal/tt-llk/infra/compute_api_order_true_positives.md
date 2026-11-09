# Compute-kernel API ordering: reviewed corpus findings

Corpus: **563 kernel files**, 3,378 units (2 architectures × 3 threads), **0 coverage
holes**. The full report is `compute_api_order_corpus.md`; the exact verdict map is
`compute_api_order_corpus_triage.yaml`.

## True positives

### Missing required uninit — 56 findings

The curated uninit contract (tilize, untilize, broadcast, fast tilize) is not met
before the kernel ends or switches operation groups. Representative files:

- `models/demos/blackhole/qwen38_flash_next/.../compute_qkv.cpp`,
  `gdn_step/.../compute.cpp`, `gr_write/.../compute.cpp`,
  `ple/.../gate_compute.cpp`, `shared_expert/.../compute_shared_eltwise.cpp`
  (`unary_bcast_init` without `unary_bcast_uninit`);
- `models/demos/deepseek_v3_b1/micro_ops/tilize_8x32/.../tilize_8x32_kernel.cpp`
  (`tilize_init` without `tilize_uninit`);
- `tt-train/.../cross_entropy_bw`, `cross_entropy_fw`, `layernorm_bw`,
  `rmsnorm_bw`, `softmax` kernels (broadcast uninit);
- `ttnn/.../eltwise_binary_*_row_col_bcast.cpp`,
  `eltwise_where_sfpu_row_col_bcast.cpp` (broadcast uninit).

Fix by calling the matching uninit (`unary_bcast_uninit`, `tilize_uninit`,
`untilize_uninit`, `fast_tilize_uninit`) at the end of the operation group.

### Missing init — 43 findings

The checker found operations with no matching init on the path. Representative
files:

- `models/demos/gpt_oss/tt/fused_decode/kernels/lm_rescore_compute.cpp`,
  `stream_*_compute.cpp`: pack-only kernels call `pack_tile` without
  `compute_kernel_hw_startup` / `pack_init`;
- `tt-train/.../sdpa_*_kernel.cpp`: `unary_bcast` without `unary_bcast_init`;
- `ttnn/.../normalization/groupnorm` and `softmax` kernels: broadcast/reduce
  helpers that reach the operation without the corresponding init on this path;
- `ttnn/.../topk.cpp`, `gated_delta_attn.cpp`,
  `eltwise_binary_row_bcast.cpp`: same pattern.

Each site should either call `compute_kernel_hw_startup` and the operation's init,
or document why the path is unreachable.

### Init / operation argument mismatch — 12 findings

`tt-train/.../sdpa_bw_kv_compute_kernel.cpp`,
`sdpa_bw_q_compute_kernel.cpp`, and `sdpa_fw_compute_kernel.cpp`.

The unpacker or datacopy init and its execute use different compile-time
arguments (e.g. `acc_to_dest` / datacopy configuration). Re-initialize the engine
for the execute's configuration or make the arguments agree.

## Triaged false positives

### `wrong-init` — 8 findings, `groupnorm_sharded_v2.cpp`

`compute_kernel_lib::reduce` computes a helper-local `constexpr bool is_sfpu`
through `is_sfpu_reduce_path<...>()`. The checker cannot evaluate that helper
constexpr, so it follows the SFPU branch (`copy_init`) and later sees the FPU
`reduce_tile`. This is a checker refinement (evaluate helper constexprs / treat
unknown `if constexpr` as a must-analysis), not a kernel defect.

### `loop-invariant-init` — 1,581 warnings, all advisory

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
