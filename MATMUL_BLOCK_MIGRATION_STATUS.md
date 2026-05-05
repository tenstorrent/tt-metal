# `matmul_block` migration status

Inventory of every `matmul_block` invocation in TTNN compute kernels and
their migration status against `compute_kernel_lib::matmul_block` (and
sister helpers in `ttnn/cpp/ttnn/kernel_lib/`). Updated each iteration of
the wransom/mm_help branch.

## Migrated

| Kernel | Helpers used |
|---|---|
| `matmul/device/kernels/compute/bmm_large_block_zm.cpp` | `matmul_block` (batch loop in helper) |
| `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | `matmul_block` + `bias_add` + `reblock_untilize` + `transpose_block` (as PreKBlockFn) |
| `conv/conv2d/device/kernels/conv_bmm_tilize.cpp` | `matmul_block` (`InitMode::None`, `pin_interm_to_captured_base`) + `bias_add` + `reblock_untilize` + `tilize` |
| `experimental/conv3d/device/kernels/compute.cpp` | `matmul_block` (`retain_in1=true`) |
| `transformer/sdpa/device/kernels/compute/compute_common.hpp` | `matmul_block` (`InitMode::None`, `retain_in0=true`) — wrapper used by every SDPA prefill / decode / ring-joint variant |
| `transformer/sdpa/device/kernels/compute/compute_streaming.hpp` | `matmul_block` (`InitMode::None`, PostFn=RecipPost) for the per-row 1×1 sum × col_identity → recip → 1/sum fusion |

## Unmigrated — production tree

### `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp:358`

All-gather-fused matmul (ring-aware, fabric-sync). Treats raw `matmul_block`
as a per-inner-dim accumulation primitive inside an explicit
`for inner_dim_idx` loop, while `compute_kernel_lib::matmul_block` is a
complete K-loop abstraction. Blockers:

- **Variable inner-dim per K-block.** `unpadded_in0_block_w` is a runtime
  value indexed by `(ring_idx + block) % ring_size` — changes every K-block.
  Helper's `MatmulBlockShape::of(...)` takes a fixed `in0_block_w`. Load-bearing.
- **Custom spill/reload reconfig.** `reload_from_cb_to_dst` does
  `copy_tile_to_dst_init_short_with_dt` + `copy_block_matmul_partials` +
  `mm_block_init_short_with_dt`. Helper's spill/reload assumes a uniform
  format and doesn't expose this hook.
- **`ENABLE_GLOBAL_CB` rd_ptr arithmetic.** Per-K-block
  `calculate_next_block_index_and_update_rd_ptr` walks the ring CB. Helper's
  `PreKBlockFn` doesn't expose per-K-block rd_ptr capture / restore.
- **Fabric sync interleaved with the matmul.** `sync2_buf` wait/pop +
  `sync_buf` reserve/push flank the matmul loops. Op-specific.
- **`mm_block_init_short_with_dt` variant.** Helper currently models
  Full / Short / None — this `_with_dt` variant isn't represented.

Code-org note: file path is under `matmul/` but content is CCL — a sibling
copy lives at
`experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`.
Whether the matmul-tree path is canonical is an open question.

## Unmigrated — experimental tree

All share the same root blocker as the gathered variant: each kernel uses
`matmul_block` as a fine-grained accumulation primitive inside a custom
loop, not as a complete K-loop. The helper currently abstracts the whole
K-loop, which doesn't match these call patterns.

| Kernel | Calls | Notes |
|---|---:|---|
| `experimental/ccl/llama_all_gather_matmul_async/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | 1 | Sibling copy of the production-tree gathered variant; same blockers. |
| `experimental/ccl/all_gather_minimal_matmul_async/device/kernels/compute.cpp` | 1 | All-gather-fused matmul, ring-aware. |
| `experimental/ccl/moe_compute/device/kernels/compute.cpp` | 4 | MoE expert routing; CCL stream-to-compute CBs. |
| `experimental/ccl/moe_gpt/device/kernels/compute.cpp` | 8 | MoE matmul with bias-as-matmul trick (`matmul(ones_tile, bias_row)`); ring-aware partials. |
| `experimental/deepseek/mla/matmul_wo/device/kernels/compute.cpp` | 1 | Custom stride-7 K-loop (`k += 7`). |
| `experimental/deepseek/moe/moe_gate_mm/device/kernels/compute.cpp` | 4 | Custom stride-2 K-loop with separate "last block" handling. |
| `experimental/matmul/group_attn_matmul/device/kernels/compute/transformer_group_attn_matmul.cpp` | 0 | Comment-only reference; no live `matmul_block` call. |

## Helper API extensions that would unlock the unmigrated work

In approximate payoff order — (1) is the most load-bearing and unlocks the
production-tree gathered variant plus most experimental call sites:

1. **Variable per-K-block inner-dim.** Either a `KBlockShapeFn` callback
   yielding the per-K-block `in0_block_w`, or a span on `MatmulBlockShape`.
   Unblocks ring-aware matmuls and the experimental custom-stride loops.
2. **Reload reconfig hook.** Template type customizing the spill/reload
   reconfig (currently a fixed format); lets callers substitute the
   `_with_dt` reload dance.
3. **Per-K-block rd_ptr hooks.** Pre/post UNPACK callbacks for ring CB
   rd_ptr capture + advance. Unblocks `ENABLE_GLOBAL_CB` callers.
4. **`InitMode::ShortWithDt` / `ShortWithBothDt`.** Formalizes the
   `_with_dt` variants the conv2d kernel (already paired via `InitMode::None`)
   and the gathered variant both use.
5. **A fine-grained primitive helper.** A `matmul_block_step` that wraps a
   single LLK `ckernel::matmul_block` call (no K-loop, no spill/reload, no
   pack) so kernels with custom outer structure can adopt the helper for
   per-tile init/reconfig hygiene without giving up their loop control.

## How this file is maintained

This file moves alongside the four-commit stack on `wransom/mm_help`:
`helper implementation` → `tests addition` → `migration / kernel
implementation` → **`status`**. Each pass on the project updates these four
commits in place rather than appending new ones.
