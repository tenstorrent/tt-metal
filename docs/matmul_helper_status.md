# Matmul Helper Library Status

**Branch**: wransom/llk3
**Last updated**: 2026-03-28

## Helpers Available

| Helper | File | Description |
|--------|------|-------------|
| `matmul_tile` | `matmul_tile_helpers.hpp` | Tile-at-a-time matmul wrapping `mm_init` + `matmul_tiles` LLK |
| `matmul_block` | `matmul_block_helpers.hpp` | Sub-blocked matmul with spill/reload wrapping `mm_block_init` + `matmul_block` LLK |
| `matmul_block_fused_bias` | `matmul_block_fused_bias_helpers.hpp` | Sub-blocked matmul + row-broadcast bias + optional SFPU activation |

All helpers live in `ttnn/cpp/ttnn/kernel_lib/` with `.hpp` + `.inl` pairs.

## Migration Status

### Migrated Call Sites

| File | Helper Used | Notes |
|------|-------------|-------|
| `matmul/device/kernels/compute/bmm.cpp` | `matmul_tile` | Core TTNN tile-matmul kernel |
| `matmul/device/kernels/compute/bmm_large_block_zm.cpp` | `matmul_block` | Core TTNN block-matmul kernel |
| `programming_examples/matmul/.../bmm_large_block_zm.cpp` | `matmul_block` | Programming example |
| `experimental/conv3d/device/kernels/compute.cpp` | `matmul_block` | Replaced local `matmul_blocks()` with library helper using `NoWaitNoPop` mode |

### Not Migratable

| File | Reason |
|------|--------|
| `bmm_large_block_zm_fused_bias_activation.cpp` | ~500 lines with many `#ifdef` variants (L1_ACC, DRAM_SHARDED, transpose, untilize, PACK_RELU). Core fused pattern matches `matmul_block_fused_bias` but the `#ifdef` paths prevent drop-in replacement. |
| `bmm_large_block_zm_fused_bias_activation_gathered.cpp` | Same as above + multi-device CCL |
| `conv2d/conv_bmm_tilize.cpp` | Fused tilize + matmul + bias with custom CB switching |
| `transformer/sdpa/compute_streaming.hpp` | Complex SDPA pipeline — matmul is one stage among many |
| `transformer/sdpa/compute_common.hpp` | Same as above |
| `experimental/minimal_matmul/compute.cpp` | Out-of-order packing (`pack_tile<true>`) + L1_ACC + ternary ops |
| `experimental/matmul/group_attn_matmul/...` | Uses `matmul_tiles` (not block), `pack_untilize_dest` |
| `experimental/deepseek/mla/matmul_wo/compute.cpp` | Ring-distributed matmul with custom tile indexing |
| `experimental/deepseek/moe/moe_gate_mm/compute.cpp` | Custom MoE dispatch with multiple matmul_block configs |
| `experimental/ccl/moe_compute/compute.cpp` | SILU activation + eltwise multiply interleaved |
| `experimental/topk_router_gpt/compute.cpp` | matmul + binary_dest_reuse + bias + topk + softmax pipeline |
| `experimental/ccl/llama_all_gather_matmul_async/...` | Fused all-gather + matmul + bias + CCL |

## Recent Changes

### WaitPopMode for matmul_block (2026-03-28)
Added `WaitPopMode` enum to `matmul_block` helper:
- `WaitAndPop` (default): helper manages `cb_wait_front`/`cb_pop_front` per K-block
- `NoWaitNoPop`: caller manages all CB synchronization externally

This enables migration of call sites where inputs are pre-waited or persistent (e.g., weight reuse across multiple matmul calls in conv3d).

### conv3d Migration (2026-03-28)
Replaced local `matmul_blocks()` function (55 lines) with library helper call using:
- `InitAndUninit`: full `mm_block_init` replaces `mm_block_init_short` + `reconfig_data_format`
- `NoReconfigure`: init handles all configuration
- `NoWaitNoPop`: weights persist across spatial patches, vol2col sync is external

**Test results**: 1536 conv3d sweep tests passed, 0 failed (Wormhole).

### matmul_block_fused_bias Helper (2026-03-27)
New helper for matmul+bias+activation pattern. Two-phase design:
1. Matmul phase: packs to interm_cb (same spill/reload as matmul_block)
2. Bias phase: reads from interm_cb, adds bias via `add_bcast_rows`, optional PostComputeFn, packs to out_cb

**Test results**: 4 GTest cases passed with PCC > 0.997 (Wormhole).

## Recommended Next Steps

1. **Extend matmul_block_fused_bias for production migration** — Add support for PACKER_L1_ACC, untilize_out, and in0_transpose to cover more of `bmm_fused_bias_activation.cpp`
2. **Add WaitPopMode to matmul_block_fused_bias** — for consistency with matmul_block
3. **Consider out-of-order packing mode** — would enable migration of `minimal_matmul` and similar kernels that pack tiles in non-sequential order
