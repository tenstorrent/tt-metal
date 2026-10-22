# Reconfiguring hardware for different DataFormats

Certain operations may require multiple input or output DataFormats. Since the Unpacker, Math, and Packer require the hardware to be configured depending on the DataFormat, the `reconfig_data_format` and `pack_reconfig_data_format` APIs provide the necessary calls for the programmer to reconfigure the DataFormats for the next operation.

## `reconfig_data_format`

This API reconfigures hardware associated with UNPACK (trisc0) and MATH (trisc1). It consists of the following 6 calls:
```
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_data_format(const uint32_t icb0_new_operand, const uint32_t icb1_new_operand)

template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_data_format(const uint32_t icb0_old_operand, const uint32_t icb0_new_operand, const uint32_t icb1_old_operand, const uint32_t icb1_new_operand)

ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand)

ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand)

ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand)

ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand)
```
There are 3 different functions (`reconfig_data_format`, `reconfig_data_format_srca`, `reconfig_data_format_srcb`), each overloaded to accept either only the new operand CB index or both the old and new operand CB index.
1. `reconfig_data_format`: reconfigures the DataFormats for both SrcA and SrcB registers. `src_order` selects how the two operands map onto SrcA/SrcB (`SrcOrder::Reverse` maps icb0 -> SrcB and icb1 -> SrcA, as matmul does).
2. `reconfig_data_format_srca`: reconfigures the DataFormats for only SrcA register
3. `reconfig_data_format_srcb`: reconfigures the DataFormats for only SrcB register

These calls change the data format and tile size only. When the tile/face geometry also changes, use `reconfig_full_operand*` instead; for a geometry-only change use `reconfig_tile_shape*` (both Wormhole/Blackhole only). See `tt_metal/hw/inc/api/compute/reconfig_data_format.h`.

The int8/unsigned state is always re-derived from the new format, so reconfiguring between FLOAT and INT8 DataFormats (ex. BFLOAT16 <-> UINT8) needs no extra flag. The `*_skip_int8` variants skip that step for callers that know no int8 boundary is crossed.

The following DataFormat reconfigurations are currently supported:
| Old DataFormat                            | New DataFormat                            | Requirements             |
|-------------------------------------------|-------------------------------------------|--------------------------|
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | None                     |
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | UINT8                                     | `DST_ACCUM_MODE==true`   |
| UINT8                                     | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | `DST_ACCUM_MODE==true`   |

## `pack_reconfig_data_format`

This API reconfigures hardware associated with PACK (trisc2), and has 2 calls:
```
ALWI void pack_reconfig_data_format(const uint32_t new_operand)

ALWI void pack_reconfig_data_format(const uint32_t old_operand, const uint32_t new_operand)
```
The function `pack_reconfig_data_format` is overloaded to accept either just the new, or both the old and new operand CB index.

The following DataFormat reconfigurations are currently supported:
| Old DataFormat                            | New DataFormat                            | Requirements                                 |
|-------------------------------------------|-------------------------------------------|----------------------------------------------|
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | None                                         |
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | UINT8                                     | None                                         |
| UINT8                                     | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | None                                         |

## Usage guidelines

- `reconfig_data_format` API should be used to reconfigure the hardware between calls to operations that use CBs of different DataFormats.
- `pack_reconfig_data_format` API is called independently of `reconfig_data_format`, when the output CB changes DataFormats
- Prefer the overloads providing both old and new operand CB indices when the current configuration is known. They compare operand descriptors, not the current hardware state: the old operand is a claim by the caller. A different buffer may serve as that reference only when its relevant descriptor fields are equivalent.
- A stale old-operand reference can make the comparison skip a required reconfiguration. For example, if SrcA is configured for FP32, passing two BF16 references does not switch it to BF16. The format guard compares both unpack source and destination formats. When operands and descriptors are compile-time known, the comparison can be constant-folded; this is not a hardware-state query.
- When the old configuration is unknown, use the new-only overload. This removes the need to identify the old operand, **not** the need to choose the correct target when restoring after a temporary operation.

## Examples:
TO DO
