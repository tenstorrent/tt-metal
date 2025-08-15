# Reconfiguring hardware for different DataFormats

Certain operations may require multiple input or output DataFormats. Since the Unpacker, Math, and Packer require the hardware to be configured depending on the DataFormat, the `reconfig_data_format` and `pack_reconfig_data_format` APIs provide the necessary calls for the programmer to reconfigure the DataFormats for the next operation.

## `reconfig_data_format`

This API reconfigures hardware associated with UNPACK (trisc0) and MATH (trisc1). It consists of the following 6 calls:
```
ALWI void reconfig_data_format(const uint32_t srca_new_operand, const uint32_t srcb_new_operand)

ALWI void reconfig_data_format(const uint32_t srca_old_operand, const uint32_t srca_new_operand, const uint32_t srcb_old_operand, const uint32_t srcb_new_operand)

ALWI void reconfig_data_format_srca(const uint32_t srca_new_operand)

ALWI void reconfig_data_format_srca(const uint32_t srca_old_operand, const uint32_t srca_new_operand)

ALWI void reconfig_data_format_srcb(const uint32_t srcb_new_operand)

ALWI void reconfig_data_format_srcb(const uint32_t srcb_old_operand, const uint32_t srcb_new_operand)
```
There are 3 different functions (`reconfig_data_format`, `reconfig_data_format_srca`, `reconfig_data_format_srcb`), each overloaded to accept either only the new operand CB index or both the old and new operand CB index.
1. `reconfig_data_format`: reconfigures the DataFormats for both SrcA and SrcB registers
2. `reconfig_data_format_srca`: reconfigures the DataFormats for only SrcA register
3. `reconfig_data_format_srcb`: reconfigures the DataFormats for only SrcB register

The following DataFormat reconfigurations are currently supported:
| Old DataFormat                            | New DataFormat                            | Special Requirements                                   |
|-------------------------------------------|-------------------------------------------|--------------------------------------------------------|
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | None                                                   |
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | {UINT8, INT8}                             | Int8 FPU Math won't work unless `DST_ACCUM_MODE==true` |
| {UINT8, INT8}                             | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | None                                                   |

## `pack_reconfig_data_format`

This API reconfigures hardware associated with PACK (trisc2), and has 2 calls:
```
ALWI void pack_reconfig_data_format(const uint32_t new_operand)

ALWI void pack_reconfig_data_format(const uint32_t old_operand, const uint32_t new_operand)
```
The function `pack_reconfig_data_format` is overloaded to accept either just the new, or both the old and new operand CB index.

The following DataFormat reconfigurations are currently supported:
| Old DataFormat                            | New DataFormat                            | Requirements |
|-------------------------------------------|-------------------------------------------|--------------|
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | None         |
| {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | UINT8                                     | None         |
| UINT8                                     | {FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B} | None         |

## Usage guidelines

- `reconfig_data_format` API should be used to reconfigure the hardware between calls to operations that use CBs of different DataFormats.
- `pack_reconfig_data_format` API is called independently of `reconfig_data_format`, when the output CB changes DataFormats
- Programmers should always use the API calls providing both the old and the new operand CB index, as this enables faster reconfiguration and dynamic checks for eligible conversions.

## Examples:
TO DO
