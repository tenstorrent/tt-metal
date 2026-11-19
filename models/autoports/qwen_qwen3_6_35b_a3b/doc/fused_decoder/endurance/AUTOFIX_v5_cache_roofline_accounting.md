# Source-bound convolution-cache footprint accounting

Updated only [summarize_fused_roofline.py](../../../tests/summarize_fused_roofline.py), accounting version `qz_ordered_l1_compact_decay_source_bound_conv_cache_v6`. Script SHA256: `189667f47b7ac856a758661ef4eba796759b4e90dd7dc7b72158203a910419f4`.

The new `convolution_history_allocation(source)` parses the actual `new_recurrent_state` AST. It validates the known zero/upload helper, BF16 `[1,3,8192]` shape and DRAM placement, then derives standard TILE padding or RM payload. Unknown helper wiring, shape, dtype or layout fails closed. Source hash and allocation line accompany the result. The implementation ledger replaces the TILE-specific key with `convolution_history_allocation_bytes`, and records layout, dtype, logical shape and padded shape. This is payload accounting, excluding allocator/bank overhead and conversion traffic.

CPU function checks passed without importing TTNN or invoking the script's report-writing `main`:

| Parsed source | SHA256 | Layout | Padded shape | Payload bytes | Useful bytes/token |
| --- | --- | --- | --- | ---: | ---: |
| Current production and immutable b163 baseline | `b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c` | TILE | `[1,32,8192]` | 524288 | 130536064 |
| [v5 cache writers candidate](v5_cache_writers.py.snapshot) | `9d33bc14b5f349a9dabfbd4e6ce6344c31047fbcfcc2716f35de0133e2d8b7f7` | ROW_MAJOR | `[1,3,8192]` | 49152 | 130536064 |

The complete useful weight/cache dictionaries are equal across both sources. The logical convolution-cache read/write bound stays `2*3*8192*2 = 98304` bytes/token. Neither the 464 KiB smaller allocation nor removed layout-conversion traffic is added to the useful-byte numerator. The execution note also distinguishes a direct recurrent-cache writer's subsequent DRAM state read from the former L1 producer/copy path without inventing or summing traffic totals.

Five negative CPU fixtures reject an unknown layout, changed shape, FP32 dtype, changed helper memory placement, and a helper that ignores its shape argument. All six existing `roofline.json` files were hashed before and after validation and remain byte-identical. Production decoder and hardware were untouched.

`python_env/bin/pre-commit run --files models/autoports/qwen_qwen3_6_35b_a3b/tests/summarize_fused_roofline.py` passed all applicable hooks, including Black23.10.1, isort5.13.2, autoflake and repository validators. No final-profile or optimization-stop claim follows from this accounting update.
