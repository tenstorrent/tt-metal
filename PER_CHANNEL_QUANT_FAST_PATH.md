# Per-Channel Quantization Fast Path — Design Document

## Problem
ttnn.quantize / ttnn.dequantize fall to a multi-pass composite for per-channel
calls even though the QUANT/DEQUANT LLK can stream a per-element fp32 scale via
operand B. The zero-point is the only scalar input.

## Solution
Route per-channel (Tensor) scale + scalar (int32) zero-point through a single
fused binary_ng pass instead of the 5-op composite fallback.

## Measured Speedup (Blackhole P150, 4096x4096)
| Operation | Axis | Composite | Fused | Speedup |
|-----------|------|-----------|-------|---------|
| dequant   | COL  | 1.00x     | 2.33x | 2.3x    |
| dequant   | ROW  | 1.00x     | 5.13x | 5.1x    |
| quant     | COL  | 1.00x     | 2.31x | 2.3x    |
| requant   | any  | 1.00x     | 3.17x | 3.2x    |

## Numerical Accuracy
| Precision | PCC vs Composite | PCC vs Torch Golden |
|-----------|------------------|---------------------|
| fp32      | 1.0000000        | 0.9999999           |
| bf16      | 0.9999986        | 0.9999972           |

## Architecture Coverage
- Blackhole: fully covered (LLK supports broadcast operand B)
- Wormhole: fully covered (same LLK interface)
- No LLK changes required

## Implementation
- dequantize: binary_ng(input, scale, DEQUANT, ..., ZERO_POINT=-zp)
- quantize:  binary_ng(input, scale, QUANT,  ..., ZERO_POINT=+zp)
- requantize: binary_ng(input, scale_recip, REQUANT, ..., ZERO_POINT=zp)
