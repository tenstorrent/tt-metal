# tilize precision matrix results

Last run: 2026-09-23, Wormhole B0 (n150), `test_tilize_numeric_formats.py::test_tilize_precision_matrix` (380 passed, 224 skipped over the whole file).

Shapes: 32x32, 32x64, 64x128, 128x512, 256x2048, 32x48 (W pad), 48x64 (H pad), 48x80 (both pad). Distributions: rand + randn (floats); full-bit-pattern randint (integers). math_fidelity is pinned HiFi4: `test_tilize_fidelity_is_noop` shows all four fidelities give identical bits.

Skipped: fp8_e4m3 pairs (constructible only on Blackhole) and the `rand` distribution for integer inputs (one distribution covers the bit patterns).

| in -> out | fp32_dest_acc_en requested | cases | worst PCC | worst median abs err | worst p99 abs err | worst rel. RMS |
|---|---|---|---|---|---|---|
| bf16 -> bf16 | True | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| bf16 -> bf16 | False | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| bf16 -> fp32 | True | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| bf16 -> fp32 | False | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| bf16 -> bfp8 | True | 16 | 0.999950 | 7.812e-03 | 2.344e-02 | 9.700e-03 |
| bf16 -> bfp8 | False | 16 | 0.999950 | 7.812e-03 | 2.344e-02 | 9.700e-03 |
| bf16 -> bfp4 | True | 16 | 0.981850 | 1.504e-01 | 4.761e-01 | 2.202e-01 |
| bf16 -> bfp4 | False | 16 | 0.981850 | 1.504e-01 | 4.761e-01 | 2.202e-01 |
| fp32 -> bf16 | True | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 2.427e-05 |
| fp32 -> bf16 | False | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 2.427e-05 |
| fp32 -> fp32 | True | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| fp32 -> fp32 | False | 16 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| fp32 -> bfp8 | True | 16 | 0.999945 | 6.836e-03 | 2.344e-02 | 8.829e-03 |
| fp32 -> bfp8 | False | 16 | 0.999945 | 6.836e-03 | 2.344e-02 | 8.829e-03 |
| fp32 -> bfp4 | True | 16 | 0.981949 | 1.553e-01 | 4.835e-01 | 2.220e-01 |
| fp32 -> bfp4 | False | 16 | 0.981949 | 1.553e-01 | 4.835e-01 | 2.220e-01 |
| u32 -> u32 | True | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| u32 -> u32 | False | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| u32 -> i32 | True | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| u32 -> i32 | False | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| i32 -> i32 | True | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| i32 -> i32 | False | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| i32 -> u32 | True | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| i32 -> u32 | False | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| u16 -> u16 | True | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| u16 -> u16 | False | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| u8 -> u8 | True | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| u8 -> u8 | False | 8 | 1.000000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Exact pairs (same dtype, bf16 -> fp32, int32 <-> uint32 bit_cast) are asserted bit-identical (`torch.equal`). Lossy pairs are held to the golden floors: fp32 -> bf16 0.999, -> bfp8 0.99, -> bfp4 0.98.
The op ignores a requested fp32_dest_acc_en=True for uint16, because fp32 DEST scrambles UInt16 pages on WH. It forces fp32_dest_acc_en on for 32-bit and uint8 pages.
bfp4 is truncated by the packer: PCC ~0.982 here, where the host's rounding gives ~0.993.
