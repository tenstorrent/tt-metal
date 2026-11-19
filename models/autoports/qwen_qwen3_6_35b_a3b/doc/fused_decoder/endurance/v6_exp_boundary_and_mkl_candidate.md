# Exp boundary diagnosis and MKL arithmetic candidate

This is CPU/source diagnosis and a prepared device component, not decoder acceptance or performance evidence. No hardware was used by the author. All acceptance gates remain unchanged. The helper uses general mathematical constants, not oracle state or reference outputs; canonical operands appear only in diagnostic setup and comparisons.

The installed Torch CPU exp implementation is MKL `vmsExp`, as selected by `AT_MKL_ENABLED()=1` and `ATen/cpu/vml.h`. Read-only runtime dispatch resolves `mkl_vml_kernel_sExp_EXHAynn` (CPU detect 0, dispatch index 6, default VML mode `0x1a02`). This is the authoritative local path; AVX512 CPU capability alone does not imply that Torch uses SLEEF exp. The exact library hash, addresses, constants, table codes, dispatch result, and disassembly hash are in [v6_mkl_exp_dispatch_cpu.json](v6_mkl_exp_dispatch_cpu.json) and [v6_mkl_exp_normal_loop.disasm.txt](v6_mkl_exp_normal_loop.disasm.txt). No dispatch override or binary patch was used.

The previous native accurate TT exp output is explained by the Juffa polynomial in `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`: the CPU true-FMA transcription reproduces the saved device decay mask with zero code differences on both chunks when evaluated on the same saved device cumulative sums. Native versus Torch normal-range differences are at most one ULP. Nonetheless those differences can change the final BF16 recurrent state. Results from [v6_exp_boundary_cpu_probe.json](v6_exp_boundary_cpu_probe.json), replacing only all four exp sites and keeping every other CPU operation canonical:

| Exp implementation | Final FP32 code differences | Final BF16 code differences |
|---|---:|---:|
| Torch MKL reference | 0 | 0 |
| Double exp rounded to FP32 | 88,211 | 16 |
| SLEEF8 u10 | 190,127 | 21 |
| Juffa true-FMA transcription | 156,312 | 23 |
| Juffa separate multiply/add transcription | 201,887 | 24 |
| Recovered MKL normal loop, rare negatives flushed | 0 | 0 |

The last row comes from [v6_mkl_exp_dispatch_cpu.json](v6_mkl_exp_dispatch_cpu.json). Flushing rare arguments also happens to preserve this particular full prefix state exactly; this is a measured fixture result, not a universal underflow-equivalence claim. More accurately rounded mathematical exp alone is therefore insufficient to reproduce this Torch state.

The normal-loop transcription uses a 128-entry high/low table for `2**(j/128)`, Cody–Waite style range reduction, a quadratic residual polynomial, and separately rounded FP32 multiplies/adds/subtracts. It does not require FMA. Decimal precision100 regeneration matches all 256 table values bit-for-bit, independently of the binary. The raw source constants and dispatch remain recorded in the JSON. The original CPU proof matches 268,483 actual normal arguments and 100,000 seeded random normal arguments exactly.

The separate [device helper](../../../tests/v6_mkl_exp_ttnn_probe.py) constructs the mathematical table during setup, uploads it once, and uses device-only arithmetic/bitcasts/bitwise operations/gather afterward. It replicates the table to two `[1,32,64,128]` FP32 tensors (2 MiB total) to avoid reshaping logical width1 inputs or changing layouts during the diagnostic runtime. This allocation is a diagnostic implementation choice, not a production memory policy or performance result. The finite input contract is `x<=0`: the normal path applies at `x>=-87.33654022216797`, smaller arguments explicitly flush to zero. Positive inputs, NaN/Inf, MKL rare handling and arbitrary positive-overflow behavior are unsupported. The actual chunk decay graph uses nonpositive arguments at all four exp sites.

[v6_mkl_exp_ttnn_cpu_frozen.json](v6_mkl_exp_ttnn_cpu_frozen.json) exercises the identical helper via a CPU adapter: 399,873 tested normal inputs have zero code differences, including actual canonical inputs, 131,072 random values, explicit boundaries, and logical width1. It reports 1,951 rare inputs separately. The unchanged double-single state graph is wrapped only at setup and `exp`; [v6_chunk64_mkl_exp_cpu_setup.json](v6_chunk64_mkl_exp_cpu_setup.json) passes CPU setup at lengths 128, 127, and 65. Existing measured source files are hashed and remain unchanged. Exact source snapshots accompany each result.

Frozen candidate hashes:

```text
8b262be19cf72ca69d553a65de656398a3efa8388f5e3ed45e9a1df5113787fa  tests/v6_mkl_exp_ttnn_probe.py
c4004cd0992298ba33c5ab21b9ecb8cb5e1d44dea6c790630643fe3a68ec27eb  tests/v6_chunk64_mkl_exp_state_probe.py
```

Parent-owned device commands, from repository root (omit `--oracle-only` only when device ownership is established):

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_mkl_exp_ttnn_probe --oracle-only --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_mkl_exp_ttnn_cpu_frozen.json
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_mkl_exp_state_probe --lengths 128 127 65 --oracle-only --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_mkl_exp_cpu_setup.json
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_mkl_exp_ttnn_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_mkl_exp_ttnn_device.json
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_mkl_exp_state_probe --lengths 128 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_mkl_exp_device.json
```

Scripts refuse to overwrite evidence; choose a new output path for reruns. The component captures intermediate values to identify whether bitcasts, integer scaling, table lookup, or FP32 arithmetic first differs on hardware. The full graph comparison is the next causal experiment; neither CPU exactness nor fewer prefix code differences proves the unchanged every-token decoder gate or the required traced speedup.

## Device component boundary and separate scaled candidate

The parent subsequently completed [v6_mkl_exp_ttnn_device.json](v6_mkl_exp_ttnn_device.json). Range reduction, indices, lookup values, polynomial, reduced result and integer exponent codes are exact on every measured case. The integer-to-FP32 exponent bitcast is the first differing intermediate. This is **not adequately explained as subnormal flushing**: the intended powers of two are normal, and the observed outputs have extra mantissa bits rather than becoming zero. The physical cause remains unidentified.

[v6_mkl_exp_bitcast_boundary.json](v6_mkl_exp_bitcast_boundary.json) preserves the saved-tensor SHA and all eleven observed mappings. Examples are `0x00800000 -> 0x00c00000`, `0x01000000 -> 0x01200000`, and `0x05800000 -> 0x05802000`. There are 10,925 affected random inputs, ranging from -86.99935913085938 to -79.71516418457031. A one-exponent adjustment would leave most of that observed range exposed.

The new [scaled helper](../../../tests/v6_mkl_exp_scaled_ttnn_probe.py) adds 16 to the scale exponent before bitcast and multiplies the order-one reduced polynomial by `2**-16` before the final product. Both transformations are exact power-of-two scalings on the stated finite nonpositive input domain: the scale moves from exponent range -126..0 to -110..16. The rare-negative flush policy is unchanged. Only the `exponent_bits` and `value` assignments of the original helper AST are changed, with full restoration proof. The original measured helper and graph wrapper remain byte-for-byte unchanged.

[v6_mkl_exp_scaled_cpu_setup.json](v6_mkl_exp_scaled_cpu_setup.json) proves identical CPU results to the original helper on all previous cases plus 48,003 inputs covering the scale/table lattice and small offsets. [v6_chunk64_mkl_exp_scaled_cpu_setup.json](v6_chunk64_mkl_exp_scaled_cpu_setup.json) passes setup for lengths 128/127/65. This remains a candidate until parent-owned device testing confirms the rescaling avoids the measured boundary.

```text
c55f1249fae3671bdf98496d582346a947f5d27163ac2bc8a43b224346738748  tests/v6_mkl_exp_scaled_ttnn_probe.py
d5f32a7252784083440452a4cb124359f29d92c81d3539ab8ddd539d28190dec  tests/v6_chunk64_mkl_exp_scaled_state_probe.py
```

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_mkl_exp_scaled_ttnn_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_mkl_exp_scaled_ttnn_device.json
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_mkl_exp_scaled_state_probe --lengths 128 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_mkl_exp_scaled_device.json
```
