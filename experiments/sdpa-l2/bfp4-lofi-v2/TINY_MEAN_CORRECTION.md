# Tiny-row mean-Q correction feasibility

## Recommendation

Use a HiFi4/FP32 matmul with mean tiles `[1,32]` first, then batch eight
distinct Q-block means from the **same head** into `[8,32]` tiles. This is a
concrete supported software path to investigate, not a new low-level opcode.
The original BF16 K remains the right operand with standard32x32 tiles and
within-tile transpose. Do not replace it with quantized K.

Blackhole's matmul API derives `partial_face = in0_height < 16`
(`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h:48`).
The no-MOP replay-length formula returns4 MVMULs per fidelity phase for
1/2/4/8x32 mean tiles,8 for16x32, and16 for32x32
(`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h:170`).
The regular matmul has the same formula and partial-face sequence at
`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:340`.

Thus a single-row correction saves approximately4x FPU instructions, not32x.
At Q128/K512/D128, full repeated32-row HiFi4 correction takes4096 MVMULs;
tiny1/8-row correction takes1024. Ordinary LoFi QK+PV takes8192 total.
These are ideal instruction counts excluding initialization, stalls, and DM:

| Correction scheme | Relative to Q128 LoFi QK+PV for all covered queries |
|---|---:|
| One mean repeated32 rows | 50% |
| One mean in tiny1 row | 12.5% |
| Eight distinct means in tiny8 rows, serving eight Q128 blocks | 1.5625% |

The last number is not an end-to-end utilization prediction. It requires
grouping eight independent means, computing their corrections together, and
reusing the same original-K read across them. For one isolated Q block the
eight-row arithmetic floor remains wasted.

## Support and constraints

Current tests explicitly cover tile heights1/2/4/8/16/32 on the reuse path,
including transposed full-width right operands, but the documented empirical
qualification is **Wormhole**, not our Blackhole device
(`tests/ttnn/unit_tests/operations/matmul/test_matmul.py:44`). Blackhole LLK
source supports the geometry; new device qualification is required. The
host-only warning in `tt_metal/impl/data_format/tile.cpp:25` is inconsistent
with newer validation/tests and should not alone rule the experiment out.

Use `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0=True`,
full-width K tiles, BF16 inputs,
HiFi4, FP32 DST/output, and packer L1 accumulation disabled. Transposed
16-wide right tiles are rejected. BFP right inputs on multicast factories
and all DRAM-sharded multicast inputs have additional <16-row restrictions
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:118,1315`).

`MVMUL` has no programmable one-row NumRows parameter: the shared ISA model
derives8 rows normally or7 in a peculiar row-broadcast mode; broadcast only
writes alternate rows. This is not a cheaper clean one-row dot primitive.
`DOTPV` is merely MVMUL without broadcast. Sources:
`tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/MVMUL.md:88`,
`DOTPV.md:3`; the Blackhole MatrixUnit page explicitly points to this shared
documentation and warns of architecture differences. Prefer the current
Blackhole partial-face LLK over undocumented broadcast behavior.

## Original-K reuse and bounded integration

First adapt the producer's actual rounded BF16 mean to a tiny tensor. Device
row-major conversion plus `ttnn.tilize(..., tile=ttnn.Tile([1,32]))` is an
available baseline API (`tilize_nanobind.cpp:50`); account for its dispatches.
A custom reader can instead extract the first16 BF16 columns of face0 and
face1 from each existing mean tile: two32-byte reads yield one64-byte tiny
tile, four tiny tiles perD128 mean. No host-generated bias is needed in the
eventual integration.

Keep groups bounded to eight Q blocks. Store only their eight means and
`8*N` FP32 corrections (8 MiB at256K,2 MiB at64K), consume those blocks, and
reuse storage for the next group. Do not materialize all-block corrections
as an implicit N²/128 intermediate. The tiny output is compact; the current
correction reader expects32 repeated rows, so either an explicit device
broadcast expansion (costed correctness baseline) or a qualified tiny-row
unpack/broadcast-to-score helper is needed. Expanding all eight back to32
rows forfeits the compact-storage benefit.

Original K must still be read somewhere. Separate correction kernels read
BF16 K once per eight-block group, in addition to the already-quantized K
attention stream. A fused original-K quantize/correction producer can reuse
the incoming BF16 CB for its first group, but extra groups must reread K or
retain it in a real cache. Quantized-K CBs cannot reconstruct original K.
No DRAM read disappears simply because correction has only one logical row.

## SFPU alternative

A direct FP32 SFPU dot is feasible: load original K and the column-matched
mean, multiply/accumulate acrossD tiles in live registers, then reduce the
16-column face rows with the already-understood shuffle/add tree. It can
share an original-K preprocessor read and avoids the public FPU precision
floor, if that floor proves fundamental to its route. But64K products per
K512/D128 group need thousands of SFPU vector/load instructions plus row
reduction, versus1024 tiny HiFi4 MVMULs. It also competes with quantization/exp
for SFPU and DST. It is a precision-control candidate, not the leading speed
candidate, absent measured overlap.

## Isolated primitive probe

`tiny_mean_matmul_probe.py` intentionally uses host-generated BF16 means and
excludes producer, upload, retiling, and attention integration from timing.
It tests exactly the same eight distinct means and original BF16 K:

- `split1`: eight independent1-row calls.
- `batched8`: one8-row call.
- `padded32`: one32-row call, each of the eight means repeated four times.
- Optional `legacy32`: eight32-row calls, each holding one repeated mean.

Every physical output row is checked against FP64; the same eight unique
rows are compared across schemes. Tiny BF16 input and FP32-zero output
upload/readback must be exact before matmul. Trace timing and replay equality
are included. The0.1% diagnostic gate is explicit; it does not certify the
unresolved public HiFi4/FP32 correction floor as ideal.

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/tiny_mean_matmul_probe.py \
  --label tiny-mean-smoke-v1 --length 1024 --iters 0
python experiments/sdpa-l2/bfp4-lofi-v2/tiny_mean_matmul_probe.py \
  --label tiny-mean-32k-v1 --length 32768 --iters 5
python experiments/sdpa-l2/bfp4-lofi-v2/tiny_mean_matmul_probe.py \
  --label tiny-mean-256k-v1 --length 262144 --iters 5
```

## Measured Blackhole primitive qualification

The main experiment runner subsequently qualified all three layouts on the
allocated Blackhole device. The table is generated from
`tiny-mean-perf-32768-v1.json` and `tiny-mean-perf-262144-v1.json`:

| K length | Eight tiny1 calls (ms) | One tiny8 call (ms) | One padded32 call (ms) | Tiny8 speedup vs eight calls / padded32 | All layouts L2 (%) |
|---|---:|---:|---:|---:|---:|
| 32,768 | 0.234198 | 0.030894 | 0.046760 | 7.58× / 1.51× | 0.031 |
| 262,144 | 1.668767 | 0.230221 | 0.324443 | 7.25× / 1.41× | 0.031 |

The eight unique FP32 output rows are bit-identical across layouts, every
physical row passes the FP64 reference check, trace replays are identical,
and tiny BF16-input/FP32-output conversion checks are exact. The program
multicasts the means and partitions N, using 64 active workers at32K and103
at256K out of110 allowed; fixed16-tile N blocks bound circular-buffer sizes.
The simple reuse factory is unsuitable for this N partitioning.

These timings cover the matmul primitive only: no mean producer, upload,
retiling, expansion, or attention integration. Tiny8 is therefore a measured
promising correction primitive, not a measured end-to-end attention speedup.
The author performed static/CPU checks; the device measurements above were
run by the main experiment runner.
