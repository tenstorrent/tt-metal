# AutoFix exact frontend follow-up

Source/CPU-only investigation. No device execution, TTNN import, or runtime/test edits. Evidence is the frozen `../endurance_gates_norm_l2_conv.py.snapshot`, `../endurance_exact_frontend_probe.json.pos*.pt`, and parent's projection microprobe JSONs. Device results below are read artifacts, not independent measurements.

## Finding and next discriminator

The first demonstrated remaining divergence is the packed linear projection on **bit-identical BF16 input norm**. Native source explicitly documents a **Wormhole hardware bug with HiFi4 plus FP32 destination accumulation (#38306)**. The current model selects exactly that combination. The smallest source-backed next experiment is the same-input projection microprobe with **HiFi3, retaining FP32 destination accumulation and every other setting**. Check operand roundtrip as the parent is already doing. If HiFi3 materially improves raw FP32 projection error and BF16 output codes, use a dedicated HiFi3 configuration for the packed decode projection and repeat the exact-frontend endurance stream. Do not reduce destination precision or loosen thresholds.

This is a supported candidate, not a proven hardware diagnosis. HiFi3 omits a fidelity phase, so it is not mathematically identical to ideal BF16 products; measure the resulting error. Source does not establish that the bug explains this exact capture.

Exact source evidence:

* `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:43–67`: `verify_numerical_configuration` identifies Wormhole, HiFi4, and FP32 accumulation, and recommends HiFi3. It warns rather than changing user configuration.
* Same file, lines 23–32: explicit user configuration is returned unchanged; lines 111–124: Metal 2.0 translation preserves fidelity and FP32 destination selection.
* `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:3415–3420`: default FP32-input matmul chooses HiFi3 on Wormhole specifically because of this bug; an explicit model HiFi4 configuration overrides the default.
* `tests/ttnn/unit_tests/gtests/test_matmul.cpp:258` and `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py:219–228`: current tests deliberately use HiFi3 with FP32 accumulation to avoid the documented bug.
* `tt_metal/hw/inc/api/compute/matmul.h:258–277`: matmul execution uses `llk_math_matmul<MATH_FIDELITY,...>`. This is not a hidden BF16-specific fidelity downgrade.

If HiFi3 does not help and operand roundtrip is exact, the next useful discriminator is a narrow projection reproduced with SFPU FP32 elementwise multiply and reduction, or an equivalent CPU/device per-product probe. More K-block or packer variants are not discriminating after the parent's existing results.

## CPU capture evidence

All 13 captured positions (128, 129, 256, 1024, 3032, 3979, 4904, 5011, 6595, 7551, 7668, 7703, 8157) have TT input norm bit-equal to HF. Consequently an upstream input norm discrepancy cannot explain the packed projection mismatch in this candidate.

CPU recomputation uses the checkpoint's actual BF16 `in_proj_qkv.weight`, the saved exact same BF16 input norm widened to FP32, and ordinary CPU FP32 linear:

| Position | TT raw FP32 QKV vs CPU FP32 relative L2 | Max absolute | TT rounded BF16 vs CPU BF16 differing codes / 8192 | CPU FP32 rounded vs CPU BF16 differing codes |
|---|---:|---:|---:|---:|
| 128 | .000292807 | .00228548 | 1243 | 74 |
| 129 | .000345493 | .00163376 | 1333 | 72 |
| 256 | .000352929 | .00293261 | 1305 | 73 |
| 1024 | .000350688 | .00208390 | 1294 | 66 |

Z raw projection has similar relative error (~.00032). At these positions A/B FP32 relative errors are ~.00027–.00045, producing 1–6 A and 4–8 B BF16 code differences among 32 elements. Therefore even explicit canonical gate rounding cannot remove errors already introduced by linear projection.

At positions 128/129/256, current Q relative errors are .004380/.002915/.002502, K .005453/.004848/.003408, V .002955/.003325/.002443. These contain the differing projected convolution history; the captures lack incoming convolution cache, so they cannot isolate a new conv implementation defect on identical operands.

Incoming recurrent state relative L2 is .010509 at128, .010566 at129, .011980 at256, .013212 at1024, .013374 at3979, .015760 at5011, .013895 at7703, .013688 at8157. Thus there is prefix error plus additional decode divergence, even when input norm is exact. The parent's HF-prefix-seeded stream removing the early failures but retaining late failures is consistent with both contributions.

Current gate outputs are often already extremely close: g relative error is below 3e-6 at every saved position except4904 (.0005203) and7551 (.00004236). Beta is bit-equal to HF at129,256,1024,3979,5011,6595,7551,7668,7703,8157. A single-step beta/softplus repair at late failing7703 therefore cannot explain the accumulated-state discrepancy. Historical projection differences remain plausible.

## Partial reload hypothesis is not supported

`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1303–1309` selects FP32 destination. Its one-M-tile, CK=1 compute path acquires destination before the entire K loop and packs after that loop (`kernels/compute/bmm_dram_sharded_multicore.cpp:195–240`). `in0_block_w=2` changes streaming blocks without reloading the accumulating FP32 destination in this path. Packer accumulation is selected through `DSMC_L1ACC` only when FP32 destination is disabled (factory1655–1656).

The legacy DRAM factory also explicitly protects FP32 partial reloads using an UnpackToDest alias (factory254–266 and1084–1094). Therefore simply enabling packer accumulation or increasing block width was not a source-backed correction for this configuration.

The parent's `projection_precision.json` then independently rules out pipeline/block/packer necessity: generic matmul and pipeline block1/2/4, packer false/true all have relative L2 ~.00033036 to CPU FP32. `projection_precision_formats.json` further shows the error persists with FP32 operand storage and split-K2/4/8. These results point upstream of final accumulation partition, toward common matrix arithmetic or operand contents.

Additional CPU falsifications on position128:

* TT FP32 outputs do not have uniformly zero low13 mantissa bits; fewer than .2% do. Final-output TF32 truncation does not explain the saved outputs.
* Simulated missing HiFi4 low×low term does not match the error: subtracting it increases mismatch from .00029279 to .00031215; error correlation with that term is only ~.092.
* Rounding or truncating each exact BF16 product to 9–14 mantissa bits also does not explain the error (error correlations remain small). Avoid claiming a specific product truncation mechanism without the fidelity/operand discriminator.

BF16 weights themselves are exact in the checkpoint. Native source and tests treat HiFi4 as full fidelity; no inspected code supports declaring a .00033 relative projection error an unavoidable BF16 accuracy floor.

## Addendum: Blackhole accumulation audit after fidelity/operand controls

The new evidence supersedes the HiFi3 recommendation above: `projection_precision_fidelity.json` reports LoFi .026608, HiFi2 .0042191, HiFi3 .000339859, HiFi4 .000330360 relative L2. HiFi3 does not improve this case. More importantly, the target is Blackhole; the quoted #38306 warning is expressly Wormhole-only and supplies no diagnosis here. Parent also verified exact operand roundtrip and exact single-product diagonal matmul. Together with split-K2/4/8 invariance, this localizes the remaining investigation to summing multiple products, not uploads or simple product rounding.

### What source proves, and what it does not

* Blackhole `_llk_math_hw_configure_<true>` writes both `ALU_ACC_CTRL_Fp32_enabled` and `ALU_ACC_CTRL_SFPU_Fp32_enabled` to true (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:31–43`). There is no BF16-destination fallback in that function.
* Blackhole matmul emits **MVMUL**, not individual SFPU FMAs. The standard full-tile replay has explicit MVMUL instructions at `tt_llk_blackhole/llk_lib/llk_math_matmul.h:389–417`; its replay iterates the requested fidelity phases at433–435. Its execute path selects the destination tile then runs this replay, lines792–830.
* `tech_reports/matrix_engine/matrix_engine.md:10–19` documents a single engine operation as an 8x16 by16x16 multiplication on both Wormhole and Blackhole. Thus the primitive groups **16 products per output**. A K256 split leaves every such 16-product group intact; it cannot distinguish loss inside that operation from mathematically exact products.
* These sources expose the FP32 **destination register** mode and MVMUL instruction schedule. They do not describe enough of the internal product-alignment/reduction circuitry to establish its intermediate bit width or rounding rule. Calling this “TF32 accumulation” or naming an exact number of mantissa bits would be unsupported. Single-product exactness does not prove a multi-product internal reduction is exact.
* No inspected public compute configuration selects an alternative high-accuracy internal MVMUL reduction. `packer_l1_acc`, full/half destination sync, block width, and external split K govern boundaries around this instruction. Changing those is not a source-backed repair for loss inside it.

### Minimal native discriminator

Use regular padded32x32 BF16 operands, FP32 output, HiFi4, and no new tiny-tile format or native kernel edits. Narrow N to32 or64 representative projection channels. Upload actual saved projection operands rather than a random distribution.

1. **Two-product pair in one 16-wide group versus two groups.** For the same two products, place live K entries at(0,1), then at(0,16), with every other K slot zero. Compare each output against the CPU FP32 sum and the sum of two independently returned single-product outputs. Repeat several pairs with different product magnitudes and opposite signs. Positions(0,1) feed one native 16-product reduction; (0,16) feed separate K faces/MVMUL contributions. This is much cheaper and more discriminating than another full projection run.
2. **K32 / K16 / K8 / K4 / K2 / K1 partial dots**, each zero-padded to32. Return the native FP32 partials and combine on CPU first; this removes the device combination path as a confound. K32 tests tile boundaries; K16 tests face boundaries; smaller live K directly reduces the number of nonzero products in a MVMUL. If accuracy sharply improves below16, it implicates the within-MVMUL sum. If K16 partials are already accurate but K32 is not, it implicates adding face results/fidelity phases to destination.
3. A useful full-projection confirmation is **16 residue masks**, with each input retaining only `k % 16 == r` for r=0..15. Each MVMUL sees at most one live K product while the matrix kernel still accumulates across tiles. Sum the 16 FP32 outputs on CPU, then with accurate TTNN FP32 sum/add. Improvement here isolates the 16-product reduction without requiring2048 separate matmul calls. This is a diagnostic; assess performance before considering it an implementation.

The pair test must retain the original products exactly when relocating them, including both the activation and corresponding weight rows. Ordinary K16 splitting without padding or a different tiny-tile shape would change native routing and weaken the conclusion.

### Existing native correctness control

A model-local native alternative requiring no C++ edits is FP32 broadcast multiply plus **accurate SFPU sum**: arrange selected weights as[N,K], widen the exact BF16 activation and weights toFP32, compute `weight * activation[None,:]`, then sum the K dimension, retaining FP32 until the HF-required BF16 boundary. Chunk N to bound temporary memory. It is a correctness control, not a promised performant replacement.

Source routing is explicit: FP32 binary multiply selects SFPU in `eltwise/binary_ng/device/binary_ng_device_operation.cpp:51–65`; `reduction/generic/device/reduce_op.cpp:198–217` selects SFPU sum for FP32 input with FP32 destination and `fast_and_approximate_mode=False` on Blackhole. This avoids the matrix engine's MVMUL reduction entirely. Likewise, a batched diagonal/single-product matmul followed by that FP32 SFPU reduction can preserve native product computation while moving only summation out of MVMUL, but costs substantially more storage/work.

No hardware result is claimed for these proposed discriminators or alternatives.
