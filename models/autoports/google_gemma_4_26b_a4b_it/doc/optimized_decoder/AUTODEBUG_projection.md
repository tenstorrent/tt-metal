# AutoDebug: decode QKV projection precision and matmul replacements

Source inspection on 2026-09-26. No hardware commands or production implementation
edits were performed by this investigator. The permitted intervention is a local
projection wrapper in `tt/optimized_decoder.py`, with tests and stage documents;
the frozen functional and fused implementations remain comparison references.

## Conclusion

The expensive QKV operation is compensating for a real operand-precision
boundary, rather than merely asking for a larger accumulator. A single ordinary
FP32-input matmul cannot reproduce the current FP32 SFPU dot products by setting
HiFi4 and FP32 destination accumulation: Blackhole matmul operands still pass
through TF32 Src registers. The BF16 checkpoint weight is already exactly
representable there; the normalized FP32 activation is the operand to preserve.

Two or three BF16 components of that activation, each multiplied by the original
BF16 weight with explicit HiFi4 and FP32 accumulation/output, are concrete
Python-only candidates. This is an approximation with a different reduction
order, not an assertion of bitwise equivalence. The first integrated two-term
experiment has one remaining failure at position 4149, so the next experiment
must distinguish activation decomposition error from GEMM accumulation/config
error and from downstream rank sensitivity.

**Update from the device owner's first probe:** three BF16 components reconstruct
the captured FP32 activation exactly, but QKV max error remains 0.03005147 while
broadcast error is 0.000009525. Therefore activation representation is no longer
a sufficient explanation of the remaining QKV discrepancy. Internal K-block
changes from 1 through 22 do not change the numerical results. See the final
section for the next controlled comparison; neither two nor three terms is yet
an accepted replacement.

## Direct observations and their limits

- `tt/fused_decoder.py:20-46` broadcasts the FP32 activation across FP32
  transposed weight rows, multiplies with SFPU, reduces over K, and concatenates
  output groups. Its default group size is 16384 (`:56`), larger than either
  actual projection width; the usual decode therefore materializes a full
  projection-sized FP32 product. `qkvl1` changes result placement, not product
  arithmetic (`:83-89`).
- Weight loading explicitly concatenates Q/K/V at TP1 and stores BF16 tiled DRAM
  weights: `models/demos/gemma4/tt/attention/weights.py:105-117,177-184`, selected
  by `tt/functional_decoder.py:59-78`. QKV does not contain a bias. Full attention
  ties K and V (`weights.py:78-81`), and `TiedQKV` retains only Q/K weight columns
  before duplicating the projected K tail (`tt/fused_decoder.py:711-729`).
- From `tests/config.json`, decode K=2816, Kt=88. Sliding has 16 Q heads and 8 KV
  heads of width 256: N=8192, Nt=256. Full has 16 Q heads and 2 KV heads of width
  512: stored QKV width 10240, but the existing tied projection computes N=9216,
  Nt=288, then duplicates the last 1024 columns. The broadcast intermediate is
  therefore 88 MiB or 99 MiB respectively. These are tensor byte counts, not
  measured bandwidth or latency claims.
- Existing `doc/functional_decoder/AUTOFIX_decode.md:320-350` records same-input
  QKV max error 0.04061413 for ordinary matmul versus 0.0000319481 for SFPU at
  position 4110, and recovery of the accepted route set. This is historical
  runtime evidence, not a measurement performed in this inspection.
- The current `compensated_sliding4096.json` records prefill PCC
  0.9985529033274025 and one failed decode check: position 4149, PCC
  0.9932774617563148. Its JSON does not record a same-input QKV comparison, so it
  does not yet identify why two components fail.

Do not conflate this with unavoidable expert-matmul error. The functional
diagnostics showed that substituting the HF routes repaired final PCC whereas
raising expert fidelity did not (`AUTOFIX_decode.md:29-40`). Nor does the
reappearance of position 4149 prove a RoPE regression: an earlier FP32-table
policy also failed there, but the current replacement retains BF16 table values.
A similar failure location can arise from different small perturbations near
the same route boundary.

## Lowered precision contract

`tt_metal/jit_build/genfiles.cpp:822-828` chooses TF32 unpack destinations for
Float32/B-family data with FP32 accumulation. `jit_build/data_format.cpp:151-159`
maps Float32 source storage to that destination. Blackhole documents the
distinction explicitly in
`tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:283-327`: SrcA/SrcB
can hold TF32, while direct DEST can hold FP32. Matmul's operands traverse
SrcA/SrcB (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h`).
More fidelity phases cannot reconstruct mantissa bits lost before multiplication.

Conversely, equal-FP32 subtraction/addition/multiplication select SFPU in
`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp:49-63`.
Use `x - typecast(high, float32)` for the residual and FP32+FP32 for combining
products. Mixed-dtype subtraction is not an adequate proof of this contract.

There is a second, distinct accumulation boundary. Current ordinary 1D multicast
matmul factories explicitly mark FP32 intermediate partials as direct-DEST reloads
(`matmul_multicore_reuse_mcast_1d_program_factory.cpp:763-801`, with corresponding
blocks near 1778, 2610, 3829 and 4791). The 2D factory does the same near 945 and
2446. Therefore it is incorrect to presume these current explicit 1D/2D paths
still narrow every partial sum to TF32. Inspect actual selected factory and
program configuration before assigning a residual error to that historical
problem.

The separate DRAM-sharded factory constructs `ComputeConfigDescriptor` without
that unpack vector (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:523-529`)
while selecting the shared blocked compute kernel. This is a reason to test its
K-block precision separately if that path is later considered; it is not an
explanation for the existing interleaved-weight two-term candidate. Start with
the ordinary 1D path rather than invoking `DramShardedLinear` merely because
`QKVLinear` inherits its callable interface.

Always pass an explicit compute configuration. For BF16 operands, supplying a
program configuration changes the inferred fidelity to LoFi unless overridden
(`matmul_device_operation.cpp:2804-2815`). Output dtype Float32 alone does not
establish HiFi4 or FP32 destination accumulation.

## Concrete candidates

1. **Two BF16 components.** Set `h = BF16(x)`,
   `r = x - FP32(h)`, `l = BF16(r)`. Compute FP32-output `h @ W + l @ W` with
   explicit HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`, and
   `packer_l1_acc=False`. Both operands of each GEMM are BF16 and exactly
   representable in TF32. Keep the full-kind tied-K tail duplication and decode
   L1 output contract. Preserve prefill's original callable.
2. **Third component.** Compute `r2 = r - FP32(l)`, `t = BF16(r2)`, and add the
   third FP32-output GEMM. This tests residual representation accuracy. For
   normal values without underflow, successive BF16 rounding errors decrease
   roughly geometrically; this does not eliminate FPU accumulation order
   differences. Record the actual reconstruction `FP32(h)+FP32(l)+FP32(t)`.
3. **Explicit 1D geometry.** Use multicast of input A, `per_core_M=1`,
   `fuse_batch=True`, no fused activation. Legal K-block divisors of 88 include
   1, 2, 4, 8, 11 and 22. An 8x8 grid gives `per_core_N=4` for sliding and 5 for
   tied full (the latter uses a partial final block); an available 8x9 grid gives
   full `per_core_N=4`. Start with output subblock 1x1, then consider 1x2/1x4
   only where divisibility permits. FP32 accumulation constrains the output
   subblock to four tiles. Never assume the device has a particular grid;
   validate before widening it.
4. **Pack components into M rows.** Concatenate h/l/(t) as two/three logical
   rows, do one GEMM, then accurate FP32 sum over M with `keepdim=True`. M=1,2,3
   all occupy one 32-row tile, so this may avoid repeated weight reads and GEMM
   launches. `concat.cpp:93-129` lowers unaligned height concatenation through
   untilize, row-major concat and tilize, so this is a measured tradeoff, not a
   free reshape. The probe supports this as a separate optional candidate.

Do not reduce W to BFP8/BFP4 in this initial projection experiment: decomposition
only repairs x, and weight quantization introduces an independent irreversible
error. Do not lower HiFi4, remove FP32 accumulation, or fuse a reduced-precision
output while adjudicating two-versus-three components. Once a candidate passes
the complete gate, those are independent optimizations with their own evidence.

An optional later alternative is a TF32 high part produced by device bitcast and
integer masking, with a residual matmul. Public `ttnn.bitcast` and integer
`bitwise_and` exist, but masking must be verified bit-for-bit, the actual unpack
rounding mode established, and scalar mask range checked. This is a secondary
hypothesis; BF16 splitting already has a simpler explicit operand contract.

## Discriminating experiment and delivered probe

`tests/probe_optimized_qkv.py` wraps the unchanged real-layer harness. It records
the normalized activation in a persistent device clone inside trace capture;
each replay rewrites that buffer. At the requested absolute HF position, after
the layer has completed and its output has been read, it runs independent QKV
comparisons outside the audited forward. It does not replace the .995 gate.

The report records:

- the current layer check, actual input/weight/component/product/output dtypes,
  tensor shapes and output memory;
- whether the uploaded weight values are exactly representable as BF16;
- CPU FP64 projection on the exact captured FP32 activation and uploaded BF16
  weights, compared against broadcast, plain FP32-input GEMM, two/three-term
  GEMMs, and requested explicit geometries;
- maximum and RMS error, relative L2, PCC, component reconstruction error,
  deterministic replay, and warmed host trace timing for each component;
- the original harness failure without suppressing it. Instrumented whole-layer
  timing is not a performance result.

Suggested initial command for the device owner (not run by this investigator):

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_qkv \
  --probe-position 4149 \
  --probe-output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149.json \
  --activation-output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149.pt \
  --qkv-blocks 1,4,8,11,22 --qkv-grids 8x8 \
  --layer 0 --length 4096 --steps 54 --real --decode \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149_harness.json
```

If three components improve reconstruction and QKV error but two-versus-three
does not restore the route, capture downstream Q/K/V, current BF16 cache writes,
attention residual, and the eighth/ninth router logit margin next. If changing
only K block changes QKV error significantly with identical reconstructed
operands, investigate accumulation/reload behavior and selected factory. If
broadcast itself is materially different from CPU on the same captured input,
do not use broadcast PCC alone as a precision oracle. Run 4110 as a controlled
historical contrast; repeat both layer kinds and the full acceptance matrix
after selecting a candidate. There must be no data-dependent host routing or
position-specific fallback in production.

## Workflow and verification

The repository-local AutoDebug skill was read and its required fresh CLI runner
was invoked from this stage directory. The nested CLI could not read source or
write `AUTODEBUG.md`: its sandbox launcher reported that neither system nor
bundled `bwrap` was available. Its zero process exit status did not indicate a
successful investigation. This report is the independent delegated source
inspection, not a claim that the failed CLI produced findings.

Only this report and the new probe were authored here. Python compilation and
Black formatting completed for the probe. No C++ source changed; no build was
required. Runtime results must be supplied by the serialized device owner.

## Follow-up: first same-input probe results

The main agent ran the delivered probe and supplied `qkv4149.json` and its log.
Reading those artifacts establishes the following independent of the original
high-level failure interpretation:

| Candidate | Input reconstruction max error | QKV max error vs CPU FP64 | Component trace host time |
| --- | ---: | ---: | ---: |
| FP32 broadcast | N/A | 0.000009525 | 2544 us |
| Plain FP32-input matmul | N/A | 0.04444308 | 125 us |
| Two BF16 components, default geometry | 0.000120163 | 0.03011156 | 268 us |
| Three BF16 components, default geometry | **0, exact** | 0.03005147 | 402 us |

All five explicit 1D K blocks (1,4,8,11,22) give identical numerical metrics to
their corresponding default two/three-term policy. Products and final outputs
are reported as FLOAT32; components and weights as BFLOAT16. The source weights
read back as exactly BF16-representable values. The log confirms Blackhole;
the Python object's legacy `WormholeComputeKernelConfig` class name is not
evidence that a Wormhole device was used. `compute_kernel_config.cpp:21-40`
returns the universal compute configuration, and the Wormhole-only HiFi4 warning
is guarded explicitly at `:45-61`.

These observations refute material decomposition error as the explanation for
the three-term result and weaken a per-internal-block partial reload hypothesis.
They do not yet distinguish final packing/narrowing, format-specific matmul
behavior, or accumulation precision inside the matrix engine. Three-term error
is approximately 3155 times broadcast's max error, well above an ordinary
last-bit reduction-order difference, so routing sensitivity alone is not a
complete account of the component discrepancy.

The probe now supports `--qkv-product-details --qkv-float32-operands`. Run with
`--qkv-blocks ''` to avoid repeating the disproven internal-block sweep. This
records each separate product against CPU multiplication of exactly those BF16
component values, compares against TF32-truncated CPU outputs, and reports the
fraction of output elements whose lowest 13 mantissa bits are zero. It also
upcasts the BF16 components to FP32 without changing their values, using either
the original BF16 weight or an exactly widened FP32 weight. Actual input,
weight and product dtypes and individual compute flags are recorded.

Interpretation of that next comparison:

- If the high product alone has the large error, the low correction has not
  caused it; inspect GEMM's mathematical/packing path on BF16 inputs.
- If product mantissas or CPU-TF32 comparisons expose final truncation, move
  the intervention to the output/packing boundary rather than adding terms.
- If merely widening storage of unchanged BF16 values changes accuracy, the
  selected data-format path is causal; retain only a validated format adaptation.
- If widening leaves the discrepancy unchanged, test external K partitioning
  (for example four 704-wide GEMMs) followed by FP32 SFPU summation. This resets
  matrix-engine accumulation between independent invocations; changing
  `in0_block_w` only preserves/reloads the same running FPU sum. Such a candidate
  needs its own same-input comparison and latency measurement.

The second probe is prepared but has not been run by this investigator. None of
these hypotheses authorizes relaxing the decoder PCC requirement or replacing
the accepted SFPU path without full regression evidence.

## Follow-up: individual products and external K reduction

The device owner's `qkv4149_products.json` eliminates two additional hypotheses:

- The first BF16-component product itself has max error 0.03003049 and relative
  L2 error 0.00036323 versus CPU multiplication of the same BF16 operands. The
  second and third products have much smaller absolute errors, commensurate
  with their magnitudes. Combining correction terms is not creating the
  material error.
- Upcasting the BF16 component values to FP32, with either BF16 or widened FP32
  weight storage, produces identical results. Thus the unchanged represented
  input values, not a BF16-versus-FP32 storage variant, determine this result.
- Only approximately 0.000122 of the first product's output elements have zero
  low 13 mantissa bits. It is not simply an output tensor rounded uniformly to
  TF32. Explicit config fields confirm HiFi4, FP32 destination accumulation,
  approximate math disabled, and packer L1 accumulation disabled.

The next test partitions the *logical* K dimension across independent GEMM
invocations, then adds their FP32 results with SFPU. This differs from the
already-tested `in0_block_w`: the shared blocked compute kernel reloads a
running partial into DEST (`bmm_large_block_zm_fused_bias_activation.cpp:91-127`)
and keeps accumulating in the matrix engine. A new GEMM starts a fresh sum.
External partitioning therefore tests whether error grows within a long FPU
reduction, even when internal partial reloads preserve their bits.

The new `--qkv-external-k` probe option supports all requested widths:

| K width per independent GEMM | Number of partitions | Tiles per partition |
| ---: | ---: | ---: |
| 32 | 88 | 1 |
| 128 | 22 | 4 |
| 256 | 11 | 8 |
| 704 | 4 | 22 |
| 1408 | 2 | 44 |

Weight slices are prepared on device before timing. Each captured invocation
decomposes the activation once, slices the corresponding component over K,
performs BF16-input/weight GEMMs with the same explicit HiFi4/FP32 configuration,
and sums FP32 partials. The report includes per-component complete projection
errors and the actual partial output dtype. This tests a legal stage-local
adaptation without changing a C++ factory or kernel.

If error decreases sharply with shorter external partitions, that supports an
accumulation-depth explanation and supplies an accuracy/latency tradeoff to
validate. If K=32 still has similar relative error, inspect the per-tile
multiply/accumulate behavior or fidelity phases instead; merely adding more
large independent GEMMs would not be a justified repair. The isolated QKV
comparison must still precede a full-model routing claim.

The new `--activation-input` mode reuses the exact saved FP32 activation and
loads only real Q/K/V checkpoint tensors, avoiding HF forward, decoder setup,
cache history and the layer replay. Earlier artifacts contain only activation;
future `--activation-output` artifacts also retain the exact BF16 projection
weight and tied-K metadata. This is explicitly a standalone component probe,
not a substitute for the real decoder gate.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_qkv \
  --activation-input models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149.pt \
  --layer 0 --qkv-blocks '' --qkv-terms 2,3 \
  --qkv-external-k 32,128,256,704,1408 --qkv-product-details \
  --probe-output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149_external.json
```

The extension passes Python compilation and Black. Its hardware execution is
left to the main agent; this investigator has performed no device commands.

## Follow-up: K=32 refutes long accumulation as the material cause

The main agent's `qkv4149_external.json` shows that the single-tile K=32 policy
still has three-term max error 0.03004599 and relative L2 error 0.00036330,
versus 0.03005147 and 0.00036327 for unpartitioned K. The change is around
5.5e-6, comparable to ordinary rounding of independent partial sums; it does
not repair the approximately 0.030 component discrepancy. K=128,256,704,1408
likewise retain it. External splitting is rejected as the accuracy repair.

The CPU oracle was re-audited against code rather than inferred from the HF
weight layout. `investigate()` reads `ttnn.to_torch(source.weight)` and the
exact captured FP32 activation, then multiplies those tensors in FP64. Its
full-attention tail duplication is applied after that multiplication. No
unpermuted HF weight participates in this comparison. The independent broadcast
path's 9.5e-6 agreement with that oracle further corroborates orientation and
values. The standalone reconstruction also reproduces those same component
metrics. Weight-layout mismatch is not supported by this evidence.

### Next hypothesis: precision inside the multiply/fidelity phases

The concrete source path programs four fidelity passes for HiFi4:
`tt_llk_blackhole/llk_lib/llk_defs.h:144` defines HiFi4 as 4, and
`llk_math_matmul.h:434-435` derives the replay-loop count from that value. The
matrix engine places the weight in SrcA and the activation in SrcB
(`llk_math_matmul.h:322-323`). The LLK golden model describes the asymmetric
mantissa phases in `tt-llk/tests/python_tests/helpers/golden_generators.py:982-1014`:
SrcA's first phase retains four explicit mantissa bits, SrcB's six; later
phases add the remaining cross products. Matmul's golden helper explicitly
swaps operands before applying those masks (`:1387-1397`).

This exposes a useful controlled intervention, but does **not** yet identify
a source defect or prove that actual hardware executes fewer phases. A cheap
CPU-only check using the saved BF16-rounded activation and cached real QKV
weights predicts that omission of only the low-SrcA times low-SrcB phase gives
max error 0.00462026 and RMS 0.000762286 in the high component. The observed
device high-component error is 0.03003049 and RMS 0.00285760. The simple
"fourth phase is absent" account is insufficient and should not be headlined.

The new probe option `--qkv-weight-mantissa 3,4` creates exact two-part BF16
weights during setup. It masks the FP32 view of the **device-read original
weight** to retain three/four explicit mantissa bits, computes the residual,
asserts that both parts are BF16-exact, and asserts their FP64 sum is exactly
the original weight. Each activation component is multiplied by both weight
components and the FP32 results are added with SFPU. This changes the multiply
operands' mantissa structure while preserving the represented weight.

The associated `--qkv-fidelity-controls HiFi2,HiFi3` adds ordinary original-weight
controls; the primary candidates retain explicit HiFi4. These tests distinguish
nominal fidelity settings from the numerical behavior they actually produce.
No lower-fidelity result is accepted merely because it happens to cancel a
route perturbation.

`--qkv-product-output` additionally saves the exact device-read original weight,
activation components, and individual product outputs. That permits CPU
emulation of candidate per-product rounding or missing cross products against
the actual signed error vector, rather than matching only a maximum error.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_qkv \
  --activation-input models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149.pt \
  --layer 0 --qkv-blocks '' --qkv-terms 2,3 \
  --qkv-weight-mantissa 3,4 --qkv-fidelity-controls HiFi2,HiFi3 \
  --qkv-product-details \
  --qkv-product-output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149_mantissa.pt \
  --probe-output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149_mantissa.json
```

This extension passes Python compilation and Black; the new hardware controls
remain with the serialized device owner. No hardware command was run here.

## Follow-up: saved vectors localize a common early-fidelity discrepancy

The main agent supplied `qkv4149_mantissa.json` and `.pt`. Weight splitting does
not repair the projection: three-bit high weight plus exact residual reduces
the complete three-term max error only to approximately 0.02497; four-bit high
weight gives approximately 0.02956. HiFi3 is approximately 0.02940, while HiFi2
is approximately 0.2228. These remain rejected component candidates.

The saved `.pt` allows a stronger CPU-only test of the high activation
component. Let `xh` retain six explicit activation mantissa bits, `xl=x-xh`,
`wh` retain four explicit weight mantissa bits, and `wl=w-wh`. The intended
fidelity contributions are `xh@wh`, `xh@wl`, `xl@wh`, `xl@wl`, with the real
device-read BF16 values used throughout.

| CPU comparison of saved high-component outputs | Result |
| --- | ---: |
| HiFi4 minus ideal four-phase sum, RMS error | 0.0028575972 |
| HiFi3 minus ideal three-phase sum, RMS error | 0.0028578032 |
| HiFi2 minus ideal two-phase sum, RMS error | 0.0028575104 |
| Correlation of `(HiFi4-HiFi3)` with ideal fourth contribution | 0.9999602604 |
| RMS residual after subtracting ideal fourth contribution from that difference | 0.0000113620 |
| Correlation of `(HiFi3-HiFi2)` with ideal third contribution | 0.9999999592 |
| RMS residual after subtracting ideal third contribution from that difference | 0.0000090462 |

The extra phases are present and numerically match their intended contribution.
The material approximately 0.002858 RMS error is already shared by the first
two fidelity phases, before adding the third/fourth contributions. This is a
narrower finding than blaming generic FP32 output packing or an absent final
phase. It does not yet identify a faulty primitive or prove a hardware defect.

Two additional CPU emulations were run against the **signed saved error vector**:

- Per-product truncation and round-to-nearest-even over 7–13 retained mantissa
  bits, for both the full BF16 product and just the phase-zero product. None
  reproduces the observed error vector. The 10-bit full-product truncation model
  has error correlation -0.0813, despite a superficially similar error scale.
  Evidence: `qkv4149_cpu_rounding.json`.
- Product alignment to a shared exponent within groups of 8/16/32 K entries,
  retaining 10/12/14/16/18/20 bits with truncation, floor, or nearest rounding.
  None meaningfully reduces the material residual; the best tested RMS remains
  0.00285744. Evidence: `qkv4149_cpu_alignment.json`.

These controls refute those particular rounding models. They do not justify a
broader claim that every internal adder implementation is ruled out. Source
inspection establishes the intended four-pass replay and FP32 accumulation
configuration, but the remaining early-phase arithmetic discrepancy needs a
minimal runtime control.

### Prepared minimal controls

`--qkv-small-controls --qkv-controls-only` runs a standalone diagnostic at the
original K=2816/N=8192 geometry using the saved activation and real BF16 weight:

- One nonzero K entry with unit and real BF16 amplitudes at K indices
  0,15,16,31,32, the maximum-magnitude activation index, and the final index.
  Unit amplitudes test weight transport; a BF16-times-BF16 scalar product fits
  in FP32 exactly and tests multiplication without a nontrivial reduction.
- Two/four/eight/16/32 contiguous real activation entries, both from K=0 and
  from the tile containing the largest activation (index 2575 in this artifact).
  This distinguishes errors that require multiple active multiply terms and
  exposes face/tile boundaries at the real geometry.
- Dense activation/weight with six/four explicit mantissa bits respectively,
  under LoFi, HiFi2, HiFi3, and HiFi4. All later ideal fidelity contributions
  are zero for this pair, so results should differ only by any real execution
  or rounding effect of requesting those extra passes.

Each control records actual dtypes, shape, fidelity, nonzero K count, full
comparison metrics, and the worst expected/actual scalar. It neither changes
the production decoder nor relaxes the acceptance requirement.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_qkv \
  --activation-input models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149.pt \
  --layer 0 --qkv-small-controls --qkv-controls-only \
  --probe-output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149_small.json
```

An independent `--qkv-fidelity-controls LoFi --qkv-product-output ...` run on
the original full operands will separate the first and second fidelity
contributions directly. The probe compiles and passes Black; all new device
controls remain unrun by this investigator.

## Follow-up: isolate dot lanes using the padded M rows

`qkv4149_small.json`, supplied by the device owner, gives the decisive contrast:

- Every tested one-hot unit and real BF16-amplitude projection is **exact**
  against the FP64 oracle, across the tested face/tile boundaries and large
  activation index. Weight transport and isolated scalar products pass.
- Two active adjacent K lanes already fail: K=0/1 has max error 0.00003242493;
  K=2560/2561 has max error 0.00025373697. More active entries increase the
  discrepancy, reaching 0.00977135 for 16 entries including the largest input.
- Dense phase-zero-only operands produce **identical** erroneous results under
  LoFi/HiFi2/HiFi3/HiFi4: max error 0.02948755 and RMS 0.00285169. Later mantissa
  cross products are zero for these inputs, so extra fidelity cannot help.

The supported boundary is now the FPU dot computation when it combines multiple
nonzero products, not input decomposition, weight orientation, final uniform
TF32 packing, an absent later fidelity phase, or a long K reduction. The source
MVMUL primitive consumes 16 K lanes at a time; its source comments describe
`D[8,16] = B[8,16] * A[16,16]` in `llk_math_matmul.h:57-65`. The observations
do not identify a specific ISA defect, but they directly support a narrower
stage-local workaround without more speculative rounding models.

For each BF16 activation component, create 16 logical M rows. Row r retains
only K positions with `k % 16 == r`. Then every dot16 in every row has at most
one nonzero product, matching the passing one-hot contrast. Project all rows
with one matmul and reduce the FP32 results over M using the accurate SFPU sum.
The sum across the 16 rows is mathematically the original component projection.

This can use work the ordinary M=1 decode GEMM already pads: concatenate the
16 masked high rows and 16 masked low rows into M=32, still one M tile. A
three-component version uses M=48/padded64. The original BF16 weight is shared
by the single GEMM; this avoids issuing 16 separate full-weight GEMMs. The
FP32 result has 32 or 48 logical rows rather than a full K-by-N broadcast
product. Setup stores only a small BF16 0/1 lane mask. These are structural
cost observations, not an unmeasured speedup claim.

The new `--qkv-lanes 16` candidate performs precisely that graph. It preserves
the original weight and tied-K duplication, uses explicit HiFi4 and FP32
destination/output, and reports lane-mask dtype, packed M rows, actual operand
dtypes and warmed component timings. The equal-BF16 0/1 mask product preserves
the component values; the subsequent FP32 sum is the intended replacement for
the inaccurate intra-dot combination.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_qkv \
  --activation-input models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149.pt \
  --layer 0 --qkv-blocks '' --qkv-terms 2,3 --qkv-lanes 16 --qkv-product-details \
  --probe-output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/qkv4149_lanes.json
```

The probe is prepared and passes compilation/Black. Retain the candidate only
if the same-input error improves and the unchanged full decoder gate passes;
one-hot evidence alone does not prove the masked-row projection's final sums.
No hardware commands were run by this investigator.

## Lane-partition component result and runtime candidate

The device owner supplied `qkv4149_lanes.json`. Its same-input, actual-device-weight
comparison confirms the proposed mitigation. All listed candidates replay exactly.
Times below are warmed traced component host measurements from that artifact;
they are not whole-layer latency or device-profiler durations.

| Projection | Maximum absolute error vs FP64 | RMS error | Component host time (us) |
| --- | ---: | ---: | ---: |
| FP32 broadcast | 0.000009525 | 0.000001566 | 2544.99 |
| Ordinary FP32 input GEMM | 0.0444431 | 0.00425655 | 124.48 |
| Two BF16 terms, ordinary GEMMs | 0.0301116 | 0.00285818 | 267.42 |
| Two BF16 terms, 16 lanes | 0.000204989 | 0.000020093 | 292.06 |
| Three BF16 terms, ordinary GEMMs | 0.0300515 | 0.00285772 | 401.73 |
| Three BF16 terms, 16 lanes | 0.000025800 | 0.000004500 | 347.98 |

The three-term lane candidate reduces maximum error by approximately 1165x
relative to the ordinary three-term candidate, and its measured component time
is approximately 7.31x shorter than broadcast. The remaining error is small
enough to justify the unchanged layer gate, not to waive it. The two-term lane
result is now limited primarily by activation reconstruction; the captured
activation reconstructs exactly with three BF16 terms. The artifact proves
FLOAT32 input, BFLOAT16 components/mask/weight, FLOAT32 product/output, HiFi4,
FP32 destination accumulation, approximate math disabled, and L1 packer
accumulation disabled.

`tt/optimized_decoder.py` now provides `LanePartitionQKV`. It uploads the 0/1
mask during construction; decode uses device typecast, subtract, masked
multiply, concat, linear and sum operations. With `qkv_lanes=16, qkv_terms=3`,
the default graph matches the passing M=48 component probe. Prefill delegates
the original projection. Global tied K/V projects stored Q/K and restores the
K tail. Original broadcast storage remains retained while acceptance is pending.
At initial integration the factory default `qkv_lanes=0` left the candidate disabled, and nonzero
lanes take precedence over the older `compensated_qkv` experiment.

Optional tuning controls are `qkv_grid=(gx,gy)`, `qkv_block_w=1`,
`qkv_subblock_w=1`, and `qkv_separate=False`. An explicit grid selects the 1D
input-multicast program with `per_core_M=ceil(lanes*terms/32)`, N coverage rounded
up across the grid, and fused batch. Setup checks the actual device grid,
K-block divisibility and FP32 output-subblock limits. Separate projection splits
Q/K/V weight columns during setup (Q/K for tied full attention), projects the
same packed activation, and concatenates reduced outputs. These tuning controls initially had source-level validation only; subsequent hardware verification is recorded below.

Python compilation and Black pass. The device owner is responsible for the
serialized unchanged 4096+128 whole-layer correctness/trace run before retaining
or promoting this candidate. No hardware commands were run by this investigator.

## Whole-layer projection validation

The device owner completed `headline_lanes16_terms3_retry_layer0.json` and
`headline_lanes16_terms3_retry_layer5.json`. Both unchanged real-weight 4096-token
prefill plus 128 successive decode checks pass the PCC 0.995 contract with
16 lanes and three activation components. Sliding layer 0 has minimum decode
PCC 0.9989250262 and median traced host time 2198.7658 us; full layer 5 has
minimum decode PCC 0.9995477099 and median traced host time 2325.0594 us. Both
reports record clean prefill/decode runtime audits, the program-cache miss
guard, and exact repeated decode equality. The constructor required explicit
shape indices because `ttnn.Shape` does not support Python slicing; the device
owner corrected this before these passing runs.

This closes the projection investigation for the stated 4096+128 validation
contract. Further tuning and longer stress coverage remain separate work; these
results do not establish universal route equality or all-length correctness.


## Parent verification after integration

The constructor's ttnn.Shape slice was corrected to explicit dimension indexing.
`headline_lanes_retry_commands.json` runs the real4096/128 whole layers with
three terms: sliding minPCC .99892503, traced host2198.77us; full .99954771,
2325.06us. Two-term runs also pass (.99894153/.99954286), at2142.68/2265.19us.
`qkv_geometry_commands.json` verifies packed and separate Q/K/V with32/64
workers,K11/22 on both layer kinds; packed64/K11 is fastest in that sweep.
A72-worker/subblock4 full-attention headline candidate also passes.
`headline_final_tune_commands.json` rejects HiFi2 QKV on both headline kinds
and LoFi sliding QKV by real-weight PCC. The selected projection retains HiFi4,
lane16 and two terms. Runtime defaults now enable this path. This fixes the
original compensated-projection blocker; final default stress/profile/review
results are recorded by the stage owner in the README and work log.
