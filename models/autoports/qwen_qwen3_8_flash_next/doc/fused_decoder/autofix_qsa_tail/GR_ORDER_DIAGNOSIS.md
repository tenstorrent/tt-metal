# GR projection backend-order hypothesis

2026-11-13; source-only investigation. No TTNN imports, device access, target forwards, arithmetic experiments, implementation edits, or changed gates. This report is the only authored file.

## Conclusion

A generic, model-local ordered projection is a plausible repair candidate, but **GR equivalence to literal HF is not yet established**. Source supports a sharply specified hypothesis: on the recorded AVX512 BF16 BRGEMM backend, contiguous GR-down with more than one row should use ten independent K1024 partials; singleton GR-down should use twenty K512 partials; GR-up should use one K320 partial. Each partial consumes adjacent BF16 pairs high member then low, rounding each accumulator update to FP32; partials combine in increasing K order, followed by BF16 round-to-nearest-even. This is derived by substituting GR dimensions into the backend configuration, not by assuming the indexer rule transfers unchanged.

The missing prerequisite is dispatch and same-operand evidence on the current layer3 fixtures. The old GR verbose evidence is unusually useful: it already records this backend for **GR-down K10240,N320**, with transposed weights. However, its M=2 operands come from layer0, not layer3 M=34/66. No exact-fixture GR-up backend log or ordered-output comparison was located. The current failure remains a literal-HF failure; neither FP64 control agreement nor an eventual isolated projection match clears the decoder gates.

## Source and dispatch evidence

Installed `python_env/lib/python3.12/site-packages/torch/version.py` reports PyTorch `2.11.0+cpu`, commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`. Installed HF `modeling_qwen4_exp.py:995` constructs bias-free down/up `nn.Linear` modules. Its GR forward normalizes, applies down, BF16 division by four and SiLU, applies up, then sigmoid and stream mixing. The [matching PyTorch linear source](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/Linear.cpp) routes ordinary bias-free linear through `matmul(input, weight.t())`. Python formulas alone do not identify the selected native kernel.

`../../functional_decoder/gr_backend_verbose_cpu.log` records oneDNN v3.10.2 commit `f1d471933dc852f956fd05389f9313c7148783d5`, eight OpenMP threads, AVX512 BF16 support, and:

```text
brg_matmul:avx512_core_bf16
src bf16 ab; weights bf16 ba; destination bf16 ab
2x10240 : 10240x320
```

It also records the historical sensitive output `down[0,0,34] = 2.171875`. This is direct old GR dispatch evidence, unlike the indexer's `1x32x2560:1x2560x640`, `abc`/`abc` projection. Both installed version and HF source identity agree with the earlier reports, but a historical log does not prove dispatch for today's exact operands.

The retained source excerpts in `../../functional_decoder/onednn_sources/` and the [pinned full configuration source](https://github.com/uxlfoundation/oneDNN/blob/f1d471933dc852f956fd05389f9313c7148783d5/src/cpu/x64/matmul/brgemm_matmul_utils.cpp) support the conditional calculation below. The full source was also inspected to check the intervening low-parallel-work branch omitted by the old excerpt: that branch can alter M/N blocking and B copying, but does not change this K choice. The AVX512 heuristic starts with K1024 for nontransposed source and K>1024, otherwise min(K,512); BRGEMM batch size is one. K partitioning requires conditions absent here: BF16 transposed-source partitioning requires power-of-two K, while huge-K partitioning is FP32-only. Neither 10240 nor 320 is a power of two. Existing retained original-source lines 866–870, 904–921, and 975–976 expose the key predicates.

| Projection and effective source geometry | Conditional K blocks | Why |
|---|---|---|
| Down, M=34/66 or a contiguous M=2..32 call | 10 × 1024 | K=10240; nontransposed source |
| Down, M=1 canonical call | 20 × 512 | Singleton tag ambiguity resolves using `helper.transA()` |
| Up, M≥1 | 1 × 320 | K=320 is below either default |

The singleton interpretation is documented and source-linked in `../../functional_decoder/onednn_q1_projection_order.md`. It still needs a GR-specific shape control. In particular, a singleton tail of a multirow HF prefill must not automatically select singleton-HF arithmetic merely because TT executes the tail as Q1. Full-call versus split-call equality must be measured, and the projection policy may need to retain original logical call geometry. Independent batched decode needs the same scrutiny: HF may flatten batch into M while TT loops requests.

The retained `jit_brgemm_kernel.cpp.excerpt.txt` and `brgemm_matmul.cpp.excerpt.txt` establish per-output accumulator reuse across increasing adjacent K pairs, independently initialized partials, beta-one FP32 addition of previous C, and final BF16 conversion. The instruction-order rationale and its finite-normal operand limitations are in `../../functional_decoder/onednn_projection_order.md`. It does **not** imply that a pair sum, a tree, ordinary matmul, or FP64 accumulation is interchangeable with this order.

## Exact missing evidence

1. Capture all literal layer3 normalized down operands and activated up operands, weights, BF16 outputs, shapes, strides, storage offsets, dtypes, and hashes for QSA34/66. The present saved control caches contain final outputs, not these operands. Keep full prefixes: row30/44 can depend on preceding attention inputs. Record attention and MLP separately; the all-GR FP64 reversal does not isolate either up projection.
2. Record verbose dispatch around the literal projection calls, with explicit markers identifying projection, original call shape, and phase. Preserve the eight-thread setting and backend/environment identity used by the existing oracle. Do not silently replace `F.linear` with a transposed contiguous matmul or alter input ranks; either can change dispatch.
3. On identical captured operands, compare every literal BF16 result against the predetermined ordered candidate, plus FP64 and the existing balanced-tree controls. Store raw ordered FP32 and BF16 results and all mismatch coordinates. Record finite/normal-product conditions; products of ordinary BF16 operands are exactly representable in FP32, but exceptional underflow/overflow/NaN behavior needs explicit emulation if encountered.
4. Repeat the same-operand literal call at original M=34/66 and concatenated M=32 chunks/tails, then Q1 and independent holdout M=1,2,31,32,33,65 inputs. Report full-versus-split differences instead of assuming chunk independence. Shape changes must never select a policy based on observed values or named failed coordinates.
5. Later TT observer evidence must establish normalized-input equality before attributing the first actual TT/HF divergence to down accumulation. If these inputs differ, a correct same-input projection alone cannot guarantee a decoder repair.

## Minimal CPU experiment design

Implement a stage test such as `tests/probe_gr_order_cpu.py` only after the parent's clean runs finish. No implementation patch is needed for this first discriminator.

The cheapest first phase need not load or run the full decoder. Read the exact cached prompts and layer3 attention GR norm/down/up checkpoint tensors; instantiate only the installed HF GR normalization or reproduce its source-exact operations. Calculate its normalized operand, then literal down, unchanged division/SiLU, and literal up. Mark each literal call for oneDNN verbose logging. This answers whether the source-derived order matches the exact attention projection operands. A later observer-only full literal decoder rerun must prove these independently constructed operands match the actual fixture and capture the MLP operands; it also reasserts bit-exact oracle equality.

Use the following fixed candidate for each output, with `block` determined only by the verified original CPU call geometry:

```text
total = undefined
for start in range(0, K, block):
    partial = float32(0)
    for k in range(start, min(start + block, K), 2):
        partial = RN32(partial + float32(x[k+1]) * float32(w[k+1]))
        partial = RN32(partial + float32(x[k])   * float32(w[k]))
    total = partial if start == 0 else RN32(partial + total)
result = RNE_BF16(total)
```

Vectorize across rows and a modest output-column block, never across the ordered K updates. This avoids materializing the full M×N×K products (about 0.86 GB FP32 for M=66 down) and avoids a reduction primitive/compiler reassociation. Independent CPU output tiles do not change per-dot arithmetic. Run literal projection once at its original complete N and M; slicing literal weights for convenience could select a different backend. Predetermine at least two independent random holdouts before inspecting results; use the real weights and BF16 operand distributions. Compare all outputs, not only the FP64-disagreeing coordinates.

Artifacts should include backend log, script/source/checkpoint/prompt hashes, tensor cache, mismatch totals by projection/shape/holdout, literal repeat equality, full-versus-split equality, exceptional-value checks, and raw failed candidates. CPU-only comparisons stay labeled diagnostic. Do not run a large full decoder intervention until the exact-operand candidate succeeds. Then start with attention-down-only ordered arithmetic; preserve the original literal whole/contribution per-row gates and separately add attention-up/MLP-up controls if needed to understand the reversal.

## Model-local implementation feasibility and limits

`tt/ordered_projection.py` and `tt/kernels/ordered_projection_compute.cpp` demonstrate a viable device mechanism: exact BF16 operands, high/low reader order, separate FP32 multiply/add, live FP32 partial accumulators, then FP32 output plus explicit `ttnn.typecast(..., bfloat16)`. `../../functional_decoder/ordered_projection_pack_review.md` shows why directly packing the accumulator is insufficient: two exact midpoint cases required the explicit RNE conversion. Existing inline kernel support in `tt/fused_decoder.py` permits a scoped implementation without editing shared TTNN or the functional decoder.

This is not a mechanical dimension substitution. The current indexer wrapper hardcodes K2560/N640, 20 output-tile cores, and two 80-tile scratch reservations. Down has 320 K tiles and only 10 output tiles: copying the entire two-scratch scheme would require 1,310,720 bytes per core before stream/output buffers. A bounded K-block streaming design or another explicitly budgeted layout is needed. Up has 320 output tiles and only 10 K tiles, requiring work scheduling across available cores rather than allocating one core per output tile. Preserve accumulation boundaries across streamed blocks and avoid partial BF16 stores. Performance and L1 fit remain unmeasured.

The narrow first candidate, if evidence supports it, is attention GR-down in the fused decoder. Do not change every GR projection simply because the mechanism is generic: all-GR FP64 already demonstrates non-monotonic downstream sensitivity. No per-input corrections, forced routes, host intermediates, or tolerance changes are justified.

## Inspected local identities

| File | SHA256 |
|---|---|
| installed `torch/version.py` | `ee3c04678fab86752390f05f0d5df9690ec1c854e4280a21731bd42bbf3d13e0` |
| installed HF `modeling_qwen4_exp.py` | `e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df` |
| `gr_backend_verbose_cpu.log` | `eaeea695a49fbad5792704b975b3de3cb49ca246940c081707c08b6ef2b9a451` |
| retained `brgemm_matmul_utils.cpp.excerpt.txt` | `f1ca8441292b2da8208a1729f6a0da4da43fe4c49c86bfa2e5dc7b2dfac4d685` |
| `tt/ordered_projection.py` | `a952d765473f0cb3e68205d9b588a2344726fabae5b30a2107035c13b255e4e4` |
| `tt/kernels/ordered_projection_compute.cpp` | `c9701ee9c1acb1c209463cf6cc19b6acc4e4fa397a0ba87f1ff62c45f2df9c77` |

Existing source provenance, instruction links, and excerpt identities remain in `../../functional_decoder/onednn_sources/provenance.json` and `q1_provenance.json`. This investigation did not verify the installed JIT's generated instructions or execute any candidate.
