# Resumed-prefill policy investigation (inspection only)

No devices were used and no implementation was changed.

## Cache and caller audit

`tt/generator.py:prefill_logits_continuation` resets once, prefills exactly `[0,start)` into `self.cache`, then invokes `[start,end)` with the same cache/page table and absolute `start`. `prefill_forward` slices the input accordingly. `tt/model.py:prefill` passes identical layer/slot state objects and absolute start to each decoder. There is no reset between prefix and suffix. The B1 pooled linear state reset clears the shared backing, which is correct for this whole-generator standalone reset. The API will need slot-scoped resets for independent multirow serving, but that is not the observed B1 failure.

`MultichipDecoder.prefill_forward` retains logical recurrent length, pads full-attention rows internally, and uses one-token updates until an unaligned absolute position reaches a page boundary. These mechanisms do not expose a definite cache ownership bug in the failing cases. Existing no-history/fresh-suffix controls fail decisively. This is not a proof that every kernel/cache operation is correct.

## Concrete shape-dependent mismatch

The selected prior-stage contract explicitly says norm output and prefill RS/AG payload use BFP8 at 512+ operation rows (`doc/optimized_multichip_decoder/README.md`, Selected implementation). `MultichipDecoder.norm` and `reduce` implement that selection using the current segment's row count.

- Full logical 513: linear layer has 513 rows; full attention pads to 544; both select BFP8 norms and reductions.
- Split at 32: prefix32 and suffix481 linear rows select BF16. Full attention prefix32 selects BF16, while suffix481 pads to512 and selects BFP8.
- Split at 68: linear prefix68/suffix445 both select BF16. Full-attention prefix68 pads to96 (BF16), suffix first uses 28 one-token decode operations for absolute positions68..95, then 417 rows pad to448 (BF16). One-token decode reductions already use BF8, while decode norms remain BF16.
- Split at512: prefix512 selects the same BFP8 path as full513; the suffix is only one logical row. This is the case that already passes.

Thus a token's precision depends on how the caller partitions prefill, before any cache reuse is considered. The observed passes/failures align with that mechanism. Projection program families also differ at512; removing BF8 only in `norm`/`reduce` makes the reduced case pass despite retaining those projection family differences, isolating the quantization boundary much more closely than a blanket BF16 run would.

`readiness/prefill_resume_check.json`: all-layer starts32/68 fail, start512 passes. `doc/full_model/reduced_resume.json`: layers0/3 starts32/68 top1=0.976744/0.961240; `reduced_resume_bf16.json`: both=1.0, all cases pass with both negative controls failing. The diagnostic implementation is `tests/full_model_resume_candidate.py`.

## Minimal discriminating next experiment

Keep original full513 reference and all weights, fidelities, cache dtypes, residual topology and projection kernels. On only the prefix/suffix execution, explicitly carry the *canonical logical prefill policy* into norm and reduce, so fragments of the original long prefill retain its BFP8 norm-output and RS/AG payload. Do not change decode globally. Run three controls: norm-only, reduction-only, and both. Cover single-token full-attention alignment helpers as prefill fragments: their norm path must receive the same prefill policy; their reduction is already BF8. Record first layer/input-normalization/projected-QKV/cache/state/output mismatch, per-position logits and top1 margins, rather than only final PCC.

A smaller first probe is the original starts32 case with both norm/reduction forced BFP8 for its short prefix and linear suffix. This directly tests the policy mismatch without the unaligned decode helper confounder. Then start68 exercises that helper.

This is an eligible **diagnostic hypothesis**, not an already proven production repair. It keeps the optimized long-prefill BFP8 computations rather than promoting them to BF16, and avoids fixed32 fragmentation. However, strict preservation of the exact existing shape-conditioned policy cannot also guarantee partition-independent numerical execution: the original policy itself deliberately chooses different dtypes for equivalent token prefixes under different partitions. A request-owned policy still requires design for prefix reuse when the final request length was not known while filling that prefix. Do not silently declare this resolved by forcing all short prompts to BFP8.

If the above probe passes, the earliest inconsistent calculation is policy selection from fragment length. Production work should make that contract explicit and test prefix keys/state compatibility; no already-validated fully policy-preserving repair has been established by these artifacts. If it fails, compare first-layer norms and per-position recurrent/cache state before attributing remaining error to kernel precision. A fixed32 candidate would avoid BF8 but would be a slower BF16-policy fallback and is not an eligible selected solution under the user contract.

## Follow-up: all-layer BF16 control invalidates a normalization-only resolution

`full_resume_bf16.json` still fails: min PCC0.919710 at start32,0.854162 at start68; top1=0.992248/0.976744. `reduced_resume_fixed32.json` passes all starts. The earlier reduced BF16 result therefore establishes one significant discrepancy, not a complete causal explanation for the stack. Fixed32 simultaneously changes norm/CCL precision, projection kernels, MLP fusion and recurrence partitioning; it does not isolate GDN.

Read `doc/full_model/autofix/AUTODEBUG.md`. Its main diagnosis concerns trace allocation, but its insistence on bounded physical profiles is also relevant: normalization alone cannot make fragment-dependent projections equivalent.

Exact remaining boundaries in `multichip_decoder.py`:

- `linear`: 512..1024 uses explicit 2D matmul for packed_input/out/down, shorter shapes use minimal_matmul, and one-token uses decode_linear.
- `finish`: <=128 rows use separate gate/up plus explicit multiply/SILU; larger rows use packed minimal_matmul with fused SwiGLU. The MLP remains LoFi in both, but intermediate accumulation/rounding and fusion boundaries need not match.
- `linear_attention`: actual q/k/v are BF16 before GDN; g is FP32, beta BF16; padded rows have g=0,beta=0. GDN output is cast to BF16 before postnorm. Its recurrent state stays FP32 throughout the caller's copy and pooled state write/read.
- Native `chunk_gated_delta_rule.cpp` computes NC from the current invocation, selects fused/phased plans based on that NC, explicitly normalizes raw flat q/k inside the prep kernel, and uses32-token recurrence blocks. Start32 preserves those internal block boundaries; start68 changes them.
- Native final-state spec is FLOAT32; initial-state remains FLOAT32 if already so. The scan carries state in FP32 CBs and the final writer uses the final-state tile size. There is no source evidence of a BF16 state-export cast. A numeric roundtrip defect remains testable, but should not be asserted from the fixed32 pass.

Added `tests/full_model_resume_policy_probe.py` (diagnostic only). It loads real layer0, records the actual long-prefill normalized projection input, packed MLP input, reduction input, and GDN Q/K/V/g/beta plus an independent initial-state copy. It then compares:

1. Full versus sliced **identical normalized inputs** through shape-selected packed-input projection.
2. Full fused packed MLP versus short separate gate/up on **identical post-attention-normalized inputs**.
3. Full versus sliced reduction inputs and native versus explicitly BFP8 normalized fragments.
4. Full GDN versus two calls on **identical preprocessed Q/K/V/g/beta**, using real device final-state -> initial-state handoff. Reports output/state PCC, max error, exact equality and first changed token row. Start32 isolates export/shape-selected implementation without changing32-token block boundaries; start68 includes genuine regrouping of recurrence math.

Suggested first commands (external serialized hardware owner must provide timeout):

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_resume_policy_probe --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/resume_policy_native.json
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.full_model_resume_policy_probe --bf16-prefill --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/resume_policy_bf16_control.json
```

These are not whole-model gates. The optional BF16 control only diagnoses the remaining all-layer failure; it is not an eligible selected implementation. Probe was syntax-compiled and passed ruff; no TTNN import or device execution occurred during authoring.

Strongest policy-preserving repair **hypothesis** after this evidence: canonical physical prefill profiles retaining BFP8/HiFi2/LoFi, plus projection/MLP execution that is consistent across continuation fragments, rather than replacing the optimized dtype policy or shrinking every chunk to32. Keep full fused projections and packed SwiGLU for long prefills; adapt short continuation staging/masking and choose a measured compatible kernel family without changing effective precision. First prove where the first discrepancy occurs. If identical-input GDN aligned32 is bit-equal while packed-input or MLP differs, investigate those projection/fusion boundaries before state serialization. If aligned32 GDN differs materially, pin fused/phased config in an isolated control and inspect first exported/imported FP32 state before changing model precision. If only unaligned68 GDN differs, preserve global32-token recurrence block phase through a partial-block continuation representation; restarting the local block phase is the candidate inconsistency, and prefix snapshots would need that partial-block state. None of these repairs is yet verified or selected.

## Canonical-BF8 continuation diagnostic

`tests/full_model_resume_candidate.py` now accepts `canonical_bf8` (norm plus reduction), `canonical_bf8_norm`, and `canonical_bf8_reduce`. The source substitutions assert each exact original match and retain native weight/fidelity/cache/projection/layout choices. A continuation wrapper sets a temporary per-layer policy from the original logical request length, enclosing both prefix and suffix execution. At start0 it disables the override, so the full513 reference remains original. The flag is restored in `finally`; ordinary decode outside this diagnostic remains native. Full-attention single-token alignment helpers receive BFP8 norm output when the norm control is active; their native decode reduction already uses BFP8 payload. Reduction output remains BF16.

This diagnostic is deliberately limited to requests fitting one original chunk per layer. It does not claim to solve policy state for multiple long chunks or reusable prefixes from requests whose final length was unknown. Native short negative-control requests retain their native policy. Reported flags make that distinction inspectable.

Use the existing readiness invocation and replace `--candidate bf16_prefill` with one of these three choices. No hardware run was performed by the author. Ruff and Python compilation passed; a host-only AST extraction also verified that both transformed decoder methods compile, without importing TTNN.
