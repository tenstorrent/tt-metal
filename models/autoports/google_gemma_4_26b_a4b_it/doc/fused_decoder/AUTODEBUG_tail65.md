# AutoDebug: sliding decode after a 65-token prompt

This is a source-only, fresh-context investigation under AutoFix. No hardware
was opened by this investigator and no implementation was edited. The parent
owns all device experiments. The frozen fused runtime SHA256 is
`0c8892be32e04202fdbddd2b06850848c63ae57660c49d03c4d149a5a88b041c`.

## Evidence and present conclusion

Real-weight, seed-42, batch-1 sliding attention prefill at length 65 passes:
fused PCC 0.9987311694824392 and functional PCC 0.9981650866707974. Both fail
only position 68 in the eight-position decode stream. Fused PCC is
0.9813712958993235 and functional PCC is 0.9809274230811537; neighboring
positions pass and repeated execution of the unchanged first position is
equal. Sources are `verified_tail65_sliding.json` and
`verified_tail65_sliding_functional_control.json`.

The parent subsequently retained the failing outputs and compared both TTNN
paths directly. `tail65_sliding_equivalence.json` records prefill PCC
0.9991649499445143, minimum decode PCC 0.9997196428713423, and position-68 PCC
0.9998401881828624. This verifies numerical preservation against the functional
baseline for this stream. It does **not** turn either path's failed HF check
into a pass or establish the numerical source of the shared failure.

At initial report creation the leading hypothesis was a shared
attention/residual error changing the discrete rank-eight/rank-nine expert
selection. The parent subsequently ran the controls below. They confirm an
inherited BF16-cache precision limitation at this input; no fused-runtime
regression or runtime fix is demonstrated.

## Source checks

- `tests/run_decoder.py` seeds before loading real weights and generating the
  prompt, generates the decode stream in order, and advances the FP32 HF
  `DynamicCache` once per logical position. TT trace warmup/capture/repeats
  overwrite the same first cache row. Subsequent inputs and both device
  position tensors refresh before replay. No obvious one-step shift appears.
- At this workload the harness allocates 1,024 tokens, or 32 pages of 32 rows,
  with reversed page order. Position 68 is logical page 2, row 4, physical page
  29. It is neither a page/tile nor a 1,024-token sliding boundary. Prefill
  internally pads 65 to 96 rows; each decode overwrites its valid row and the
  attention mask excludes future padding. `PrecisePagedAttention` reads at
  most all 32 allocated pages and masks absolute positions above 68. There is
  no evident rounded-read under-allocation or boundary cliff here.
- Both paths inherit BF16 paged K/V and BF16 uploaded RoPE tables. HF uses
  FP32 tables and FP32 cache. `routing_precision.Router` already uses FP32
  SFPU decode projection and selects logits before BF16 route storage, so
  probability rounding **before** top-k is not the implemented path.
- `tests/run_decoder.py --diagnostic` only probes initial decode. Existing
  `tests/probe_sequence_decode.py --inspect-position 68` instead records the
  failing position from the same traced stream and needs no runtime edits.

## Focused verify/refute experiments

1. **Shared residual drift changes expert selection.** Run:

   ```sh
   python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_sequence_decode \
     --decoder functional --layer 0 --length 65 --real --decode --steps 8 \
     --verify-program-cache --inspect-position 68 \
     --stage-report models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/tail65_functional_stages.json \
     --save-inputs models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/tail65_inputs.pt \
     --output models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/tail65_functional_diagnostic.json
   ```

   Compare `hf_routes`, `cpu_same_residual_routes`, and `tt_routes`. If CPU on
   the TT residual selects the TT set while HF selects another, the selection
   change originates before the router. `attention_pcc`, `residual_pcc`,
   `sdpa_same_qkv_pcc`, and `cpu_sdpa_o_routes` distinguish upstream K/V/Q
   differences from attention arithmetic. If CPU on the same residual selects
   HF's set, localize the router arithmetic instead. The extra readbacks and
   comparison matmul are diagnostic only and do not supply performance data.

2. **BF16 cache/RoPE precision alone reproduces the rank discontinuity.** Run
   the existing CPU-only control after the above saves the exact input stream:

   ```sh
   HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python_env/bin/python \
     -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_long_decode_precision_controls \
     --length 65 --steps 8 \
     --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/tail65_inputs.pt \
     --output models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/tail65_cpu_precision.json
   ```

   This asserts exact agreement with the device harness's seeded inputs and
   compares FP32 HF against cache-only and cache-plus-RoPE BF16 rounding.
   Inspect position 68's selected expert IDs, rank-eight/rank-nine gap and
   stage PCC. Reproduction establishes sensitivity to the existing precision
   boundary, not emulation of TT arithmetic or universal irreducibility.

3. **Fallback if these controls do not explain the error.** Use the existing
   same-Q/K/V CPU SDPA control before changing precision. A large discrepancy
   there calls for an exact cache/page-table readback or eager-versus-trace
   replay control at position 68. A high-precision passing control would
   motivate a single boundary-specific candidate, not a blanket precision
   change. Do not reduce the PCC threshold or reject logical length 65.

## Completed controls and verdict

`tail65_functional_stages.json` records position-68 attention PCC
0.9999978049318797 and residual PCC 0.9999971752061487. HF chooses expert 10
at the cutoff, while TT chooses expert 70. CPU routing on the **same TT
residual** selects the TT set, as does CPU SDPA and output projection on the
same actual paged Q/K/V. Same-Q/K/V SDPA PCC is 0.99999985012578; same-input
QKV PCC is 0.999999999344115. Thus the observed set change is upstream of
router selection, and the attention kernel is closely matching its actual
inputs. Higher router projection fidelity is not the missing fix.

`tail65_cpu_precision.json` verifies the exact saved device inputs match CPU
seed reconstruction and records `ttnn_imported=false`. With **all arithmetic
and RoPE kept FP32**, rounding only stored K/V through BF16 reproduces the
same expert-10-to-70 substitution and the only failing position, 68, at PCC
0.9814915923967622. The FP32 reference's rank-eight/rank-nine logit gap is
0.0004329681396484375. Attention and router-input PCC stay above 0.999998,
then expert output PCC falls to 0.9104965613860845 after the discrete route
change. Adding BF16 RoPE storage gives the same failure and selected set at
PCC 0.9815510237835411; RoPE rounding is not required to trigger this case.

The precision controls retain identical real BF16 checkpoint values converted
to FP32 for CPU arithmetic, identical input tensors, identical HF operators,
and the same logical sequence. Cache-only rounding is the independent
variable. Actual TT caches remain BF16 TILE `[32,8,32,256]`, reversed page
table, 32-token page/update rows; no allocation, mapping, layout, or precision
contract was changed. The diagnostic reconstructs logical K/V using the
actual page table and actual cache tensors rather than a nearby shape.

**Verified:** the existing BF16 cache boundary alone is sufficient to reproduce
the near-tied MoE route discontinuity. Both TT implementations preserve that
boundary, and their direct comparison passes 0.995 for prefill and all eight
decode outputs. This is a controlled inherited numerical limitation, not a
fusion regression. It is not a claim that every possible precision-policy
change is physically incapable of avoiding the case; changing the public
cache contract would be a separate capability decision.

Keep the frozen runtime. Add a durable mixed-64-plus-32 prefill-tail regression
that captures the default functional and fused outputs at logical length 65,
compares prefill and all eight decode outputs at the unchanged 0.995
threshold, and retains both exact FP32-HF result records including the
position-68 failure. Require the full-attention counterpart to retain its HF
passes. Do not mark the anomalous sliding HF check as passed, silently discard
position 68, loosen the threshold, or impose an alignment restriction. The
final accuracy report should distinguish functional-equivalence acceptance
from this explicitly controlled HF outlier. Exact diagnostic commands and
process results are in `tail65_diagnostic_commands.json`.
