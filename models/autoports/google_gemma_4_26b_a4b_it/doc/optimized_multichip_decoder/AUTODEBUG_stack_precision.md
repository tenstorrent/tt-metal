# AutoDebug: consecutive sliding-layer stack precision

Source-only investigation; no device execution, TTNN import, or runtime edits.

## Evidence

The repaired full semaphore grid produces finite, identical TP4 replicas and
identical repeated traces. The unchanged 0.995 gate still fails on consecutive
layers 0 and 1 with the real 4096-token input and 128 decode steps:

| Control | Layer 0 minimum | Layer 1 minimum | Layer 1 positions below 0.995 |
| --- | --- | --- | --- |
| QKV4 / attention CCL8 / MoE CCL8 | 0.997197 | 0.986605 | 4103, 4117, 4155, 4160, 4165, 4207 |
| QKV8 / attention CCL16 / MoE CCL8 | 0.997611 | 0.992153 | 4103, 4117, 4155, 4160, 4207 |
| QKV8 / attention CCL8 / MoE CCL16 | 0.997660 | 0.989809 | 4103, 4117, 4155, 4160, 4207 |

Artifacts: `stack_full_grid_actual_context.json`,
`stack_full_grid_baseline_precision.json`, `stack_qkv8_moe16_attention8.json`.
The repeated isolated positions, including 4103 and 4117, do not describe a
simple page-boundary cliff. This is suggestive of routing sensitivity, not proof.

## Source-supported hypotheses

1. **Decode precision differs in several untested upstream groups.** Sliding
   defaults in `tt/optimized_decoder.py:386-418` use BFP8 expert gate/up weights,
   unchanged BF16 expert inputs, BFP8 shared gate/up weights, HiFi2 QKV and HiFi4
   attention output projection. TP4 uses BFP4 expert gate/up and BFP8 expert
   inputs (`tt/multichip_decoder.py:894-909`), BFP4 shared gate/up
   (`:326-336`), and LoFi QKV/output projection (`:668-669`). Raising only QKV
   and CCL payloads does not align these groups. A layer 0 output above 0.995
   can still cross a layer 1 routing boundary and produce a larger layer 1
   difference. Verify each group separately, using identical input fixture,
   weights, cache/page table, and full semaphore grid.

2. **The router can amplify upstream drift, but an intrinsic TP4 router policy
   difference is not supported.** Both sliding paths use GeneralizedRouter,
   centered FP32 scores, BF16 score conversion, direct BF16-weight/FP32-input
   projection, HiFi4, K block 22, and indexed top-8 experts
   (`optimized_decoder.py:598-609,819-981`; `multichip_decoder.py:868-890,961`).
   TP4 moves the single-core gate to (10,9). Capture normalized residual,
   routes, selected indices, shared/routed contributions, and output at 4103
   and 4117. Compare expert sets (ordering alone is irrelevant) and the CPU
   float32 8th/9th score margin on each captured residual. Matching routes
   with a large routed-output difference points to expert arithmetic instead.

3. **TP1 is an approximation, not an HF oracle.** The stack harness runs TP1
   and TP4 on their own consecutive outputs and caches
   (`tests/test_multichip_stack.py:86-246`) and does not execute HF. Both use
   the rounded-score router and low-precision weights. Different TP1/TP4
   routes cannot establish which is closer to HF. CPU HF routing on each
   exact captured residual separates upstream-input drift from router error;
   a same-input layer-1 control separates propagation from local arithmetic.
   The 0.995 TP1 comparison remains required regardless of this distinction.

4. **Shared CCL buffers are a lower-priority remaining control.** Identical
   ranks and replay make the repaired semaphore bug an unlikely explanation
   for the present isolated low-PCC positions. However, deterministic
   collective arithmetic/lifetime errors are not refuted by equality alone.
   If precision-group controls do not help, rerun with `--no-pool-ccl` and
   compare boundary outputs. Do not conclude collective correctness solely
   from replica equality.

## Focused experiments

Use `probe_stack_precision.py` around the unchanged stack harness. Each switch
is independent and applies only to TP4; the output records actual policy,
selected layers, and wrapper hash. BFP8 gate controls reload original state
with exactly the current shard packing, rather than widening BFP4 tensors.
Expert activation control changes `decode_activation_dtype`, the attribute
actually used at `optimized_decoder.py:199-200`.

Start with the actual failing defaults and change only one of:

1. `--probe-expert-gate-bf8`
2. `--probe-expert-activation-bf16`
3. `--probe-shared-gate-bf8`
4. `--output-fidelity HiFi4` (existing harness option)

Use layers 0 and 1, length 4096, and 128 steps for acceptance. A short first-22
step run can cheaply assess the two earliest worst positions (4103/4117) but
cannot pass the stage. To locate cross-layer propagation, repeat a useful
control with `--probe-layers 0` and `--probe-layers 1` independently.
`--probe-attention-precision baseline` restores QKV8; preserve the chosen
attention/MoE CCL settings explicitly when comparing with an existing control.

No hypothesis is yet verified. Do not weaken the PCC gate or call the stage
complete. Any passing policy needs the original full stack, final single-layer
correctness/trace contracts, and new performance evidence before selection.

## Parent-run independent controls

The parent executed all four controls at length 4096 / 128 steps with the
unchanged QKV4, attention CCL8, and MoE CCL8 policies. All remained below 0.995:

| One changed group | Layer 1 minimum at position 4103 | Artifact |
| --- | --- | --- |
| Expert gate/up BFP8 | 0.990188 | `stack_control_expert_gate8.json` |
| Expert activation BF16 | 0.987297 | `stack_control_expert_activation16.json` |
| Shared gate/up BFP8 | 0.989331 | `stack_control_shared_gate8.json` |
| Attention output HiFi4 | 0.986903 | `stack_control_output_hifi4.json` |

These refute each single precision change as a complete fix; they do not refute
precision sensitivity. Expert and shared gate changes help more than activation
or output fidelity, but route-boundary measurements remain needed to explain it.

The parent then ran QKV8 + expert gate/up BFP8 + attention CCL16 while retaining
the other selected policies. `stack_qkv8_expert_gate8_attention16.json` passes
the complete length-4096 / 128-step layer-0-to-1 stack with minimum PCC
0.99653849. Compared with the already failing QKV8 / attention CCL16 control,
this isolates the added expert gate/up precision as sufficient to cross the
gate in that policy context. It does not establish the cheapest passing policy
or prove routing disagreement as the mechanism. Parent is testing alternatives
before changing defaults or making a performance claim.

`diagnose_stack_routes.py --layers 0 1 --length 4096 --steps 22
--capture-positions 4103 4117 --output <path>.json` retains the full prefill and
advancing decode sequence. It writes `.routes.json` and `.routes.tensors.pt`
even when the original PCC assertion fails. The diagnostic records TP1/TP4
boundaries, raw router scores, top-8 expert IDs, centered BF16 score margins,
and CPU HF routing on each exact attention residual. It performs no HF
attention/prefill/model run. Device clones preserve the pooled collective
boundaries before another layer overwrites them; retained handles and these
copies perturb allocation, so acceptance still requires an uninstrumented run.
The route wrapper also accepts every `--probe-*` precision-control switch, so
failing and passing configurations can be inspected through the same path.

Both new Python wrappers passed Black and AST parsing. Neither wrapper was
executed on hardware by the investigating subagent. No implementation or
acceptance-harness source was changed by this investigation.

## Follow-up: mixed sliding/full stack, layers 0 to 5

**Validity correction:** the following 0->5 experiments skip real layers 1..4.
Their full-layer inputs are an artificial composition, even though the layer-0
input and both layers' weights come from the real model. Retain them as stress
and localization evidence. Do not use their precision failures alone to veto
the faster full-layer policy; an adjacent 4->5 run on recorded boundary-4
activations is required. The proposed controls below remain diagnostic, not a
precision-selection obligation. This follows `.agents/skills/optimize/SKILL.md`
OPT-012, which requires real target-activation evidence for a precision veto.

After the parent selected sliding QKV8 / expert gate8 / attention CCL16, the
4096-token / 128-step mixed stack still fails. Full defaults retain QKV4,
expert gate4, expert activation8, LoFi QKV/output/expert math, and attention
CCL8. TP1 full defaults already use expert gate4, expert activation8, LoFi
QKV/output/expert math, and shared gate4; TP1 full QKV weights are BFP8.

| Evidence | Layer 0 minimum | Layer 5 minimum | Worst full position |
| --- | --- | --- | --- |
| `stack_selected_mixed.json` | 0.997644 | 0.973639 | 4160 |
| `stack_mixed_qkv8.json`, only full QKV8 | 0.997644 | 0.983449 | 4142 |

The QKV8 control preserves both prefill outputs exactly (full prefill PCC
0.9990718784542094). `_Projection.prefill` owns the original BFP8 weight;
decode precision changes replace only `_Projection.weight`
(`multichip_decoder.py:118-155,810-819`). This is a decode-sensitive result,
not proof of a full-prefill or cache defect. Full KV-cache inputs still reflect
the original sliding prefill, and no same-cache oracle has yet excluded that
contribution.

Use the full-QKV8 control as the common baseline. The focused matrix is:

1. Full attention CCL16: `--attention-precision baseline
   --full-attention-ccl-dtype bfloat16`. This isolates a remaining attention
   truncation/reduction boundary absent in TP1. Do this before raising full
   expert gate precision, which already matches the TP1 dtype.
2. Sliding shared gate8: `--attention-precision baseline --probe-layers 0
   --probe-shared-gate-bf8`. Parent reports this alone fails at full PCC
   0.9842445; it is not a sufficient repair.
3. Sliding expert activation BF16: same baseline plus `--probe-layers 0
   --probe-expert-activation-bf16`.
4. Sliding QKV HiFi2: same baseline plus `--probe-layers 0
   --probe-qkv-hifi2`. The wrapper now sets the selected layer's constructor
   argument, leaving full QKV LoFi. This matches TP1 sliding fidelity without
   changing its BFP8 weights or prefill computation.
5. Sliding output HiFi4: same baseline plus `--probe-layers 0
   --probe-output-hifi4`.

Cases 2-5 change only sliding decode, preserving the full layer's prefilled KV
inputs. Each is an independent test, not an assumed improvement. If the full
attention CCL16 control helps, carry it as a separately recorded baseline;
avoid silently mixing it into the earlier results. Acceptance remains full
4096/128 and PCC >= 0.995, followed by stage correctness/performance checks.

The existing route wrapper supports `--layers 0 5` and all these control flags.
Capture positions 4142 and 4191 with at least **96 steps**; 47 steps reaches
4142, whereas 51 does not reach 4191. Capture each layer's input, attention
output/residual, shared/routed outputs and route IDs. A collapse before the full
router prioritizes attention/CCL or upstream input; preserved attention residual
with a routed-branch collapse prioritizes expert arithmetic and routing-weight
sensitivity. Compare each full router against HF on its own residual before
attributing its different route to a TP1 reference defect. Do not infer which
complete stack is closer to HF from this local comparison.

## Actual adjacent sliding/full validation

The old stack harness always loaded `actual_text_layer0_4096_128.pt`, then
applied only the two requested layers. Consequently `--layers 0 5` was not an
actual target-model composition. Merely changing it to `--layers 4 5` while
retaining layer-0 input would also be invalid. The parent has added a fixture
argument, boundary metadata validation, fixture hashing, and adjacency metadata.

`create_layer4_fixture.py` creates the required actual boundary without loading
or executing a complete model. It uses the existing CPU-only
`load_real_layer`, `evaluate_layer`, and `save_fixture` helpers. Starting from
the original raw FP32 layer-0 prefill/decode tensors, it executes HF layers
0, 1, 2, and 3 sequentially, loading and releasing one checkpoint layer at a
time. The exact 4224 token IDs remain unchanged. FP32 states remain unrounded
between HF layers; only the saved boundary transport rounds to BF16, matching
the existing fixture contract. Chunked eager attention uses the helper's
explicit sliding cache and original absolute positions.

Run from the repository root:

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.doc.optimized_multichip_decoder.create_layer4_fixture \
  --output-dir models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_multichip_decoder \
  --threads 4 --prefill-chunk-size 1024
```

The output is `actual_text_layer4_4096_128.pt`, its normal fixture JSON, and
`actual_text_layer4_4096_128_capture.json` containing command, source fixture hash,
reference-source hashes, evaluated layers, and timing. Imports are CPU-only;
the script does not import TTNN or access a device. Black and AST checks passed;
the investigating subagent did not run the capture.

Validate the selected runtime with `--layers 4 5 --length 4096 --steps 128
--fixture <output-dir>/actual_text_layer4_4096_128.pt`. Preserve PCC >= 0.995,
trace-repeat equality, replica equality, and direct device handoff. A failure
on this actual adjacent path can justify precision changes; a pass limits the
0->5 evidence to its artificial stress composition. The valid 0->1 results
remain independent evidence for the selected sliding policy.

Boundary-4 decode activations contain their original 4096-token history and
absolute positions. Do not reuse them at `--length 33` or another shorter
prefix. Such a real contextual-boundary test requires a separately evaluated
HF boundary fixture with that exact prefill/decode split. Layer-0 embeddings
do not have this contextual dependency, which made the earlier shortening
less restrictive.

The parent completed the first 4096/128 capture through all four HF layers;
that initial run's manifest is `actual_text_layer4_capture.json`.
The generator now accepts `--length` and `--steps`. For a valid 33/128 fixture,
add `--length 33 --steps 128`: it selects raw layer-0 prefix rows 0..32 and
the original raw layer-0 decode rows, preserving token IDs from source ranges
`[0,33)` and `[4096,4224)`, then re-executes HF layers 0..3 at new positions
0..160. This is an explicit spliced teacher-forced token sequence. It does not
slice or reinterpret contextual boundary-4 decode activations. Provenance
records the skipped corpus range, source and target ranges, and a separate
selected-token manifest; capture manifests include length/steps in their names
so multiple cases retain their evidence.
