# AutoFix: routed decode accuracy

**Current status:** the integrated device-only implementation passes real
batch32 for both attention kinds and all nine request-reuse cases, including
31/32/33, 1023/1024/1025, 2047/2049 and repeated33. Threshold remains 0.995 per
reported case/slot. The earlier layer-dependent fidelity policy below was
superseded by the final repair described at the end of this report. The subsequent 4096-token/128-step checks also pass after the decode-QKV
repair documented below. Long-context, continuation and performance stage
regression checks remain owned by the parent task.

Starting evidence: `AUTODEBUG.md`, `real_full_33_diagnostic.log`. Real layer 5,
128 experts/top-8, BF16 weights, S=33 prefill passed PCC 0.99904094, while traced
decode failed at 0.98322794 (required 0.995). Attention PCC 0.999531 and shared
MLP norm PCC 0.999939 hid a routed norm PCC of 0.900598.

## Controlled experiments

Command (run from the repository root):

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_moe_decode
```

The probe reproduces seed 42 and the real checkpoint at the original dimensions.
It captures the TT residual and normalized expert input, and compares CPU and TT
components on identical tensors. CPU substitutions occur only in the probe.

| Control | Final PCC | Interpretation |
| --- | ---: | --- |
| Original TT decode | 0.98322794 | Original failure reproduced |
| CPU router on TT residual, TT experts | 0.98455066 | Router implementation alone does not account for upstream drift |
| CPU router and CPU experts on TT tensors | 0.98462944 | Expert computation is not the material error |
| Original HF route IDs/weights, TT experts | 0.99987250 | Expert selection is causal |
| HiFi4 expert matmuls | 0.98345997 | Refuted; retained no expert fidelity change |

Baseline expert output on identical inputs and routes is PCC 0.99966240; HiFi4
worsens it to 0.99905369. Original HF routes include experts 21 and 40; the TT
router replaces 40 with 7. The exact CPU router on the TT residual replaces 21
with 7 instead. TT residual PCC is 0.99988984, but these discrete replacements
change the routed sum substantially. Rounding the original HF residual to BF16
keeps its correct route set.

Attention projection controls:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_moe_decode --attention-fidelity hifi4
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_moe_decode --attention-fidelity hifi4_fp32
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_moe_decode --attention-fidelity hifi2_fp32 --projection qkv
```

HiFi4 without FP32 accumulation worsens residual accuracy. HiFi4 with FP32
accumulation restores the correct CPU route set, but the BF16 router still
changes an expert. Narrowing to the original HiFi2 fidelity and FP32 accumulation
on QKV alone also restores the CPU route set. Neither logits-top-k alone nor
FP32 logits after BF16 normalization fixes the remaining router error. FP32
router normalization, scaling, projection and ranking agrees with the same-input
CPU route set and produces final PCC 0.99987093. Preserving HF's softmax-before-topk
order also passes, at 0.99985056; that smaller semantic change is retained.

Evidence: `decode_moe_probe*.log` and `.json`. Two intermediate runs failed only
while querying unavailable MeshDevice memory statistics after numerical results;
the final QKV-only control completed successfully.

## Retained implementation

`tt/routing_precision.py` and setup wiring in `tt/functional_decoder.py`:

- QKV retains BF16 weights/output and HiFi2 math fidelity, with FP32 destination
  accumulation enabled and packer L1 accumulation disabled. O projection,
  shared MLP, sparse experts, and BF16 cache remain unchanged.
- Router retains HF normalization/scaling/softmax/top-k/sum-normalize order,
  using FP32 internal tensors and HiFi4/FP32 accumulation. Only selected routing
  values are stored as BF16 for sparse expert matmul.
- The QKV callable adapter uses the imported attention dispatch interface; it
  does not modify global functions or source outside this autoport.

The focused diagnostic temporarily wraps TTNN functions to run isolated A/B
controls; production forward code does not monkeypatch any operations.

## Trace recovery

The first implementation used `zeros_like(fp32_scores, dtype=bfloat16)`, which
falls back to host creation in `creation.cpp:full_like_impl` and rejects capture.
It was changed to device `typecast` followed by same-dtype `zeros_like`.
The failed capture stalled device close. Captured `trace_write_triage*` evidence
showed no running operation and healthy Ethernet; terminated only that runner,
ran bounded `tt-smi -r`, verified all four p300c chips with `tt-smi -ls --local`,
then passed a 1x1 open/close smoke. Evidence is in `trace_write_reset.log`,
`trace_write_health.log`, and `trace_write_smoke.log`.

## Final verification

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder --layer 5 --length 33 --real --decode --steps 4 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/real_full_33_fixed.json
```

Pass: prefill PCC **0.99886105**, first traced decode PCC **0.99985056**, four
changed-input/position replay steps (positions 33–36) minimum PCC **0.99985056**,
repeated replay bit-equal, runtime prefill audit clean. Capture runs inside the
harness's device-only guard. Log: `real_full_33_fixed.log`. All thresholds remain
0.995. Device closed successfully, and the main agent took hardware ownership
for broader sliding/full, padding, cache, batch and context checks.

This is a Python-only change; no C++ build is required. New Python files were
formatted with Black for Python 3.10 and syntax checked. The original failure is
fixed with a focused traced regression. Broader shape/context coverage belongs
to the stage validation matrix; this report makes no performance claim.

## Boundary follow-up

The wider real-weight sliding request-reuse matrix exposed three more failing
decode inputs (S31, S1025, S2049). See `AUTODEBUG_boundaries.md` for the fresh
source-only hypothesis report. The initial full-attention fix remains verified,
but does not establish this wider contract.

First focused control:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_boundary_decode
```

Exact seed19 S31 input reproduces eager PCC **0.98488777**, identical to the
recorded reused trace. Cache K/V and current query PCC exceed **0.999979**.
The TT router agrees with CPU routing on the TT residual, both replacing HF
expert 85 with 26. Original HF routing substitution passes **0.99947729**.
CPU SDPA using the exact current TT Q/K/V and the same mask restores expert 85
and final PCC **0.99938515**. TT SDPA versus that same-QKV CPU result is
**0.99983367**: a small attention error causes a discrete expert replacement.
HF BF16 execution still chooses the original route and matches FP32 HF at
**0.99996594**, so reference dtype is not the cause.

The imported decode SDPA defaults in `sdpa_decode.cpp` are HiFi2, approximate
math, and FP32 destination accumulation disabled. Independent controls showed:

| S31 control | Final PCC | Result |
| --- | ---: | --- |
| FP32 destination accumulation only | 0.98534928 | Refuted |
| Disable approximate math only | 0.99953902 | Verified |

At S1025, exact SDPA math alone still fails **0.99227678**. Original routing
substitution repairs it (**0.99961601**), while oracle output projection,
oracle SDPA+output projection, and individual oracle Q/K/V substitutions do not.
Oracle Q/K/V together pass **0.99982491**. Raising only QKV fidelity from HiFi2 to
HiFi4 with its existing FP32 accumulation passes **0.99975158**; S2049 with the
same policy passes **0.99916394**. HF BF16 controls on both inputs pass above
**0.99995**, preserving the original route sets.

Commands add `--sdpa-compute hifi2_exact --qkv-fidelity hifi4` and
`--request-index 5` / `6` to `tests.probe_boundary_decode`. Evidence is in
`boundary_probe_*` logs/JSON. CPU SDPA on exact TT Q/K/V still changes the route
at S1025 under this policy: numerical errors interact around a close ranking
boundary. The verification matrix, rather than a general precision claim, is
the acceptance evidence.

The follow-up policy raises QKV fidelity to HiFi4 and disables approximate math
for decode SDPA. The local `tt/decode_attention.py` single-request implementation
reuses existing head transforms and preserves BF16 cache/output and SDPA
accumulation. It delegates prefill to the existing attention instance. The
functional decoder's fixed slot loop supplies one request at a time.

The eight-request sliding matrix then passes every prefill and traced decode:
minimum prefill PCC **0.99588285**, minimum decode PCC **0.99799301**. Former
failures S31/S1025/S2049 pass at **0.99962003 / 0.99975158 / 0.99916394**.
Evidence: `reuse_sliding_fixed.json`, `.log`. The same policy also passes the
full-attention reuse matrix, but regresses the separate original seed42 full-S33
decode to **0.98433118**. Therefore global HiFi4 QKV was rejected.

One-variable regression control:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_policy --qkv-fidelity hifi2 --sdpa-compute exact --layer 5 --length 33 --real --decode --steps 4 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/full33_hifi2_exact.json
```

Restoring only QKV HiFi2 for the full-attention layer, retaining exact SDPA,
passes all four trace steps at minimum PCC **0.99983047**, with identical repeated
replay. The resulting QKV policy is HiFi4 for sliding attention and HiFi2 for
full attention, with FP32 accumulation in both; SDPA uses exact HiFi2 in both.
This is selected by the existing layer-type configuration, independent of
inputs, positions, routing IDs, or test seeds.

## Broad numerical validation exposed additional causes

The layer-dependent QKV policy above is **not sufficient for acceptance**. Full
attention request reuse still fails the first S33 at 0.9922984
(`reuse_full_final.json`). Batch32 S33 has five failing sliding slots
(5,15,17,25,27), four failing full slots (9,11,21,29), while every prefill passes,
trace auditing is clean, and repeated replay is identical. See
`batch32_sliding_final.json` and `batch32_full_final.json`.

CPU controls supplied by the parent investigation establish that FP32 HF
computation with only BF16 cache storage passes all 64 batch slots with unchanged
routes. BF16 RoPE tables plus BF16 cache also pass all 64. Thus neither is an
intrinsic accuracy limit for this test. Diagnostic precision changes remain in
`tests/probe_policy.py` and `tests/probe_fp32_attention.py` until a complete
passing policy is demonstrated; they are not claimed as a retained repair.

Focused batch32 sliding controls, each compared against its named predecessor:

| Diagnostic policy | Failing decode slots | Evidence |
| --- | --- | --- |
| FP32 input norm/all phases, QKV/head storage, device paged FP32 attention | 17,25 | `batch32_sliding_front_attn_fp32.json` |
| Above + FP32 O projection, post-attention norm and residual | 17 (0.98250818) | `batch32_sliding_attention_residual_fp32_retry.json` |
| Above + exact HF BF16 prefix cache | 17 (0.98151085) | `batch32_sliding_oracle_prefix.json` |
| Residual policy + FP32 RoPE table | 8,17 | `batch32_sliding_rope_fp32_gather.json` |
| Residual policy + FP32 attention output storage | 8,17 | `batch32_sliding_output_fp32.json` |
| Residual policy + explicit exact/FP32 head RMSNorm | 5,17,27 | `batch32_sliding_exact_norm_retry.json` |

The first residual experiment (`batch32_sliding_attention_residual_fp32.json`)
is **invalid**: a diagnostic closure reused the post-attention norm weight for
input normalization. Distinct closure variables correct the probe; only the
`_retry` artifact is valid. The first exact head-norm run also has no numerical
result because diagnostic reshape used padded volume; the corrected retry uses
logical dimensions.

Current localization after the corrected residual policy: TT and CPU router on
the exact same TT residual both select expert108 instead of HF expert76 for
slot17 (residual PCC 0.99995220). This independently reconfirms upstream residual
drift. Explicit head-norm accumulation raises slot17 residual PCC to 0.99998522
but still crosses the route boundary. At slot8, CPU routing on the residual
changes expert5 to41 while TT routing retains5, showing why a single passing
seed or a fidelity compensation is insufficient evidence.

## Final causal findings and retained repair

The detailed same-input controls distinguish four separate boundaries:

1. Native RoPE narrows products even with FP32 input/output, and the small decode
   core group ignores its requested compute configuration. With exact head norms,
   failing sliding cases had QKV PCC 0.99999976–0.99999988 but post-RoPE Q PCC
   only 0.99999440–0.99999523. Device SFPU `q*cos + rotate_half(q)*sin`, retaining
   the supplied BF16 table values, repairs the previously failing 5/17/27 slots.
   Source evidence: `AUTODEBUG_fp32.md`; runtime:
   `batch32_sliding_detailed_stages.json` and `batch32_sliding_elementwise_rope*`.
2. Top-k after FP32 softmax can reorder close logits. Sliding slot25 raw scores
   correctly select expert11, while rounded probabilities select15; slot28
   similarly changes45 to31. Select logits first, then softmax the selected
   logits. This is algebraically identical to full softmax followed by top-k
   sum-normalization. Evidence: `batch32_sliding_router_detail.json`.
3. Native paged update repacking of an FP32 update corrupts the intended BF16
   rounding. Even exact HF current QKV plus exact BF16 prefix cache leave current
   V PCC 0.99999577 against the intended BF16 value (maximum error 0.03125).
   Q after RoPE on the same BF16 tables is meanwhile PCC
   0.999999999999997. Explicit device BF16 conversion before paged update repairs
   every sliding slot: minimum final PCC 0.99968503 in that control. Evidence:
   `batch32_sliding_oracle_qkv_prefix_routes.json`, `batch32_sliding_cast_cache*`.
4. Full slot11 retains the correct expert72 in CPU routing on the exact TT
   residual, but TT's FP32 projection selects111. On identical scaled inputs,
   score error reaches 0.000918. BF16 high/low decomposition reduces error to
   0.000538 but still flips the rank. The small decode router instead computes
   128 FP32 elementwise dot products and SFPU sums, avoiding FPU Src narrowing.
   All full slots then pass, minimum PCC 0.99982832. Evidence:
   `batch32_full_cast_cache_routes_retry.json`, `batch32_full_split_router*`,
   `batch32_full_sfpu_router*`.

The retained implementation preserves BF16 checkpoint weights, paged cache,
RoPE table values, shared/routed MLP compute and final output. It uses FP32
input/head/post-attention normalization, QKV/head storage, RoPE arithmetic,
attention output projection, attention residual and routing arithmetic. Decode
attention gathers a device-selected logical page range, computes FP32 logits,
softmax and weighted values, and uses the original caller's page table and
positions. Sliding decode selects only the required logical window plus boundary
page. The embedding primitive currently untilizes the underlying cache; no
performance improvement is claimed.

Prefill must generate accurate K/V too: reverting the entire prefill front end
produces four failing sliding slots (`batch32_sliding_decode_only.json`). Local
TP1 prefill therefore uses the same accurate head construction and native BF16
prefill SDPA. Sliding chunks prepend the previous window as K/V history and
matching disposable Q rows; full chunks reuse the paged chunked SDPA helper.
All production changes are local classes/functions under this autoport; no
module-global monkeypatch or CPU oracle is retained.

High/low BF16 decomposition of QKV and attention matmuls was removed after both
batch32 matrices passed without it (`batch32_*_no_splits.json`). FP32 RoPE table
storage and expert HiFi4 changes were rejected. The production code is in
`tt/precision_ops.py`, `tt/precise_attention.py`, `tt/decode_attention.py`,
`tt/routing_precision.py`, wired by `tt/functional_decoder.py`.

Integrated direct-harness validation (not the diagnostic policy driver):

| Test | Minimum prefill PCC | Minimum decode PCC | Status |
| --- | ---: | ---: | --- |
| Sliding real batch32 S33 | 0.99765820 | 0.99964134 | All32 pass; trace repeat equal, audit clean |
| Full real batch32 S33 | 0.99785365 | 0.99982261 | All32 pass; trace repeat equal, audit clean |
| Sliding nine-request reuse | 0.99569189 | 0.99976075 | All pass; audit clean, trace reused |
| Full nine-request reuse | 0.99922349 | 0.99982880 | All pass; audit clean, trace reused |

Commands, from repository root:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.batched --batch 32 --layer 0 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/batch32_sliding_integrated.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.batched --batch 32 --layer 5 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/batch32_full_integrated.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.request_reuse --layer 0 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/reuse_sliding_integrated.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.request_reuse --layer 5 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/reuse_full_integrated.json
```

Python compilation and Black formatting pass for the changed implementation and
probe files. No C++ source changed; a build was not required. All hardware runs
were serialized and closed their mesh device in `finally`.


The original seed42 full-layer S33 reproducer also passes after integration:
prefill PCC **0.99943957**, four traced decode steps minimum **0.99987781**,
with clean audits and identical repeated replay. The first warm K/V cache rows
at position33 are **bitwise equal** to their supplied device BF16 updates.
Evidence: `full33_integrated.json`, `cache_write_integrated.json` and
`full33_integrated.log`.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_cache_write --cache-report models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/cache_write_integrated.json --layer 5 --length 33 --real --decode --steps 4 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/full33_integrated.json
```

Hardware lease returned to the parent after this check; no further device work
is performed by this subtask. Parent validation is responsible for the final
stage-wide readiness conclusion. The decoder's current position contract states
valid nonnegative positions, and precise attention currently uses the stage's
32-token cache pages.


## 128-step decode follow-up: QKV Src precision

The first integrated sliding 4096-token/128-step run exposed one additional
failure at absolute position 4110, PCC **0.99494137**. Prefill passed 0.99852634;
trace auditing and repeated replay were clean. TT routing and CPU routing on
its exact residual both replace HF expert 76 with 64. The attention residual PCC
is 0.99998076. CPU exact SDPA and O projection on the same TT Q/K/V also select 64,
localizing the material discrepancy before that boundary.

An independent FP32 RoPE-table control repairs 4110 but introduces 4149
(PCC 0.99328420). It is rejected. Exact-input CPU controls confirm that BF16
cache with FP32 RoPE itself changes 4149's expert 109 to 49 (PCC 0.99340531), while
BF16 cache plus the original BF16 RoPE tables passes all 128 steps, minimum
PCC 0.9990941. Exact saved device inputs match the CPU reconstruction bitwise;
there is no RNG mismatch. See `long_decode_hf_precision_control.json`.

The retained additional repair affects only S1 QKV projection. It computes
FP32 SFPU row dots in groups of 256 output rows, avoiding FPU Src narrowing and
bounding the temporary product. Prefill retains its existing matmul, and the
supplied BF16 RoPE/cache values are unchanged. Setup holds an FP32 copy of the original BF16 weight values, arranged with
one output projection per row.

At 4110, a paired device comparison against CPU linear on the exact same
normalized input measures maximum QKV error **0.04061413** for the prior FPU
projection versus **0.0000319481** for the SFPU result (approximately 1271×
smaller). The correct expert 76 is restored and final PCC rises to 0.99987218.
The complete sliding 128-step run passes at minimum 0.99904322; full attention's
128-step run also passes at minimum 0.99977388. This is a measured accuracy
repair; no performance improvement is claimed.

Latest integrated regression artifacts:

| Test | Minimum decode PCC | Artifact |
| --- | ---: | --- |
| Sliding 4096/128 | 0.99904322 | `headline_sliding_exact_qkv_integrated.json` |
| Full 4096/128 | 0.99977388 | `headline_full_exact_qkv_integrated.json` |
| Sliding batch 32 | 0.99968658 | `batch32_sliding_exact_qkv_integrated.json` |
| Full batch 32 | 0.99982852 | `batch32_full_exact_qkv_integrated.json` |
| Sliding 9-request reuse | 0.99978697 | `reuse_sliding_exact_qkv_integrated.json` |
| Full 9-request reuse | 0.99982091 | `reuse_full_exact_qkv_integrated.json` |

Paired same-input evidence is in
`headline_sliding_exact_qkv_integrated_stages.json`; the baseline and rejected
RoPE controls are `headline_sliding_stages.json` and
`headline_sliding_fp32rope_stages.json`. All normal-harness checks retain
threshold 0.995, real 128-expert/top-8 weights, device-only trace auditing and
identical repeated replay. The additional production change is confined to
`QKVLinear` in `tt/routing_precision.py`.

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_sequence_decode --inspect-position 4110 --stage-report models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/headline_sliding_exact_qkv_integrated_stages.json --layer 0 --length 4096 --real --decode --steps 128 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/headline_sliding_exact_qkv_integrated.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder --layer 5 --length 4096 --real --decode --steps 128 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/headline_full_exact_qkv_integrated.json
```

All six latest integrated regression runs completed with exit status zero and
closed their devices before hardware was returned to the parent. Python
compilation and Black formatting also pass for the additional source/probe.

## Maximum-context full attention: corrupt identity gather

The subsequent full-attention length 262144 gate passes sampled prefill
PCC 0.99627766 but fails traced decode at positions 262143/262142 with
PCC 0.10160415/0.11483141. This is a cache addressing failure, not another
precision policy adjustment. See `AUTODEBUG_long_context.md` for the source
checks and per-operation localization.

`probe_long_attention.py` proves the first incorrect operation is the
row-major page-table gather: its 8192 identity indices are bitwise exact but
the selected physical IDs are corrupt. Replacing only that identity gather
with the input table gives bitwise-correct cache indices and K/V contents,
and maximum-width attention PCC **0.99999976096** against CPU. A bounded
no-cache probe confirms native gather succeeds through 1920 indices and
fails at 1921, 2048, 8192, matching the row-major multi-core selector boundary.

The retained production change is confined to `tt/precise_attention.py`:
full attention directly uses the caller's page table because it reads all
logical pages in table order. Sliding attention and all numerical settings
are unchanged. No C++ changes or context reduction are involved.

| Verification | Result | Evidence |
| --- | --- | --- |
| Maximum-shape attention A/B | PCC 0.99999976096; K/V bitwise exact | `long_attention_identity_gather.json` |
| Native gather boundary | 128/1920 exact; 1921/2048/8192 corrupt; direct table always exact | `page_gather_boundary.json` |
| Integrated real full 4096/128 | Minimum decode PCC 0.9997738820 | `headline_full_identity_gather_integrated.json` |
| Integrated real full batch 32 | Every slot passes; minimum decode PCC 0.9998285179 | `batch32_full_identity_gather_integrated.json` |

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.probe_page_gather --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/page_gather_boundary.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder --layer 5 --length 4096 --real --decode --steps 128 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/headline_full_identity_gather_integrated.json
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.batched --layer 5 --batch 32 --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/batch32_full_identity_gather_integrated.json
```

These commands exited zero and closed their devices. Runtime audits pass
and repeated traced outputs are identical in the two model checks. Device
lease was released to the parent at 19:37 UTC for full 262144/262143 reruns and
the remaining stage gates. Full-layer maximum-context verification is still
pending in this subtask report; the component result alone is not a stage pass.
