# AutoFix: haiku precision versus request state

Starting evidence: [AUTODEBUG_haiku.md](AUTODEBUG_haiku.md). Selected extended p0
ends with a 5/7/6 answer; pinned HF and prior Stage 7 outputs give 5/7/5. The
selected and prior TT generations first differ at generated index 31. This is
not a page or SDPA chunk boundary. No runtime defect is established.

## Controlled experiment

`tests/probe_datatype_qualitative.py` uses the normal constructor and one explicit
policy per fresh process. It reserves/asserts B1 cache capacity 1088, 34 pages,
and history capacity 1023, then runs two native full64 generations of the exact
HF extended p0 S60/G1024. Each normal `generate` resets request state. Request 0
captures traces; request 1 exercises their reuse. Policy/source/reference hashes,
actual allocation descriptors, device page tables, resolved SDPA dimensions,
trace counters, EOS-trimmed tokens/text, and deterministic equality are recorded.
The original HF prompt, budget, and shared suite artifacts remain untouched.

Run from the repository root through the existing serialized hardware launcher:

```bash
QWEN_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/configs/head_bfp4_lofi.json models/autoports/qwen_qwen3_8_27b/tests/run_datatype_experiment.sh haiku_selected_retry models.autoports.qwen_qwen3_8_27b.tests.probe_datatype_qualitative --host-greedy
QWEN_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/configs/baseline_bfp4_lofi_head_bfp8_hifi2.json models/autoports/qwen_qwen3_8_27b/tests/run_datatype_experiment.sh haiku_baseline models.autoports.qwen_qwen3_8_27b.tests.probe_datatype_qualitative
QWEN_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_27b/doc/datatype_sweep/configs/head_bfp4_hifi2.json models/autoports/qwen_qwen3_8_27b/tests/run_datatype_experiment.sh haiku_head_hifi2 models.autoports.qwen_qwen3_8_27b.tests.probe_datatype_qualitative
```

The launcher provides environment/source snapshots, command logging, and exit
status. Add `--host-greedy` to a fresh diagnostic invocation if sampler localization
is needed. It adds a third request using existing host full-logit argmax, keeps
allocation geometry fixed, and records comparison against native request 0.
This changes sampling and trace delivery together; a difference requires further
same-prefix localization and equal-max tie checks before blaming the sampler.
Diagnostic timings are not candidate performance evidence.

## Completed controls

All three commands above ran through the coordinator's serialized hardware
session, reached report status `complete`, and have `.exit_status` **0**. This
report author inspected their saved JSON/log/exit artifacts with host-only reads.

| Policy and artifact | Native repeat equality | Tokens including EOS; EOS index | Final syllables |
| --- | --- | --- | --- |
| [Selected BFP4/LoFi](haiku_selected_retry.json) | Exact | 347; 346 | 5/7/6 |
| [Baseline BFP8/HiFi2](haiku_baseline.json) | Exact | 418; 417 | 5/7/5 |
| [BFP4/HiFi2](haiku_head_hifi2.json) | Exact | 347; 346 | 5/7/6 |
| [Pinned HF p0](../full_model/hf_qualitative_extended.json) | Existing reference | 284; 283 | 5/7/5 |

Selected and BFP4/HiFi2 produce the **same entire 347-token completion**, with:

```text
Data flows in light                 2+1+1+1 = 5
Models learn from hidden signs      2+1+1+2+1 = 7
A new truth emerges                 1+1+1+3 = 6
```

The baseline retains the first two lines and ends `Clarity forms now`
(3+1+1 = 5). HF ends with `models learn from patterns deep` (2+1+1+2+1 = 7)
and `answers emerge now` (2+2+1 = 5), after the same first line. These are
conventional syllable counts of the actual final answers, not the model's own
reasoning claims. All completions reach EOS; the failures are not truncation.
The unchanged shared prompt is exactly `Write a haiku about machine learning.`
It does not explicitly demand the numbers 5/7/5. This report applies the strict
conventional 5/7/5 audit to the requested haiku form; the observed six-syllable
third line remains a real limitation under that audit, not an invented extra
constraint supplied to generation.

Selected exactly reproduces the saved Stage 8 extended p0 tokens, and baseline
exactly reproduces saved Stage 7 extended p0. Their first mismatch remains index
31 (selected token 6970, baseline token 30), whose decoder input absolute position
is 90: page 2/offset 26 and k64 chunk 1/offset 26, away from a boundary.

All three controls have identical recorded source/reference hashes on their ten
common paths and identical initial/final geometry dictionaries: full64, B1,
capacity 1088, 34 pages, device table `[[0, ..., 33]]`, history allocation 1023;
full-attention KV tensors `[34, 1, 32, 256]`, BFP8, tiled/interleaved DRAM;
local Q/KV heads 6/1, head dimension 256, page size 32, SDPA q32/k64/grid11x10.
Policies differ only in head weight dtype/fidelity and `config_id`. Actual head
and final-norm compute attributes are recorded; all use FP32 destination
accumulation, no approximate math, and packer accumulation; norm stays HiFi2.

Each first native request captures traces and each second request reuses them.
Both record one reset, 1023 model/sampling replays and history appends, and one
history readback per request. The selected third request uses CPU argmax over
1024 full-logit readbacks with host token feedback and produces the identical
347-token EOS-trimmed completion. Its allocated history remains 1023 even though
its delivery path does not use history (`last_perf.history_capacity` is 0 for
that host path). Equality claims cover saved tokens through first EOS; the helper
executes all 1024 generation steps but does not retain post-EOS tokens.

## Hypothesis ledger

- **H1, head precision — verified prompt-specific sensitivity.** Raising fidelity
  from LoFi to HiFi2 with BFP4 head weights changes no saved token and does not fix
  the haiku. At fixed HiFi2, changing head weights from BFP4 to BFP8 restores the
  baseline 5/7/5 answer. Thus head weight precision is a sufficient intervention
  at HiFi2 for this prompt, while fidelity alone is not. This does not establish
  that BFP8/LoFi also restores it; that fourth policy was not run. Nor does this
  prove a low-precision arithmetic defect or generalize to other prompts.
- **H2, request/capture state — unsupported for this reproduction.** Every native
  repeat is deterministic at identical geometry, and fresh selected/baseline
  streams exactly match their prior-stage artifacts despite the isolated request
  order. The proposed capture/reuse or retained-capacity explanation is not
  needed to explain the observed delta. This is not a universal trace-state proof.
- **H3, sampler/delivery — unsupported for this reproduction.** Selected native
  device sampling and CPU argmax agree exactly through EOS despite different
  delivery paths. No sampler patch is justified. Full same-prefix hidden/logit
  oracle substitution was not performed, so no source/kernel root cause is claimed.

## Status

The coordinator's first `haiku_selected` attempt completed native request 0 but
exited 1 during the probe's post-generation geometry collection: `page_host` was
`None`. The generator intentionally invalidates this optional host mirror when
using the bound device table. This was a diagnostic assumption error, not evidence
of a runtime page-table defect. The probe now compares the actual device-table
readback and omits the host memoization state. Original `haiku_selected` and
`haiku_selected_native0.json` artifacts are preserved; retry with experiment name
`haiku_selected_retry` so the corrected two-request result cannot overwrite them.
Syntax and Black checks below passed again after this correction.

No model/runtime edits or fix are proposed. Hardware work was serialized by the
coordinator; this author ran no TTNN imports or device operations. Host checks:
`python_env/bin/python -m py_compile models/autoports/qwen_qwen3_8_27b/tests/probe_datatype_qualitative.py`
and `python_env/bin/python -m black --check --target-version py312 models/autoports/qwen_qwen3_8_27b/tests/probe_datatype_qualitative.py`
passed. `isort --check-only` could not run because isort is not installed. This
Python/docs change requires no C++ build.

Final status: **observed precision-sensitive qualitative limitation; selected
haiku remains incorrect**. BFP8/HiFi2 corrects this specific output in the control;
the selected policy remains unchanged. No runtime defect was demonstrated, and
no runtime fix is justified by these experiments. These focused p0 controls do
not re-evaluate or supersede shared-suite story, code, or other prompt artifacts.
The user's selection rule remains the fastest evaluated configuration passing
the minimum top-1/top-5 accuracy gates. Existing candidate benchmarks rank the
BFP8/HiFi2 baseline below selected; the diagnostic measurements here do not
rerank them or establish a new qualitative selection gate. The numeric selection
rule and candidate performance ranking are unchanged;
the passing numeric gate must coexist with this explicitly recorded quality
limitation and the independent stage-review decision. Do not describe the
selected shared qualitative suite as wholly instruction-correct.
