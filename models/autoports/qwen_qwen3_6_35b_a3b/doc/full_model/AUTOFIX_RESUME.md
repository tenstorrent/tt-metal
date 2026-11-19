# AutoFix: retained-state resumed-prefill numerical gate

Assessed 2026-11-14T14:09:39+00:00 under the AutoFix skill, following [AUTODEBUG_RESUME.md](AUTODEBUG_RESUME.md) and the completed accepted-decoder parity experiment. This assessment used source/JSON inspection only. It opened no device, reran no HF/model workload and changed no production implementation.

**Historical source scope:** source descriptions and line references below refer to that assessment's tree. Subsequent formatting/import-order edits and the separate singleton K/V lifetime repair changed source bytes. The recorded parity is not a rerun of those later files. [Source provenance audit](source_provenance_audit.md) records the comparison; [AUTOFIX_CONTEXT.md](AUTOFIX_CONTEXT.md) documents the later allocation repair and its distinct acceptance evidence. No numerical result or resumed-prefill gate has been relabeled.

**Outcome: AutoFix has not produced a verified policy-preserving repair. The all40 resumed-prefill gate remains FAIL and the full-model stage remains incomplete.** The new reduced parity test refutes the proposed bounded-executor/accepted-decoder mismatch on its measured inputs. It does not convert the failed acceptance into a pass. No remaining concrete source-supported repair intervention is selected from the available evidence. Further numerical localization is possible; this report does not claim that every potential defect has been excluded or that satisfying the gate is mathematically impossible.

## Failing contract

The user requires genuine retained-state continuation, preserving the optimized multichip decoder, cache dtype split, activation/fidelity/CCL policy and inter-layer layout. Recomputing the entire prefix, selecting a slower fallback, changing precision or rounding policy to force agreement, or relaxing the checker is not an acceptable repair.

The installed checker compares each implementation's uninterrupted prefill with prefix-plus-suffix execution. Its unchanged gates are mean PCC >=0.99, minimum per-position PCC >=0.98, top1 agreement >=0.98, and equal final-position top1. It tests seven cases at context2048 on all40 layers. Ten negative-control comparisons fail as expected; all seven top1/final-token gates pass. Three numerical gates fail:

| Total length / start / suffix | TT mean PCC | TT minimum PCC | Required failing gate |
|---|---:|---:|---|
| 257 / 128 / 129 | 0.9935122896 | 0.8585700580 | Minimum >=0.98 |
| 1025 / 260 / 765 | 0.9931221526 | 0.9686653286 | Minimum >=0.98 |
| 1025 / 1024 / 1 | 0.9870205088 | 0.9870205088 | Mean >=0.99 |

The raw failure remains [readiness/prefill_resume_check.failed.json](../../readiness/prefill_resume_check.failed.json). These are eager prefill comparisons: the artifact records zero model/sampling trace replays and unchanged program-cache entries9602. Decode-trace performance and the prior mixed-batch rounding repair are separate issues.

## Diagnosis checked against source at the recorded assessment time

The wrapper retains the same cache between prefix and suffix (`tt/generator.py:683–694`); only `start==0` resets convolution/recurrent state (`:398–401`). Tokens use the absolute `[start:end]` interval and selected logits undergo one relative-index conversion. Absolute positions and page windows match full-attention fill/decode calls. Bounded transport writes valid rows, excluding padding. AutoDebug found no contradictory reset, indexing, ownership or completion transition.

The recurrent consumption boundary in `tt/prefill.py:127–130` exactly matches the accepted `tt/multichip_decoder.py:1544–1545`: BF16 rounding followed by FP32 backing. Singleton dispatch and projection K are also inherited (`prefill.py:132–142`, `multichip_decoder.py:1547–1556`). A one-row tail of a longer call uses chunk recurrence/K1024; a public singleton uses recurrent-step/K512. The singleton's gating path differs as well. Removing these distinctions would change accepted arithmetic rather than fix an isolated port inconsistency.

The pinned HF model has analogous stored-state rounding and a cached-singleton kernel switch. Its matched raw-book BF16 control fails all three case-level numerical gates. HF mean/minimum PCC are0.98702738/0.49634269, 0.99161743/0.95285638 and0.98846549/0.98846549 respectively. This establishes a known numerical continuation sensitivity in the original model. It does not prove identical TT/HF failing rows, a common root cause, or permission to waive TT acceptance. The control explicitly records one independent deterministic measurement per case, reused under ten seed labels; it is not ten independent trials or a generated-text quality verdict.

## Discriminating experiment and interpretation

The completed experiment directly addressed the concrete repair hypothesis: **bounded full-model prefill mishandles the accepted decoder's continuation state, row layout or terminal boundary.** It used exact HF-control token IDs, real layers0/3, a1x4 mesh, context4096, independent retained caches, and suffixes128:129,260:765,1024:1. Each path prefills its own prefix once and then consumes only the suffix. This is genuine continuation, not prefix recomputation inside the candidate.

The first diagnostic reference failed before producing a comparison because the legacy list-of-attention-results retained too many L1 tensors. The recorded error is a static-CB overlap: live L1 address887488 versus CB end888384. Production bounded prefill already drains each result into its fixed residual workspace. A **test-only** reference wrapper immediately moved each legacy attention result to DRAM, then restored the original method in `finally` (`tests/run_prefill_equivalence_probe.py:348–365`). That changes reference storage lifetime, not decoder arithmetic, dtype, weights or policy. It is not a production repair, a relaxed numerical gate, or evidence that the production path had the same allocation failure.

With that explicit reference drain and `--require-exact`, [resume_legacy_parity_drain.json](prefill_equivalence/resume_legacy_parity_drain.json) reports PASS, no failures, and `all_compared_tensors_exact=true`. Independent JSON inspection verified all78 metric records: finite, exact equality, PCC1, maximum error0 and zero unequal elements. These include both layer outputs and conv/recurrent/K/V on all four chips for all three suffixes, full vocabulary logits, and final-row versus callback logits. All six recorded source hashes matched the inspected files at the original assessment time. The later provenance audit finds two still identical (`tt/model.py`, `tt/slot_state.py`) and four changed; this historical pass does not establish byte identity or fresh parity for the later tree.

The generic `top1` fields below1 in two exact-logit comparisons are not token divergence: that helper compares `argmax(candidate)` against the first element of `topk(reference)`, whose tie ordering can differ. The explicit `greedy_token_agreement` is1, mismatch lists are empty, and tensors are exactly equal. This observation does not alter the acceptance checker or its gates.

Limits are material: this experiment covers two layers, context4096 and reduced-stack activations. The failed check uses all40 layers/context2048. It compares **bounded versus accepted continuation**, not uninterrupted versus resumed all40 computation. Exact reduced parity does not prove every layer/value/backend allocation correct, nor explain the three all40 PCC failures.

## Hypotheses and disposition

| Hypothesis / proposed intervention | Observation | Disposition |
|---|---|---|
| Lost/reset prefix, incorrect positions, page mapping or output slice | Assessment-time source is coherent; corrupted-history controls fail; selected accepted-decoder parity is exact | No demonstrated wrapper defect to repair; reduced cases refute this cause within their scope |
| Bounded padding, singleton layout or terminal conversion changes accepted suffix output | Three measured continuations have exact outputs, states and logits on every chip | Refuted for the measured two-layer cases; no additional layout patch selected |
| Skip the recurrent BF16 consumption boundary | Both accepted decoder and HF cache semantics contain the boundary | Would change required policy; not a defect-backed fix |
| Force singleton K1024/chunk recurrence, or make full prefill use singleton execution everywhere | Existing dispatch is inherited and also appears in passing cases; HF switches paths too | Numerical mechanism, not an isolated inconsistent calculation; no production intervention selected |
| Disable prefill stochastic rounding or change grouped/indexed MoE selection | Selected prefill keeps its policy; the proven deterministic batched-decode repair is already separate | No same-operand resumed-prefill defect established; no global rounding or routing change |
| Legacy diagnostic L1 overlap explains production numerical failure | Error exists only in the undrained diagnostic reference; drained reference and production bounded path match exactly | Test-reference lifetime repaired only; does not repair or explain the all40 gate |
| HF also fails, therefore accept TT | HF counterpart fails the same cases, but TT raw gate is explicit | Record limitation; gate remains FAIL |

The earlier suggested K512/K1024, recurrent/chunk and grouped/indexed same-operand comparisons would quantify differences between selected numerical branches. After exact accepted-decoder parity, they still lack a demonstrated caller/callee inconsistency that a policy-preserving production change would correct. They are not represented here as completed tests, proven fixes, or mandatory speculative sweeps.

## Commands and recorded results

The following are the exact recorded Python argv, rendered as executable commands. They are evidence of completed parent-run experiments, not instructions issued or runs performed by this auditor. Environment setup used the lane environment; the parity run pins tracking enabled and watcher/profiler disabled in its artifact.

All40 acceptance — FAIL, three PCC cases above:

```bash
python /home/ttuser/qwen36-35b-a3b/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/tests/run_full_model_resume_check.py --mode resume
```

Pinned HF control — completed; all three corresponding HF cases fail; `tt_gate_waived=false`:

```bash
python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_resume_control/run_control.py
```

Initial accepted-decoder parity — ERROR in undrained legacy L1 reference, no numerical acceptance conclusion:

```bash
python /home/ttuser/qwen36-35b-a3b/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/tests/run_prefill_equivalence_probe.py --section full --tail-lengths '' --prompt-lengths '' --continuations 128:129,260:765,1024:1 --token-fixture models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_resume_control/prompt_tokens.json --require-exact --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/prefill_equivalence/resume_legacy_parity.json
```

Drained-reference parity — PASS with exact equality:

```bash
python /home/ttuser/qwen36-35b-a3b/tt-metal/models/autoports/qwen_qwen3_6_35b_a3b/tests/run_prefill_equivalence_probe.py --section full --tail-lengths '' --prompt-lengths '' --continuations 128:129,260:765,1024:1 --token-fixture models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_resume_control/prompt_tokens.json --drain-legacy-attention --require-exact --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/prefill_equivalence/resume_legacy_parity_drain.json
```

This assessment independently checked the JSON status/metrics, source hashes, token-fixture correspondence and reference-wrapper implementation. No additional hardware command was run. The only file changed by this assessment is this report; parent test-harness changes are already pinned by the parity artifact.

## Remaining uncertainty and unresolved outcome

There is no observed earliest all40 inconsistent operation and no complete TT per-position/intermediate archive. TT aggregate metrics cannot identify whether its worst rows match HF positions135/300; singleton position1024 is unambiguous. Reduced parity does not eliminate a value-dependent or later-layer defect, a context-geometry-specific backend defect, or an error shared by both accepted and bounded paths. The entire native binary/dependency chain is not pinned by every historical artifact.

A further **diagnostic**, if pursued, can preserve policy: for the original1025/start1024 case, record the requested final row after each real layer and the linear state's bytes immediately before the final-token update, keeping the actual continuous and resumed branches unchanged. Compare only after accounting for the intentional BF16 public-call boundary. An earliest unexplained mismatch could then motivate a same-operand accepted-operation/native-kernel check. This would supply new localization evidence; it is not a currently supported repair experiment, a guaranteed solution or a measured result. The reference comparison must retain full depth and original context/tokens; a reduced pass cannot substitute for acceptance.

AutoFix therefore **failed to restore the required resumed-prefill passing contract in this attempt**. No justified minimal production patch has been identified from AutoDebug plus the new discriminating control. Preserve the failed artifact and policy, record the HF-known limitation, and keep this gate and the full-model stage incomplete. Remaining performance/context/review requirements are not passed by this report.

## Evidence hashes

Paths are relative to the model directory. These hashes pin the original assessment inputs; source hashes are also recorded inside the artifacts. The JSON evidence remains unchanged. The `preserved_policy.md` hash is its historical assessment snapshot: that living document has since gained boundary/provenance documentation and is not asserted to retain this hash.

| Artifact | SHA256 |
|---|---|
| `readiness/prefill_resume_check.failed.json` | `cce222c5dd38c329839310732c038a41d835e39238e5210539509f0cf102ed62` |
| `doc/full_model/qualitative/hf_resume_control/result.json` | `fd3f6a7f10f899d6359b30da776dcd491bc94eda3c65c69548223811356832ae` |
| `doc/full_model/prefill_equivalence/resume_legacy_parity.json` | `f3ef63be8da14c89ea63236090c6d608d7a3bf06d3cf92efc84050e2085712c3` |
| `doc/full_model/prefill_equivalence/resume_legacy_parity_drain.json` | `7256aa9a678138c0e79d8a22d16ff6f3e0c38c15a262635f24c8478a0c5c8e66` |
| `doc/full_model/AUTODEBUG_RESUME.md` | `8983a3e372a364cb320ffa80c8091a66585efefd5525e9af5b6272f8e232318d` |
| `doc/full_model/preserved_policy.md` | `26f9d17454547a76a1c560079ce6cde24b973f7bb9a43944f54683717ba59801` |
