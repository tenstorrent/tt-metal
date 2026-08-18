# Stage Review

Verdict: clean-pass

## Required Work
- None.

## Other Concerns
- The large context probes are capability/shape/runtime probes, not HF-vs-TTNN PCC runs at 262143/262144 tokens. I am not making this required work because the stage also records HF-vs-TTNN PCC above 0.995 for both layer kinds, batch 1 and batch 2 synthetic cases, and real-weight seq 1 and seq 5 cases in `logs/correctness_full.log:46-131`; the long probes then separately prove the public paths accept the advertised and near-advertised logical lengths in `logs/context_probe_full_prefill_262144_sparse.log:45-64`, `logs/context_probe_full_prefill_262143_sparse.log:45-64`, `logs/context_probe_linear_prefill_262144_chunked.log:45-64`, and `logs/context_probe_linear_prefill_262143_chunked.log:45-64`.
- Full-attention advertised prefill at exactly 262144 was recorded before the final `tests/test_functional_decoder.py` mtime, while `functional_decoder.py` was not modified after that run. The post-edit 262143 full-attention run, 262143 linear run, real-weight rerun, full correctness rerun, and fallback audit are current. I do not classify this as stale enough to block because the implementation under review predates the exact-262144 log and the current test body still invokes the same public path.
- `doc/functional_decoder/logs/py_compile.log` is empty. That is weak as a standalone syntax-check artifact, but later pytest imports and executes `tt/functional_decoder.py`, `tests/test_functional_decoder.py`, and `tests/conftest.py`, so syntax validity is independently covered by `logs/correctness_full.log:13-247` and `logs/runtime_fallback_audit.log:56-82`.
- The live worktree includes an untracked `tt_metal/third_party/tt-cluster-descriptors/` directory outside the autoport. I found no functional-decoder evidence depending on it and no optimized/full-model/vLLM files under the autoport, so this is a checkpoint hygiene concern rather than a functional-decoder blocker. The final stage commit should exclude or separately account for that outside-autoport state.

## Hard-Check Gaps
- No HF-PCC check is recorded at the longest 262143/262144-token contexts. Existing evidence satisfies the stage gate by combining smaller real/synthetic HF parity with long shape/capability probes, but a future robustness pass should add at least one reduced-output long-context numerical control if the harness can do it without impractical host memory.
- Full-attention batch-2 decode uses a reversed page table but identical `current_pos` values across users. This exercises per-user page-table rows, but it does not test different current positions per batch item. This is useful future coverage, not a blocker for this stage contract.
- The runtime fallback audit is source-AST based plus a fallback-exception run on the synthetic representative prefill/decode cases. It does not run the real-weight cases under fallback exceptions, but the audited runtime functions are shared and the synthetic run exercises both layer kinds' measured runtime paths.

## Anomaly Ledger
- Observed anomaly: Historical token-stepped linear-attention prefill timed out at the advertised 262144-token context.
  Evidence: `AUTODEBUG_linear_context.md:14-19` records three pre-fix 262144 timeout probes and healthy `tt-smi` evidence; `AUTOFIX.md:17-25` records replacement with the 64-token chunked TTNN gated-delta implementation; current logs pass at both `262144` and `262143` in `context_probe_linear_prefill_262144_chunked.log:45-64` and `context_probe_linear_prefill_262143_chunked.log:45-64`.
  Affected path: Historical `_QwenLinearAttention.prefill_forward`; current `_QwenLinearAttention.prefill_forward` in `tt/functional_decoder.py:763-796`.
  Control or comparison: Current linear advertised and near-max chunked probes pass, and small synthetic/real HF parity still passes after the chunked change.
  Likely subsystem: Linear gated-delta prefill dispatch scaling.
  Investigation performed: Read `AUTODEBUG_linear_context.md`, `AUTOFIX_linear_chunked_design.md`, `AUTOFIX.md`, current implementation, and current context logs.
  Resolution: fixed.

- Observed anomaly: An earlier trace-PCC harness attempted capture before warmup and hit TTNN trace write/read guards.
  Evidence: `work_log.md:257-275` records `TT_FATAL: Writes are not supported during trace capture` and `TT_FATAL: Reads are not supported during trace capture`; the current helper warms before capture in `tests/test_functional_decoder.py:342-351`; traced decode HF parity and trace controls pass in `logs/correctness_full.log:46-85` and `logs/correctness_full.log:106-146`.
  Affected path: Test trace lifecycle, not the final measured decode path.
  Control or comparison: Watcher traced run passes in `logs/watcher_correctness.log:129-164`; triage summary has all checks passing in `triage/trace_capture_unwarmed/triage-summary.txt`.
  Likely subsystem: TTNN trace capture setup.
  Investigation performed: Read work log, trace helper code, correctness log, watcher log, and trace triage summary.
  Resolution: controlled.

- Observed anomaly: Long advertised-context tests validate execution and shape, while PCC evidence is concentrated at short target-shape cases.
  Evidence: Long context tests assert output shapes in `tests/test_functional_decoder.py:785-830`; HF parity tests assert PCC in `tests/test_functional_decoder.py:354-463` and real-weight parameterization includes seq 1 and seq 5 in `tests/test_functional_decoder.py:588-608`; PCC values in `logs/correctness_full.log:46-131` all exceed 0.995.
  Affected path: Public `FunctionalDecoder.prefill_forward` and `decode_forward` context capability claims.
  Control or comparison: The context contract records no capability reduction and lists exercised lengths `[1, 5, 33, 65, 1025, 262143, 262144]` in `doc/context_contract.json:73-181`; all cited artifacts exist.
  Likely subsystem: Context-length evidence strategy.
  Investigation performed: Read tests, context contract, README/work log, correctness log, and individual long context logs; validated artifact references with a small read-only Python script.
  Resolution: controlled.

- Observed anomaly: Live worktree has untracked state outside `models/autoports/qwen_qwen3_6_35b_a3b`.
  Evidence: `git status --short --untracked-files=all models/autoports/qwen_qwen3_6_35b_a3b tt_metal/third_party/tt-cluster-descriptors` reports the autoport tree plus `tt_metal/third_party/tt-cluster-descriptors/`; that directory's files have timestamps around 2026-08-18 15:13, earlier than the final functional-decoder artifacts.
  Affected path: Repository checkpoint/scope hygiene, not the decoder implementation path.
  Control or comparison: Search under the autoport found only functional-decoder code/tests/docs/logs and no optimized-decoder, multichip, full-model, vLLM, generator, serving, datatype, or release implementation files.
  Likely subsystem: Workspace dependency/submodule state.
  Investigation performed: Ran read-only `git status`, `find`, and `rg` over the autoport and the untracked third-party path.
  Resolution: controlled.

## Scope Inspected
- Goal/skill paths: user-provided functional-decoder stage contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/functional-decoder/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: `models/autoports/qwen_qwen3_6_35b_a3b/doc/context_contract.json`; `doc/functional_decoder/README.md`; `doc/functional_decoder/work_log.md`; `doc/functional_decoder/runtime_fallback_audit.md`; `doc/functional_decoder/AUTOFIX.md`; `doc/functional_decoder/AUTODEBUG_linear_context.md`; `doc/functional_decoder/AUTOFIX_linear_chunked_design.md`; `doc/functional_decoder/logs/correctness_full.log`; `doc/functional_decoder/logs/real_weight_multitoken_moe.log`; `doc/functional_decoder/logs/runtime_fallback_audit.log`; `doc/functional_decoder/logs/watcher_correctness.log`; `doc/functional_decoder/watcher/final/generated/watcher/watcher.log`; `doc/functional_decoder/logs/context_probe_full_prefill_262144_sparse.log`; `doc/functional_decoder/logs/context_probe_full_prefill_262143_sparse.log`; `doc/functional_decoder/logs/context_probe_linear_prefill_262144_chunked.log`; `doc/functional_decoder/logs/context_probe_linear_prefill_262143_chunked.log`; `doc/functional_decoder/logs/context_probe_traced_decode_advertised.log`; `doc/functional_decoder/logs/py_compile.log`; `doc/functional_decoder/logs/tracy_perf_summary.log`; `doc/functional_decoder/tracy/{linear_attention,full_attention}/*_ops.csv`; `doc/functional_decoder/tracy/{linear_attention,full_attention}/*_perf_report.{txt,csv,console.log}`; raw Tracy CSVs under `doc/functional_decoder/tracy/raw/reports/2026_08_18_22_31_32/`; trace triage logs under `doc/functional_decoder/triage/trace_capture_unwarmed/`.
- Code paths: `models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py`; `models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py`; `models/autoports/qwen_qwen3_6_35b_a3b/tests/conftest.py`; local HF reference source `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`.
- Commands run: read-only `wc -l`, `sed`, `nl -ba`, `ls -l`, `stat`, `find`, `git branch --show-current`, `git status --short`, `rg`, `head`, `tail`, `cat`, and small read-only Python scripts to inspect HF source locations, validate `context_contract.json` artifact references, and sum `tt-perf-report` CSV `Device Time`. No pytest, TT hardware command, reset, server, vLLM run, or implementation-file modification was performed.

## Residual Risk
- This review did not rerun hardware tests by instruction; it relies on the provided artifacts and read-only artifact analysis.
- The perf evidence is intentionally functional-stage scale: warmed prefill and warmed traced decode for representative short sequence lengths, not optimized or production throughput evidence.
- Full-model interactions across all 40 layers, multichip behavior, generator semantics, and vLLM serving are outside this functional-decoder stage and remain for later bringup stages.
