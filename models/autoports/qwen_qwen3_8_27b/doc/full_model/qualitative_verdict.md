# Qualitative review

Stage verdict: clean-pass with the limitations below. The repaired default path passes AIME and shared bounded checks, escapes the original copying loop, and passes the independent long factual check. Paired HF controls and final-default reproduction are complete; clean performance and independent stage review are complete (`stage_review_final.md`). This is not an AIME solved-problem accuracy result. Controls use the exact HF checkpoint revision and normal tokenizer chat template. Three shared prompts are fixed in ../qualitative_prompts.json; their rendered prompts, token IDs and raw outputs are in qualitative/. No packaged shared-suite asset exists in the installed plugin, so this repo-local suite is retained for subsequent stages.

- AIME24 prompt0,100generated tokens: HF says “Let's parse carefully.”; TT says “Let's parse.”. First token divergence is index15. Thereafter both coherently restate the same walk/coffee problem, without wrong-language drift, mechanical repetition or corrupt feedback. The raw15/100 positional agreement is shifted by the omitted word, not evidence that85% of the continuation is semantically wrong. Teacher forcing is99%top1 and100%top5/top100. Both100-token outputs stop before solving the problem. Sources: autoregressive/hf_completion.txt, tt_completion.txt, autoregressive_meta.json and teacher_forcing_accuracy.json.
- explain,128-token cap: both explain blue light scattering in clear English. Both spend tokens planning the requested three sentences and truncate the final answer at the budget. TT reaches more of the final explanation; the HF control is also incomplete. No repetition loop or language drift. Sources: qualitative/explain_hf.json and explain_tt.json.
- structured: both compute Alice4/Bob3 and produce the requested JSON. The first TT run continued beyond im_end, a real wrapper stopping bug. The fixed run stops at token114, including EOS248046; no later template text is returned. Both checkpoint EOS IDs248044/248046 are recognized. Sources: qualitative/structured_hf.json, structured_tt.json and full_validation_04.log; failed behavior remains in full_validation_01.log.
- code,128-token cap: both give an order-preserving seen-set implementation. TT produces a complete function and begins an example; HF also provides the function and an unfinished example within its reasoning. Both examples truncate at the fixed budget. No doubled tokens, gibberish or cross-request content. Sources: qualitative/code_hf.json and code_tt.json.

`check_degenerate_output.py --hf-model Qwen/Qwen3.8-27B --missing-artifacts critical --scope autoregressive` passed. Its adjacent duplication rate is0; artifact stage_checks.log (the recognized AIME completion only). Human inspection above includes the shared suite in addition to that mechanical AIME check.

Fixed-length throughput probes explicitly set stop_on_eos=False to measure128 token-out steps. They are not used as qualitative verdicts. Normal high-level generation stops output at EOS. In a mixed batch, already-finished rows remain physical device slots until the other rows finish; no per-token host mask or token-feedback reconstruction is introduced. External schedulers use low-level active-slot controls.

## Long-fixture failure and narrow repair

`before_geometry/benchmark_b1_perf.json:throughput_fixture_qualitative_control` contains512 TT
tokens with normal EOS stopping enabled. TT spends the entire budget repeating
the supplied context inside its opening quotation and never reaches an answer.
`benchmark_hf_control_512.json` uses the exact same4096 input token IDs (SHA256
ec5606540969599d781d96ad892b0cded3f605d8f000669886cfaad9411899e2) and reaches coherent several-paragraph prose.
HF fixed-length mode does not explain the difference: its displayed output has no
EOS before the cap. B8/B32 fixed128-token outputs move beyond literal restatement;
this is a useful batch-dependent control, not a512-token quality pass. AutoDebug and AutoFix isolated the sensitivity to prefill arithmetic; the fixture
was not waived as throughput-only. See `AUTOFIX_repetition.md`.


The final default changes only native out-projection K grouping48→4, retaining
fusion, TP4, requested dtypes/fidelity, CCL, KV split and residual strategy.
`default_long_quality_cases.json` uses a fresh default-built all64 generator, with
no geometry override. `default_geometry_reproduction.json` proves every output
ID and completion string matches the read diagnostic candidate:

- Original4096: first512 tokens exactly match the candidate and already escape
  literal copying at55. At1024, reasoning has ended at887 and coherent English
  prose has begun; output is truncated without EOS. HF ends reasoning at214 and
  already writes prose within512. This substantial answer-delay limitation is
  retained; neither equal-budget quality parity nor full answer completion is claimed.
  Extending the budget verifies that the changed trajectory terminates reasoning;
  it is not the repair itself.
- Nearby4095, same thinking-enabled chat-template policy: TT ends reasoning at325,
  supplies a coherent five-paragraph answer and stops at EOS892. Paired HF ends
  reasoning at285 and writes coherent prose, truncated at512 without EOS.
- Independent4096 distinct records, explicitly thinking-disabled in both backends:
  TT returns strict JSON with R0101,1459,scheduled, allthree expected facts correct,
  and stops at EOS47. Paired HF returns the same facts in valid JSON and stops at EOS38; whitespace formatting differs.

Root read all completions. No mechanical copying loop, wrong-language drift or
cross-request content appears in these repaired outputs. They do not establish
broad task accuracy or eliminate sensitivity to quantization. The both-K4 and
out-K16 alternatives still reproduce the original loop and remain rejected.
Exact templates, token IDs and hashes are in `long_quality_cases.json`; paired
controls are `long_quality_cases_hf.json` and `long_quality_cases_tt_out.json`.
