# Qualitative verdict

Final default all64-layer outputs preserve every recorded stage06 token for the
shared short suite and all three long cases (`qualitative_comparison.json`). The
current run uses the checkpoint chat template, device greedy sampling and16-step
EOS-aware collector windows. Model/revision, tokenizer, EOS configuration and
reused byte-identical HF controls are identified in qualitative_prompt_format.json
and hf_control_provenance.json.

The AIME100-token continuation begins “We need solve math problem” and restates
Aya's walk/coffee timings coherently, as does HF. It is a bounded continuation,
not a claim of solving the whole problem. Prefill and teacher forcing each score
99/100 top1,100/100 top5,100/100 top100 reference token positions.

`structured` returns the correct `{"alice_apples":4,"bob_apples":3}` and EOS.
`explain` discusses sunlight and scattering in English; `code` gives a seen-set
order-preserving implementation. Explanation and code truncate at the128-token
cap, as their HF controls do. Reasoning text is emitted by this checkpoint's chat
mode; there is no new wrong-language, repeated-subword or cross-request behavior.

The original repeated-text4096 fixture remains coherent but spends887 tokens in
reasoning and reaches the1024 cap before finishing its final prose. This is exactly
the prior stage's investigated trajectory; the HF control starts final prose much
earlier. Stage07 does not claim HF parity or label that cap a completed answer.
The nearby4095 fixture completes five coherent paragraphs and emits EOS at token892.
The distinct4096 factual case returns the three exact requested JSON facts and EOS
at token47. Its default16-step collector schedules49 tokens, versus the earlier
128-window policy's possible129; clean paired latency measurements are separate
in benchmark_b1_perf.json:eos_window_control.

Long cases preserve trace IDs/capture counts and original prefix identity.
`default_long_quality_cases.json` contains the actual rendered-input hashes,
outputs and diagnostic metrics. Automated degeneracy checks supplement this direct
reading; they do not establish quality for untested prompts or dataset-wide accuracy.
