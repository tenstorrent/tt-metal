# Supplemental qualitative anomaly audit

The frozen blind comparison passes with **0/60 TT and 0/60 HF failures**. This is a narrow repetition/gibberish/off-topic/wrong-language result. Direct inspection finds unsupported process remarks that must remain visible: **six selected TT samples, four previous optimized-full-model TT samples, and zero pinned HF samples**. The absence in HF is real; this audit does not claim HF exhibits the same call remarks. The [JSON audit](qualitative_anomaly_audit.json) preserves exact completions, identities, source paths and counts.

## Unsupported call remarks

| Artifact/backend | Prompt and seed | Exact excerpt |
|---|---|---|
| Selected TT | q02/6, blind sample0096 | “Need maybe call again? It said MUST call immediate. Already called with content.” |
| Selected TT | q04/0 | “Already calling.” |
| Selected TT | q04/2 | “We can call immediately.” |
| Selected TT | q04/9 | “We have already called.” |
| Selected TT | q05/4 | “We called.” |
| Selected TT | q06/4 | “We already called.” |
| Prior optimized TT | q02/0, q04/2, q04/9, q05/4 | “We called” / “We have called” in otherwise topical thinking |
| Pinned original HF | All six greedy and 60 sampled records inspected | No unsupported call remark found |

None of the six rendered prompts includes a tool, call instruction, previous call or verbosity setting. The strongest anomaly, selected q02/6, asserts an absent “MUST” instruction before a coherent supervised/unsupervised-learning explanation. This is an unsupported instruction/action reference; its origin is **unproven**. It is not established as cross-request leakage, nor established as harmless expected text. Prior TT occurrences establish that this class predates this selected policy; they do not prove the new policy cannot affect its frequency. Six versus four of60 is descriptive, not a causal or statistically established regression finding.

The source path supplies saved prompt token IDs afresh for each request, creates explicit sampled seeds0–9 and invokes traced device sampling. `generate` applies sampling parameters, calls `reset`, then prefills the supplied prompt. Reset invalidates checkpoints, clears convolution/recurrent/PLE/private attention K/V and history, resets positions, sampler outputs and seeds. Start-zero prefill resets again and replaces sampler prompt history from the current prompt. No concrete prompt concatenation or missing reset bug was found in this inspection. Source inspection alone does not prove per-request tensor state or exclude another runtime/sampler defect. Exact source locations and limits are in the JSON.

## Markers, verbosity and answer quality

Both selected TT and HF have42/60 sampled outputs containing `</think>`; selected TT has14 configured `<|im_end|>` endings and HF15. The raw chat template starts assistant thinking, and logical output retains the first configured EOS inclusively. No other angle-pipe control-token form appears in the sampled completions. These delimiters are explained serialization, distinct from the unsupported call statements.

Unsupported “desired oververbosity9” or equivalent references occur in four selected TT and four HF sampled outputs. That is direct both-backend evidence for verbosity meta commentary, **not** evidence that HF shares call remarks. Prompt formatting is byte-identical to the previous optimized validation and copied HF output is byte-identical to the original full-model control. System text specifies xhigh reasoning effort but no verbosity9 or call instruction.

The judge also found actual answer-quality nits beyond the frozen categories: HF q01/5 (`sample0016`) ends its haiku with the ordinarily four-syllable “Minds born from code”; TT q04/1 (`sample0012`) says entropy “always increases over time” without the equilibrium/nondecreasing qualification; TT q04/8 (`sample0100`) reasons confusedly about the zeroth versus first/second/third law list. These remain limitations, not general accuracy passes. Many stories/functions/answers end at the shared128-token cap. Cap-limited incompleteness is excluded by the frozen rubric; neither runnable code nor complete long answers were validated by this suite.

## Disposition under the cap

Propose carrying the unsupported process-text cause and broader subject-matter checks as **nonblocking, outside-frozen-rubric follow-up**, subject to the stage owner's independent review. The frozen comparison passes and call remarks remain on-topic and occur in prior TT evidence. This is not a root-cause resolution or permission to label them normal HF behavior. Preserve especially q02/6 for later fresh-process and reordered-request controls with exact histories/logits/sampler state and matched original-HF scoring. No new experiment, test, device access, runtime edit or scorer change was performed; this CPU-only audit leaves all frozen ratings and outputs intact.
