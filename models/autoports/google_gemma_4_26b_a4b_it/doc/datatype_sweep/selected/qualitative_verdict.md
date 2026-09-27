# Selected-policy qualitative check

Verdict: pass for this 128-token shared-suite smoke. All six TT completions,
their HF controls, the refreshed baseline controls and the separate sky
explanation were read. Exact token equality with baseline is not expected after
changing precision and is not claimed: all six continuations differ. No visible
mechanical repetition, doubled subwords, wrong-language behavior, prompt echo,
cross-request text or gibberish was observed.

`qualitative_tt.json` includes the exact rendered chat prompts, message roles,
prompt tokens, HF outputs and revision, tokenizer class, chat-template presence,
greedy generation settings and TT outputs. The model is instruct/chat; these are
not raw-completion quality verdicts. Every buffered 128-token output preserves
the corresponding EOS-stopping streaming prefix.

| Prompt | Selected output evidence | Comparison and verdict |
| --- | --- | --- |
| shared_0 | “Data flows through nodes” / “Learning how to see.” | Coherent three-line ML poem; HF and baseline supply different coherent poems. |
| shared_1 | Fruit-label teacher analogy, “This is an apple.” | Correct supervised-learning setup, like both controls; cut at the 128-token budget. |
| shared_2 | Elara discovers a brass clockwork heart | Coherent requested story continuation; controls choose a compass and Elian. Creative divergence is expected. |
| shared_3 | “Energy cannot be created or destroyed” | Correct first-law explanation, as in both controls; the long requested explanation is budget-truncated in all paths. |
| shared_4 | “Bonjour, comment allez-vous aujourd'hui ?” | Valid French formal/informal variants; same task and language as HF/baseline. |
| shared_5 | `def fibonacci_sequence(n):` | Coherent iterative-function introduction; selected and controls are truncated inside code at the same fixed output budget. This smoke does not establish executable completed code. |

The stored `<turn|>` marker in short completions is the tokenizer's normal turn
terminator, also present in HF and baseline because artifacts decode with
`skip_special_tokens=False`; it is not generated text corruption. Long outputs
stop at the explicit128-token limit, with the same limitation in controls.

The separate sky explanation correctly introduces the white-light spectrum
before its128-token cutoff, as the HF control does. The existing degenerate-output
checker exits0 with no findings (`degeneracy.json`, `degeneracy.log`), zero
adjacent duplication and0.0316 trigram-loop fraction. Low free-running exact
token agreement with HF is informational, not a teacher-forcing accuracy gate.
Full-model teacher forcing independently passes94%/100%/100%.
