# Full-model qualitative verdict

Date: 2026-07-20

The six shared readiness prompts were rendered as base-model completion prompts because the exact Falcon3-7B-Base tokenizer snapshot has no chat template. HF and TT both used the same rendered text, tokenizer, greedy policy, and 100-token budget. The machine checker reports no critical degeneration. Human review passes the suite with one advisory limitation: the TT haiku repeats a coherent stanza, while the other five TT completions remain coherent and task-relevant.

| Prompt | First divergence | Human verdict |
| --- | ---: | --- |
| Machine-learning haiku | 1 | Advisory pass. The stanza is fluent, grammatical, on topic, and contains no prompt/control leakage or language drift, but repeats mechanically through the 100-token budget. HF also continues beyond the requested haiku into other verse forms, so neither base-model greedy continuation stops cleanly; TT's repetition is the weaker behavior. |
| Supervised vs. unsupervised | 2 | Pass. Correctly distinguishes labeled from unlabeled data and continues with a coherent explanation. Some concepts are restated, but there is no token doubling, collapse, language drift, or leakage. |
| Inventor story | 2 | Pass. Diverges early from HF's quantum-computer story but remains a coherent English story about a young inventor, fermentation, and a garden. No repetition or request leakage is visible. |
| Thermodynamics | 13 | Pass. Correctly states and explains the zeroth law and starts the first law. It closely tracks the HF control before a harmless wording divergence and remains coherent. |
| English to French | 3 | Pass. Produces a correct French translation, then follows the same multilingual continuation pattern as HF. The later Spanish/German/Japanese/Chinese text is base-completion continuation rather than wrong-language drift in the requested translation. |
| Fibonacci Python | none | Strong pass. TT and HF match for all 100 generated tokens and produce the same syntactically coherent function. |

Across the suite there is no token doubling, special/control-token leakage,
prompt echo, cross-request leakage, or early-divergence quality collapse.

The haiku received a focused control rather than a dismissal. Host-eager
greedy, traced free-running greedy, traced greedy after synchronized trace
release/reset/recapture, and traced teacher-forcing on the host-greedy prefix
produce the same 100 tokens with no divergence. Both traced free runs finish
with current and rotary positions 107, direct sampled-token feedback, and zero
token H2D copies. This refutes split-greedy semantics, trace replay, feedback,
position advancement, reset, page-table handling, and cache contamination as
causes of the loop. It is the inherited BFP4/BFP8/LoFi model's greedy result.

A supplemental seeded on-device top-k 8, top-p 0.9, temperature 0.7 run
produces a coherent, non-repetitive haiku and related English explanation. It
demonstrates the supported stochastic path but does not replace or conceal the
weaker required greedy artifact. The haiku loop therefore remains a documented
qualitative limitation. The other prompts are unrelated and coherent,
including a 100-token exact HF match on Fibonacci.

Artifacts:

- `results/qualitative_suite/qualitative_prompt_format.json`
- `results/qualitative_suite/prompt_verdict_inputs.json`
- `results/qualitative_suite/prompt_*/{rendered_prompt,hf_completion,tt_completion}.txt`
- `results/qualitative_suite/prompt_*/autoregressive_meta.json`
- `results/qualitative_suite/degenerate_output.json`
- `results/full_model_qualitative_control.json`
- `results/haiku_host_greedy.txt`
- `results/haiku_traced_greedy.txt`
- `results/haiku_seeded_sampled.txt`
