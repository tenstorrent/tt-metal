# Native serving qualitative review

All twelve shared completions and all eight extended completions were read in
full. They are coherent, on topic, and free of mechanical repetition, gibberish,
unexpected language drift, and cross-request contamination. This is a serving
quality pass with a retained instruction-quality limitation: the sampled haiku
does not satisfy conventional meter. It is not a claim that every answer fully
satisfies every instruction.

The all-64-layer B1 server uses the selected precision, context 262144, async
decode and native split sampling; host compatibility and allocation tracking are
unset. `readiness_vllm/primary_run_config.json` records the exact configuration.
The shared runner sends all six user messages through `/v1/chat/completions`,
using the pinned checkpoint revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and its actual chat template.
`readiness_vllm/qualitative_prompt_format.json` contains the template hash,
rendered prompts, message roles and token IDs. Prompt lengths are 60, 67, 75, 61,
66 and 62 tokens; these are valid non-aligned serving requests.

The template opens a reasoning response. Raw reasoning and `</think>` are present
in both serving and HF/direct-TT controls, rather than being evidence of request
contamination. The endpoint is not configured with a reasoning-content parser.
Exploratory counting in the haiku reasoning terminates with extended budgets; it
is not a mechanical token loop.

| Prompt | Greedy verdict | Sampled verdict |
| --- | --- | --- |
| 0: machine-learning haiku | Extended 476 tokens, EOS. “Data flows in light / Models learn from hidden signs / A pattern is born” scans 5/7/5. | Seed42 extension, 139 tokens, EOS. “Data learns patterns / weights adjust, models emerge / answers from noise” scans 5/7/4: final line is short. |
| 1: supervised/unsupervised learning | Correct labeled-answer versus unlabeled-pattern distinction, with photo/customer examples. Complete in the shared 256-token budget. | Correct, simple explanation and examples; complete in the shared budget. |
| 2: story | Extended 1174 tokens, EOS. An inventor's key opens doors of regret and leads to reconciliation with her mother. Coherent ending, no unrelated request content. | Seed42 extension, 392 tokens, EOS. A key removes sounds; reversing it restores the kingdom and laughter. Consistent fairy-tale logic and a completed ending. |
| 3: thermodynamics | Extended 268 tokens, EOS. Correct first/second/third laws, work-by-system sign convention, isolated-system entropy, perfect-crystal limit. | Seed42 extension, 246 tokens, EOS. Same correct concepts in different wording. |
| 4: French translation | “Bonjour, comment allez-vous aujourd'hui ?” is correct and complete. | Same correct translation; requested French is not wrong-language drift. |
| 5: Fibonacci | Extended 400 tokens, EOS. Correct list and nth-number implementations. | Seed42 extension reaches EOS with a correct first-n-values function. |

The initial shared requests use temperature 0 and temperature 0.7/top-p 0.9,
256 output tokens, and no explicit seed. Server generation configuration supplies
top-k 20 for sampled requests. Prompts 0/2/3/5 reach that budget, so their prefixes
alone were not judged complete. Extensions use 1024 tokens, or 2048 for story;
all eight finish with `stop`. All four extended greedy texts preserve the exact
shared greedy prefix. The sampled extensions are new seed42 native controls
(temperature 0.7, top-k 20, top-p 0.9), not continuations of the initial unseeded
samples. Exact token counts, requests and responses are in
`readiness_vllm/qualitative_extended.json`.

Controls come from the completed datatype stage's `tt_qualitative.json`,
`tt_qualitative_extended.json`, and `tt_qualitative_story_2048.json`, which retain
matching rendered prompts, prompt IDs and HF/selected-TT texts. These ignored
artifacts were restored from historical stage8 object `3f48cac393a` after the
operator's artifact-history sanitization. Prompt IDs 3/4 match the initial
selected-TT text exactly after removing only its terminal EOS marker; 0/1/2/5
differ in wording or reasoning length. No token-identical claim is made for
those four. The scientific explanation, translation and code remain correct;
the creative stories differ while remaining coherent. HF also truncates some
initial 256-token outputs.

The prior selected-TT haiku ends “A new truth emerges” (six syllables); the HF
control satisfies meter. Earlier datatype-stage controlled experiments associate
that specific prior error with BFP4 head weights. Here the greedy answer satisfies
meter and the sampled answer does not. The comparable control establishes that
meter errors already exist at the selected policy; this small serving suite does
not isolate the cause of every sampled word choice. It does not establish a new
serving quality regression or justify changing the selected precision.

After inspecting the generated code, restricted host execution checked the greedy
and sampled list functions at n=-1/0/1/2/8/10. The greedy nth-number function was
checked at n=0/1/2/10 and rejects n=-1. All checks pass. The functions import
nothing; only inspected list append, range and ValueError calls were allowed.
`readiness_vllm/qualitative_control_comparison.json` records comparisons and code
results. The packaged degeneracy gate is recorded separately in the work log.

Exact artifacts: `readiness_vllm/vllm_qualitative_outputs.json`,
`final_qualitative_runner.log`, `qualitative_extended.json`,
`qualitative_extended.log`, `qualitative_prompt_format.json`, and
`qualitative_control_comparison.json`. The six-prompt suite and short controls
are limited evidence, not a broad quality benchmark.
