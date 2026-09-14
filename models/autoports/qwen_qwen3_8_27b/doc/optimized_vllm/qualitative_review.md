# Native serving output review

All twelve shared completions and all eight extended completions were read in
full. They are coherent and on topic, with no mechanical repetition, unexpected
language drift, gibberish or cross-request contamination. All six shared greedy
texts and all eight extended token streams/texts match the completed Stage9
controls exactly. This is a serving-quality pass, with the same controlled
sampled-haiku meter limitation as Stage9.

The native full 64-layer B1 server uses the selected policy, context 262144, actual
`/v1/chat/completions`, async scheduling and device split sampling. Host
compatibility, profiler, Watcher and allocation tracking are unset. The pinned
checkpoint's chat template is applied with `add_generation_prompt=True`.
`qualitative_prompt_format.json` rechecks the current template and shared prompt
source hashes, rendered prompts and exact token IDs. Logical prompt lengths
are60,67,75,61,66,62; none is constrained to internal page/tile alignment.

| Prompt | Reviewed result |
| --- | --- |
|0, machine-learning haiku | Greedy extension476 tokens/EOS gives “Data flows in light / Models learn from hidden signs / A pattern is born”,5/7/5. Seed42 extension139 tokens/EOS gives “Data learns patterns / weights adjust, models emerge / answers from noise”,5/7/4. Both are token-identical to Stage9. |
|1, supervised versus unsupervised | Both complete shared answers correctly distinguish labeled prediction from unlabeled pattern/group discovery, with cat/dog and customer examples. |
|2, story | Greedy1174 tokens/EOS develops a regret-opening key and ends in reunion with the inventor's mother. Seed42 sampled392 tokens/EOS follows a silence-removing key through loss and restoration. Both complete, coherent and exactly match Stage9. The fresh unseeded shared story about a glowing seed/lantern is a coherent budget-limited prefix, not the same sample. |
|3, thermodynamics | Greedy268 and seed42 sampled246 tokens/EOS explain conservation of energy, isolated-system entropy and perfect-crystal absolute-zero limits. The work-by-system sign convention is stated. Both exactly match Stage9. |
|4, translation | Greedy and sampled final answer: “Bonjour, comment allez-vous aujourd'hui ?”. Requested French is not language drift. |
|5, Fibonacci | Greedy400 and seed42 sampled331 tokens/EOS provide correct first-n list functions; greedy also provides nth-number calculation. Restricted execution of inspected functions passes n=-1,0,1,2,8,10, including the nth function's negative-input rejection. |

Shared requests use256 output tokens, greedy temperature0 or sampled
temperature0.7/top-p0.9 with model-default top-k20 and no explicit seed. Sampled
wording therefore differs from Stage9's unseeded outputs and is not a token
regression test. Prompts0/2/3/5 need extensions to judge complete answers.
Extensions use1024 tokens, or2048 for the story, with sampled seed42/top-k20.
All eight finish with `stop`; each greedy extension retains the exact shared
prefix. Seed42 extensions are new reproducible controls, not continuations of
the unseeded samples. Exact requests, token IDs and counts are retained.

The template opens reasoning. Raw reasoning and `</think>` appear in the exact
Stage9 controls and earlier HF/selected-TT controls as well. No reasoning-content
parser is configured. The haiku's syllable exploration finishes in the extension;
it is not an endless decode loop. Its sampled final line remains four syllables,
and no claim is made to repair that selected-model instruction-quality error.
The exact same seeded tokens in `before/qualitative_extended.json` control this
anomaly directly. Earlier HF/selected-policy evidence is discussed in
`../vllm_integration/qualitative_review.md` and retained under `../datatype_sweep/`.

Artifacts: `vllm_qualitative_outputs.json`, `qualitative_extended.json`, their
runner logs, `native_quality_server.log`, `qualitative_prompt_format.json` and
`qualitative_control_comparison.json`. The packaged shared degeneracy gate exits0
with no findings; the same checker also passes all eight extensions. The native
server closes its mesh and its owning guard completes cleanup after SIGINT.
This small prompt suite is not a broad model-quality evaluation.
