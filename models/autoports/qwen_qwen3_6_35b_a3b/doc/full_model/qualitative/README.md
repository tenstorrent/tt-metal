# Shared qualitative HF controls

The six prompts come from the installed plugin's
`runtime/readiness_check/vllm_prompts.txt`, preserving each user message and
using the pinned HF tokenizer chat template with `add_generation_prompt=True`.
The default thinking mode is retained. This stage requests 128 output tokens;
the later serving runner normally requests 256. Compare TT to these controls
at 128 tokens, or rerun both sides with an equal changed limit.

Run from the repository root after activating `python_env` and sourcing
`/home/ttuser/qwen36-35b-a3b/lane-env.sh`:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 timeout 7200 python -u models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/run_hf_controls.py --mode greedy --max-new-tokens 128 > models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_greedy.log 2>&1
```

`hf_greedy_outputs.json` contains the exact revision, tokenizer, full rendered
prompts and token IDs, generation settings, output token IDs, text, timings,
and seed. One model instance serves all prompts; CPU threads are capped at 8.
A resumed invocation reuses only entries whose metadata and prompts match
exactly. `hf_greedy_verdict.json` is a reading of those actual outputs, not a
TT comparison or stage pass.

## Before investigating a qualitative failure

Prepare a concrete failure rubric **before** inspecting sampled outputs. The
baseline threshold is TT failure rate more than 20 percentage points above HF,
using at least ten matched seeds per prompt. A missing EOS at the token cap is
not a failure. Expected reasoning markers are allowed. A same-rate defect is
recorded as an HF model limitation instead of a TT-specific regression.

The sampled control runner is prepared but not run proactively. Example for a
failure on `q01` (omit `--prompt-id` for all six prompts):

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 timeout 14400 python -u models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/run_hf_controls.py --mode sampled --seeds 10 --prompt-id q01 --max-new-tokens 128 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/hf_sampled_q01.json
```

Use the pinned model card's general thinking recipe on both sides:
`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
presence_penalty=1.5, repetition_penalty=1.0`. Its source is the pinned
snapshot's `README.md:663`. Presence penalty applies once per generated token
ID already seen, excluding the prompt, using an HF logits processor. For a
non-thinking experiment use `--thinking false` on both sides: the model card
prescribes temperature 0.7 and top-p 0.80 with the other settings unchanged.
The precise-coding recipe is a distinct experiment (temperature 0.6,
presence penalty 0.0) and is not selected for this mixed general suite.

`blind_compare.py rubric-template` prints a rubric template. Fill its content
rule and save it before sampling. Produce TT output with the same JSON schema,
matching prompt IDs, rendered prompts, settings and seeds. Then:

```bash
python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/blind_compare.py prepare --hf hf_sampled_q01.json --tt tt_sampled_q01.json --rubric locked_rubric.json --output-dir blind_q01
```

Give an independent judge **only** `blind_cases.json` and `rubric.json`; keep
`blind_key.json` hidden. After the judge writes `scores.json` with `case_id`,
boolean `failed`, and `reason`, run:

```bash
python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/blind_compare.py reveal --scores scores.json --output-dir blind_q01
```

The script verifies at least ten matched seeds and aligned generation settings
before constructing a randomly ordered unlabeled packet. It reports each
prompt's HF/TT failure rates only after blind scores are fixed. These helpers
never import TTNN or open devices.
