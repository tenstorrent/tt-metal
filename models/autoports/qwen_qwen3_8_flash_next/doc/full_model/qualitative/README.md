# Matched qualitative controls

The paired review is complete within the frozen rubric: all six prompts have
HF **0/10** and TT **0/10** blind-scored sampled failures. The independent judge
read 120 label-hidden examples; the owner froze its ratings hash before joining
labels. `verdict.json` records the bound hashes, per-prompt rates, concrete
reasons and source scope. The actual judge used a fresh `fork_turns="none"`
agent with inherited model/reasoning defaults, not an explicit xhigh override.
The rule remains strictly more than 20 percentage points worse per prompt.

The owner also read all six greedy pairs and both AIME completions for coherence,
repetition, language drift and early divergence (`greedy_review.json` and
`autoregressive_review.json`). Both AIME outputs remain coherent after wording
diverges at generated token 6. Shared examples include “learning with a teacher,”
the requested French “Bonjour, comment allez-vous aujourd’hui ?”, and the start
of `def fibonacci(n):`. Final mathematical accuracy, complete factual coverage
and executable code are unassessed at the 128-token cap. Both backends have
45/60 sampled and 5/6 greedy outputs reach that cap; checkpoint thinking and
truncation are shared control behavior excluded by the frozen rubric.

The 66 HF controls were reused only after exact cache-key checks, with the same
checkpoint, tokenizer, rendered prompts, seeds and generation settings. Their
original generation-call time was 7,600.67 seconds. Both HF and TT now have 16
EOS-terminated outputs and 50 cap-limited outputs across the 66-request suite.
EOS normalization preserves raw tokens and excludes post-EOS text from judging.

TT evidence is all48 B1 from `../acceptance_text_v1.json`, generator hash
`6b38b367a6fca1b9e212561a7c6e4883964b047883b6faaad7885a726249dc84`, before B>1
factored integration and formatting. It includes the shared workspace and
device token feedback; the B1 semantic path remains unchanged. It does not prove
B32/factored behavior, current top-k accuracy, performance or maximum context.
The process completed all saved outputs but exited 1 after device teardown
because the final reporter passed `phase` twice. The raw report is retained;
`../acceptance_text_v1_process_status.json` records the failure and reporter-only
fix, validated by eight CPU tests. Generation was not rerun, and this review
does not relabel that process as a successful exit or a completed model stage.

`run_controls.py` uses the six shared prompts from the installed
`readiness_check/vllm_prompts.txt`. It preserves their user roles and renders
exact HF tokenizer chat-template prompts with default thinking enabled. The
sampled controls use the model card's thinking recipe: temperature 1.0, top-k 20,
top-p 0.95, presence penalty 0, repetition penalty 1. Seeds 0–9 and a 128-token
cap are shared by HF and TT. One greedy continuation per prompt accompanies the
sampled controls. The 128-token cap may end inside thinking; absence of EOS or a
final answer due only to that cap is not a quality failure.

Stopping comes from the pinned HF `generation_config.json`, whose EOS IDs are
`[248046, 248044]`. Both sides' logical output ends at the first such token,
including that EOS. TT may physically produce its fixed 128-token canvas;
`raw_generated_token_ids` preserves it, while the completion and judged token
sequence exclude every post-EOS token. Every row records its EOS source, first
EOS position, discarded suffix length, and whether it reached the token cap.
The blind-packet builder rejects missing or inconsistent stopping metadata.
Existing exact HF generations can acquire this metadata without regeneration:
run `run_controls.py --normalize-existing` after generation finishes. This does
not change generation parameters or bypass exact reference-cache matching.

`rubric.json` freezes the failure rule before generation: compare labels-hidden
samples, and treat TT as worse only when its failure rate exceeds HF's by more
than 20 percentage points over the ten matched seeds for the same prompt.
HF-only results cannot establish a comparative verdict.

Historical CPU control command (no TT device access); the completed controls
were reused for this verdict, not regenerated:

```bash
source python_env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 timeout 10800 \
  python -u models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative/run_controls.py \
  > models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative/hf_run.log 2>&1
```

The original full48 HF model is retained across calls, with exact BF16 mmap
weights and exact selected PLE rows. Every generation starts with a fresh HF
request cache. Sample cache reuse requires the unmodified `ReferenceCache` key
match for exact model/revision/tokenizer/rendered tokens/library versions and
settings including seed, dtype, attention, token limit, and sampling recipe.
Completed calls persist incrementally to `hf_outputs.json` and the exact-match
cache, so a timeout can resume safely.

Before the six-prompt suite, the script calls the installed
`run_autoregressive._cached_hf_greedy` for the AIME prompt and 128 tokens.
`hf_reference/prompt.txt` is rendered using the HF template. Tokenization follows
the installed runner exactly, including its `.strip()` before encoding, which
may differ from the teacher-forcing `.refpt` prompt's final newline. The `.refpt`
is never substituted into the greedy cache. `aime_autoregressive_hf.json` stores
these exact tokens and completion; its cache entry is owned by the installed
runner and can be reused by the full `run_autoregressive` invocation with this
prompt path and `--max-new-tokens 128`.
Its `seconds` field measures this cache invocation and must not be reported as
uncached HF generation performance. The AIME cache was populated before the
shared suite; the successful suite run reused it exactly.

A device-owning full-model probe can import this module and call:

```python
from models.autoports.qwen_qwen3_8_flash_next.doc.full_model.qualitative.run_controls import run_tt
run_tt(generator, host_sampling=False)
```

This uses the already-open generator and never opens a second mesh. It supplies
exact prepared prompt token IDs and explicit sampling seeds. If the generator
cannot support a seed on the selected device-sampling path, it raises: this is
work for the stage. `host_sampling=True` is available only as separately labeled
compatibility evidence (`tt_host_sampling_outputs.json`); it does not establish
the optimized token-out sampling path.

The completed paired-packet preparation used this interface:

```bash
python models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative/blind_compare.py \
  --tt models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative/tt_outputs.json
```

Give an independent judge only `blind_packet.json`, `rubric.json`, and
`blind_ratings_template.json`. Do not give it
`blind_labels_DO_NOT_GIVE_TO_JUDGE.json`, backend output filenames, or owner
conclusions. Freeze its per-example boolean ratings, concrete reasons and
snippets before joining the labels:

The ratings template is an object containing `packet_sha256`, `rubric_sha256`,
and a `ratings` list. Preserve both hashes and fill the entries inside that
list. The comparator rejects ratings for a different packet or rubric, and
verifies that the packet and hidden labels have not changed. Preparing the same
packet again preserves the existing template; preparing different evidence
requires archiving the prior blind artifacts first. Prepare only after both
HF and TT outputs are complete.

```bash
python models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative/blind_compare.py \
  --ratings models/autoports/qwen_qwen3_8_flash_next/doc/full_model/qualitative/blind_ratings.json
```

The comparator requires all ten matched seeds per prompt and side. Record any
HF-matched limitation in the full-model work log, and in eval limitations when
applicable. Mechanical token feedback defects remain required work; a matched
sampling-rate comparison does not waive a proven stale-token/position defect.
