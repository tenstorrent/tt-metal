# AutoFix Report: Stage 07 qualitative LM-head drift

## Starting evidence

- Stage 06 control:
  `doc/full_model/qualitative/vllm_qualitative_outputs.json`.
- Rejected Stage 07 block-3 observation (preserved before final-default rerun):
  `doc/optimized_full_model/qualitative_block3_rejected/vllm_qualitative_outputs.json`.
- Prompt 5 changes at its first generated token from the Stage 06/HF
  Fibonacci continuation to a newline followed by corpus-style Python-question
  autocomplete. Prompts 0, 2, and 4 also take different continuations.
- The two `qualitative_prompt_format.json` files and the two
  `qualitative_rendered_prompts.json` files are byte-identical. The HF token
  outputs are identical across stages. This rules out a prompt, tokenizer, HF
  revision, chat-template, or generation-mode change.
- The only production math change between the stages is the tied LM-head
  placement/program family. Stage 06 used an interleaved auto-selected
  `MatmulMultiCoreReuseMultiCast1D` projection with `in0_block_w=2`. Stage 07
  uses split DRAM-sharded matmuls, currently eight input shards and
  `in0_block_w=3`. Both use BF16 weights, BF16 logits, HiFi2,
  `fp32_dest_acc_en=False`, and the same device softcap.

This investigation was source/artifact-only. It did not open TT devices or
change implementation files.

## Observation matrix

The first Stage 06/Stage 07 TT divergence and the number of equal TT tokens are:

| Prompt | First difference (zero based) | Equal positions | Interpretation |
|---|---:|---:|---|
| Haiku | 11, `evolve` -> `grow` | 46/64 | Same coherent template, alternate word |
| Supervised learning | none | 64/64 | Exact Stage 06/Stage 07/HF match |
| Story | 17, `Write` -> `Complete` | 19/64 | Same first sentence, alternate corpus continuation |
| Thermodynamics | none | 64/64 | Exact Stage 06/Stage 07/HF match |
| French | 1, `What` -> `Translate` | 1/64 | Stage 07 improves HF exact prefix from 1 to 10 tokens |
| Fibonacci | 0, ` The` -> newline | 2/64 | Stage 07 regresses immediately |

Across all six prompts, Stage 07 has 155/384 positional matches to HF versus
136/384 for Stage 06. The sum of per-prompt HF exact-prefix lengths improves
from 134 to 142. These are descriptive controls, not an accuracy gate, but they
show that the change is mixed rather than a systematic qualitative collapse.

The independent 100-token autoregressive artifacts use the same prompt and HF
tokens in both stages. Stage 06 and Stage 07 TT share their first two tokens and
then diverge at token 2; both remain coherent English. Autoregressive
trajectories compound the first changed greedy choice and therefore do not
localize the cause.

Readiness accuracy shifts from 91/100 to 90/100 top-1 for both prefill and
teacher forcing, while top-5 and top-100 remain 100/100. The logs contain only
aggregate counts, not TT candidate IDs, scores, or margins. This is consistent
with close-candidate changes, but does not prove them.

## Hypothesis experiments

### Hypothesis 1: TP or split vocabulary ordering is wrong

**Prediction:** a shard/split permutation should cause broad, repeatable token
remapping, low aligned logit agreement, and failures at vocabulary boundaries.

**Inspection:** for local split offset `s` and mesh column `d`, `tt/model.py`
constructs

```text
W[:, d * 65536 + s : d * 65536 + s + 8192]
```

for each `d`, concatenates the four pieces, and applies
`ShardTensorToMesh(dim=-1)`. Mesh column `d` therefore receives exactly its
piece. The projection iterates `s = 0, 8192, ..., 57344` and concatenates its
outputs along vocabulary, reconstructing
`W[:, d * 65536 : (d + 1) * 65536]`. Host composition and the custom greedy
sampler both apply the same mesh-column order and `d * 65536` global offset.
The common non-greedy sampler constructs the same offsets.

Existing real-target evidence also checks this boundary:

- synthetic sampler cases cover global IDs 0, 32767, 32768, 65535, 65536, and
  262143, including deterministic ties and traced replay;
- reduced real model logits assert custom device greedy equals composed host
  argmax;
- Stage 07 prompts 1 and 3 remain exact for all 64 tokens, including tokens
  from different TP vocabulary ranges;
- full-model prefill and teacher forcing retain 100/100 top-5 and top-100.

**Result:** no contradictory offset, concatenation, mapper, or sampler index was
found.

**Verdict:** refuted at high confidence for a systematic permutation/layout
bug. An isolated kernel defect still requires aligned-logit evidence to exclude
absolutely, but it is not supported by the current symptom matrix.

### Hypothesis 2: custom split-greedy chooses the wrong token

**Prediction:** device greedy should disagree with host argmax on the exact same
Stage 07 logits.

**Inspection:** the reduced real-model regression performs exactly that check
and passes. The custom sampler explicitly chooses the lower global token ID for
equal BF16 scores, and its synthetic boundary/tie tests pass. The qualitative
artifacts do not retain host logits for prompt 5, so the exact full-model token
has not yet been checked this way.

**Verdict:** refuted for the exercised real and synthetic shapes; low residual
uncertainty remains for the exact prompt-5 logits. The decisive A/B below
includes this comparison.

### Hypothesis 3: BF16 accumulation/rounding changes a close top candidate

**Prediction:** the legacy and optimized heads should retain high aligned-logit
agreement, while one or more close top candidates change order or become an
exact BF16/softcap tie. A full trajectory then diverges after the first changed
token. Changing only legal matmul accumulation geometry may change the first
choice without changing weights or vocabulary indices.

**Evidence:** the optimized program changes K blocking and input-shard geometry
while retaining BF16/HiFi2 and BF16 output. Device softcap is monotone before
finite-precision rounding, but BF16 tanh/output can create plateaus. The model
already has observed equal maxima in real BF16 softcapped logits, which is why
the sampler has an explicit lower-token tie rule. The mixed divergence
positions, two exact 64-token controls, improved French continuation, and
unchanged top-5/top-100 accuracy match this prediction.

**Result:** verified at the LM-head accumulation-geometry intervention
boundary. A legal DRAM-sharded candidate changed only the terminal matmul from
eight input shards/block 3 to four input shards/block 2 while retaining the
same tied BF16 weights, BF16 logits, HiFi2, softcap, decoder hidden state
producer, sampler, prompt, and generator. Its Fibonacci output matches Stage
06 exactly for all 64 generated tokens; the eight-shard/block-3 default differs
at token 0. The candidate's first HF difference is again token 1 with 1/64
positional matches, exactly the Stage 06 result.

This verifies accumulation geometry as the cause and intervention boundary. It
does not by itself distinguish an exact post-softcap tie from a small
pre-softcap score-order swap; the aligned-logit artifact remains the direct
numeric record for that narrower question.

**Verdict:** verified.

### Hypothesis 4: request reset, trace lifetime, or cross-prompt state changes prompt 5

**Prediction:** prompt 5 run alone should differ from prompt 5 run after the
first five prompts, or repeated isolated runs should be nondeterministic.

**Inspection:** `generate()` begins with `reset()`, which releases both traces,
zeros dirty KV cache, synchronizes that clear, and resets counters before the
next prefill. The warning about allocation with an active trace is the already
documented second split-trace capture warning; sampler/model persistent buffers
are prewarmed and allocated before capture. Prompts 1 and 3 exactly reproduce
their Stage 06 64-token sequences despite being run in the same six-prompt
session.

**Verdict:** refuted for the reported Fibonacci regression. The same harness,
request lifecycle, prompt, cache/reset path, trace, and sampler recover the
entire Stage 06 sequence when only LM-head accumulation geometry changes.

## Smallest decisive real-target diagnostic

Do not use another 64-token prose comparison as the primary experiment. Run one
prompt-5 prefill and inspect its first sampler-ready row.

1. Produce the pre-final hidden row once and apply final RMSNorm once.
2. From that identical normalized tensor, run both terminal projections:
   the Stage 06 interleaved auto program and the Stage 07 DRAM-sharded program.
   Retaining both tied heads temporarily costs one additional TP-local BF16
   weight shard/device and avoids decoder noise between runs.
3. Read back aligned pre-softcap logits and aligned post-softcap logits. Record:
   PCC, maximum/mean absolute difference, global argmax, ordered top-10 IDs and
   scores, top-1/top-2 margin, exact-equality groups, and values around split/TP
   boundaries 8191/8192 and 65535/65536.
4. On the Stage 07 device logits, compare custom split-greedy with composed host
   `torch.argmax`.
5. Repeat prompt 5 once in isolation and once after another request to exclude
   state/lifetime sensitivity.

Interpretation is decisive:

- high aligned PCC with only close candidates swapping verifies numeric
  accumulation/tie sensitivity;
- collapsed aligned PCC that recovers after an 8192- or 65536-block reorder
  verifies an ordering bug;
- host argmax differing from custom greedy verifies a sampler bug;
- isolated versus sequenced differences verify a reset/lifetime bug.

A cheaper controlled adjudication resolved the intervention boundary. The
four-input-shard, `in0_block_w=2`, split-8192 candidate measures 339.823107
reduced steady t/s/u versus 339.834362 for the eight-shard/block-3 default: a
0.0033% reduction, within measurement noise. On the full 60-layer Fibonacci
run, the candidate TT IDs and completion match Stage 06 exactly 64/64, while
the block-3 default differs at token 0. Evidence is preserved in:

- `candidates/lm_head_dram4_split8192_block2_perf.json`;
- `candidates/lm_head_dram4_split8192_block2.xml`;
- `candidate_fibonacci_block2/vllm_qualitative_outputs.json`;
- `candidate_fibonacci_block2/qualitative_prompt_format.json` and
  `qualitative_rendered_prompts.json`;
- `candidate_fibonacci_block2.log`.

The same-hidden aligned-logit diagnostic should still be collected after the
default selection. It will quantify the verified geometry effect and close the
narrower exact-tie-versus-close-score question, but it is no longer needed to
choose the intervention boundary.

## Fix ledger

The verified selection is the four-input-shard/block-2, split-8192
DRAM-sharded LM head. It restores the Stage 06 Fibonacci result at effectively
identical reduced performance without changing model dtype/fidelity, decoder,
cache, CCL, sampler, or host-boundary policy.

- Keep the legal four-shard/block-2 DRAM geometry as the default candidate.
  Do not change decoder/KV/CCL dtypes or start the `$datatype-sweep` frontier
  here.
- Keep the vocabulary-ordering, sampler, and request-lifecycle hypotheses
  refuted; the selected fix must not alter those contracts.
- Do not substitute force-argmax, generic full-vocabulary TopK, a replicated
  logit stream, or a host boundary for the canonical split-greedy path.

After selecting the default, rerun and retain the same-hidden aligned-logit A/B,
the final six-prompt qualitative control, full prefill and teacher-forcing
accuracy gates, traced token-out, and reduced performance/watcher checks. The
single-prompt candidate proves the cause and selection; it does not replace
those final full-stage gates.

## Final status

**Resolved: select four input shards with `in0_block_w=2`.** The full-model
controlled contrast restores Stage 06's Fibonacci token IDs exactly 64/64 and
changes only LM-head accumulation geometry. Reduced throughput is 339.823107
t/s/u versus 339.834362 for block 3, a negligible -0.0033%. This verifies the
cause and identifies a performance-neutral fix; it is not a prose-based
limitation waiver.

No implementation/default was changed by this report-only update. The selected
default still requires source integration followed by the same-hidden
aligned-logit artifact, final six-prompt qualitative run, accuracy gates, traced
token-out, and reduced watcher/performance reruns before Stage 07 can close.
