# Sampler all-gather AutoDebug probe

## Status

Instrumentation is ready but has not been run on TT hardware.  The parent agent
was notified before device use so this probe can be serialized with the other
full-model work.  No sampler implementation change is proposed by this report.

The unresolved observation entering this investigation is that the global
argmax of the host-composed per-rank `_trace_logits` differs from both device
`argmax` and device `topk` after the sampler's full-vocabulary
`all_gather_async`.  Prior isolated tests refuted the reduction width, BFP8
reduction, output aliasing, and a controlled TP4 gather as general causes.

## Instrumentation

`models/common/sampling/tt_sampling.py` has an inspection-only opt-in hook that
retains the exact tensor returned by the real sampler `all_gather_async` before
untilize/reduction.  It adds no operation to the sampled graph.

`tests/full_model_perf.py --gather-debug-probe` performs one destructive
diagnostic sequence in the same full-model capture state:

1. Replay the model trace and snapshot every `_trace_logits` shard and address.
2. Replay the captured sampler and snapshot the retained gather, then snapshot
   `_trace_logits` again to detect mutation or aliasing.
3. Compare every gathered rank with `torch.cat(input_shards, dim=-1)` over the
   full padded extent using PCC, max absolute error, allclose, dtype/shape, and
   row-0 top-8 values and indices.
4. Release only the sampler trace and run eager sampling on the identical
   `_trace_logits` object.  Record input/gather addresses and compare eager
   gather against both host composition and the captured gather.

All trace-owned gather data is copied to host before sampler trace release.  No
model replay follows the release.  The public host oracle still trims to valid
vocabulary for semantic argmax; gather correctness is checked over the padded
full extent.

## Decision table

- Captured gather disagrees with host composition while eager gather agrees:
  captured sampler trace/address state is the leading cause.
- Captured and eager gathers agree with each other but disagree with unchanged
  input composition: the real full-model gather/path configuration is the
  leading cause.
- `_trace_logits` changes across captured sampling: input mutation/aliasing is
  the leading cause and the post-sampler host oracle was invalid.
- Both gathers agree with host composition but device tokens disagree with its
  top-1: gather is exonerated; reduction/output feedback remains responsible.
- Input shard ordering or addresses differ: the host oracle/composition or
  trace tensor identity is invalid, and no gather conclusion should be drawn.

## Exact command

```bash
python models/autoports/qwen_qwen3_6_27b/tests/full_model_perf.py \
  --prompt models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_chat_prompt.txt \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/artifacts/full_model_gather_debug.json \
  --prompt-tokens 128 --decode-tokens 1 \
  --feedback-overwrite-probe --gather-debug-probe \
  2>&1 | tee models/autoports/qwen_qwen3_6_27b/doc/full_model/logs/full_model_gather_debug.log
```

The expected semantic assertion currently fails, so the structured probe JSON
is printed to the log before the exception; the normal output artifact may not
be written.

## Static verification

```text
python -m py_compile models/common/sampling/tt_sampling.py models/autoports/qwen_qwen3_6_27b/tests/full_model_perf.py
git diff --check
```

Both checks pass.  A fresh source-only review found the probe location correct
and requested the pre/post input snapshots and captured-versus-eager numerical
comparisons now included here.  Its sandboxed AutoDebug runner could not launch
because bundled bubblewrap was unavailable; it made no edits and used no TT
hardware.
