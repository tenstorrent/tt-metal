# AutoFix: the GPQA blocker is a sliding-cache read-side wrap bug

Written 2026-08-17 on a second machine (`qb2-120-p04t03`, 4×P300C) from the
handoff in `doc/HANDOFF.md`. **The Stage 11 accuracy blocker is a functional
decode bug, not the numerical policy.** It reproduces in three minutes on one
prompt and the fix is three lines.

## The defect

`ttnn.experimental.paged_update_cache` (the cache **write**) is given
`cache_position_modulo=SLIDING_CACHE_TOKENS` so a sliding layer's K/V lands in
ring slot `pos % 1024`. `ttnn.transformer.paged_scaled_dot_product_attention_decode`
(the cache **read**) was not. The op's own nanobind documentation states the
consequence:

> `cache_position_modulo` (in tokens) treats the cache as a circular buffer:
> every page_table lookup uses `cur_pos % cache_position_modulo`. **Required when
> the cache is sized for sliding-window-only allocation (vLLM SlidingWindowSpec);
> without it, positions past the bounded capacity collapse onto physical block 0
> and silently corrupt the cache.**

25 of 30 layers are sliding, and their cache is allocated at exactly
`SLIDING_CACHE_TOKENS = 1024` (`tt/model.py:275`). So every generation whose
**absolute position** passes 1024 starts attending to the wrong tokens.

Sites (all three decode paths had it):

| file | write has modulo | read had modulo |
|---|---|---|
| `tt/optimized_decoder.py:1122` / `:1139` | yes | **no** |
| `tt/multichip_decoder.py:875` / `:886` | yes | **no** |
| `tt/functional_decoder.py:921` / `:938` | yes | **no** |

### The fix, in the shape the code already had

The reason the read drifted from the write is that **each call site assembled its
own cache view**: `_cache_view_kwargs(prefill=...)` returned the block-size view,
and every caller then bolted `cache_position_modulo` on separately — so one
caller could simply not do it.

So the modulo moves into that accessor, which now returns the complete view:

```python
def _cache_view_kwargs(self, *, prefill: bool, cache_position_modulo: int | None = None) -> dict[str, int]:
    """Every op that touches the paged cache takes its view from here."""
```

and every paged-cache op — `paged_fill_cache`, `paged_update_cache`, and both
SDPA reads — now takes `**cache_view` from a single call per decode/prefill.
`_fill_prefill_cache` no longer accepts a caller-built `fill_kwargs` either; it
derives the view itself from the modulo it is already given. After this, the
write and the read of a given call cannot disagree, in any of the three decoders.

`cache_position_modulo=1024` satisfies the op's constraints — it is a multiple of
the sliding block size (64) and `>= sliding_window_size` (1024).

### Regression test

`tests/test_functional_decoder.py::test_bounded_modulo_decode_reads_across_wrap`
is the read-side counterpart of the stage-04 write-side test. It decodes
`sliding_window + 80` positions twice — once through a bounded 1024-token cache
with the modulo, once through an unbounded cache without it — and requires equal
attention output at positions 1023, 1024, 1025 and the last step. Because
`sliding_window` makes both configurations attend to the same 1024 keys, any
read-side wrap error appears at the first probe past the window. 13 s on one
Blackhole chip. Evidence:
`doc/functional_decoder/bounded_modulo_decode_across_wrap.json`.

The shared `models/common/modules/attention/attention_1d.py` is **not** affected:
it rejects `sliding_window` together with paged attention outright
(`attention_1d.py:1466`). This defect is specific to this autoport's
hand-written bounded sliding cache.

## Evidence

Free-running greedy generation, shipped precision policy, traced decode, the
`gpqa_diamond_cot_zeroshot` prompt shape recorded in `tti_eval_gpqa_cot.json`
(the graded documents are in a gated dataset and were not available):

| prompt tokens | absolute pos 1024 falls at generated index | output coherent through | fully degenerate by |
|---:|---:|---:|---:|
| 157 | 867 | ~890 | ~925 |
| 637 | 387 | ~370 | ~410 |

The onset tracks **absolute position**, not generated count — which is what
distinguishes a cache-wrap defect from precision drift or a trace/state problem.
Before the wrap the model is fully coherent and reaches the *correct* answer
(`8V_0/(3\pi)`, choice A, at generated index ~800 in the 157-token-prompt run).
After it, output collapses to token soup (`"lue lue lidth and and and"`) and
never emits EOS — it runs to the token cap.

That is exactly the Stage 11 signature the earlier AutoDebug could not explain:

- `meta_gpqa_cot` allows 32,768 generated tokens, so essentially every request
  crosses absolute position 1024 → corrupted reasoning, and requests that
  degenerate never emit EOS. `AUTODEBUG_GPQA_DIVERGENCE.md` records "the last
  four are reported together at 3:20:54" — four requests exhausting the cap
  together, which it flagged as "a useful lead, not a finding".
- `meta_ifeval` PASSES because its TT run recorded no `max_gen_toks`
  (`tti_eval_ifeval.json`), so lm-eval's API default of 256 applied: with short
  prompts those requests never reach position 1024.

## Why every earlier gate missed it

- Stage 04 tested the sliding boundary as logical lengths **1024 vs 1025** with a
  cache-content PCC check. That validates the **write** side, which was already
  correct. No stage ran sustained decode across the boundary.
- The advertised-context evidence (`position 262143` decode, `S=262143` prefill)
  are capacity probes. `READINESS_ANALYSIS.md` notes the decode probe used "a
  rolled page table" — rolling the table by hand supplies the wrap that the
  missing kwarg would otherwise have to perform, so the probe passes while the
  serving path fails.
- Prefill is unaffected: sliding prefill attends over live chunked K/V rather
  than through the paged cache, so prefill PCC at S=33 and the 262,144-token
  capacity prefill are both genuinely clean.
- Longest correctness-measured generation on the branch was 100 tokens, and the
  first wrap cannot occur before position 1024.

## Verification

Same prompts, shipped precision policy, before vs after the three-line fix:

| run | prompt | generated | EOS | decode t/s/u | outcome |
|---|---:|---:|---|---:|---|
| `eos_shipped` (before) | 157 | 2048 (cap) | no | 28.97 | degenerate after pos 1024 |
| `eos_fixed` (after) | 157 | 884 | **yes** | 28.63 | correct, `\boxed{A}` |
| `eos_filler20` (before) | 637 | 1100 (cap) | no | 28.82 | degenerate after pos 1024 |
| `eos_filler20_fixed` (after) | 637 | 875 | **yes** | 28.78 | correct, `\boxed{A}` |

Throughput is unchanged (the spread is run-to-run noise; the two 637-token runs
differ by 0.1 %).

Cross-path check with the fix in place: traced decode with the shipped on-device
sampler and eager decode with an exact host argmax produce **bit-identical
884-token streams**, both terminating on EOS (`audit_fixed_1200.json`), with 0
sampler mismatches over 884 step-level A/Bs.

Scope note: hardware verification here covers the optimized/multichip decode path
that the full model and the vLLM adapter use. The identical fix in
`functional_decoder.py` is by inspection — that path is exercised by
`tests/test_functional_decoder.py`, which was not re-run on this machine.

`doc/tti_release/experiments/sliding_wrap/` carries the runner and the before
and after artifacts. Reproduce with:

```bash
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' python \
  doc/tti_release/experiments/sliding_wrap/tt_longgen.py \
  --max-new-tokens 2048 --max-seq-len 8192 --out /tmp/after.json
# and again with --filler-repeats 20 to move the prompt length and confirm the
# failure follows absolute position 1024 rather than the generated count.
```

## What this does NOT explain, and what was measured instead

The residual per-token disagreement with HF *before* the wrap is real but small
and is **not** a defect. Measured on this machine, before-wrap only:

| measurement | result |
|---|---|
| TT/HF greedy disagreement, identical prefix, 512 steps | 8 (1.56 %) |
| ...ranks of TT's token in HF's distribution | 7 of 8 at rank 2, 1 an exact HF tie; none worse |
| ...HF top1−top2 margin at disagreements vs agreements | median 1.25 vs 15.19 |
| ...trend with decode index | 2.34 % (0–255) → 0.78 % (256–511): no accumulation |
| HF vs **itself**, only its own chunking changed | 1 of 512 (0.20 %) |
| HiFi2 everywhere + BFP8 decode MLP + FP32 logits | 9 of 512 (1.76 %) — no improvement, +28 % decode time |
| shipped chunked-topk device sampler vs exact argmax, same logits | 0 mismatches in 1024 step-level A/Bs |

So the handoff's ranked hypothesis 1 (a precision fix) is refuted by direct
measurement, and the shipped sampler is exonerated. The disagreements that remain
sit at near-ties, and they are structural for this architecture: on 20,070
top-8-of-128 routing decisions (669 tokens × 30 layers) measured on the HF
reference, **9.8 % have an 8th/9th expert gap below 1 % relative** (47 % below
5 %) — an average of 2.9 layers per token whose expert set is a coin flip.
Compounding this, HF computes the router in bf16 (`Gemma4RMSNorm.forward` returns
`.type_as(hidden_states)`) while this port computes routing in FP32, so the port
is the *more* precise side and must resolve tied experts differently by
construction. Exact greedy-trajectory agreement with the reference is not an
achievable target for this model; quality is, and it has to be measured
statistically over many documents rather than by trajectory match.

## Next steps

1. Re-run the Stage 11 `meta_gpqa_cot` / `meta_ifeval` gate with the fix. The
   IFEval run must set `max_gen_toks` explicitly (1,280, matching the HF
   control) — its recorded `gen_kwargs` did not, so the passing row was measured
   at the API's 256-token default.
2. Add a gate for sustained free-running decode past `2 × sliding_window` and
   assert coherence plus EOS, not just cache PCC at one boundary.
3. Re-check the multichip and vLLM decode paths for any other write/read kwarg
   asymmetry; the pattern here was that the two calls built their cache view
   independently.
