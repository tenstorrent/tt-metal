# Record / trace layout and input sequence

This describes what is stored in `.pt` traces and how :class:`TraceReplayBaseAdapter` maps them to
``decode_state`` (including ``last_hidden_state``) for fusion / NextN drafts.

## Two on-disk formats

``load_trace_or_mtp_reference`` picks the loader from the top-level dict keys:

| Format | Detection | Loader |
|--------|-----------|--------|
| **MTP reference** | ``hidden_states`` and ``next_tokens`` present | :func:`load_mtp_reference_bundle` |
| **Collect-base trace** | otherwise (expects collect script keys) | :func:`trace_bundle_from_collect_payload` |

---

## A) `collect_base_trace_gpu.py` payload

Written by ``scripts/collect_base_trace_gpu.py`` (full base model on GPU).

| Key | Shape / type | Meaning |
|-----|----------------|--------|
| ``model_id`` | str | HF model id used when collecting |
| ``prompt`` | str | Original text (for humans) |
| ``prompt_token_ids`` | ``list[int]`` | Full **prefill** only (special tokens per tokenizer) |
| ``step_next_tokens`` | ``list[int]`` | Greedy **next** token at each decode step (length = recorded steps) |
| ``step_last_hidden`` | ``[num_steps, hidden_size]`` | Base model **last-layer** hidden (batch dim squeezed), **before** argmax at that step |
| ``step_multi_layer_hidden`` | optional ``[num_steps, …]`` | If base exposes aux hidden (e.g. EAGLE3) |
| ``topk_tokens`` / ``topk_scores`` | per-step top-k | Diagnostics |
| ``base_impl``, ``…`` | meta | Collection settings |

**Step indexing (collect loop):** for step ``i``, the code appends ``state.last_hidden_state``, **then** takes ``argmax`` of ``state.next_token_logits`` as ``step_next_tokens[i]``, **then** runs ``forward_decode`` with that token. So ``step_last_hidden[i]`` is the hidden **immediately before** predicting the token that becomes ``step_next_tokens[i]`` — i.e. the state used to produce logits for position ``i``.

**Full token sequence implied by the trace:**

```text
[prompt_token_ids[0] … prompt_token_ids[L-1]]  |  step_next_tokens[0]  |  step_next_tokens[1]  | …
```

The **input context** for step ``i`` (``i >= 0``) is: prefill + ``step_next_tokens[0:i]``. The **token being predicted** at step ``i`` is ``step_next_tokens[i]``.

---

## B) MTP reference payload (e.g. DeepSeek v3 test / MTP I/O)

| Key | Shape | Meaning |
|-----|--------|--------|
| ``hidden_states`` | ``[num_steps, batch, hidden_size]`` | Same role as ``step_last_hidden`` (per step, per batch row) |
| ``next_tokens`` | ``[num_steps, batch]`` | Greedy next token per step |
| ``start_tokens`` | ``[batch]`` | **Per-row first context token id** when the file has no long prefill. Replay sets ``prompt_token_ids = [start_tokens[batch_index]]``. This is *not* a natural-language string — for the default ``mtp_full_model_seq128.pt`` fixture, rows are synthetic ``0, 1, 2, …``. |
| ``metadata.start_token_id`` | scalar (optional) | Test harness default (often ``0``). **Do not** use this instead of ``start_tokens[batch_index]`` for row ``batch_index > 0`` or hiddens and prefix diverge. |
| ``logits`` | optional ``[num_steps, batch, vocab]`` or ``[num_steps, vocab]`` | Stored base logits for verification / confidence (recommended for replay) |
| ``metadata`` | dict | Often ``model_id``, ``num_prefill_tokens``, nested ``input_ids``, etc. |

**Prefill resolution:** :func:`_mtp_prefill_token_ids` tries many keys (``prompt_token_ids``, ``input_ids``, ``metadata.num_prefill_tokens`` + long ``input_ids`` = prefill + steps, etc.). If nothing matches, ``prompt_token_ids`` becomes **only** ``[start_tokens[batch_index]]`` — often **wrong** if the real run used a long prompt: hiddens were computed with the full context, but replay thinks the prefix is one token.

**Batch:** ``batch_index`` (default 0) selects one row from ``hidden_states`` / ``next_tokens``.

Converted :class:`TraceBundle` sets ``tokenizer_hub_id`` to NextN for tokenizer-only loads when appropriate.

---

## How replay builds ``DecodeState``

Implementation: :class:`TraceReplayBaseAdapter` in ``trace_replay_base.py``.

- **Trace position** ``pos`` = number of **generated** tokens after prefill that match the greedy prefix:
  ``pos = len(committed) - len(trace.prompt_token_ids)``.
- ``_state_from(pos, valid)`` sets:
  - ``last_hidden_state`` = ``step_last_hidden[pos]`` (when ``valid`` and in range),
  - ``next_token_logits`` = ``step_next_logits[pos]`` if present, else a synthetic one-hot at the record greedy token.

So at the start of a speculative round, ``decode_state.last_hidden_state`` is exactly the **recorded base hidden used to predict the next token** at that position — correct input for ``hnorm`` in MTP fusion **if** the trace’s base stack matches the model that produced ``last_hidden`` (e.g. full R1).

**Draft last token id:** adapters use ``committed[-1]`` = last token in context = the token **at** the last position of the current prefix, which is the same convention as “current token” for MTP embed in the same step.

---

## Checks that reading is “correct”

1. **Prefill must match:** ``engine.generate(prefix_token_ids=trace.prompt_token_ids)`` must use the **same** ``prompt_token_ids`` as in the file; ``forward_prefill`` asserts equality.
2. **MTP files:** Prefer payloads that include real **prefill** ids + ``num_prefill_tokens`` (or equivalent) so ``prompt_token_ids`` is not reduced to a single ``start_tokens`` id.
3. **Logits:** If ``step_next_logits`` / ``logits`` is absent, verification uses synthetic logits — acceptance stats are still defined but base “confidence” lines are misleading.
4. **Alignment:** ``_decode_state_for_committed`` marks ``valid=False`` if ``committed`` diverges from greedy ``step_next_tokens``; then replay hidden/logits may not correspond to a real base forward for that prefix.

Fusion-only works when (1)–(2) hold and the hidden size / tokenizer match NextN config; SGLang-structure draft adds sensitivity to decoder + norm + head on top of the same ``fused`` pipeline.
