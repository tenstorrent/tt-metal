# XTTS-v2 TTNN — Known Bugs

Running log of known bugs in the XTTS-v2 TTNN bringup, to return to when fixing.
Parent/context: `CLAUDE_XTTS_TTNN.md` (master). Per-block detail: `CLAUDE_XTTS_*.md`.

Status legend: 🔴 open · 🟡 worked around · 🟢 fixed

---

## BUG-1 🟡 `TTNNGPTDecoder` returns garbage when `max_seq` >> actual sequence length
- **Block:** 3 (GPT) — see `CLAUDE_XTTS_GPT.md`.
- **Discovered:** 2026-07-17, during full-pipeline integration (GPT prefill+decode on TT).
- **Component:** `models/experimental/xtts_v2/tt/ttnn_xtts_gpt_decode.py` → `TTNNGPTDecoder`
  (the non-traced KV-cached decode).

### Symptom
Decode is correct when the KV cache is sized tightly to the sequence, but produces garbage
when the cache is much larger than the actual decode position:
- same emb/prompt → teacher-forced next-code agreement **96%** at `max_seq=256`,
  **0%** at `max_seq=736` (first generated code flips 81 → 405).
- Prefill path (`TTNNGPTCore`) is unaffected; the traced decoder
  (`TTNNGPTTracedDecoder`) is expected unaffected (it uses `cur_pos_tensor`).

### Root cause
`TTNNGPTDecoder._attn_decode` calls
`ttnn.transformer.scaled_dot_product_attention_decode(..., cur_pos=[self.pos])`
with a **Python-int** `cur_pos`. The large unused/zero region of the preallocated cache
beyond `cur_pos` is not masked cleanly, and the error grows with the number of unused
slots. (Cache is `[1, n_head, max_seq, head_dim]`, zero-initialised.)

### Why tests missed it
`tests/test_gpt_decode_pcc.py` sizes `max_seq = round_up(S)` (tight), so the unused region
is small and the error stays negligible (PCC 0.9997). No test exercised a large `max_seq`.

### Fix (planned)
1. Switch `TTNNGPTDecoder` to a device **`cur_pos_tensor`** (int32 `[1]`, updated in place)
   instead of `cur_pos=[int]` — mirror what `TTNNGPTTracedDecoder` already does with
   `paged_update_cache(update_idxs_tensor=...)` + `sdpa_decode(cur_pos_tensor=...)`.
2. Add a regression test: decode with a large `max_seq` (e.g. 736) and assert PCC vs the
   prefill golden stays >0.999.

### Current workaround
Callers size `max_seq` close to the real sequence length. The temp pipeline
(`$CLAUDE_JOB_DIR/tmp/pipe/phase_tt.py`) sets `max_seq = round_up(S_hint)` from coqui's
sequence length and caps `max_new` accordingly.

### Repro
Feed a fixed prompt/emb through `TTNNGPTDecoder` twice, once with tight `max_seq` and once
with a large one, and compare `mel_head` argmax agreement (or latent PCC) vs a golden — the
large-cache run diverges.
