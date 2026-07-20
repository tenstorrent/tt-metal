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

---

## BUG-2 🟢 `ttnn.conv2d` OOMs on L1_SMALL when the device is opened without `l1_small_size`
- **Block:** 2 (ResNet speaker encoder) — see `CLAUDE_XTTS_SPEAKER_ENCODER.md`.
- **Discovered:** 2026-07-20, first on-device run of `tests/test_speaker_pcc.py`.
- **Component:** any `ttnn.conv2d` call (here the speaker encoder's conv1).

### Symptom
First `ttnn.conv2d` aborts with:
`Out of Memory: Not enough space to allocate 992 B L1_SMALL buffer across 62 banks ...
bank size is 0 B`. Backtrace goes through `sliding_window::move_config_tensor_to_device` /
`UntilizeWithHaloProgramFactory` (the conv halo step). Exit is non-zero; nothing computes.

### Root cause
`ttnn.open_device(device_id=0)` defaults `l1_small_size=0`, so the L1_SMALL region has zero
banks. `ttnn.conv2d`'s halo/sliding-window path allocates its config tensor in L1_SMALL and
has nowhere to put it. The Block-1/Block-3 tests never hit this because they use no conv2d.

### Fix (applied)
Open the device with a non-zero L1_SMALL, e.g. `ttnn.open_device(device_id=0, l1_small_size=32768)`.
(pytest's `device` fixture already sets an l1_small_size; this only bites the standalone
`__main__` device open.) With that, the full conv-heavy ResNet runs at PCC 0.99972.

---

## BUG-3 🟡 HiFi-GAN vocoder: bf16 convs cap waveform PCC at ~0.96; fp32 needs a bigger L1_SMALL
- **Block:** 4 (HiFi-GAN vocoder) — see `CLAUDE_XTTS_HIFIGAN.md`.
- **Discovered:** 2026-07-20, first on-device run of `tests/test_hifigan_pcc.py`.
- **Component:** `models/experimental/xtts_v2/tt/ttnn_xtts_hifigan.py` (the whole generator).

### Symptom
With bf16 conv activations the final waveform sits at **PCC ~0.961** (below the 0.99 gate),
even though every per-stage transpose output is ~0.999. Switching to fp32 activations at the
usual `l1_small_size=32768` then OOMs on L1_SMALL:
`Out of Memory: Not enough space to allocate 35280 B L1_SMALL buffer ... bank size is 32768 B`.

### Root cause
Two coupled effects:
1. **bf16 precision.** Error accumulates across the 12 ResBlocks (each 6 convs + residuals).
   Per-stage that error is tiny (transpose outputs ~0.999), but the final **32→1 `conv_post`**
   is a channel-reduction that extracts a linear combination in which the accumulated error
   does *not* cancel, and on an **oscillatory** waveform (PCC-sensitive) it shows as ~0.96.
   Weight precision is not the lever (fp32 weights + bf16 acts was *worse*, 0.937); the
   activation dtype is. HiFi3 vs HiFi4 compute config made no meaningful difference for bf16.
2. **fp32 L1_SMALL.** The conv sliding-window/halo config tensor is larger in fp32 than bf16,
   so bf16's 32768 L1_SMALL (BUG-2) is not enough. More DRAM slices don't help — the whole
   L1_SMALL region fills.

### Fix (applied)
Run the generator in **fp32 activations** (`TTNNHifiganGenerator(dtype=ttnn.float32)`,
`preprocess_hifigan_parameters(conv_dtype=float32, lin_dtype=float32)`) and open the device
with **`l1_small_size=65536`** (both the standalone `__main__` and the pytest `device_params`
fixture). Result: **waveform PCC 0.9983**, all per-stage ≥ 0.9998. bf16 remains available via
the `dtype=` knob for a faster/lower-fidelity run (~0.96).

### Aside (cost debugging time, not a device bug)
A ttnn intermediate stashed on-device for later inspection reads back as garbage after the
forward completes (its DRAM buffer gets reused). The **forward path itself is correct** — it
looked like the first `conv_transpose2d` was broken (0.27) when it was actually 0.9999. Fix in
the test: snapshot each per-stage tensor to host (`ttnn.to_torch`) at capture time, not after.
