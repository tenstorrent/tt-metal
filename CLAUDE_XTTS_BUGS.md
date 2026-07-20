# XTTS-v2 TTNN — Known Bugs

Running log of known bugs in the XTTS-v2 TTNN bringup, to return to when fixing.
Parent/context: `CLAUDE_XTTS_TTNN.md` (master). Per-block detail: `CLAUDE_XTTS_*.md`.

Status legend: 🔴 open · 🟡 worked around · 🟢 fixed

---

## BUG-1 🟢 `sdpa_decode` returns garbage when the KV-cache length is an ODD number of tiles
- **Block:** 3 (GPT) — see `CLAUDE_XTTS_GPT.md`.
- **Discovered:** 2026-07-17, during full-pipeline integration (GPT prefill+decode on TT).
- **Root-caused & fixed:** 2026-07-20.
- **Component:** `models/experimental/xtts_v2/tt/ttnn_xtts_gpt_decode.py` → both
  `TTNNGPTDecoder` (non-traced) and `TTNNGPTTracedDecoder` (traced).

### Symptom
Decode is correct when the KV cache is sized to certain lengths and garbage at others:
- same emb/prompt → teacher-forced next-code agreement **96%** at `max_seq=256`,
  **0%** at `max_seq=736` (first generated code flips 81 → 405).
- Latent PCC vs the prefill golden: **0.9997** at `max_seq=256`, **0.631** at `max_seq=736`.
- Originally (mis)attributed to "large unused cache region not masked" and to the
  int-vs-tensor `cur_pos`. **Both wrong** — see below.

### Root cause (corrected)
`ttnn.transformer.scaled_dot_product_attention_decode` produces a wrong result whenever the
**K/V cache sequence length is an *odd* number of 32-tiles** (i.e. not a multiple of 64),
independent of `cur_pos` and independent of whether `cur_pos` is a Python int or a
`cur_pos_tensor`. Measured with a fixed cache slice (`cur_pos` held at 0..63):

| cache tiles | 8 (256) | 20 (640) | 21 (672) | 22 (704) | 23 (736) | 24 (768) | 28 (896) | 32 (1024) |
|---|---|---|---|---|---|---|---|---|
| latent PCC | 0.9997 | 0.9997 | **0.631** | 0.9997 | **0.631** | 0.9997 | 0.9997 | 0.9997 |

Odd tile counts (21, 23) fail; even tile counts pass. `max_seq=256` (8 tiles, even) worked
by luck; `max_seq=736` (23 tiles, odd) is the failure the pipeline hit. The non-traced and
traced decoders fail **identically** (bit-for-bit PCC 0.6310340), which is what disproved the
`cur_pos_tensor` hypothesis — the traced decoder already uses `cur_pos_tensor` and still fails.
Passing an explicit `SDPAProgramConfig` (`k_chunk_size` 0/32) does **not** help either. This
is an `sdpa_decode` flash-decode kernel bug (even-tile work-split assumption).

### Fix (applied)
Round the allocated KV-cache seq length **up to a multiple of 64** (even tile count) in both
decoders' `__init__`: `self.max_seq = ((max_seq + 63) // 64) * 64`, and allocate the zero
cache with `self.max_seq`. Trace-compatible (fixed graph), no per-step slicing, negligible
memory cost. Verified: requests `max_seq` 256→256, 605→640, 736→768 all give PCC **0.99972**
on both the non-traced and traced decoders.

### Why tests missed it
`tests/test_gpt_decode_pcc.py` sizes `max_seq = round_up(S)` to a multiple of 32 — which for
the short golden (S=64 → 64 = 2 tiles) happened to be even. See the regression test added at
`tests/test_gpt_decode_pcc.py` (odd-tile `max_seq` case).

### Repro (historical)
Feed the golden `inputs_embeds` [1,64,1024] through `TTNNGPTDecoder` at `max_seq=736`
(pre-fix) vs `768` and compare latent PCC vs `golden/gpt/latents.pt`: 0.631 vs 0.9997.

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
