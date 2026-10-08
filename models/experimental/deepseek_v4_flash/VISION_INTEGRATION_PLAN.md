# DeepSeek-V4-Flash-Vision: integration plan

Plan for adding image input to the TT-NN DeepSeek-V4-Flash port, using the
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` checkpoint and its reference code
(`inference/{image_processor,vision,model}.py`, `encoding/encoding_dsv4.py` in the
HF snapshot).

## Summary

The vision model is a separate ~0.47B-parameter encoder (ViT + Aligner) whose output
replaces the input embeddings at image positions. The 43-layer language model is
unchanged except for three prefill-only behaviours:

1. **Embedding merge** - image positions get ViT/Aligner vectors or one of four learned
   marker vectors instead of a vocabulary lookup.
2. **Image-span attention** - inside `[IMAGE_START .. IMAGE_END]` the sliding-window part
   of attention is bidirectional; everything else stays causal.
3. **Image-aware MoE routing** - image positions rank experts with `bias_vl` instead of
   `bias`, and the 3 hash-routed layers use score top-k for them instead of `tid2eid`.

Decode is untouched: after prefill, image tokens only exist as KV-cache entries.

The hard part is not the ViT. It is the prefill scheduling: every image span has to land
entirely inside one prefill chunk, and the prompt has to be prefilled past the last
`IMAGE_END`. Today's server prefills only the 128-aligned prefix and replays the rest
through decode, which breaks the most common VLM prompt (an image followed by a short
question). Fixing that ("ragged commit") is the critical-path item.

Recommended order:

| Phase | Deliverable |
| --- | --- |
| 0 | Text-only model running from the Vision-Exp checkpoint |
| 1 | Host preprocessing, CPU goldens for the vision tower and the image-aware LLM |
| 2 | LLM-side changes (embedding merge, span mask, router) in eager prefill, ViT on host |
| 3 | Span-aware chunk scheduling and ragged prefill commit; end-to-end image demo test |
| 4 | ViT + Aligner in TT-NN |
| 5 | Traced prefill, OpenAI server and chat CLI integration |
| 6 | Accuracy and performance evaluation |

---

## 1. What the reference does

### 1.1 Prompt encoding

`encoding/encoding_dsv4.py` turns OpenAI `image_url` / Anthropic `image` content blocks
(or inline `<image>path</image>` text) into the special token `<｜deepseek_image｜>` plus
an ordered list of image records. The local copy
`models/experimental/deepseek_v4_flash/encoding_dsv4.py` has none of this.

`image_processor.prepare_vl_inputs` replaces each placeholder with a block of fake ids
`vocab_size + type`, where `type` is one of
`IMAGE_START=0, IMAGE_PAD=1, IMAGE=2, IMAGE_NEW_LINE=3, IMAGE_END=4`
(ids `129280..129284`).

### 1.2 Pixels to patches (`load_image`)

- RGB; upscale below 147,456 px; aspect capped at 8:1.
- Size picked so the block is at most 384 LLM tokens; letterboxed with gray (127),
  stretched only for extreme aspect ratios.
- Normalised to `[-1, 1]`, bf16, cut into raster-order 14x14 patches
  `[n_vit_h * n_vit_w, 3, 14, 14]`.

Examples (verified on CPU): `carrots.jpeg` 1024x701 -> 42x61 patches -> 14x21 = 294
image tokens, block length 311; `corn.jpeg` 450x308 -> 23x34 patches -> 8x12 = 96 image
tokens, block length 107.

### 1.3 Encoder (`vision.py`)

| Part | Shape |
| --- | --- |
| `PatchEmbed` | `Linear(588 -> 1024)` with bias, on the flattened patch (not a conv) |
| 32 x `Block` | pre-RMSNorm; attention with `wqkv`/`wo` (with bias), 16 heads x 64; SwiGLU MLP `w1: 1024 -> 2x2816`, `w2: 2816 -> 1024` (no bias) |
| Position | 2D RoPE only (no CLS, no absolute embedding): of the 32 rotary frequencies per head, 16 encode the row and 16 the column; rotate-half on the two 32-dim halves |
| Attention | Full bidirectional over one image, no mask |
| `norm` | final RMSNorm |
| `Aligner` | zero-pad the feature grid to multiples of 3, concatenate each 3x3 neighbourhood (9216), `Linear(9216 -> 4096) -> GELU -> Linear(4096 -> 4096)` |

Size: ~411M parameters in the ViT blocks plus ~55M in the Aligner, about 0.93 GB in bf16.
A maximum-size image is about 3,240 patches, roughly 4 TFLOP per image
(~2.7 linear, ~1.4 attention).

### 1.4 Token-block layout (`build_image_block`)

- Each grid row gets an `IMAGE_NEW_LINE`; an odd row count gets an extra `IMAGE_PAD` row.
- Rows are emitted two at a time, interleaved column by column
  (`(r0,c0), (r1,c0), (r0,c1), (r1,c1), ..., NL, NL`); `perm` maps the Aligner's raster
  output into this order.
- Leading `IMAGE_PAD`s put `IMAGE_START` at a position that is 3 mod 4, and trailing pads
  make the body a multiple of 4. This keeps the image body aligned with the ratio-4
  compressor windows.
- The whole block is at most 384 tokens (3 x 128).

### 1.5 Language-model changes (`model.py`)

- **Merge** (`merge_image_embeddings`): `params = [image_start, image_pad, image_pad,
  image_newline, image_end]`; `block = params[types]`, then `block[types == IMAGE] =
  embeds[perm]`; written into `h` at the block's offset. The vocabulary embedding returns
  zeros for ids >= vocab, so nothing leaks through.
- **Attention** (`get_image_visible`, `get_window_topk_idxs_visible`): for a query at
  absolute position `p` inside a span `[s, e]` (both markers included), the window keys
  are `[min(p - 127, s), e]`. Outside spans (including the leading compress pads) it is
  the normal `[p - 127, p]`. This applies to the window part of every layer type; the
  compressed-entry part and the CSA indexer stay causal.
- **MoE gate**: `image_mask = input_ids >= vocab_size` (all five marker types).
  - Learned layers: `ranking = scores + where(image_mask, bias_vl, bias)`.
  - Hash layers 0-2: `indices = where(image_mask, topk(scores + bias_vl), tid2eid[ids])`.
  - Routing weights are always the unbiased scores at the chosen experts.
  - The `mtp.*` (DSpark) layers also carry `bias_vl`.
- **Generation**: the image must be prefilled in one chunk (decode asserts no image ids),
  and image prompts run alone.

---

## 2. Where this lands in the TT port

| Concern | Current TT code | What has to change |
| --- | --- | --- |
| Prompt encoding | `encoding_dsv4.py` (local, text only) | Port the image path from the snapshot's `encoding/encoding_dsv4.py`, keeping the local reasoning-effort handling |
| Image preprocessing | none | New host module, copied from `inference/image_processor.py` (MIT licensed) |
| Vision encoder | none | Host torch first (Phase 2), then TT-NN (Phase 4) |
| Weights | `tt/weight_loader.py` HF-name map, `tt/weight_cache.py` | Load `vision.*`, `aligner.*`, `image_{start,end,newline,pad}`, `layers.*.ffn.gate.bias_vl` (all bf16, no scale) |
| Config | `configuration_deepseek_v4.py` has no `vision_*` | Add `vision_*` fields, read from the checkpoint's `config.json` |
| Embedding | `DeepSeekV4PrefillModel.embed` (`tt/model.py`): device `ttnn.embedding`, then repeat to `hc` streams | Override image rows before the repeat |
| Id validation | `_host_ids` rejects ids >= `vocab_size` | Keep raw ids for the image mask; clamp to 0 for the embedding and `tid2eid` lookups |
| Prefill attention mask | `tt/prefill/attention.py` `_mask`, `mask_tables_host` (traced) | Window columns follow the span rule; entry columns unchanged |
| Maskless sliding path | `_attend_maskless` / `_sliding_attend` | Cannot express spans; force the masked path on chunks that contain an image |
| Prefill MoE | `tt/prefill/moe.py` `route()` | Per-token bias for learned layers; top-k fallback for image rows in hash layers |
| Decode MoE / attention | `tt/decode/*` | No change (decode never sees image ids); add a guard |
| Prefill chunking | uniform `chunk_size`, multiples of `ALIGNMENT=128` | Span-aware chunk boundaries |
| Prefill -> decode commit | `commit_prefill_state` requires `T % 128 == 0` | Ragged `T` (see 3.5) |
| Server | `demo/server.py` `_content_text` returns 400 for image parts; prefill only from position 0 | Accept image blocks; image turns go through prefill |
| CPU golden for the LLM | `models/demos/deepseek_v3_d_p/reference/deepseek_v4/modeling_deepseek_v4.py` (HF style) | Add the three image behaviours |

The official reference `inference/model.py` cannot run on this box: it needs `tilelang`
CUDA kernels (`kernel.py`), and there is no GPU. `vision.py` and `image_processor.py` are
pure torch and run on CPU, so they are the golden for the vision tower. For the language
model, the existing HF-style CPU reference used by `tests/prefill/full_model_reference.py`
gets the image behaviours ported into it (Phase 1).

---

## 3. Design

### 3.1 Where the vision encoder runs

**First host torch, then device.** Running the snapshot's `vision.py` on host CPU unblocks
all the language-model work (Phases 2-3) with an exact golden. It costs seconds per image
on CPU, which is fine for tests.

**Device placement (Phase 4):**

- **`p150x8`** (8-stage pipeline, all chips busy): put the ViT on the stage-0 submesh,
  next to the embedding table. Weights are ~0.93 GB bf16 (~0.5 GB as bfloat8_b), well
  within free DRAM. The ViT then runs inside the same exclusive window the server already
  takes for traced prefill (drain decode, run, restore). It must follow the L1/GCB rules in
  `PREFILL_DECODE_INTERLEAVE_NOTES.md`: decode GCBs and L1 tensors alias under the large
  CBs of non-decode programs, so they must be restored afterwards.
- **`galaxy32`** (two 1x4 TP stages; 24 chips idle): put the ViT on an idle chip. It can
  then encode while other users decode, and only the merge plus prefill need the exclusive
  window.

The output is at most `[384, 4096]` bf16 (3 MB) per image, so moving it to stage 0
through the host is cheap.

### 3.2 Embedding merge

Do the merge on device in `DeepSeekV4PrefillModel.embed`, before the `hc` repeat:

```
emb = ttnn.embedding(ids_clamped, table)          # [1, T, D]
emb = ttnn.where(img_mask, img_embeds, emb)        # img_mask [1, T, 1], img_embeds [1, T, D]
```

- `img_embeds` is the per-chunk `[T, 4096]` block (Aligner output gathered by `perm`, plus
  the four marker vectors by type), built from the encoder output.
- Building `img_embeds` on the host for each chunk is simplest. It moves to the device once
  the ViT is on device (Phase 4).
- Only stage 0 needs `img_embeds`. Every stage needs `img_mask` for routing (3.4); the ids
  are already uploaded per stage in `_stack`.
- Traced prefill: `img_embeds` and `img_mask` become persistent input buffers, written
  before each chunk replay, like the ids. Text-only chunks write `img_mask = 0`.

### 3.3 Image-span attention mask (prefill only)

TT prefill attention is dense masked SDPA over the key layout
`[tail (128) | chunk (T) | compressed entries | pad]`.

For a query at absolute position `p` in a span `[s, e]`, the window keys are
`[min(p - 127, s), e]`. Because a span fits inside its chunk (3.5) and the tail holds the
128 tokens before the chunk, every needed key is already in the window columns. So the
change is confined to the window part:

- `_mask` (eager): after building the causal window,
  `window |= in_span(query) & in_span(key) & (key_abs <= e) & (key_abs >= min(p - 127, s))`.
  The span intervals are passed in per chunk.
- `mask_tables_host` / traced path: the static window table cannot encode spans. For
  chunks that contain an image, upload a full per-chunk window mask
  (`[1, 1, T, 128 + T]`, about 2.4 MB at `T = 1024`) into a persistent buffer used instead
  of the static table.
- Entry columns and the CSA lightning-indexer selection (`_select_entries`) are unchanged
  (causal). `IMAGE_START` / `IMAGE_END` are inside the span; the leading compress pads are
  not.
- Sliding layers (0, 1) normally take the maskless path. On image chunks they must use the
  masked path (a per-chunk flag, or always masked when the prompt has an image).

Write a host function `image_window_mask(start, T, spans)` that mirrors
`get_window_topk_idxs_visible` exactly, and unit-test it against the reference on random
span layouts. This needs no device.

### 3.4 MoE routing with `bias_vl`

In `tt/prefill/moe.py` `route()`:

- **Learned layers:** precompute `delta = bias_vl - bias` once per layer; then
  `ranking = scores + bias + img_mask * delta`.
- **Hash layers:** `hash_idx = embedding(ids_clamped, tid2eid)`;
  `vl_idx = topk(scores + bias_vl, k)`; `indices = where(img_mask, vl_idx, hash_idx)`;
  route with `SparseRouting(scores, indices=indices)`.
  - `scores` is only `[T, 256]`, so the extra top-k is cheap. It can be skipped when the
    chunk has no image rows.
  - The hash router currently drops `_bias_tile`; the vision checkpoint ships `bias` for
    the hash layers too, but with `vl` only `bias_vl` is used for image rows.
- **Decode routers:** `DeepSeekV4TopKRouter` and `DeepSeekV4HashRouter` stay as they are.

### 3.5 Prefill scheduling (critical path)

There are three constraints: an image span (at most 384 tokens) must not straddle a chunk
boundary, the prefill must cover every image, and decode must never see an image id.

**a) Span-aware chunk boundaries.** Eager prefill uses uniform `chunk_size`. Change it to
take a list of chunk boundaries, each a multiple of 128 and at most the maximum chunk size.
A host scheduler picks the largest chunks that do not cut a span. With chunks of at least
512 tokens and spans of at most 384, a valid split always exists.

Traced prefill is captured at one chunk size, so it cannot move boundaries. Options:

- **(i)** Eager prefill for image prompts (simplest; do this first).
- **(ii)** Capture a few extra chunk sizes (for example 1024 and 512, plus 128 for the
  remainder) and compose them. Only viable if the trace memory allows it.
- **(iii)** One chunk only: image prompts up to the chunk size (1024) run as a single
  chunk. This already covers one or two images with a short question.

Recommendation: (i) for bring-up; then (iii) plus (i) as a fallback in the server; (ii)
only if measurements show eager prefill is the bottleneck.

**b) Ragged commit.** The server prefills `floor(T / 128) * 128` tokens and replays the
remaining fewer-than-128 tokens through decode. A typical prompt ends
`... IMAGE_END <short question> <｜Assistant｜>`, so the end of the image often falls in
that replayed tail. Those image tokens would be processed causally (wrong) and carry image
ids into decode (asserted against in the reference).

Fix: prefill the prompt padded up to a multiple of 128, and commit exactly `T` real
tokens. This is exact:

- Padding sits after every real token. Text attention is causal, and spans never include
  padding, so no real token can see a pad.
- An entry whose window includes a pad only becomes visible at positions at or beyond the
  end of that window, which are all pads.

What `commit_prefill_state` / `_commit_*` need for a ragged `T`:

- Sliding ring: write the last 128 real rows so that ring slot = `pos mod 128` (today it
  assumes slot `j` holds token `T - 128 + j`).
- Compressed entries: only the complete windows `floor(T / rate)`.
- Partial compressor windows: fill the decode `win_*` accumulators for CSA (rate 4), HCA
  (rate 128) and the indexer compressor from the last `T mod rate` real rows. Prefill has
  to export those raw per-token compressor inputs. The CSA overlap half (`prev_*`) must be
  built from real rows only.
- Drop the `% ALIGNMENT` checks in `commit_prefill_state` and `_host_ids`; keep the
  128-alignment of the padded compute length.

This also helps text-only serving: prompts no longer need fewer-than-128-token decode
replays after prefill.

Interim, if ragged commit is not ready: refuse (or warn on) prompts whose last `IMAGE_END`
falls past `floor(T / 128) * 128`. For demos, a "deviation" mode can replay that tail and
measure the quality hit, but it is not a release mode.

**c) Follow-up turns.** The server prefills only at position 0 and feeds later turns
through decode. V1: a turn that adds an image resets the session and re-prefills the whole
conversation from 0 (the reference has the same restriction: no context without a
prefilled KV cache). Later: prefill continuation from an arbitrary position. That is the
same ragged-start problem in reverse, plus the tail/compressor state as prefill input.

### 3.6 Decode

No functional change. Add a host-side guard so no id >= `vocab_size` is ever written into
a decode packet. Today an id >= vocab would read out of bounds in the device embedding and
`tid2eid` tables.

### 3.7 DSpark / MTP

`tt/decode/dspark.py` (speculative decoding) is not used by `demo/server.py`. The vision
reference never runs `forward_spec` with images. DSpark's prefill of its own KV over image
positions would embed the fake ids as zeros, which is undefined behaviour that the
reference never exercises. Keep DSpark disabled for sessions with images in V1. Evaluate
its accept rate after an image prompt as a follow-up (the checkpoint does ship
`mtp.*.ffn.gate.bias_vl`).

### 3.8 Weights, config and cache

- The vision snapshot already uses native names (`vision.blocks.N.attn.wqkv.weight`,
  `aligner.w1.weight`, `image_start`, `layers.N.ffn.gate.bias_vl`). Add pass-through and
  HF-style rules to `tt/weight_loader.py`.
  - All vision tensors are bf16 with no `.scale`, so they need no dequantisation.
  - Shapes: `patch_embed.proj` `[1024, 588]`, `wqkv` `[3072, 1024]`, `mlp.w1`
    `[5632, 1024]`, `aligner.w1` `[4096, 9216]`, `image_*` `[4096]`.
- Config differences between Vision-Exp and the text checkpoints:
  - `vision_*` fields (new).
  - `rms_norm_eps = 1e-20` (vs `1e-6`); the TT code reads `config.rms_norm_eps`. Confirm it
    survives the bf16/fp32 paths.
  - `num_nextn_predict_layers = 3`.
  - `num_hash_layers = 3` with no `mlp_layer_types`. Check that the TT config derives the
    hash layers correctly.
- `WeightCache` keys files by tensor name + dtype + layout only. The demos namespace the
  cache by checkpoint basename. Keep that, so the Vision-Exp text weights (a different
  fine-tune) can never be served from a `-0731` / `-DSpark` cache.
- The Vision-Exp snapshot is currently the only DeepSeek-V4 checkpoint in this machine's HF
  cache. `_DEFAULT_MODEL_DIR` in the demo tests points at `-0731`.

### 3.9 Prompt encoding and serving API

- Merge the snapshot's image handling into the local `encoding_dsv4.py`:
  `IMAGE_PLACEHOLDER`, `process_image_messages`, `_process_image_blocks`,
  `parse_tagged_text`. Keep the local `reasoning_effort` behaviour; the snapshot changed
  the allowed values and default. Add tests built from the snapshot's `encoding/tests`.
- `demo/server.py`:
  - `_content_text` / `_normalized_messages` keep `image_url` (http, https, data URLs) and
    Anthropic `image` blocks.
  - The tokenizer must contain `<｜deepseek_image｜>` (the Vision-Exp `tokenizer.json`
    has it).
  - Image decoding and resizing run off the device thread.
  - Image turns go through the prefill path even when `--no-prefill` is set (or fail
    clearly).
  - Add limits: maximum images per request, maximum image bytes, URL fetch timeout.
- `demo/chat_cli.py`: accept `<image>path</image>` in the prompt (the same syntax as the
  reference `example_vl.txt`). The CLI is decode-replay only, so it needs the prefill
  model for image turns.

---

## 4. Phased plan

Device commands in this plan (pytest, demos, server) are run by the person at the box,
per the repo's device-access rules.

### Phase 0 - text-only model on the Vision-Exp checkpoint

- Point the loader at the Vision-Exp snapshot; build a separate tile cache (first run is
  more than an hour).
- Fix config derivation (`num_hash_layers`, `rms_norm_eps`, MTP count) and make sure the
  loader ignores `vision.*` / `aligner.*` / `image_*` / `bias_vl` for now.
- **Exit:** existing text tests (`tests/prefill/test_full_model_prefill_verify.py`,
  `tests/decode/test_full_model_decode_demo.py`) pass with the Vision-Exp checkpoint, and
  text chat quality is sane.

### Phase 1 - host pieces and goldens (no device)

- New package `models/experimental/deepseek_v4_flash/vision/`:
  - `image_processor.py` (copied, with a reference/attribution header).
  - `reference_vision.py` (the snapshot's `vision.py`, loading bf16 weights from the
    snapshot).
  - `image_spans.py`: span extraction from ids, `image_window_mask(...)`, chunk scheduler.
- Ported encoding (3.9) with tests.
- Extend the HF-style CPU reference with the embedding merge, span-aware window mask and
  `bias_vl` routing, and check it against hand-built cases from the snapshot's `model.py`
  logic (the index functions are pure torch and can be imported directly).
- **Exit (host-only pytest):** preprocessing matches the reference exactly (types, perm,
  patch tensors) on the example images and on random sizes and aspect ratios;
  `image_window_mask` matches `get_window_topk_idxs_visible` on random layouts; the scheduler
  never cuts a span.

### Phase 2 - language-model changes in eager prefill (ViT on host)

- `embed()` override (3.2), id clamping and image mask, span mask in `_mask`, forced masked
  path on image chunks (3.3), router changes (3.4), `bias_vl` / `image_*` weight loading.
- Tests (device):
  - `tests/prefill/test_attention.py`: add image-span cases (span at chunk start, middle,
    end, two spans, span next to the first chunk's empty tail) for sliding, CSA and HCA
    layers.
  - `tests/prefill/test_moe.py`: add image-mask cases for a hash layer and a learned layer.
  - `tests/prefill/test_decoder_layer.py`: one layer with an image block.
- **Exit:** per-layer PCC on image chunks at the same thresholds the text tests use today;
  routing indices match the golden exactly on image rows.

### Phase 3 - scheduling, ragged commit, end-to-end

- Variable chunk boundaries in eager prefill (3.5a); ragged commit with export of
  partial-window state (3.5b).
- New `tests/prefill/test_image_prefill_decode_demo.py`: the reference's two-image
  carrots/corn prompt (`example_vl.txt`) through host ViT, eager prefill, ragged commit and
  decode. Check: last-token logits PCC against the CPU golden over the full prompt, and
  that the answer names carrot / corn in order.
- Text-only ragged-commit regression: prefill+commit of a prompt with `T mod 128 != 0`
  matches pure decode replay.
- **Exit:** correct answers on the example prompt and a small image set; ragged-commit
  regression passes.

### Phase 4 - ViT and Aligner in TT-NN

- New `tt/vision/{vit.py, aligner.py}`. Starting points:
  - `models/demos/qwen3_vl/tt/` (`vision_block.py`, `vision_attention.py`, `rope.py`,
    `patch_merger.py`): closest structure.
  - `models/tt_transformers/tt/multimodal/mistral_24b/vision_rope.py` (Pixtral 2D RoPE).
- Specifics to get right:
  - Patch embed as a linear with K = 588; pad K to 608 (tile multiple).
  - RoPE table per `(n_h, n_w)`: row frequencies then column frequencies, rotate-half,
    built on host and cached.
  - Fused `wqkv` with bias; SwiGLU with `w1` split into gate and up halves.
  - Pad the sequence to a bucket (multiple of 128, up to ~3,328) with a key-padding mask in
    non-causal SDPA, so the program cache sees a few shapes instead of one per image size.
  - Aligner 3x3 unfold as reshape/permute (or a host-built gather index) after zero
    padding; exact (erf) GELU.
  - Apply `perm` with `ttnn.embedding` over a row index, or fold it into the unfold gather.
  - Multiple images: run sequentially at first; pack with a block-diagonal mask later if
    multi-image throughput matters.
- Precision: start with bf16 weights and activations at HiFi4; then try bfloat8_b weights
  and HiFi2, keeping them only if end-to-end answers and PCC hold.
- Placement per profile (3.1). On `p150x8`, add the ViT to the exclusive prefill window and
  verify decode traces still replay correctly afterwards (GCB/L1 restore).
- **Exit:** per-block PCC >= 0.99 and full ViT + Aligner output PCC >= 0.99 against the CPU
  golden on the example images and a sweep of sizes; Phase 3 demo still correct with the
  device ViT.

### Phase 5 - traced prefill and serving

- Persistent `img_embeds` / `img_mask` / per-chunk window-mask buffers in traced prefill
  (3.2, 3.3); single-chunk traced mode for image prompts up to the chunk size, with eager
  fallback (3.5a).
- Server: image content blocks, preprocessing off the device thread, image turns forced
  through prefill, full re-prefill for follow-up image turns (3.5c), limits, DSpark off for
  image sessions.
- Chat CLI `<image>path</image>` support.
- **Exit:** `curl` with an `image_url` block returns a correct answer; mixed text and image
  users on the server; decode throughput for other users unchanged outside the prefill
  window.

### Phase 6 - evaluation

- **Accuracy:** a small VQA / OCR / chart subset, compared against the reference outputs
  published for the checkpoint (or a GPU run elsewhere if available).
- **Performance:** ViT time per image vs patch count; prefill time for image prompts
  (eager vs traced); time to first token for one and two images.

---

## 5. Risks and open questions

| Risk / question | Mitigation |
| --- | --- |
| Ragged commit touches the decode state layout (ring rotation, `win_*` accumulators, CSA overlap, indexer) | Text-only regression against pure decode replay before any image work depends on it |
| Traced prefill cannot move chunk boundaries | Single-chunk traced mode plus eager fallback; measure before adding traces |
| ViT CBs on `p150x8` stage 0 clobber decode L1/GCB state | Run inside the existing prefill exclusive window and reuse its restore path; or dedicate a chip on `galaxy32` |
| `rms_norm_eps = 1e-20` vs `1e-6` | Phase 0 PCC; check that no fused norm clamps or ignores eps |
| Vision-Exp text weights differ from `-0731`; no local text golden other than the CPU reference | Phase 0 exit criteria on this checkpoint |
| DSpark behaviour after image prompts is undefined in the reference | Disabled for image sessions in V1 |
| Per-chunk window-mask upload cost (about 2.4 MB per 1024-token chunk) | Only for chunks that contain an image; negligible next to the ViT |
| Without the official GPU reference, the end-to-end golden is our ported CPU reference | Cross-check the ported index functions against the snapshot's pure-torch helpers; sanity-check answers on the bundled examples |
| Is the indexer's causal top-k over entries that include image tokens enough for long multi-image prompts? | Same as the reference; nothing to change, but include a long multi-image prompt in Phase 6 |

---

## Appendix: reference numbers

| Quantity | Value |
| --- | --- |
| Patch size / ViT dim / heads / layers / MLP inner | 14 / 1024 / 16 x 64 / 32 / 2816 (SwiGLU) |
| Downsample | 3x3 -> 9216 -> 4096 -> 4096 |
| Max LLM tokens per image (incl. markers and pads) | 384 |
| Min pixels / max aspect | 147,456 / 8:1 |
| Fake image ids | `129280 + {START 0, PAD 1, IMAGE 2, NEWLINE 3, END 4}` |
| Window for a query at `p` in span `[s, e]` | keys `[min(p - 127, s), e]` (sliding window 128) |
| ViT + Aligner parameters / bf16 size | ~0.47B / ~0.93 GB |
| Compute per max-size image | ~4 TFLOP |
