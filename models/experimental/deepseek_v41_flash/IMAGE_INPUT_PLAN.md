# Image input for DeepSeek-V4.1-Flash (plan)

How `smanoj/metal_2` (branch `ssinghal/dsv4p1`, `models/demos/blackhole/deepseek_v41_flash`) serves image
prompts, what is different in this tree, and how to add the same thing here, including where the new device
work runs.

## 1. What image input needs (the checkpoint's `inference/model.py`)

The local checkpoint (`DeepSeek-V4.1-Flash-1017`) has everything: `vision_config` (ViT of 32 blocks, dim 1024,
16 heads, MLP 2816, patch 14, 3x3 downsample), `image_token_id = 129264`, 309 vision tensors (`vision.*`,
`aligner.*`, `image_start` / `image_end` / `image_newline`, all bf16), and `ffn.gate.bias_vl` (fp32 [384]) on
all 40 layers and the 3 MTP layers. The reference does four things differently for an image prompt:

1. **Prompt expansion.** `inference/image_processor.py` resizes each image (at most 1024 LLM tokens, about
   1302 x 1302 px), cuts it into 14 x 14 patches, and expands the placeholder into a span
   `[START] + ([IMAGE] * w + [NEWLINE]) * h + [END]`. Every position of the span carries `image_token_id`;
   a `token_types` array (TEXT = -1, otherwise >= 0) tells them apart.
2. **Embedding override.** `merge_image_embeddings`: IMAGE slots take `aligner(ViT(patches))` rows, the
   delimiters take the learned `image_start` / `image_end` / `image_newline` vectors.
3. **Routing.** Every layer's `Gate` ranks image-span tokens with `bias_vl` instead of `bias` (the routing
   weights still come from the unbiased scores).
4. **Engram.** Image tokens are cached as `DEAD` in the n-gram hash state, so no n-gram spans them, and the
   Engram gate is masked to 0 at image positions.

Decode needs nothing new: generated tokens are text. The MTP / DSpark layers are called without an image
mask in the reference, so they need nothing either.

## 2. How metal_2 does it

| Piece | File (metal_2) | What it does |
| --- | --- | --- |
| Config | `tt/vision_args.py` | `VisionArgs` from `config.json` (no ttnn, so it also works in the vLLM frontend). |
| Host preprocessing | `tt/image_processor.py` (vendored copy of the checkpoint's), `tt/vision_prompt.py` | Resize plan, patchify, span token types, prompt expansion. A test checks that the vendored copy matches the checkpoint's. |
| Image encoder | `tt/vision.py` `DSV41Vision` | ViT + aligner on the 4x8 mesh: **tensor parallel over the 8 columns** (2 heads, 1/8 of the MLP and the aligner per column, all-reduce along the row), **a different image on each of the 4 rows** (one wave = up to 4 images). 2D RoPE through `rotary_embedding_llama`; full attention inside an image via SDPA `cu_window_seqlens` (the padded tail is its own window). The 3x3 unfold is a `ttnn.embedding` row gather, with `aligner.w1` stored in the gather's order. Padded-length buckets (patches 256..9216, aligner tokens 64..1024), every program compiled at startup, the ViT trunk traced for buckets up to 3072 patches. About 140 MiB of weights per chip, 100-160 ms per wave. Output goes back to the host as `[span_len, 5120]` bf16 rows. |
| Per-request state | `tt/vision_state.py` `VisionSeq`, `VisionBook` | Locates the spans in the token ids, encodes each image once (an LRU keyed by the sha1 of the pixels), and returns per-chunk windows `(image mask [C], override rows [C, 5120])`. The state lives as long as the request, because a re-prefill rebuilds the prompt from token ids that only carry the placeholder id. |
| Prefill | `tt/prefill_model.py`, `tt/device_head.py` | Persistent per-chunk device buffers `ov` (override rows) and `vl` (image mask), uploaded per chunk with the tokens. Embedding becomes `emb * keep + override` with `keep = 1 - vl` (exact, since keep is 0 or 1). Engram rows are multiplied by `keep` on the host (value = `wkv(0)` = 0, so the contribution is exactly the reference's masked gate). The hasher gets `token_mask = ~image` (DEAD). |
| Router | `tt/router.py`, `tt/moe_weights.py`, `tt/prefill_unified_moe.py` | Loads `gate.bias_vl` when `DSV41_VISION=1`; per-token bias = `bias + vl * (bias_vl - bias)`. |
| Serving | `tt/vision_vllm.py`, `tt/generator_vllm.py` | vLLM multimodal processor, `SupportsMultiModal`, up to 16 images per prompt. |
| Tests | `tests/test_vision_*.py` | PCC against the CPU reference (0.9977-0.9987 on 5 shapes), host tests (processor equals the checkpoint's, hash mask, windows), splice and identity (bit-exact without images), determinism, interference between users, trace equals eager. |

## 3. What is different here

metal_2 runs the whole model on one 4x8 mesh: tensor parallel over 8 columns and one user per mesh row.
This tree is an **8-stage pipeline of 1x4 submeshes** (TP4, five layers per stage), one decoding user per
stage, joined by D2D sockets, with a traced, pipelined prefill (`DeepSeekV41TracedPrefill`). Several
chunks are in flight at once. Because of that:

* Per-chunk inputs cannot sit in one persistent buffer rewritten before each replay, as in metal_2, because
  an earlier chunk's trace may still be reading it. Inputs reach the trace through the **packet** (H2D on
  stage 0, then forwarded D2D stage to stage) or through **H2D sockets** (the Engram rows). Image input has
  to use the same two channels.
* The router runs on every stage, so every stage needs the image mask. The embedding runs on stage 0
  only, so only stage 0 needs the override rows.
* The server drains in-flight decode steps before a prefill (`_prefill_on_device`), so the device has a
  natural window for the vision encode at the start of the prefill.
* This tree has the ops the ViT needs: SDPA `cu_window_seqlens`, `rotary_embedding_llama`,
  `nlp_create_qkv_heads`, and `_tp_all_reduce` for the 1x4 all-reduce.

## 4. Where the new device work runs

### What each stage holds today (eight stages, five layers each)

| Stage | Layers | Extra residents | Compressed-KV role |
| --- | --- | --- | --- |
| 0 | 0-4 | embedding table, Engram layer 1 (sockets in L1), packet H2D | kv + index source 2 (ratio 2) |
| 1 | 5-9 | | kv + index source 8 (ratio 2) |
| 2 | 10-14 | Engram layer 14 | kv + index source 14 (ratio 2) |
| **3** | **15-19** | **none** | **ratio-2 consumer only: no kv source, no index source** |
| 4 | 20-24 | | kv source 20 (ratio 1), index sources 20, 24 |
| 5 | 25-29 | | ratio-1 index source 28 (full `comp_kv` copy) |
| 6 | 30-34 | | ratio-1 index source 32 |
| 7 | 35-39 | `lm_head`, DSpark (3 MTP layers with experts), D2H | ratio-1 index source 36 |

Stage 3 is the lightest stage in DRAM and in L1. It holds no embedding, Engram, head or DSpark, and no
compressed-KV pool that grows with the context (ratio-1 pools hold one entry per token and dominate at long
context).

### Size of the new work

* **Tower weights:** about 485 M parameters, about 970 MB in bf16 (ViT 411 M, aligner 73 M). At TP4 that is
  **about 245 MB per chip** (about 30 MB per DRAM bank), or about 130 MB per chip in bfloat8_b. Persistent
  input buffers for the buckets add about 40 MB per chip. Activations at the largest bucket (9216 patches)
  are about 20 MB per tensor per chip, and transient.
* **Image mask:** C int32 slots in the packet, 4 KB at a 1024-token chunk.
* **Override rows:** one 10 KB bf16 row per token, through an H2D socket on stage 0, only for chunks that
  contain image tokens.
* **`bias_vl` delta:** one [1, 1, 1, 384] tile per layer on its own stage. This is negligible.

### Options for the tower

| | Placement | Per-chip DRAM | Images per wave | Notes |
| --- | --- | --- | --- | --- |
| **A (start here)** | TP4 on **stage 3** | ~245 MB on stage 3's 4 chips | 1 | Simplest. Lightest stage. The other 28 chips are untouched. A 16-image prompt runs 16 waves in a row. |
| **B (scale-out)** | TP4 on a list of stages, e.g. `3,1,5,6` (never 0 or 7) | ~245 MB on each chosen stage | one per stage | metal_2's "different image per row" trick, with stages as the rows. Same code as A, plus round-robin dispatch: enqueue a wave on every vision stage without blocking, then read them all back. Cuts TTFT for multi-image prompts by about the number of stages. |
| C | One `(8, 4)` tensor on the full mesh (metal_2's layout transposed: TP over the 4 columns, one image per row) | ~245 MB on all 32 chips | 8 | Closest to metal_2's code. But it mixes parent-mesh workloads with the submesh traces and sockets, and it puts weights on stages 0 and 7, which are the tightest. Too risky. |
| D | ViT pipelined over the stages (4 blocks per stage) | ~30 MB on every chip | 8 in flight | Smallest footprint and best throughput. But it needs its own sockets (L1 FIFOs are scarce), its own traces, and 7 extra hops of latency. Only worth it if DRAM becomes the limit. |
| E | TP1, one image per chip of a stage | ~1 GB per chip | 4 | No CCL. About 4x the per-image latency of TP4, and the most memory. |

**Recommendation:** build A, with a `stages` argument from day one so that B is a config change
(`DEEPSEEK_V41_VISION_STAGES`, default `3`). Keep stages 0 and 7 off the list. Order B's list by measured free
DRAM (stage 3, then 1, then 2, 5, 6). Weights stay bf16 (metal_2's PCC was measured in bf16); switch to
bfloat8_b only if a long-context config doesn't fit.

### The rest of the new work

* **Embedding override:** stage 0, inside the stage-0 trace, as `emb * keep + override`.
* **Image mask:** every stage reads it from the packet. It is decoded once per stage in `_step_inputs`,
  into a [1, 1, T, 1] tile.
* **Router change:** stays on each layer's own stage.
* **Engram masking:** host only (the hash state and the row gather).
* **Patchify and resize:** host, in the HTTP request thread, before the turn is queued. The device window
  then only covers the ViT.

### When the encode runs

In `_V41Prefiller._prefill`, after the decode drain and after the decode prefetcher is stopped (the same
stop `prefill_traced` performs), and before `prefill_traced`. The ViT's L1 circular buffers must not share
L1 with the prefetcher's global CBs. On stage 3, the decode pipeline is stalled for the encode (about one
wave per image). In option B, the stall is the slowest stage's wave.

Build order: allocate the tower weights and the bucket input buffers **before** the KV pools are planned
and before any trace is captured. A persistent tensor allocated under a live trace can be overwritten by
replays, and the pools should see the memory as used. Capture the vision trunk traces (phase 4) last.

## 5. Work items

### Phase 0: measure (device; ask before running)

Build the server's model at the default config and the longest-context config you intend to serve.
For each stage, log `ttnn.get_memory_view(dev, DRAM)` (free and largest contiguous block per bank) and the
L1 view. This confirms stage 3 as the first vision stage and gives the order for option B.

### Phase 1: host side (no device)

* `tt/config.py`: `vision_args(model_dir)` reads `vision_config` and `image_token_id`. Add `vision_weights(loader)`,
  thunks for `vision.*`, `aligner.*` and the three delimiters (bf16, no dequant).
* `tt/vision_prompt.py`: load the checkpoint's own `inference/image_processor.py`, as `engram_lookup._engram_module`
  does for `engram.py`, instead of vendoring a copy. Expose span planning, patchify, and `token_types` for a
  prompt.
* `encoding_dsv41.py` already turns OpenAI `image_url` blocks into `<｜deepseek_image｜>` placeholders plus
  image records. Add the placeholder expansion to a span (via `prepare_vl_inputs`) in the server's prompt
  build.
* `tt/vision_state.py`: port `VisionSeq` (per-chunk windows) and a per-session `VisionBook` (encoder rows per
  image, LRU by pixel hash). Key it by **session**, not by request: a continued turn or a bounded replay can
  re-run image positions.
* `tt/engram_lookup.py`: `__call__` / `hash_session` take `token_mask`, pass it to `NgramHashState`
  (`DEAD` at image positions), and zero the gathered rows there. Session export and import then carry
  `DEAD` automatically.

### Phase 2: vision tower (device)

**Done, single stage:** `tt/vision.py` `DeepSeekV41Vision` (one image at a time, eager, TP4 on one 1x4
submesh), `config.load_vision_config`, and `tests/test_vision.py`. The stage is `config.VISION_STAGE`
(`DEEPSEEK_V41_VISION_STAGE`, default 5: a KV-only prefill leaves stages 5-7 idle). The inputs are uploaded
per image rather than kept in persistent buffers, and the trunk is not traced yet (phase 4).

* `tt/vision.py` `DeepSeekV41Vision(config, loader, submeshes, cache)`: a port of metal_2's `DSV41Vision`
  to a list of 1x4 stage submeshes. Weights are column- or row-sharded over the 4 chips (`ShardTensorToMesh`
  dim 3 or 2) and replicated per stage. The all-reduce is `_tp_all_reduce`. Use `WeightCache` files under
  `vision/`. Keep the bucket scheme, the gather-based 3x3 unfold, the 2D RoPE tables, and the
  `cu_window_seqlens` fence.
* `encode_many(images)` sorts by size, assigns waves round-robin over `submeshes`, enqueues every stage
  without blocking, then reads back `[span_len, 5120]` host rows.
* `warmup()` compiles every bucket at startup.
* Test: `tests/test_vision.py`, PCC against the checkpoint's `ViT` + `Aligner` on CPU for 5 image shapes
  (9 to 7200 patches); target PCC >= 0.997, as metal_2 got.

### Phase 3: prefill integration (device)

**Done, as an extended embedding table** rather than the packet slots and override sockets planned here. The
plan missed one thing: the server never prefills a whole prompt. `commit_prefill_state` takes a multiple of
the 128-token window, so decode replays the prompt's last 1-127 tokens (128-255 with the KV-only prefill), and
those can be image tokens. Decode would have needed the overrides and the mask too. Both paths now take an image
token the same way:

* **Token ids past the vocabulary.** `DeepSeekV41Model(image_rows=R)` appends `R` zero rows to the shared
  embedding table on stage 0 (`ttnn.concat`, before anything else allocates). The three learned delimiters
  sit at `V + 0/1/2` (start, newline, end), and an image's IMAGE rows at `V + 3 + slot`. `vision.span_ids` maps a
  span's `token_types` and slots to its ids. `write_embedding_rows` writes rows with
  `deepseek_v4_prefill_write_rows` into a `[1, 1, V + R, D]` view, in fixed 32-row blocks, so it is one program.
  Prefill (eager and traced) and decode embed these ids with no trace or packet change. The session's `tokens`
  hold them too, so a replay or a re-prefill re-embeds the same rows.
* **Image mask from the ids.** `vision.image_mask(ids, V)` computes `ids >= V` on device. It runs once per
  stage per step: in decode's `_step_context` and in the traced prefill's `_step_inputs`, from the ids already
  there. The eager `_stack` uploads it. All of this happens only when `image_rows > 0`.
* **Router.** `DeepSeekV41TopKRouter.bias_table` is `[2, E]` bf16: row 0 is `bias - max(bias)`, row 1
  `bias_vl - max(bias_vl)`. `ranking_bias(image_mask)` gathers each token's row (`ttnn.embedding`), which is
  added to the scores in place of the broadcast bias. A text token gets exactly the text bias's bf16 values.
  `image_mask` goes through `DeepSeekV4MoE.decode_static`, `DeepSeekV4PrefillMoE.route` / `forward`, both V4.1
  decoder layers, and `V41PrefillStaticStep.image_mask`.
* **Engram.** `EngramHostLookup` hashes ids `>= V` as `DEAD` (the reference's `token_mask`), and their rows are
  zeroed in prefill (`__call__`) and decode (`_write_engram_rows`). Zero rows give exactly zero contribution,
  because `wkv` has no bias.
* **Server.**
  * `V41GenerationServer` takes image parts in user messages, inline base64 only; URLs and paths are refused.
    It allows at most 16 per request.
  * Each image becomes a placeholder text block tagged with its sha1, in the checkpoint's own block layout. The
    render then matches `process_image_messages` byte for byte, and a continued conversation is matched by its
    images too.
  * `_render_ids` replaces each placeholder token with the span from `_ImageBook`. The book decodes images on
    the HTTP thread. At admission (no step in flight) it encodes on the vision stage, writes the rows, and
    keeps them while any conversation (a session, a host conversation, a queued turn) holds the image. It
    evicts least recently used otherwise.
  * The tower is built and every bucket compiled before the KV pools. All buckets run again after the decode
    capture, so a lack of memory beside the decode state shows at startup.
  * Speculative decode refuses `image_rows`, because the drafter embeds on the host vocabulary.

### Phase 4: performance (device)

* Trace the ViT trunk per bucket up to about 3072 patches (metal_2: 1.7-1.9x on small images, nothing past
  3072). Capture last, after the decode and prefill traces.
* Option B: add stages to `DEEPSEEK_V41_VISION_STAGES` and measure TTFT for 1, 4 and 16 images.

## 6. Tests

| Test | Device | Checks |
| --- | --- | --- |
| Host processor and template | no | Our span expansion and `token_types` equal the checkpoint's `prepare_vl_inputs`; Engram `token_mask` writes `DEAD`. |
| `test_vision.py` | yes | Tower PCC against the CPU reference on 5 shapes, and on each vision stage (option B). |
| Identity | yes | A text-only prompt with vision enabled is bit-identical to vision disabled (mask 0, no override). |
| Prefill PCC | yes | Image prompt logits against `tests/reference.py` with `images` / `token_types`, eager and traced, with an image span straddling a chunk boundary. |
| Router | yes | Image-token top-6 matches the reference `Gate` with `bias_vl` on a few layers. |
| Serving | yes | 1 image, 2 images, 16 images, 17 images (400), an image in turn 1 followed by a text turn 2 (continued prefill), an edited conversation with a swapped image (no stale KV), and an end-to-end check of a number and colours read from synthetic images (metal_2's 48/48 check). |

## 7. Open questions

* Free DRAM per stage at the long-context configs (phase 0). If stage 3 does not fit the tower in bf16,
  use bfloat8_b weights before considering option D.
* Whether the stage-3 decode stall during an encode (estimated 150-300 ms per image at TP4, against
  metal_2's 100-160 ms at TP8) is acceptable while other users are decoding, or whether image encodes should
  be batched across queued requests, as metal_2's `VisionBook.update_many` does.
* Whether the image-mask ops on text chunks are measurable. If they are, also make the mask vision-only
  through the variant key.
