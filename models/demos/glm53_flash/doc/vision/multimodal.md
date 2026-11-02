# GLM-5.3-Flash multimodal (image + text) path

Branch `arg/glm53_mm` (worktree `~/ashai/glm53/wt_mm`, from `arg/glm53_mesh` @ d1d7a64d33c). Code
`tt/multimodal.py`; models `tt/model.py` (`GLM53Model.prefill_streams`, `generate(streams=)`) and
`tt/mesh_model.py` (`GLM53MeshModel.prefill_streams`); HF side `reference/hf_multimodal_reference.py`
(+ `generate_streaming(x0=)` in `reference/hf_layer_reference.py`); tests `tests/test_multimodal.py`; demo
`demo/multimodal_demo.py` with `demo/sample_prompts/shapes_hello.png`.

## What HF does (`Glm5NextModel.forward`)

```
inputs_embeds = embed_tokens(input_ids)                               # bf16 [1, S, 4096]
image_embeds  = visual(pixel_values, image_grid_thw).pooler_output    # [N_img, 4096] -> bf16
inputs_embeds = inputs_embeds.masked_scatter(input_ids == image_token_id, image_embeds)
hidden        = inputs_embeds.unsqueeze(2).expand(-1, -1, 4, -1)      # the 4 mHC streams
```

The text model has no rotary embedding (`qk_rope_head_dim == 0`), so an image token is an ordinary
sequence position: the only multimodal work on the text side is the input streams.

## Device path

| step | where | what |
|---|---|---|
| `encode_multimodal(processor, image, question, think=False)` | host | `Glm5NextProcessor.apply_chat_template` with `{"type": "image"}` + the question: `<|begin_of_image|>` + `grid.prod()//4` x `<|image|>` (154854) + `<|end_of_image|>` + question + `<|assistant|><think>`; `think=False` appends `</think>` (154842) so the answer starts immediately (a checkable caption). 448x448 image -> grid (1, 32, 32), 1024 patches `[1024, 1176]`, 256 image tokens |
| `image_features(device, pixel_values, grid_thw, tower=None)` | device | `tt/vision.py` `TtVisionTower` (fp32 activations, bf16 weights, bf16 output): `prepare_inputs` -> tower -> per-image `[tokens, 4096]` host features (bf16 values, HF's cast). On a mesh the tower weights are replicated (`from_torch` without a mapper) and every chip runs it; chip 0's output is read. Without a resident `tower` it is built, run and released (the single-chip model needs the DRAM for the streamed text layers) |
| `multimodal_streams(model, input_ids, features)` | device | the text path for the ids (`TtEmbedding`: `ttnn.embedding` of the bf16 table into fp32 + `mhc_expand`), then `streams = streams * keep + img` with host-built `keep` (1.0 on text rows, 0.0 on image rows) and `img` (features rounded to bf16, repeated over the 4 streams, 0 elsewhere): fp32 ops with 1.0 / 0.0 operands, bit-exact on the text rows. `T_pad = ceil(S / (32 * rows)) * 32 * rows`; on a multi-row mesh `keep` / `img` are row-sharded on dim 2 (`mesh.upload(rows=2)`) like the ids (`rows=3`), i.e. each row holds its slab of the prompt, the layout `GLM53MeshModel.prefill` builds for text |
| `prefill_multimodal(model, input_ids, features, user=0)` | device | `model.prefill_streams(streams, S, user=user)`: the layers exactly as the text `prefill` (which now calls `prefill_streams` after embedding the ids); `GLM53Model` returns the last-token logits, `GLM53MeshModel` `(streams_local, logits_last, owner_row)` |
| decode | device | unchanged (text tokens); `GLM53Model.generate(ids, n, streams=...)` runs the prefill from the streams, the mesh demo / test replay the traced decode step |

## HF reference (`bringup/references/glm53_flash/MM_shapes_hello`)

`reference/hf_multimodal_reference.py --image demo/sample_prompts/shapes_hello.png --question "What shapes
and what word are in this image? Answer in one sentence." --generate 16`: the same `encode_multimodal`
prompt (286 tokens: 256 image tokens at rows 11..266, `</think>` appended), the tower in fp32 on the CPU
(`features.pt`: `fp32`, and `bf16_model` = the tower run in bf16 as the shipped model does), the scatter
into the bf16 embeddings (`inputs_embeds.pt`), then all 45 text layers in bf16 streamed one module at a time
per token (`generate_streaming(x0=streams)`: `generate.json`, `text.txt`). CPU time: tower 2 s, prefill
pass of 286 tokens 457 s, decode passes 620-1030 s (16 tokens over three host reboots, resumed from
`generate_state.pt`). Result: `[785, 2168, 5610, 264, 2518, 12665, 11, 264, 6303, 9331, 11, 264, 6176, 21445, 11, 323]`
= **'The image contains a red circle, a blue square, a green triangle, and'** (first token 785 'The' at logit
21.75, runner-up 19.0): the model reads the image correctly through this prompt.

## Validation

(filled in as the device runs complete; see `_agents/vision/STATUS.md` for the live state)

| check | result |
|---|---|
| processor encoding == reference ids (286 tokens, 256 image tokens at rows 11..266) | yes |
| tower on device vs HF fp32 features (this image; 1 chip, build + run + release 25.7 s) | PCC 0.98823 (HF-bf16 floor on this image 0.99086; bar min(0.995, floor - 0.03) = 0.961) |
| streams from HF features: text rows and image rows vs HF inputs_embeds, text rows vs the device text embedding | bit-exact (PCC 1.000000, max diff 0 on both row sets); all 4 streams identical |
| streams from TT features: image rows PCC vs HF inputs_embeds | 0.98823 (= the tower PCC), text rows bit-exact |
| single chip, 45 layers streamed (`GLM53Model`, fullmodel tensorbin cache), `GLM53_MM_N=3` per job with the checkpointed continuation | first token 785 `'The'` == HF; **16/16 greedy tokens == HF** over six jobs (`'The image contains a red circle, a blue square, a green triangle, and'`, every HF token in the TT top-5 at every step); step-0 top-5 ids identical to HF's (TT logits 21.375 / 19.125 / 18.625 / 18.375 / 17.125 vs HF 21.75 / 19.0 / 18.625 / 18.5 / 18.0); device argmax == host argmax on every decode step. Prefill pass 417 s, decode passes 389 / 370 s, tower 7.5 s: 1192 s per job, so 3 tokens is the maximum under the 1500 s device-job cap (16 tokens would take ~85 min) |
| mesh-model path on a (1,1) mesh (`demo/multimodal_demo.py`, `GLM53_DEMO_LAYERS=0-5`: resident `GLM53MeshModel.prefill_streams`, tower resident on the mesh device, traced decode after the multimodal prefill) | runs: tower build 3.2 s / forward 65 ms, TTFT 416 ms for the 286-token prompt (6 layers), traced decode 29.97 ms/step = 33.4 tok/s/user; text meaningless by construction (6 of 45 layers) |
| mesh (`GLM53_MESH_SHAPE` 2x4 / 8x4, `-k mesh`): first token == HF; 16 traced tokens; multi-row stream sharding | pending a mesh grant |

## Reuse of the repo's reference implementations (standing rule)

| this branch | sibling | verdict |
|---|---|---|
| `multimodal_streams` merge (host keep/img masks, fp32 multiply + add on the packed streams) | `models/demos/qwen3_vl/tt/common.py::merge_vision_tokens_single_user_ttnn` (`ttnn.scatter` of the device image embeddings at the `image_token_id` rows; torch twin `merge_vision_tokens`) | replace: scatter into the `[T, 4096]` device embedding before `mhc_expand` (no host masks, no readback) |
| `image_features` (tower on device, host readback per image) | `models/demos/qwen3_vl/tt/model.py::DropInVisionTransformer.forward` / `forward_single_user` (per-image loop, device embeddings straight to the merge) | replace together with the merge |
| `tt/vision.py` tower | `models/demos/qwen25_vl/tt/{vision_block,vision_attention,vision_mlp,vision_rmsnorm,rope,patch_merger}.py` + `qwen3_vl/tt/model.py::VisionTransformer` | port target: GLM5Next = the Qwen2.5-VL block structure + per-head q/k RMSNorm + 2x2 downsample conv + LayerNorm/GELU/SwiGLU-10240 merger |
| `prefill_streams` on both models | `qwen3_vl/tt/model.py::Transformer.prepare_inputs_prefill` (the stack takes embeddings) | same pattern, keep |
| `encode_multimodal` | the qwen VL demos' processor-driven prompts | same pattern, keep (`</think>` append is GLM-specific) |
| the generate / decode loops (`GLM53Model.generate`, mesh test, `multimodal_demo.py`) | `models/tt_transformers/tt/generator.py::Generator` via an adapter like `qwen3_vl/tt/generator.py` | replace when the GLM text stack gets a Generator adapter (not a `TTTransformer`: mHC streams, KDA/DSA state) |

## Video (frames -> tokens -> features)

HF (`Glm5NextProcessor` / `Glm5NextModel.get_video_features`): the video processor samples frames at 2 fps (the
checkpoint's `fps`, up to `max_frames` 2048), resizes them within the token budget and stacks every 2 consecutive
frames (`temporal_patch_size`) into one patch group `[h*w, 1176]` (the same patch layout as an image, whose 2
temporal slices are a duplicated frame), so `video_grid_thw = [[t, h, w]]` with `t = frames / 2` groups. The prompt
holds, between `<|begin_of_video|>` and `<|end_of_video|>`, one run of `h*w/4` `<|image|>` tokens per group,
each followed by `<|end_of_image|>` and the text `"<t>.<d> seconds"` (the group's timestamp): the `<|video|>`
token (154855) exists only in the chat template; every vision token in the ids is `image_token_id` (154854). The
tower runs the video as `t` images: the grid is flattened to `[1, h, w] * t` (one attention window and one 2D
rope table per group, no temporal position) and the merged features are scattered into the frames' runs in
prompt order; an image token belongs to a video when a `<|begin_of_video|>` precedes it without its
`<|end_of_video|>` (HF `get_placeholder_mask`'s cumsum rule).

| piece | where | what |
|---|---|---|
| `encode_prompt` / `encode_multimodal_video(processor, frames, question, fps=2.0)` | host | frames (`[T, H, W, 3]` uint8, a list of images, a directory of frame files: `load_frames`; `synthetic_clip(image, n)` = n rotated copies of one image) + `VideoMetadata(fps)` through the chat template + processor -> ids, `pixel_values_videos [t*h*w, 1176]`, `video_grid_thw [1, 3]`. A 4-frame 448x448 clip -> `[[2, 32, 32]]`, 2048 patches, 512 vision tokens in two runs, 549 tokens |
| `tt/mm_inputs.py` (`placeholder_runs`, `feature_rows`, `chunk_scatter`, `flatten_grid`, `vision_batches`) | host, no ttnn | image runs vs video spans from the ids; per item the rows its features take (by vLLM placeholder range or by order; a prefix that ends inside an item yields its rows so far); the per-device-chunk split; the grid flattening and the tower's window batching |
| `vision_features(tower, items, max_patches)` (`video_features`) | device | the flattened windows batched to `max_patches` per tower call (4096 = four 448x448 images / eight temporal groups; a window above the budget runs alone), per item `[n_tokens, 4096]` host features |
| `multimodal_streams` / `prefill_multimodal` | device | unchanged: the vision rows are all `image_token_id` rows in prompt order, so a video prompt is the image path with more rows |

Per-frame cost (the tower is linear in windows; the (2,4) block measured one 1024-patch window at 60-65 ms): one
temporal group of a 448x448 clip (2 frames, 1024 patches) costs one image, ~60 ms and 68 KB of fp32 transient per
patch (`tt/memory_plan.py::VISION_TRANSIENT_BYTES_PER_PATCH`: residual + gate|up + the two SwiGLU slices at the
MLP peak) = 68 MiB per group, freed before the text layers run; a 16-frame clip = 8 groups = 2048 vision tokens,
~0.5 s of tower in two 4096-patch calls of 272 MiB each. The HF CPU reference: fp32 tower 3 s / bf16 2 s for the
2048-patch clip. What the planner reserves: `vision_transient_bytes(batch_patches)` (one tower call's working set)
in `reserve_bytes`'s transient term (it precedes the layers' transient rather than adding to it), plus the tower's
weights when they are built per request (`weights_resident=False`); a single window larger than the batch budget
(the processor allows images up to 8000 tokens = 32000 patches: 2.1 GiB of transient) exceeds the reserve the
1 M profile models and is the one shape the planner does not cover.

Validation: `tests/test_serving_image_host.py` (the bookkeeping for `t > 1`: runs, flattening, batching, straddling
chunks), `tests/test_multimodal_video_mesh.py` ((2,4) block: the clip's tower features vs the HF fp32 reference
`bringup/references/glm53_flash/MMV_shapes_clip4` per group, streams vs HF `inputs_embeds` with the text rows
bit-exact, prefill + traced decode vs `generate.json` when present), `demo/multimodal_demo.py`
(`GLM53_MM_CLIP_FRAMES=4` / `GLM53_MM_VIDEO=<frames dir>`). The reference: `reference/hf_multimodal_reference.py
--clip-frames 4 --generate 0` (features + embeddings in seconds; `--generate 16` adds the 45-layer CPU generation).
Device runs of the video path are pending a block grant (written device-free on 2026-10-27).

## Served path (vLLM + the TT plugin)

`tt/generator_vllm.py::GLM53FlashForConditionalGeneration` (selected by `TT_MODEL_CLASS_OVERRIDES`; the text class
`GLM53FlashForCausalLM` is untouched, bit for bit) adds vLLM's `SupportsMultiModal` and registers upstream's
`Glm5NextMultiModalProcessor` / `Glm5NextProcessingInfo` (`vllm.models.glm5next`, with the Glm4v dummy-inputs
builder: what upstream's own `Glm5NextForConditionalGeneration` registers), so `vllm serve` accepts `image_url` /
`video_url` chat content, vLLM's processor produces the pixel patches and expands the placeholders, and the plugin
(`model_runner._gather_multi_modal_inputs`) forwards per batch row the item lists `pixel_values` /
`image_grid_thw` / `pixel_values_videos` / `video_grid_thw` plus `mm_placeholder_offsets` / `mm_placeholder_lengths`
to `prefill_forward`. The adapter:

| step | where |
|---|---|
| `initialize_vllm_model`: `GLM53_VLLM_IMAGES` (auto = the class's `supports_multimodal`) builds `VisionRuntime` BEFORE any warm-up capture (a resident tower allocated after a capture could alias a trace's transients); the planner takes the tower's bytes (`vision_tower_bytes`: 1.05 GiB per chip bf16, replicated) as static when resident, else inside the per-request reserve (`vision_transient_bytes`) | `tt/generator_vllm.py`, `tt/memory_plan.py` |
| `warmup_model_prefill` (eager phase): one dummy 288-token image prompt through the tower + the eager stream prefill into slot 0 (`GLM53_VLLM_WARMUP_MM=1`): the tower's programs, the 512-bucket prefill and the scatter ops compile at boot | `_warmup_multimodal` |
| `prefill_forward`: rows with items are parsed (`mm_items`) and checked against the prompt (`feature_rows` on the whole prefix `tokens[i, :end]`) before any slot is taken; the plan is forced EAGER (the per-length traces embed the ids inside the capture); `_prefill_one_mm` encodes the items whose rows intersect `[start, end)` once (`VisionRuntime.encode`: eager, temporaries freed, the tower released in the per-request mode), then per 16 K device chunk either the plain text `prefill` (no vision row in the chunk: bit-identical to text) or `chunk_streams` + `prefill_streams(streams, n, user, start)` with the chunk's rows scattered; a straddling item takes its first rows with the first chunk and the rest with the next (`chunk_scatter`); rows past the chunk end are never written | `tt/generator_vllm.py`, `tt/multimodal.py` |
| decode: unchanged (text tokens) | |

Chunk boundaries: vLLM's scheduler never ends an engine chunk inside a multimodal item (the plugin sets
`disable_chunked_mm_input` with chunked prefill), which is why the catalog's `max_num_batched_tokens` is 32768 (a
video item can reach ~31.9 K tokens: the vLLM processor's 30 000 vision-token cap + timestamps); the adapter walks
a 32 K engine chunk in two 16 K device chunks and tolerates an engine chunk that does end inside an item (the rows
so far are scattered, the next chunk re-derives the item's rows and re-encodes it). A multimodal prompt is served
by the eager prefill whatever its length: the 286-token image prompt pads to the 512 bucket instead of the 512
trace; a traced variant would need per-length `keep` / `img` inputs beside the ids (240 MB bf16 over the 256-2048
ladder on 8 rows) and is the next optimization.

Memory, the 1 M profile (`tt/memory_plan.py`, planner at 8x4 / user-partitioned / 512 MB trace region): the text
profile fits 4 users per row at 1 M with 0.12 GiB to spare; the resident tower (1.05 GiB per chip) does not fit
there (4 users per row reach 944 K), and a per-request tower with the default 4096-patch batch misses by 0.07 GiB
(its 1.13 GB + 272 MiB of transient exceeds the reserve's 1 M whole-prefix copy). With `GLM53_VLLM_VISION_RESIDENT=0`
and `GLM53_VLLM_VISION_BATCH_PATCHES=1024` (one 448x448 window per tower call: 1.13 GB + 68 MiB) the transient hides
under that term and the 1 M x 4 statement stands, at the cost of uploading the tower (1.13 GB per chip, 32 chips)
on every image request. The 128 K profile keeps the tower resident: 28 users per row still fit beside it; at
256 K the resident tower costs 16 -> 12 users per row, at 512 K 8 -> 4. Knobs: `GLM53_VLLM_IMAGES`,
`GLM53_VLLM_VISION_RESIDENT`, `GLM53_VLLM_VISION_BATCH_PATCHES`, `GLM53_VLLM_WARMUP_MM`.

Tests: `tests/test_serving_image_host.py` (host: the plugin contract on fakes, the scatter bookkeeping, the adapter
with a fake mesh model + fake tower, the class declaration against a fake vLLM api, the planner numbers) and
`tests/test_serving_image_mesh.py` ((2,4) block: the adapter driven with the plugin-shaped call for the demo image,
the image straddling a 1024-token chunk boundary and the synthetic clip, against the harness path on the same
model instance). Not covered: multimodal prompts under the per-row wave prefill (refused), prefix caching of
multimodal items (declared off), a traced multimodal prefill.

## Open issues

- The single-chip test yields 3 tokens per ~17-21 min device job (the 1500 s job cap, not a model limit; the
  weight-streamed pass over 45 layers takes 290-450 s); it checkpoints its ids (`GLM53_MM_CKPT`) and a later run
  prefills prompt + those ids (teacher-forced on the device's own tokens) and continues. Logs
  `_logs/agents/vision/single_chip_*.log` (jobs: 3 + 3 + 3 + 3 + 3 + 1 tokens, 1192 / 1017 / 1013 / 1082 / 1281 /
  381 s). The mesh test does the 16 tokens in one job (resident model, traced decode in seconds).

- The sample image (flat shapes on white) is a moderately ill-conditioned input for the tower: HF itself in
  bf16 scores PCC 0.9909 against its fp32 run (test_vision.py documents the gradient image at 0.86). The
  tower bar in `test_multimodal.py` is therefore `min(0.995, floor - 0.03)` as in `test_vision.py`.
- The HF reference runs the tower in fp32 and the text layers in bf16 (the shipped `from_pretrained(bf16)`
  model runs the tower in bf16 too); the TT tower (fp32 activations, bf16 weights) is closer to the fp32 run.
- Multi-image and video prompts are wired (`encode_prompt` takes any order of images, videos and text; the served
  path takes the plugin's per-item lists); their device runs are pending a block grant.
