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
| single chip, 45 layers streamed (`GLM53Model`, fullmodel tensorbin cache), `GLM53_MM_N=3` per job with the checkpointed continuation | first token 785 `'The'` == HF; 12/12 greedy tokens == HF after four jobs (`'The image contains a red circle, a blue square, a'`, every HF token in the TT top-5; jobs 5-6 for tokens 13-16 in progress); step-0 top-5 ids identical to HF's (TT logits 21.375 / 19.125 / 18.625 / 18.375 / 17.125 vs HF 21.75 / 19.0 / 18.625 / 18.5 / 18.0); device argmax == host argmax on every decode step. Prefill pass 417 s, decode passes 389 / 370 s, tower 7.5 s: 1192 s per job, so 3 tokens is the maximum under the 1500 s device-job cap (16 tokens would take ~85 min) |
| mesh (`GLM53_MESH_SHAPE`): first token == HF; traced decode | pending |

## Reuse of the repo's reference implementations (standing rule)

| this branch | sibling | verdict |
|---|---|---|
| `multimodal_streams` merge (host keep/img masks, fp32 multiply + add on the packed streams) | `models/demos/qwen3_vl/tt/common.py::merge_vision_tokens_single_user_ttnn` (`ttnn.scatter` of the device image embeddings at the `image_token_id` rows; torch twin `merge_vision_tokens`) | replace: scatter into the `[T, 4096]` device embedding before `mhc_expand` (no host masks, no readback) |
| `image_features` (tower on device, host readback per image) | `models/demos/qwen3_vl/tt/model.py::DropInVisionTransformer.forward` / `forward_single_user` (per-image loop, device embeddings straight to the merge) | replace together with the merge |
| `tt/vision.py` tower | `models/demos/qwen25_vl/tt/{vision_block,vision_attention,vision_mlp,vision_rmsnorm,rope,patch_merger}.py` + `qwen3_vl/tt/model.py::VisionTransformer` | port target: GLM5Next = the Qwen2.5-VL block structure + per-head q/k RMSNorm + 2x2 downsample conv + LayerNorm/GELU/SwiGLU-10240 merger |
| `prefill_streams` on both models | `qwen3_vl/tt/model.py::Transformer.prepare_inputs_prefill` (the stack takes embeddings) | same pattern, keep |
| `encode_multimodal` | the qwen VL demos' processor-driven prompts | same pattern, keep (`</think>` append is GLM-specific) |
| the generate / decode loops (`GLM53Model.generate`, mesh test, `multimodal_demo.py`) | `models/tt_transformers/tt/generator.py::Generator` via an adapter like `qwen3_vl/tt/generator.py` | replace when the GLM text stack gets a Generator adapter (not a `TTTransformer`: mHC streams, KDA/DSA state) |

## Open issues

- The single-chip test yields 3 tokens per ~20-min device job (the 1500 s job cap, not a model limit); it
  checkpoints its ids (`GLM53_MM_CKPT`) and a later run prefills prompt + those ids (teacher-forced on the
  device's own tokens) and continues, so the 16-token comparison accumulates over jobs; the mesh test does
  the 16 tokens in one job (resident model, traced decode in seconds).

- The sample image (flat shapes on white) is a moderately ill-conditioned input for the tower: HF itself in
  bf16 scores PCC 0.9909 against its fp32 run (test_vision.py documents the gradient image at 0.86). The
  tower bar in `test_multimodal.py` is therefore `min(0.995, floor - 0.03)` as in `test_vision.py`.
- The HF reference runs the tower in fp32 and the text layers in bf16 (the shipped `from_pretrained(bf16)`
  model runs the tower in bf16 too); the TT tower (fp32 activations, bf16 weights) is closer to the fp32 run.
- Multi-image prompts and video are not wired (one image per prompt; `image_features` already splits per
  image and `multimodal_streams` concatenates in prompt order, so several images need only the processor call).
