# hf_golden

Prefill reference outputs ("goldens") from upstream HuggingFace models, so a device run can be PCC'd layer by layer
against something independent. HuggingFace does all the compute: transformers' own model code, or the checkpoint
repo's remote code pinned to a commit. These scripts only orchestrate it.

Needs only `torch`, `transformers`, `safetensors` and `huggingface_hub` (`requirements.txt` pins the versions the
goldens were made with). Nothing here imports tt-metal; the folder works copied out on its own.

## Scripts

One per checkpoint. Each one's header lists every tensor it stores: the key in the `.safetensors` file, the shape,
the dtype, the layers it exists on and what it is.

| Script | Checkpoint | HF implementation |
| --- | --- | --- |
| `generate_golden_gemma_4_31b_it.py` | google/gemma-4-31B-it | transformers `gemma4` |
| `generate_golden_gemma_4_26b_a4b_it.py` | google/gemma-4-26B-A4B-it | transformers `gemma4` (MoE) |
| `generate_golden_gpt_oss_120b.py` | openai/gpt-oss-120b | transformers `gpt_oss`, MXFP4 experts dequantized |
| `generate_golden_minimax_m3.py` | MiniMaxAI/MiniMax-M3 | transformers `minimax_m3_vl` |
| `generate_golden_deepseek_v4_flash.py` | deepseek-ai/DeepSeek-V4-Flash | transformers `deepseek_v4`, FP8/FP4 dequantized |
| `generate_golden_deepseek_v4_pro_0813.py` | deepseek-ai/DeepSeek-V4-Pro-0813 | same class as Flash |
| `generate_golden_glm_5_2.py` | zai-org/GLM-5.2 | transformers `glm_moe_dsa` |
| `generate_golden_whole_model.py` | Qwen/Qwen3.8-27B, Llama-3.1-8B, Mistral-Medium-3.5 | whole model, one forward (see below) |

Not covered yet:
- **DeepSeek-V4.1-Flash**: transformers has no implementation and the repo ships no modeling code.
- **MiMo-V2.6-Flash-RL** (in progress): the script passes its selftest, but a full-depth run peaks at ~525 GB of
  host memory; layer weights are not being released yet.
- **Kimi-K3** (deferred): its remote code needs `fla` Triton kernels (CUDA only), forces flash-attention, and does
  not import under transformers 5.12.

```
python generate_golden_<checkpoint>.py --isl 10240                          # the trace
python generate_golden_<checkpoint>.py --isl 1024 --num-layers 4 --root /tmp/smoke   # a diagnostic, never a grade
python generate_golden_<checkpoint>.py --selftest [--device cuda]           # layer by layer == one HF forward
```

`generate_golden_whole_model.py` is the original script behind the Qwen, Llama and Mistral traces. It loads the whole
model, runs one forward, and writes the older format the device-side grader (`galaxy_prefill_kv_pcc.py`) reads:
`kv_cache/layer_<i>.safetensors` (`key_cache_layer_<i>`, `value_cache_layer_<i>`, or GDN conv/recurrent state) and
`final_hidden.safetensors`. It computes in fp32 and writes to `/mnt/models/agentic-prefill-goldens` by default. Every
other script here uses the layer-by-layer engine and the layout below.

## How it works

`engine.py` holds the loop and `common.py` the pieces under it.

The model is built on the meta device, so nothing is allocated. Each stage (embedding, layer i, final norm,
lm_head) is materialized from only the checkpoint shards holding it, run by the upstream module, recorded with
forward hooks, and sent back to meta. Memory stays one layer deep.

Anything already on disk is used in place, with no hub access needed: a local checkpoint dir, your HF cache, or
`$HF_GOLDEN_SHARED_CACHE` (default `/mnt/models/huggingface/hub`). That covers shards, config, tokenizer and remote
code. A branch such as `main` pins to the commit of the copy on disk; pass `--revision <commit>` to choose another.
Any shard not on disk is downloaded to `--tmp` and deleted after its last use.

A script subclasses `LayerwiseGolden` and overrides only what its model needs:
- what the upstream text model passes each layer (`prepare`, `layer_kwargs`)
- how checkpoint tensors become HF parameters, dequantization included (`convert`)
- which tensors to store (`Recipe`)
- a tiny config for `--selftest`

`--selftest` builds a tiny random checkpoint with every layer kind of the real model. It streams that checkpoint
layer by layer and compares the result with one whole-model `from_pretrained` forward: exact on CPU, within a
tolerance on GPU. Run it first on any new machine.

## Trace layout

```
<root>/<hf-repo>/
  wikitext-2-raw-v1-train.tokens.safetensors   the whole text under this model's tokenizer (built once)
  isl<N>/                                      the first N tokens prefilled
    hidden/embed.safetensors                   model.embed_tokens.output
    hidden/layer_XX.safetensors                model.layers.<i>.output (+ sub-layer outputs with --sublayers)
    hidden/final.safetensors                   model.last_hidden_state, logits (last position, fp32)
    cache/layer_XX.safetensors                 past_key_values.layers.<i>.<attr>, routing, indexer top-k, ...
    metadata.json                              revision, versions, dtypes, every stored tensor and what it is
```

Naming:
- Tensor names are the HF module path relative to the text backbone, plus `.output`, so a reader can find the
  producing module upstream.
- Cache state is named after the HF cache object: `past_key_values.layers.<i>.<attr>`.
- When a shape depends on the layer kind (sliding vs full, ...), `metadata.json` maps each kind to its shape.

Defaults:
- `--root` is `$HF_GOLDEN_ROOT`, else `./generated/agentic-prefill-goldens`. The text file
  `wikitext-2-raw-v1-train.txt` must be in the root, or passed with `--text`.
- Compute is bf16 on CPU and GPU (`--compute-dtype float32` is available). Storage is bf16, except router logits,
  routing weights and logits, which are stored fp32. Those are compute-dtype values widened to fp32, not values
  recomputed in fp32.
- Sliding-window layers store K/V for every position. The device needs only the last window.
- Models with attention sinks (gpt-oss, MiMo, DeepSeek-V4) run `--attn eager`, the only kernel that applies the
  sinks.
- MoE routing near the top-k cut is sensitive to precision (about 1% of expert picks change between bf16 and fp32 on
  Gemma-4-26B). Compare routing and indexer top-k as sets, with a tolerance.
