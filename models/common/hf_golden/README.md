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
| `generate_golden_xing4_0_29b_a4b.py` | XingChen-AGI/Xing4.0-29B-A4B | repo remote code (MLA + mHC + MoE) |
| `generate_golden_mimo_v2_6_flash_rl.py` | XiaomiMiMo/MiMo-V2.6-Flash-RL | repo remote code, FP8/MXFP4 dequantized (see below) |
| `generate_golden_glm_5_3_flash.py` | zai-org/GLM-5.3-Flash-BF16 | transformers `glm5_next` (KDA + DSA + mHC), needs transformers >= 5.17 |
| `generate_golden_hy4_preview.py` | tencent/Hy4-preview | transformers `hy_v4` (DSA + sinks + IHC), needs transformers >= 5.17 |
| `generate_golden_whole_model.py` | Qwen/Qwen3.8-27B, Llama-3.1-8B, Mistral-Medium-3.5 | whole model, one forward (see below) |

Not covered yet:
- **DeepSeek-V4.1-Flash**: transformers has no implementation and the repo ships no modeling code.
- **Kimi-K3** (deferred): its remote code needs `fla` Triton kernels (CUDA only), forces flash-attention, and does
  not import under transformers 5.12.

```
python generate_golden_<checkpoint>.py --isl 10240                          # the trace
python generate_golden_<checkpoint>.py --isl 1024 --num-layers 4 --root /tmp/smoke   # a diagnostic, never a grade
python generate_golden_<checkpoint>.py --selftest [--device cuda]           # layer by layer == one HF forward
python generate_golden_<checkpoint>.py --isl 51200 --attn-chunk 4096        # long ISL: attention 4096 query rows at a time
```

**transformers >= 5.17** (GLM-5.3-Flash, Hy4-preview): the repo's `python_env` pins 5.12.1 for tt-metal, so run these
through a uv overlay that adds a newer transformers on top of it for that one command, nothing installed:

```
uv run --no-project --python python_env/bin/python --with transformers==5.19.0 python generate_golden_hy4_preview.py --isl 10240
```

**MiMo-V2.6-Flash-RL:** the script passes its selftest and a full-depth run completes, but that run peaked at ~525 GB
of host memory at isl 256, so layer weights are not all being released yet. Check memory before a long run.

**MiniMax-M3 at long ISL:** the upstream lightning indexer builds a whole [4, N, N] fp32 score matrix (~105 GB peak at
isl 56320 with its masked copy). `HF_GOLDEN_INDEXER_CHUNK=4096` builds it 4096 query rows at a time (~4 GB); every op
after the projections reads only its own query row, so the trace is bit-identical (checked against the unchunked run
at isl 10240, all 468 tensors). At startup the script checks the chunked forward against the installed transformers'
`MiniMaxM3VLIndexer.forward` on a tiny random indexer and exits if upstream has changed. isl 56320 on one H200:
2.4 min, ~9 GB GPU peak, 50 GB on disk.

**DeepSeek-V4 (Flash, Pro-0813) at long ISL:** the indexer scorer holds [N, 64 heads, N // 4] fp32 scores (~200 GB
each at isl 56320). `HF_GOLDEN_INDEXER_CHUNK` builds them a slice of query rows at a time; weights_proj still runs on
the whole sequence, because that bf16 linear rounds differently by row count and per-slice calls flipped indexer
top-k and then MoE routing. Checked identical at isl 10240 on the real models (Flash 573/573 tensors, Pro 822/822),
and against the installed transformers at startup. Eager attention needs `--attn-chunk` too: 1024 on Flash, 256 on
Pro (128 heads); 4096 OOMs on Flash (55 GiB fp32 logits per slice). `HF_GOLDEN_INDEX_SCORES=0` stores the top-k
only, not the scores (~3.2 GB per CSA layer at isl 56320). isl 56320: Flash 5.6 min, 83 GB. The 4 mHC streams make
hidden states large: [N, 4, hidden] per layer, 3.2 GB per Pro layer.

**GLM-5.3-Flash at long ISL:** transformers' torch KDA holds [64 heads, N / 64, 64, 64, 128] fp32 intermediates
(~118 GB each at isl 56320) and the DSA indexer [N, 32 heads, N / 4] fp32 pool scores (~100 GB).
`HF_GOLDEN_KDA_CHUNK=4096` runs KDA 4096 positions at a time, carrying its fp32 recurrent state (the reference's only
cross-chunk dependency); `HF_GOLDEN_INDEXER_CHUNK=2048` scores 2048 query rows at a time, every projection still on
the whole sequence. Both are checked against the installed transformers at startup, and the trace is bit-identical to
the unchunked one (layers 0-8 at isl 10240, all 66 tensors). It needs no prefix: wikitext ppl at isl 10240 is 2.87
raw, 3.06 with `[gMASK]<sop>` and 17.5 under its chat template. isl 56320 on one H200 with `--attn-chunk 2048`: 4.1
min from cached weights, 89 GB GPU peak, 117 GB on disk (70 GB trimmed to 15 hidden layers).

**`--device cuda` under torch 2.11:** `torch.cuda.set_device` rejects a device without an index; pass `cuda:0`.

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

The token cache (`<root>/<hf-repo>/wikitext-2-raw-v1-train.tokens.safetensors`) is built by each model's own
tokenizer and records its fingerprint (sha256 of the fast tokenizer's full serialization) and the text's sha256. It is
reused only when both match and it was built for the same repo; otherwise the run exits and says why. A cache from
before the fingerprint is re-tokenized once and compared. The fingerprint is also in every trace's `metadata.json`.

On a CUDA device each stage's tensors are read with parallel `O_DIRECT` preads into a pinned host buffer, copied to
the device in one transfer, and routed experts are fused there (`direct_read` in `common.py`; bytes unchanged,
checked bit-identical against the safetensors path). safe_open's mmap faults one 4 KiB page at a time, which in a
memory-limited container whose page cache is full ran at ~0.3 GB/s: ~45 s per MiniMax-M3 MoE layer, now ~2 s.
`HF_GOLDEN_DIRECT_READ=0` restores the safetensors path; `HF_GOLDEN_READ_THREADS` sets the read threads (default 8).

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

`trim_trace.py` copies a trace keeping the layer hidden states of only some layers (embed, final and every layer's
cache stay complete), hard-linked when on the same filesystem, with metadata.json's inventory and a `hidden_layers`
entry saying what was kept and why, and a fresh SHA256SUMS. At isl 56320 the 4-stream mHC hidden states dominate:
DeepSeek-V4-Pro-0813 201 GB -> 79 GB keeping 19 of 61 layers, V4-Flash 83 GB -> 40 GB keeping 17 of 43. Keep each
event layer with its input layer, so it can be run alone from golden input:

```
python trim_trace.py <root>/deepseek-ai/DeepSeek-V4-Flash/isl56320 <trimmed-root> \
    --layers 0-5,7,8,25,26,28-30,39-42 --reason "..."
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
- Models with attention sinks (gpt-oss, MiMo, DeepSeek-V4, Hy4) run `--attn eager`, the only kernel that applies the
  sinks.
- `--attn-chunk N` runs the model's own attention on N query rows at a time, so no [heads, N, N] score matrix exists
  whole (needed past ~10k for eager attention, and for MLA models whose head sizes keep sdpa on its math path). Each
  query row's softmax reads only its own row, so traces are bit-identical to unchunked ones (checked at isl 8192 on
  Gemma-4, gpt-oss and Xing4.0, and by every selftest with chunking forced). It does not cover N^2 work outside the
  attention function, such as DSA indexers (for MiniMax-M3's, see `HF_GOLDEN_INDEXER_CHUNK` above).
- A script's `prefix` (literal text, special tokens included) is prefilled before the token cache, recorded as
  `prefix_ids` in `metadata.json`. Instruct checkpoints set it to their chat template up to the assistant turn, so the
  text reads as the model's own reply: raw wikitext is far out of distribution for them (Gemma-4-26B-A4B-it: ppl ~400
  raw, ~22 framed, at 4k tokens). `--prefix ''` turns it off.
- MoE routing near the top-k cut is sensitive to precision (about 1% of expert picks change between bf16 and fp32 on
  Gemma-4-26B). Compare routing and indexer top-k as sets, with a tolerance.
