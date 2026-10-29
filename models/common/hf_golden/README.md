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
| `generate_golden_minimax_m3_1_flash.py` | MiniMaxAI/MiniMax-M3.1-Flash | M3's class, all layers sparse, MXFP8/NVFP4 dequantized |
| `generate_golden_deepseek_v4_flash.py` | deepseek-ai/DeepSeek-V4-Flash | transformers `deepseek_v4`, FP8/FP4 dequantized |
| `generate_golden_deepseek_v4_1_flash.py` | deepseek-ai/DeepSeek-V4.1-Flash-1017 | DeepSeek's reference `inference/model.py` (no transformers `deepseek_v41`), FP8/FP4 dequantized, Engram tables memory-mapped |
| `generate_golden_deepseek_v4_pro_0813.py` | deepseek-ai/DeepSeek-V4-Pro-0813 | same class as Flash |
| `generate_golden_glm_5_2.py` | zai-org/GLM-5.2 | transformers `glm_moe_dsa` |
| `generate_golden_xing4_0_29b_a4b.py` | XingChen-AGI/Xing4.0-29B-A4B | repo remote code (MLA + mHC + MoE) |
| `generate_golden_mimo_v2_6_flash_rl.py` | XiaomiMiMo/MiMo-V2.6-Flash-RL | repo remote code, FP8/MXFP4 dequantized (see below) |
| `generate_golden_mimo_v2_6_pro_rl.py` | XiaomiMiMo/MiMo-V2.6-Pro-RL | Flash-RL's class on the Pro repo (8 qkv TP shards) |
| `generate_golden_glm_5_3_flash.py` | zai-org/GLM-5.3-Flash-BF16 | transformers `glm5_next` (KDA + DSA + mHC), needs transformers >= 5.17 |
| `generate_golden_muse_glimmer_30b.py` | meta-models/Muse-Glimmer-30B | transformers `muse_glimmer` (text backbone), needs transformers >= 5.17 |
| `generate_golden_muse_spark_open.py` | meta-models/Muse-Spark-Open-816B-A42B | transformers `muse_spark` (text backbone, latent MoE), needs transformers >= 5.20 |
| `generate_golden_hy3.py` | tencent/Hy3 | transformers `hy_v3` (GQA + MoE) |
| `generate_golden_hy4_preview.py` | tencent/Hy4-preview | transformers `hy_v4` (DSA + sinks + IHC), needs transformers >= 5.17 |
| `generate_golden_kimi_k2_7_code.py` | moonshotai/Kimi-K2.7-Code | transformers `kimi_k25` (DeepseekV3 text model), INT4 experts dequantized, needs transformers >= 5.17 |
| `generate_golden_kimi_k3.py` | moonshotai/Kimi-K3 | repo remote code (KDA + MLA + AttnRes + latent MoE), MXFP4 experts dequantized, KDA in fla's torch references, needs `fla-core` (CUDA) |
| `generate_golden_k2_horizon.py` | IFM/K2-Horizon-375B-A23B | repo remote code (GQA + partial RoPE + MoE), plain bf16 |
| `generate_golden_nemotron_3_ultra.py` | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 | transformers `nemotron_h` (Mamba-2 + latent MoE + NoPE GQA), needs `--with kernels==0.12.3` |
| `generate_golden_llama_3_1_8b_instruct.py` | meta-llama/Llama-3.1-8B-Instruct | transformers `llama` |
| `generate_golden_qwen3_8_27b.py` | Qwen/Qwen3.8-27B | transformers `qwen3_5` (Gated DeltaNet + gated GQA), torch GDN reference |
| `generate_golden_mistral_medium_3_5.py` | mistralai/Mistral-Medium-3.5-128B | transformers `mistral3` (`ministral3` text model), FP8 weights dequantized |
| `generate_golden_whole_model.py` | Qwen/Qwen3.8-27B, Mistral-Medium-3.5 (`isl10240`, older layout) | whole model, one forward (see below) |

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

**MiMo-V2.6-Flash-RL:** no prefix (wikitext ppl at isl 10240: 4.779 raw, 4.780 chat; no BOS token). The ~525 GB
host peak once noted here came from the pre-direct-read load path: with weights read straight to the GPU and
dequantized there, host memory stays at ~2 GB for the whole run. isl 56320 with `--attn-chunk 1024`: 5.5 min from
cached weights, 107 GB GPU peak, 36 GB on disk. Runs are deterministic, but NOT prefix-consistent across ISLs: the
router's fp32 logit matmul rounds differently by row count (max 1.6e-5 at layer 1), which flips a few top-8 picks
and grows with depth (at isl 10240 vs the first 10240 rows of isl 56320: 96-99.8% of expert picks agree, last-layer
hidden PCC 0.974). Compare routing as sets, with a tolerance.

**MiMo-V2.6-Pro-RL:** same remote code and formats as Flash-RL; qkv_proj is stored as 8 TP shards (Flash: 4).
Prefix: the chat template up to a no-think assistant turn (wikitext ppl at isl 5120 over the same 5000 tokens: 2.087
chat, 2.114 `<|endoftext|>`, 2.125 raw). isl 56320 with `--prefill-chunk 5120 --attn-chunk 512`: 11.4 min, 141.7 GB
GPU peak (use `--attn-chunk 256` for headroom), 54 GB trimmed to 38 layers; its first 10240 rows equal an isl 10240
run with the same flags bit for bit (419/419 tensors). The early sliding layers send almost all attention to the
sink, so their attention output is near 0: compare those with an absolute tolerance, not PCC alone.

**Kimi-K2.7-Code:** `[BOS]` prefix (wikitext ppl at isl 10240 over the same 10100 tokens: 2.897 `[BOS]`, 2.937 raw,
3.200 chat). The tokenizer is remote code on tiktoken, which reads even local files through `blobfile`: add
`--with blobfile` to the overlay. Routed experts are compressed-tensors INT4 (group 32), dequantized bit-identically
to compressed_tensors 0.17 (checked at run time on the first MoE layer). transformers' MLA cache already holds the
latent (normed c_kv and roped k_pe), so it is stored as is. isl 56320 with `--prefill-chunk 5120`: 2.2 min, 54 GB GPU
peak, 64 GB on disk (38 GB trimmed to 28 layers); its first 10240 rows equal the isl 10240 run bit for bit (426/426
tensors).

**K2-Horizon-375B-A23B:** `<|ifm|begin_of_text|>` prefix, the BOS its tokenizer prepends (wikitext ppl at isl 10240
over the same 10100 tokens: 6.266 BOS, 6.278 raw, 6.325 chat). Its router computes the logits in bf16 (stored widened
to fp32), and RoPE rotates dim pairs (j, j + 64) for j < 32 only. isl 56320 with `--prefill-chunk 5120`: 2.2 min,
41 GB GPU peak, 57 GB on disk (36 GB trimmed to 26 layers); its first 10240 rows equal the isl 10240 run bit for bit
(359/359 tensors), ppl 4.146 over all 56318 tokens. Layer 1 writes one outlier channel (max |x| ~1e3 at rms 0.7) and
layers 59-60 blow the residual up to max |x| ~1.1e6: compare those relatively.

**Nemotron-3-Ultra-550B-A55B:** 108 single-mixer blocks (48 Mamba-2, 48 latent MoE, 12 attention without RoPE).
No prefix (wikitext ppl at isl 10240 over the same 10100 tokens: 5.073 raw, 5.075 `<s>`, 5.238 chat). The Mamba-2
kernels come from the kernels hub (`--with kernels==0.12.3`; newer kernels break transformers 5.12.1's loader), so
don't run it with HF_HUB_OFFLINE. transformers runs any Mamba call after the first as a single decode token, so the
script carries the conv and fp32 SSM state across `--prefill-chunk` chunks itself (`mamba_prefill`). Smoke needs
`--num-layers 8`: the cache's length comes from an attention block, the first being block 7. isl 56320 with
`--prefill-chunk 5120`: 2.9 min, 49 GB GPU peak, 102 GB on disk (42 GB trimmed to 39 blocks); its first 10240 rows
equal the isl 10240 run bit for bit (278/278 tensors), ppl 3.345 over all 56319 tokens. Mamba blocks 6 and 13 write
the first outliers (max |x| ~330), blocks 54-69 grow them to ~3.6e3 and the last blocks to ~8.8e3 at block 107.

**MiniMax-M3 at long ISL:** the upstream lightning indexer builds a whole [4, N, N] fp32 score matrix (~105 GB peak at
isl 56320 with its masked copy). `HF_GOLDEN_INDEXER_CHUNK=4096` builds it 4096 query rows at a time (~4 GB); every op
after the projections reads only its own query row, so the trace is bit-identical (checked against the unchunked run
at isl 10240, all 468 tensors). At startup the script checks the chunked forward against the installed transformers'
`MiniMaxM3VLIndexer.forward` on a tiny random indexer and exits if upstream has changed. isl 56320 on one H200:
2.4 min, ~9 GB GPU peak, 50 GB on disk.

**Kimi-K3:** remote code written for transformers 4.56+, run on 5.12 with shims (OutputRecorder alias, the forced
flash_attention_2 put back to `--attn`, a plain causal mask for the selftest's whole-model forward). The remote
KDA layer calls fla's Triton `chunk_kda`; by default it runs fla's own pure-torch references instead
(`naive_kda_lowerbound_gate` + `naive_recurrent_kda`, fp32, the ones fla's tests grade `chunk_kda` against), checked
against `chunk_kda` at startup on CUDA; `HF_GOLDEN_KDA=triton` keeps the kernel. AttnRes blocks are carried between
layers, so no `--prefill-chunk`; the final stage reads `norm` and `output_attn_res_*` through a tuple stage prefix.
A dequantized MoE layer is ~59 GB in bf16. For isl 56320, AttnRes reads run 4096 tokens at a time and `moe_infer`
is replaced by a same-arithmetic version without upstream's extra [N x 16, 3584] copies; on a GPU under 120 GiB
(H100) the routed experts stay in host RAM and are copied to the device per call (`HF_GOLDEN_EXPERTS=auto|device|host`;
needs ~70 GB free host RAM). Run through a uv overlay:

```
uv run --no-project --python python_env/bin/python --with fla-core==0.6.0 --with einops --with blobfile \
    --with compressed-tensors==0.20.0 python generate_golden_kimi_k3.py --selftest --device cuda
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run ... python generate_golden_kimi_k3.py --isl 56320 --attn-chunk 1024
```

**MiniMax-M3.1-Flash:** M3's architecture with every layer sparse, stored MXFP8 (dense linears) + NVFP4 (routed
experts); transformers 5.12 dequantizes neither combination, so convert() does (the script header has the formulas
and the real-checkpoint check). `HF_GOLDEN_SERVING=hopper` instead emulates vLLM serving it on Hopper (PR #63232):
e4m3 KV and index-K caches read back by prefill, fp32 router, and one top-16 block selection per KV head, where
transformers shares the max over the 4 index heads between all heads. That last one is not rounding: it changes which
keys attention sees once N > 2048. It writes `isl<N>-serving-hopper` beside the plain trace; at startup its attention
and indexer are checked against upstream's with quantization off and the selection shared. Chat prefix (wikitext ppl at
isl 5120 over the same 4880 tokens: 11.34 chat, 293 `]~!b[` only, 594 raw; variants of the framing, fp32 compute and
the serving emulation all within 2%). isl 56320 with `--prefill-chunk 5120 --attn-chunk 4096` and
`HF_GOLDEN_INDEXER_CHUNK=4096` on one H100 from cached weights: 2 min, 72 GB GPU peak, 50 GB (33 GB trimmed to 31
layers: 0-11, 20-22, 26-28, 37-39, 50-59); the serving trace needs `--attn-chunk 2048` on an H100 (4096 OOMs): 6 min,
48 GB, 51 GB (34 GB trimmed). Both equal their isl 10240 runs on the first 10240 rows bit for bit (474/474 tensors),
also unchunked. Perplexity rises with context: 21.2 at isl 10240, 61.9 at 56320 (21.7 / 45.1 / 93.5 / 160.4 by
quarter; serving 63.1), not explained by chunking (an unchunked isl 10240 run scores the same); kept as is.

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

**Hy3:** plain GQA + MoE, no prefix (wikitext ppl at isl 10240: 5.61 raw, 5.81 after BOS, 5.88 under its chat
template). isl 56320 on one H200 with `--attn-chunk 4096`: 2.1 min from cached weights, 40 GB GPU peak, 56 GB on disk.
Run from a cache downloaded by commit with `--revision <commit>`: offline, `main` has no ref to resolve.

**Hy4-preview:** transformers' hy_v4 (5.17 - 5.19) rotates the MLA's rope dims half-split, but the checkpoint
expects them interleaved, as SGLang (which Tencent's model card points to) runs it: wikitext ppl at isl 10240 is 199
half-split, 3.91 interleaved. The script rebinds the MLA's rope to interleaved and checks upstream's call sites at
startup (`interleaved_mla_rope`). Its DSA indexer builds [N, 32, N] fp32 scores (~406 GB at isl 56320):
`HF_GOLDEN_INDEXER_CHUNK=2048` scores 2048 query rows at a time, bit-identical (layers 0-3 at isl 10240, all 30
tensors). No prefix (3.914 raw, 3.927 BOS, 3.927 chat). Use the BF16 checkpoint: Hy4-preview-FP8 is a post-hoc
MXFP8 quantization with ~3% weight error per matmul. isl 56320 with `--attn-chunk 1024`: 11.1 min from cached
weights, 123 GB GPU peak, 226 GB on disk (71 GB trimmed to 18 hidden layers).

**Muse-Glimmer-30B:** dense; 3 sliding-window (2048, RoPE) layers per full NoPE layer, gated GQA, sandwich norms,
normed embeddings, logits x output_multiplier then softcapped. BOS prefix (wikitext ppl at isl 10240: 7.02 BOS, 7.11
raw, 7.15 chat; score PPL with the multiplier and softcap applied, or it reads in the billions). isl 56320 with
`--attn-chunk 4096`: 0.9 min from cached weights, 18 GB GPU peak, 41 GB on disk. Its K/V on
sliding layers keep every position (make_cache: DynamicCache(config) would keep only the last 2047), and the first
10240 positions of that trace are bit-identical to a bare isl 10240 run (hidden states and K/V, all 52 layers).

**DeepSeek-V4.1-Flash-1017:** transformers has no `deepseek_v41` (huggingface/transformers#48755), so this runs
DeepSeek's own reference implementation from the repo's `inference/` (pinned with the checkpoint) under the engine:
`build()` wraps its `Transformer`, `stages()` drives its `Block`s, and its TileLang `kernel.py` is replaced by torch
`sparse_attn` / `hc_split_sinkhorn` (checked against the kernels at startup when tilelang is installed) with the
in-graph activation rounding off (the unquantized model, as V4-Flash). Cross-layer state (compressed KV, index keys,
top-k, candidate blocks) passes through the reference's own `shared_attn`; the mHC pre mix is carried per layer. The
two ~98 GB Engram tables are memory-mapped from their shards and only the hashed rows are read. Indexer scores, the
sparse-attention gather and `hc_post` run in token chunks, so isl 56320 fits an H100 (~30 GB of weights per layer).
The selftest compares against one whole, unchunked reference forward; it executes the downloaded reference code. Chat prefix (wikitext ppl at
isl 10240 over the same 10100 tokens: 1.683 chat, 1.751 BOS only, 1.842 raw). Unchunked: CSA2's cross-layer carry is
not supported by `--prefill-chunk`. isl 56320 on one H100 from cached weights: 4 min, 54 GB GPU peak, 95 GB on disk,
ppl 2.069 over all 56272 tokens (1.72 / 1.94 / 2.09 / 2.62 by quarter).

```
uv run --no-project --python python_env/bin/python --with sympy --with pillow \
    python generate_golden_deepseek_v4_1_flash.py --selftest
uv run ... --with tilelang==0.1.12 python generate_golden_deepseek_v4_1_flash.py --isl 56320
```

**Muse-Spark-Open-816B-A42B:** Muse-Glimmer's attention, embedding and logits (plus a weightless o_norm before the
output gate), fp32 residual gates instead of residual adds, and MoE on all 62 layers: 256 experts, top 8, run in a
4096 latent with a norm after each expert. bf16 checkpoint (1.64 TB; gated, manual approval), ~27 GB per layer, so
an H100 needs no expert offload. BOS prefix borrowed from Glimmer, not yet measured. The selftest runs on CPU (no
remote code, no Triton):

```
uv run --no-project --python python_env/bin/python --with transformers==5.20.2 \
    python generate_golden_muse_spark_open.py --selftest
uv run ... python generate_golden_muse_spark_open.py --isl 56320 --attn-chunk 1024
```

**Llama-3.1-8B-Instruct:** `<|begin_of_text|>` prefix, the BOS its tokenizer prepends (wikitext ppl at isl 10240
over the same 10100 tokens: 8.358 BOS, 8.428 raw, 8.623 chat). isl 56320 on CPU (64-core EPYC 9354P) with
`--prefill-chunk 5120 --attn-chunk 4096`: 18.5 min, 4.9 GB host peak, 22 GB on disk; its first 10240 rows equal the
isl 10240 run with the same flags bit for bit (99/99 tensors), ppl 6.842 over all 56318 tokens.

**Qwen3.8-27B:** chat prefix up to a no-think assistant turn, as Flash-Next (wikitext ppl at isl 10240 over the same
10100 tokens: 7.430 chat, 7.445 raw, 8.175 `<|endoftext|>`; no BOS). transformers 5.12's cache holds the Gated
DeltaNet recurrent state in bf16, rounding it at every `--prefill-chunk` boundary and in the trace; the script keeps it
fp32 (`make_cache`), so chunked bf16 runs equal one whole forward bit for bit. It refuses to run with fla /
causal_conv1d installed: the torch GDN reference only. isl 56320 on CPU with `--prefill-chunk 5120 --attn-chunk 4096`:
29.5 min, 5.3 GB host peak, 40 GB on disk; its first 10240 rows equal the isl 10240 run bit for bit (195/195
tensors), ppl 6.400 over all 56264 tokens.

**Mistral-Medium-3.5-128B:** `<s>` prefix, BOS (wikitext ppl at isl 10240 over the same 9600 tokens: 6.271 BOS, 6.408
raw, 6.445 chat with its default Le Chat system prompt). The FP8 e4m3 linears are dequantized by transformers'
`Fp8Dequantize` (per-tensor `weight_scale_inv`), as `from_pretrained` does without an FP8 GPU; activations stay bf16
and the static activation scales go to `metadata.json` unapplied. isl 56320 on CPU with `--prefill-chunk 5120
--attn-chunk 4096`: 143 min (~98 s / layer), 14 GB host peak, 136 GB on disk; its first 10240 rows equal the isl 10240
run bit for bit (267/267 tensors), ppl 4.033 over all 56318 tokens.

**`--device cuda` under torch 2.11:** `torch.cuda.set_device` rejects a device without an index; pass `cuda:0`.

`generate_golden_whole_model.py` is the original script behind the older Qwen3.8-27B and Mistral `isl10240` traces. It loads the whole
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
    input_ids.safetensors                      input_ids [N] int32: the exact prefill ids (prefix + token cache)
    hidden/embed.safetensors                   model.embed_tokens.output
    hidden/layer_XX.safetensors                model.layers.<i>.output (+ sub-layer outputs with --sublayers)
    hidden/final.safetensors                   model.last_hidden_state, logits (last position, fp32)
    cache/layer_XX.safetensors                 past_key_values.layers.<i>.<attr>, routing, indexer top-k, ...
    metadata.json                              revision, versions, dtypes, every stored tensor and what it is
```

Read a trace's prefill ids from `input_ids.safetensors`, not from the token cache: the cache holds the wikitext ids
only, and a model with a prefix (metadata `prefix_ids`) prefilled those first.

`backfill_input_ids.py` adds that file to traces generated before the engine wrote it, rebuilt from metadata.json and
the token cache and checked against the trace's own `hidden/embed.safetensors` (equal ids must have equal embedding
rows, different ids different ones: a prefix shift fails at once, no weights needed). It only adds the file, readable
by all; metadata.json and SHA256SUMS are left as they are. Dry run unless `--write`:

```
python backfill_input_ids.py [<root>] [--only <hf-repo>[/isl<N>]] [--check-tokenizer] [--write]
```

`trim_trace.py` copies a trace keeping the layer hidden states of only some layers, always including decoder
layers 0-4 (embed, final and every layer's cache stay complete), hard-linked when on the same filesystem, with metadata.json's inventory and a `hidden_layers`
entry saying what was kept and why, and a fresh SHA256SUMS. At isl 56320 the 4-stream mHC hidden states dominate:
DeepSeek-V4-Pro-0813 201 GB -> 79 GB keeping 19 of 61 layers, V4-Flash 83 GB -> 40 GB keeping 17 of 43. Keep each
event layer with its input layer, so it can be run alone from golden input:

```
python trim_trace.py <root>/deepseek-ai/DeepSeek-V4-Flash/isl56320 <trimmed-root> \
    --layers 0-5,7,8,25,26,28-30,39-42 --reason "..."
```

`trace_ppl.py` scores a trace's wikitext perplexity from its stored final hidden state: logits = last_hidden_state @
lm_head.T at every position, cross-entropy of each next token, over the same wikitext tokens whatever the prefix
(`--skip`, `--count`). The lm_head is read from the checkpoint; models with logit multipliers or softcaps need them
applied first.

```
python trace_ppl.py <root>/<hf-repo>/isl10240 <checkpoint snapshot> lm_head.weight --skip 1 --count 10100
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
- `--prefill-chunk N` runs every layer as a chunked prefill: N rows at a time, each chunk attending to keys [0, end)
  through the growing cache (HF's multi-step prefill path). Every row-local op then has the same shapes whatever the
  ISL, so traces at any two ISLs that are multiples of N share their prefix bit for bit: MiMo-V2.6-Flash-RL at isl
  10240 vs 56320 with N = 5120, all 48 layers' hidden states, K/V, router logits and expert picks (unchunked, its
  fp32 router logits round by row count and routing drifts). Same ppl (4.414 vs 4.413). Selftested on MiMo, Hy3 and
  Muse-Glimmer; refused for models that hand a top-k from layer to layer (shared DSA indexers).
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
