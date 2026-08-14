# Measured comparison: autoport vs stock demos Gemma 4 31B on BH QuietBox 2

Date: 2026-08-14 UTC
Host: `qb2-120-p02t03` — 2x Blackhole `p300c` (4 chips), 11x10 grid, `MeshShape(1,4)`.

Both sides served through vLLM on the same four chips and benchmarked with a
byte-identical client. This is the first measured p300x2 number for the stock
path that I could find anywhere; TTI publishes only a `theoretical` target.

## What was run

Identical client for both, the exact command the readiness runner emits:

```bash
vllm bench serve --backend vllm --model <served-name> --base-url <url> \
  --endpoint /v1/completions --dataset-name random \
  --random-input-len 128 --random-output-len 128 --num-prompts 8 \
  --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
  --max-concurrency 1 --temperature 0.0
```

| | Autoport | Stock demos |
| --- | --- | --- |
| vLLM model class | `models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM` | `models.demos.gemma4.tt.generator_vllm:Gemma4ForCausalLM` |
| Checkpoint | `google/gemma-4-31B` (base) | `google/gemma-4-31B-it` |
| `max_model_len` | 113,280 | 49,152 (TTI's documented ceiling) |
| `max_num_seqs` | 32 | 1 (TTI `max_concurrency`) |
| `block_size` | 64 | 64 |
| `sample_on_device_mode` | `all` | `decode_only` (TTI spec) |
| `trace_region_size` | 268435456 | 200000000 (TTI spec) |
| Async scheduling | **enabled** | **disabled** — `supports_async_decode: False` |
| Weight precision | BFP8 attention, **BFP4 MLP**, BFP8 LM head, LoFi | **BF16 throughout** |

Both selected the mesh via `MESH_DEVICE=P150x4`, which TTI's own spec requires on
QB2: "the custom p300_x2 descriptor laid the TP collectives over the wrong fabric
links and corrupted decode logits".

## Results

8/8 requests completed, 0 failed, on both sides.

| Metric | Autoport | Stock demos | Autoport advantage |
| --- | ---: | ---: | ---: |
| Benchmark duration | 44.36 s | 69.64 s | 1.57x |
| TTFT, median | 100.91 ms | 106.25 ms | 1.05x |
| TTFT, mean | 1005.70 ms | 111.17 ms | **0.11x (worse)** |
| TTFT, p99 | 6832.38 ms | 134.49 ms | **worse** |
| Decode t/s/u, from mean TPOT | 27.98 | 14.78 | 1.89x |
| Decode t/s/u, from median TPOT | 30.03 | 14.78 | 2.03x |
| Decode t/s/u, from median ITL | 34.11 | 14.81 | 2.30x |
| Output throughput, aggregate | 23.08 tok/s | 14.71 tok/s | 1.57x |
| GPU KV cache | 103,872 tokens | 21,824 tokens | 4.76x |
| Max concurrency at own ceiling | 8.62x @ 113,280 | 3.26x @ 49,152 | — |

Against TTI's published `theoretical` target for this platform
(`model_performance_reference.json`: 46 ms TTFT, 37 t/s/u):

| | Best decode t/s/u | Share of 37 target |
| --- | ---: | ---: |
| Autoport | 34.11 | **92%** |
| Stock demos | 14.81 | 40% |

Neither implementation reaches the target. The stock path reaches 40% of
Tenstorrent's own theoretical figure for its own supported configuration.

## How much of the decode gap is actually the autoport

The 2x decode difference is **not** a like-for-like numerics comparison. Three
contributors, in order of likely size:

1. **Weight precision.** The autoport ran its Stage 08 selected policy: BFP8
   attention, BFP4 MLP, BFP8 LM head, LoFi fidelities. The stock path ran BF16
   for `wqkv`, `o_proj`, `gate_proj`, `up_proj`, and `lm_head` (confirmed in its
   tensor-cache generation log). Decode at batch 1 is weight-bandwidth bound, and
   BFP4 MLP weights are roughly a quarter of BF16. Selecting and accuracy-validating
   that config is a real autoport contribution — 0.92 top-1 / 1.00 top-5 against
   the reference — but it is a different numerical configuration, not the same
   math running faster.
2. **Async decode.** The autoport declares `supports_async_decode: True` and ran
   with async scheduling; the stock path declares `False` and vLLM disabled it.
3. **Parallelisation and trace design.** TP4 layout, collectives policy, split
   device sampling, and trace reuse.

Isolating (1) would need the autoport re-run at BF16, or the stock path at BFP8/BFP4.
Neither was done here.

The KV-capacity result **is** attributable to design: the autoport's hybrid cache
(50 sliding layers at physical 1024 plus 10 full-attention layers at 262,144)
against the stock path allocating a full-length buffer for all 60 layers, which is
the reason TTI documents 49,152 as this platform's ceiling. That difference is
also *conservative* toward the autoport, which carried a 4.8x larger KV pool and a
2.3x longer context while still decoding faster.

## Where the autoport is worse

**Cold-start TTFT.** Mean 1005.70 ms with p99 6832.38 ms against a 100.91 ms
median: the first request pays trace capture. The stock path is flat by
comparison (mean 111.17, median 106.25, p99 134.49) — no capture spike. Over a
long-lived server this amortises to nothing, and steady-state medians are within
5%. For a cold-start-sensitive or short-lived deployment it is a genuine
regression, and the mean TTFT should never be quoted as this model's TTFT
without the median beside it.

## Shared code, so read the comparison narrowly

The autoport is not an independent implementation. Its final serving path imports
from `models/demos/gemma4`: `Gemma4DecoderLayer` (instantiated per layer, with its
`self_attn`, `shared_mlp`, and three layernorms driven directly), eleven symbols
from `tt.attention.operations` (`apply_qkv_projection`, `apply_rope`,
`apply_rope_decode_peruser`, `apply_per_head_norm`, `split_qkv_heads_*`,
`prefill_sdpa_program_config`, `effective_block_size`, and two chunking
constants), `attention.kv_cache.init_kv_cache`, `ccl.{CCLManager, ccl_allreduce,
ccl_allgather}`, `rms_norm.RMSNorm`, `config.{MeshConfig, ModeConfig}`, and
`model_config.Gemma4ModelArgs`.

So the attention math, RoPE, head splitting, SDPA configuration, KV-cache
construction, collectives, and RMSNorm are the *same code* on both sides. What
differs is the assembly: parallelisation strategy, cache design, terminal/LM-head
path, sampling, tracing, and the serving adapter. Attribute the results to that,
not to kernel quality.

## Reproducing

Autoport side, via the readiness runner (writes into
`readiness_vllm/`, which overwrites committed evidence — restore with
`git checkout --` afterwards):

```bash
export TT_GEMMA4_TEXT_VER=gemma4_31b_autoport   # needs the plugin selector, see below
export GEMMA4_31B_AUTOPORT_DIR=$PWD/models/autoports/google_gemma_4_31b
export GEMMA4_31B_TENSOR_CACHE=/home/mvasiljevic/models/tt_cache/gemma4_31b_full
export GEMMA4_31B_VLLM_HOST_SAMPLING_COMPAT=1   # logprob/determinism checks only
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/google_gemma_4_31b --hf-model google/gemma-4-31B \
  --stages serve,benchmark --mesh-device P150x4 --port 8000 \
  --max-num-seqs 32 --max-model-len 113280 --block-size 64 \
  --tt-config '{"sample_on_device_mode":"all","trace_region_size":268435456,"fabric_config":"FABRIC_1D","trace_mode":"all","enable_model_warmup":true}' \
  --additional-server-args='--served-model-name google/gemma-4-31B --async-scheduling' \
  --benchmark-prompt-len 128 --benchmark-output-len 128 \
  --benchmark-num-requests 8 --benchmark-concurrency 1 --no-benchmark-ci-serving
```

Stock demos side, direct api_server with TTI's config:

```bash
export TT_GEMMA4_TEXT_VER=demos GEMMA4_MAX_TOKENS_ALL_USERS=49152
export GEMMA4_PAGE_BLOCK_SIZE=64 MESH_DEVICE=P150x4
export TT_CACHE_PATH=/home/mvasiljevic/models/tt_cache/gemma4_31b_it_demos  # MUST be writable
python -m vllm.entrypoints.openai.api_server \
  --model <gemma-4-31B-it snapshot> --served-model-name google/gemma-4-31B-it \
  --block_size 64 --max_num_seqs 1 --port 8100 --max_model_len 49152 \
  --additional-config '{"tt":{"sample_on_device_mode":"decode_only","trace_region_size":200000000,"fabric_config":"FABRIC_1D","enable_model_warmup":true}}'
```

Reset the devices between runs — see `host_runbook_qb2_p300x2.md`; this host
timed out on device-0 ethernet core 29-25 three times, always on the first mesh
open after a prior process tore down.

Selecting the autoport does **not** require patching the plugin. The TT plugin
already supports `EXTRA_MODELS_DIR`, a directory of bundles each carrying a
`vllm_metadata.json` with `arch` and `main_class`, registered before the built-in
map; `TT_VLLM_BUILTIN_MODELS=0` disables the built-in map entirely. That is the
upstream-intended mechanism and needs no change to `tenstorrent/vllm`.

## Artifacts

Raw `vllm bench serve` results for both sides are the authority for every number
above. They were produced outside the repo to avoid overwriting committed
evidence; the autoport run additionally passed the non-aligned 149-token check
and the logit-determinism check (`token_id:108`, exact top-20, stable across two
runs and three batch positions).
