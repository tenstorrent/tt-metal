# Inkling-2-Mini on one Blackhole p100a

Batch-1 decode (and chunked prefill) of
[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single Blackhole p100a card.

## Hardware

| | |
|---|---|
| card | one Blackhole p100a: 28 GB DRAM in 7 banks, no Ethernet, opened as a 1x1 mesh |
| DRAM use | weights ~20.5 GiB; the far KV cache takes 19,584 B per token of `--max-ctx` |
| context | `--max-ctx 196608` (the demo's default) leaves ~2.4 GiB of DRAM free; 327680 is the largest that fits |
| host | conversion peaks at ~3 GB RSS; the checkpoint is ~19 GB on disk and each weight cache ~21 GB |

Every number in this file was measured on a p100a. The expert layouts are made for the card's DRAM bank count
(see "Weights and cache conversion"); no other card was run.

## The model

| | |
|---|---|
| size | ~26B parameters, ~5B active per token |
| layers | 24; layers 6-17 run three times per token (48 layer applications) |
| attention | local (1024-token sliding window) and global (layers 5, 11, 17, 23); learned relative bias, no RoPE |
| residual | block attention residual over depth, cross-pass messages and message gates in the loop, short convs |
| MoE | 128 routed experts, top-8, sigmoid router with bias, softcapped SwiGLU |
| hidden / vocab | 2304 / 200058 (padded to 200064 on device) |
| source weights | NVFP4 experts in layers 0-19, MXFP8 in 20-23, bf16 / fp32 elsewhere |

On the card the experts are BFP4 or BFP8 (per tensor, `convert/default_dtype_map.json`), attention and the
unembed BFP8, router and residual queries fp32.

Code map: `convert/` checkpoint -> weight cache (and the GPTQ rounding of the experts), `tt/` device modules
(`model.py` wires them, `generator.py` traces the decode step, `prefill_model.py` chunked prefill,
`spec_generator.py` speculative decode), `tests/` per-module checks against the Hugging Face reference and the
full-model test, `demo/` the demo, golden check and perf harness.

## Setup

Build tt-metal and activate its python env as usual. The scripts take their data from these variables
(or the matching command-line flags); no path is built in:

| variable | what |
|---|---|
| `INKLING_CKPT` | the Hugging Face checkpoint directory (safetensors, config.json, tokenizer, mtp.safetensors) |
| `INKLING_CACHE_DIR` | the directory that holds the run modes' weight caches and the drafter's (default `model_cache/inkling2_mini`) |
| `INKLING_MODE` | `default` or `fast`: which expert weights load (see "Opt-in modes"; `--fast` in the demo) |
| `INKLING_CACHE` | the converted weight cache the model loads (overrides the mode's) |
| `INKLING_TILE_CACHE` | a cache with the experts as tiles, for the chunked prefill and its test (overrides the mode's) |
| `INKLING_GOLDEN` | CPU golden outputs (per-prompt hidden states and logits), for the full-model test, the golden check, the demo and the perf gate |
| `INKLING_PERF_DIR` | where perf.py appends its results and keeps its baseline |
| `INKLING_REF_PYTHON` | a python with transformers >= 5.24 (Inkling2), for the HF-reference tests and `tests/make_golden.py` |
| `INKLING_CONFIG` | the model config.json for those tests (default `$INKLING_CKPT/config.json`) |
| `INKLING_SPEC`, `INKLING_SPEC_MAX_CTX`, `INKLING_MTP_CACHE`, `INKLING_PF_GATE_BF16` | see "Opt-in modes" |

## Weights and cache conversion

Get the checkpoint (`huggingface-cli download thinkingmachines/Inkling-2-Mini --local-dir $INKLING_CKPT`), then
convert it. Host only, one layer at a time:

```
D=$INKLING_CACHE_DIR
python -m models.demos.inkling2_mini.convert --ckpt $INKLING_CKPT --out $D/tile
python -m models.demos.inkling2_mini.convert.relayout_sticks --src $D/tile --dst $D/bank \
    --layout bank --dtypes BFP4,BFP8
python -m models.demos.inkling2_mini.convert.pf_layout --src $D/bank --tiles $D/tile --dst $D/pf
export INKLING_TILE_CACHE=$D/tile INKLING_CACHE=$D/pf
```

1. `convert` writes a cache with every expert as tiles; the chunked prefill reads that one. A rerun converts only
   the units (global, L00 ... L23) whose settings changed; the model refuses a cache whose manifest is not complete.
2. `relayout_sticks` rewrites the expert tensors of every layer in the bank layout, where each expert column lives
   in one DRAM bank and the decode MoE (`tt/fused/moe_bank.py`) reads it from its own bank. It costs disk only for
   the rewritten tensors (every other file is hard-linked) and checks each one against its source.
3. `pf_layout` regroups the same tiles so that the tensor prefetcher can stream the experts of a layer ahead of
   its step. Decode loads the bank or the PF cache; with a PF cache the expert prefetch is on.

The bank and PF layouts are laid out for the card's DRAM bank count, 7 on a p100a (`convert/stick.py`
`N_BANKS`). The loader refuses a cache made for another count (`relayout_sticks --n-banks 8` writes one for 8
banks).

The converter rounds each expert weight to the nearest BFP value. The caches the two run modes name
(`own3-def8-*`, `own3-l8-*`) hold the same tensors with the expert weights rounded by GPTQ instead; the next
section makes them. Without them, set `INKLING_CACHE` and `INKLING_TILE_CACHE` to the converter's output as
above: every script and test takes them.

ttnn opens the card even for host-side tensor conversions. To convert or run the host-only tests without
the card, point it at a mock cluster:

```
export TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml
```

## GPTQ-rounded experts: the run modes' caches

GPTQ rounds the weights of an expert one input column at a time and moves each column's rounding error onto the
columns not yet rounded, so that the expert's output on calibration inputs changes least. The values stay on the
card's BFP grid (one exponent per 16 outputs at one input), so nothing changes in the kernels:

| cache | expert dtypes | expert tensors rounded by GPTQ |
|---|---|---|
| `own3-def8` | as `convert/default_dtype_map.json`: the plain cache's dtypes, bytes and speed | all 48 (22 BFP4, 26 BFP8) |
| `own3-l8` | layers 0-19 all BFP4; layers 20-23 BFP8 | the 40 of layers 0-19; layers 20-23 are the plain cache's |

Four steps make both from the tile cache `$D/tile` above. Only the capture runs on the card.

| step | runs on | time | needs |
|---|---|---|---|
| 1. `gptq_calib data`, `corpus`, `weights` | host | a minute | the tokenizer, network for `data`, the source files below |
| 2. `gptq_calib capture` | card | the model load (~35 s) and 1.4 to 2.2 s per 2048-token sequence | the tile cache; 0.43 GB of disk per 2048-token sequence (24 GB for the set below) |
| 3. `gptq_calib solve` | host, CPU only | 2 to 3 minutes per layer with 40 threads on an 80-core host; 37 layer solves for the two caches | the checkpoint and the capture; 9 to 15 GB of RAM per process |
| 4. `variant`, then the layout steps | host | `variant`: 45 s per layer; per cache 2.3 min for the bank layout and 2.6 min for the PF layout | disk for the rewritten expert tensors only (the rest is hard-linked): 19 GiB per bank or PF cache |

```
G="python -m models.demos.inkling2_mini.convert.gptq_calib"
V="python -m models.demos.inkling2_mini.convert.variant"
C=<a work directory>
T=<tt-metal at commit d1a10279368>; A=$T/tt_metal/impl/allocator
PY=<CPython 3.13 Lib>; CLAP=<clap_builder 4.6.2>; HASH=<hashbrown 0.17.1>     # the table below says which

# 1. the public texts, the calibration set (47 sequences, 108,544 tokens: the table below), the weights
$G data --out $C/data
B="--text book:$C/data/pg"
$G corpus --out $C/ids1.pt --chat chat:12:$C/data/ultrachat.jsonl \
    --ask "code:6:$T/ttnn/cpp:.py,.cpp,.hpp:Explain what this code does." \
    --ask "instruct:3:$T/tech_reports:.md:Summarize this document." \
    ${B}84.body.txt#0.2 ${B}84.body.txt#0.6 ${B}98.body.txt#0.2 ${B}98.body.txt#0.6 \
    ${B}1400.body.txt#0.2 ${B}1400.body.txt#0.6 ${B}345.body.txt#0.2 ${B}345.body.txt#0.6
$G corpus --out $C/ids2.pt --first 29 --reasoning-effort none \
    --chat chat:4:$C/data/oasst.jsonl --chat instruct:6:$C/data/dolly.jsonl \
    --text code:$PY/argparse.py --text code:$PY/json/decoder.py \
    --text code:$CLAP/src/derive.rs --text code:$HASH/src/map.rs \
    --text code:$T/tt_metal/impl/program/program.cpp --text code:$A/allocation_request.cpp,$A/allocator.cpp
$G corpus --out $C/ids_long.pt --seq 8192 --first 45 \
    --text long:$C/data/pg1400.txt@0.4 --text long:$C/data/pg345.txt@0.8
$G weights --ids $C/ids1.pt $C/ids2.pt $C/ids_long.pt --out $C/weights.json \
    --share chat=.348377,instruct=.125949,code=.262755,book=.175260,long=.087668

# 2. the MoE inputs of every layer application, on the card
for f in ids1 ids2 ids_long; do $G capture --ids $C/$f.pt --out $C/cap --cache $D/tile; done

# 3. the solves: every expert tensor of layers 0-19 on the BFP4 grid; then layers 6-17 and 19-23 with their
#    BFP8 tensors on the BFP8 grid (--gu8 / --dn8: the layers whose gate_up / down the dtype map keeps BFP8)
OMP_NUM_THREADS=40 $G solve --capture $C/cap --weights $C/weights.json --layers 0-19 --threads 40 \
    --out $C/codes-l8
OMP_NUM_THREADS=38 $G solve --capture $C/cap --weights $C/weights.json --layers 6-17,19-23 \
    --gu8 12-15,17,20-23 --dn8 6-17,19-23 --threads 38 --out $C/codes-def8

# 4. the tile caches, then the decode layouts as in "Weights and cache conversion"
$V --base $D/tile --out $D/own3-l8-tile --experts-from codes:$C/codes-l8 --layers 0-19
$V --base $D/tile --out $D/own3-def8-tile --experts-from codes:$C/codes-def8 --layers 6-17,19-23 \
    --take "$D/own3-l8-tile:L0[0-5]/experts.*" --take "$D/own3-l8-tile:L18/experts.*"
for m in own3-l8 own3-def8; do
    python -m models.demos.inkling2_mini.convert.relayout_sticks --src $D/$m-tile --dst $D/$m-bank \
        --layout bank --dtypes BFP4,BFP8
    python -m models.demos.inkling2_mini.convert.pf_layout --src $D/$m-bank --tiles $D/$m-tile --dst $D/$m-pf
done
```

With `INKLING_CACHE_DIR=$D` the two run modes then find `own3-def8-pf`, `own3-def8-tile`, `own3-l8-pf` and
`own3-l8-tile`. `variant` reads every file it writes back and refuses a tensor that differs from its codes
(`gptq_exact` in the manifest), and records where each tensor came from.

The calibration set: 47 sequences in three files, none of them a golden or an evaluation prompt. Nothing is drawn
at random; the order of the rows and files decides. Chat data goes through the chat template (`ids2.pt` with
`reasoning_effort` "none"); `--text` sequences are raw text.

| kind | sequences | text | weight |
|---|---:|---|---:|
| chat | 12 | `HuggingFaceH4/ultrachat_200k`, test_sft, rows 0-25 | 1.1540 |
| code | 6 | tt-metal `ttnn/cpp` at commit `d1a10279368`: the first source files over 4000 bytes in path order, each as "Explain what this code does." | 1.1605 |
| instruct | 3 | tt-metal `tech_reports/*.md` at that commit, each as "Summarize this document." | 0.7417 |
| book | 8 | Project Gutenberg 84, 98, 1400 and 345 (`N-0.txt` of the mirror), text between the marker lines, 2048 tokens from 1/5 and 3/5 of its tokens | 1.1611 |
| chat | 4 | `OpenAssistant/oasst_top1_2023-08-25`, test, English threads from row 190 on (rows below are held out) | 1.1540 |
| instruct | 6 | `databricks/databricks-dolly-15k`, English rows from row 59 on: instruction and context as the user turn, response as the answer | 0.7417 |
| code | 6 | raw source, the first 2048 tokens of: `argparse.py` of Debian 13's `libpython3.13-stdlib` 3.13.5-2+deb13u5 (CPython 3.13.5 plus Debian's gettext patch); `json/decoder.py` of CPython 3.13.5; `clap_builder` 4.6.2 `src/derive.rs`; `hashbrown` 0.17.1 `src/map.rs`; tt-metal `program.cpp`; tt-metal `allocation_request.cpp` + `allocator.cpp` | 1.1605 |
| long | 2 (8192 tokens) | Project Gutenberg 1400 from 0.4 and 345 from 0.8 of the file's characters | 0.5808 |

The shares give each kind a fixed part of the Hessian rows, whatever its token count, and the weights above.

What a rebuild reproduces (checked for the caches the numbers in this file were measured on):

- Token ids: the same. The id files of the measured run were not kept, its capture was. On every one of the
  108,544 positions the reference model's first-layer router picks at least 5 of the 8 experts that capture holds
  for the position (98.6 to 99.2 % of all picks per sequence; another text gives 17 to 33 %). The weights file is
  identical.
- Solve: the same codes, bit for bit, from the same captured inputs with the same thread counts (`--threads` and
  `OMP_NUM_THREADS` 40 and 38 as above, torch 2.15, the same 80-core aarch64 host): all 37 layer solves were
  run again with the code of this folder and compared.
- The thread count and the machine matter. With 16 threads, 752 of the 170 million values of one layer-0 tensor
  changed. On a 12-core x86 host about 6 % of the values of layers 0-5 did. The output error of each tensor on
  the calibration inputs stayed the same (within 0.02 %): such a cache is as good a rounding, but it is not
  the measured cache. With those six x86 layers in place of the measured ones, the KL to the bf16 model is
  0.0110 against 0.0109, and the KL between the two caches 0.0034 (4,068 evaluation positions; the experts'
  rounding emulated on the CPU reference model, every other weight bf16).
- Caches: the same bytes. `variant` wrote the 6 tensors of three layers again from the kept codes, byte-equal to
  the files in the caches.
- Capture: not repeated. The captured inputs depend on the prefill's numerics, and the measured caches' capture
  ran before the prefill's faster defaults went in (K-blocked LoFi projections, bf16 gate messages). `capture`
  uses the defaults of the day, so a capture made now differs in the low bits, and some codes with it.
- Not measured: a rebuild from nothing (a new capture with today's prefill, every solve, both caches) against
  the measured caches. By the times in this section it takes about 4 minutes on the card for the capture, about
  1.5 hours for the 37 solves on the 80-core host (5 to 6 minutes per solve on a 12-core host) and about 40
  minutes for `variant` and the layouts. Its KL to the bf16 model then needs a card run over an evaluation set;
  no tool for that is in this folder.
- Solve, measured with 40 threads on the 80-core host: 78 to 180 s per layer with the host otherwise idle
  (about 1.5 hours for the 37 solves in sequence, half that in two processes); about 10 minutes per layer when
  the host ran other jobs. 9 to 15 GB of RAM per solve process over the 108,544 tokens.
- Against plain rounding, on the calibration inputs, the output error of the BFP8 tensors was 0.23 to 0.46 times
  (gate_up) and 0.45 to 0.68 times (down) with GPTQ. KL to the bf16 model, same bytes: 0.0086 with plain rounding,
  0.0067 for `own3-def8`.
- `gptq_codes.py` also reads the GPTQ package of Lottolabs' Inkling-2-Mini P150 release (`--experts-from
  lotto:DIR`); the caches above do not use it.

## Run the demo

```
python -m models.demos.inkling2_mini.demo.demo --prompt "Write a haiku about a mountain lake at dawn." \
    --max-ctx 4096 --gen 64 --skip-app-trace --perf-pos 2048
```

The prompt goes through the model's chat template and the answer is printed. Without `--prompt`, the demo
generates the golden prompts (`--golden`, default `$INKLING_GOLDEN`) and compares prompt 0 with the golden
greedy tokens. It also checks that a traced decode step replays bit-equal and matches the eager step, and
measures decode speed at the `--perf-pos` positions. While it starts it prints one progress line per phase on
stderr (open device, weights, traces).

The same runs as tests:

```
pytest models/demos/inkling2_mini/demo/demo.py                  # all four
pytest models/demos/inkling2_mini/demo/demo.py -k golden        # default arguments, prompt 0 against the golden
pytest models/demos/inkling2_mini/demo/demo.py -k "prompt or fast or spec"
```

Each one skips when its weight cache, the golden prompts, the checkpoint or the drafter weights are not there.

## Serve with vLLM

`tt/generator_vllm.py` is the model class for the vLLM TT plugin
([tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin), upstream vLLM 0.32): an OpenAI
server (`/v1/chat/completions`, `/v1/completions`, streamed or not) for one request at a time. Install vLLM and
the plugin into the tt-metal python env as the plugin's README says (`source docs/install-vllm-tt.sh`), then:

```
INKLING_CKPT=<checkpoint dir> models/demos/inkling2_mini/vllm/serve.sh [more vllm serve arguments]
```

Any OpenAI client then works with base URL `http://127.0.0.1:8000/v1` (any API key) and model `Inkling-2-Mini`.
`vllm/serve.sh` is the launch line below with its switches as environment variables: `HOST`, `PORT`, `CTX`
(`--max-model-len`), `SPEC=0` (the plain class), `FAST=1`, `PREFILL=<tokens>` (128; see "Chunked prefill" below). What
it runs with `SPEC=0`:

```
export INKLING_VLLM_PREFILL=128
export TT_MODEL_CLASS_OVERRIDES=Inkling2ForCausalLM=models.demos.inkling2_mini.tt.generator_vllm:Inkling2ForCausalLM
vllm serve models/demos/inkling2_mini/vllm --tokenizer $INKLING_CKPT --served-model-name Inkling-2-Mini \
    --max-model-len 196608 --max-num-seqs 1 --shutdown-timeout 60 \
    --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
    --additional-config '{"tt": {"worker_l1_size": 1411136, "trace_region_size": 134217728, "sample_on_device_mode": "all"}}'
```

| argument | what it is for |
|---|---|
| `models/demos/inkling2_mini/vllm` | a small config.json in place of the checkpoint's: the transformers release vLLM 0.32 pins does not know its model type. The tokenizer and the chat template are the checkpoint's (`--tokenizer`) |
| `TT_MODEL_CLASS_OVERRIDES` | registers the class with the plugin (the plugin itself is not changed) |
| `--max-model-len` | the context: the far KV cache is allocated for it (19,584 B per token), as `--max-ctx` in the demo |
| `--max-num-seqs 1` | the model is batch 1 and keeps one sequence of state on the card |
| `--shutdown-timeout` | lets the engine close the card on Ctrl-C (vLLM kills it at once otherwise) |
| `--reasoning-parser`, `--tool-call-parser` | vLLM's parsers for the model's reply format: the reasoning comes apart from the answer, also when streamed (that needs both) |
| `worker_l1_size`, `trace_region_size` | the device options `tt/device.py` opens the card with for the demo; the plugin opens the card here |
| `sample_on_device_mode` | the adapter returns the token: the on-device argmax for a greedy request, its own draw from the logits (temperature, top-k, top-p, seed) for a sampled one. Without it vLLM's host sampler does all sampling, and sorts the 200K vocabulary for every token of a request with top-k or top-p (about 13 ms a token) |

`"optimizations": "performance"` in the same tt config picks fast mode; `$INKLING_MODE`, `$INKLING_CACHE` and
`$INKLING_CACHE_DIR` work as in the demo.

- Without the chunked prefill (below) the prompt is fed by the traced decode step token by token: time to the
  first token is about the prompt length divided by the decode speed. A request that continues the conversation
  on the card feeds only its new tokens; vLLM's own prefix cache and chunked prefill are off.
- Requests with penalties, logprobs, `min_p`, `logit_bias` or structured output are sampled by vLLM's host
  sampler (the plugin routes them there).
- Greedy output equals the demo's tokens (golden prompt 0, 64 tokens).

Speculative decode (see "Opt-in modes") runs through the plugin's model-owned drafter contract. Select the
other class, give the traces more room and add the speculative config:

```
export TT_MODEL_CLASS_OVERRIDES=Inkling2ForCausalLM=models.demos.inkling2_mini.tt.generator_vllm:Inkling2SpecForCausalLM
# (536870912 without the chunked prefill: serve.sh with PREFILL=0)
vllm serve ... (as above, with "trace_region_size": 335544320) --no-async-scheduling \
    --speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 2}'
```

- The plugin's `propose_draft_tokens` call runs the draft pass (and commits the pass before it), a `[1, 3]` decode
  call runs the verify pass, and a plain decode call the plain step. The drafter declines above
  `modes.spec_max_ctx()` and when its cost model expects no gain.
- Greedy requests speculate, and sampled ones that the adapter samples (`sample_on_device_mode`): each verify
  row's token is one draw of the adapter's sampler, and a draft is kept only when the draw is that token. The
  replies are the plain class's, token for token: greedy, and sampled for the same seed.
- Requests for vLLM's host sampler (penalties, logprobs, ...) run plain steps; vLLM refuses `min_p` and
  `logit_bias` on a speculative launch.
- It needs the drafter weights (`convert.mtp`) and DRAM for them; when they are missing it says so and decodes
  plain.

Chunked prefill (`PREFILL=<tokens>` in `serve.sh`, i.e. `INKLING_VLLM_PREFILL`; on by default from 128 tokens,
`PREFILL=0` or an empty `PREFILL=` turns it off): a new prompt of at least that many tokens runs through the
chunked prefill instead of token by token, and so does a prompt that continues the conversation on the card with
at least that many new tokens (a long paste): the steps feed up to a multiple of 128, then the prefill goes on from
there. The reply after such a prompt comes from the prefill's numerics, not the token-by-token feed's. The process
keeps the one PF-layout decode cache and reads the decode's attention weights (no second copy:
`tt/fused/pf_pages.py` copies a layer's projection tiles for the one matmul that uses them); the prefill's expert op reads its expert tensors in place (`gate_up_extra` /
`down_extra`; a build without them rebuilds the plain tensors per layer with `tt/pf_tiles.py`, about 10 ms a
layer). The prefill's logits are bit-equal to a TILE-cache process's with the same passes.

Measured through vLLM on one p100a, first token, streamed, greedy, warm (the first prompt of a new pass size
compiles its kernels: 8.4 s for 8,424 tokens):

| launch | `--max-model-len` | tokens per pass | 8,424-token prompt | 33,000-token prompt | reply |
|---|---:|---:|---:|---:|---:|
| token by token (`PREFILL=0`) | any | | 126.5 s | about 500 s | 64 / 77 tok/s |
| plain (`SPEC=0`) | 196608 | 4096 | 2.7 s | 16.6 s | 62.1 to 62.7 tok/s |
| speculative (the default launch) | 196608 | 1024 | 7.1 to 7.2 s | 28.9 s | 77.6 to 77.8 tok/s |

A conversation that goes on (8,192-token prompt, 24-token replies, `--max-model-len 196608`), first token:

| launch | + 4,096-token paste | then + 8,192-token paste | fed token by token |
|---|---:|---:|---:|
| speculative | 5.8 s | 7.9 s | about 64 s and 128 s |
| plain, passes of 1024 (`INKLING_VLLM_PREFILL_PASS=1024`) | 5.7 s | 7.6 s | |

- It costs DRAM: 0.41 GiB for the prefill's own tensors, 0.26 GiB per 1024 tokens of a pass and room beside it
  (1 GiB beside 8192 tokens, less for a smaller pass), and with speculative decode the verify steps' 0.74 GiB
  after it. The server picks the largest pass size that fits at start and says what it uses, or how much is
  short: speculative decode at 196608 gets passes of 1024 (1.31 GiB free before the verify steps, 0.77 after),
  plain 4096. `INKLING_VLLM_PREFILL_PASS=<tokens>` caps it: two launches with the same passes give the same
  tokens, other pass sizes round otherwise (1024 against 4096 passes: the 8.4K reply parts at token 37).
- The prefill's programs need a 48 KB kernel-config ring (`serve.sh` sets it): speculative greedy 77.2 to
  77.3 tok/s against 77.5 to 77.7 with the 64 KB ring. `PREFILL=0` keeps the 64 KB ring.
- A continued conversation in the speculative launch: the prefill does not write the drafter's context of the
  new tokens (its drafter pass hung the card in one such case, not understood yet), so the reply after a paste
  decodes slower (48 to 53 tok/s over 24 tokens against 78 after a new prompt). Its tokens equal the plain
  launch's.
- Decode keeps its speed: speculative decode runs with it (the prefill also writes the drafter's context of a new
  prompt), and the decode traces and the expert prefetch stay across a prefill (their L1 is freed and taken back
  at the same addresses): no slow steps after a prefill.
- The reply after a prefill is the demo's prefill reply, not the token-by-token one: for that prompt they part at
  token 30 (in a TILE-cache process too).

Measured on one p100a, default mode, `--max-model-len 196608`, at 4K positions (plain = `Inkling2ForCausalLM`,
spec = `Inkling2SpecForCausalLM`; the same tokens in both columns of a row):

| | vLLM plain | vLLM spec | demo |
|---|---:|---:|---:|
| greedy, tok/s (golden prompt 0, 300 tokens, streamed) | 64.2 to 64.7 | 75.7 | 66.2 with a token read per step, 68.8 without |
| one-sentence chat answer, greedy (66 tokens), s after the first token | 0.98 | 0.82 | |
| sampled, tok/s (temperature 0.8, top-k 40, top-p 0.95, two seeds) | 60, 60 | 66 to 68, 59 to 60 | |
| time to first token, 25-token prompt | 0.37 s | 0.38 s | |
| time to first token, 8,424-token prompt | 126.5 s (66.6 tok/s) | 127.3 s (66.2 tok/s) | |
| greedy, tok/s after that prompt (64 tokens) | 62.5 | 74.9 | |
| start to `/health` 200 (weights in the page cache) | 39 to 70 s | 82 to 113 s | |
| DRAM free while serving | 2.84 GiB | 0.89 GiB | |

## Run the tests

```
pytest models/demos/inkling2_mini/tests models/demos/inkling2_mini/convert
```

| test | what | needs |
|---|---|---|
| `tests/test_model.py` | end to end accuracy: the full model against the CPU golden on two prompts. Teacher-forced per-layer PCC >= 0.99, logits PCC >= 0.98, golden top-1 in the top-5 at >= 98 % of the positions; token-only logits PCC >= 0.98 | card, weight cache, `INKLING_GOLDEN` |
| `tests/test_prefill.py` | chunked prefill against the golden | card, tile cache, `INKLING_GOLDEN`; over 5 minutes on a cold kernel cache |
| `tests/test_attention*.py`, `test_moe*.py`, `test_residual.py`, `test_router*.py`, `test_head*.py`, `test_fused*.py`, `test_pf_*.py` and the other module tests | one device module or kernel each against the Hugging Face modules or a torch reference | card; some need `INKLING_REF_PYTHON` or the weight cache |
| `tests/test_mtp*.py`, `test_verify_rows.py`, `test_moe_rows.py` | the drafter and the verify pass of speculative decode | card, weight cache, drafter weights, `INKLING_CKPT` |
| `tests/test_convert_pack.py`, `tests/test_perf_gate.py`, `tests/test_startup.py`, `convert/test_cache.py`, `convert/test_gptq.py` | host only (with the mock cluster above) | nothing (`test_gptq.py::test_corpus`: the tokenizer in `INKLING_CKPT`) |
| `convert/test_dequant.py` | NVFP4 / MXFP8 unpacking against transformers | `INKLING_REF_PYTHON` |

- A test skips when what it needs is missing: without `INKLING_REF_PYTHON` (and transformers < 5.24 in the
  tt-metal env) the HF-reference tests skip; without a weight cache and `INKLING_GOLDEN` the full-model and
  prefill tests skip.
- `tests/test_attention_fast.py::test_use_time` is a benchmark (marker `models_performance_bare_metal`).
- The golden: `$INKLING_REF_PYTHON models/demos/inkling2_mini/tests/make_golden.py $INKLING_CKPT $INKLING_GOLDEN`
  runs the stock transformers model in bf16 on the CPU and writes the reference tensors. Host only; it needs about
  100 GB of RAM (`prompt0 prompt1` as arguments write only what `test_model.py` and the demo read).

## Run perf

```
python -m models.demos.inkling2_mini.demo.perf --golden $INKLING_GOLDEN --results $INKLING_PERF_DIR \
    --label my-change [--long code1500 prose8k] [--decode 4096 131072]
```

Each part (golden gate, decode at 4K / 32K / 128K, long-prompt prefill) runs in its own process and prints
one JSON line, appended to `results.jsonl`. The gate fails below absolute bars (forced per-layer PCC 0.99,
logits PCC 0.98, top-1 90 %, top-5 98 %) and, if `baseline.json` exists, on a regression against it.
`--set-baseline` writes the baseline only from a run that passes the gate. Set `INKLING_CARD_LOCK` to a lock
file each part should hold, and `INKLING_CARD_RESET` to a command to run after a part times out.

## Measured numbers

One p100a, batch 1, the GPTQ-rounded caches of the two run modes. Decode: the traced step, median of 5 x 10
steps, token read not included. Prefill: one traced call over the whole prompt, second replay.

| | default mode | fast mode |
|---|---:|---:|
| decode step at position 2048 (4K context) | 14.6 ms (68 tok/s) | 13.0 ms (77 tok/s) |
| decode step at position 122880 (128K context) | 19.8 ms (51 tok/s) | 18.2 ms (55 tok/s) |
| chunked prefill, 8424-token prompt | 1.60 s (5280 tok/s) | 1.58 s (5340 tok/s) |
| chunked prefill, 32768 tokens | 7.27 s (4510 tok/s) | 7.19 s (4560 tok/s) |

Quality against the CPU golden (bf16 Hugging Face model), four short prompts, 143 positions:

| | default mode | fast mode |
|---|---:|---:|
| teacher-forced per-layer PCC, lowest of the 48 applications | 0.9973 | 0.9973 |
| teacher-forced logits PCC, lowest prompt | 0.99997 | 0.99997 |
| teacher-forced top-1 agreement | 98.6 % | 98.6 % |
| token-only greedy top-1 agreement | 130 / 143 | 131 / 143 |
| KL vs the bf16 model | .0067 | .0096 |
| chunked prefill, 1478-token prompt, teacher-forced: logits PCC | 0.99991 | 0.99991 |
| chunked prefill, 8424-token prompt: logits PCC | 0.9884 | 0.9867 |

## Opt-in modes

Each of these is off unless asked for.

### Fast mode

`--fast` in the demo or `INKLING_MODE=fast` in every script loads a second set of expert weights. The two sets
differ only in how many expert layers are 4-bit; the rest of the model is the same.

| mode | experts | decode cache | prefill cache |
|---|---|---|---|
| default | own3-def8: GPTQ, BFP4/BFP8 as `convert/default_dtype_map.json` | `own3-def8-pf` | `own3-def8-tile` |
| fast | own3-l8: GPTQ, layers 0-19 all BFP4 | `own3-l8-pf` | `own3-l8-tile` |

The caches are sub-directories of `$INKLING_CACHE_DIR` (`modes.py`). Fast mode is about 1.5 ms per token faster
and less exact (the tables above). Prefill uses the same LoFi K-blocked defaults in both modes.

### Speculative decode

A small drafter proposes 2 tokens, one verify pass checks them with the full model, and the accepted ones are kept.
The tokens are the same as plain greedy decode; text the drafter predicts well comes out faster.

| where | how |
|---|---|
| demo | `--spec` or `INKLING_SPEC=1`: each prompt also runs through spec decode; prints `spec_equals_plain`, tok/s |
| code | `tt/spec_generator.py`: `SpecGenerator.generate_rows`, `generate_rows_from` |

It needs the drafter weights: `python -m models.demos.inkling2_mini.convert.mtp` writes them once to
`$INKLING_CACHE_DIR/mtp`, where both run modes find them (`$INKLING_MTP_CACHE` points elsewhere), and
`$INKLING_CKPT/mtp.safetensors` at run time.

Measured on one card, ms/token, 128 to 192 greedy tokens, plain = the plain step in the same harness:

| run | plain | spec | tokens / pass |
|---|---|---|---|
| 4K, code prompt, default mode | 15.15 | 13.29 | 2.23 |
| 4K, prose prompt, default mode | 15.15 | 12.10 | 2.35 |
| 4K, code prompt, fast mode | 13.68 | 13.20 | 1.76 |
| 4K, prose prompt, fast mode | 13.68 | 10.62 | 2.40 |
| 130K real prompt, prefill feeds the drafter, default mode | 20.65 | 16.92 | 1.91 |
| 130K real prompt, prefill feeds the drafter, fast mode | 19.02 | 17.84 | 1.68 |

These rows were measured before the stash steps and the expert prefetch beside the drafter went in (they took the
plain step beside the drafter from 15.46 to 14.80 ms at 4K) and were not timed again in that harness.

- DRAM: a drafter that covers N positions keeps 4,608 B per position (two of its three layers read the last 1,032
  positions only and keep rings), beside the far KV's 19,584 B per token of `--max-ctx`.
  `spec_generator.build(..., spec_ctx=N)` sizes it; past N the tapped model runs its quiet step with the same
  tokens, so spec decode does not lower the context limit. `modes.SPEC_MAX_CTX` (143,360;
  `INKLING_SPEC_MAX_CTX`) is the N a caller should use. The demo's `--spec` covers its whole `--max-ctx`, which
  fits up to about 150K in default mode.
- A chunked prefill gives the drafter no context for the prompt unless it is told to: set
  `Inkling2Prefill.drafter = <the DSparkDrafter>` before `prefill()` and every chunk also writes the drafter's
  context (`DSparkDrafter.commit_chunk`). A process that runs that prefill has to load the TILE cache; spec tokens
  equal plain there too, but its expert step runs a verify pass row by row, so spec gains nothing in that process
  today.

### Prefill gate messages in fp32

`INKLING_PF_GATE_BF16=0`: the chunked prefill keeps the loop gate messages and outputs fp32 (default: bf16).
The 8424-token prefill then takes 1.65 s against 1.60 s.
