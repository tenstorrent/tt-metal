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
| 1. `gptq_calib corpus`, `weights` | host | seconds | the tokenizer, the texts below |
| 2. `gptq_calib capture` | card | the model load (~35 s) and 1.4 to 2.2 s per 2048-token sequence | the tile cache; 0.43 GB of disk per 2048-token sequence (24 GB for the set below) |
| 3. `gptq_calib solve` | host, CPU only | about 10 minutes per layer for the set below on a shared 80-core host; 37 layer solves for the two caches | the checkpoint and the capture; about 10 GB of RAM at 43K tokens |
| 4. `variant`, then the layout steps | host | `variant`: 45 s per layer; per cache 2.3 min for the bank layout and 2.6 min for the PF layout | disk for the rewritten expert tensors only (the rest is hard-linked): 19 GiB per bank or PF cache |

```
G="python -m models.demos.inkling2_mini.convert.gptq_calib"
V="python -m models.demos.inkling2_mini.convert.variant"
C=<a work directory>; T=<a tt-metal checkout>

# 1. the calibration set: 45 sequences of 2048 tokens and 2 of 8192 (the table below), then the weights
$G corpus --out $C/ids.pt \
    --chat chat:12:ultrachat.jsonl --ask "code:6:$T/ttnn:.py,.cpp,.hpp:Explain what this code does." \
    --ask "instruct:3:$T/tech_reports:.md:Summarize this document." \
    --text book:pg84.txt@0.2 --text book:pg84.txt@0.6 --text book:pg98.txt@0.2 --text book:pg98.txt@0.6 \
    --text book:pg1400.txt@0.2 --text book:pg1400.txt@0.6 --text book:pg345.txt@0.2 --text book:pg345.txt@0.6 \
    --chat chat:4:oasst.jsonl --chat instruct:6:dolly.jsonl \
    --text code:argparse.py --text code:json/decoder.py,json/encoder.py,csv.py,heapq.py \
    --text code:<clap_builder src/*.rs> --text code:<hashbrown src/*.rs> \
    --text code:$T/tt_metal/impl/program/program.cpp --text code:<$T/tt_metal/impl/allocator/*.cpp>
    # <...>: the files, comma-separated; pgN.txt: the plain text of Project Gutenberg book N
$G corpus --out $C/ids_long.pt --seq 8192 --first 45 --text long:pg1400.txt@0.4 --text long:pg345.txt@0.8
$G weights --ids $C/ids.pt $C/ids_long.pt --out $C/weights.json \
    --share chat=.348,instruct=.126,code=.263,book=.175,long=.088

# 2. the MoE inputs of every layer application, on the card
$G capture --ids $C/ids.pt --out $C/cap --cache $D/tile
$G capture --ids $C/ids_long.pt --out $C/cap --cache $D/tile

# 3. the solves: every expert tensor of layers 0-19 on the BFP4 grid; then layers 6-17 and 19-23 with their
#    BFP8 tensors on the BFP8 grid (--gu8 / --dn8: the layers whose gate_up / down the dtype map keeps BFP8)
$G solve --capture $C/cap --weights $C/weights.json --layers 0-19 --out $C/codes-l8
$G solve --capture $C/cap --weights $C/weights.json --layers 6-17,19-23 --gu8 12-15,17,20-23 --dn8 6-17,19-23 \
    --out $C/codes-def8

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

The calibration set of the measured caches: 47 sequences, 108,544 tokens, none of them a golden or an evaluation
prompt. Chat data goes through the chat template; `--chat` reads one JSON list of messages per line.

| kind | sequences | text | weight |
|---|---:|---|---:|
| chat | 12 | `HuggingFaceH4/ultrachat_200k`, test_sft conversations | 1.1540 |
| chat | 4 | `OpenAssistant/oasst_top1_2023-08-25`, English test threads | 1.1540 |
| instruct | 6 | `databricks/databricks-dolly-15k`: instruction and context as the user turn, response as the answer | 0.7417 |
| instruct | 3 | tt-metal `tech_reports/*.md`, each as "Summarize this document." | 0.7417 |
| code | 6 | tt-metal ttnn sources, each as "Explain what this code does." | 1.1605 |
| code | 6 | raw source: CPython 3.13 `argparse.py`; `json/decoder.py`, `json/encoder.py`, `csv.py`, `heapq.py`; `clap_builder` 4.6.2 and `hashbrown` 0.17.1 `src/*.rs`; tt-metal `impl/program/program.cpp` and `impl/allocator/*.cpp` | 1.1605 |
| book | 8 | raw text of Project Gutenberg 84, 98, 1400 and 345, from 0.2 and 0.6 of each | 1.1611 |
| long | 2 (8192 tokens) | Project Gutenberg 1400 from 0.4 and 345 from 0.8 | 0.5808 |

The weights give each kind a fixed share of the Hessian rows (chat .348, instruct .126, code .263, book .175,
long .088), whatever its token count.

What was measured, and what was not:

- The numbers in "Measured numbers" are those of the caches made this way with an earlier revision of these
  tools: the same solver, with the sequences packed by scripts that `corpus` replaces. Their token-id files were
  not kept. A rebuild gives caches of the same kind; the experts are bit-equal only for the same token sequences.
- Solve, measured: about 10 minutes per layer over the 108,544 tokens, and about 1 hour for 24 layers with
  about 10 GB of RAM over 43K tokens, both with 40 threads on an 80-core host that ran other jobs; 154 s and
  5.8 GB for layer 0 over 4096 tokens with 10 threads on a 12-core host. The RAM of the 108,544-token solve was
  not recorded.
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
