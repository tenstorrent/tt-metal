# Inkling-2-Mini on one Blackhole p100a

Batch-1 decode (and chunked prefill) of
[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single Blackhole p100a card
(28 GB DRAM, no Ethernet, 1x1 mesh).

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
unembed BFP8, router and residual queries fp32. The weights take about 20.5 GiB.

Code map: `convert/` checkpoint -> weight cache, `tt/` device modules (`model.py` wires them,
`generator.py` traces the decode step, `prefill_model.py` chunked prefill), `tests/` per-module checks
against the Hugging Face reference, `demo/` the demo, golden check and perf harness.

## Setup

Build tt-metal and activate its python env as usual. The scripts take their data from these variables
(or the matching command-line flags):

| variable | what |
|---|---|
| `INKLING_CKPT` | the Hugging Face checkpoint directory (safetensors, config.json, tokenizer) |
| `INKLING_MODE` | `default` or `fast`: which expert weights load (see "Run modes"; `--fast` in the demo) |
| `INKLING_CACHE` | the converted weight cache the model loads (overrides the mode's) |
| `INKLING_TILE_CACHE` | a cache with the experts as tiles, for the chunked prefill and its test (overrides the mode's) |
| `INKLING_PF_GATE_BF16` | `0`: prefill keeps the loop gate messages and outputs fp32 (default `1`: bf16, faster) |
| `INKLING_GOLDEN` | CPU golden outputs (per-prompt hidden states and logits), for the golden check and perf gate |
| `INKLING_PERF_DIR` | where perf.py appends its results and keeps its baseline |
| `INKLING_REF_PYTHON` | a python with transformers >= 5.24 (Inkling2), for the HF-reference tests |
| `INKLING_CONFIG` | the model config.json for those tests (default `$INKLING_CKPT/config.json`) |

## Run modes

Two sets of expert weights, picked with `--fast` (demo) or `INKLING_MODE=fast` (every script).
They differ only in how many expert layers are 4-bit; the rest of the model is the same.

| mode | experts | decode cache | prefill cache | decode 4K / 128K | KL vs bf16 model |
|---|---|---|---|---:|---:|
| default | own3-def8: GPTQ, BFP4/BFP8 as `convert/default_dtype_map.json` | `own3-def8-pf` | `own3-def8-tile` | 14.6 / 19.9 ms | .0067 |
| fast | own3-l8: GPTQ, layers 0-19 all BFP4 | `own3-l8-pf` | `own3-l8-tile` | 13.1 / 18.5 ms | .0096 |

The caches live in `/work/inkling-bringup/cache/` (`modes.py`). Prefill uses the same LoFi K-blocked
defaults in both modes, with the loop gate messages stored bf16 (`INKLING_PF_GATE_BF16=0` turns that off).

## Speculative decode (off by default)

`tt/spec_generator.py` (`SpecGenerator.generate_rows`, `generate_rows_from`): a small drafter proposes 2 tokens, one
verify pass checks them with the full model, and the accepted ones are kept. The tokens are the same as plain greedy
decode. Nothing turns it on: the demo and the perf harness run the plain step. It needs the drafter
weights (`python -m models.demos.inkling2_mini.convert.mtp`, or `$INKLING_MTP_CACHE`).

Measured on one card, ms/token, 128 to 192 greedy tokens, plain = the default plain step in the same harness:

| run | plain | spec | tokens / pass |
|---|---|---|---|
| 4K, code prompt, default mode | 15.15 | 13.29 | 2.23 |
| 4K, prose prompt, default mode | 15.15 | 12.10 | 2.35 |
| 4K, code prompt, fast mode | 13.68 | 13.20 | 1.76 |
| 4K, prose prompt, fast mode | 13.68 | 10.62 | 2.40 |
| 130K real prompt, chunked prefill, default mode | 20.52 | 22.01 | 1.44 |
| 130K real prompt, chunked prefill, fast mode | 18.99 | 19.62 | 1.48 |
| 130K real prompt, prefill feeds the drafter, default mode | 20.52 | 20.97 | 1.49 |
| 130K real prompt, prefill feeds the drafter, fast mode | 18.99 | 18.70 | 1.54 |

A chunked prefill gives the drafter no context for the prompt unless it is told to: set
`Inkling2Prefill.drafter = <the DSparkDrafter>` before `prefill()` and every chunk also writes the drafter's context
(`DSparkDrafter.commit_chunk`). Without it fewer drafts are accepted: the 4K prompts with the drafter context
cleared give 1.7 to 1.9 tokens per pass and 15.3 to 16.4 ms/token; with the context from the prefill they give the
same 2.35 to 2.44 tokens per pass as after decode. At 130K the feed lifts 3-row passes from 1.58 to 1.68 tokens, and
a 3-row pass there costs 1.7 plain steps, so spec is about level with plain. Spec decode also runs without the
expert prefetch (the drafter's programs and the prefetch ring do not fit in L1 together).

## Convert the weights

Host only, one layer at a time (peak host RSS ~3 GB; the checkpoint is ~19 GB, the cache ~21 GB):

```
python -m models.demos.inkling2_mini.convert --ckpt $INKLING_CKPT --out $INKLING_TILE_CACHE
python -m models.demos.inkling2_mini.convert.relayout_sticks --src $INKLING_TILE_CACHE --dst $INKLING_CACHE \
    --layout bank --dtypes BFP4,BFP8
```

The first step writes a cache with every expert as tiles; the chunked prefill reads that one
(`INKLING_TILE_CACHE`). A rerun converts only the units (global, L00 ... L23) whose settings changed; the model
refuses a cache whose manifest is not complete.

The second step rewrites the expert tensors of every layer (BFP4, BFP8 and mixed) in the bank layout, where each
expert column lives in one DRAM bank and the decode MoE (`tt/fused/moe_bank.py`) reads it from its own bank. It
costs disk only for the rewritten tensors (every other file is hard-linked) and checks each one against its
source. It is ~1.9 ms/token faster than the stick layout (`--layout stick`, read by the older fused kernel).
Both layouts sum the down projection expert by expert, a different order than the tile path:
the BFP8 layers round differently (token-only 127 -> 125 on the short gate), with equal code1500 results.

Sticks and bank layouts are laid out for the card's DRAM bank count, 7 on a p100a (`convert/stick.py`
`N_BANKS`). The loader refuses a cache made for another count; for a p150 (8 banks) use `relayout_sticks --n-banks 8`.

ttnn opens the card even for host-side tensor conversions. To convert or run the host-only tests without
the card, point it at a mock cluster:

```
export TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml
```

## Run the demo

```
python -m models.demos.inkling2_mini.demo.demo --prompt "Write a haiku about a mountain lake at dawn." \
    --max-ctx 4096 --gen 64 --skip-app-trace --perf-pos
```

The prompt goes through the model's chat template and the answer is printed. Without `--prompt`, the demo
generates the golden prompts (`--golden`) and compares prompt 0 with the golden greedy tokens. It also
checks that a traced decode step replays bit-equal and matches eager, and measures decode speed at
`--perf-pos` positions.

## Run the tests

```
pytest models/demos/inkling2_mini/tests models/demos/inkling2_mini/convert
```

- Without `INKLING_REF_PYTHON` (and transformers < 5.24 in the tt-metal env) the HF-reference tests skip;
  without the mode's caches (or `INKLING_CACHE`) and `INKLING_GOLDEN` the full-model and prefill tests skip.
- Host only (with the mock cluster above): `tests/test_convert_pack.py` (converter packing vs the modules),
  `tests/test_perf_gate.py`, `convert/test_cache.py`. `convert/test_dequant.py` runs in the reference python.
- `tests/test_model.py`: full model vs the golden, teacher-forced per-layer PCC >= 0.99, logits PCC >= 0.98.
- `tests/test_prefill.py` takes over 5 minutes on a cold kernel cache and needs a tile-layout cache.
- `tests/test_attention_fast.py::test_use_time` is a benchmark (marker `models_performance_bare_metal`).

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

## Expected numbers

Measured on one p100a (7 DRAM banks), traced decode, median of 5 x 10 steps:

| | 4K context (pos 2048) | 128K context (pos 122880) |
|---|---|---|
| decode step | ~40.2 ms (~24.9 tok/s) | ~47.1 ms (~21.2 tok/s) |

Quality against the CPU golden (bf16 HF reference), four short prompts (143 positions): teacher-forced
per-layer PCC >= 0.999 at the last layer applications, token-only greedy top-1 agreement at 124 positions,
logits PCC >= 0.99. Chunked prefill of
the 1478-token code prompt matches the teacher-forced golden at logits PCC 0.9999.
