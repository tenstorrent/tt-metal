<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-S-2.1 on a TT-QuietBox 2

[`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) (117.6B parameters, 8.45B active, 1M-token
context) as an OpenAI-compatible vLLM server on four Blackhole ASICs (a TT-QuietBox 2 or four P150 cards).

## Status (2026-10-13)

The first version of this branch (2026-10-06, commit `8a556f0422e`) vs now, both measured on 2026-10-13 with the
same test, `demo/perf_direct.py` (no vLLM: device time plus reading the token back; the first version's model code run
with this tree's tt-metal build). DFlash numbers: decode over a 2048-token answer.

| | First version | Now | Speedup |
|---|---:|---:|---:|
| Batch-1 decode, 128 / 8K-token input | 18.6 / 18.3 tok/s | 90.0 / 84.8 tok/s | 4.8x / 4.6x |
| Batch-1 TTFT, 128-token prompt | 199 ms | 60.5 ms | 3.3x |
| Batch-1 TTFT, 8K-token prompt | 10.95 s | 0.72 s | 15x |
| Batch-32 decode, 128 / 8K-token input | 12.3 / 11.1 tok/s/user | 32.6 / 25.1 tok/s/user | 2.7x / 2.3x |
| Batch-32 TTFT (last of 32 users), 128 / 8K-token prompts | 6.41 / 350 s | 0.42 / 23.4 s | 15x / 15x |
| Batch-1 DFlash, AIME24 / real text 128-8K | 43.1 / 21.5-51.7 tok/s | 181.3 / 137.6-215.5 tok/s | 4.2x / 2.7-6.4x |
| Batch-32 DFlash, real text 128-8K | - (batch 1 only) | 29.8-38.0 tok/s/user | |

The first version prefilled a batch-32 request one prompt at a time (no packed prefill); the test does the same for
it.

Against the targets (50% of speed of light, measured without vLLM, tables below):

- Met: batch-1 TTFT at 128 tokens (60.5 ms vs 66 ms) and batch-32 decode at every length (25-33 vs 18-20
  tok/s/user).
- Unhittable on this machine: each needs less time than work that cannot be removed already takes.
  - Batch-1 decode without DFlash (85-90 vs 152-158 tok/s): needs 6.3-6.6 ms per token; weight reads at the best
    measured 357 GB/s (4.5 ms), 96 all-reduces (1.3 ms), attention (1.1 ms) and LM head (0.4 ms) already take 7.3 ms.
  - Batch-1 TTFT from 1K tokens up (139 vs 66 ms at 1K): each chip reads all 64 local experts' weights every layer
    (~42 ms), plus prefill collectives (~20 ms) and attention (~7 ms).
  - Batch-32 TTFT (11.6 vs 1.60 s at 4K): prefill collectives alone take ~58 ms per 4K tokens, so for 32 prompts
    they exceed the target at every length (0.46 / 0.93 / 1.86 / 3.7 s vs 0.39 / 0.78 / 1.60 / 3.31 s at 1K-8K). At
    128 tokens the 32 prompts are one 4K-token prefill: 58 ms of collectives plus 31 ms of expert matmuls vs 66 ms.
- Batch-1 DFlash speculative decoding over a 2048-token answer, this run: above the batch-1 decode target at 128 /
  1K / 4K (175 / 216 / 204 vs 158 / 157 / 155 tok/s), below it at 2K / 8K (142 / 138 vs 156 / 152). A round costs the same as on 2026-10-10 (draft 2.3 ms + check 16.5-19.2 ms); what moves
  is acceptance, because near-tie token flips change the answer between runs: 2K accepted 1.9 drafts per round here
  vs 3.2 on 2026-10-10, when all five 2048-token cells were above target (188 / 200 / 194 / 202 / 160 tok/s).
- Batch-32 DFlash (new; in the model, not yet in the server) is 17-22% faster than plain batch-32 decode over a
  2048-token answer: 38.0 / 37.1 / 34.5 / 33.6 / 29.8 vs 32.6 / 30.5 / 29.5 / 28.6 / 25.1 tok/s/user at 128 / 1K /
  2K / 4K / 8K. The gain is smaller than at batch 1 because batch-32 decode already reads each expert's weights once
  for all 32 users: checking 96 rows (3 per user) touches more of each chip's 64 experts (~62 vs ~46 if tokens routed
  at random), does 3x the per-row attention work and costs 68-81 ms, about 2x a decode step (31-40 ms), while a round
  commits 2.7-2.9 tokens per user (1.7-1.9 drafts accepted).

## Results

### Accuracy

Laguna's next-token predictions vs the original model in fp32, over the AIME24 prompt (235 tokens, or cut to 128)
plus a fixed 100-token answer; batch 32 is the worst of 32 identical users. "top-1, traced" is the same top-1 check
through the traced decode path the server runs (token picked on device). `tests/test_accuracy.py`, 2026-10-13.

| Prompt (AIME24) | Batch | top-1 | top-5 | top-100 | top-1, traced | PCC |
|---|---:|---:|---:|---:|---:|---:|
| 235 tokens | 1 | 0.98 | 1.00 | 1.00 | 0.99 | 0.97 |
| 235 tokens | 32 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 |
| 128 tokens | 1 | 0.96 | 1.00 | 1.00 | 0.97 | 0.97 |
| 128 tokens | 32 | 0.97 | 1.00 | 1.00 | 0.97 | 0.96 |
| Bar | | >= 0.90 | >= 0.98 | 1.00 | >= 0.90 | >= 0.95 |

### Performance

Target = 50% of speed of light (SoL), the roofline limit from [All About Transformer
Inference](https://jax-ml.github.io/scaling-book/inference/) computed by `demo/roofline.py`. Measured with
`demo/perf_direct.py`: no vLLM, device time plus reading the token back, 2026-10-13. Batch 32: 32 prompts arrive
together; TTFT is the last user's; total decode throughput is 32x the per-user speed.

Decode, tok/s per user (measured / target):

| Input tokens | Batch 1 | Batch 32 |
|---:|---:|---:|
| 128 | 90.0 / 158 | 32.6 / 20 |
| 1,024 | 86.8 / 157 | 30.5 / 20 |
| 2,048 | 86.5 / 156 | 29.5 / 19 |
| 4,096 | 85.4 / 155 | 28.6 / 19 |
| 8,192 | 84.8 / 152 | 25.1 / 18 |

Time to first token (measured / target):

| Input tokens | Batch 1 | Batch 32 |
|---:|---:|---:|
| 128 | 60.5 ms / 66 ms | 0.42 s / 0.066 s |
| 1,024 | 139 ms / 66 ms | 2.90 s / 0.39 s |
| 2,048 | 238 ms / 66 ms | 5.83 s / 0.78 s |
| 4,096 | 386 ms / 66 ms | 11.6 s / 1.60 s |
| 8,192 | 716 ms / 103 ms | 23.4 s / 3.31 s |

DFlash speculative decoding, batch 1: decode tok/s over the first 2048 generated tokens (normal decode is 85-90 tok/s
at these lengths). The server runs DFlash for one request at a time (`--max-num-seqs 1`); batch-32 DFlash (below)
runs in the model only so far. A round drafts 15 tokens, then checks up to the first 5 in Laguna in one step (a 3-row
check when 2 or fewer are checked) and keeps the matching ones plus one of Laguna's own. It checks fewer when the
draft is unsure (once the product of its top-1 probabilities drops below 0.2), which reads fewer expert weights.
18-22 ms per round (draft 2.3-2.4 ms, check 15.9-19.2 ms) vs 11.1-11.8 ms per token for normal decode, so DFlash wins
when more than ~0.9 drafts per round are accepted:

| Prompt | Input tokens | Decode tok/s (drafts accepted per round) | Target |
|---|---:|---:|---:|
| AIME24 | 235 | 181.3 (2.7) | |
| Real text | 128 | 174.8 (2.5) | 158 |
| Real text | 1,024 | 215.5 (3.7) | 157 |
| Real text | 2,048 | 141.5 (1.9) | 156 |
| Real text | 4,096 | 203.6 (3.4) | 155 |
| Real text | 8,192 | 137.6 (1.8) | 152 |

Target = the batch-1 decode target.

DFlash speculative decoding, batch 32 (`tt/dflash_batched.py`): 32 users with different real-text prompts (other
stretches of the same reports) each draft 7 tokens, then one Laguna step checks 96 rows: every user's own next token
plus the 64 drafts the draft model is most confident in (product of its top-1 probabilities). Decode tok/s per user
until the users average 2048 generated tokens (users run at different speeds: 965-4346 tokens each), next to plain
batch-32 decode:

| Input tokens | Batch-32 DFlash | Plain batch-32 decode | ms per round (draft + check) | Drafts accepted per round |
|---:|---:|---:|---:|---:|
| 128 | 38.0 | 32.6 | 76.6 (8.9 + 67.5) | 1.91 |
| 1,024 | 37.1 | 30.5 | 77.6 (9.0 + 68.3) | 1.88 |
| 2,048 | 34.5 | 29.5 | 82.4 (11.6 + 70.4) | 1.85 |
| 4,096 | 33.6 | 28.6 | 83.4 (9.2 + 74.0) | 1.80 |
| 8,192 | 29.8 | 25.1 | 90.0 (9.0 + 80.7) | 1.68 |

At 8K the run stopped at an average of ~1,870 tokens, when the fastest user reached the 4,096 positions the test
keeps past the prompt.

Real text = a "summarize this document" request over a technical report. With the AIME24 answer fed in, the device
draft accepts 2.6-2.7 drafts per round, as the draft model does on CPU (2.64): the acceptance rate is the draft
model's own. Single runs vary by up to ~30%, since the output (and so the acceptance) shifts with near-tie tokens
(real text at 2K over 2048 tokens: 194 tok/s at 3.2 drafts per round on 2026-10-10, 142 at 1.9 on 2026-10-13).

## Quick start

Set up once:

```bash
git clone --branch jerrywangTT/laguna-s --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal
export MODEL_DIR="$PWD/models/demos/laguna"
sudo ./install_dependencies.sh            # only if this host has never built tt-metal
"$MODEL_DIR/setup_vllm.sh"                 # builds tt-metal (1-3 h fresh) and the vLLM environment (30-45 min)
"$MODEL_DIR/.venv/bin/hf" auth login       # after accepting the model's terms on Hugging Face
"$MODEL_DIR/.venv/bin/hf" download poolside/Laguna-S-2.1   # 235 GB; otherwise downloaded at first start
```

Start, check and stop the server (port 8000):

```bash
"$MODEL_DIR/serve_vllm.sh"            # ready when ~/laguna-logs/latest.log prints "Application startup complete"
curl -fsS http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "poolside/Laguna-S-2.1", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "user", "content": "Write a Python function that checks whether a number is prime."}]}'
"$MODEL_DIR/serve_vllm.sh" stop       # also resets every Tenstorrent ASIC in the system (tt-smi -r)
```

A start takes ~3 min (the first also converts the weights, ~20 min). For answers without thinking, send
`"chat_template_kwargs": {"enable_thinking": false}`.

## Reproduce the results

From the repository root, with no server running.

### Accuracy test

```bash
REPO=$PWD MODEL_DIR=$PWD/models/demos/laguna
# once: fp32 reference scores from the original model (CPU only, ~5 min, ~30 GB of memory)
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python -m models.demos.laguna.tests.gen_streamed_reference --dtype fp32 \
  --output generated/laguna_reference/readiness_aime24_chat_s.refpt \
  --save-logits generated/laguna_reference/Laguna-S-2.1-aime24-logits.pt
# the tests (~6 min): batch 1, then batch 32 (worst user); each prints top-1, top-5, top-100, traced top-1 and PCC
# and fails if any is below its bar
cd /tmp && env -u TT_METAL_HOME PYTHONPATH=$REPO \
  $MODEL_DIR/.venv/bin/python -m pytest -s $MODEL_DIR/tests/test_accuracy.py
```

For the 128-token prompt, make its reference once and point the tests at it:

```bash
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python -m models.demos.laguna.tests.gen_streamed_reference --dtype fp32 \
  --prompt-len 128 --output generated/laguna_reference/readiness_aime24_chat_s_p128.refpt \
  --save-logits generated/laguna_reference/Laguna-S-2.1-aime24-p128-logits.pt
cd /tmp && env -u TT_METAL_HOME PYTHONPATH=$REPO \
  LAGUNA_REFERENCE_TOKENS=$REPO/generated/laguna_reference/readiness_aime24_chat_s_p128.refpt \
  LAGUNA_REFERENCE_LOGITS=$REPO/generated/laguna_reference/Laguna-S-2.1-aime24-p128-logits.pt \
  $MODEL_DIR/.venv/bin/python -m pytest -s $MODEL_DIR/tests/test_accuracy.py
```

### Perf test

```bash
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/perf_direct.py   # performance tables, ~10 min
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/perf_direct.py --modes dflash --dflash-prompt aime,text --dflash-tokens 2048   # DFlash table
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/perf_direct.py --modes dflash32 --dflash-tokens 2048 --input-lens 128   # batch-32 DFlash, one length per run
PYTHONPATH=$REPO $MODEL_DIR/.venv/bin/python $MODEL_DIR/demo/roofline.py      # SoL and targets, no device
```

The first-version column of the status table: the same script against commit `8a556f0422e`'s `models/` tree (with
this tree's `demo/perf_direct.py` and `demo/roofline.py` copied in), ahead of this repository on `PYTHONPATH`, run
from outside the repository:

```bash
mkdir -p /tmp/laguna-first && git archive 8a556f0422e models | tar -x -C /tmp/laguna-first
cp $MODEL_DIR/demo/perf_direct.py $MODEL_DIR/demo/roofline.py /tmp/laguna-first/models/demos/laguna/demo/
ln -s $REPO/tech_reports /tmp/laguna-first/tech_reports
cd /tmp && PYTHONPATH=/tmp/laguna-first:$REPO $MODEL_DIR/.venv/bin/python \
  /tmp/laguna-first/models/demos/laguna/demo/perf_direct.py           # and --modes dflash --dflash-prompt aime,text
```

## Serving options

Set in front of `serve_vllm.sh`; experimental ones also need `LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`.

| Feature | Variables | Notes |
|---|---|---|
| DFlash speculative decoding | `TT_LAGUNA_DFLASH=1` | experimental; draft [`Laguna-S-2.1-DFlash`](https://huggingface.co/poolside/Laguna-S-2.1-DFlash); 1 greedy request |
| N-gram speculative decoding | `TT_LAGUNA_SPEC_DECODE=1` | experimental; 1 greedy request; not measured on this version |
| Prefix caching | `TT_LAGUNA_PREFIX_CACHE=1` | experimental; 1 request |
| Concurrent requests | `LAGUNA_MAX_NUM_SEQS=N` | N up to 32; shared context pool (below) |

Context per request (one ~1.3M-token pool shared by all requests):

| Concurrent requests | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Context per request, if all equal | 1,048,576 (model limit) | 653K | 326K | 162K | 80K | 39K |

Only the 12 full-attention layers keep every past token; the 36 sliding-window layers keep 512.

## Weight formats

The bf16 checkpoint (235 GB) does not fit in ~127 GB of device memory; the first start converts it:

| Weights | Share of parameters | Stored as | Size |
|---|---:|---|---:|
| Routed experts | 97% | `bfloat4_b` (4-bit; 16 values share one 8-bit exponent) | 63.9 GB |
| Attention, shared expert, layer 0, LM head | 3% | `bfloat8_b` (8-bit) | 3.9 GB |
| Router, embedding, norms | <1% | bf16 | 0.7 GB |
| KV cache | - | `bfloat8_b` | - |
