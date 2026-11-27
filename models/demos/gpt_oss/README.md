# GPT-OSS: Mixture of Experts Language Model

Inference implementation for GPT-OSS models on Tenstorrent Wormhole and Blackhole accelerators.

**Model Source**: [GPT-OSS on HuggingFace](https://huggingface.co/gpt-oss) (custom MoE architecture)

**Target Hardware**:
- **LoudBox**: Single Wormhole device (1×8 configuration)
- **Galaxy**: Multi-device Wormhole mesh (4×8 configuration)
- **QuietBox 2**: 4 Blackhole devices (1×4 configuration); batch-1 decode uses the fused decode path ([tt/fused_decode](tt/fused_decode/README.md)), prefill the routed-expert path ([Optimizations on QuietBox 2](#optimizations-on-quietbox-2-gpt-oss-20b))

**Current Status**: This model is under active development.
- ✅ Supported: Prefill up to sequence length 128, batch size 1, total sequence length 4096
- 🚧 In Progress: Extended sequence lengths, larger batch sizes

## Quick Start

```bash
# Set model path using HF_MODEL environment variable
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b"

# Run text generation demo on Galaxy (4×8 mesh)
cd tt-metal/models/demos/gpt_oss/demo
pytest text_demo.py -k "4x8 and prefill_128"

# Run text generation demo on QuietBox 2 (1×4 mesh)
pytest text_demo.py -k "1x4 and prefill_128"
```

## Optimizations on QuietBox 2 (gpt-oss-20b)

gpt-oss-20b on QuietBox 2 (4 Blackhole chips, 1x4 mesh, TP=4). Decode (#60161, #61050) and prefill (this code) were
optimized separately; each has its own section below.

| | Before | Now | Where |
|---|---:|---:|---|
| Batch-1 decode, ms/token (ISL 128) | 18.35 | **2.38** | [Batch-1 decode](#batch-1-decode-60161) (#60161) |
| DFlash decode, tok/s/user (ISL 256 / 1K / 32K) | 415 / 405 / 338 (plain decode) | **772 / 718 / 699** | [DFlash](#dflash-speculative-decoding-61050) (#61050) |
| Batch-1 time to first token, ms (1K / 8K / 32K prompt) | 1,233.5 / 9,715.6 / 39,259.9 | **55.6 / 180.1 / 788.3** | [Prefill](#prefill-optimizations-batch-1-and-batch-32) |
| Batch-32 time to first token, ms (32 x 1K prompts) | 14,704.1 | **634.9** | [Prefill](#prefill-optimizations-batch-1-and-batch-32) |

## Decode optimizations

### Batch-1 decode (#60161)

The decode path in [tt/fused_decode](tt/fused_decode/README.md). It turns on only for gpt-oss-20b, batch 1, on
Blackhole 1x4 (`fused_decode_supported()` in `tt/fused_decode/config.py`). Per decode token:
- **Experts:** only the 4 routed experts' weights are read. Own DRAM-streaming kernels read each bank's expert columns
  in large contiguous pieces (the sparse matmul read one 576-byte tile per request and stalled near 220 GB/s), with the
  bias, SwiGLU and the weighted sum of the 4 experts inside the kernels.
- **Projections:** QKV, o_proj and the router run on the same streaming kernel; it writes the exact layout the next op
  reads, so the reshape / move / slice ops around them are gone. Router top-4 + softmax run in its writer.
- **Between blocks:** one op per layer boundary does the all-reduce over the fabric, the residual add and the next
  RMSNorm, inside the next matmul's program (the stock all-reduce was 7 ops on one core, 47.6 µs per call). The hidden
  vector is one flat row the next matmul reads in one request.
- **LM head and sampling:** an own LM-head streaming kernel (1,218 → 307 µs per token, ~508 GB/s); each core keeps its
  top 32 candidates, the chips exchange them inside the LM-head program and the greedy pick is written straight into the
  next decode input.
- **Decode inputs:** the embedding row and the RoPE rows are written by one op instead of 10.

### DFlash speculative decoding (#61050)

Greedy batch-1 decode with the [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) drafter
(downloaded from HuggingFace on first use). Each round, the drafter proposes 7 tokens from the target's hidden states;
the target verifies the 8-row block [last token, 7 drafts] in one pass and keeps the drafts up to the first mismatch,
plus its own next token. The output is the target's greedy output (up to BF16 rounding of near-ties).

```bash
export HF_MODEL=openai/gpt-oss-20b
pytest models/demos/gpt_oss/demo/dflash_demo.py -k "isl_1k"   # isl_256, isl_512, isl_1k, ... isl_64k, isl_128k
pytest models/demos/gpt_oss/tests/test_dflash.py               # accuracy vs. the target's own greedy decode (1K, 16K)
```

The demo and the test download the drafter and the prompt's books (gutenberg.org) on first use; without network
access they skip (under CI they fail). `DFLASH_DRAFT_PATH` points at a local copy of the drafter checkpoint instead
of the HuggingFace cache.

A round is three passes, captured as one device trace and replayed once per round; the host reads one small result
per round:
- draft pass: [tt/dflash/fused_draft.py](tt/dflash/fused_draft.py);
- 8-row verify pass: [tt/dflash/verify.py](tt/dflash/verify.py);
- context pass (adds the accepted tokens' target features to the drafter's K / V cache).

What makes a round fast (a round is mostly weight reads: ~1.6 GB per chip, mainly the experts any of the 8 rows picks):
- **One verify pass for 8 rows:** every weight is read once for all 8 rows (instead of one decode step per draft), and
  the MoE runs the union of the rows' experts as one program; the most-used experts of the last round are prefetched
  while the router runs.
- **The round stays on the device:** the passes hand drafts, verify inputs and context rows to each other on the
  device, the acceptance is computed on the device, the round is one trace, and the next round is queued before the
  host reads this one's 16-integer result.
- **The drafter on the fused-decode streaming kernels**, with its q / k prep + K / V cache write and its MLP as one
  program each.
- **Attention:** a small 8-row kernel for the sliding-window layers, and "block attention" for the full-attention
  layers, where the 8 rows share every K / V cache read (131,072 tokens: 18.01 → 8.20 ms per round).
- **Two-stage LM head:** a 4-bit pass keeps each core's 8 best candidates; only those are re-scored at 8 bits.
- **DRAM readers placed per bank and per chip:** each reader on its bank's row (464 → 490-504 GB/s per chip), weights
  balanced over the banks, slow cores left idle. Also fixed plain decode on `main` (2.84 → 2.38 ms/token).

The drafter attends to the last 1024 positions. The chat template writes "Current date: ..." into the prompt, so the
demo pins that date (`PROMPT_DATE`): the answer, and with it the tokens accepted per round, changes with the date.

| Prompt length (tokens) | Tokens per round | Draft (ms) | Verify (ms) | Round (ms) | DFlash (tok/s/user) | DFlash, mean of 5 dates | Plain decode (tok/s/user) |
|---|---|---|---|---|---|---|---|
| 256 | 4.03 | 1.02 | 3.79 | 4.842 | 813.4 | 771.9 | 415.2 |
| 512 | 3.04 | 1.02 | 3.69 | 4.749 | 641.7 | 689.3 | 408.9 |
| 1,024 | 4.11 | 1.06 | 3.88 | 4.967 | 831.3 | 717.5 | 404.9 |
| 2,048 | 3.32 | 1.05 | 3.87 | 4.957 | 670.7 | 696.0 | 401.0 |
| 4,096 | 3.34 | 1.05 | 3.83 | 4.923 | 675.4 | 703.1 | 393.1 |
| 8,192 | 3.46 | 1.06 | 3.98 | 5.074 | 681.8 | 638.6 | 375.8 |
| 16,384 | 3.31 | 1.05 | 4.07 | 5.162 | 644.1 | 665.7 | 360.9 |
| 32,768 | 3.76 | 1.06 | 4.38 | 5.474 | 687.7 | 698.9 | 337.9 |
| 65,536 | 3.94 | 1.06 | 4.83 | 5.921 | 665.1 | 636.2 | 302.3 |
| 130,752 | 3.37 | 1.11 | 5.65 | 6.739 | 499.8 | 482.2 | 258.5 |

QuietBox 2, 256 new tokens. Round and DFlash: the demo above (one `-k isl_*` case per row: the chat-template
Gutenberg prompt dated 2026-10-17). Draft and verify: the same rounds with each pass timed on its own (a device
synchronize after each pass; the round also holds the context pass, ~0.1 ms). Mean of 5 dates: the same prompt dated
2026-10-15 .. 2026-10-19. Plain decode: the fused batch-1 decode on the same build, a random-token prompt of the same
length (130,943 tokens for the last row); `pytest models/demos/gpt_oss/demo/text_demo.py -k "1x4 and prefill_128"`
gives 2.38 ms per token (420 tok/s/user).

| File | Role |
|---|---|
| `tt/dflash/runner.py` | `DFlashRunner`: prefill, trace capture, the round loop |
| `tt/dflash/draft.py` | the drafter's weights, K / V cache, prompt context and per-round inputs |
| `tt/dflash/fused_draft.py` | the draft pass and the context pass |
| `tt/dflash/verify.py` | the 8-row verify pass of the target |
| `tt/dflash/qk_prep.py`, `kv_write.py` | the drafter's q / k norm + RoPE, its K / V cache write |
| `tt/dflash/row_attention.py`, `block_attention.py` | verify attention: sliding-window layers, full-attention layers; drafter attention |
| `tt/dflash/moe_fused.py` | the verify MoE and the drafter MLP as one program per device |
| `tt/dflash/lm_head.py`, `lm_rescore.py`, `argmax.py` | the two-stage LM head and the on-device acceptance |
| `tt/dflash/capture.py` | the target's prefill features for the drafter |
| `reference/dflash.py` | torch reference of the drafter and of DFlash greedy decoding |
| `tt/fused_decode/kernels/` | the device kernels of these programs (shared with the fused batch-1 decode they extend) |

## Prefill optimizations (batch 1 and batch 32)

Prefill by @gtobarTT (branches `gtobarTT/gpt-oss-20b-prefill-optimizations` and `gtobarTT/gpt-oss-20b-prefill-b32`),
on QuietBox 2 next to the decode paths above. Time to first token (TTFT), ms:

| Prompt length (tokens) | Batch 1: before | **Batch 1: now** | Batch 32 (all 32 users): before | **Batch 32: now** |
|---:|---:|---:|---:|---:|
| 128 | 164.3 | **41.1** | 2,731.6 | **127.1** |
| 512 | 1,231.7 | **47.5** | 14,706.6 | **345.9** |
| 1,024 | 1,233.5 | **55.6** | 14,704.1 | **634.9** |
| 2,048 | 2,312.0 | **68.3** | 28,417.1 | **1,176.2** |
| 4,096 | 4,851.5 | **105.3** | 56,047.6 | **2,458.1** |
| 8,192 | 9,715.6 | **180.1** | 113,700.0 | **5,239.4** |
| 32,768 | 39,259.9 | **788.3** | 464,260.5 | **25,828.0** |
| 131,072 | 155,734.8 | **5,907.2** | does not fit | does not fit |

Before: `main` 35f4cd54629; now: this code, back to back on one QuietBox 2. Random prompts (batch 1) and the
harness's text prompts (batch 32: 32 different prompts in one prefill call), untraced, logits read back to host.
`main` pads 512 to 1,024. 32 users x 128K tokens of KV cache do not fit in DRAM.

What makes prefill faster:
- **MoE computes only the routed experts** (`tt/experts/prefill_routed.py`). Before, prefill ran every token through
  all 32 experts. Now each chip computes the (token, expert) pairs routed to its experts, with the experts placed over
  the chips to balance the load (`prefill_routed_placement.py`). Most of the speed-up (1K: 1,198 → 137 ms).
- **Attention on the whole core grid**, with SDPA chunk sizes per prompt length and HiFi2 math.
- **Fewer, smaller transfers between the 4 chips:** the o_proj all-reduce as one op on unpadded weights, bf8
  payloads, 2 fabric links.
- **Sequence-parallel residual from 1K tokens:** each chip keeps a quarter of the rows between layers instead of all
  of them, so the residual adds and RMSNorms run on S / 4 rows.
- **2D matmul program configs** for the QKV and o_proj projections.
- **Batch 32 in one forward pass** (`tt/batched_prefill.py`): the 32 users' prompts go through each layer together (in
  groups of users at 4K and 8K), each user bit-identical to its own batch-1 prefill; on-device sampling of the first
  tokens.

Then, on top of the merged branches (batch-1 TTFT without → with each change, measured back to back on the tree it was
added to):
- **The expert op reads its input rows through an index.** A gather used to copy every token's row 4 times (once
  per chosen expert) into a buffer of 4 x C rows, a quarter of which each chip reads, before the expert op read the
  copy back. Now `unified_routed_expert_moe` reads row `tok_of_row[r]` of the token-ordered input directly
  (`x_row_index`). 8K: 230.7 → 216.5 ms.
- **The expert op writes rows, not tiles.** The combine needs rows, so a separate untilize converted the whole
  4 x C-row output (0.77 ms per layer at 8K). Now the down matmul's last pack untilizes each tile-row and the writer
  stores rows (`output_row_major`); the expert outputs also stay bf16 instead of being rounded to bf8.
  8K: 217.4 → 201.2 ms; 32K: 939.0 → 873.7 ms.
- **The combine reads the expert rows where they are.** An embedding gather copied each token's 4 expert rows into a
  [block, 4, 2880] buffer that `post_combine_reduce` then read. Now `post_combine_reduce` reads row `pos[t, j]` of the
  expert output itself (`combine_row_index`) and never reads the rows of experts on other chips (3 of every 4).
  8K: 191.1 → 180.6 ms; 32K: 834.8 → 794.7 ms.
- **The scatter copies its input row in one transfer.** `ttnn.scatter` (it builds the buffer-row → token map)
  copied its input row from a scratch buffer one element at a time (~300 µs per layer at 8K); now the NoC reads it
  straight into the output page. 8K: 196.5 → 191.1 ms.
- **Routing weights read once per token.** `post_combine_reduce` read each (token, expert) weight with its own DRAM
  read, from a [C, 4, 1] copy made by a reshape (63 µs per layer at 8K). Now one read per token of the [C, 4]
  weights. 8K: 199.1 → 196.5 ms.
- **Attention writes its heads side by side** (`output_concat_heads`) instead of a separate concat op.
  8K: 182.6 → 179.7 ms.
- **Router on one K block up to 128 tokens per chip** (77.5 → 12.3 µs per layer at 128 tokens).

All of these are bit-identical except the bf16 expert outputs. The accuracy gates pass, and batch 32 stays
bit-identical to batch 1.

Running next to the decode paths: the routed-expert op and the 2D projection configs fit their L1 below the buffers
DFlash keeps resident (`Model.set_prefill_l1_end`); the routed path is used for gpt-oss-20b only (gpt-oss-120b keeps
the dense prefill); a read-only weight cache builds the routed expert weights in memory.

```bash
# batch-1 TTFT (one prompt length per entry; 128 ... 128k)
PREFILL_LENGTHS=128,1k,8k,32k,128k pytest models/demos/gpt_oss/tests/perf/prefill_ttft.py -k 1x4
# batch-32 TTFT (32 prompts in one prefill call)
PREFILL_LENGTHS=128,512,1k,2k,4k,8k pytest models/demos/gpt_oss/tests/perf/prefill_ttft_b32.py -k 1x4
# prefill accuracy gate vs HF (8 prompts + an 8K window) and batch-32 vs batch-1 prefill
python models/demos/gpt_oss/tests/perf/prefill_gate.py --mode v4
python models/demos/gpt_oss/tests/perf/prefill_gate_b32.py
```

## Configuration

### Model Selection
```bash
# GPT-OSS-20B (faster, recommended for development)
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b"

# GPT-OSS-120B (higher quality, requires more memory)
export HF_MODEL="/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-120b"
```

## Testing

```bash
# Run all tests
pytest models/demos/gpt_oss/tests/unit/ -v

# Run specific test files
pytest models/demos/gpt_oss/tests/unit/test_modules.py -v     # Core components
pytest models/demos/gpt_oss/tests/unit/test_model.py -v       # Full model accuracy
```

### Test Files Overview

| File | Purpose | Tests |
|------|---------|-------|
| **`test_modules.py`** | Core MoE components | • Attention component<br>• RMSNorm<br>• TopK router<br>• Experts<br>• Full MLP pipeline<br>• Complete decoder layer |
| **`test_model.py`** | Full model integration | • End-to-end accuracy<br>• Teacher forcing<br>• Reference model comparison |
| **`../test_dflash.py`** (`pytest models/demos/gpt_oss/tests/test_dflash.py`) | DFlash speculative decoding (QuietBox 2) | • Teacher-forced top-1 agreement with the target's greedy decode<br>• Mean accepted tokens per round |

### Host sampling readback

GPT-OSS declares `supports_compact_host_logits` and
`supports_selective_host_readback` for the vLLM plugin. The plugin supplies
`sample_rows` in scheduled request order. Host processing preserves this order
and returns independently owned FP32 logits without vocabulary padding.

Model construction prepares the selective readback programs before any trace
is captured. Reads use aligned power-of-two slot ranges, clipped at the batch
boundary. For 32 slots per device, there are 62 slice programs and five reusable
device buffers. The buffers contain 31 rows in total. A full-range read uses
the original decode output directly. Some scattered requests transfer extra
rows because their enclosing range is larger than the exact active interval.

Serving reads use only these prepared programs and buffers. They reject an
input specification that differs from the prepared row-major BF16 DRAM logits.
Keep the program cache enabled throughout model construction and serving.
Slices and transfers use command queue 0, so each transfer completes before
a later slice can overwrite its buffer. Each read has a separate host tensor.
Consumers must wait on all returned events before processing host output.
Inactive generator ranks have no transfer event; the event list is not indexed
by rank. Calls without `sample_rows` retain the full readback path.
