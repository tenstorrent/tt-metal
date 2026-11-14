# Fused Decode Module

Batch-1 decode path for gpt-oss on Blackhole, one token per device: gpt-oss-20b on 1x4 (QuietBox 2, TP=4 over one mesh
row) and on one chip (P150, TP=1), and gpt-oss-120b on 1x4.

## When it is used

`config.fused_decode_supported()` turns it on only for: Blackhole, mesh (1, 4) at TP=4 or mesh (1, 1) at TP=1, EP=1,
32 experts (or 128 on 1x4), no throughput experts, 1 token per device, a compute grid of at least 10x9 and 8 DRAM banks.
Every other mesh, batch size and prefill keeps the original decode path; `GPT_OSS_FUSED_DECODE=0` keeps it everywhere.

## Switches

Every switch defaults to the fastest configuration; the expert formats are the most accurate that fit (one chip,
gpt-oss-120b) or the fastest (gpt-oss-20b on 1x4, where BFP8 is the accurate option).

| Environment variable | Default | Effect |
|---|---|---|
| `GPT_OSS_FUSED_DECODE` | `1` | `0`: the original decode path everywhere |
| `GPT_OSS_SPLIT_ATTN` | `1` | `0`: the paged SDPA decode op instead of the split decode attention |
| `GPT_OSS_LM_RESCORE` | `1` | `0`: the single-stage BFP8 LM head (1x4; one chip always uses it) |
| `GPT_OSS_FAST_TOPK` | `1` | `0`: the router's generic top-k pass (the same output) |
| `GPT_OSS_FUSED_EXPERT_DTYPE` | `bfp4` | gpt-oss-20b on 1x4: `bfp8` decode expert copies (exact for MXFP4, ~0.7 ms slower) |
| `GPT_OSS_ND_EXPERT_DTYPE`, `GPT_OSS_ND_DOWN_DTYPE` | `bfp8` | ND experts (one chip, gpt-oss-120b): `bfp4` for less DRAM |
| `GPT_OSS_ND_LAYER_DTYPES` | unset | ND experts: per-layer dtypes, e.g. `"0-11:bfp8/bfp4"` (`../bfp4_quant.py`) |
| `GPT_OSS_EXPERT_BFP4` | `kmse` / `mse` | Rounding of BFP4 expert tensors: `kmse`, `mse` or `max` (`../bfp4_quant.py`) |

The dtype and rounding switches are read when the model is built; the others also per decode call (a trace keeps the
form it was captured with).

## Attention

Every layer's attention (sliding-window and full) runs inside the decode QKV program as the split decode attention
(`split_attn.py`, `kernels/sda_*`): kv head g's 8 query heads are split over slice cores, each slice a contiguous range
of the attended positions (flash decoding). The slices read the chunks already in the paged cache while the QKV weights
stream, Q and the newest chunk once the QKV writers signal; each slice runs an online softmax, the slices of a kv head
all-gather their row maxima and the leader merges their rescaled partial states in one step. Sliding layers use 2
slices (the older chunks and the newest one), full layers 32 at TP=4 and 5 at TP=1.

## Experts

- **gpt-oss-20b on 1x4:** decode-only expert copies streamed per DRAM bank (`ExpertGateUpStream`, `ExpertDownStream`)
  next to prefill's copies. By default both are BFP4 with least-squared-error shared exponents; the decode copies are
  K-grouped: every tile is stored transposed so a BFP4 group is 16 consecutive K values inside one MXFP4 block, and the
  stream kernels transpose the tiles back on unpack (`W_TRANSPOSED`). `GPT_OSS_FUSED_EXPERT_DTYPE=bfp8` stores BFP8
  decode copies instead.
- **One chip and gpt-oss-120b on 1x4:** the experts are stored once, for prefill and decode: gate / up / down
  ND-sharded in one `[1, 1, K, 32]` DRAM shard per (expert, output tile column), dealt round-robin over the banks
  (`nd.py`). Prefill's sparse matmuls read them through their tensor accessors; `NdExpertGateUp` / `NdExpertDown`
  stream the routed experts' K runs. BFP8 by default (exact for the checkpoint's MXFP4 values).

gpt-oss-120b on 1x4 (36 layers, 128 experts, top-4) also uses:
- each device's shard of the intermediate dimension zero-padded from 720 to 736 (whole tiles); gate|up runs one core
  per (column, routed expert), down one core per output column; the decode-only bias tiles are packed 8 columns per
  tile;
- a router top-4 of 128 on 4 router cores (one 32-expert logit column each), merged on the column-0 core
  (`ROUTER_MERGE` in `kernels/stream_linear_writer.cpp`);
- a vocab-sharded embedding table (`embed.py`): 0.29 instead of 1.16 GB per device; the decode-inputs op sends the
  token's row (or zeros) as the all-reduce partial of the first layer boundary (site "entry"), prefill all-reduces the
  local rows;
- prefill fused-bias tables built per call and a 2048-token prefill expert chunk with BFP8 experts.

Per-device DRAM after the gpt-oss-120b build (usable 34.1 GB): BFP8 experts 31.1 GB, 32.9 GB in all with a 16k-token
KV cache; BFP8 fits up to ~32k tokens of context. For longer contexts set `GPT_OSS_ND_DOWN_DTYPE=bfp4` (BFP8 gate|up,
BFP4 down: 28.9 GB with a 128k-token KV cache).

## LM head

The streamed LM head splits the vocabulary over the TP devices. On 1x4 it runs in two stages (`terminal.py`,
`LM_RESCORE` in `kernels/stream_linear_*`): a BFP4 copy of the LM-head stream gives every output, then each LM-head
core recomputes the columns holding its 6 best ids from the BFP8 weights. The sampler candidates are those BFP8 values,
so the greedy pick is the BFP8 head's.

## Performance

Traced decode step (decode + on-device greedy pick, mean over 3 x 160 replays after a real-text prefill), ISL 128 / 8k,
`TT_MM_THROTTLE_PERF=2`; teacher-forced top-1 / mean KL against the fp32 HF model (gpt-oss-20b: 4 prompts x 256 decode
positions; gpt-oss-120b: 12 prompts x 160):

| Model, mesh | Experts | Step, ISL 128 / 8k | Top-1 / KL |
|---|---|---|---|
| gpt-oss-20b, one chip | BFP8 (default) | 8.61 / 9.34 ms | 94.8% / 0.011 |
| gpt-oss-20b, 1x4 | K-grouped BFP4 (default) | 1.99 / 2.20 ms | 92.6% / 0.029 |
| gpt-oss-20b, 1x4 | BFP8 (`GPT_OSS_FUSED_EXPERT_DTYPE=bfp8`) | 2.67 / 2.88 ms | 95.6% / 0.016 |
| gpt-oss-120b, 1x4 | BFP8 (default) | 4.02 / 4.33 ms | 94.5% / 0.022 |

## Structure

```
fused_decode/
├── __init__.py      # Public classes + fused_decode_supported
├── config.py        # Support check, decode-only weight dtypes, reader counts, attention slices, switches
├── inputs.py        # DecodeInputs: embedding row + Q/K RoPE rows in one op
├── boundary.py      # DecodeBoundary: all-reduce + residual add + next RMSNorm in one op (inter-layer contract)
├── stream.py        # DRAM-streaming matmuls: LinearStream (QKV, o_proj, router, LM head), expert gate|up / down
├── split_attn.py    # split decode attention, inside the QKV program
├── terminal.py      # DecodeTerminal: streamed LM head, folded logits, top-32 exchange, greedy pick / sampling
├── nd.py            # routed experts streamed from the ND-sharded expert tensors prefill also reads
├── embed.py         # vocab-sharded embedding table (gpt-oss-120b)
├── union.py         # multi-row MoE over the union of the routed experts (DFlash verify)
└── kernels/         # Metalium kernels of the ops above (run through ttnn.generic_op)
    ├── boundary_*.cpp          # boundary.py
    ├── decode_inputs.cpp       # inputs.py
    ├── nd_*.cpp, nd_stream.hpp # nd.py
    ├── sda_*                   # split_attn.py
    ├── stream_*.cpp            # stream.py, union.py
    ├── router_topk.hpp         # router top-k + softmax inside the router LinearStream writer
    └── terminal_*.cpp          # terminal.py
```

## One decode token

1. `DecodeInputs` writes the token's embedding row and the Q/K cos/sin rows (replaces 10 ops).
2. Per decoder layer (`DecoderLayer._decode_forward` in `../layer.py`):
   - boundary: all-reduce of the previous partial sums + residual add + input RMSNorm, run inside the QKV op;
   - QKV `LinearStream` (+ RoPE and the K / V cache write) with the split decode attention in the same program ->
     o_proj `LinearStream`, which writes this device's partial sum and sends it over the fabric;
   - boundary (post-attention norm), run inside the router `LinearStream` (top-4 + softmax in its writer);
   - the routed-expert gate|up and down ops read only the 4 routed experts' weights; the down op writes and sends the
     partial sum.
3. `DecodeTerminal.lm_head` runs the last boundary (+ final norm) inside the streamed LM head. It keeps a
   top-32 per writer core, merges and exchanges them over the fabric, and the greedy pick writes the next token
   straight into the decode input. Requests with penalties, log-probs or a grammar mask instead convert the logits
   to the original decode layout on device (`DecodeTerminal.stock_logits`) and use the common sampler.

Between layers the residual stream is a flat BF16 vector (hidden value h at byte 2h) on one core, replicated
on every device; `boundary.py` documents this contract. At TP=1 there is no all-reduce: each boundary reads the
producer's partial sum directly.

## Weights

The streamed ops read decode-only copies of the QKV, o_proj, router and LM-head weights (and, for gpt-oss-20b on 1x4,
of the experts), stored per DRAM bank. They are written to the ttnn weight cache on the first build with HF weights.
`ModelArgs.mark_weight_cache_complete` records `fused_decode_weights` and the expert format
(`fused_expert_cache_tag`), so a cache built without them is reloaded (or set `GPT_OSS_FORCE_MODEL_LOAD=1`).

## Usage

```python
from models.demos.gpt_oss.tt.fused_decode import fused_decode_supported

# Model and DecoderLayer call this themselves; nothing to set up by hand.
use_fused = fused_decode_supported(mesh_device, mesh_config, hf_config, use_throughput_experts, tokens_per_device)
```

## Demo

```bash
# QuietBox 2 (1x4 mesh), batch 1: decode runs on this path (HF_MODEL: gpt-oss-20b or gpt-oss-120b)
pytest models/demos/gpt_oss/demo/text_demo.py -k "1x4 and prefill_128"

# One chip (P150), batch 1
pytest models/demos/gpt_oss/demo/text_demo.py -k "1x1 and prefill_128"
```

## Tests

```bash
# Host only (runs in CI): which configurations take the fused decode path
pytest models/demos/gpt_oss/tests/unit/test_fused_decode_config.py

# QuietBox 2 (Blackhole 1x4) or one chip (-k 1x1), skipped in CI: 2-layer fused decode vs the HuggingFace reference,
# 8 decode steps (HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-120b: the 128-expert path, 1x4 only)
HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_fused_decode.py
```
