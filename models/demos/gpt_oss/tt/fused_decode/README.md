# Fused Decode Module

Batch-1 decode path for gpt-oss-20b on Blackhole 1x4 (QuietBox 2): TP=4 over one mesh row, one token per device.

## When it is used

`config.fused_decode_supported()` turns it on only for: Blackhole, mesh (1, 4), TP=4, EP=1, 32 experts,
no throughput experts, 1 token per device, a compute grid of at least 10x9 and 8 DRAM banks.
Every other mesh, batch size and prefill keeps the original decode path.

## Structure

```
fused_decode/
├── __init__.py      # Public classes + fused_decode_supported
├── config.py        # Support check, decode-only weight dtypes, reader counts, SDPA chunk sizes, path switches
├── inputs.py        # DecodeInputs: embedding row + Q/K RoPE rows in one op
├── boundary.py      # DecodeBoundary: all-reduce + residual add + next RMSNorm in one op (inter-layer contract)
├── stream.py        # DRAM-streaming matmuls: LinearStream (QKV, o_proj, router, LM head), expert gate|up / down
├── terminal.py      # DecodeTerminal: streamed LM head, folded logits, top-32 exchange, greedy pick / sampling
└── kernels/         # Metalium kernels of the ops above (run through ttnn.generic_op)
    ├── boundary_*.cpp         # boundary.py
    ├── decode_inputs.cpp      # inputs.py
    ├── stream_*.cpp           # stream.py
    ├── router_topk.hpp        # router top-k + softmax inside the router LinearStream writer
    └── terminal_*.cpp         # terminal.py
```

## One decode token

1. `DecodeInputs` writes the token's embedding row and the Q/K cos/sin rows (replaces 10 ops).
2. Per decoder layer (`DecoderLayer._decode_forward` in `../layer.py`):
   - boundary: all-reduce of the previous partial sums + residual add + input RMSNorm, run inside the QKV op;
   - QKV `LinearStream` -> fused Q/K RoPE -> paged SDPA -> o_proj `LinearStream`, which writes this device's
     partial sum and sends it over the fabric;
   - boundary (post-attention norm), run inside the router `LinearStream` (top-4 + softmax in its writer);
   - `ExpertGateUpStream` and `ExpertDownStream` read only the 4 routed experts' weights; the down op writes
     and sends the partial sum.
3. `DecodeTerminal.lm_head` runs the last boundary (+ final norm) inside the streamed LM head. It keeps a
   top-32 per writer core, merges and exchanges them over the fabric, and the greedy pick writes the next token
   straight into the decode input.

Between layers the residual stream is a flat BF16 vector (hidden value h at byte 2h) on one core, replicated
on every device; `boundary.py` documents this contract.

## Weights

The streamed ops read decode-only copies of the QKV, o_proj, router, expert and LM-head weights (about +3.2 GB
of DRAM per device), stored per DRAM bank. They are written to the ttnn weight cache on the first build with HF
weights. `ModelArgs.mark_weight_cache_complete` records `fused_decode_weights` so a cache built without them
is reloaded (or set `GPT_OSS_FORCE_MODEL_LOAD=1`).

## Usage

```python
from models.demos.gpt_oss.tt.fused_decode import fused_decode_supported

# Model and DecoderLayer call this themselves; nothing to set up by hand.
use_fused = fused_decode_supported(mesh_device, mesh_config, hf_config, use_throughput_experts, tokens_per_device)
```

## Demo

```bash
# QuietBox 2 (1x4 mesh), batch 1: decode runs on this path
pytest models/demos/gpt_oss/demo/text_demo.py -k "1x4 and prefill_128"
```

## Tests

```bash
# Host only (runs in CI): which configurations take the fused decode path
pytest models/demos/gpt_oss/tests/unit/test_fused_decode_config.py

# QuietBox 2 (Blackhole 1x4), ~45 s, skipped in CI: 2-layer fused decode vs the HuggingFace reference, 8 decode steps
HF_MODEL=models/demos/gpt_oss/configs/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_fused_decode.py
```
