# Testing 25K Full-Sequence Prefill with Real KV Cache

Uses `prefill_runner.py` in standalone mode — the full `GptOssPrefillPipeline` with a real
KV cache allocated for 25600 tokens.

## Prerequisites

- 4×8 BH Galaxy mesh available
- GPT-OSS 120B weights at a known path
- tt-metal environment activated

## Steps

```bash
# 1. Set env vars
export DEEPSEEK_V3_HF_MODEL=/path/to/gpt-oss-120b
export PREFILL_MAX_SEQ_LEN=25600
export PREFILL_STANDALONE=1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x8_mesh_graph_descriptor.textproto

# 2. Create a 25K-token input file
python3 -c "import json; json.dump({'task_id': 1, 'token_ids': [1]*25000}, open('standalone_input.json', 'w'))"

# 3. Run
cd /path/to/tt-metal
python3 models/demos/gpt_oss_d_p/tt/runners/prefill_runner.py
```

## What to look for

- No crash during `compile()` (warm-up forward with 25600 dummy tokens)
- No OOM or kernel error during `pipeline.prefill()`
- Log line: `[standalone] task_id=1 first_token=<id>` printed to stdout
- Timing log: `[prefill timing] ... pipeline.prefill()=<N> ms`

## Notes

- The pipeline pads every input to exactly `PREFILL_MAX_SEQ_LEN` tokens so the compiled
  trace is reused across calls. `actual_isl=25000`, padded to `25600`.
- KV cache is allocated at shape `[1, 1, 25600, 64]` per device (~1.6 MB per tensor per
  layer, well within 32 GB DRAM per device).
- Migration is disabled by default (`PREFILL_ENABLE_MIGRATION=0`), so `on_layer_complete`
  is a no-op. This is correct for a standalone test with no decode node.
- To use real tokens from an actual prompt instead of padding tokens, replace step 2 with:
  ```python
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("/path/to/gpt-oss-120b", trust_remote_code=True)
  ids = tok("Your 25K-token prompt here...")["input_ids"][:25000]
  import json; json.dump({"task_id": 1, "token_ids": ids}, open("standalone_input.json", "w"))
  ```
