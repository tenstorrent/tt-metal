# GPT OSS Token Accuracy Testing

## Overview

Top1/Top5 token accuracy testing for GPT OSS using the "Tale of Two Cities" benchmark with teacher forcing. Uses `.refpt` reference files (same approach as `tt_transformers` and the `tt-xla` repo) for 1:1 cross-repo comparison.

**Configuration:** 64 token prefill + 64 token decode (128 total, split in half).

## How to run

```bash
# GPT OSS 20B on 1x8 mesh
pytest models/demos/gpt_oss/demo/text_demo.py -k "token_accuracy and mesh_1x8" -v
```

## What was changed

**`models/demos/gpt_oss/demo/text_demo.py`:**
- Added `TokenAccuracy` class (loads `.refpt`, implements teacher forcing and top1/top5 computation)
- Added `"token_accuracy"` parametrize config to `test_gpt_oss_demo` (batch=1, 64 decode tokens, greedy sampling, trace disabled)
- Teacher forcing inserted before `decode_forward` in the decode loop
- Accuracy reported after decode loop completes

**`models/demos/gpt_oss/tests/reference_outputs/gpt-oss-20b.refpt`:**
- Pre-generated reference file copied from `tt-xla` repo
- Contains: `reference_tokens` [1, 128], `top5_tokens` [127, 5], `top1_tokens` [127], `library_versions`

## Key constraints

- **Trace must be off** (`enable_decode_trace=False`) — teacher forcing changes input tokens dynamically
- **Batch size = 1** — teacher forcing only applies to user 0
- **Greedy sampling** (`temperature=0`) — deterministic for accuracy measurement
- **`stop_at_eos=False`** — must run all 64 decode tokens

## TODO

- [ ] Run first test and record baseline top1/top5 numbers
- [ ] Compare with tt-xla results for the same model
- [ ] Set accuracy thresholds and enable `run_in_ci=True`
- [ ] Add `generate_reference_outputs.py` for GPT OSS (to regenerate `.refpt` when needed)
- [ ] Add GPT OSS 120B `.refpt` file and test config
