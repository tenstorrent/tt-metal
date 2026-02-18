# DeepSeek Weight Dequantization Plan

## Motivation

We need a reliable way to convert DeepSeek-V3 Hugging Face checkpoints from mixed FP8+scale format into a dequantized safetensors checkpoint that:

- keeps non-quantized tensors unchanged,
- dequantizes quantized tensors using their `*_scale_inv` companions,
- and writes a valid sharded safetensors checkpoint back to disk.

The output must remain in standard HF safetensors layout so existing loaders continue to work.

## Goals

- Download a DeepSeek-V3-compatible checkpoint from Hugging Face (or use a local checkpoint).
- Dequantize all tensors that have corresponding `*_scale_inv` tensors.
- Preserve all other tensors unchanged except optional dtype normalization.
- Write output as safetensors shards plus `model.safetensors.index.json`.
- Keep memory bounded by processing shard-by-shard, not full-model materialization.
- Provide deterministic, automatable validation.
- Provide a fast test workflow that does not require full DeepSeek weights.

## Current Status

Implemented in:

- `models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py`
- `models/demos/deepseek_v3/tests/test_deepseek_weight_dequantization.py`

Current behavior is strict and fail-fast:

- FP8 tensors must have matching `*_scale_inv` tensors.
- Orphan scale tensors are rejected during preflight.
- Index/shard mismatches are rejected before conversion starts.
- Input and output directories must be different.

## Non-goals

- Changing tensor names or model architecture.
- Repartitioning model parallel structure or reshaping tensors.
- Producing TTNN cache artifacts.
- Supporting arbitrary quantization schemes beyond DeepSeek’s FP8 + inverse-scale block dequantization.
- Optimizing for maximum throughput in v1 over correctness and robustness.

## Current Repo Primitives to Reuse

- Dequantization primitive:
  - `models/demos/deepseek_v3/utils/config_helpers.py` (`dequantize`).
- Quantized-state helpers:
  - `models/demos/deepseek_v3/utils/test_utils.py` (`add_inv_scale_to_state_dict`, `dequantize_state_dict`).
- Lazy/indexed safetensors access pattern:
  - `models/demos/deepseek_v3/utils/lazy_state_dict.py`.
- HF download pattern:
  - `scripts/download_hf_artifacts.py` (uses `snapshot_download`).

Reusing these avoids diverging from existing DeepSeek logic in the repo.

## Functional Requirements

- Input modes:
  - HF repo id (`--repo-id`) + optional token.
  - Pre-downloaded local directory (`--input-dir`).
- Output mode:
  - Directory containing sharded safetensors and index JSON.
- Dequantization behavior:
  - For each key `k`, if `k_scale_inv` exists, dequantize `k`.
  - By default, omit `*_scale_inv` tensors from output.
  - Optionally retain `*_scale_inv` tensors via flag.
- Dtype behavior:
  - Default output dtype for dequantized tensors: `bfloat16`.
  - Non-quantized tensors can be left unchanged by default.
- Compatibility:
  - Output directory should be consumable by standard HF safetensors loading.

## Inputs and Outputs

## Input Checkpoint Layout

- `model.safetensors.index.json` with `weight_map`.
- One or more shard files like `model-00001-of-000NN.safetensors`.
- `config.json` with `quantization_config.weight_block_size`.
- Optional tokenizer and metadata files.

## Output Checkpoint Layout

- Updated shard files in safetensors format.
- Updated `model.safetensors.index.json`.
- Copied config/tokenizer/auxiliary files from input.

## Output Invariants

- Every output tensor key maps to exactly one output shard.
- Every entry in output `weight_map` exists on disk.
- All dequantized tensors are no longer FP8 payload tensors.
- If `--keep-scale-inv` is false, no `*_scale_inv` keys remain.

## Dequantization Semantics

Given:

- quantized tensor `W`
- inverse-scale tensor `S_inv`
- block shape `[Bh, Bw]` from config

Use the exact repo helper behavior:

- repeat/interleave inverse scales to tensor resolution,
- multiply elementwise in float,
- crop to original shape,
- cast to target dtype.

This matches existing behavior in `dequantize(...)` and avoids algorithm drift.

## High-level Architecture

The converter is a host-only Python script:

1. Resolve input checkpoint location.
2. Read config + index metadata.
3. Run preflight structural validation.
4. Iterate shards in deterministic order.
4. For each tensor key in shard:
   - load source tensor lazily,
   - load `*_scale_inv` lazily if needed (possibly from another shard),
   - dequantize or pass-through,
   - append to an incremental output buffer.
5. Flush buffered output to safetensors shards by byte budget.
6. Emit updated index JSON.
7. Copy auxiliary non-weight files.

## Data Access Strategy

Use cached `safe_open` handles keyed by filename:

- avoids reopening/mmap churn,
- supports cross-shard `k` + `k_scale_inv` lookup,
- keeps memory bounded to active tensors.

No full-state-dict materialization is allowed in default flow.

## Sharding Strategy

Output shards are rebuilt by size budget:

- Tensors are buffered and flushed when `--max-output-shard-size-mb` budget is reached.
- Temporary shards are renamed into canonical names:
  - `model-00001-of-000NN.safetensors`, etc.
- `weight_map` is regenerated from emitted output keys.
- `k_scale_inv` keys are dropped unless `--keep-scale-inv`.

Benefits:

- bounded output buffering memory,
- deterministic, canonical shard naming,
- robust behavior for very large checkpoints.

## Index Rewrite Strategy

Build new `weight_map` as keys are emitted.

Output index fields:

- `weight_map`: regenerated from emitted keys.
- `metadata.total_size`: recomputed from output tensors.
- preserve any unknown metadata fields where safe.

## CLI Design

Script:

- `models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py`

Implemented arguments:

- `--repo-id`: HF repo id, optional when `--input-dir` is provided.
- `--input-dir`: local checkpoint path, optional when `--repo-id` is provided.
- `--output-dir`: required.
- `--hf-token`: optional.
- `--cache-dir`: optional.
- `--dtype`: `bfloat16|float16|float32` for dequantized tensors.
- `--keep-scale-inv`: retain `*_scale_inv` keys.
- `--overwrite`: allow writing into non-empty output directory.
- `--max-output-shard-size-mb`: output shard flush budget (default `5120` = 5 GiB).

Behavior rules:

- exactly one of `--repo-id`, `--input-dir` is required.
- input and output directories must be different.
- fail fast on structural inconsistencies before conversion.
- deterministic processing order.

## Error Handling Policy

Hard errors (default fail):

- missing `model.safetensors.index.json`,
- missing shard referenced by index,
- index key not present in referenced shard,
- orphan `*_scale_inv` without matching base key,
- missing scale tensor when a quantized tensor is expected to dequantize,
- non-FP8 tensor paired with `*_scale_inv`,
- non-floating-point `*_scale_inv`,
- incompatible tensor/scale dimensionality for block dequantization,
- invalid output directory state without `--overwrite`.

Soft handling:

- none for missing-scale on FP8 tensors; conversion must fail to avoid silent corruption.

## Performance and Resource Plan

## Memory

- Stream per tensor and per shard.
- Keep small handle cache for source shards.
- Bound output buffering with `--max-output-shard-size-mb`.
- Avoid full checkpoint load.

## Disk

Expected output size can be significantly larger than FP8 input.

Preflight:

- estimate output bytes using source dtypes and target dtype policy,
- compare with free disk space,
- fail early with actionable error message.

## Time

- Dominated by tensor IO and dequantization compute.
- Parallelism should be conservative in v1 to avoid memory spikes.

## Validation Plan

## Runtime Validation in Script

- Preflight validation of index/shard consistency.
- Strict FP8/scale pairing checks.
- Tensor accounting summary logs (`seen`, `dequantized`, `passthrough`, `scales_kept`).
- Emit regenerated index with recomputed `metadata.total_size`.

## Automated Tests

### 1) Unit: Synthetic Tiny Checkpoint (Primary Fast Path)

Create a tiny synthetic DeepSeek-style checkpoint in temp dir:

- include several quantized keys with `*_scale_inv`,
- include several normal non-quantized keys,
- split across 2+ shards,
- include at least one cross-shard pair:
  - `k` in shard A, `k_scale_inv` in shard B.

Generation approach:

- Build small torch tensors.
- Use `add_inv_scale_to_state_dict(...)` for realistic quantized payload generation where possible.
- Save shards with `safetensors.torch.save_file`.
- Write index JSON with `weight_map`.

Assertions:

- converted keys match expected set,
- dequantized values match `dequantize(...)` reference,
- pass-through values unchanged,
- output index valid,
- `*_scale_inv` dropped/retained according to flag.

### 2) Unit: Negative Cases

- missing shard file referenced by index,
- missing `k_scale_inv`,
- orphan `k_scale_inv` without base tensor,
- non-FP8 tensor with `k_scale_inv`,
- index key missing in referenced shard,
- input_dir == output_dir,
- malformed block shape,
- corrupt index JSON.

### 3) Integration: Optional Small Real Model

Optional smoke test using a small local HF checkpoint that has safetensors, only to verify CLI plumbing and index rewriting. This does not replace synthetic quantized tests.

## Test Execution Note

Because tests live under `models/demos/deepseek_v3/tests/`, they inherit DeepSeek collection hooks that require `MESH_DEVICE` to be set during collection. For host-only smoke tests, run:

```bash
MESH_DEVICE=N150 pytest -q models/demos/deepseek_v3/tests/test_deepseek_weight_dequantization.py
```

## Why Tiny Synthetic Checkpoints Are the Main Test Strategy

Small public models usually do not use DeepSeek-style FP8+`_scale_inv` layout, so they cannot test the dequantization path. Synthetic checkpoints let us:

- test the exact dequantization logic,
- test cross-shard lookup edge cases,
- and run quickly in CI.

## Implementation Plan

## Phase 1: Script Skeleton and IO

- Add CLI parser and argument validation.
- Add input resolver (`snapshot_download` or local path).
- Add index/config loader.
- Add output directory creation and metadata copy.

## Phase 2: Dequantization Pipeline

- Add shard handle cache.
- Add tensor conversion loop.
- Add shard writer and index rewrite.

## Phase 3: Validation and Diagnostics

- Add key accounting checks.
- Add optional sampling-based numeric checks.
- Add dry-run and disk preflight reporting.

## Phase 4: Tests

- Add synthetic checkpoint builder fixture.
- Add core conversion tests.
- Add failure-mode tests.

## Phase 5: Documentation

- Add usage examples to DeepSeek README or script docstring.
- Add troubleshooting notes.

## Acceptance Criteria

- Script converts a synthetic sharded checkpoint end-to-end and passes tests.
- Output checkpoint loads with safetensors and has valid index map.
- Dequantized tensors numerically match reference dequantization.
- Non-quantized tensors remain unchanged.
- Script handles cross-shard scale lookup correctly.
- Disk/missing-file failures are clear and actionable.

## Example Commands

Local checkpoint:

```bash
python models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py \
  --input-dir /path/to/deepseek-checkpoint \
  --output-dir /path/to/deepseek-checkpoint-dequantized \
  --max-output-shard-size-mb 5120
```

HF download + convert:

```bash
python models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py \
  --repo-id deepseek-ai/DeepSeek-R1-0528 \
  --output-dir /path/to/dequantized-out \
  --hf-token $HF_TOKEN
```

## Risks and Mitigations

- Risk: output size explosion.
  - Mitigation: preflight disk estimate and fail-fast.
- Risk: cross-shard scale lookup bugs.
  - Mitigation: explicit cross-shard synthetic tests.
- Risk: behavior mismatch with existing DeepSeek dequantization.
  - Mitigation: call existing `dequantize(...)` helper directly.
- Risk: accidental metadata incompatibility.
  - Mitigation: preserve non-conflicting metadata and validate index.

## Open Questions

- Should non-quantized tensors be cast to target dtype or remain original dtype by default?
- Should `*_scale_inv` be retained by default for auditability, or dropped by default for size?
- Should we support writing a single `model.safetensors` output mode in addition to sharded output?

## Recommended Defaults for V1

- Dequantize only tensors with matching `*_scale_inv`.
- Drop `*_scale_inv` keys by default.
- Cast dequantized tensors to `bfloat16`.
- Leave non-quantized tensors unchanged.
- Fail on missing scale for FP8 tensors.
- Use output shard size budget of 5 GiB (`--max-output-shard-size-mb=5120`).
