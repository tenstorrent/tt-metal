# AutoFix: full-prefill DRAM-sharded LM head

## Starting evidence

- Source diagnosis: `AUTODEBUG_PREFILL_LM_HEAD.md`.
- Original failing artifact: `run_prefill_check.log` from the AIME24
  `run_prefill_check` command. Its 249 logical rows are tile-padded to physical
  M=256, then `_terminal` attempts to apply the fixed `(32, 672)` width shard
  and fails with `Shard height 32 must match physical height 256`.

## Hypothesis experiment

- **Hypothesis:** normalize once, partition logical sequence rows into slices
  of at most 32, project every slice with the already-selected one-tile
  DRAM-sharded LM head, concatenate logits along M, then softcap once.
- **Focused static experiment:** AST inspection confirmed the pre-fix sharded
  `_terminal` had one unconditional `to_memory_config(normed, ...)` and no
  logical-M branch. TTNN validation in
  `matmul_device_operation.cpp:1364-1376` requires `M == per_core_M` and
  `M == 1`; the selected factory also rejects multiple K blocks per shard when
  `per_core_M > 1`. Dynamic-M reshaping is therefore not a valid local fix.
- **Verdict:** verified. Logical row tiling is the smallest repair that reuses
  the proven BF16/HiFi2, split-8192, 8-core program unchanged.

## Fix and regression

- Added `_sequence_tile_ranges`, covering arbitrary positive and non-aligned
  logical M.
- Extracted `_project_sharded_lm_head_tile`, which owns exactly one 1..32-row
  normalized tile and retains the selected sharding, program, dtype, fidelity,
  vocab split, and DRAM concat behavior.
- `_terminal` still normalizes once. M<=32 takes the original fast branch. For
  M>32 it slices the normalized tensor, projects each tile, concatenates along
  sequence, and applies the existing softcap once to the complete result.
- The opt-in reduced full-model hardware test now requests all logits for
  logical M=33 and asserts `[1, 33, 262144]` before continuing its existing
  sampler-ready prefill and traced decode checks.
- The no-device contract test covers M=1, 31, 32, 33, 63, 64, 65, 149, and the
  exact readiness M=249; it checks complete, contiguous, <=32-row ranges and
  that final RMSNorm occurs once before tiling.

## Verification

```bash
python -m py_compile \
  models/autoports/google_gemma_4_31b/tt/model.py \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py \
  models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py

MPLCONFIGDIR=/tmp/matplotlib LD_LIBRARY_PATH=$PWD/build/lib \
  pytest -q models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py
# 20 passed

git diff --check -- \
  models/autoports/google_gemma_4_31b/tt/model.py \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py \
  models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py
```

## Final status

The AutoFix fork intentionally performed only inspection and cheap gates; the
stage owner then serialized the required P150_X4 verification. The M=33 reduced
hardware regression passed (`autofix_reduced_prefill.xml`). The original
249-row AIME24 `run_prefill_check` also passed after the repair with top-1
90/100, top-5 100/100, and top-100 100/100 (`run_prefill_check.log`). The
pre-fix failure is retained separately as `run_prefill_check_pre_autofix.log`.
The hypothesis is therefore proven on the exact target path and this AutoFix is
complete.
