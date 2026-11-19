# Causal scorer tile guard proposal

Prepared while B32 hardware validation owns the frozen selected production source. **Applied after B32 clean close at14:45UTC.**

[causal_standard_tile_guard.patch](causal_standard_tile_guard.patch) adds only an explicit standard32x32, untransposed-tile check to the existing common validator in `tt/causal_indexer_scores.py`. It covers query, compressed keys, positions, and caller-supplied/generated outputs for both packed-Q1 and general-Q paths. Existing physical geometry, dtype, allocation, device, and alias checks remain unchanged. No kernel, arithmetic, dispatch, program descriptor, cache key, or allocation behavior changes for supported tensors.

The kernels hard-code standard face offsets and BF16/FP32 page sizes. TILE layout and padded shape alone do not exclude custom tiles. This patch rejects those unsupported inputs before native submission. Existing standard-tile component and full-model numerical evidence remains relevant within that explicit scope; it does not establish behavior for rejected custom tiles.

After the active B32 job closes, from the checkout root:

```bash
git apply --check models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/causal_standard_tile_guard.patch
git apply models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/causal_standard_tile_guard.patch
python3 models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/check_causal_standard_tile_guard.py
python3 -m py_compile models/autoports/qwen_qwen3_8_flash_next/tt/causal_indexer_scores.py
python_env/bin/python -m black --check --target-version py312 models/autoports/qwen_qwen3_8_flash_next/tt/causal_indexer_scores.py
```

The metadata check executes the actual nested validator AST, applying the proposed guard only in memory when production is still unpatched. Its25 cases cover five tensor roles with accepted standard tiles and rejected16x32,32x16, within-face-transposed, and face-transposed metadata. It imports no TTNN module and opens no device; it does not claim hardware support for constructing every rejected tile combination.

Refresh the selected source manifest before the already planned final performance run. That run exercises the accepted standard-tile path after applying the guard. A new broad numerical or datatype sweep is unnecessary for this rejection-only change; retain B32 evidence with its original source manifest and explicitly identify the guard-only source delta.

Applied-source metadata check:25/25 pass. Final45-file pre-commit hooks pass. `final_sources_v4.json` binds the rejection-only delta; final resident and full48 mixed-prefill evidence follow this guard.
