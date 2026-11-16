# Proposed decode weight accounting by precision group

This is a host-only proposal. `benchmark_work_precision_groups.patch` is deliberately not applied to `tools/benchmark_work.py`; the tests apply it only to a temporary copy.

The change preserves the current arithmetic path whenever attention, MLP gate/up, and down all use BFP4. This keeps current baseline and head-only rate-work numbers exactly equal. For other supported policies, it weights each projection group's physical element count using the existing policy label: `bfp4`/`bfp4_hifi2` = 576/1024 bytes, `bfp8`/`bfp8_lofi` = 1088/1024 bytes, and `bf16` = 2 bytes. Fidelity suffixes do not change storage size. It rejects missing/extra groups, per-layer exceptions, unsupported labels, and non-BFP8 KV.

The group split follows `tt/multichip_decoder.py`: attention comprises Q/K/(V for sliding layers) plus O projection; MLP comprises gate and up; down is a separate group. The decoder's state-dict path pads gate/up to intermediate width 22528 and down's input width to 22528, then pads O/down output width to 5632 (`multichip_decoder.py:121-146`). The decoder selects attention policy for `self_attn.*`, down policy only for `mlp.down_proj.weight`, and the MLP policy for gate/up (`:153-166`); `_finish` calls gate, up, then down separately (`:522-526`). The full-attention layer uses shared K/V (`benchmark_work.py:11,18` and decoder's shared K/V path), while sliding attention has distinct V.

For a down-only BFP4→BFP8 change, the analytical increase is `(1088-576)/1024 * Σ_layers(n_layers * 5632 * 22528)` bytes. This is a modeled full-model weight-read estimate, not a memory-controller measurement. KV remains fixed BFP8. Precision-specific L1 reuse and actual memory traffic remain outside this model.

Run the standalone host suite with:

```sh
python_env/bin/python -m unittest models.autoports.google_gemma_4_31b_it.doc.agentic_evals.setup.swe85-precision-bisect.candidate-serving-prep.test_benchmark_work_precision_groups -v
```

The suite covers exact baseline/head-only equality against the current implementation, analytical down BFP4→BFP8 byte delta, equivalent storage widths for fidelity variants, and rejection of unsupported layer exceptions/KV/labels/group sets. No model, device, or serving workload is used.

# Proposed explicit KV-pool override

`kv_pool_ids_override.patch` and `test_kv_pool_ids_override.py` are also host-only proposals; the patch is applied only to temporary copies by the test. The proposed `GEMMA4_KV_POOL_IDS` accepts canonical positive decimal IDs only for B1/B8/B16/B32 at the advertised `max_model_len=262144`. It conflicts with the legacy `GEMMA4_B8_KV_POOL_IDS` variable. The old B8 override behavior, including its default/no-op behavior for other batch sizes, is retained exactly. The generic value and its source, total normalized tokens, minimum IDs and reserves are exposed in the existing `serving_pool_budget` event. The separate allocator check (`validate_pool_memory` and `memory_plan["fits"]`) is unchanged and remains the measured-free-memory launch guard.

For requested IDs `N` and batch `B`, the proposed synthetic budget is `32*N - 32*B - 8*1024*B`. This carries forward the existing autoport's 32-token pool-ID mapping, one 32-token output reserve per sequence, and eight 1024-token sliding allowances per sequence. Requiring `N >= ceil((262144 + 32*B + 8*1024*B)/32)` ensures one advertised-length context per sequence plus those reserves; the minima are B1=8449, B8=10248, B16=12304 and B32=16416. These are validation minima, not selected pool sizes. The old B8 setting remains 20480 IDs.

Superseded source audit: an earlier inspection used `bringup/artifacts/v5-main-harness/tt-vllm-plugin`, which is not the serving plugin imported by this Python environment. Its conclusion about missing sliding-window reserves was incorrect for the active serving path; retained here only to document the correction.

Correct serving provenance: `python_env/lib/python3.10/site-packages/__editable__.vllm_tt_plugin-0.1.0.pth` points to `bringup/artifacts/v4-serving-plugin/src`. `tools/serving_paths.py:plugin_root()` resolves that editable checkout. In its `vllm_tt_plugin/worker.py`, `get_num_available_blocks_tt` reserves `max(cache_config.block_size, output_tokens_per_step) * max_batch` at lines 784–799, adds up to eight `sliding_window * max_batch` allowances at 801–831 when the model enables hybrid KV groups, then rounds by `cache_config.block_size` at 833. It subsequently applies `max(heuristic_blocks, admission_floor)` at 840–870, where the floor is derived from upstream KV specs and includes the output reservation. Gemma 4's autoport sets `_HYBRID_KV_CACHE_GROUPS_ENABLED = True`, returns 32-token KV specs, and uses a 1024-token sliding window. For this AR path, output width defaults to 1, so the reservation is one 32-token block per sequence.

With those current dimensions, the proposed budget `32*N - 32*B - 8*1024*B` cancels the worker's output and sliding heuristic reserves, so its rounded heuristic pool is exactly `N` blocks. The context-preserving minima remain B1=8449, B8=10248, B16=12304, B32=16416. The separate upstream admission floor can raise a requested `N`; it does not lower it. Current `pool_plan.json` records a 4902-block single-max-context admission bound for the 4096-token in-flight setting. Even adding a conservative `B` output blocks leaves this below all four proposed minima. Larger or otherwise changed in-flight configuration can increase the floor, so the `serving_pool_budget` event reports the requested normalized budget/IDs, not a guarantee of the worker's final count. Device memory fit remains checked by the unchanged model-owned `validate_pool_memory` guard at allocation.

The KV override tests verify unchanged defaults and legacy behavior across B1/B8/B16/B32, exact accounting/logging fields for the generic override, invalid/conflicting/subminimum rejection, and retention of the measured-fit guard. Together with the precision-group tests, run:

```sh
python_env/bin/python -m unittest \
  models.autoports.google_gemma_4_31b_it.doc.agentic_evals.setup.swe85-precision-bisect.candidate-serving-prep.test_kv_pool_ids_override \
  models.autoports.google_gemma_4_31b_it.doc.agentic_evals.setup.swe85-precision-bisect.candidate-serving-prep.test_benchmark_work_precision_groups -v
```
