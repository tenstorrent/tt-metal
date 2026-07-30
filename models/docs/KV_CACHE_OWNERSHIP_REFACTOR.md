# KV Cache Ownership Refactor

**Status: implemented (model-owns, deferred init).**

## Summary

The TT model **owns** its KV cache. Each attention layer holds its own cache in
`attention.layer_past`, rather than the cache living in the vLLM runner /
generator and being threaded into every forward call.

The forwards and warmups no longer take a `kv_cache` argument:
`prefill_forward`, `decode_forward`, `warmup_model_prefill`, and
`warmup_model_decode`.

## Allocation

A single allocation routine (`init_kv_cache`) writes each
`attention.layer_past`. There are two entry points:

- **Demo / standalone:** the model is built with `create_kv_cache=True`
  (default), so the cache is allocated at construction time. (This flag
  replaces the old `use_paged_kv_cache`; the shape still comes from
  `paged_attention_config`.)
- **vLLM (deferred init):** the model is built with `create_kv_cache=False`,
  skipping allocation at construction. vLLM later calls
  `Model.allocate_kv_cache(per_layer_specs)` (or
  `allocate_kv_cache_per_layer` for hybrid models), which runs the SAME routine
  parameterized by vLLM's specs and installs the cache onto the model. Deferred
  init keeps a single vLLM path that covers both uniform and hybrid models and
  future-proofs for memory profiling.

There is no separate `install_kv_cache(prebuilt_tensors)` step and no
`generator.kv_cache` / `runner.kv_caches` handle. The old free functions that
built tensors externally (`allocate_vllm_kv_cache*`) now live inside the model.

## Sentinel

`layer_past` is `None` until allocated. `_assert_kv_cache_ready()` runs at the
top of every public forward / warmup and raises a clear error if the cache was
never installed, with a backstop in attention. This is a model-readiness check,
not generator ownership.

## per_layer_specs (vLLM)

`per_layer_specs` is a list (one entry per attention layer, in model
layer-index order) of `(shape, dtype, tensor_idx)` tuples, built by the plugin's
`_build_per_layer_specs` from vLLM's `kv_cache_config`. `num_blocks` is consumed
from the passed specs and never recomputed.

`allocate_kv_cache` composes two orthogonal KV-sharing axes:

1. **`tensor_idx` (HMA buffer packing):** layers carrying the same `tensor_idx`
   share the SAME tensor object. This comes from vLLM core
   (`kv_cache_config.kv_cache_tensors[i].shared_by`, tied to BlockPool block
   IDs).
2. **`kv_shared_layer_map` (architectural KV reuse):** e.g. Gemma 4, where the
   last `num_kv_shared_layers` reuse an earlier layer's cache (from
   `hf_config.num_kv_shared_layers`). Applied on top of the buffer packing.

Per-layer page-table routing (`page_tables_per_layer`) is a separate concern
from the cache itself and is unchanged.

## Trace caveat

Reassigning `layer_past` after a trace has been captured leaves the trace
pointing at a stale tensor. Re-warm up (re-capture traces) if the cache is
reallocated.

## See also

- [tech_reports/LLMs/vLLM_integration.md](../../tech_reports/LLMs/vLLM_integration.md) — model-interface signatures.
- vLLM TT plugin README — KV cache ownership section (`model_runner.py` `initialize_kv_cache`).
