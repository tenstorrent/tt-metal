# Cache contract integration (source-only)

`tt/cache_contract.py` is model-neutral storage/lifecycle support, not a vLLM adapter. It reads the selected layer's `kv_dtype`; it does not choose a new precision policy. Pinned interface: `vllm-tt-plugin` commit `042a94863ad27d88207bcf71abdf67122c6f4c27`, `docs/MODEL_SETUP.md`, `docs/MODEL_LIFECYCLE.md`, and `model_runner.py` `_kv_cache_shape` / `_build_per_layer_specs` / `_allocate_kv_caches`.

Current physical policy is BF16 tiled K/V `[physical_pages,1,32,256]` per chip for each full-attention layer, BF16 row-major convolution `[1,3,2048]` and FP32 tiled recurrent `[1,8,128,128]` per chip for every request/linear layer, and INT32 row-major page tables. There is no heterogeneous full-attention KV dtype in the current selected model. The precision split is preserved across state types; future selected per-layer KV differences are validated rather than silently homogenized. Head ownership stays `[0,0,1,1]` on TP4; no gather or cache replication fallback is added.

`describe_cache(model)` supplies JSON-compatible specifications and the attention-ordinal → original HF layer mapping. The plugin's allocator counts attention layers, so names are `model.layers.0.self_attn` through `.9.self_attn`, mapping to HF 3,7,...39. It advertises original `num_key_value_heads=2`, leaving the plugin's TP formula to produce one local head. `aggregate_token_capacity` is the configured pool capacity (maximum 262144), not 32 multiplied by context, and is a source contract, not new measured capacity evidence.

`allocate_kv_cache_per_layer(model, specs, batch=...)` accepts triples `(shape,dtype,tensor_idx)` in full-attention order and returns the existing full-stack nested cache representation. It adds separately allocated recurrent/conv state at linear layers. Equal `tensor_idx` shares the exact K/V objects; unequal indices allocate separate pools. Shape/dtype disagreement for one shared index is rejected before allocating. `allocate_kv_cache` wraps uniform legacy plans and rejects selected layer dtype disagreement. Minimum physical pages remain `8*ceil(context/256)`, including SDPA's rounded read window. The existing `new_kv_state` owns actual allocation and initial page-table padding.

Initial identity page tables are placeholders for multi-request or cross-layer shared pools. A scheduler must assign disjoint writable logical pages. `validate_page_ownership(cache, tables, slots=..., prompt_lens=...)` checks CPU tables for all live requests and sharing layers, excluding padded read-only entries. Do not validate only the newly submitted rows: retained requests also own pages. Page allocations are scheduler data, so this validation belongs at admission/page-table change, never per resident decode token.

Small root integration changes:

```python
# QwenGenerator constructor, immediately after selecting self.cache:
from .cache_contract import validate_cache
validate_cache(model, self.cache, batch=max_batch_size)

# Model-neutral model methods (before construction-time generator binding):
def cache_spec(self):
    from .cache_contract import describe_cache
    return describe_cache(self)

def allocate_kv_cache_per_layer(self, per_layer_specs, *, batch=1):
    from .cache_contract import allocate_kv_cache_per_layer
    return allocate_kv_cache_per_layer(self, per_layer_specs, batch=batch)

# build_generator: after model creation, before QwenGenerator(...):
cache = kwargs.get("kv_cache")
specs = kwargs.get("per_layer_cache_specs")
if cache is not None and specs is not None:
    raise ValueError("Provide kv_cache or per_layer_cache_specs")
if specs is not None:
    cache = model.allocate_kv_cache_per_layer(specs, batch=kwargs.get("max_batch_size", 1))
# Pass kv_cache=cache into the existing constructor; prepare happens afterward.

# QwenGenerator lifecycle hooks:
def release_request(self, slot):
    from .cache_contract import release_request
    return release_request(self, slot)

def release_persistent_capture(self):
    return self.close()
```

`validate_cache` checks exact slot count, logical context matching the configured generator, local shapes/layout/dtypes/DRAM placement, distinct per-slot mutable storage, separate K/V allocations, whole-pair sharing, and one aggregate pool per full layer. It is metadata-only. A plain external list remains supported; helper-specific metadata is not required.

`release_request` requires drained output leases, then zeros only that slot's conv/recurrent buffers, clears its position/activity, resets its sampler parameters/RNG and history, and calls common `seed_manager.release_slot`. It preserves physical shared KV, page mappings, other slots and trace addresses. Inactive token bits remain masked and are overwritten on authoritative input reload. It rejects active device penalties, consistent with this stage's declared host-logit route for penalties. Failed mutations poison state until whole-generator reset. `note_state_slots_moved` need not repeat any gather: existing `SlotState.apply` already permutes all identities and request data; a future wrapper may have additional request-ID bookkeeping.

Validation performed on 2026-11-14: `python -m py_compile .../tt/cache_contract.py`; CPU import and synthetic metadata check passed valid shared BF16 descriptors and ordinal mapping, and rejected six malformed plans (count, capacity, local head geometry, precision, shared-index geometry mismatch, Boolean index). No devices opened, no hardware allocation/release test run by this subagent.

Serving inventory closure remains evidence-dependent. These helpers provide source implementations for `get_kv_cache_spec`, `allocate_kv_cache_per_layer`, optional `allocate_kv_cache`, external cache binding before `initialize_vllm_model`'s future factory mapping, and `release_request`. Required follow-up: root integration; actual external-cache prefill/decode and slot-release isolation evidence; shared-page ownership admission integration; measured full-stack capacity for `get_max_tokens_all_users`; generator-wide warmup/capture evidence; shutdown release evidence. Do not mark those rows supported from this document alone. No vLLM class, registration, server or adapter was implemented.
