# Qwen3.6-27B Galaxy server — DeltaNet-as-vLLM-mamba-cache rework plan

## Why
The server decode currently works only in **eager** mode (~3 tok/s) and the **trace**
path degrades over length + corrupts across requests. Root cause: we override
`architectures -> Qwen3ForCausalLM` (pure attention), so vLLM never manages the
DeltaNet recurrent state. We hold it in internal model buffers + a manual
`clear_state()` hack, which doesn't survive trace reuse across requests.

The fix (proven by `tt-inference-server@atupe/qwen35-9b-inf-server`, impl
`qwen35_9b_blackhole`): let vLLM treat the model as a **hybrid SSM/attention** model
and manage the DeltaNet state as a **mamba cache** it allocates + resets per request.

## Blueprint (vLLM `Qwen3NextForCausalLM`)
- `vllm/vllm/model_executor/models/qwen3_next.py`:
  - `class Qwen3NextGatedDeltaNet(nn.Module, MambaBase)` — the linear/delta layer
    declares itself a mamba layer.
  - `Qwen3NextForCausalLM(..., IsHybrid)`.
  - GatedDeltaNet.forward reads `self_kv_cache[0]`=conv_state, `[1]`=ssm_state, with
    `conv_state_indices` / `ssm_state_indices` = per-request cache slots.
- `vllm/vllm/v1/kv_cache_interface.py:244 MambaSpec(KVCacheSpec)` — shapes/dtypes of
  the (conv_state, ssm_state) cache; vLLM sizes + resets it per request.

## Changes (file-level)
1. **Config / registration** (tt-inference-server plugin + catalog):
   - hf_overrides `architectures -> ["Qwen3NextForCausalLM"]` (NOT Qwen3ForCausalLM)
     so vLLM marks the model `is_hybrid` + builds a MambaSpec for linear layers.
   - Register `TTQwen3NextForCausalLM -> qwen3.6 TT class` in plugin `__init__.py`.
   - Config must expose the per-layer type (linear_attention vs full_attention) +
     GatedDeltaNet dims (conv_kernel, head_dim, n_v_heads, etc.) so vLLM/the runner
     can size the MambaSpec. Extend `qwen3_5_config.py`.
   - Catalog: `no-enable-prefix-caching: true` (avoids align-mamba chunked-prefill),
     `trace_region_size: 256MB`.
2. **Plugin `tt_worker.get_kv_cache_spec`**: return a HYBRID spec — `MambaSpec` for the
   linear-attn layers (conv_state + ssm_state shapes), `FullAttentionSpec` for the
   full-attn layers. Currently returns FullAttentionSpec for all.
3. **Plugin `allocate_kv_cache` / model_runner**: allocate the mamba cache tensors
   (conv_state, ssm_state) on the mesh with the model's row-shard layout; thread the
   per-request `state_indices` into decode/prefill.
4. **Model `qwen36_delta_attention.py` (the deep part)**: make the DeltaNet
   forward_prefill / forward_decode read/write the **passed** conv_state + ssm_state at
   the per-request `state_indices`, instead of the internal `dn_state_buffer` /
   `conv_state_buffer`. Drop `clear_state()` (vLLM resets via fresh slots).
   - TT reference to adapt: the `qwen35_9b` model (tt_metal_commit 07e4d5fe7c4; likely
     near branch `alnah005/deltanet_work`) already does TT-side mamba-cache integration.
5. **Remove** the manual per-request `clear_state()` + `reset_gather` hack in
   `generator.prefill_forward_text` once vLLM owns the state.

## Order of work
(1) get the qwen35_9b TT model as the concrete template ->
(2) plugin hybrid kv_cache_spec + allocate (MambaSpec) ->
(3) model DeltaNet forward using passed cache + state_indices ->
(4) config/registration (Qwen3NextForCausalLM) -> (5) test trace decode coherence +
cross-request + perf (target ~25 tok/s like the demo).

## Keep working in the meantime
Eager decode (`override_tt_config.trace_mode: false`) is the committed, correct
fallback. Do this rework on a branch; don't destabilize the eager server.
