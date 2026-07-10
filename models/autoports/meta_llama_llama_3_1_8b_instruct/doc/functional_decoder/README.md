# Functional Decoder

Target: `meta-llama/Llama-3.1-8B-Instruct`

This stage translates the layer math from the forge-emitted TTNN graph at
`/localdev/mvasiljevic/ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0`.
The supplied graph is decode-shaped: batch 32, one active token, cache length 128. This artifact is
the repo-local functional prefill decoder and intentionally leaves decode pending.

## Runtime Contract

- `FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, batch=32, **kwargs)`
- `prefill_forward(hidden_states, *, position_cos, position_sin, attn_mask=None)`
- `decode_forward(...)` raises `NotImplementedError("decode path pending emitted-decode forge version...")`
- Runtime input shape: `hidden_states [1, batch, seq, 4096]`, default batch 32.
- Runtime output shape: `[1, batch, seq, 4096]`.
- Runtime batch must match the configured batch; this stage validates the emitted batch 32 workload.
- Runtime path has no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, or host fallback.

## Translated Semantics

- RMSNorm epsilon is Llama/HF `1e-5`; the Qwen-style `1e-6` and q/k norm notes do not apply to this emit.
- QKV is fused from HF weights. The actual Llama emit uses `[Q, V, K]` for layers 0-30 and a layer-31 decode reorder to `[Q, K, V]`; this functional single-layer prefill uses layer-0 `[Q, V, K]`.
- RoPE uses HF Llama-3 scaled RoPE tables built outside runtime and applied in TTNN runtime with the Llama `rotate_half` formula.
- Attention uses GQA with 32 Q heads, 8 KV heads, head_dim 128, and SDPA scale `1/sqrt(128)`.
- MLP is SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

## Validation

| Command | Result |
| --- | --- |
| `timeout 60 tt-smi -ls --local` | unavailable: `tt-smi` not on PATH |
| `timeout 120 python - <<'PY' ... open_mesh_device(MeshShape(1, 1)) ... PY` | `MESH_SMOKE_OK` |
| `timeout 600 pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py --tb=short -s` | 5 passed |

PCC results from the full test command:

| Test | PCC |
| --- | ---: |
| Synthetic weights, batch 32, seq 1 | 0.998992 |
| Synthetic weights, batch 32, seq 8 | 0.998662 |
| Real weights, layer 0, batch 32, seq 4 | 0.999998 |

## Limitations

- Prefill-only functional stage.
- Decode is pending the emitted-decode integration contract and is a documented stub.
- No paged KV path is implemented.
- The forge source graph is decode-shaped, not prefill-shaped.
- HF advertises context 131072; this stage only validated single-layer prefill through seq 8.
- `doc/context_contract.json` records `current_supported_context=8` for the current evidence. The repo-local checker currently fails non-DRAM context reductions; this is recorded as a checker/stage-contract mismatch rather than hidden by overstating support.
- `ttnn.experimental.rotary_embedding` padded prefill sequence to 32 on this path, so the runtime uses an equivalent TTNN rotate-half formula for arbitrary prefill sequence correctness.

`forge_sharding_recommendations.json` records emitted layout/program-config provenance for the future optimize stage; those sharding choices are not copied into the functional runtime.
