# DeepSeek-V4-Flash MoE Cache Population Design

## Goal

Add a standalone Python CLI that opens a TP1 or TP4 mesh, loads one selected
DeepSeek-V4-Flash MoE layer from its safetensors checkpoint, and populates the
same on-disk ttnn cache namespace used by the production model.

## Scope

The selected layer's complete MoE is cached:

- learned router weights and correction bias, or hash-router weights and
  token-to-expert table when the layer uses `hash_moe`;
- all shared-expert projections;
- all routed-expert gate/up/down weights, including TP-specific transformed
  entries.

The script does not construct attention, hyper-connections, norms, or other
decoder-layer weights, and it does not run inference.

## Interface

Create `models/experimental/deepseek_v4_flash/tools/cache_moe_weights.py`.
The command accepts a positional layer ID and a restricted `--tp` option:

```text
python models/experimental/deepseek_v4_flash/tools/cache_moe_weights.py \
    LAYER_ID --tp {1,4} [--model-dir MODEL_DIR] [--cache-dir CACHE_DIR]
```

The model directory defaults to the existing V4-Flash HuggingFace cache
location. The cache directory defaults to `DEEPSEEK_V4_CACHE_DIR`, or the
existing temporary cache location when that variable is unset. The cache root
includes the checkpoint directory basename, matching `DeepSeekV4Model`.

## Architecture and data flow

The CLI resolves the checkpoint snapshot, loads
`DeepseekV4Config`, selects the system profile for the opened mesh, and creates
a 1-device mesh for TP1 or a 1x4 mesh for TP4. It builds the production
`DeepSeekV4TopKRouter` or `DeepSeekV4HashRouter`,
`DeepSeekV4PreloadedExperts`, and `DeepSeekV4MLP` classes with a
`WeightCache` rooted at `layers.<layer_id>.mlp`.

The router and shared-expert weights are passed through the same lazy loader and
dequantization path used by the model. Constructing `DeepSeekV4PreloadedExperts`
loads or creates every routed-expert cache entry. Construction of the router
and shared MLP loads or creates their entries. No forward call is needed.

## Validation and errors

Before opening the device, validate that the layer ID is non-negative and less
than `config.num_hidden_layers`, that the layer's `mlp_layer_types` is either
`moe` or `hash_moe`, and that TP4 has a four-device mesh. Missing checkpoint
tensors, invalid cache paths, unsupported model geometry, and cache conversion
errors propagate with actionable messages.

The CLI reports the resolved snapshot, cache root, layer, TP size, and a
successful completion message. It closes the mesh in a `finally` block.

## Testing

Add host-only tests for parser defaults/restrictions, cache namespace
construction, and layer-type validation. Device-backed cache population is
verified manually with the CLI because it opens the shared accelerator.
