# Gemma4 cache layout

## On disk

| Location | Contents |
|---|---|
| Local directory supplied as `HF_MODEL`, or Hugging Face's hub cache (normally `$HF_HOME/hub`) | Original checkpoint weights, model configuration and tokenizer files. `HF_HOME` defaults to `~/.cache/huggingface`. |
| TT cache root | `TT_CACHE_PATH` if set; otherwise the local checkpoint directory, or `$HF_HOME/tt_cache/google--gemma-4-31B-it` for the model ID. |
| `<TT cache root>/tensor_cache_<dtype>_mesh<rows>x<cols>/` | Converted weight cache for a dtype and mesh layout, e.g. `tensor_cache_bf16_mesh8x4`. |
| `*.tensorbin` beneath that directory | Serialized TT tensors, including converted/sharded device weights. Module-specific dtype settings also affect filenames. |
| `.host_weights.pt` in that directory | PyTorch dictionary of token embeddings and decoder-layer scalars needed on the CPU. |
| `.weights_complete.<variant-digest>` in that directory | Completion metadata identifying the model, layers, mesh, build settings and cached tensor files. |

A valid completion marker lets initialization load device weights from tensorbins
and CPU weights from `.host_weights.pt`, skipping the full checkpoint load.
`GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1` forces checkpoint loading in the demo.
Cache-path resolution requires existing directories; it does not create or mirror them.

## In memory

- **CPU:** embeddings and layer scalars loaded from `.host_weights.pt`; the full
  state dict is also loaded when building from the checkpoint.
- **Device DRAM:** loaded model weights, global/sliding KV caches, RoPE tables,
  metadata tensors and persistent communication buffers. These runtime caches
  are separate from the weight files on disk.
- **Device trace memory:** captured execution traces, within the reserved trace region.

Implementation: `tt/model_config.py` resolves paths, `tt/common.py` selects the
loading path, and `models/common/weight_cache.py` manages host weights and completion markers.
