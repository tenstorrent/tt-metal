# Integrating TT Models into vLLM

## Overview
vLLM is an [open-source LLM serving library](https://github.com/vllm-project/vllm). We use vLLM to serve our models in production because of the features it enables. On the serving side, vLLM supports continuous batching (see [LLMs Tech Report - Continuous Batching](./llms.md#34-continuous-batching) for more info) and [paged attention](https://arxiv.org/pdf/2309.06180). In addition, vLLM provides an OpenAI-compatible server which is useful for deployment.

Tenstorrent maintains a [fork of vLLM](https://github.com/tenstorrent/vllm/tree/dev) for serving models on Tenstorrent hardware. The [README](https://github.com/tenstorrent/vllm/tree/dev/tt_metal/README.md) has instructions for setting up the environment and running the inference example.

**Quick Links for vLLM's public docs**:
- vLLM Docs Homepage: https://docs.vllm.ai/en/latest
- Contributing to vLLM: https://docs.vllm.ai/en/latest/contributing/overview.html
- Architecture Overview: https://docs.vllm.ai/en/latest/design/arch_overview.html

## Implementation Requirements for Model Integration
In order to add vLLM support to a new Tenstorrent model, the following requirements must be met:

1. **The model must implement paged attention** using the TT-NN `paged_fill_cache`, `paged_update_cache`, and `paged_scaled_dot_product_attention_decode` ops (see [LLMs Tech Report - Attention](./llms.md#24-attention) and [LLMs Tech Report - Prefill and Decode](./llms.md#32-prefill-and-decode) for more info). An example usage of these ops is in the `forward_prefill` and `forward_decode` functions in [models/tt_transformers/tt/attention.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/attention.py) (part of the [TT-Transformers](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) library).

2. **The model generation class must conform to a specific interface**. An example generation class is `LlamaForCausalLM` in [models/tt_transformers/tt/generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py). The class must have the following functions:
    - `initialize_vllm_model`: class method which returns an instance of the model. In vLLM, this function is used by `TTModelLoader::load_model` in [tt_loader.py](https://github.com/tenstorrent/vllm/blob/dev/vllm/model_executor/model_loader/tt_loader.py).
      ```python
      initialize_vllm_model(cls, hf_config : transformers.PretrainedConfig, mesh_device : ttnn.MeshDevice, max_batch_size : int)
      ```
    - `allocate_kv_cache`: returns the paged kv cache which will be passed to the model during inference. The `kv_cache_shape` argument has shape `(num_blocks, num_kv_heads, block_size, head_size)`. In vLLM, this function is used by `TTCacheEngine::_allocate_kv_cache` in [tt_worker.py](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_worker.py).
      ```python
      allocate_kv_cache(kv_cache_shape : tuple, dtype : torch.dtype, num_layers : int, mesh_device : ttnn.MeshDevice)
      ```
    - `prefill_forward`: returns the prefill output logits on host. The `tokens` argument has shape `(batch_size, max_prompt_len)` and has been zero-padded along the last dim to the length of the longest prompt in the batch. `page_table` has shape `(batch_size, num_blocks)` and has been zero-padded along the last dim to the max number of blocks in the batch. `prompt_lens` has shape `(batch_size)`. In vLLM, this function is used by `TTModelRunner::_execute_model_single_step` in [tt_model_runner.py](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py). In the same file, the preparation of the inputs takes place in `TTModelRunner::prepare_model_input`.
      ```python
      prefill_forward_text(tokens : torch.Tensor, page_table : torch.Tensor, kv_cache : list, prompt_lens : torch.Tensor)
      ```
    - `decode_forward`: returns the decode output logits on device.
    - `read_decode_output`: 

4. 


## vLLM modifications
On the vLLM side there may be additional changes needed to support the new model.

- Modify [`tt_loader.py`](https://github.com/tenstorrent/vllm/blob/dev/vllm/model_executor/model_loader/tt_loader.py) if the model requires a different initialization.
- Modify [`tt_model_runner.py`](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py) if it is missing functionality for the new model.
