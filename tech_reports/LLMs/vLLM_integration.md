# Integrating TT Models into vLLM

## Overview
vLLM is an [open-source LLM serving library](https://github.com/vllm-project/vllm). We use vLLM to serve our models in production because of the features it enables. On the serving side, vLLM supports continuous batching (see [LLMs Tech Report - Continuous Batching](./llms.md#34-continuous-batching) for more info) and [paged attention](https://arxiv.org/pdf/2309.06180). In addition, vLLM provides an OpenAI-compatible server which is useful for deployment.

Tenstorrent maintains a [fork of vLLM](https://github.com/tenstorrent/vllm/tree/dev) for serving models on Tenstorrent hardware. The [vLLM README](https://github.com/tenstorrent/vllm/tree/dev/tt_metal/README.md) has instructions for setting up the environment and running the inference example.

**Quick Links for vLLM's public docs**:
- vLLM Docs Homepage: https://docs.vllm.ai/en/latest
- Contributing to vLLM: https://docs.vllm.ai/en/latest/contributing/overview.html
- Architecture Overview: https://docs.vllm.ai/en/latest/design/arch_overview.html

## Implementation Requirements for Model Integration
In order to add vLLM support to a new Tenstorrent model, the following requirements must be met:

1. **The model must implement paged attention** using the TT-NN `paged_fill_cache`, `paged_update_cache`, and `paged_scaled_dot_product_attention_decode` ops (see [LLMs Tech Report - Attention](./llms.md#24-attention) and [LLMs Tech Report - Prefill and Decode](./llms.md#32-prefill-and-decode) for more info). An example usage of these ops is in the `forward_prefill` and `forward_decode` functions in [models/tt_transformers/tt/attention.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/attention.py) (part of the [TT-Transformers](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) library).

2. **The model generation class must conform to a specific interface**. An example generation class is `LlamaForCausalLM` in [models/tt_transformers/tt/generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py). The class must have the following functions:
    - `initialize_vllm_model`: class method which returns an instance of the model. In vLLM, this function is used by `TTModelLoader::load_model` in [tt_loader.py](https://github.com/tenstorrent/vllm/blob/dev/vllm/model_executor/model_loader/tt_loader.py). The `max_batch_size` argument represents the total batch size when combining all data parallel (DP) groups, and `tt_data_parallel` represents the DP factor of attention (for a fully DP model this is just the number of model replicas).
      ```python
      initialize_vllm_model(cls, hf_config : transformers.PretrainedConfig, mesh_device : ttnn.MeshDevice, max_batch_size : int, tt_data_parallel : int, optimizations : str)
      ```
    - `allocate_kv_cache`: returns the paged kv cache which will be passed to the model during inference. The `kv_cache_shape` argument has shape `(max_num_blocks, num_kv_heads, block_size, head_size)`. In vLLM V0, this function is used by `TTCacheEngine::_allocate_kv_cache` in [tt_worker.py (V0)](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_worker.py). In vLLM V1, it's used by `TTModelRunner::initialize_kv_cache` in [tt_model_runner.py (V1)](https://github.com/tenstorrent/vllm/blob/dev/vllm/v1/worker/tt_model_runner.py).
      ```python
      allocate_kv_cache(kv_cache_shape : tuple, dtype : torch.dtype, num_layers : int)
      ```
    - `prefill_forward` (**text-only models**): returns the prefill outputs on host. The `tokens` argument has shape `(batch_size, max_prompt_len)` and has been zero-padded along the last dim to the length of the longest prompt in the batch. `page_table` has shape `(batch_size, num_blocks)` and has been zero-padded along the last dim to the max number of blocks in the batch. `prompt_lens` has shape `(batch_size)`. The `sampling_params` argument is a dataclass with sampling attributes such as `temperature`, `top_p`, `top_k` (note: the current default in vLLM is to not pass in this argument and instead sample on host, unless sampling on device is enabled explicitly). For fully-DP or DP-attention models, the `empty_slots` argument is used to indicate request positions in the global DP batch (only to be used for determining the DP group of a request, not for the actual request index in the batch which can change). In vLLM V0, this function is used by `TTModelRunner::_execute_model_single_step` in [tt_model_runner.py (V0)](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py). In the same file, the preparation of the inputs takes place in `TTModelRunner::prepare_model_input`. In vLLM V1, this function is used by `TTModelRunner::execute_with_model_input` in [tt_model_runner.py (V1)](https://github.com/tenstorrent/vllm/blob/dev/vllm/v1/worker/tt_model_runner.py) and the preparation of the inputs takes place in `TTModelRunner::_prepare_model_inputs`.
      ```python
      prefill_forward(tokens : torch.Tensor, page_table : torch.Tensor, kv_cache : list, prompt_lens : torch.Tensor, sampling_params : TTSamplingParams, empty_slots : list)
      ```
    - `decode_forward` (**text-only models**): returns the decode outputs on host if `read_from_device=True` (default True in V1 and False in V0) otherwise on device. The `tokens` argument has shape `(max_batch_size, 1)` and has been zero-padded along the batch dim to the max batch size (along with `start_pos` with shape `(max_batch_size)` and `page_table` with shape `(max_batch_size, max_num_blocks)`). For fully-DP or DP-attention models, each DP group's batch is padded and the batches are concatenated. The decode inputs are intentionally padded to `max_batch_size` and `max_num_blocks` since the default behaviour in vLLM is to use `enable_trace=True` and TT-NN tracing requires constant input shapes. Similar to `prefill_forward`, the `sampling_params` argument is a dataclass with sampling attributes such as `temperature`, `top_p`, `top_k` (note: the current default in vLLM is to not pass in this argument and instead sample on host, unless sampling on device is enabled explicitly). In vLLM V0, this function is used by `TTModelRunner::_execute_model_single_step` in [tt_model_runner.py (V0)](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py). In vLLM V1, this function is used by `TTModelRunner::execute_with_model_input` in [tt_model_runner.py (V1)](https://github.com/tenstorrent/vllm/blob/dev/vllm/v1/worker/tt_model_runner.py).
      ```python
      decode_forward(tokens : torch.Tensor, start_pos : torch.Tensor, page_table : torch.Tensor, kv_cache : list, enable_trace : bool, read_from_device : bool, sampling_params : TTSamplingParams)
      ```
    - `read_decode_output` (V0 only): returns the ttnn decode outputs on host. The `tt_out` argument is the output of `decode_forward`. This function is intentionally separate from `decode_forward` to implement the asynchronous output processing + multi-step scheduling optimization (for more info see [vLLM v0.6.0 Blog Post](https://blog.vllm.ai/2024/09/05/perf-update.html)) where the execution of `decode_forward` (with `read_from_device=False`) is nonblocking on CPU, allowing for overlap of vLLM output processing (of previous decode iterations) with the current decode execution. In vLLM, this function is used by `TTModelRunner::_execute_model_single_step` in [tt_model_runner.py (V0)](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py).
      ```python
      read_decode_output(tt_out : ttnn.Tensor, async_read : bool)
      ```
    - `process_decode_output_host` (V0 only): returns torch decode outputs on host. The `tt_out` argument is the output of `read_decode_output`. The `is_tokens` argument specifies whether `tt_out` is logits or tokens. In vLLM, this function is used by `TTModelRunner::_execute_model_single_step` in [tt_model_runner.py (V0)](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py).
      ```python
      process_decode_output_host(tt_out : ttnn.Tensor, is_tokens : bool)
      ```
    - `warmup_model_prefill` (V0 and V1): wrapper function that calls the model specific prefill warmup function that compiles the model and captures the traces (for prefill). The `kv_cache`, `enable_trace` and `sampling_params` are passed from vLLM. The `enable_trace` [True/False] argument determines whether tracing in prefill will be used or not. The `sampling_params` [List] argument is a list of all types of samplings that a model can perform. The `sampling_params` list argument contains `TTSamplingParams` objects, and always contains at the end of the list a `None` value which resembles host sampling.
      ```python
      warmup_model_prefill(kv_cache : list, enable_trace : bool, sampling_params : list)
      ```
    - `model_capabilities`: Class dictionary that lets VLLM know about the
    features supported by this model. We use it in
    [tt.py](https://github.com/tenstorrent/vllm/blob/dev/vllm/platforms/tt.py)
    to decide about enabling or disabling those features in VLLM.
    Currently there is only one feature (`supports_prefix_caching`)
    that we describe here, but the list is expected to grow. Example:
    `model_capabilities={"supports_prefix_caching": True}`
3. **(Multi-modal models only)** Currently, we only support image+text input modalities. An example generation class is `Gemma3ForConditionalGeneration` in [models/tt_transformers/tt/generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py). For more info on multi-modal models see also [vLLM Docs - Multi-Modal Support](https://docs.vllm.ai/en/latest/contributing/model/multimodal.html)). These models have the same interface requirements as the text-only models, as well as the following:
   - `prefill_forward` (**image+text models**): same as text-only models with an additional kwarg (`images` for V0, `pixel_values` for V1) for the image inputs.

4. **(V0 encoder-decoder multi-modal models only)** An example generation class is `MllamaForConditionalGeneration` in [models/tt_transformers/tt/generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py). These models have the same interface requirements as the text-only models, as well as the following:
   - `max_cross_attn_tokens`: class property which returns the max number of tokens in cross attention. In vLLM, this property is used in [tt_model_runner.py (V0)](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py) for padding cross page tables to `max_cross_blocks`.
   - `prefill_forward` (**image+text models**): returns `logits, cross_attention_masks, full_text_row_masked_out_mask`. The input arguments are the same as the text-only models with the addition of `images` and `cross_page_table` (the page table for paged cross attention which accesses the same `kv_cache` as self attention). `cross_page_table` has shape `(batch_size, num_vision_blocks)`.
     ```python
      prefill_forward(tokens : torch.Tensor, images : List[List[PIL.Image.Image]], page_table : torch.Tensor, kv_cache : list, prompt_lens : torch.Tensor, cross_page_table: torch.Tensor, sampling_params : TTSamplingParams, empty_slots : list)
      ```
   - `decode_forward` (**image+text models**): same as the text-only models with the additional input arguments `cross_attention_masks` (output from prefill), `full_text_row_mask_out_mask` (output from prefill), and `cross_page_table`. `cross_page_table` has shape `(max_batch_size, max_cross_blocks)`.
     ```python
      decode_forward(tokens : torch.Tensor, start_pos : torch.Tensor, cross_attention_masks : list, full_text_row_masked_out_mask : list, page_table : torch.Tensor, kv_cache : list, cross_page_table : torch.Tensor, enable_trace : bool, read_from_device : bool, sampling_params : TTSamplingParams)
      ```
   - A custom vLLM input processor for preprocessing encoder/decoder tokens may also be required (for an example, see `MllamaMultiModalProcessor` in [models/tt_transformers/tt/generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py).

## Testing the Model in vLLM
Once the model meets all of the requirements specified in [Implementation Requirements for Model Integration](#implementation-requirements-for-model-integration), it can be tested in vLLM by following the instructions in the [vLLM README](https://github.com/tenstorrent/vllm/tree/dev/tt_metal/README.md) and doing the following:
1. The model needs to be registered using `ModelRegistry.register_model` and added to the list of supported models in [vllm/platforms/tt.py](https://github.com/tenstorrent/vllm/blob/dev/vllm/platforms/tt.py).
2. Testing offline inference, continuous batching, and performance (see [Running the Offline Inference Example](https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md#running-the-offline-inference-example)).
3. Testing various sequence lengths of increasing size using the `--test_increasing_seq_lens` option of [examples/offline_inference_tt.py](https://github.com/tenstorrent/vllm/blob/dev/examples/offline_inference_tt.py).
4. Testing model serving of asynchronous requests with the server example (see [Running the Server Example](https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md#running-the-server-example)).

## vLLM Modifications
Occasionally, additional changes may be needed on the vLLM side to support a new model (e.g. supporting a new type of model, inputs of a different modality, or customizing the KV cache initialization). If this is the case, please make a pull request to the [dev branch](https://github.com/tenstorrent/vllm/tree/dev) (which acts as our main branch, and is not permitted to be committed to directly). The main files for our TT vLLM backend are the following:
- [`tt.py`](https://github.com/tenstorrent/vllm/blob/dev/vllm/platforms/tt.py): handles platform definition and model registration.
- [`tt_loader.py`](https://github.com/tenstorrent/vllm/blob/dev/vllm/model_executor/model_loader/tt_loader.py): handles model initialization.
- [`tt_worker.py (V0)`](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_worker.py) / [`tt_worker.py (V1)`](https://github.com/tenstorrent/vllm/blob/dev/vllm/v1/worker/tt_worker.py): handles device initialization and kv cache management.
- [`tt_model_runner.py (V0)`](https://github.com/tenstorrent/vllm/blob/dev/vllm/worker/tt_model_runner.py) / [`tt_model_runner.py (V1)`](https://github.com/tenstorrent/vllm/blob/dev/vllm/v1/worker/tt_model_runner.py): handles input preparation and model execution.
