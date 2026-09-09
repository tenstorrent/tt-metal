# Integrating TT Models into vLLM

## Overview
vLLM is an [open-source LLM serving library](https://github.com/vllm-project/vllm). We use vLLM to serve our models in production because of the features it enables. On the serving side, vLLM supports continuous batching (see [LLMs Tech Report - Continuous Batching](./llms.md#34-continuous-batching) for more info) and [paged attention](https://arxiv.org/pdf/2309.06180). In addition, vLLM provides an OpenAI-compatible server which is useful for deployment.

Tenstorrent integrates with upstream vLLM through the
[vLLM TT plugin](https://github.com/tenstorrent/vllm-tt-plugin). Its README has
instructions for setting up the environment and running inference.

**Quick Links for vLLM's public docs**:
- vLLM Docs Homepage: https://docs.vllm.ai/en/latest
- Contributing to vLLM: https://docs.vllm.ai/en/latest/contributing/overview.html
- Architecture Overview: https://docs.vllm.ai/en/latest/design/arch_overview.html

## Implementation Requirements for Model Integration
In order to add vLLM support to a new Tenstorrent model, the following requirements must be met:

1. **The model must implement paged attention** using the TT-NN `paged_fill_cache`, `paged_update_cache`, and `paged_scaled_dot_product_attention_decode` ops (see [LLMs Tech Report - Attention](./llms.md#24-attention) and [LLMs Tech Report - Prefill and Decode](./llms.md#32-prefill-and-decode) for more info). An example usage of these ops is in the `forward_prefill` and `forward_decode` functions in [models/tt_transformers/tt/attention.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/attention.py) (part of the [TT-Transformers](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) library).

2. **The model generation class must conform to a specific interface**. An example generation class is `LlamaForCausalLM` in [models/tt_transformers/tt/generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py). The class must have the following functions:
    - `initialize_vllm_model`: class method which returns an instance of the model. In vLLM, this function is used by `TTModelLoader::load_model` in [src/vllm_tt_plugin/loader.py](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/loader.py). The `max_batch_size` argument is the engine-local capacity: per-rank for standard multi-process DP and the merged capacity across in-process lanes for lane-DP. `tt_data_parallel` is the number of TT KV-cache replicas/submeshes owned by that model instance.
      ```python
      initialize_vllm_model(cls, hf_config : transformers.PretrainedConfig, mesh_device : ttnn.MeshDevice, max_batch_size : int, max_seq_len : int, tt_data_parallel : int, optimizations : str | None)
      ```
    - `allocate_kv_cache`: returns the paged kv cache which will be passed to the model during inference. The `kv_cache_shape` argument has shape `(max_num_blocks, num_kv_heads, block_size, head_size)`. This function is used by `TTModelRunner::initialize_kv_cache` in [src/vllm_tt_plugin/model_runner.py](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/model_runner.py).
      ```python
      allocate_kv_cache(kv_cache_shape : tuple, dtype : torch.dtype, num_layers : int)
      ```
    - `prefill_forward` (**text-only models**): returns the prefill outputs on host. The `tokens` argument has shape `(batch_size, max_prompt_len)` and has been zero-padded along the last dim to the length of the longest prompt in the batch. `page_table` has shape `(batch_size, num_blocks)` and has been zero-padded along the last dim to the max number of blocks in the batch. `prompt_lens` has shape `(batch_size)`. The `sampling_params` argument is a dataclass with sampling attributes such as `temperature`, `top_p`, `top_k` (note: the current default in vLLM is to not pass in this argument and instead sample on host, unless sampling on device is enabled explicitly). `empty_slots` identifies the exact engine-local physical state slots being initialized: rank-local under standard multi-process DP and merged across in-process lane-DP. This function is used by `TTModelRunner::submit_prefill` in [src/vllm_tt_plugin/model_runner.py](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/model_runner.py).
      ```python
      prefill_forward(tokens : torch.Tensor, page_table : torch.Tensor | None = None, kv_cache : list | None = None, prompt_lens : torch.Tensor | None = None, empty_slots : list[int] | None = None, enable_trace : bool = True, sampling_params : TTSamplingParams | None = None, start_pos : torch.Tensor | list[int] | None = None, **kwargs)
      ```
    - `decode_forward` (**text-only models**): returns the decode outputs on host if `read_from_device=True` (default True) otherwise on device. The `tokens` argument has shape `(max_batch_size, 1)` and has been zero-padded along the batch dim to the max batch size (along with `start_pos` with shape `(max_batch_size)` and `page_table` with shape `(max_batch_size, max_num_blocks)`). For fully-DP or DP-attention models, each DP group's batch is padded and the batches are concatenated. The decode inputs are intentionally padded to `max_batch_size` and `max_num_blocks` since the default behaviour in vLLM is to use `enable_trace=True` and TT-NN tracing requires constant input shapes. Similar to `prefill_forward`, the optional `sampling_params` argument is a dataclass with sampling attributes such as `temperature`, `top_p`, `top_k`. For eligible structured decode, the plugin sets `defer_device_sampling=True` and deliberately omits `sampling_params`; `decode_forward` then returns an opaque one-shot payload containing device logits and authoritative decode state instead of sampling immediately.
      ```python
      decode_forward(tokens : torch.Tensor, start_pos : torch.Tensor, page_table : torch.Tensor, kv_cache : list, enable_trace : bool = True, read_from_device : bool = True, sampling_params : TTSamplingParams | None = None, reset_batch : bool = False, prompt_tokens : torch.Tensor | None = None, output_tokens : torch.Tensor | None = None, slot_remap = None, defer_device_sampling : bool = False, grammar_bitmask : torch.Tensor | None = None, skip_trace_precompile : bool = False, **kwargs)
      ```
      The plugin later completes that payload after vLLM produces its packed
      `int32` grammar mask:
      ```python
      sample_decode_on_device(deferred_decode, *, sampling_params : TTSamplingParams, grammar_bitmask : torch.Tensor)
      ```
      The deferred payload owns position, reset, token-history, slot-remap,
      trace, and reload state. It is consumed exactly once. The grammar mask
      has shape `(batch_size, ceil(vocab_size / 32))`, with one bit per token
      and `1` meaning allowed. Rows follow the immutable submitted TT row
      order, including all-allowed rows for plain requests and padding gaps.
      Multi-model execution concatenates complete model-local slot blocks.
      `slot_remap[i] = j` means destination slot `i` inherits sampler state
      from source slot `j`; each accepted decode applies it exactly once before
      advancing per-slot sampling state.
    - `warmup_model_prefill`: compiles or captures the model's prefill variants. The plugin invokes it once without tracing and again with tracing when configured.
      ```python
      warmup_model_prefill(kv_cache : list, enable_trace : bool, can_sample_on_device : bool)
      ```
    - `warmup_model_decode`: compiles or captures decode and sampling variants. `can_sample_device_grammar=True` adds grammar-on variants, excluding logprob configurations that the plugin keeps on host. Traced grammar requires the plugin's normal two-phase order: compile with `enable_trace=False`, then capture with `enable_trace=True`.
      ```python
      warmup_model_decode(kv_cache : list, enable_trace : bool, max_batch_size : int, num_blocks : int, can_sample_on_device : bool, can_sample_device_grammar : bool = False, read_from_device : bool = True, greedy_only : bool = False, skip_trace_precompile : bool = False, sampling_trace_variants_prepared : bool = False)
      ```
      `grammar_bitmask` on `decode_forward` is used only by model warmup to
      capture the grammar-on sampling trace. Runtime grammar arrives later via
      `sample_decode_on_device`.
      With `trace_mode=all`, the plugin also calls these hooks around prefill
      trace capture so both host- and device-sampled decode buffers exist before
      any trace is live:
      ```python
      prepare_device_grammar_decode_trace_warmup(kv_cache : list, max_batch_size : int, num_blocks : int) -> bool
      capture_prepared_device_grammar_decode_trace() -> None
      ```
      Returning `False` from preparation keeps structured outputs on host.
      Returning `True` requires the capture hook to exist.
    - `enable_device_grammar`: allocates the persistent grammar-mask state after
      the plugin has selected device grammar for this runtime and before model
      warmup captures traces. It returns whether activation succeeded and sets
      `device_grammar_enabled` accordingly.
      ```python
      enable_device_grammar() -> bool
      ```
    - `model_capabilities`: Class dictionary that lets VLLM know which optional
      TT backend features the model supports. We use it in
      [platform.py](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/platform.py)
      to validate requested features and enable or disable them in VLLM. Missing
      keys default to `False`. The recognized sampling keys include
      `supports_prefix_caching`, `supports_async_decode`,
      `supports_sample_on_device`, and `supports_device_grammar`.
      `supports_prefix_caching` controls automatic
      prefix caching. `supports_async_decode` allows async scheduling for models
      that can submit decode with `read_from_device=False` and later read the
      output asynchronously. `supports_sample_on_device` allows the
      `sample_on_device_mode` TT config option. `supports_device_grammar`
      additionally certifies late packed-mask application before decode token
      selection. It requires `supports_sample_on_device`; the initial contract
      excludes structured prefill, row-sharded sampling, async structured
      overlap (launch with `--no-async-scheduling`), logprobs, and block-output
      models. The loaded generator exposes `device_grammar_enabled` after
      `enable_device_grammar()` runs; a false value makes the plugin retain host
      sampling.
      Example:
      `model_capabilities={"supports_prefix_caching": True, "supports_async_decode": True, "supports_sample_on_device": True, "supports_device_grammar": True}`
3. **(Multi-modal models only)** Currently, we only support image+text input modalities. An example generation class is `Gemma3ForConditionalGeneration` in [models/tt_transformers/tt/generator_vllm.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator_vllm.py). For more info on multi-modal models see also [vLLM Docs - Multi-Modal Support](https://docs.vllm.ai/en/latest/contributing/model/multimodal.html)). These models have the same interface requirements as the text-only models, as well as the following:
   - `prefill_forward` (**image+text models**): same as text-only models with an additional kwarg (`pixel_values`) for the image inputs.

## Testing the Model in vLLM
Once the model meets all of the requirements specified in [Implementation Requirements for Model Integration](#implementation-requirements-for-model-integration), it can be tested in vLLM by following the instructions in the [vLLM TT plugin README](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/README.md) and doing the following:
1. The model needs to be registered using `ModelRegistry.register_model` in the TT plugin's registration path, primarily [src/vllm_tt_plugin/model_registry.py](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/model_registry.py) and [src/vllm_tt_plugin/platform.py](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/platform.py).
2. Test offline inference, continuous batching, and performance using the plugin README.
3. Test increasing sequence lengths with [examples/offline_inference_tt.py](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/examples/offline_inference_tt.py).
4. Test asynchronous requests with the server instructions in the plugin README. For Galaxy text models, prefer its documented single-process lane path.

## vLLM TT Plugin Modifications
TT-specific behavior belongs in
[tenstorrent/vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin),
not in upstream vLLM or the deprecated `tenstorrent/vllm` fork. The main plugin
files are:
- [`platform.py`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/platform.py): handles platform definition, config validation, and runtime-class selection.
- [`model_registry.py`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/model_registry.py): handles TT model registration.
- [`loader.py`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/loader.py): handles model initialization.
- [`worker.py`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/worker.py): handles device initialization and KV cache management.
- [`model_runner.py`](https://github.com/tenstorrent/vllm-tt-plugin/blob/main/src/vllm_tt_plugin/model_runner.py): handles input preparation and model execution.
