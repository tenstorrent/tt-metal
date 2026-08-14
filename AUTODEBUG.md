# AutoDebug: Mistral tokenizer-regex and active-trace allocation warnings

## Headline findings

1. **The preserved tokenizer warning does not describe the tokenizer used for API request tokenization.** The live TT-specific vLLM tokenizer construction passes `fix_mistral_regex=True`. A decisive CPU-only A/B using the exact checkpoint revision shows the registered API tokenizer is token-for-token identical to the corrected HF tokenizer and differs materially from `fix_mistral_regex=False`. The warning is emitted by a separate default processor/template probe during API chat-template warmup.

2. **The allocator warning is a real trace-lifetime invariant violation during startup warmup.** Decode and sampling traces remain retained after capture, which marks subsequent allocation unsafe. `warmup_model_decode()` then calls `reset_kv_cache()`, whose `ttnn.fill(..., output_tensor=cache)` is first compiled only after the traces are active. A program-cache miss allocates device-side program resources and triggers the warning. The smallest safe fix is to compile the cache-reset fill during the pre-capture eager phase, then repeat the reset after trace capture from the program cache.

No implementation code was edited and no hardware workflow was run during this investigation.

## Direct observations

- `server_release_v9.log:33` records the engine configuration with `tokenizer_mode=tt_mistral_small_24b_hf` and the exact snapshot `9527884be6e5616bdd54de542f9ae13384489724`.
- The allocator warning occurs once at `server_release_v9.log:92`, immediately after KV-cache sizing and before engine initialization completes.
- The regex warning occurs later in the APIServer process at `server_release_v9.log:102`, between “Warming up chat template processing” and “Chat template warmup completed.”
- The release then starts the API server and completes its preserved traffic. That absence of visible corruption is useful run evidence, but is not proof that allocating while traces are live is generally safe.

## Tokenizer construction and decisive A/B

### Live construction path

- `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py:660-667` changes `model_config.tokenizer_mode` to `tt_mistral_small_24b_hf` when `TT_MISTRAL_TEXT_VER=mistral_small_24b_autoport`.
- `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/entrypoints.py:19-33` registers both `MistralSmall24BTokenizer` and `MistralSmall24BRenderer` for that mode.
- `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/mistral_small_24b_tokenizer.py` constructs the HF tokenizer with `fix_mistral_regex=True`; its renderer owns that tokenizer for request rendering/tokenization.
- vLLM creates the renderer from the updated model config (`vllm/vllm/v1/engine/async_llm.py:135-147`) and uses its tokenizer for input and output processing.

The metadata/config flag is therefore not merely copied documentation: it is on the live registered renderer/tokenizer path recorded by the server.

### A/B experiment

Using Transformers 5.10.2 and the exact cached snapshot, the investigation constructed:

- `AutoTokenizer.from_pretrained(snapshot, fix_mistral_regex=False)`
- `AutoTokenizer.from_pretrained(snapshot, fix_mistral_regex=True)`
- `renderer_from_config()` with `tokenizer_mode=tt_mistral_small_24b_hf`, then its registered API tokenizer

The shortest discriminator, `hello world`, produced:

```text
fix=False: [16114, 1392, 3011]
fix=True:  [29706, 4304]
live API:  [29706, 4304]
```

A full chat-template control also had `live API == fix=True` and `live API != fix=False`. The preserved artifact `models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/tti_release/tokenizer_regex_ab.json` strengthens this with six exact eval-prompt controls (three IFEval and three GPQA): every live-plugin token-ID sequence and SHA prefix exactly matches `fix=True`; every `fix=False` sequence differs, beginning at token index 3.

### Why the warning still appears

The warning timestamp and process identify an APIServer chat-template warmup probe, not the EngineCore renderer construction. The probe asks Transformers for a processor/template using the checkpoint defaults. That secondary construction omits `fix_mistral_regex=True`, so Transformers truthfully warns about that temporary tokenizer object. The registered TT renderer continues to tokenize served prompts with the corrected tokenizer.

Conclusion: the warning is noisy and should be removed by forwarding the corrected tokenizer kwargs into the processor/template probe (or avoiding the redundant tokenizer construction), but there is no evidence of an unfixed live request tokenizer.

## Active-trace allocation

### Source invariant

`tt_metal/impl/allocator/allocator.cpp:100-112` states the contract: allocations made while a trace is retained may be corrupted when the trace executes unless those allocations die before replay. `MeshDeviceImpl::end_mesh_trace()` marks allocations unsafe after capture (`tt_metal/distributed/mesh_device.cpp:1318-1347`); allocations become safe again only after all retained traces are released (`mesh_device.cpp:1261-1277`).

### Startup causal chain

1. `warmup_model_decode(enable_trace=False)` returns immediately at `generator_vllm.py:314-317`; it therefore does not compile the KV-cache reset operation during the ordinary pre-capture warmup phase.
2. The traced warmup calls `prepare_decode_trace()` at `generator_vllm.py:327-333`.
3. `generator.py:670-712` eagerly compiles decode and sampling before capture. It does not call `reset_kv_cache()`.
4. `generator.py:719-741` captures and ends the model and sampling traces. After `end_trace_capture`, both trace buffers remain retained in `_trace_model_id` and `_trace_sampling_id` (`generator.py:765-769`), so allocator state is unsafe.
5. Control returns to `warmup_model_decode()`, which calls `self.generator.model.reset_kv_cache(kv_cache)` at `generator_vllm.py:334-337`.
6. `model.py:468-474` implements that reset as repeated in-place `ttnn.fill`. Because fill was not compiled earlier for these cache tensors, the first program-cache miss allocates device-side program resources while traces are live, matching the warning timing at `server_release_v9.log:92`.

This is more specific than the prior broad classification of “some startup allocation”: the first post-capture cache-reset fill/program compilation is the allocation site supported by control flow and timing.

### Risk and smallest fix

The warning represents a real unsupported allocation order, even though the preserved run showed no corruption symptom. The reset output tensors are the long-lived vLLM KV-cache buffers and are passed as `output_tensor`, so the suspicious new allocations are most plausibly program/cache resources created on the fill cache miss rather than replacement KV tensors. Static inspection cannot prove their exact addresses or demonstrate corruption/non-corruption without allocator instrumentation on hardware.

Smallest safe source change:

- During the pre-capture eager phase, call `reset_kv_cache(kv_cache)` once so all fill programs/resources are compiled while allocations are safe.
- Preserve the existing post-capture reset to clear warmup-written KV values. It should then be a program-cache hit and allocate no new trace-unsafe resources.
- Keep program-cache misses forbidden during capture as today; optionally assert the program-cache miss count is unchanged around the post-capture reset in a regression test.

Releasing the traces around reset is not equivalent because it destroys the traces needed for serving. Removing the post-capture reset is also unsafe because warmup has modified caller-owned cache contents.

## Verification required after a fix

A targeted hardware startup smoke should verify all of the following, without rerunning full release evals:

1. No allocator active-trace warning during `initialize_cache` / `warmup_model_decode`.
2. Program-cache miss count is unchanged by the post-capture reset.
3. Both decode and sampling traces remain retained and replay successfully.
4. KV-cache contents are zero after warmup and the first real request remains correct.

## Remaining uncertainty

- Static evidence identifies the first-time `ttnn.fill` program-cache allocation as the warning source with high confidence, but only instrumentation or a hardware startup smoke can name the exact underlying Metal buffer allocation and prove the fix eliminates it.
- The exact upstream vLLM processor helper responsible for the redundant tokenizer warning should be patched at the narrowest boundary after confirming its kwargs contract. This does not block the conclusion that live request token IDs are corrected.
