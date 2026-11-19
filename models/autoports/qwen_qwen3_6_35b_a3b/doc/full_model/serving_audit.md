# Full-model serving contract audit

This document records the pinned interface audit and subsequent CPU-side reconciliation of parent-owned device evidence. `serving_contract.json` currently has 86 rows: 51 supported generator contracts, five pending capacity/resume rows, and 30 explicit optional exclusions. Full40 prefill/teacher top-k gates pass; this is not a full-model stage or serving-stack acceptance claim. This audit installed no serving stack, implemented no adapter, and opened no device.

## Pinned sources

- Plugin: `tenstorrent/vllm-tt-plugin`, commit `042a94863ad27d88207bcf71abdf67122c6f4c27`, clean detached checkout `/home/ttuser/qwen36-35b-a3b/vllm-tt-plugin`.
- Recommended upstream vLLM: `0.31.0` (`v0.31.0`) in both plugin `README.md` and `docs/install-vllm-tt.sh`. Keep the pair for this run.
- The installation script resolves vLLM common requirements at that tag, builds with `VLLM_TARGET_DEVICE=empty`, installs vLLM `--no-deps --no-binary vllm`, and then the plugin editable. It adds CPU torchvision `0.26.0 --no-deps` and numba `0.61.2`. `docs/vllm-overrides.txt` pins opencv-python-headless `4.11.0.86`, NumPy `>=1.24.4,<2`, and transformers `>=5.10.4,<5.15.0`. These are future installation inputs; this audit did not change the environment.
- Registration precedence is `TT_MODEL_CLASS_OVERRIDES`, then `EXTRA_MODELS_DIR`, then built-ins. The later adapter must resolve under the checkpoint architecture in every process. No plugin patch is needed merely to register the new adapter.

## Generator-facing signatures

The signatures below describe the pinned consumer contract. The JSON maps them to implemented neutral generator primitives and their scoped evidence. In particular, explicit reloads use `decode_forward_v1`; legacy `decode_forward` remains the readiness method. A later vLLM adapter supplies interface spelling/factory glue; no adapter work belongs in this stage.

```python
# Future factory mapping to build_generator / full model.
initialize_vllm_model(hf_config, mesh_device, max_batch_size,
                      max_seq_len=..., tt_data_parallel=1, optimizations=None)
get_max_tokens_all_users(model_name=..., num_devices=..., tt_data_parallel=...,
                         max_model_len=..., max_num_seqs=...)
get_kv_cache_spec(vllm_config)
allocate_kv_cache_per_layer(per_layer_specs)

prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, start_pos,
                enable_trace, empty_slots=None, sampling_params=None,
                page_tables_per_layer=None)
decode_forward(tokens, start_pos, *, page_table, kv_cache, enable_trace,
               read_from_device, sampling_params=None, page_tables_per_layer=None,
               slot_remap=None, prompt_tokens=None, output_tokens=None,
               reload_inputs, reload_page_table, reload_sampling_params,
               reset_sampling_state)
read_decode_output(output, *, async_read=True)
process_decode_output_host(output, *, is_tokens)
warmup_model_prefill(*, kv_cache, can_sample_on_device, enable_trace)
warmup_model_decode(*, kv_cache, max_batch_size, num_blocks,
                    can_sample_on_device, enable_trace)
release_request(slot)
note_state_slots_moved(moves)
release_persistent_capture()
```

`per_layer_specs` contains `(shape, dtype, tensor_idx)` triples despite an older source docstring calling them pairs. Shape is `(num_blocks, heads_per_device, block_size, head_size)`; repeated `tensor_idx` means shared storage. Preserve the selected K/V dtype split and retain separately owned recurrent/conv state. The plugin cache allocation result is opaque and returned to every warmup/forward. A missing token-budget hook silently selects 131072 tokens; implement an evidence-backed budget instead.

Prefill receives complete CPU token history `[B,T]`. For row `i`, process only `tokens[i,start_pos[i]:prompt_lens[i]]`; the end is absolute, not a chunk length. Starts may be NumPy or torch and need not be tile aligned. `empty_slots` selects recurrent/conv device state, whereas page-table row `i` selects that execution row's K/V. Return the logical last token as host logits `[B,1,V]` or sampled IDs in submitted order. Intermediate chunks are host-sampled/suppressed by the plugin and must not advance device RNG.

Decode tokens are CPU `[B,1]`; positions `[B]` have `-1` for inactive rows, including gaps. Inactive rows must preserve other requests and must not consume invalid positions. `slot_remap[i]=j` means new slot `i` takes old slot `j`; apply a snapshot permutation once before forward, including off-bucket recurrent/conv state. Host-sampling transitions still require remapping the dormant sampler. `note_state_slots_moved` changes identities, not the device state again.

Use explicit reload contract version 1 in the future adapter. The four commands are independent: full input reload; page-table-only reload; sampling parameter reload; sampling-state reset. The plugin never sets full and page-only reload together. A sampling reset implies authoritative full inputs. When full reload is false, host tokens and positions may be one step stale and must not seed any new forward/RNG state. Forward advances resident position once; sampling writes to the same persistent token input; readback is observational. `supports_async_decode=False` causes the plugin to force full reload every decode, so a real resident-serving implementation and evidence are needed before claiming that capability.

Readback must retain buffers/events until consumed. Token output is `[B,1]`; host logits contain real vocabulary only and own writable storage. The plugin reads logits for unsupported device requests. Optional selective host readback may remain disabled by its documented default; that does not remove ordinary full-vocabulary compatibility mode.

The generator must own two-phase preparation even outside vLLM. Prepare prefill/chunk/decode/sampling variants and persistent tensors before capture; restore KV and RNG state. Keep traces until model unload. `release_persistent_capture` runs while the mesh remains open; the plugin does not call a generic model `shutdown`.

## Request inventory and required evidence

The ordinary `tests/tt` files cover per-row temperature/top-k/top-p/seed, negative seeds, seeded reproducibility across batches/slots, unseeded variety, k=1 greedy, mixed request isolation, repetition/presence/frequency penalties, sampled/top-K/all-vocabulary logprobs, min-p, bad words, logit bias, allowed IDs, min tokens, structured choice/regex/JSON, and chunked-prefill recall at aligned and ragged starts. The JSON maps each feature to a row and source. Tests use k=50 as well as device-eligible k<=32; the host-logit path is mandatory compatibility behavior, not grounds for reducing request capability.

On the existing 1x4 mesh, the pinned plugin routes **every logprob request to host sampling** (`check_perform_device_sampling` requires 8 or 32 devices). Positive top-K device logprobs are additionally restricted by `hf_config.model_type == "gpt_oss"`. Do not invent a capability key to bypass this. Full logits in explicit compatibility mode provide the required request behavior. `max_device_top_k` must reflect the actual candidate bound, normally 32. Missing device penalty support must be explicitly False because its documented default is True.

Optional speculative, block-output, pipeline stage, grammar-on-device, selective host readback, and folded lane-DP extensions have explicit disabled declarations in `tt/serving_capabilities.py`. Their presence in plugin tests does not make them new full-model requirements. The speculative suite's optional admission, lossless verification, sampled controls, capacity, cancellation, termination and async tests were inventoried; no speculative support is claimed.

Qwen carries FP32 Gated DeltaNet recurrent matrices and convolution tails. `supports_prefix_caching=False` is the documented `recurrent_state` exclusion: KV blocks alone cannot recover this state at a cached prefix hit. This does **not** waive the user's explicit resumed-prefill check. The public continuation API must retain/restore matching prefix state (or explicitly reconstruct it), test the required starts and negative controls, and leave `readiness/prefill_resume_check.json`. A generic APC hit with only vLLM KV blocks remains unsupported unless a complete recurrent-state snapshot contract is implemented and tested.

Before any row becomes supported, attach an existing path beneath this model directory containing implementation tests or result artifacts. No source-inspection claim alone proves device correctness. The five remaining missing rows are measured aggregate capacity and retained-state continuation/chunked-prefill acceptance; they are stage work, not deferred vLLM work. Full40 capacity32 trace acceptance now passes; final context/resume/performance/review acceptance remains separate from transport support.

## Sources inspected

`serving_contract.json.documents_read` is the full file inventory. Read the interface and linked setup/execution/lifecycle/capability/reload documents; inspected speculative admission/extension requirements, scheduling chunk and async sections, and grammar contract. Source inspection covers loader factory, worker budget/specification consumers, model-runner allocation/prefill/sampling/warmup, async decode/reload/readback, model-input sampling dataclass, config/admission declarations, and logprob payload conversion. Every `tests/tt/test*.py` and `tests/tt/spec/test*.py` was inspected through its AST test names, docstrings, assertions and request fields; ordinary request callsites and speculative README admission were read. Optional speculative bodies unrelated to the disabled extension were not treated as model requirements.

Initial source inspection used `git clone`, `git checkout --detach <pin>`, `rg`, `sed`, and Python AST extraction. Follow-up work writes this inventory and runs CPU fake-owner tests. Device artifacts are read from serialized parent runs. Both external checkouts remain source-only; this audit performed no installs, server execution, device commands, or plugin modifications.

## Required trace reservation by the mesh owner

Before opening the mesh, its owner must call
`QwenServingCapabilities.recommended_trace_region_size(max_batch_size)` and pass
that byte count as `ttnn.open_mesh_device(..., trace_region_size=...)`. The input
is the configured slot capacity, even when only slot zero is active. The
generator cannot enlarge an existing mesh reservation after capture starts.

| Configured capacity | Reservation per device | Graph policy |
|---|---:|---|
| 1 | 200,000,000 bytes (decimal 200 MB) | Optimized solo model plus sampler trace |
| 2–32 | 536,870,912 bytes (512 MiB) | Fixed-capacity model plus sampler trace |

The full40 capacity32 model trace required **467,361,792 bytes**. The initial
200,000,000-byte reservation failed during trace creation, recorded in
`logs/full_batch32_final.log:3761`. This is a known insufficient trace reservation,
not evidence of a smaller model context or a reason to use eager/single-chip
decode. The 512 MiB retry passes in `full_batch32_trace_budget.json`: all40 layers
compute on all four chips, with8551 program entries and zero growth. Inactive
slots1..31 state payloads were checked on chip0 only. Two resident model/sampler
steps performed zero token/position/page refreshes or captures, one explicit
activity update, and two observational token readbacks. This is not an all-chip
inactive-payload or maximum-context claim. The batch-one reservation already
supports the full40 accuracy runs. Include
the appropriate reservation in each prepared DRAM census without adding live
trace buffer bytes a second time. The separate maximum-context acceptance is
batch one and therefore still uses 200,000,000 bytes.

No generic `fabric_config` dictionary change enables this reservation. Preserve
FABRIC_1D and the selected 1x4 mesh, and supply the reservation at mesh creation.
The configured aggregate KV budget is shared across requests: 32 slots does not
advertise 32 independent 262144-token KV pools.

Authoritative decode reloads validate submitted logical page ownership using
the supplied positions. Resident/page-only commands deliberately avoid device
position readback; the caller owns their valid extents and off-batch page
ownership. This is part of the explicit low-level contract, not an implicit
per-token host state rebuild.

The pinned plugin does **not** discover `recommended_trace_region_size`. Its
`src/vllm_tt_plugin/worker.py:1042` defaults to only **50,000,000 bytes** and reads
the explicit `tt.trace_region_size` launch configuration at the next two lines.
The later serving handoff must therefore set that existing option to the
reservation above; merely exposing the neutral helper is insufficient. This
audit records the required mapping without adding a plugin adapter or launching
vLLM. The additional `tt.trace_region_size` inventory row records this previously
implicit launch prerequisite.
