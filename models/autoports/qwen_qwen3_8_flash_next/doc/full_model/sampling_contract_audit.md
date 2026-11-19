# Full-model serving and sampling audit

Current acceptance evidence is summarized in the final section. The initial
audit below is retained as the design/rejection history, including blockers
that were subsequently repaired and validated.

Initial CPU-only audit, 2026-11-13. This is a design/rejection record, not a device result. At audit time the target directory had the accepted multichip decoder but no `tt/model.py` or `tt/generator.py`. `serving_contract.json` therefore deliberately records the unimplemented required surfaces as `missing`. The stage owner must replace those statuses with actual implementation/test evidence; this audit cannot pass the stage.

## Pinned source pair

- Plugin: `/home/ttuser/qwen38-flash/vllm-tt-plugin`, repository `tenstorrent/vllm-tt-plugin`, detached commit `042a94863ad27d88207bcf71abdf67122c6f4c27`.
- Recommended upstream vLLM: `v0.31.0`, built from source with `VLLM_TARGET_DEVICE=empty`; both pinned README and `docs/install-vllm-tt.sh` agree.
- Installation script reads upstream `requirements/common.txt`, applies `docs/vllm-overrides.txt` (`numpy>=1.24.4,<2`, OpenCV `4.11.0.86`, Transformers `>=5.10.4,<5.15.0`), installs CPU torchvision `0.26.0`, numba `0.61.2`, then vLLM without dependencies and the plugin editable. These are read-only future serving requirements. Nothing was installed and no server was launched.
- Second sampler source: `/home/ttuser/qwen38-flash/tt-transformers`, repository `tenstorrent/tt-transformers`, commit `25ebcdb3311bb86d1cb4ceff77bb8bf481f4cc73`. Cloned source for comparison only; not installed and not imported into the target runtime.

Commands used to materialize source:

```bash
git clone https://github.com/tenstorrent/vllm-tt-plugin.git /home/ttuser/qwen38-flash/vllm-tt-plugin
git -C /home/ttuser/qwen38-flash/vllm-tt-plugin checkout --detach 042a94863ad27d88207bcf71abdf67122c6f4c27
git clone --depth 1 https://github.com/tenstorrent/tt-transformers.git /home/ttuser/qwen38-flash/tt-transformers
```

## Selected integration candidate

Select repository-local `models.common.sampling.generator.SamplingGenerator`. It owns `TTSampling`, `TTPenalties`, `SeedManager`, per-mode trace slots, runtime parameter normalization, and request-state transitions. This avoids another package dependency and separate implementations of trace ownership and penalties. Selection is based on source contracts; final greedy strategy and terminal performance still require device measurement.

Reject external `tt_transformers.modules.sampling.sampling_1d.Sampling1D` as the default integration candidate, not as an incorrect sampler. It supports the target 1D mesh, padded vocab masking, candidate all-gathers, `tt_out_tok`, per-call device k/p/temp/seeds and sampled-token logprobs. Its `Sampling1DConfig` is declarative, and `load_device_buffers()` must resolve lazy buffers before capture. It owns no trace cache or complete request penalty/RNG lifecycle; the full-model generator would have to supply those. The richer repository-local wrapper fits the pinned plugin's request-state contract with less new code. Neither implementation justifies custom sampler math.

| Contract | Repository common SamplingGenerator | External Sampling1D |
|---|---|---|
| Vocab-sharded 1x4 logits | Local top-k plus candidate all-gather | Same |
| Feedback output | `sample(..., tt_out_tok=tokens)` | `decode_forward(..., tt_out_tok=tokens)` |
| Trace lifetime | Internal `_TraceKey(penalties_on, log_probs_on, force_argmax, grammar_on, bucket)` | Generator-owned trace wrapper needed |
| Warmup | `precompile(..., all_configs=True)` and `capture_trace(..., skip_precompile=True)` | Resolve buffers and warm exact forward variants before capture |
| Penalties | `TTPenalties`, prompt/output counts and remap/reset helpers | Caller must apply and track penalties |
| Request seeds | `SeedManager`; explicit seeds currently select eager sampling | Persistent/override seed tensor; caller owns stream policy |
| Logprobs | Distributed calculator, sampled and optional top-k payload | Sampled-token calculator; argmax gives none |
| Grammar | Persistent mask and grammar-keyed traces | No equivalent full grammar wrapper |
| Parameter changes | Persistent parameter tensor uploads at transitions | Per-call tensor inputs |

### Exact runtime contract to implement

1. LM head output stays vocabulary-sharded across the four ranks, BF16 tiled `[1,1,32,V_padded/4]` for the sampler. Derive padding from real HF vocabulary; mask invalid IDs. Preserve the decoder's per-rank hyperconnection residual streams through the stack; any terminal projection/gather is localized after the last layer.
2. Construct common sampler args with real `vocab_size`, tile/mesh-aligned `padded_vocab_size`, `cluster_shape=(1,4)`, `max_top_k=32`, `max_batch_size` matching the prepared batch capacity (the sampler physically rounds it to at least 32), `sampling_dp=1`, and the accepted CCL resources. Check the actual CCL helper supports the selected topology and stable outputs; `line_all_gather` may accept `buffer_key`.
3. Default candidate is `args.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"]=False` (or omit that config: default is False). The same config needs `num_links` and `topology` if present. Greedy is semantically `k=1,p=0,temp=1`, while local top-k remains physical width 32. Gathering width 32 from each rank yields a tile-aligned candidate row. Do not set physical `max_top_k=1`. The common normalizer maps public temperature zero onto greedy k/temp; use `format_sampling_params` for the API representation.
4. Benchmark a separate, semantically identical force-argmax configuration against split greedy. Force-argmax gathers full logits then argmax; comments calling it faster are not target-device evidence. Do not accept a dominant sampling cost without repairing the LM-head/sampler boundary.
5. Allocate persistent token, position, page-table, cache/state, sampler params, RNG/penalty and CCL buffers before capture. Warm every prefill/chunk, model-decode and sampling program signature first. `precompile(all_configs=True, tt_out_tok=persistent_tokens)` handles common sampler axes. Use `compile_token_update=True` if penalty updates will be captured; restore warmup-mutated request/KV/RNG state.
6. Capture model trace to sampler-ready logits; capture sampler with those exact logits and `tt_out_tok=persistent_tokens`, `skip_precompile=True`. `sample(enable_trace=True)` validates exact input identities. Do not defer an unprepared first sampler capture behind live model traces.
7. Model trace advances current position/RoPE state exactly once; sampler writes next token into the existing token input; unchanged page tables require zero per-step copies. Reads only observe results. State refresh belongs to request/scheduler transitions.
8. Common mode traces remain cached across greedy/sampled transitions. `reset_trace()` releases all slots and is teardown-only. Inspect actual sampler trace condition when counting traced steps, not merely the requested enable flag.
9. Explicit seeds are a real integration caveat: `SamplingGenerator.sample` uses its internal trace only when `not seed_manager.has_active_request_seed()` (current `generator.py:675`). Explicit seed updates are eager; labeling them traced is wrong. Preserve an explicit host compatibility path for tests/features, or repair traced device-owned seed handling with evidence. The unseeded generator also initializes/reseeds according to `SeedManager`'s init/SKIP state machine; warmup must not leak RNG state.
10. Top-k above 32 or unbounded stochastic sampling cannot silently clamp the public distribution. Return full logits through explicit host compatibility for those requests; the pinned plugin already selects that mode with `max_device_top_k=32`.

The local common wrapper has one source-level acceptance blocker: `SamplingGenerator.capture_trace` currently encloses the complete begin/forward/end capture in `trace_allocation_tracker.corruptible_allocation_scope` (`models/common/sampling/generator.py:610`). Current tracing skill explicitly forbids exempting a complete capture. Repair that wrapper to acknowledge only exact retained tensors with reviewed backing-storage/alias lifetimes, or use a generator-owned compliant wrapper around the common sampling operations. Do not accept the wrapper unchanged or disable the tracker. Its bucket helper also acknowledges `(logits, output)`; each backing allocation must have a valid lifetime against every live model/sampler trace.

Both candidates use lowest-index tie correction over gathered candidates. Their documented corner case is more than 32 equal maxima within one vocabulary shard: unstable local top-k may drop the absolute lowest ID. This remains a semantic/determinism test case for the chosen greedy strategy; no device tie result is claimed here.

## Serving API consequences

Read `docs/MODEL_INTERFACE.md`, setup, execution, lifecycle, capabilities, reload, speculative, grammar and scheduling contracts. Source inspection includes `TTModelRunner.check_perform_device_sampling`, submit paths, and `tests/tt` request-isolation, seed, penalty, logprob, host-only, structured-output and chunked-prefill cases. The JSON enumerates every documented capability and required call/request surface; optional speculative/stage/grammar extensions use documented defaults, not new plugin knobs.

The pinned plugin sends `prefill_forward` complete token history `[B,T]`; per-row absolute start and exclusive end select each chunk. `empty_slots` place recurrent state, independently of page-table row order. Decode uses `start_pos=-1` for inactive rows and may send `slot_remap` on host-sampling calls too. Every recurrent/conv/PLE and dormant sampler slot must remap exactly once, including off-batch state. A row omitted from a prefill is still live.

For future `decode_input_update_contract=1`, honor four independent commands: reload all inputs, reload only page table, reload sampling params, and reset sampling state. With `reload_inputs=False`, host token/position values are intentionally stale. Derive no position, RoPE, token, seed counter or other forward state from them. `reset_sampling_state` implies full authoritative inputs. The plugin disables resident reuse unless `supports_async_decode` is declared and proven, even for synchronous scheduling.

On the selected **four-device** mesh, this plugin pin routes **all logprob requests to host** (`model_runner.py:4062` allows device logprobs only for 8 or 32 devices). The full-model compatibility path must return real full-vocabulary logits. Min-p, bad words, logit bias, allowed IDs, minimum tokens and custom logits processors also route to host; grammar uses host unless the optional device grammar declaration and deferred methods are provided. These are explicit caller-selected compatibility boundaries, not fallback allowed inside measured token-out decode.

Gated DeltaNet, convolution and PLE/hash state give this model the documented `recurrent_state` prefix-cache exclusion: KV block reuse alone cannot reconstruct nonzero-prefix recurrent state. The serving JSON records `supports_prefix_caching=False` under that exact exclusion, pending any actual recurrent checkpoint mechanism. This does **not** waive the user's resumed-prefill gate: retained-state continuation and both missing-history controls still require implementation and passing evidence. Chunked continuation of a live slot is distinct from cross-request KV-only APC.

No plugin or vLLM implementation was modified. No tests were marked passing from inspection, and no hardware was touched.


## Hardware-backed greedy strategy decision, 2026-11-14

Keep repository-common `SamplingGenerator` split greedy as the current final
strategy: semantic `top_k=1, top_p=0, temperature=1`, physical local top-k width
32, and `allow_force_argmax=False`. Reject the common force-argmax strategy for
this target on measured latency; it passed correctness. No custom sampler math
is needed. This updates the initial source-only strategy decision above; it
is not an all-layer performance or full-stage acceptance claim.

Evidence: [sampler_ccl_smoke_v1.json](sampler_ccl_smoke_v1.json),
[sampler_ccl_smoke_v1.log](sampler_ccl_smoke_v1.log),
[terminal_timing_v1.json](terminal_timing_v1.json), and
[terminal_timing_v1.log](terminal_timing_v1.log). Both reports passed. The smoke
loads no model; timing uses only real layers 0, 1, 3, batch 1, capacity 128,
17 prompt tokens, two warmup steps and 12 measured steps. These numbers must
not appear as full-model TTFT or throughput.

| Warmed host-window metric | Common split greedy | Common force argmax |
|---|---:|---:|
| Model trace, logits only | 9.677250 ms | 9.677844 ms |
| Sampling trace, frozen logits | 0.540895 ms | 2.811883 ms |
| Read existing token output | 0.022906 ms | 0.032273 ms |
| Reduced token-out decode | 10.306138 ms | 12.573694 ms |

Both samplers consume the same persistent frozen real logits in correctness
checks. At prompt lengths 3, 7, and 17, both returned the full-vocabulary
reference argmax IDs 56680, 93282, and 142093 respectively. Full-logit readback
and host argmax are correctness-only checks outside timing and token feedback.
After equivalent request reset, both produced the identical 14-token device
feedback sequence (two warmup plus 12 measured steps):
`29463, 87107, 226836, 1179, 31839, 88093, 66003, 236065, 123128, 79448,
66645, 3558, 37628, 209214`.

For each candidate the token-out counter delta was 14 model replays, 14
sampling replays and 14 token readbacks, with **zero token, position or
page-table refreshes**, zero cache resets, zero captures and zero releases.
The read token is observed, not uploaded as feedback. Model trace 0 and
sampling traces 2/3 remained resident; program-cache entries stayed 10905.
All sampler objects/resources were allocated and warmed before capture.
Clean timing excludes watcher, allocation diagnostics and device profiling;
JSON records profiler environment. Host windows include trace submission and
boundary synchronization. Isolated regions are not additive kernel timings.
The isolated sampling/(model+sampling) share is 5.29% for split greedy and
22.51% for force argmax: neither dominates this reduced stack. Full-model
clean timing and separate per-op attribution remain independent evidence.

### CCL adapter provenance and greedy corner cases

The first profile attempt failed during common force-argmax precompile with
`AttributeError: Collectives has no attribute
get_and_cycle_barrier_semaphore_handle`; see
[terminal_ops_v1.json](terminal_ops_v1.json). This was an adapter/resource
contract defect, not a sampling result. `tt/multichip_decoder.py:Collectives`
now lazily owns a standard `models.common.tt_ccl.TT_CCL` provider and delegates
barrier semaphore cycling, all-gather semaphore cycling and link count.
`prepare_sampling_collectives()` creates those resources once before generator
prepare/capture; existing decoder gather/sum paths stay intact. The provider
is owned for trace lifetime. No post-capture device allocations or per-token
resource construction are required. The compared common 1x4 force-argmax path
uses its Linear collective; this rejected candidate does not change the
selected decoder or split-greedy collective policy.

The no-model smoke preallocates six full-vocabulary BF16 input patterns and
both outputs, warms both samplers, then captures and replays them. Ten input
selections include repeated transitions; each is replayed twice, covering all
32 physical sampler rows and all four replicated output ranks. Unique maxima
on each rank agree exactly between samplers; for row `r` on winning rank `s`
the actual selected ID is `62080*s + 13 + 97*r`. Every selected token's input
value equals that row's global maximum, and repeat replay is deterministic.

The distributed 65-way ties select IDs `7*r` (0 through 217), equal between
samplers. The distinct local-shard corner case places all 65 maxima in shard
`r % 4`; both select `[13, 62093, 124173, 186253]` repeated eight times. Thus
the tested >32-per-local-shard corner is semantically greedy and deterministic,
including its lowest-ID choice. This evidence covers those inputs, not a
universal proof that unstable top-k always preserves the lowest index of
arbitrary ties. Greedy semantic correctness is separately checked by selected
value equality to the global maximum.

Reproduction (clean environment; watcher absent, profiler and tracker off):

```bash
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
 timeout 180 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
 --sampler-smoke-only --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/sampler_ccl_smoke_v1.json
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
 timeout 3600 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.profile_full_model_terminal \
 --iterations 12 --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/terminal_timing_v1.json
```

## Final implementation and current profiler evidence

The selected common sampler's math is unchanged. `QwenGenerator._sample` owns
the strict model-specific trace wrapper around common `_run_sampling`, keeping
the exact logits and token-output buffers persistent. It does not invoke the
common capture-wide corruptible-allocation exemption. `DeviceSeedState` advances
explicit request seeds on device; the common seed hash and sampler remain the
semantic reference. `seeded_sampling_device_v2.json` passes 42 checks spanning
reset, row order, migration, inactive slots and replay. Thus the initial common
wrapper's eager explicit-seed limitation is repaired for this generator, rather
than hidden by claiming an eager path was traced.

`terminal_final_ops_v1.json` and `terminal_final_perf_v1/coverage.json` are the
current-source final reduced operation profile: real layers 0/1/3, batch 1,
capacity 262144, two warmups and one measured eager iteration. All 21488
host/device/report operations and six signposts agree, with no skipped rows.
The measured split and force-argmax wrappers select the same actual greedy
token 243975 from frozen real logits. Merged kernel sums are 0.429263 ms split
and 2.886563 ms force-argmax; split is 3.673% of reduced model-plus-split kernel
time. Force-argmax remains rejected at 6.724 times the split kernel sum.
`terminal_final_analysis.md/.json` records units, CCL merge semantics, source
attribution and the profiler metadata limitations. These eager kernel sums are
not trace latency or full-model throughput, and do not replace the earlier
tie-pattern or traced feedback checks.

The final all-48 clean public generation result is
`acceptance_perf_final_v1.json`: prompt 128 / output 128, TTFT 7038.672 ms and
127 steady decodes at 138.754269 ms/token (7.206985 tokens/s/user). Every steady
step replays model and sampling traces and reads the requested token; there
are no host token/position/page-table refreshes, resets, captures or releases.
All 12 QSA layers end at position 255 on all four ranks. Batch-32 current
resident-feedback evidence is in `batch32_hf_factored_v2.json`; its QSA
position readback is rank 0, while the all-rank position evidence above is B1.

The explicit host-sampling compatibility mode is separately selected by the
caller for unsupported device distributions/features and full-vocabulary
logits. It is not entered by measured token-out generation. The final
`runtime_fallback_audit.md` and serving contract record its boundaries and
cache/reset ownership. No custom sampling distribution or vLLM adapter was
introduced.
