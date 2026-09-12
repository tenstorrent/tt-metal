# Full model

Measured full-model batch-1 **TTFT 97.24 ms; token-out decode 39.06 tokens/s/user**
at S128/G128, warmed canonical split traces on four Blackhole p300c devices.
AIME teacher forcing: **99.56 ms TTFT; 38.76 tokens/s/user**, S203/G100.
Source: `readiness_confirmed.json`, clean process and launcher exit 0.
Status: validated; independent xhigh `stage-review: clean-pass`
([report](STAGE_REVIEW.md)). Local checkpoint SHAs are recorded in `work_log.md`.

## Model and precision contract

`tt/model.py` implements the pinned HF text autoregressive path: BF16 embedding,
48 linear-attention and 16 full-attention layers, final RMSNorm (HF gamma + 1),
and untied vocabulary-column TP4 LM head. The checkpoint is
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Embedding columns are divided across four devices and gathered once at entry.
The model never replicates the full weights or runs a single-chip fallback.

Every block is the unchanged selected `MultichipDecoder` with shared `TT_CCL`:
BFP4/LoFi projections, BF16 activation/residual/norm/CCL, FP32 projection
accumulation and recurrent state, BFP8 paged K/V. Prefill fills use BFP8 K/V;
decode updates use BF16 K/V. Page size32, one full KV head and12 linear value
heads per rank. Ring fabric and two CCL links are preserved. The B1 residual
stays L1 width-sharded on40 cores between layers; B2..32 uses the inherited
public DRAM layout. There are no wrapper conversions or gathers between blocks.
The [Stage5 rejection ledger](../optimized_multichip_decoder/work_log.md),
[inter-layer contract](../optimized_multichip_decoder/inter_layer_contract.md),
and [collective contract](../optimized_multichip_decoder/collective_contracts.md)
remain authoritative.

The new terminal path uses BF16 norm/activations, BF8 head weights, HiFi2 and
FP32 accumulation. A40-core sharded final norm feeds an8-core input reshard,
eight DRAM-bank head chunks (8192 local vocabulary columns, last chunk4736),
and a local logits concat. Decode input block10 fits L1; block20 exceeded it.
The full-vocabulary prefill-logits path keeps its interleaved head copy.
These terminal choices do not change the decoder policy.

## Readiness and text quality

Fresh `readiness_aime24_chat.refpt` contains100 sequential greedy HF predictions
and top100 token sets from AIME24 index0, using the exact tokenizer chat template.
Sibling metadata records model/revision, tokenizer, prompt-source hash, command,
generation settings and strict checkpoint assignment. The prompt has203 tokens.

| Standard runner | Top1 | Top5 | Top100 | Tokens |
| --- | ---: | ---: | ---: | ---: |
| run_prefill_check | 99% | 100% | 100% | 100 |
| run_teacher_forcing | 98% | 100% | 100% | 100 |

`run_autoregressive` writes both completions and token IDs under `autoregressive/`.
It reuses the fresh HF control only after exact token-ID and generation-length
assertions. The shared runner now preserves prompt whitespace; stripping the
template's final newline incorrectly changed203 tokens to202. No prompt is
truncated in a quality check. The S128 benchmark fixture is latency-only.

`hf_qualitative_256.json` and `tt_qualitative.json` contain all six shared prompts,
rendered prompts, token IDs and 256-token greedy outputs. HF controls use the same
pinned model, left-padded batch6 with explicit masks and positions. Each TT request
uses the full64-layer generator, fixed device feedback, and a reset between requests.
The matched cap-1024 controls and TT results are `hf_qualitative_extended.json`
and `tt_qualitative_extended.json` for prompts 0/2/3/5. TT completes the haiku,
story, thermodynamics and Fibonacci answers at 418/531/433/337 tokens. HF completes
prompts 0/3/5 at 284/411/237; its coherent story remains capped at 1024. Prompts
1/4 already completed in the six-prompt 256-token run. The extended run exits 0.
See `qualitative_review.md` for direct reading, exact prefix comparisons and the
bounded-generation caveat. The longer TT haiku has different early wording; it
is a matched-budget quality check, not a claim of exact old-prefix continuation.

## Public generator API and state ownership

`build_generator(model_dir, mesh_device, **kwargs)` returns the standard Metal
readiness `Generator`. `generate(..., enable_trace=True)` is the standalone token
API; `prefill_logits` returns logical CPU logits for readiness checks. Cache setup,
padding, chunking, positions, page fill and logical output slicing belong to the
generator. Valid prompt lengths need no tile/page/chunk alignment.

| API | State contract |
| --- | --- |
| `model.allocate_cache(batch_size=B, capacity=C)` | All layers, B1..32, C1..262144, ceil(C/32) pages per slot |
| `bind_cache(cache, page_table)` | Caller retains ownership; table covers capacity, INT32 row-major interleaved DRAM, stable device tensor identity |
| `prefill_forward(tokens, page_table=..., kv_cache=..., prompt_lens=..., slots=..., start_pos=...)` | Mixed logical lengths, distinct fixed slots, continuation offsets; internal stack chunks4096 |
| `decode_forward(tokens=None, start_pos=None, page_table=..., kv_cache=..., active_slots=..., read_from_device=False)` | Explicit initial token/positions; omitted values reuse persistent device feedback and incremented positions; inactive full positions=-1 and linear state preserved |
| `set_sampling_params(top_k=..., top_p=..., temperature=..., seed=...)` | k1..32, p0..1, positive temperature; request updates stable common parameter tensors; inverse temperature passed to native sampler |
| `reset()` | Explicitly zero the currently bound cache/state, reset tokens/positions/seeds; ordinary same-shape requests reuse compiled traces |

`slots` contains integer prefill slot indices; `active_slots` is a B-element
boolean decode mask. Binding/replacing a cache invalidates the position budget. Changing the active-slot
set requires authoritative positions. Inactive output token lanes are ignored by
the scheduler; active state advances coherently. Passing an unchanged host table
does not copy it; passing the bound device table gives its owner responsibility for
in-place changes and invalidates the CPU comparison shadow. A different device
table requires rebinding. Invalid geometry/IDs/dtypes are rejected before binding.

New prefill signatures can compile persistent program buffers, so both traces are
released *before* an unseen signature executes. Known prefill signatures reuse
traces. This repairs the allocation-tracker failure without allowing unsafe buffers
or restricting public prompt shapes. See `AUTOFIX_trace_prefill.md`.

High-level `generate` returns the requested fixed token count; it does not stop
on EOS. Readiness permits this contract. The qualitative runner retains the first
EOS completion, and a low-level scheduler can deactivate finished slots explicitly.
This distinction explains fixed-length replay counters for shorter saved answers.

Explicit `host_sampling=True` supports greedy host-sampling compatibility. It
returns CPU logits in low-level decode and owns host argmax/feedback in `generate`.
Non-greedy compatibility requests are rejected; top-k/top-p use the device path.
Neither TP4 common sampler supplies logprobs; this generator does not advertise them.

## Canonical split sampling and fallback audit

Both common sampler implementations were inspected before token-out integration;
`AUTODEBUG_sampling.md` records their state, shape, tracing, seed, penalty, logprob,
mesh and output contracts. `Sampling1D` has a broken method-arity/local-buffer path
in this checkout and lacks the selected request handling. The implementation uses
common `TTSampling` directly; it adds no custom sampler.

The model trace produces local `[1,1,32,62080]` BF16 logits. Logical B1..32 is
padded to32 physical sampling lanes. Local top32 followed by TP4 candidate gathering
feeds the shared greedy k1/p0/T1 sampling op, or the same top-k/top-p path. Persistent
UINT32 token tensor `[1,1,1,32]` is passed as `tt_out_tok` and consumed by the next
embedding. Model trace increments signed positions and RoPE indices; sampling trace
advances seeds. No per-token host token/position/RoPE/table construction is used.
The common candidate gather calls native `ttnn.all_gather`; current legacy
`num_links`/`topology` arguments are deprecated and ignored. The native operation
resolves routing from the configured fabric. Final profiler attributes record
`axis_num_links={0;2}`, `axis_topology={Linear;Ring}`, and `FABRIC_1D_RING` for
both 32-to-128 candidate gathers. Each rank contributes one tile; this routing
metadata does not claim that both links carry useful payload for that tiny input.
The unchanged decoder and embedding entry use explicit experimental ring/two-link
collectives. The source diagnosis's earlier adapter recommendation is superseded
by its final integration disposition and these runtime rows.

Low-level `read_from_device=False` returns the same persistent token tensor without
host readback. High-level generation reads output token IDs for the caller, never
uses those IDs to refresh model input unless teacher forcing or explicit host
compatibility was requested. Full logits cross to the host only for readiness,
explicit compatibility, or test assertions. There are no Torch computations inside
the model or sampling graph; the runtime contract probe guards both and steady replay.
Setup-time HF loading, tokenizer work, RoPE-table preparation, cache binding and
prefill uploads remain outside decode traces. There is no automatic reset/retry,
precision fallback, host model, replicated model or alternate topology fallback.

## Performance evidence

Reduced probes load one real layer of each kind (0 and3), real terminal weights,
and the same cache/trace/sampling contracts. Their space-token output is only a
pipeline smoke, never a qualitative result.

| Reduced S128/G128 candidate | Token-out tokens/s | Decision |
| --- | ---: | --- |
| Interleaved head + common split greedy | 400.30 | Baseline |
| Interleaved head + common force argmax | 226.64 | Reject; slower full-logit gather |
| DRAM head, block10 | 415.69 | Legal after adapting block20 L1 overflow |
| DRAM head + sharded norm | 433.09 / 434.09 | Selected; logits PCC .9999766, greedy tokens equal baseline |

`tracy/profile_split_drained` contains advice-enabled per-device baseline reports.
The sampler window is about0.49ms; it does not dominate full-stack token-out decode.
The baseline head is about0.996ms and final norm0.095ms; these motivated the selected
DRAM/sharded terminal. Sampling stays canonical and semantically greedy throughout.
Final selected profiling, advice dispositions and stack accounting are recorded in
`performance.md`. Raw captures are archived with exact paths/hashes in
`profile_archive.json`; compact compressed CSVs and rendered reports stay local.

## Context, checks and status

`memory_capacity_plan.json` recomputes all64-layer weights, both head layouts,
all-layer KV, recurrence, RoPE, tables, traces, CCL/sampling and bounded scratch.
`../context_contract.json` preserves262144 with no capability reduction. The actual
full model passed262143+last-position decode and262144 prefill. Batch and context
are a joint physical allocation: B32 short prompts are tested separately from B1
maximum context; B32 times maximum context is not claimed to fit.

`work_log.md` and `commands.log` record exact commands, source hashes, artifacts,
failures and remediation. Readiness, full-stack B32/state, watcher, context,
qualitative and degeneracy checks pass with clean process exits. Independent
xhigh `stage-review` returns `clean-pass`; checkpoint SHAs are in the work log. Watcher exposed an oversized single-packet read in the
DRAM-sharded matmul reader. The two-line repair selects the existing packet-splitting
helper using the actual row size; it preserves the terminal layout and program.
The prescribed Docker build wrapper was unavailable. The existing local CMake
`ttnn` build and runtime install passed, as did all three focused TP4 watcher
regressions and the original reduced generator watcher run. Exact commands and
logs are in `work_log.md` and `AUTOFIX_dram_head.md`. No vLLM work or push is part
of this stage.


## Full-stack state verification

`contract_full_b32.json` passes with all 64 layers, allocation tracking and 32
fixed slots. Mixed prompts of 31 and 33 tokens occupy slots 31 and 0. The probe
checks exact logits after physical page permutation, unchanged-table copy elision,
caller-owned device tables, persistent token-buffer identity, coherent positions,
frozen inactive recurrence/convolution state, reset determinism and explicit host
sampling parity. Unaligned continuation (31 + 2 versus 33) has PCC 0.99921447.
A guarded model/sampling replay rejects host computation inside either graph;
steady replay has zero token, position, RoPE or page-table refreshes. The process
and launcher both exit 0 (`contract_full_b32.exit_status`). `watcher_fixed.json`
repeats the contract on real representative layers with all Ethernet checks enabled.


## Runtime warning disposition

The generic allocator warning in untracked runs concerns allocations made while
traces exist. Eager prefill temporaries are freed before replay; newly compiled
persistent buffers require the explicit trace lifecycle repair above. The tracked
shape matrix and full-stack B32 contract verify that replay has no unsafe live
allocations. The warning is not suppressed or used to waive the tracker.

The fabric packet-size advisory is inherited from Stage5's selected CCL policy;
that stage's collective/rejection ledger remains unchanged. Actual CCL dtypes,
ring geometry and kernel timings are visible in the final reduced reports.
The HF CPU control reports unavailable optional fused linear-attention libraries
and uses its Torch implementation. This is the explicitly requested host reference,
not a TT runtime fallback. The optimized TT generator never invokes that model.
Native pytest shutdown emits nanobind reference warnings also seen in collection;
all three test bodies pass, and the standalone watcher/full-model processes close
cleanly without that harness warning. The actual NoC watcher assertion was fixed
and rerun, as recorded in `AUTOFIX_dram_head.md`.
