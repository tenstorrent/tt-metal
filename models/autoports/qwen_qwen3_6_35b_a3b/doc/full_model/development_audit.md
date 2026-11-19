# Development audit: full-model wrapper and generator

CPU source inspection only, 2026-11-14. This is a development audit, not an independent stage-review verdict. Inspected the initial 205-line `tt/model.py` and 311-line `tt/generator.py`, their decoder/sampler dependencies, readiness contract and callback runner, sampling output validation, plus-one implementation, and trace-correctness rules. No runtime imports, device access or tests occurred. Line references describe this source snapshot and may move during development.

The accepted decoder residual is **replicated BF16**; the wrapper correctly retains it. This audit does not request a different decoder residual or dtype policy.

## Highest-priority findings

### P1: full-context generation computes an avoidable 30.3 GiB logits tensor per chip

`model.py:165-171` always calls `terminal(hidden)` for every prefill token; `generator.py:149-151` slices the last token only after the full LM head. The checkpoint config has vocabulary 248320 and context 262144. On TP4 the BF16 output alone is `262144 * (248320 / 4) * 2 = 32,547,799,040` bytes per chip, before weights/cache/norm temporaries. This prevents ordinary max-context generation for an implementation reason, not an inherent model-capacity limit, and spends full-vocab work on unused tokens even for short generation.

Required fix: select the last logical hidden row before terminal norm/LM head for token-out prefill; for `return_all_logits=True`, apply the terminal path/readback in bounded chunks and return the required host result. Preserve absolute cache updates and output order. Test a non-aligned prompt through both last-only and all-logits paths. Do not reduce advertised context based on this avoidable allocation.

### P1: post-capture prefill logits remain live at replay

`generate` captures both traces through `prepare`, then runs eager prefill (`generator.py:267-270`). The returned device logits (and its reshape alias) stay referenced by local `logits` through `_sample` and the entire decode loop (`274-290`). They were allocated after both captures, are not acknowledged, and are not needed once copied to persistent `sampler_logits`. Allocation tracking must reject this live buffer at the first sampling replay. Without tracking, another trace can overwrite it; retaining unused post-capture storage obscures the lifetime invariant.

Required fix: complete the copy, release every temporary/alias before replay, or use preallocated stable terminal output buffers. Do not exempt a whole prefill/capture scope. Add a focused fresh-process tracker run that performs prefill after capture then sampling/model replay, with program-cache allocations included and the positive control active.

### P1: preparation only covers the first logical prompt and one immutable slot tuple

`prepare` returns unconditionally once `prepared` is true (`166-168`). `_plan` later allocates and retains device RoPE/position tensors per exact `(length,start)` (`120-124`; decoder `prepare_positions` calls `from_torch(...,device=...)`). New logical lengths/starts also compile embedding, slices, decoder chunks, terminal and cache-fill programs after live traces. Thus two `generate` calls of different lengths, a resumed prefill, or mixed prefill/decode can create persistent unsafe allocations. There is no coordinated preparation of supported physical buckets or sampling modes. `prepare(slots=new)` after first capture silently does nothing, and `decode_forward` then rejects the new signature (`233-234`).

Required fix: establish bounded physical prefill/decode/sampling signatures and prepare their buffers/programs before first capture. Reuse runtime position indices and preallocated inputs where possible. Deferred first capture may only consume already-prepared state. Test interleaved new logical lengths/starts and slot orderings on one generator with unchanged existing trace IDs, zero recaptures and tracker replay after the final eager path. Do not fix this with request-time trace retirement.

### P1: noncontiguous/reordered slots feed sampled tokens to the wrong requests

`model.decode` reads token input at physical `slot` (`model.py:188`) but concatenates hidden rows in `slots` order (`193-196`). `_decode_into_sampler` packs those rows into rows `0..B-1` (`generator.py:161-164`). Sampling writes its packed output directly into all 32 entries of `self.tokens`; `_read_tokens` then indexes physical slot IDs (`214-217`). Example: `slots=(2,)` samples the active request into token entry 0 and writes a dummy-row token into entry 2. The next decode reads entry 2, not the previous prediction. `(1,0)` swaps feedback. Sampling dummy rows also overwrites token state of unscheduled slots.

Required fix: choose a consistent fixed-slot sampler layout or trace a device scatter back into persistent token input, preserving unscheduled rows. Prove `(2,)`, `(1,0)` and a gapped active set against solo requests over two or more token-feedback steps. Do not repair this with a host readback/writeback loop.

### P1: serving state keywords are silently ignored, and inactive rows execute

`prefill_forward` accepts arbitrary kwargs but ignores `empty_slots` and defaults to `range(B)` (`137-140`), so a request admitted into slot 2 overwrites slot 0's recurrent/conv state. `decode_forward` ignores `slot_remap`, all four reload commands, `sampling_params`, `read_from_device`, and `enable_trace` (`219-240`), while defaults force host token/position reload. `start_pos=-1` rows are not filtered: `_refresh` writes -1 and `device_positions` casts it to uint32 for RoPE lookup (`model.py:175`), then executes cache and recurrent updates. Absence of a slot from the captured tuple is not support for the plugin's inactive-row contract.

Required fix: implement explicit keyword semantics and active masks/fixed slots now; reject unsupported keywords loudly until implemented. Snapshot-remap every recurrent/conv/sampler slot once before forward, including host-sampling transitions. Honor page-only reload without host token/position refresh. Cover incoming slot placement, permutation, inactive gaps and unchanged page-table replay with counters.

### P1: sampled requests and explicit host-logit compatibility are not implemented

The common sampler remains at constructor defaults k=1/p=0/temp=1. `prepare` warms only that mode and captures `_run_sampling(...penalties_on=False,grammar_on=False)` (`175`, `190-192`). No sampling parameters or seeds are ever applied. `generate(**kwargs)` and low-level calls discard sampling requests. `host_sampling=True` reads logits then performs its own host argmax (`201-210`); `decode_forward` still returns token IDs (`241`). A caller needing top-k/top-p, penalties, structured constraints or logprobs therefore cannot obtain the required logits boundary or requested sampling behavior.

Required fix: implement parameter/state commands and prepare/cache actual common sampler graph modes. Keep greedy and sampled traces semantically correct and reusable. Expose a separate explicit low-level logits compatibility output; caller-owned host sampling must not be replaced by unconditional generator argmax. Maintain the measured token-out path on device. Validate mixed per-row params, seeds, mode alternation and host/device transitions.

### P1: externally owned caches cannot be decoded

Prefill accepts an external cache, but `decode_forward` rejects `cache is not self.cache` (`233`). `prepare` always uses `self.cache` and has no external cache argument. There is therefore no supported path to prepare/decode a caller-owned cache, despite the explicit low-level signature. `_bind_pages` also assumes every attention layer has the same external table and exact default per-request capacity; `page_tables_per_layer` is ignored.

Required fix: bind the configured serving cache pool and its stable page-table buffers during generator preparation, with correct per-layer table routing and dtype policy. Reuse those bindings through request changes. Test prefill and multiple decodes using externally allocated shared/disjoint pages without cache replacement or single-chip/host fallback.

## Additional contract and measurement issues

- **P2 — reset is incomplete:** `reset` only zeroes K/V/conv/recurrent tensors (`81-86`). It leaves uploaded page mappings, `_host_pages`, tokens, positions and sampler state unchanged. A public reset after externally remapping pages does not restore freshly constructed semantics. Restore metadata/state in place, and reset only appropriate request slots during serving. Full `reset` must preserve trace buffers and IDs.
- **P2 — duplicated full-capacity cache distorts physical capacity:** `self.zeros` allocates a second full tensor for every K/V/recurrent/conv tensor (`generator.py:74-79`), and each of up to 32 slots owns max-context K/V by default (`model.py:151-162`). Include these bytes in the census now, but prefer warmed in-place fills or bounded shared reset sources and a physical page pool before treating allocation failure as a hard model limit. The current memory census stops before generator caches/zero sources/sampler/trace buffers, so it cannot establish the final full-model KV budget.
- **P2 — missing decode bounds and vector validation:** decode validates neither active positions against configured/cache context nor consistency/uniqueness of slots, token rows, prompt-length/start arrays and slot capacity. Prefill uses `zip` and can silently omit rows when arrays differ (`144`). Reject malformed inputs before writes and handle `-1` only as the documented inactive sentinel.
- **P2 — warmup restoration and timing:** warmup samples into persistent tokens and may modify sampler RNG; reset restores only cache tensors. `generate` performs first-time full warmup/capture inside the readiness timer. Report warmed TTFT separately and prove request/RNG state restored. After `close`, `prepared` remains True although traces are cleared (`295-299`); define closed-object behavior explicitly instead of later KeyError.
- **P2 — async outputs lack per-submission lifetime:** `read_tokens=False` returns shared mutable `self.tokens`, and there is no `read_decode_output`/`process_decode_output_host` contract. The next replay overwrites it. This may serve a synchronous caller that reads immediately, but cannot be advertised as deferred async output without retaining the submitted read buffer/event until consumption.

## Checks that did not identify a bug

- Teacher forcing callback order matches the installed readiness contract: callbacks for steps 0 through N-2 supply the next input; the final callback at `generator.py:288-289` records step N-1 once without feeding another decode. N=1 records exactly one callback. `run_teacher_forcing._run_one_entry` wraps `(step,predicted)` correctly. No off-by-one callback repair is indicated by source.
- The persistent sampler token tensor shape `[1,1,1,32]`, row-major uint32, matches `SamplingDeviceOperation` output requirements for padded 32-row input logits. The bug is slot routing, not this output shape itself. Candidate width must remain tiled; max_top_k=32 across four shards meets that structural rule.
- `ttnn.plus_one` returns/updates the input allocation in place and accepts row-major int32. Ignoring its Python return is therefore valid; whether the captured input identity and positions actually advance correctly still needs runtime evidence.
- Both model and sampling traces use CQ0 and persistent logits/tokens allocated before capture. Serial producer-before-consumer ordering is a sensible baseline once post-capture temporaries and late compilation are fixed. No blanket allocation exemption appears in this generator-owned capture wrapper.
- Decoder construction currently preserves its accepted defaults, including replicated residual and BF16 paged KV policy. The independently chosen LM-head BFP8/HiFi2 policy still needs the stage accuracy/performance evidence; source inspection alone neither accepts nor rejects it.

## Suggested first discriminating checks

1. Run the existing reduced real-layer wrapper on a short prompt with allocation tracking through first prefill-to-sampling and two decode replays; resolve the first unsafe allocation rather than restarting the whole model.
2. On the same prepared generator run two different prompt lengths and a nonzero-start continuation, ending each with decode replay. Record program-cache growth and trace IDs.
3. Verify token and position tensors for a nonzero slot, swapped two-slot order and inactive gap before/after two replays.
4. Verify an external shared cache and changed/unchanged page tables, then mixed sampling/host-logit modes.
5. Only after these pass, run full-model readiness and max-context acceptance using last-row terminal prefill.

These are proposed future owner checks, not experiments run by this audit.
