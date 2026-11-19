# Runtime and trace boundary audit

The unchanged decoder remains the stage5 selected TP4 path. `tt/model.py`,
`tt/multichip_decoder.py`, `tt/flat_prefill.py` and `tt/bounded_prefill.py` are
unchanged from the accepted full-model checkpoint. No precision override,
replicated residual, single-chip fallback or HF execution was added.

`decode_forward(..., reload_inputs=False, read_from_device=False)` now returns
borrowed persistent device storage. With device sampling this is exactly
`generator.tokens`, the common sampler's `tt_out_tok` and the next model trace's
input. Without sampling it is `generator.logits`, in physical sampler-ready
layout. Callers must consume it before a later replay overwrites it. Inactive
rows remain scheduler-owned; the device return is not a new host sentinel API.
The ordinary host-return defaults retain their prior shape and behavior.

The measured resident loop calls this public API, submits nonblocking model and
sample traces, then synchronizes once at the window boundary. It performs zero
per-token token/position/RoPE/page-table refreshes, token/logit reads, host
sampling, allocations or synchronizations. Diagnostic endpoint reads occur
after separate measured windows and do not reconstruct feedback. The streaming
`generate` API still reads caller-visible output tokens for its list/EOS and
teacher-forcing contracts; its performance is reported separately.

The additional first-chunk prefill trace binds the existing B1/P128 token,
position, page, working-state and stable-logit buffers. All-layer warmup occurs
before any capture. Its graph resets the fresh GDN slot, imports working state,
executes unchanged embedding/RoPE/layers/norm/head, copies sampler-ready logits,
and publishes GDN state. Arbitrary/nonaligned/resumed/batched inputs keep the
prepared general path. Changing input data or page contents never retires a
trace. There are no allocation acknowledgments or corruptible scopes in the
new helper. Every persistent input predates all traces; temporary values die
before replay; CQ0 orders producer/consumer operations. Existing persistent CCL
and cloned common-sampler outputs retain their ownership contracts.

Sampling parameter caching compares the complete padded temperature/k/p
signature; seeds are deliberately excluded and continue to reset per request.
A slot permutation invalidates the cached signature because parameters moved on
device. Greedy/stochastic/greedy transitions update the same persistent buffers
and retain the same generic split-sampling trace. Unsupported device penalties,
logprobs and grammar retain the explicit compatibility boundary from stage6.

`tests/optimized_runtime_audit.py` patches six transfer/synchronization APIs while
every graph is captured and while public resident decode executes. A positive
control proves the sentinels are active. Strict allocation runs include program
cache allocations and end each eager prefill transition with decode. Tests also
compare eager/traced complete generated token sequences, resident endpoint
history, positions, changed prompt/page data and stable trace/program identities.
The separate tracker positive control must raise before these checks count as
acceptance. These sentinels cover executed public Python transfer APIs; they are
not a claim of static proof over every library internals path.

All20 acceptance receipts and source-lineage checks pass in `hard_gates.json`.
The one-line standalone-loader import repair is verified by
`verify_import_resolution.py`/`import_resolution_proof.json`: prior package-mode
probes resolve the same helper class, and all readiness jobs rerun on exact
repaired source. Raw source hashes are preserved, not relabeled. `run_acceptance.py` is the serialized command manifest and every job's
`.command.json` records source hashes, environment, timing and exit status.
Generic setup warnings about active-trace allocations and L1 semaphore placement
are assessed by strict tracking and measured capacity, rather than suppressed.
No reset, trace eviction, policy fallback or device recovery has been needed.

Independent stage review: clean-pass, round1 (`stage_review_round1.md`), with no runtime audit blocker.
