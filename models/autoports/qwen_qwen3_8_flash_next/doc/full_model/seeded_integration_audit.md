# Seeded generator integration audit

CPU source review; no hardware run by this audit. Reviewed the current
`tt/generator.py`, `tt/device_seed.py`, common sampler, and plugin commit
`042a94863ad27d88207bcf71abdf67122c6f4c27` execution/reload source.

## Initial findings and follow-up status

The root agent subsequently fixed findings1/2 in source: input reload now
saves existing tokens and restores inactive rows after uploading; prefill has
an explicit sampled-ID path using supplied parameters and destination slots.
This audit then fixed `set_sampling_params` only, closing finding4's prefill
scatter and raw parameter-validation defects. These are source fixes awaiting
the corresponding device probes; the original failure descriptions follow
so the tests' purpose remains explicit. Findings3/5 require call-site policy
and validation review, beyond the setter's ownership.

1. `decode_forward(reload_inputs=True)` constructs zeros for all32 token rows
   and uploads the entire persistent `decode_tokens`. This destroys dormant
   token feedback outside active_slots before the sampling wrapper snapshots
   it. `_restore_inactive_tokens` then faithfully restores those zeroed values.
   Required fix: upload new tokens into a persistent staging allocation, then
   device-concatenate staged rows for active_slots and existing decode_tokens
   rows for every other slot; copy the result into decode_tokens. Compile this
   copy/slice/concat shape before capture. Do not read resident tokens to host
   to fill the gaps. Test: sample slots0/1, reload only slot0, then continue
   slot1 with reload_inputs=False and compare its state/token to a solo run.

2. `prefill_forward` currently accepts but ignores `sampling_params` through
   kwargs. The pinned plugin sends prefill params only for device sampling
   and expects sampled IDs; first-token seeds must be registered in
   empty_slots order before that draw. Existing standalone `generate` sets
   params first and is different from this low-level entrypoint. Add explicit
   prefill sampling composition using the already available return_device
   and `_sample(slots)` operations, or document and implement a named generator
   helper that the later adapter directly calls. Intermediate prefill chunks
   must not consume a draw. Merely ignoring the argument is not support.

3. `reload_sampling_params=True, reset_sampling_state=False` changes the
   host `request_seeds` list but leaves device seed keys/counters untouched.
   The pinned `plan_decode_reload` currently sends both flags together for
   transitions, so this does not break its ordinary path. The public independent
   command contract still needs defined seed-update behavior; a changed seed
   must not be silently ignored. Keep live counters when updating a seed key
   without a state-reset command, or reject an unsupported combination before
   any mutation until that operation exists.

4. `set_sampling_params` formats request lists as physical rows0..31. A direct
   submission with slots=(2,0) and two request-ordered params requires explicit
   scatter to physical slots; otherwise the request at slot2 gets padding
   defaults. Pinned ordinary decode uses front-packed rows; lane decode
   supplies physical stable-slot rows, so the additional slots API must state
   its parameter ordering rather than implicitly mixing conventions.

5. Nonidentity remap mutates cache, seed, token and page-shadow state before
   parameter validation and page-table-shape checks. If a later check raises,
   the plugin does not commit its ownership map and may resubmit the same map,
   causing a second device permutation. Validate all submission shapes,
   parameter ranges, cache bindings and commands before the first mutation.

## Correct pieces and proof still needed

The integrated device seed update precedes the common sampler's own
manual_seed/draw. Explicit seed counters advance on device only for active
rows. Prefill seeds use the exclusive prompt end, and transition decode seeds
use authoritative start_pos+1, matching the next-token absolute-position
convention. Slot migration moves the helper's complete seed/counter/mode state;
negative/full64 seeds retain common modulo64 semantics. The old SeedManager
does not also advance the borrowed seed tensor. These are source findings;
reproducibility still requires actual token tests.

`tests/probe_seeded_sampling.py` constructs a sampler-only QwenGenerator object
and invokes its real `_sample` implementation unchanged. It loads no model
weights. It uses the real248320 vocabulary,62080 local TP4 width,32 physical
rows, top-k32/top-p0.95/temperature1.3, and32 candidate tokens spread across
all4 vocabulary shards. Every row has identical logits to isolate RNG state.

Checks include first capture versus cached replay, identical negative seeds
across physical rows, a distinct seed with high bits,8-token reset sequences,
fresh row-order permutation, midstream three-slot migration, a dormant-row
pause/resume, exact counter increments, preserved inactive feedback and equal
outputs on all4 ranks. The cold-capture counter check detects an extra logical
draw during capture. It uses mandatory trace and program-cache allocation
tracking and never invokes the common sampler's broad corruptible capture
scope. A bounded600-second command is in the script docstring.

The script was Black/syntax checked only. No device JSON was fabricated.
It does not test decoder outputs, full input reloads, top-k distribution
statistics, host-seed bit parity or throughput. Those require their own
appropriate integration/performance evidence.

The probe now also invokes real `prefill_forward` with a synthetic immutable
terminal-logit provider in place of model.forward_chunk. It tests first-token
seeded IDs, duck-typed plugin parameters, sparse/reordered empty_slots,
dormant token/counter preservation, and no draw when prefill sampling is not
requested. The synthetic provider is explicitly reported in each result;
this remains generator/sampler evidence, not decoder accuracy evidence.

`tests/test_sampling_params_cpu.py` AST-extracts the current setter and common
formatter without importing TTNN. Six tests pass, covering required defaults,
duck typing, per-slot merging, num_logprobs=None, scalar broadcasting, invalid
raw values rejected before common clamping/inversion, seed/list validation,
and explicit host compatibility. Evidence: `sampling_params_cpu.log`.
