# Seed continuity investigation

2026-11-13. AutoFix hypothesis investigation; source inspection and isolated
host Python checks only. No model execution, TT access, server changes, plugin
changes, or source edits. This report is the only artifact written by this
investigation. Plugin inspected at `042a94863ad27d88207bcf71abdf67122c6f4c27`.

## Finding

The secondary finding in `AUTODEBUG.md` is confirmed. A seeded request with
default penalties loses its logical RNG offset on the first decode after
device-sampled prefill, and again on layout/membership transitions. The first
decode must use the request's initial device seed plus one, since prefill
already sampled its first output; current code instead reconstructs the initial
seed. This is independent of sampling-dataclass normalization in the adapter.

The smallest complete repair for ordinary requests admitted through this
generator is generator-owned per-request sampling-position metadata, combined
with the already-authoritative decode positions at explicit reloads. Keep the
four reload commands intact. Do not solve this by discarding reset commands,
forcing host token feedback, or inspecting `start_pos` on resident steps.

## Source evidence

Paths below are relative to the model directory unless marked `P` (plugin).

- `P/src/vllm_tt_plugin/async_decode.py:409-411` invalidates the resident decode
  chain after prefill. `plan_decode_reload` at lines 425-484 requests input,
  parameter, and sampling-state reloads for the first decode and layout/mode
  transitions. That is intentional, authoritative behavior.
- `P/src/vllm_tt_plugin/model_runner.py:2679-2694` supplies prompt/output
  histories for penalty use. Without penalties they are absent. Bad-words
  output history alone is not forwarded as sampler history:
  `async_decode.py:1391-1394` forwards histories only with prompt history.
- `tt/generator.py:_decode_preflight` normalizes absent histories before the
  reset. `tt/sampling_state.py:normalize_histories` makes missing history empty.
  Thus absent and explicitly empty output history become indistinguishable.
- `tt/generator.py:_reset_sampling_state` derives each seed as
  `hash(request_seed) % 500000 + 1 + len(output_history)`.
- `prefill_forward` initializes this seed with zero outputs and samples once;
  `_sample_logits` advances it once. The plugin's next decode reset reconstructs
  zero outputs again when penalties are disabled.
- `SamplingSlotState.apply_slot_remap` already moves device seeds correctly.
  A subsequent full reset overwrites those remapped seeds with initial seeds.
- Common `SamplingGenerator.reset_sampling_params` updates sampling controls
  and penalty controls, not this generator's device seed stream. The lost
  continuity specifically comes from `_reset_sampling_state`, not merely
  `reload_sampling_params`.

The existing `tests/full_model_sampling_transition_probe.py` supplies explicit
output histories for its resume/first-decode checks. Its prior success would
not cover the plugin's omitted-history case.

## Recommended minimum implementation

Separate the RNG offset from penalty token reconstruction **inside the model**.
An internal optional offset argument to `_reset_sampling_state` is sufficient;
it need not become a plugin kwarg or capability. When explicit output history
exists, keep its length authoritative, including an explicitly empty history.
When it is absent and no penalties require tokens, use request metadata and
validated reload positions to compute the logical offset.

For an ordinary fresh request whose completed prompt has length `L`, the input
token at absolute decode position `p` is generated output number
`p - L + 1`. Consequently, immediately before sampling that decode's output:

```text
next_device_seed = initial_hashed_seed + (p - L + 1)
```

Example: a 129-token prompt samples output 1 during prefill. First decode reads
that token at position 129 and needs offset 1. At position 133 it needs offset
5. This formula also accounts for host-sampled intervals before returning to
device sampling; it does not attempt to make the host and TT samplers produce
identical random draws.

The required bookkeeping is small but must follow request ownership:

1. Record a prefill sampling origin per `(cache binding, slot)` for **both** host
   and device prefill. Update the admitted prefix end on successive prompt
   chunks, so a final host-sampled prefill followed by device decode also has
   an origin. Use the validated `input_rows`/`active_slots` mapping; do not infer
   it from packed row numbers. Record after successful admission. For ordinary
   prompt chunks, the last prefill end is `L`; intermediate chunks do not
   consume an output draw.
2. Preserve whether `output_tokens` was omitted before normalization. During
   `_decode_preflight`, compute and validate offsets for active reset slots
   from the **validated** positions, the effective post-remap metadata, and
   explicit histories if present. All reset commands already require
   `reload_inputs=True`, so this need never read a stale resident-step input.
   Reject negative/out-of-range inferred offsets before device writes.
3. Apply the requested remap once, including this new metadata, then execute
   reloads and seed reconstruction in the existing order. Forward the computed
   offsets to `_reset_sampling_state`; do not change `SamplingSlotState`'s
   device seed uploader or sampling kernels.
4. Handle inactive/dormant slots deliberately. A position of `-1` is not a
   valid origin calculation and must not become a negative seed offset. Retain
   a last-known offset per request, accounting for completed submitted samples
   before it becomes inactive. This can use existing `host_positions` plus
   `position_advances` at transitions; alternatively update a small host counter
   at successful `_execute` submissions, using the submitted active mask.
   Such bookkeeping uses no token readback. Do not advance these counters from
   `_sample_logits`, which is also called during warmup and capture. Resetting
   an unowned inactive slot can use offset zero; resetting a dormant owned slot
   must retain its known logical offset. Preserve explicit scoped reset
   semantics and untouched peers.
5. Clear metadata on actual release/reset and replace it on fresh admission;
   carry it with permutations. Cache identity matters: multiple prepared cache
   bindings must not share slot origins accidentally. Snapshot/restore any
   bookkeeping affected by setup probes, or keep capture outside its updates.

For explicit authoritative output histories, the existing seed calculation is
already correct. If those histories describe a rewind or a different logical
offset, refresh the stored position relationship from the supplied count and
authoritative decode position so a later omitted-history transition does not
undo that reset. Explicitly empty histories mean zero outputs, not “preserve
the previous stream.”

This is a generator metadata repair; changing shared sampling math or plugin
interfaces is unnecessary for these ordinary transitions. The existing
`SamplingSlotState` remap/reset primitives remain usable as-is. Keep penalty
history reconstruction separate: knowing a count cannot recreate token IDs.

## Alternatives and boundaries

**Preserve the current device seed whenever histories are missing:** tempting
and smaller, but insufficient. A host-sampled interval leaves the device seed
behind the logical output count; an authoritative rewind/reset also requires
reconstruction. This silently weakens the reset command's meaning.

**Use `len(output)` unless zero, then preserve:** also wrong. Explicit empty
history is meaningful, and normalization has already discarded absence.

**Use `start_pos` directly as the count:** counts prompt tokens as generated
outputs. Reading it during resident execution additionally trusts intentionally
stale host state. Subtract the admitted prompt origin and consult it only on
an authoritative reload.

**Always ask the plugin to send histories:** can supply the count but extends
the shared integration behavior and transports output tokens merely to obtain
their length. It is unnecessary for the identified ordinary request lifecycle.

**Resume/preemption caveat:** a newly introduced prefill may contain both the
original prompt and generated outputs. Plugin `submit_prefill` supplies prefix
end/start positions but no original prompt/output boundary. The model cannot
infer that boundary for an unknown resumed request from token IDs and absolute
positions alone. The same information gap already affects presence/frequency
penalties. Do not claim the proposed fresh-admission origin covers all resumed
requests. If existing request metadata survives and identifies the origin, use
it; if explicit histories arrive, use them. A released/recreated request with
omitted histories needs additional authoritative information or an explicitly
validated lifecycle strategy. No correctness claim for that case is made here.

The seeded claim is distinct from `seed=None`: current resets choose new random
bases for unseeded requests. Extending deterministic continuity to unspecified
seeds would require preserving each request's randomly chosen base too.

## Host check performed

Executed `python_env/bin/python` with an inline AST harness extracting the actual
`normalize_histories`, `_sampling_histories`, and `_reset_sampling_state`
functions. No TT, plugin, vLLM, or model module was imported. The formatter and
hash were replaced with transparent stubs; hash result 100 makes the initial
device seed 101. A fake slot-state sink recorded the reset value.

```text
first_decode_missing actual_next_seed 101 expected_next_seed 102
first_decode_explicit actual_next_seed 102 expected_next_seed 102
later_transition_missing actual_next_seed 101 expected_next_seed 106
later_transition_explicit actual_next_seed 106 expected_next_seed 106
proposed_position_offset_checks PASS: first decode, later transition,
destination remap, explicit empty
```

These checks reproduce the offset loss independently of sampler math. The
position-offset checks are arithmetic discrimination, not execution of a
finished repair. Source remains unchanged by this investigation.

## Discriminating validation after implementation

Host contract tests should reproduce the plugin command sequence with no
penalties and omitted histories: device-prefill then first decode; several
resident steps then forced membership reload; full slot permutation; host
prefill then device decode; a host-sampled interval; release/reuse with the same
seed; inactive/dormant slots; explicit history rewind; explicit empty history.
Feed sentinel objects that raise on access as resident `tokens`/`start_pos` to
prove the repair never consults stale inputs. Assert exact pre-sample seed
values, not only sampled-token equality. Verify all four supplied commands
still execute and remaps execute exactly once.

Then extend the reduced transition probe using two otherwise identical seeded
runs with default penalties, supported stochastic controls, and traces captured
before admission. One run continues uninterrupted; the other forces the exact
plugin reset commands **without histories** after prefill and after several
decode steps, and swaps request slots. Compare per-request next seeds and
output sequences after undoing the slot permutation. Read seeds only as test
instrumentation. In production, ordinary resident replay must still perform
no extra token/position uploads or readbacks. A host/device interval needs the
logical-offset assertion separately, since different sampler implementations
need not produce identical tokens. No hardware result is claimed here.
