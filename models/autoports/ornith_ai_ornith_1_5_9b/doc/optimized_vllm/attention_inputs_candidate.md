# Attention input construction and refresh candidate

CPU-only preparation; `attention_inputs_candidate.patch` is not a latency claim.
No runtime source or device was changed by the preparing agent.

`make_attention_inputs` constructs identical immutable plans during repeated
validation and per-layer refresh. An eight-entry LRU now reuses that work. Every
call still checks scalar extent/context constraints and snapshots caller page
maps into a new normalized tuple. The cache key includes physical bucket,
context, physical page count, start, consumed count, every mapping element and
page offset. Cached construction retains all mapping/causal-prefix validation.
Changing an in-place scheduler list produces a different key; failed validation
never becomes a cached result. The frozen result contains only integers/tuples,
not model, cache, device or mutable caller references. At most eight plans and
their keys remain process-owned; their sizes are bounded by configured context
and physical bucket. Full-attention adapters with different offsets/capacities
cannot incorrectly share a key.

Each attention adapter separately records the values actually present in its
four persistent inputs. Constructor uploads initialize this record. A refresh
first verifies cache backing identity as before, constructs/gets the validated
plan, then transfers only changed fields. A later start normally changes RoPE,
fill pages and start, while an unchanged scheduler mapping retains attention
pages. A changed mapping is compared by content. Plan equality alone does not
authorize skipping another adapter's copies.

The current plan clears before transfer; each changed field invalidates before
copy and commits after success. Thus a partial failure blocks attention and
cannot leave stale metadata that skips repair on retry. Request/slot resets do
not mutate these input tensors, and new adapter construction establishes new
metadata. All geometry, precision, page semantics, prepared device allocations
and kernel calls remain unchanged.

Eight tests in `tests/vllm_attention_inputs_test.py` passed against the unapplied
candidate. They execute the actual pure functions and refresh writer, covering
immutable results, mutable maps, scalar keys, invalid mappings, cache size,
changed/unchanged fields, independent adapters, partial transfers and replaced
cache backing. After application, run that module without the temporary source
override. The existing `--refresh-cache-control --all-layers` device comparison
now clears the plan LRU on every baseline call and forces all attention field
copies. Its candidate exercises actual caches, records `attention_input` copy
counts and requires fewer copies with exact outputs/state and retained traces.

Neither source inspection nor copy counts establish a robust four-millisecond
serving gain. Keep responsive 1:2 and the same 2048 workload/config for the next
serving discriminator, then require the final acceptance runs. Do not select a
result from noisy reruns or alter the ITL bound.
