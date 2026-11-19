# Slot migration and sampling reset audit

This is a CPU source audit against plugin commit
`042a94863ad27d88207bcf71abdf67122c6f4c27` and the current repo common sampler.
`tt/slot_state.py` is implemented and structurally tested; its TTNN operations
have **not** been executed by this audit. Hardware trace/permutation acceptance
is still required. No model/generator runtime file was edited by this audit.

## Fixed-address migration

The plugin `src/vllm_tt_plugin/model_runner.py::_decode_state_slot_remap`
constructs a full permutation, including unscheduled slots. `remap[i]=j`
means destination `i` receives old source `j`. Duplicate sources, short maps,
and new-request sentinels are invalid for this entrypoint. Validate the entire
submission before mutating any subsystem; the plugin only commits ownership
after `decode_forward` accepts it. Do not execute the remap a second time in a
host ownership callback.

For the current independent per-slot KV allocations:

1. Allocate one extra `model.create_cache(capacity=...)` scratch cache before
   preparation when the configured batch exceeds one. Its full memory cost
   must appear in capacity accounting. `SlotStateMover(caches,scratch)` checks
   shape, dtype, layout and Python tensor aliasing; caller must also ensure
   distinct underlying storage if external TT tensor views are accepted.
2. Run `mover.warmup()` before any trace capture. It copies slot0 into scratch
   and back, warming every tensor copy shape and preserving live values.
   Warm `remap_token_buffer(decode_tokens, range(max_batch_size), warmup=True)`
   at the same point; it exercises all 32 slice offsets, concat and final copy.
   This extra concat/copy needs hardware validation on ROW_MAJOR uint32.
3. At a nonidentity transition, call `mover.apply(remap)` on CQ0, then
   `remap_token_buffer(decode_tokens, remap)`. Migration moves tensor values;
   neither the RequestCache objects nor tensor allocations move. Existing
   trace pointers and PLE hash history aliases remain valid.
4. Reindex `page_tables` upload shadows with
   `remap_slot_mapping(page_tables, remap)`. If direct high-level compatibility
   sampling keeps per-slot HostSampler objects, move those objects with the
   same helper, including dormant/off-batch objects. Never recreate a seeded
   HostSampler at an ordinary slot move, which would restart its RNG.
5. Apply authoritative input/page refresh commands only after migration.
   Refreshing positions alone never restores GDN/PLE/indexer state.

Copied persistent tensors are GDN conv and recurrent, PLE conv and token
history, QSA position/page table/K/V/raw three-key history/compressed-key cache.
The dtype split is preserved by same-dtype copies. Host `cache.position` is
scheduler bookkeeping, moved separately. PLE hash input/output buffers are
overwritten each invocation, their controls are constants, and QSA number
arrays are immutable shared constants. Prefill output and sample logits are
overwritten before each decode consumes them and need no migration. Resident
sampled token feedback does need migration, even when forward reload is off.

This helper does not implement a shared global KV pool: copying shared pages
could corrupt other requests. Such a pool needs a distinct ownership contract.
Changing a per-slot page permutation changes how existing physical pages are
interpreted; the external owner must supply a mapping consistent with actual
page contents. Merely uploading a new permutation does not relocate history.

## Sampling policy and dormant state

`models/common/sampling/generator.py::SamplingGenerator.apply_slot_remap`
moves SeedManager's host request RNG bookkeeping, then invalidates device
sampling parameters and penalty history. It does **not** gather those buffers.
Pad the scheduler permutation with identity to physical width32 and call this
method once on every remap, including host-sampling calls. A future device
sample must call `validate_decode_state_commands` and require both
`reload_sampling_params=True` and `reset_sampling_state=True` after a move.
Only `commit_decode_state_commands(...,sampling_state_slots=None)` clears
whole-device invalidation. Calling the private `_run_sampling` bypasses this
validation, so the generator wrapper must enforce it itself.

The common penalty update counts all32 sampled rows and has no active-row mask.
Inactive logits being finite does not fix this. Declare
`model_capabilities['supports_device_penalties']=False`, reject nonneutral
penalties on direct device-sampling calls, and keep `_penalties_active=False`
for the measured path. The pinned plugin's `model_runner.py` routing already
sends penalized requests to its own host sampler. This preserves penalties;
do not silently neutralize them. Device penalty buffers are then never read
as request history. A mode transition rebuilds sampler parameters and RNG
before using it again. Future device penalty support requires actual masked
counting and complete history remapping/rebuild, plus tests.

Set `max_device_top_k=32` in model capabilities; the plugin already routes
non-greedy unbounded/larger-k requests to host. The explicit CPU compatibility
API supplies request-local seeding; seeded support evidence must test that API
through the generator. A `seeded_sampling_policy` model capability is not read
by this plugin and must not be advertised as routing support. The plugin has
an existing operator policy in additional_config, but that is not evidence of
model support and must not close a missing full-model requirement. Under the
plugin's default `auto` policy a seeded request can still reach the device
path: this remains a model integration gap unless the generator implements
that contract. Direct unsupported seeded device requests must raise rather
than drop their seed or silently switch the measured token-out path to host.
This audit introduces no plugin option or implementation and does not close
the seeded serving row on the basis of future launch configuration.

Unseeded device RNG resides in per-core hardware registers, not slot-gatherable
buffers; the plugin contract explicitly exempts it from slot migration.
On every commanded sampling reset, including `seed=None`, call
`seed_manager.reset_seed([None]*32, list(range(32)))` and `get_new_values()`
once to enqueue initialization. The sampling trace copies MAX_UINT32 SKIP
into the persistent seed tensor after drawing; subsequent draws advance
hardware RNG without per-token host seed uploads. Do not derive/reset seed
counters from stale host positions on resident-input decode.

## Authoritative command inputs

| Input/command | Authority and required behavior |
| --- | --- |
| `slot_remap` | Host scheduler ownership metadata, applied once across all allocated slots before reading state; independent of reload booleans. |
| `reload_inputs` | Tokens, absolute positions and page tables are authoritative; upload into existing allocations. False means token/position values may deliberately lag the device. |
| `reload_page_table` | Page tables only; mutually exclusive with full input reload. Do not upload tokens or positions. |
| `reload_sampling_params` | Use this submission's sampling parameters, padded/scattered by scheduler slot; reloading params must not silently clear output history. |
| `reset_sampling_state` | Requires reload_inputs. Rebuild mutable history/RNG from authoritative state, including unseeded RNG initialization. A whole-device sampler invalidation needs a whole-device rebuild. |
| `prompt_tokens`, `output_tokens` | Authoritative complete prompt and generated history for a commanded penalty rebuild, not a delta or just the latest input token. The plugin forwards these on device-sampling calls when applicable. In host mode the plugin owns histories and sampling. |
| `sampling_state_slots` | Optional common-sampler subset reset; not sufficient to clear whole-device invalidation after a permutation. |
| negative position/inactive row | Do not advance model cache state. Plugin compatibility sampler handles live request rows; common all32 penalty updates are unsupported. |
| `reset_batch` | Reject when declaring decode_input_update_contract=1. |

Defaults for direct decode callers: reload_inputs=True; the other commands
False. The version1 plugin sends all four explicitly. Under host sampling,
return raw logits to the plugin and do not apply HostSampler a second time.
The CPU HostSampler is for explicit standalone compatibility tests/generation.

## Validation

`tests/test_slot_state_cpu.py` exhaustively verifies every permutation through
six slots, distinct cycle lengths, independent cycles, all persistent fields,
unchanged tensor identities, identity no-op, invalid-map atomic validation,
alias/geometry rejection, off-batch/dormant metadata and device-metadata
rejection. Five unittest tests passed; no TTNN import or device access.

Command:

```text
timeout 60 env PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.test_slot_state_cpu
```

Evidence: `doc/full_model/slot_state_cpu.log`. Hardware acceptance still needs
a two/three-slot cycle with different histories and pages, equality against
unmoved solo continuations, a dormant host/device sampling transition, no
trace reallocation, and inactive-row isolation. CPU tests do not prove TTNN
copy/gather support, trace safety or performance.
