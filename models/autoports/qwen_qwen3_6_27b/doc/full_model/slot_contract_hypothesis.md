# Slot and inactive-row contract investigation

Source-only AutoFix hypothesis investigation, 2026-11-14. No TT device was
opened, no implementation was edited, and no hardware correctness claim is
made. Sources: tt-metal HEAD `7be8c4348666622686a6bdf5efa72e73ecf4c9b0`
plus the parent's in-progress model/generator, and pinned vllm-tt-plugin
`042a94863ad27d88207bcf71abdf67122c6f4c27`. The parent is editing the generator
concurrently; findings below describe the reviewed code and the required
closure evidence, rather than asserting that every finding remains unfixed.

## Confirmed source findings

### Tiled `slice_write` changes the destination storage

The original `release_request` and mixed-prefill writeback call
`ttnn.experimental.slice_write(source, target, ...)` for both recurrent
tensors. Conv is row-major `[B,3,2560]`; recurrence is tiled interleaved
`[B,12,128,128]` (see `MultichipDecoder.create_recurrent_state`).

`ttnn/cpp/ttnn/operations/experimental/slice_write/slice_write.cpp` sets
`rm_only_not_sharded` for this tiled interleaved case. The device branch then
assigns its non-const `Tensor& output_tensor` to a newly untilized tensor,
writes there, and assigns it to a newly tiled tensor. This is not an in-place
update of the original tiled allocation. Even if numerical output is correct,
captured decode addresses and `reset_pairs` references can become inconsistent.
An eager preparation call does not fix a later replacement of destination
storage.

Minimal repair: apply the slice to a separate preallocated row-major staging
tensor, then copy the tiled result into the original persistent tensor; or
merge the selected row through `where` and `copy` into the original tensor.
Keep the existing optimized BF16 persistent recurrence layout. Never call the
tiled destination form directly on a tensor whose address a retained trace
uses. Verify all per-device buffer addresses before and after release,
mixed-prefill writeback, and traced decode, under allocation tracking.

### Prefill page rows and recurrent slots are different indices

The plugin's `MODEL_EXECUTION.md` explicitly defines prefill row `i` as
`tokens[i,start_pos[i]:prompt_lens[i]]`, using `page_table[i]`, while its
recurrent destination is `empty_slots[i]`. The original generator first
requires `page_table.shape == self.page_host.shape`, then selects the cached
table by destination `slot`. A one-row prefill into slot 2 either fails the
shape check or reads slot 2's table instead of submitted row 0's table.

Minimal repair: validate submitted row tables independently of fixed slot
capacity, map row `i` into the table/state belonging to `empty_slots[i]`, and
preserve every omitted slot. Bounded prefill should consume the submitted
row table while committing recurrent state to its destination slot. Preserve
the separate decode table contract. Multiple KV groups require respecting
`page_tables_per_layer`; silently swallowing that keyword would be wrong.

### Inactive recurrent rollback alone is insufficient

The initial decode graph advanced all unsigned RoPE positions; the sample
graph overwrote every token row and incremented every seed. Meanwhile only
signed positions used `skip_negative_entries=True`, and the new model rollback
preserved GDN conv/recurrent rows. A dormant slot can still belong to a live
request. Its sampler/RoPE history must not accidentally advance while inactive.
This also eventually makes an inactive RoPE gather out of bounds despite its
attention position remaining -1.

The parent has begun adding device `where` restoration of inactive RoPE,
tokens, and seeds. Verify integer values remain UINT32 throughout; BF16
conversion would destroy large token/seed values. This checkout has integer
`where` support and tests in `test_where.py`, including mixed float predicates
and INT32 values, so an exact integer select is a plausible repair.

Attention's -1 behavior is supported by source: paged update reader/writer
kernels skip update_idx=-1, and SDPA decode reader/compute/writer kernels
return for UINT32_MAX. GDN does not consume positions, so its transaction
remains necessary. The model's clone/update/where/copy transaction is
semantically sound per row and keeps the persistent destination address;
actual broadcast/layout execution still needs a hardware probe.

### Placeholder page rows must not silently disable optimized prefill

`MODEL_EXECUTION.md` describes inactive lane-DP rows with position -1 and
zeroed page-table rows. The initial `_bind_pages` conservatively applies
`lease.certified &= certify_host_page_rows(all_rows)`. Repeated zero entries
in ordinary padding therefore permanently revoke prefill bulk eligibility,
even if they represent no owned pages. This affects future requests as well.

Use valid ownership evidence for the pages a prefill can access, such as
separately owned private prefill tables or explicit active/range certificates.
Do not undo `CertifiedPagedState`'s deliberate revocation after public raw
tensor exposure. A changed table with real aliasing must still disable the
unsafe bulk path.

### Reload commands are not the same as `resident_inputs`

The reviewed low-level decode accepts `resident_inputs` and `**kwargs`, but
does not implement the four version-1 commands: `reload_inputs`,
`reload_page_table`, `reload_sampling_params`, and `reset_sampling_state`.
The pinned plugin sends these as commands. `reload_inputs=False` means host
tokens/positions may be stale. Silently swallowing these keywords would
overwrite resident feedback because `resident_inputs` defaults false.

The parent must either map all four commands explicitly in the generator's
serving API or clearly document the exact future adapter mapping without
claiming direct support. Declaring version 1 requires rejecting legacy
`reset_batch`; page-only updates must leave token/position/sampler state
unchanged. Sampling configuration and RNG reset must be separate from mere
forward-input reload. Seeded transitions should reproduce the documented
counter convention rather than restarting every continuing request at zero.

## Remap assessment

The full-capacity permutation validation and gather direction are correct:
destination `i` reads old `remap[i]`. The plugin's
`TTModelRunner._decode_state_slot_remap` builds a complete permutation over
state-slot capacity, including requests outside the current decode bucket.
Gather into an independent result followed by copy avoids cycle clobbering.
The reviewed code covers recurrence, conv, resident tokens, k/p/temperature,
device seeds, and greedy tie mask, including when host sampling is selected.

The following remain explicit integration/probe work:

- Remap execution is eager after capture and creates gather/layout temporaries.
  Preparing its kernels does not prove allocator safety after capture. A
  separately retained remap trace with stable index buffers is a straightforward
  bounded solution; otherwise allocation tracking must prove safety.
- Sampling `_scale_logprobs` and any enabled penalty/history/host SeedManager
  state are not in the custom remap list. They are currently dormant in the
  reviewed `_run_sampling(...penalties_on=False)` path. Keep unsupported
  features explicit, or move/reset every state owner when enabling them. The
  common `SamplingGenerator.apply_slot_remap` exists but intentionally marks
  device state as requiring authoritative reload, so do not combine it
  blindly with the custom device gather.
- Positions and active flags are omitted from remap. This is safe only while
  a remap always requires a complete authoritative input reload of the full
  fixed capacity. If bucketed decode is enabled, preserve/remap off-batch
  position/RoPE/active state or define it as reconstructible on its next use.
- Current decode expects exactly the configured token/position row count.
  Advertise only that prepared batch size until bucket handling exists;
  full-capacity remaps must never be truncated to the active token count.
- Validate all token, position, page, slot, and sampling arguments before
  executing a remap or clearing a slot. A later Python shape error after a
  device remap leaves the plugin's uncommitted ownership map stale. The
  original prefill also clears start-zero slots before all row interval
  validation, which can erase a live slot on a rejected call.
- `release_request` may clear recurrence without wiping KV, because page
  ownership is external and future valid positions overwrite cache entries.
  It must also dispose of any actual request-owned sampler history on reuse,
  through explicit request/reset commands; clearing recurrence alone is not
  a complete sampling reset. Do not clear omitted but still-owned slots.

## Minimal discriminating hardware probes for parent

Use the smallest reduced model with one GDN and one full-attention layer,
fixed capacity 3 or 4, then repeat relevant acceptance cases on all64.

1. Fill each recurrent row with a distinct finite pattern; release one row.
   Assert the released row is exactly zero, every other row is bitwise
   unchanged, and all persistent cache addresses are unchanged. Repeat after
   capture and before a decode replay.
2. Apply a three-cycle remap `[2,0,1]` to patterned conv/recurrent and distinct
   sampler parameters/seeds. Compare against CPU `index_select`. Repeat with
   host sampling and then device sampling, verifying each remap is applied
   exactly once and unchanged identities do nothing.
3. Prefill a single nonaligned prompt into a nonzero slot using a one-row
   page table. Snapshot other slots. Decode that slot and compare with an
   isolated same-prompt baseline. Repeat resumed-prefill with nonzero start.
4. Decode positions `[p,-1,q]`, unique seeds above 65536 and tokens above
   65536. After multiple resident steps, inactive recurrent state, token,
   seed, and safe RoPE position remain bitwise unchanged; signed position
   remains -1. Active rows match a no-pause baseline. Include inactive zero
   page rows without writing page0 or permanently losing eligible prefill.
5. Run identity, three-cycle, release/reuse, and changed/unchanged page-table
   cases after capture with `TT_METAL_TRACE_ALLOC_TRACKING=1`, its positive
   control, stable trace IDs, and per-device persistent addresses recorded.
6. Deliberately supply stale host tokens/positions with version-1
   `reload_inputs=False`, then page-only reload. Assert resident decode follows
   its prior sampled token and device position, and page-only mutation causes
   no token/position refresh. Reject unsupported/reset legacy keywords before
   any state mutation.

These probes distinguish numerical correctness from address/ownership
correctness; passing only eager output comparisons does not close the trace
contract.
