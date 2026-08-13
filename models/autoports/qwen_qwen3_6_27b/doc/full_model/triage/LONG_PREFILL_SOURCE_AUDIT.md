# Long-Prefill Source Audit

## Scope and starting evidence

This is a source-only AutoFix audit of the final full-model public prefill path.
No TT device was opened and no implementation file was changed.  The audit
compared:

- `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`,
  `tt/optimized_decoder.py`, and the inherited linear-prefill reducer in
  `tt/functional_decoder.py`;
- `doc/context_contract.json`, the B1/B32 full-model capacity artifacts, and
  the earlier per-layer S=192511 capacity evidence;
- `doc/full_model/STAGE_REREVIEW.md` and the public-contract/mixed-slot tests.

The relevant earlier evidence is not a public full-wrapper execution.  The
S=192511 passes are target-shape *single decoder layer* capacity runs (one full
attention layer and one linear attention layer).  The full-model capacity JSON
allocates the accounted resident tensors but does not execute 64 layers or the
generator at S=192511.

## Hypothesis experiments

### Hypothesis: default prefill still allocates sequence-by-vocabulary logits

Experiment: trace `Qwen36Generator.prefill_forward(return_all_logits=False)`
through `Qwen36Model.prefill_forward`, `select_prefill_terminal_rows`, and
`terminal_forward`, including the B>1 LM-head branch.

Result: refuted in the current source.  The generator passes `prompt_lens` as
`logit_positions`.  After all decoder layers, the model slices one device row
per fixed slot from `[1,B,S,5120]`, concatenates those rows along batch to
`[1,B,1,5120]`, pads only the row dimension to 32 for the retained LM-head
program, projects each batch slot separately, concatenates along batch, and
slices the result back to one logical row.  Therefore the normal path's largest
LM-head activation is proportional to `B * 32 * vocab`, not `S * vocab`.

For B1, the selected source tensor is `[1,1,1,5120]`; terminal padding produces
`[1,1,32,5120]`.  For B>1, per-slot slices remain `[1,1,32,5120]` through the
one-tile DRAM-sharded projection and are concatenated to `[1,B,32,local_vocab]`
before the one-row slice.  There is no host hidden-state or logits selection.

Remaining uncertainty: the selected sequence start can be non-tile-aligned
(for S=192511 it is row 192510).  Source shapes are consistent, and small
non-aligned public calls have run, but the exact near-limit slice/layout and
allocation must still be exercised on hardware.  Also,
`return_all_logits=True` deliberately retains the old sequence-by-vocabulary
behavior and cannot be treated as a near-context-capable mode without separate
evidence.

Verdict: the rereview's terminal-allocation bug is fixed for the default public
mode, but the advertised maximum remains unproven.

### Hypothesis: non-aligned and mixed prompt lengths corrupt prefix state

Experiment: follow generator metadata through both layer kinds for an arbitrary
physical length and mixed `prompt_lens`, including S=192511 (3007 full 64-token
chunks plus a 63-token tail).

Result: source contract is internally coherent.

- The public generator accepts an arbitrary logical/physical extent up to its
  limit and builds exact-length positions.
- Linear attention receives one mask and four convolution selectors per
  64-token chunk.  The last chunk uses its logical tail length.  Masked rows
  compose the identity affine transform with zero bias; selectors retain the
  exact four-token state at each row's logical end.  The balanced concatenator
  bounds retained chunk references logarithmically.
- Full attention above 32768 fills K/V in 32768-token page-aligned chunks,
  pads only the final query chunk to a tile, and slices its result back to the
  logical tail.  The 32768 boundary is divisible by page size 64.  Terminal
  selection occurs only after the complete layer stack.
- For a shorter positive row in a mixed physical batch, full-attention padding
  positions are filled, but causality preserves every prefix output and decode
  overwrites the current position before reading it.  A zero-length inactive
  row uses an all-`-1` cache-fill page table and contributes an explicitly zero
  terminal row.

Verdict: no source-level masking/cache-fill indexing bug was found.  This is
not a substitute for the required long public-wrapper run.

### Hypothesis: long-prefill temporary state is released safely

Experiment: enumerate explicit deallocation and retained Python/decoder
references across a public prefill.

Result: **verified source bug** in request metadata lifetime.  The generator
uploads approximately `ceil(S/64) * 5` device tensors (`sequence_mask_tt` plus
four selector tensors per chunk).  Every decoder assigns those same lists to
`_sequence_masks` and `_conv_state_selector_chunks`; neither the model nor the
generator clears those attributes after prefill, and the generator never
deallocates the metadata tensors.  At S=192511 this is 15,040 device Tensor
objects retained by all 64 layer objects after the request.  Repeated prefills
replace the attributes but make release dependent on Python destruction and
async allocator timing, and the first call retains the entire metadata set for
the model lifetime.  This is a real serving leak/lifetime defect and makes a
near-limit capacity run less trustworthy.

There is a related stale-handle issue: `_cache_page_table` on every layer keeps
referencing the temporary inactive-slot page table after the generator has
explicitly deallocated it.  Decode does not consult this attribute and the next
prefill overwrites it, so no current wrong-result path was found, but the
request-scoped binding should be cleared at the same boundary.

The long full-attention implementation explicitly deallocates K/V projection
temporaries and query tensors.  Other loop locals become unreachable as the
loop advances or method exits; whether their async release is sufficiently
prompt cannot be proven from source alone.  The required hardware experiment
must capture allocator snapshots by layer/chunk and after return, and repeat a
prefill to detect retained allocations.

Verdict: verified; implementation should clear request-scoped layer bindings
and explicitly release uploaded masks/selectors only after queued consumers
have completed (or at an equivalent proven ordered lifetime boundary).

## Context-contract conclusion

The current `192511` public claim still requires a concrete all-64-layer
`Qwen36Generator.prefill_forward(..., return_all_logits=False)` hardware pass.
The B1 `C=262144` capacity artifact proves weight/cache/state residency only;
the earlier S=192511 runs prove each layer kind in isolation.  Neither proves
the full wrapper's peak transient live set, 64-layer request-metadata lifetime,
terminal non-aligned slice, or successful host logits result.  Consequently
`full_model.status: implemented_and_physically_validated` and the note calling
192511 the largest tested full-layer prompt overstate current evidence.

Do not reduce the advertised limit from source reasoning.  First fix the
metadata lifetime defect and run the focused probes below.  Reduce only if the
real full-wrapper run finds a hard physical boundary, then bracket the largest
passing non-aligned length and the smallest adjacent failure.

## Focused post-reboot experiments

1. **Reduced wrapper shape/layout probe (B1 and B2).** Use one real layer of
   each kind with physical S=32769 and mixed logical lengths `[32769,32767]`.
   Run default terminal-only logits, assert output `[B,1,vocab]`, compare the
   selected logits with the corresponding rows from a small-enough
   `return_all_logits=True` control, and log every selected/terminal tensor's
   logical shape, padded shape, layout, dtype, and memory config.
2. **Metadata lifetime A/B.** Snapshot per-device DRAM before prefill, after
   prefill synchronization/readback, and after request metadata cleanup.  Run
   the same reduced prefill twice.  The second post-return allocation level
   must equal the first within stable program-cache allocations; no set of
   15,040 request tensors may remain model-owned.
3. **Public full-wrapper maximum.** With all 64 official-weight layers, B1,
   generator-owned paged cache/table, physical and logical S=192511, invoke
   public `prefill_forward(return_all_logits=False)`.  Require successful
   `[1,1,248320]` host output, finite/nonzero logits, populated full-attention
   cache at first/last pages, populated final linear state, and a subsequent
   decode at position 192511.  Record elapsed time and allocator peak/free
   blocks.  This is the missing capability proof.
4. **Near-limit non-aligned mixed control.** If physically feasible for B>1 at
   its separately supported context, use unequal logical lengths ending in a
   63-token tail and prove each terminal result matches its all-logits row on a
   reduced control, inactive padding does not change prefix state, and decode
   starts at each logical length.
5. **Failure bracketing only if needed.** If experiment 3 fails from a proven
   allocator limit after metadata cleanup, binary-search non-aligned physical
   extents and test the immediate practical boundary.  Update code/JSON/docs
   only to the largest real full-wrapper pass, preserving the separate 262144
   decode-cache contract.

## Final status

Source-only result: **still incomplete**.  Terminal hidden-row selection fixes
the sequence-by-vocabulary allocation for the default mode and no new
shape/masking/cache-index bug was found, but request-scoped mask/selector
metadata is retained after prefill and the 192511 public capability still lacks
the required full-wrapper hardware proof.
