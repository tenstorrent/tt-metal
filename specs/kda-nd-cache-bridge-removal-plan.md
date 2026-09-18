# KDA direct-ND bridge removal plan

Date: 2026-09-14

Branch: `momcilo/kda-nd-cache`

Parent bead: `tt-metal_tracker-ef4`

Prior attribution: `artifacts/kda-nd-cache/kda-nd-cache-performance-analysis.md`

## Verdict and scope

Remove the two SP-only placement bridges in measured-risk order:

1. replace the redundant recurrent `all_broadcast` with a local
   `ttnn.to_memory_config` conversion and measure it before considering a new
   writer kernel;
2. preserve the required convolution tail exchange, but let the QKV reader
   select canonical ND history on SP rank zero and the interleaved predecessor
   tail on later ranks, eliminating the initial-history staging conversion.

Do not change the cache contract, recurrence math, required SP summary gather,
required convolution-tail exchange, activation layouts, or generic TTNN
behavior. A fused recurrent ND writer is a measured follow-up, not part of the
initial implementation.

The branch is currently 139 commits behind `origin/main` and 11 commits ahead,
with merge base `3f254861838`. Rebase and same-base measurement therefore
precede optimization.

## Target data flow

```text
canonical ND state
  recurrent
    -> existing distributed affine prefix
    -> identical local carry on every SP rank
    -> local interleaved-to-ND conversion
    -> canonical ND replacement

  convolution
    -> rank 0 QKV reader reads canonical ND history
    -> rank > 0 QKV reader reads gathered predecessor tail
    -> existing fused writer emits canonical ND replacement
```

No operation may communicate or convert state solely to make two physical
layouts match. The first recurrent step intentionally retains a local
conversion as a low-risk measurement point; it removes the unnecessary fabric
communication but is not claimed as the final fused design.

## Phase 0: rebase, build, and baseline

Bead: `tt-metal_tracker-ef4.1`

1. Fetch `origin/main` and `origin/momcilo/kda-nd-cache` and record both SHAs.
2. Preview the rewrite with commit lists and `git range-diff`; preserve a local
   backup ref before rebasing.
3. Rebase the direct-ND commits onto the verified latest `origin/main`, resolve
   changes using current KDA and Metalium patterns, and synchronize submodules.
4. Build from the repository root:

   ```bash
   ./build_metal.sh --build-type Release --enable-ccache
   ```

5. Capture or reuse a provenance-matched latest-main measurement, then run the
   unchanged rebased direct-ND SP1xTP8, SP2xTP4, and SP4xTP2 matrix.

Commit concern: rebase/compatibility fixes only. Do not mix either bridge
optimization into this commit.

Gate: Release build passes; real Kimi-K3 accuracy and exact ND state contract
pass on all layouts; same-base layer medians exist before optimization.

## Phase 1: local recurrent ND conversion

Bead: `tt-metal_tracker-ef4.2`

In `_distributed_affine_prefix`, retain the final carry locally on every rank
and replace:

```python
final_state = ttnn.all_broadcast(
    carry,
    cluster_axis=sequence_parallel_axis,
    memory_config=state_memory_config,
)[0]
```

with:

```python
final_state = ttnn.to_memory_config(carry, state_memory_config)
```

Do not add an output buffer, wrapper, mode flag, or new kernel in this phase.
The returned tensor remains the replacement state owned by the caller.

Validation:

- exact recurrent ND shard config and DRAM placement;
- exact equality of recurrent state across SP replicas;
- output, recurrent-state, and convolution-state PCC;
- trace capture and replay plus program-cache reuse;
- SP1/SP2/SP4 trace-wall medians;
- targeted device profile proving `all_broadcast` is absent and attributing the
  local copy.

Expected result: replace the 171.9/379.0 us SP2/SP4 broadcast with a local tiled
copy expected to be in the tens of microseconds. Investigate before proceeding
if the local copy exceeds 50 us at SP4 or if SP4 improves by less than 250 us.

Commit concern: `perf(kda): replace recurrent ND broadcast with local copy`.

## Phase 2: dual-source convolution history

Bead: `tt-metal_tracker-ef4.3`

### Python orchestration

Keep the existing projected-tail gather. Change `exchange_convolution_carry`
so it no longer slices the canonical ND initial history into interleaved DRAM
or concatenates that converted tensor with predecessor tails. It should produce
only the interleaved predecessor-history source and the existing final-state
source.

Pass two history tensors to `qkv_causal_conv1d_silu`:

- `initial_history`: canonical `[1,3,64]`-sharded ND DRAM cache;
- `predecessor_history`: interleaved history derived from gathered projected
  tails. Its rank-zero content is unused.

### Operation and kernel contract

Derive the SP rank from the operation's mesh dispatch coordinate, following an
existing mesh-coordinate-aware TTNN program-factory pattern. Prefer a
compile-time `is_first_sp_rank` specialization over an inner-loop runtime rank
branch.

The QKV reader selects the semantic source by rank and addresses that source by
its tensor accessor type:

```cpp
if constexpr (is_first_sp_rank) {
    read_history(initial_history, ...);       // ND accessor
} else {
    read_history(predecessor_history, ...);   // interleaved accessor
}
```

`read_history` remains templated on the accessor layout. The existing fused ND
state writer remains unchanged.

Do not create a heterogeneous-layout mesh tensor, convert predecessor tails to
ND, or fold the tail collective into the QKV kernel in this phase.

Validation:

- focused reader coverage for ND rank-zero and interleaved later-rank sources;
- SP1 direct history behavior remains unchanged;
- SP2/SP4 per-rank convolution state agrees with the CPU reference;
- real and patterned cache identity, trace replay, and program-cache rebinding;
- targeted profile contains no three-row ND-to-interleaved staging copy;
- QKV kernel time remains within run-to-run noise of the Phase 1 result.

Expected result: recover approximately 122 us at SP4 without changing the
required tail exchange.

Commit concerns:

1. operation/kernel dual-source contract and focused tests;
2. Python SP orchestration switch and layer validation.

Keep these separate if the kernel contract can be validated independently.

## Phase 3: recurrent writer decision

Bead: `tt-metal_tracker-ef4.4`

After Phases 1 and 2, retain the local conversion unless both conditions hold:

1. its measured cost materially limits the end-to-end result; and
2. direct ND does not beat main plus the optimized adapter by at least the
   larger of 20 us or three median-absolute-deviations across session medians.

If both hold, prototype one KDA-local operation that fuses the final affine
addition with a layout-aware ND writer. It must write the already-local result
on every SP rank and perform no collective. Reject the prototype if its gain is
not larger than measurement noise or if it adds another intermediate state.

Record the decision even when no kernel is added.

## Phase 4: final matrix and report

Bead: `tt-metal_tracker-ef4.5`

Run the authoritative Release build, focused KDA operation tests, KDA component
and state-contract tests, and the real Kimi-K3 layer performance matrix on all
three layouts. Use synchronized warm trace-wall medians and targeted device
profiles; do not sum device-program durations as a wall-time model.

Report one same-base table containing:

- latest main;
- rebased direct ND before bridge changes;
- direct ND after local recurrent conversion;
- direct ND after convolution staging removal;
- latest main plus optimized-adapter export and round-trip costs.

Final acceptance:

- build and all focused/full accuracy tests pass;
- exact canonical ND shapes, page sizes, distribution, and SP replication pass;
- no recurrent placement broadcast and no convolution initial-history staging
  copy remain;
- SP4 direct ND beats main plus the optimized adapter by a material margin, or
  the direct design is rejected with measured evidence;
- Markdown results are committed with exact commands, SHAs, PCC, medians, and
  residual uncertainty.

## Commit and delivery discipline

Each semantic concern is committed only after its focused validation. Preserve
the untracked supplied HTML report. Push with `--force-with-lease` only after
the rebased remote range and lease are reverified; subsequent commits use normal
pushes. Do not combine the plan, rebase fixes, recurrent change, convolution
kernel contract, orchestration switch, or final report into one commit.
