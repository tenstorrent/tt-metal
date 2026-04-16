# BH Fast-Tilize BFP Overhead Reduction Plan

## Executive Summary

This document captures a detailed plan for analyzing and reducing the BFP-output
overhead in the Blackhole fast-tilize path.

Current conclusion:

- The BFP overhead is **not fully removable**.
- The overhead is **likely reducible**.
- The most credible near-term path is to remove the **software-managed
  per-tile launch/stall structure** in the current BFP fast-pack path and fold
  more of that work into a **single BFP-specific MOP + replay design**.
- We should **not** bet the main plan on undocumented or currently-unused
  packer features such as `Add_l1_dest_addr_offset` or auto-last generation
  until they are validated on Blackhole silicon.

In short:

- Some BFP cost is fundamental because BFP tile boundaries are semantically
  real.
- Some of the cost is an artifact of the current implementation strategy.

## Scope

This plan is specific to:

- Blackhole LLK fast-tilize pack path
- BFP output formats in the fast-tilize pipeline
- Primarily `Float16_b -> Bfp8_b` and `Float16_b -> Bfp4_b`
- Secondary consideration: `Float32 -> Bfp8_b` and `Float32 -> Bfp4_b`

This plan does **not** attempt to redesign regular tilize.

## Branch and Inputs Used

Repository context used for this analysis:

- Main workspace: `tt-metal`
- Active vendored LLK checkout:
  `tt_metal/third_party/tt_llk`
- Active branch in vendored LLK:
  `pjosipovic/ttsim-bh-llk-support`
- Head at time of analysis:
  `bb0b5575` (`Rewrite fast-tilize perf tests to mirror regular tilize CI matrix`)
- BFP support commit of interest:
  `a56fa4a9` (`Add BFP output support (Bfp8_b, Bfp4_b) for BH fast-tilize pack`)

Primary artifacts used:

- `BH_FAST_TILIZE_PERF_COMPARISON.md`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/instructions/assembly.yaml`
- `/home/developer/ttsim-private/src/tensix.cpp`
- Public ISA docs in `tenstorrent/tt-isa-documentation`

## Observed Performance

From `BH_FAST_TILIZE_PERF_COMPARISON.md`:

- Fast `Float16_b -> Float16_b` steady-state is about `26 cyc/tile`
- Fast `Float16_b -> Bfp8_b` steady-state is about `36-37 cyc/tile`
- Reported delta is about `~10 cyc/tile`

Relevant report notes:

- BFP output adds overhead due to per-tile `PACK|THCON` stall and L1 address
  update.
- Fast tilize with BFP output is still pack-bound.

This is the right starting model: the BFP penalty is inside the pack stage, not
 unpack or math.

## Current BFP Implementation Summary

The current BFP fast-pack path in
`tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h` does the following:

1. Loads a BFP replay that emits exactly one tile and sets `Last=1` on the
   final PACR.
2. Programs a BFP MOP with `outerloop=1`, `innerloop=1`.
3. Runs that MOP once per tile.
4. After each tile:
   - advances `OUTPUT_ADDR`
   - writes the new address into `THCON_SEC0_REG1_L1_Dest_addr` using
     `REG2FLOP`
   - issues `STALLWAIT(PACK|THCON)` before launching the next tile

Relevant code points:

- Replay layout and BFP replay:
  `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h:18-38`
- BFP replay with `Last=1`:
  `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h:126-143`
- BFP MOP shape:
  `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h:180-205`
- Per-tile run + stall loop:
  `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h:306-325`

## Why BFP Costs More Than Flat Float16_b

### 1. BFP tile closure is real hardware work

The `PACR` ISA explicitly defines `Last=1` as the tile-closing condition.

Relevant reference:

- `tt_metal/third_party/tt_llk/tt_llk_blackhole/instructions/assembly.yaml:1242-1355`

The BFP path uses `Last=1` at the end of **every tile**, because a BFP tile is
not just a flat byte stream:

- there is an exponent section
- there is a mantissa section
- tile-local shared exponent assembly must terminate cleanly at tile boundaries

That means BFP cannot simply inherit the flat-output strategy of "many tiles in
one MOP, one terminal close at the end of the unit" without changing the
meaning of the output.

### 2. The current code pays a software-visible per-tile launch cost

The current BFP implementation launches one MOP per tile from software, rather
than describing a unit of multiple BFP tiles as one higher-level MOP.

That introduces:

- per-tile MOP launch overhead
- per-tile thread-side control overhead
- per-tile `STALLWAIT` overhead

This part is **not fundamental**. It is a design choice in the current LLK.

### 3. The current code pays a per-tile L1 destination reprogramming cost

Today the BFP path updates `L1_Dest_addr` every tile via:

- `ADDDMAREG`
- `REG2FLOP`

Relevant code:

- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h:199-202`

This is required by the current implementation because:

- the packer is not configured to auto-advance the L1 destination
- the code does not currently use `WRCFG`-style address update replay in the
  BFP fast path

Some form of per-tile address progression is still required. The question is
whether we can make it cheaper.

## Fundamental Cost vs Avoidable Cost

| Component | Fundamental? | Current Payment Mechanism | Likely Reducible? |
|---|---|---|---|
| Per-tile BFP close (`Last=1`) | Yes | Final PACR in BFP replay | No |
| Per-tile shared-exp finalization | Yes | Implicit in packer at tile close | No |
| Per-tile MOP launch from software | No | `ckernel_template::run()` inside tile loop | Yes |
| Per-tile `STALLWAIT(PACK|THCON)` from software | No, in current form | Explicit thread-side stall after every tile | Yes, likely |
| Per-tile L1 dest update | Probably yes in some form | `ADDDMAREG + REG2FLOP` per tile | Maybe partially |
| Extra replay-programming overhead for unused flat replay in BFP mode | No | Loads both replay variants unconditionally | Yes |

The practical expectation is:

- We should not expect BFP to reach the flat fp16 floor.
- We should expect to recover some part of the `~10 cyc/tile` gap.

## Current Design Constraints

### Packer destination programming on BH

The standard BH helper `program_packer_destination()` writes the destination
address through `WRCFG`, with explicit `STALLWAIT(THCON)` before the write.

Relevant code:

- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h:536-553`

The existing BH untilize pack path already contains a replay-based address
update sequence:

- `ADDDMAREG`
- `STALLWAIT(THCON)`
- `WRCFG`
- `NOP`

Relevant code:

- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack_untilize.h:94-108`

This is important because it gives us a proven BH-side pattern for replaying
address updates between repeated pack operations.

### WRCFG scheduling rule

Public Blackhole ISA docs state that software must ensure the instruction
immediately after `WRCFG` does not consume the configuration that was just
written.

Reference:

- `https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/WRCFG.md`

The relevant point is that `WRCFG` is not self-synchronizing with its consumer.
So any proposal that depends on per-tile config updates must preserve this
ordering rule.

### STALLWAIT semantics

Public Blackhole ISA docs state that `STALLWAIT`:

- latches a wait condition
- blocks selected future instruction classes
- keeps blocking until the selected conditions are met

Reference:

- `https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md`

This reinforces that current BFP software-side stalling is a conservative way
to avoid launching the next tile while PACK and THCON work from the previous
tile is still in flight.

## Important Subtlety: MOP vs Wait-Gate Semantics

One subtle point is worth recording explicitly.

The public ISA docs describe `MOP` and `REPLAY` as instructions that are
handled by the MOP/replay machinery and "never reach Tensix itself". The same
docs describe `STALLWAIT` as operating through the Wait Gate.

Relevant references:

- `tt_metal/third_party/tt_llk/tt_llk_blackhole/instructions/assembly.yaml:3339-3419`
- `https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md`

That creates an open question:

- Does the current `STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON)`
  strictly gate the next `TTI_MOP(...)` launch in the way the comments imply?

This does not invalidate the performance analysis, but it **does** mean we
should not overclaim the exact mechanism until it is validated on silicon.

The plan below treats this as an implementation risk and includes an explicit
validation step.

## Recommended Direction

### High-level recommendation

Implement a **BFP-specific multi-tile MOP** that:

- preserves per-tile `Last=1`
- preserves per-tile address progression
- avoids software relaunch + software stall for every tile
- reuses the replay/end-op structure already proven in BH pack-untilize

### Why this is the best next step

It is the best next step because:

- it attacks the most obviously avoidable overhead
- it stays inside the current LLK programming model
- it does not depend on undocumented packer features
- it is compatible with the correctness reason that forced per-tile BFP close
  in the first place

## Proposed Design

### Design goal

Keep:

- one BFP tile replay with `Last=1` on the final PACR
- one logical BFP tile close per output tile
- one logical L1 destination advance per output tile

Change:

- move from "software loop over 1-tile MOPs" to "one MOP describes a multi-tile
  unit"

### Replay buffer usage

Current replay usage in `llk_pack_fast_tilize.h`:

- `[0..15]` BFP tile replay
- `[16..31]` flat-output tile replay

For BFP mode only, we can likely repurpose the flat replay region and instead
store:

- `[0..15]` BFP tile replay
- `[16..19]` address-update replay

Candidate address-update replay body:

1. `ADDDMAREG(OUTPUT_ADDR += OUTPUT_ADDR_OFFSET)`
2. `STALLWAIT(THCON)` or equivalent serialized config-consumer barrier
3. `WRCFG(OUTPUT_ADDR -> THCON_SEC0_REG1_L1_Dest_addr)`
4. `NOP`

This mirrors the existing BH untilize approach rather than inventing a new
BH-only trick.

### BFP MOP shape

Instead of:

- `outerloop=1`
- `innerloop=1`
- software loop over tiles

Use:

- `outerloop=unit_dim`
- `innerloop=1`
- replay body = one BFP tile replay
- one end op for `W` advance
- one end op for address-update replay

Conceptually:

```cpp
ckernel::ckernel_template tmp(
    unit_dim,
    1,
    lltt::replay_insn(REPLAY_BFP_TILE_OFFSET, REPLAY_BFP_TILE_LEN),
    TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));

tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011));
tmp.set_end_ops(
    TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010),
    lltt::replay_insn(REPLAY_ADDR_UPDATE_OFFSET, REPLAY_ADDR_UPDATE_LEN));
```

This sketch is intentionally conceptual. The exact choice of:

- `start_op`
- `loop_op`
- `last_inner_loop_instr`
- `last_outer_loop_instr`
- end-op ordering

must be validated against actual BFP tile closure behavior and W counter
semantics.

### Expected benefit

This should reduce or remove:

- thread-visible relaunch overhead per tile
- thread-visible stall bookkeeping per tile
- some replay setup waste in BFP mode

This should **not** remove:

- the tile-close cost itself
- the need for address progression itself

So this is a **reduction** plan, not an **elimination** plan.

## Alternative Paths Considered

### Option A: Use `Add_l1_dest_addr_offset`

There is an exposed pack config bit for `Add_l1_dest_addr_offset`.

Evidence:

- `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h:35-49`
- `tt_metal/third_party/tt_llk/tests/hw_specific/blackhole/inc/cfg_defines.h`

Why this is not the main recommendation:

- current BH LLK does not use it
- `ttsim-private` explicitly treats it as unimplemented in PACR execution
- we do not yet have proof that it behaves in the exact way needed for fast
  tilize BFP output on silicon

This is a good experimental branch, not a good mainline plan.

### Option B: Use pack auto-last generation

There are hints of auto-last-related functionality:

- `auto_set_last_pacr_intf_sel` in pack config
- pack counters that disable or imply auto-last generation

Why this is not the main recommendation:

- current LLK explicitly programs pack counters to disable auto-last generation
- there is no validated BH fast-tilize use of this mechanism in the current
  codebase
- it is unclear whether it can represent the exact "close each BFP tile at the
  correct tile boundary" behavior we need

Again: possible experiment, not the main path.

### Option C: Keep current BFP path and only micro-tune the stall

This is low-risk but low-upside.

Possible tweaks:

- test whether `PACK` alone is sufficient instead of `PACK|THCON`
- test whether a `NOP` is sufficient after the address write
- test whether `REG2FLOP` can be replaced by `WRCFG`

Why this is not enough by itself:

- it leaves the one-tile-per-run software structure intact
- it is unlikely to recover the full avoidable portion of the measured gap

## Detailed Implementation Plan

### Phase 1: Instrument and lock down current behavior

Goals:

- preserve a correctness baseline
- preserve a performance baseline
- avoid mixing optimization with latent correctness regressions

Tasks:

1. Record current BFP correctness coverage:
   - `Bfp8_b`
   - `Bfp4_b`
   - `Float16_b` input
   - `Float32` input
2. Record current steady-state perf numbers for:
   - `rt in {1,2,4,8}`
   - `ct in {2,4,8}`
3. Add focused comments in the fast-tilize pack file that separate:
   - fundamental BFP close requirement
   - current implementation-specific stall pattern

No functional change should happen in this phase.

### Phase 2: Split BFP and flat replay loading more aggressively

Goals:

- avoid loading unused replay sequences in BFP mode
- reserve replay slots for address-update replay

Tasks:

1. Refactor replay loading in
   `tt_llk_blackhole/llk_lib/llk_pack_fast_tilize.h`
2. In BFP mode:
   - load BFP tile replay
   - load address-update replay
3. In flat mode:
   - keep the existing flat replay

Expected impact:

- small perf gain
- cleaner structure for the larger BFP MOP change

### Phase 3: Convert BFP path from software-tiled to MOP-tiled

This is the main optimization phase.

Goals:

- replace the software loop over `run(); stallwait; run(); stallwait`
- move tile-to-tile progression into one BFP-specific MOP description

Tasks:

1. Introduce a dedicated BFP MOP config helper, separate from the current
   `_llk_pack_fast_tilize_mop_config_bfp_()`.
2. Preserve tile-local `Last=1` semantics in the replay.
3. Add an address-update replay as one of the MOP end operations.
4. Ensure `W` progression and `OUTPUT_ADDR` progression stay aligned.
5. Remove the explicit per-tile software loop in
   `_llk_pack_fast_tilize_block_()` for BFP mode.

Expected impact:

- medium to large fraction of the avoidable BFP overhead

### Phase 4: Evaluate `WRCFG`-based update vs `REG2FLOP`-based update

This phase should be driven by measured results, not assumption.

Goals:

- choose the cheaper safe per-tile address update mechanism

Candidates:

- Current: `ADDDMAREG + REG2FLOP`
- Candidate: `ADDDMAREG + WRCFG + NOP`

Why this matters:

- BH untilize already uses the `WRCFG` replay model
- `WRCFG` has explicit documented scheduling requirements
- it may be more natural for replay/end-op usage than `REG2FLOP`

### Phase 5: Only then experiment with hardware auto features

Only after Phases 2-4 are done and measured should we try:

- `Add_l1_dest_addr_offset`
- pack auto-last generation
- shared-exp-assembler-related toggles

Criteria for doing this:

- the primary replay/MOP path does not recover enough performance
- silicon-only experiments are available
- correctness checkers can validate exponent/mantissa layout exactly

## Validation Plan

### Functional validation

Required matrix:

- `Float16_b -> Bfp8_b`
- `Float16_b -> Bfp4_b`
- `Float32 -> Bfp8_b`
- `Float32 -> Bfp4_b`
- `ct = 2, 4, 8`
- `rt = 1, 2, 4, 8`

Required checks:

- exact output match vs existing golden
- no tile-boundary corruption
- no exponent-section drift
- no position mismatch at 16-value BFP block granularity

### Performance validation

Measure against the current report methodology:

- same profiler zone
- same loop factor
- same matrix as `perf_fast_tilize_full.py`

Key comparisons:

- old BFP fast path vs new BFP fast path
- new BFP fast path vs flat fp16 fast path
- pack-isolate zone before and after

### Silicon vs ttsim validation

Important limitation:

- `ttsim-private` currently marks some potentially relevant packer features as
  unimplemented, especially `Add_l1_dest_addr_offset`.

Therefore:

- mainline design should stay within behaviors already exercised by BH LLK
- silicon is the source of truth for final performance conclusions

### MOP/STALLWAIT validation

Because of the MOP/Wait-Gate subtlety described above, explicitly validate:

1. whether the current BFP `STALLWAIT` is actually gating MOP launch exactly as
   intended
2. whether the proposed single-MOP design preserves correct tile sequencing
3. whether any observed gain comes from real issue reduction rather than lucky
   timing

## Expected Outcome

Best realistic outcome:

- recover a meaningful portion of the current `~10 cyc/tile` BFP penalty
- keep correctness fully intact
- preserve the simple flat-output fast path unchanged

Realistic steady-state target:

- Better than `~36-37 cyc/tile`
- Probably **not** as low as `~26 cyc/tile`

The exact target should be driven by measurement, but a plausible success band
would be to reclaim several cycles per tile rather than just one.

## Risks

### Risk 1: Tile boundary correctness regressions

BFP failures can look deceptively "almost right":

- correct values in wrong places
- correct mantissas with wrong shared exponents
- correct first tile and corrupted later tiles

This is the primary correctness risk.

### Risk 2: MOP ordering assumptions are wrong

If the current software-side `STALLWAIT` is not gating MOP launch in the exact
way implied by comments, then simply moving logic into a single MOP may expose
ordering assumptions that were previously masked.

### Risk 3: Address update remains dominant

Even after removing software per-tile launch overhead, the per-tile address
update may remain expensive enough that the net gain is modest.

### Risk 4: Feature exploration burns time with low probability of payoff

Underdocumented features like `Add_l1_dest_addr_offset` may be tempting, but
they can consume substantial time without yielding a production-safe solution.

## Recommended Order of Work

1. Preserve current baseline numbers and correctness.
2. Refactor replay loading so BFP mode has room for an address-update replay.
3. Implement the single-MOP BFP design.
4. Compare `REG2FLOP`-based and `WRCFG`-based address update variants.
5. Only if needed, explore auto-address / auto-last hardware features.

## Success Criteria

This plan is successful if all of the following are true:

- BFP output remains bitwise correct against current golden output
- no new silicon hangs appear
- steady-state BFP fast-tilize performance improves materially
- flat `Float16_b -> Float16_b` performance remains unchanged
- the final implementation does not rely on unvalidated hardware features

## Final Recommendation

Proceed with a **BFP-specific replay/MOP restructuring** as the main path.

Do **not** frame the work as "remove BFP overhead entirely".

Frame it as:

- preserve per-tile BFP closure
- eliminate avoidable software-visible per-tile control overhead
- reuse proven BH replay/update patterns
- validate aggressively on silicon

That is the highest-confidence route to recovering performance without turning
the fast-tilize BFP path into a fragile hardware experiment.
