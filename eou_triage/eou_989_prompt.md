# Issue tt-llk#989 — Fix `pack_reads_per_xy_plane` Default and Reduce Handling

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-llk/issues/989
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`pack_reads_per_xy_plane` is a hardware packer counter field (8 bits at offset `[8:15]` in the
`PACK_COUNTERS_SEC*` registers) that controls when the Y-position counter resets to zero for
edge-mask purposes. In Blackhole and Wormhole B0 the struct is:

```cpp
typedef struct
{
    std::uint32_t pack_per_xy_plane        : 8;
    std::uint32_t pack_reads_per_xy_plane  : 8;  // <-- this field
    std::uint32_t pack_xys_per_til         : 7;
    std::uint32_t pack_yz_transposed       : 1;
    std::uint32_t pack_per_xy_plane_offset : 8;
} pack_counters_t;
```

**The bug:** `configure_pack()` (and `reconfig_packer_data_format()`) initialise
`pack_reads_per_xy_plane` to `face_r_dim` (= 16). That is wrong. The **correct default is 1**,
because when the field equals 1, the Y-position counter resets on every read, making all other
packer operations agnostic to this setting.

**Only reduce operations** (row/col/scalar) depend on a non-1 value — they need
`pack_reads_per_xy_plane = FACE_R_DIM` (16) to read a full face worth of rows before resetting.
The right ownership model is:

- `_llk_pack_reduce_mask_config_` **sets** `pack_reads_per_xy_plane` to `FACE_R_DIM`
- `_llk_pack_reduce_mask_clear_` **resets** it to `1`

A past bug ([tt-metal#17132](https://github.com/tenstorrent/tt-metal/issues/17132)) caused by this
mismatch was patched at the wrong layer (compute API, in tt-metal PR#17486). This issue moves the
fix to where it belongs: the LLK library.

---

## 2. Architectural layers involved

```
┌──────────────────────────────────────────────────────────────────┐
│ Compute kernel / compute API layer                               │
│   tt_metal/hw/inc/api/compute/                                   │  ← no changes here
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│ LLK lib (per arch) — root cause                                  │
│   tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/               │
│   common/inc/cpack_common.h     ← configure_pack / reconfig      │
│   llk_lib/llk_pack_common.h     ← reduce_mask_config / clear     │
└──────────────────────────────────────────────────────────────────┘
```

**Quasar is out of scope.** It has a different packer architecture and does not use
`pack_reads_per_xy_plane` in the same register layout. Do not touch Quasar files.

---

## 3. Complete scope — every change needed

Run these greps first to confirm nothing has shifted:

```bash
# All places that currently set pack_reads_per_xy_plane to face_r_dim (not 1)
grep -rn "pack_reads_per_xy_plane" \
    tt_metal/tt-llk/tt_llk_blackhole/ \
    tt_metal/tt-llk/tt_llk_wormhole_b0/

# Confirm _llk_pack_reduce_mask_config_ and _llk_pack_reduce_mask_clear_ are in the right files
grep -rn "_llk_pack_reduce_mask_" \
    tt_metal/tt-llk/tt_llk_blackhole/ \
    tt_metal/tt-llk/tt_llk_wormhole_b0/
```

### 3.1 `configure_pack()` — change default from `face_r_dim` to `1`

Both architectures initialise the counter register to `face_r_dim` inside `configure_pack()`.
Change the single field assignment in each file:

| File | Location | Change |
|---|---|---|
| `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h` | line ~617, inside `configure_pack()` | `pack_reads_per_xy_plane = face_r_dim` → `pack_reads_per_xy_plane = 1` |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cpack_common.h` | line ~729, inside `configure_pack()` | same |

Update the comment on the same line from "Number of reads per face / Used for resetting tile
position generator for edge masks" to something like "Default 1 — makes non-reduce operations
agnostic to this counter; reduce sets it via _llk_pack_reduce_mask_config_".

### 3.2 `reconfig_packer_data_format()` — same default change

Both architectures also set `pack_reads_per_xy_plane = face_r_dim` inside the reconfig function.
Apply the same `→ 1` change:

| File | Location | Change |
|---|---|---|
| `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h` | line ~513, inside `reconfig_packer_data_format()` | `pack_reads_per_xy_plane = face_r_dim` → `pack_reads_per_xy_plane = 1` |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cpack_common.h` | line ~594, inside `reconfig_packer_data_format()` | same |

Update the adjacent comment ("Some initialization methods modify this configuration register…")
accordingly.

### 3.3 `_llk_pack_reduce_mask_config_()` — add set to `FACE_R_DIM`

Currently these functions configure the packer edge-offset masks but **never touch**
`pack_reads_per_xy_plane`. After the `TTI_STALLWAIT` that already exists in both functions, add
the RMW call to set `pack_reads_per_xy_plane = FACE_R_DIM`.

**Blackhole** (1 packer — `SEC0` only):

```cpp
// In tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h
// After the existing TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK) line, before TTI_WRCFG:

cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(FACE_R_DIM);

// Then the existing TTI_WRCFG lines follow as-is
```

**Wormhole B0** (4 packers — `SEC0`–`SEC3`):

```cpp
// In tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_common.h
// Same position (after TTI_STALLWAIT, before TTI_WRCFG):

cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(FACE_R_DIM);
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC1_pack_reads_per_xy_plane_RMW>(FACE_R_DIM);
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_RMW>(FACE_R_DIM);
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC3_pack_reads_per_xy_plane_RMW>(FACE_R_DIM);
```

The existing pattern in `_llk_pack_rows_init_` (in `llk_pack_rows.h`) uses `cfg_reg_rmw_tensix`
in exactly this way — follow that precedent.

### 3.4 `_llk_pack_reduce_mask_clear_()` — add reset to `1`

Symmetrically, after the `TTI_STALLWAIT` in the clear function, add the RMW to reset
`pack_reads_per_xy_plane` back to `1`:

**Blackhole:**

```cpp
// After TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK), before TTI_WRCFG:
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);
```

**Wormhole B0:**

```cpp
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC1_pack_reads_per_xy_plane_RMW>(1);
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC2_pack_reads_per_xy_plane_RMW>(1);
cfg_reg_rmw_tensix<PACK_COUNTERS_SEC3_pack_reads_per_xy_plane_RMW>(1);
```

### 3.5 Validation function — verify but do not blindly change

`cpack_common.h` (both BH and WH) contains `are_packers_configured_correctly()`. Its
`ProgramByFace` branch checks `counters.pack_reads_per_xy_plane == face_r_dim`.

This function is only called in `ProgramByFace` mode (never in `ProgramByTile` mode, where the
check is skipped with `true`). After the fix the check remains semantically correct: it is called
**after** `_llk_pack_reduce_mask_config_` sets the field to `FACE_R_DIM`, so the assertion should
still pass. **Do not change this function** unless you find a call path where it is invoked
before the reduce config.

### 3.6 Out of scope

- **`tt_metal/tt-llk/tt_llk_quasar/`** — Quasar has no `pack_reads_per_xy_plane` register in
  the same layout. Do not touch.
- **Compute API layer workaround** — [tt-metal PR#17486](https://github.com/tenstorrent/tt-metal/pull/17486)
  added a workaround at the compute API layer for [tt-metal#17132](https://github.com/tenstorrent/tt-metal/issues/17132).
  Once this LLK-layer fix is in, that workaround may be removable, but that cleanup is tracked
  separately under [tt-metal#17641](https://github.com/tenstorrent/tt-metal/issues/17641). Do not
  touch compute API files in this PR.
- **`_llk_pack_rows_init_`** — Already sets `pack_reads_per_xy_plane = 1` correctly. No change
  needed.

---

## 4. Branch and worktree setup

`tt_metal/tt-llk/` is a plain directory in the `tt-metal` repo — no submodule.

```bash
cd /localdev/ncvetkovic/work/tt-metal

git fetch origin main
git checkout -b ncvetkovic/989-pack-reads-per-xy-plane FETCH_HEAD

# Verify: should be 0 commits ahead of origin/main before you start
git log --oneline origin/main..HEAD   # expected: empty
```

If you need an isolated worktree:

```bash
git worktree add ../tt-metal-989 ncvetkovic/989-pack-reads-per-xy-plane
```

Never base on another in-flight feature branch.

---

## 5. Suggested execution order

### Phase A — Change the default in `configure_pack` and `reconfig_packer_data_format`

1. Edit BH `cpack_common.h` line ~617 and ~513.
2. Edit WH `cpack_common.h` line ~729 and ~594.
3. Confirm grep returns zero `pack_reads_per_xy_plane = face_r_dim` hits (excluding comments):
   ```bash
   grep -rn "pack_reads_per_xy_plane = face_r_dim" \
       tt_metal/tt-llk/tt_llk_blackhole/ tt_metal/tt-llk/tt_llk_wormhole_b0/
   # Expected: no hits
   ```

### Phase B — Add `pack_reads_per_xy_plane` management to the reduce functions

4. Edit BH `llk_pack_common.h`: add `cfg_reg_rmw_tensix` call in `_llk_pack_reduce_mask_config_` and `_llk_pack_reduce_mask_clear_`.
5. Edit WH `llk_pack_common.h`: same (4 packers).
6. Confirm the four functions each contain a `pack_reads_per_xy_plane` RMW:
   ```bash
   grep -A2 -B2 "pack_reads_per_xy_plane_RMW" \
       tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h \
       tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_common.h
   ```

### Phase C — Build and test

```bash
# Build (run for both arch targets)
./build_metal.sh --build-tests

# LLK standalone tests
cd tt_metal/tt-llk/tests
pytest --compile-producer -n 8 -x python_tests/
pytest --compile-consumer -x python_tests/
cd -
```

If a test hang occurs: `pkill -9 -f pytest && tt-smi -r`

### Phase D — Invariant check

```bash
# Must show no remaining wrong defaults:
grep -rn "pack_reads_per_xy_plane = face_r_dim" \
    tt_metal/tt-llk/tt_llk_blackhole/ tt_metal/tt-llk/tt_llk_wormhole_b0/
# Expected: empty

# Must show the RMW calls in all four reduce functions:
grep -n "pack_reads_per_xy_plane_RMW" \
    tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h \
    tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_common.h
# Expected: 2 hits in BH (config + clear), 8 hits in WH (4 packers × 2 functions)
```

---

## 6. Commit message

```
fix(llk): fix pack_reads_per_xy_plane default and reduce ownership

pack_reads_per_xy_plane was initialised to face_r_dim (16) in configure_pack
and reconfig_packer_data_format. The correct default is 1, which makes all
non-reduce packer operations agnostic to this counter.

Only reduce operations depend on this setting. Transfer ownership to the reduce
layer: _llk_pack_reduce_mask_config_ now sets pack_reads_per_xy_plane to
FACE_R_DIM, and _llk_pack_reduce_mask_clear_ resets it to 1.

Applies to Blackhole (1 packer) and Wormhole B0 (4 packers). Quasar is
unaffected (different packer architecture).

Resolves tt-llk#989
```

---

## 7. PR creation

```bash
cd /localdev/ncvetkovic/work/tt-metal
git push origin ncvetkovic/989-pack-reads-per-xy-plane

gh pr create \
  --base main \
  --head ncvetkovic/989-pack-reads-per-xy-plane \
  --title "fix(llk): fix pack_reads_per_xy_plane default and reduce ownership" \
  --body "$(cat <<'PRBODY'
### Summary

`pack_reads_per_xy_plane` is a hardware packer counter that controls when the
Y-position counter resets for edge-mask purposes. It was incorrectly initialised
to `face_r_dim` (16) in `configure_pack()` and `reconfig_packer_data_format()`
in both Blackhole and Wormhole B0.

The correct default is **1** — when set to 1, the counter resets on every read,
making all non-reduce packer operations agnostic to this setting. Only reduce
operations (row/col/scalar) require a non-1 value.

This PR:
- Changes `configure_pack()` and `reconfig_packer_data_format()` to initialise
  `pack_reads_per_xy_plane = 1` in both BH and WH B0.
- Adds `cfg_reg_rmw_tensix<PACK_COUNTERS_SEC*_pack_reads_per_xy_plane_RMW>(FACE_R_DIM)`
  at the start of `_llk_pack_reduce_mask_config_()` (after the existing STALL) to
  set the field when entering reduce mode.
- Adds the symmetric reset to 1 in `_llk_pack_reduce_mask_clear_()`.

Root cause of past bug: [tt-metal#17132](https://github.com/tenstorrent/tt-metal/issues/17132)
was patched at the compute API layer (PR#17486) rather than at the LLK layer where
the incorrect default lives. This PR is the correct fix.

Resolves [tt-llk#989](https://github.com/tenstorrent/tt-llk/issues/989)

### What's changed

**Blackhole** (`tt_metal/tt-llk/tt_llk_blackhole/`)
- `common/inc/cpack_common.h` — `configure_pack()` and `reconfig_packer_data_format()`: `pack_reads_per_xy_plane = face_r_dim` → `= 1`
- `llk_lib/llk_pack_common.h` — `_llk_pack_reduce_mask_config_()`: add `cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(FACE_R_DIM)` after STALLWAIT
- `llk_lib/llk_pack_common.h` — `_llk_pack_reduce_mask_clear_()`: add `cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1)` after STALLWAIT

**Wormhole B0** (`tt_metal/tt-llk/tt_llk_wormhole_b0/`) — identical changes; 4 packers (SEC0–SEC3) updated in the reduce functions instead of 1.

### What's intentionally unchanged

- **Quasar** — Different packer architecture; does not use `pack_reads_per_xy_plane` in this register layout.
- **`_llk_pack_rows_init_`** — Already sets `pack_reads_per_xy_plane = 1` correctly; no change needed.
- **Compute API layer** — The existing workaround in tt-metal PR#17486 is left in place; its removal is tracked separately under [tt-metal#17641](https://github.com/tenstorrent/tt-metal/issues/17641).
- **`are_packers_configured_correctly()`** — The `ProgramByFace` check (`pack_reads_per_xy_plane == face_r_dim`) remains correct because it is only called after `_llk_pack_reduce_mask_config_` has set the field to `FACE_R_DIM`.

### Invariant after this change

```bash
# No wrong defaults remain:
grep -rn "pack_reads_per_xy_plane = face_r_dim" \
    tt_metal/tt-llk/tt_llk_blackhole/ tt_metal/tt-llk/tt_llk_wormhole_b0/
# Expected: empty

# Reduce functions own the field:
grep -n "pack_reads_per_xy_plane_RMW" \
    tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h \
    tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack_common.h
# Expected: 2 hits in BH, 8 hits in WH
```

### Type of change

- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring

### Checklist

- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)
PRBODY
)"
```

---

## 8. Definition of Done

- [ ] `grep -rn "pack_reads_per_xy_plane = face_r_dim" tt_metal/tt-llk/tt_llk_blackhole/ tt_metal/tt-llk/tt_llk_wormhole_b0/` returns zero hits.
- [ ] BH `llk_pack_common.h`: `_llk_pack_reduce_mask_config_` contains `PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW` with value `FACE_R_DIM`; `_llk_pack_reduce_mask_clear_` contains same macro with value `1`.
- [ ] WH `llk_pack_common.h`: same for SEC0–SEC3 (8 total RMW calls across the two functions).
- [ ] No Quasar files modified.
- [ ] `./build_metal.sh --build-tests` succeeds for both Blackhole and Wormhole B0 targets.
- [ ] LLK standalone test suite passes (produce + consume steps).
- [ ] PR open with description matching section 7 (no placeholder text remaining).

---

## 9. Common pitfalls

1. **Mixing up which packer count to use.** Blackhole has `NUM_PACKERS = 1` — only `SEC0`. Wormhole B0 has `NUM_PACKERS = 4` — must update `SEC0`, `SEC1`, `SEC2`, `SEC3`. Grep `_llk_pack_rows_init_` in both architectures for the correct pattern to copy.

2. **Placing the RMW in the wrong order.** The `TTI_STALLWAIT` in the reduce functions already serialises access to the packer config registers. Place the `cfg_reg_rmw_tensix` call **after** `TTI_STALLWAIT` and **before** the `TTI_WRCFG` instructions that configure the edge-offset masks, so all config writes happen in one serialised block.

3. **Changing `reconfig_packer_data_format` but forgetting `configure_pack`, or vice versa.** Both functions write the same counter field. Both must change. Use the invariant grep from section 5 Phase D to confirm zero `= face_r_dim` hits remain.

4. **Touching Quasar.** Quasar has no `PACK_COUNTERS_SEC*_pack_reads_per_xy_plane_RMW` macros. Any attempt to add these calls to Quasar files will fail to compile.

5. **Removing the compute API workaround in PR#17486.** That's a separate issue (tt-metal#17641). Keep the two changes decoupled so this PR can be reviewed and landed independently.

6. **Stacking on the wrong base.** Branch from `FETCH_HEAD` of `origin/main`, not from any local branch that may carry unrelated eou_triage or ai-agents commits.
