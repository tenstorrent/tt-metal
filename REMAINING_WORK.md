# tt-llk blaze promotions — everything still to do

Single actionable index of what is left, with a plan per item. **Everything in this document is
outstanding**: closed items are removed rather than struck through, and what is done — results,
sweeps, and the findings that came out of building them — lives in
[`BLAZE_PROMOTION_TESTS_DONE.md`](BLAZE_PROMOTION_TESTS_DONE.md). Item letters are therefore not
contiguous; the gaps are closed work, and the DONE document is where they went.

- **Detail for every defect below** — mechanism, what is measured versus inferred, blast radius,
  reproduction, fix options with trade-offs, and the tripwire that flips when it is fixed — is in
  [`blaze_llk_promotion_test_strategy.md`](blaze_llk_promotion_test_strategy.md) **§13**, one
  dossier per item (C1–C6, F). Read the dossier before starting on an item; two of them record
  fixes that were tried and did not work.
- **Forensic detail** for A5 also sits in that document at §9, and §3 covers the
  `mul_reduce_scalar` dead ends behind C4. Bare "§n" below means that document.
- All three promotion PRs (**#52709, #52713, #52727**) have merged, so nothing below is blocked
  on a promotion landing.

Where an item says "uncalled", that means no driver under `tt_metal/tt-llk/tests/sources/`
references the symbol. State verified against the tree on 2026-08-20.

---

## At a glance

| # | Item | Type | Size | Blocked on |
|---|------|------|------|-----------|
| **A5** | `eltwise_mul_scalar` HiFi init — untested, rationale disproved | test | unknown | C2 |
| **B1** | `custom_mm` vs `compressed_custom_mm` divergence guard | test, **outside tt-llk** | ~1 d | needs an owner (interim static guard landed) |
| **C1** | `dense_packing` W-stride not format-aware | **defect** | ~0.5 d once decided | owner decision |
| **C2** | `eltwise_mul_scalar` HiFi workaround rationale does not hold | **question** | — | #52709 author |
| **C3** | `topk_xl` → `eltwise_binary` reconfig escape | **defect**, pre-existing | unknown | needs an owner |
| **C4** | `mul_reduce_scalar` re-entry needs a DEST-section boundary | **defect** | unknown | needs an owner |
| **C5** | OOB metadata read shipped to main with #52727 | **defect** | minutes | branch **pushed**, needs a PR |
| **C6** | `top32_rm` 32-bit unpack: partial chunk sorts against stale Dest | **defect** | ~0.5 d | needs an owner (xfail pins it) |
| **D1** | `mul_reduce_scalar_chunked_tile` ships with no caller | cleanup | — | C4 decides it |
| **E** | PR mechanics — title and body | chore | minutes | — |
| **F** | `test_matmul_custom_compressed` intermittent — host/BRISC desync | **defect**, nightly-only | unknown | needs an owner |

**Nothing left is tt-llk test work.** The two remaining test items cannot be done in this repo
as it stands: A5 needs C2's answer first (and may end in deleting the code rather than testing
it), and B1 needs CB metadata the tt-llk harness does not have — verified, see B1. Six are
product decisions or defects needing an owner (C1–C6), three of them defects in shipping ops with
a reproducer already attached (C1, C4, C6); C5's fix is written, pushed and verified and needs
only a PR. The rest are one cleanup (D1), the PR's own metadata (E), and one intermittent-failure
investigation (F).

---

## A. Functional test gaps

### A5. `eltwise_mul_scalar` HiFi init — untested, and its rationale does not hold

**Unchanged, still blocked on C2.** Smaller than it looks: the underlying shapes **are** covered
generically — `test_eltwise_binary.py` sweeps `DEST_TO_SRCA`/`DEST_TO_SRCB` dest-reuse and
`BroadcastType.Scalar`. What has no test is the **HiFi init sequence** specifically.

Resolve C2 first. If the workaround's real mechanism is something else the test to write
changes completely, and if it is inert the honest outcome may be deleting it rather than
testing it. Read §9 for the earlier attempt, which hung the device as first written.

---

## B. Gap that cannot live in tt-llk

### B1. Nothing can catch `custom_mm` vs `compressed_custom_mm` divergence

**Unchanged.** `custom_mm_uninit_restore_test.cpp` **replicates** the uninit body rather than
calling `custom_mm_block_uninit` / `compressed_custom_mm_block_uninit`, because a tt-llk test
cannot include `tt_metal/hw/inc/api/compute`. The two bodies are currently identical; if they
diverge, every existing test keeps passing. Copilot raised the same point independently on
#53130.

**Plan.** A compute-kernel test under `tests/tt_metal/` that calls the real entry points. ~1 d,
but it needs an owner who works in that tree.

**Why it cannot be done from tt-llk — checked on 2026-08-20, not assumed.** The include path is
*not* the obstacle: the tt-llk test build has the Metal `llk_api` directory on it, which is how
`top32_rm_test.cpp` now drives the `llk_math_deepseek_top32_rm_*` wrappers and how
`deepseek_moe_gate_test.cpp` includes `llk_sfpu/` headers. The obstacle is one layer up: the
Compute API resolves its operands through CB metadata (`get_operand_id`,
`get_operand_dst_format`, …) that the JIT build emits from the circular-buffer config, and the
tt-llk harness has no CB emulation at all — nothing under `tests/helpers/include/` provides
those symbols. That is why `mul_reduce_scalar_test.cpp` *names* the Compute API in a comment and
then expands its sequence at the LLK level instead of including it. Anyone tempted to retry this
in tt-llk needs to add CB metadata to the harness first, which is a bigger job than the test.

**What exists already, and why it does not close this.** `test_custom_mm_uninit_parity.py` is a
device-free static gate that fails loudly if the two uninit bodies diverge or if the driver's
replicated W-stride expressions drift from the headers (detail in the DONE document). A text
match cannot say the functions *work*, only that they still say the same thing, so the
metal-side test calling the real entry points is still the item — the guard only removes the
silent-failure mode.

---

## C. Product issues needing a decision, not a test

### C1. `dense_packing` W-stride is not format-aware — **defect**

`set_packer_strides` (`cpack_common.h:301-305`) derives the field as
`TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(pack_src_format)`, while
`custom_mm.h:69` / `:261` and `compressed_custom_mm.h:69` / `:262` hardcode `* 2`. On a Float32
pack source both halves are 2x off: init programs 1024 where 2048 is correct, and the uninit
restores 2048 where 4096 is correct. Measured at 0.25 match.

**Owner:** whoever owns `custom_mm.h`. **Decision needed**, then ~half a day:

- **Option 1 — guard.** `LLK_ASSERT` in `*_block_init` that
  `datum_size_in_bytes(pack_src_format[out_cb]) == 2` when `dense_packing` is set. No API
  change; turns silent corruption into a loud failure; leaves 32-bit unsupported.
- **Option 2 — full fix.** Derive the datum size in init from `out_cb_id` and **add an
  `out_cb_id` parameter to `*_block_uninit`**, which currently takes none. Correct on 32-bit,
  but changes a signature that `matmul.hpp`, `flash_mla.hpp`, `dram_streaming_matmul*.hpp` and
  `matmul_custom_compressed_kernel.cpp` all call.

The `xfail` in `test_custom_mm_uninit_restore.py` flips to XPASS when either lands. It is a real
detector, not a label: it uses the marker form, so the body builds and runs and the comparison
actually happens under XFAIL.

### C2. The `eltwise_mul_scalar` HiFi workaround's mechanism does not survive review

**Unchanged.** `deepseek_binary_dest_reuse_tiles_init`'s HiFi branch hardcodes
`DEFAULT_TENSOR_SHAPE` and attributes a HiFi4 fix to the shorthand init "mis-specialising the
tile shape". But `get_effective_math_fidelity<ELWMUL, f>()` is the identity for ELWMUL,
`acc_to_dest` is 0 in both arms, and the shorthand resolves the shape from the CB regardless of
fidelity — so tensor shape is the only difference, and on a standard 4-face CB
`get_operand_tensor_shape` returns exactly `DEFAULT_TENSOR_SHAPE`, making the HiFi arm
bit-identical to the shorthand it replaces. Meanwhile the paired execute *does* derive the shape
from the CB, so on non-default geometry init and execute disagree.

Combined with the measured fact that forcing `DEFAULT_TENSOR_SHAPE` on a 2-face tile
**deadlocks the MATH_PACK handshake**, the workaround is either inert (4-face CB) or hangs
(2-face CB). The cited failing config is `gated_local_reduce` at HiFi4 (0.70 → 0.9996), and that
measurement is not explained by the stated mechanism.

**Owner:** the #52709 author. **Needed:** either the real mechanism, or a corrected comment. A5
is blocked on this.

### C3. Pre-existing `topk_xl` → `eltwise_binary` reconfig escape — **defect**

**Unchanged.** Reproduces on clean `main` with every promotion change stashed, so it is
unrelated to the promotions. Recorded because A2 shares the sort headers, so whoever picks that
up will see a failure in this area and assume it is theirs. **Bisect single-file-then-target
before blaming your own driver.** `tt-smi -r` must **not** be used to paper over it. Needs an
owner. See also F, which may or may not be the same thing.

### C4. `mul_reduce_scalar` re-entry needs a DEST-section boundary — **defect**

Located by the `mul_reduce_scalar_chunked_tile` investigation. The reproducer is in the tree —
`tests/sources/mul_reduce_scalar_reenter_test.cpp` +
`tests/python_tests/test_mul_reduce_scalar_reenter.py`, a ~40-line driver that runs the
known-good non-chunked sequence twice over the same input. On BH p100a:

| Configuration | Result |
|---|---|
| `passes=1`, either mode (control) | correct |
| `passes=2`, DEST-section boundary between passes | correct, and **bit-identical** across passes |
| `passes=2`, one shared DEST section | **wrong — all 12 variants, 9.27x to 9.93x golden** |

So the family **is** re-enterable. What is broken is re-entry with no
`dest_section_done` / `wait_for_dest_available` pair in between: that handshake restores whatever
the second `_llk_math_mul_reduce_scalar_init_` does not.

**And that is exactly how the chunked op is built.** `mul_reduce_scalar_chunked_tile`
(`rmsnorm.h:105`) documents that the caller "must ... acquire DST before calling", then
re-enters every batch inside that one section, with `if (batch > 0) mul_reduce_scalar_init(...)`
as its only restoration attempt. The reverted chunked driver reported 5-30x golden and "not a
clean multiple of anything"; this reproduces 9.3-9.9x, also non-integer. Same signature, so
very likely the same defect — now with a minimal reproducer instead of a full chunked
implementation.

**Owner needed.** Two shapes the fix could take, and it is not this document's call which:

- **In the LLK** — make `_llk_math_mul_reduce_scalar_init_` (or `switch_to_reduce`) restore
  whatever the section boundary restores. Right if re-entry inside a section is meant to work.
- **In the compute API** — have `mul_reduce_scalar_chunked_tile` close and reacquire the DEST
  section per batch, or document that it cannot be used as written. Right if the per-batch
  handshake is considered the caller's job.

**Whoever takes it:** the reproducer is `--test test_mul_reduce_scalar_reenter.py`, and the
`single_dest_section` axis is the whole experiment. The 12 failing variants are xfail (marker
form, so the body runs), and flip to XPASS the moment re-entry inside one section restores
state. Do **not** re-investigate the accumulator fill or a missing UNPACK/MATH barrier —
§3 records both as tried on silicon and disproved, and this result explains why neither moved
the number.

### C5. The out-of-bounds metadata read shipped to main — **defect**

`#52727` merged **without** the fix for the out-of-bounds remainder read Copilot found on
#53130. Verified on 2026-08-19: `grep 'rem_iters != 0'` on main's
`llk_unpack_AB_compressed_custom_mm.h` returns nothing, so the unguarded
`meta_ptr[full_iters]` is live on main.

The guard exists only on this branch (`54e218ebbce`). Reachable inside the documented ranges
whenever `kt_dim * ct_dim` is a multiple of 10 — `kt_dim=10, ct_dim=1` is the smallest case.

**What it costs, stated precisely so nobody over- or under-reacts.** At `rem_iters == 0` the
remainder loop never runs, so the word read past the buffer is *never used* and no golden can
see it — confirmed by running the boundary test against the unguarded kernel, where it passes.
It is a memory-safety defect, not a wrong-answer defect: an L1 read of whatever follows the
metadata buffer.

**The branch is pushed; what is left is opening the PR.**
`ldjurovic/compressed-mm-oob-guard`, cut from `main`, carries both guards cherry-picked (unpack
side and math side) and nothing else: 2 files, +30/-12, no compute-API change. Verified on BH
p100a from that worktree — `test_matmul_custom_compressed.py` **582 passed** (main's count; the
6 metadata-word-boundary variants that reach the guarded case live on #53130's branch, so run
those there).

The guard will otherwise ride in on #53130 whenever it merges, which is fine but leaves main
carrying the defect in the meantime.

---

### C6. A partially-filled chunk on `top32_rm`'s 32-bit unpack branch sorts against stale Dest — **defect**

`llk_unpack_A_top32_rm_api.h` forks on the src format. The 16-bit branch has the unpacker clear
SrcA to -infinity (`TTI_UNPACR_NOP ... CLR_SRC_NEGINF`) before unpacking, so a chunk that fills
fewer than 4 faces leaves the rest at -infinity and they lose every comparison. **The 32-bit
(unpack-to-dest) branch clears nothing**, and the `ZEROACC` loop inside `_llk_math_top32_rm_`
covers only `num_faces` faces, so the untouched part of the tile keeps whatever Dest held.

Measured on BH p100a: a 160-element row (two full 64-element chunks plus a 32-element one) of
Float32 values in [-80, 79] returned a top-32 containing **11026.0, 10041.0, 9058.0** and more —
values that are not in the input at all, evidently left in Dest by an earlier kernel.

**Latent in the consumer rather than harmless**, which is the reason nobody has hit it:
`top32_rm_dev_compute.cpp` does call this branch with `num_faces=2`, but only for its **uint32
index** tile, and an index slot can only be selected if the paired value slot wins — the value
tile is bf16, so its padding is -infinity and never wins. The defect is invisible until someone
puts *values* through the 32-bit branch, which the family's doc tables permit.

**Pinned, not just described:** `test_top32_rm_32bit_partial_chunk` is a non-strict `xfail`, so
it flips to XPASS the moment the branch starts clearing its tile.

**It is also the last thing standing between the `top32_rm` sweep and complete coverage of the
family.** Everything else is done — both modes, both widths, the mixed 1024+tail shape, and the
7 Metal wrappers — and the one shape left out is the Metal dev test's own `row=3232`, which ends
in a 32-element chunk. Fix this and that shape becomes a one-line addition to
`PRE_SORTED_ROW_ELEMENTS`.

**The obvious fix was tried on 2026-08-20 and does not work — read §13.6 before repeating it.**
Extending that `ZEROACC` loop to all 4 faces passes in isolation (`--k partial_chunk` XPASSes)
and **still fails inside a full-suite run**, twice in a row, so a Dest clear from the math thread
does not remove what the sort is reading. It also forces `num_faces` out of the math half and out
of 6 call sites, for no gain. Reverted.

**Fix options now**, in the order §13.6 recommends: give the 32-bit branch the 16-bit branch's
-infinity semantics via SrcA (the only option that makes the two branches agree, but it needs
both TRISCs to know the face count); or reject `num_faces < 4` on that branch at compile time and
document partial chunks as 16-bit only; or, caller-side and available today with no LLK change,
pad the chunk to a full 64 elements with -infinity in L1 and always pass `num_faces=4`.

---

## D. Review comment resolved but not fixed

- **D1 — `mul_reduce_scalar_chunked_tile` ships with no caller and no test.** C4 says the op is
  broken as written, not merely untested. Removal is a legitimate outcome, and is the cheaper one
  unless someone wants the chunked form to work.

---

## E. PR mechanics

- **The title still reads `[do not review]`.** Re-checked 2026-08-20 against the API:
  `[do not review][LLK] Add tests for newly promoted kernels from blaze`. This is the single
  thing blocking anyone from looking at the PR.
- **The body is still the untouched template**, with an empty Summary and PR Category — Copilot
  raised it too, and it is now inaccurate as well as empty, since the diff is no longer just
  tests. A drafted title, Summary and Notes-for-reviewers sit in
  [`pr-53130-replies.md`](pr-53130-replies.md).
- **21 review threads have no reply.** Drafts for all of them are in that same file, grouped by
  fixed / already-fixed / deliberate-won't-fix. Nothing has been posted: the sessions that wrote
  them had no GitHub write access (`gh` absent, outbound `curl` blocked).
- **`ldjurovic/compressed-mm-oob-guard` is pushed and needs a PR against main** — that is C5,
  and opening the PR is the only step left on it.
- Safety refs, local-only, delete once you are satisfied: `backup/llk-tests-pre-rebase` (first
  rebase) and `backup/pre-rebase-20260820` (the 2026-08-20 one).

---

## F. Intermittent `test_matmul_custom_compressed` failures — needs an owner

**Diagnosed; what is left is the fix.** Under repeated runs of the suite on BH p100a the failure
reproduces roughly 2 in 6, and `run_test.sh`'s triage caught the state:

```
Unpacker/Math/Packer mailboxes = 0x0 (KERNEL_STARTED)
TRISC0/1/2  in_reset=True
BRISC       pc=0x368, unchanged  (spinning)
BriscCounter=0x118 (280)   host Python counter: 281
```

All three TRISCs sit in soft reset while BRISC spins one command behind the host — a **host↔BRISC
command-protocol desync**, not an LLK compute bug. `get_tensix_state` then failed to halt BRISC,
so the device was already unresponsive.

**The symptom has two shapes, and this is the part an owner will otherwise miss.** As well as
hanging, the same suite has produced a **wrong answer**: `shape=(1, 64, 32), formats=('bfp4',)`
at PCC -0.033, 587/588, which did not reproduce (that variant passes 17/17 in isolation, and the
suite then passes 588/588). Looking only for a hang will miss half of it.

**Scope.** Every failing variant reproduced — `clustered`, `interleaved`, `single` — is
`@pytest.mark.nightly`, and the PR gate filters `not nightly`. So this affects nightly runs, not
the gate.

**Before reproducing it yourself:** back-to-back runs are not how CI runs it and may be the
aggravating factor rather than an independent trigger; and it wedges the device
(`PcieHangError`, all devices unhealthy) often enough that you should expect to `tt-smi -r`
between attempts. That is the sanctioned remedy for a runtime timeout here — not for a reconfig
escape. Whether this is the same phenomenon as C3 is still unproven: C3 is a golden mismatch
under a specific test ordering, this is primarily a hang, so they are probably different.

## Environment setup, for whoever picks this up next

One-time setup, kept here because none of it is in `tests/README.md` and each blocker presents
as something worse than it is:

**The short version: run `source tests/setup_external_testing_env.sh`.** §8 of the DONE
document already says so, and it does the whole job — creates `tests/.venv`, installs
`requirements.txt` with the `--index-strategy unsafe-best-match` that the multi-index
requirements file needs, and fetches SFPI. The three blockers below are what you hit if you
*do not* find that script first, recorded because each one presents as something worse than it
is:

1. **No `tests/.venv`.** `.claude/scripts/run_test.sh` requires it (exits 3, `ENV_ERROR`) and
   activates it for `ttexalens`. The ambient `/opt/venv` had `tt-exalens 0.3.11` where
   `requirements.txt` pins **0.3.29**, and `CallstackEntry` moved modules between those
   versions. That ImportError blocks *collection of the entire suite* from `conftest.py`, so it
   reads like a broken repo rather than one stale dependency.
   **Confirmed again on 2026-08-20:** `tests/.venv` (0.3.29) is present and `run_test.sh` works,
   while `/opt/venv` is still 0.3.11 — so `python -m pytest` with the ambient interpreter still
   fails at collection. If you are going to reach for pytest anyway, the pinned versions are
   `tt-exalens 0.3.31` **plus `tt-umd 0.9.9`**; 0.3.31 against the ambient `tt_umd 0.9.3` fails
   later and more confusingly, on
   `TopologyDiscoveryOptions.device_init_failure_action`. Use `run_test.sh`.
2. **No SFPI toolchain** (`tests/sfpi/`) — every compile fails with
   `riscv-tt-elf-g++: not found`, which looks like a broken driver rather than a missing
   toolchain. `tests/setup_testing_env.sh` fetches and sha256-verifies it (7.69.0), but note it
   *also* runs `pre-commit install`; run only the download half if you do not want git hooks
   added to your checkout.
3. **`uv` cache** — if `~/.cache` is a dangling symlink (it was, to a non-existent
   `/localdev/$USER/.cache`), `uv venv` fails on cache init. `mkdir` the target or set
   `UV_CACHE_DIR`.

Both `tests/sfpi/` and `tests/.venv` are gitignored, so none of this shows up in `git status`.

**Use `run_test.sh`, never `pytest` directly** — the tt-llk `CLAUDE.md` requires it, and it
serialises silicon access with `flock`, kills stale processes, and triages hangs. Its default
`--maxfail 10` will stop a run early; pass `--maxfail` higher when you want a full count.

**On method.** Nearly every test on this branch passed on its first hardware run, and in every
case a mutation is what established it was not vacuous — a deliberately broken helper, a halved
addr_mod stride, an unguarded load, a wrong chunk stride. Several of those mutations changed
what the test claimed: one exposed that the `tile=0` variants prove nothing about the helper,
another that the OOB read cannot be detected by any golden. **A test that passes first try has
not yet been shown to test anything** — budget for the mutation, not just the test.

---

## Explicitly out of scope

**Perf tests.** There is no perf coverage for any promoted family (nor for `topk_xl` or
`sampling`), and 56 functional test modules have no perf counterpart. This was reviewed and
**deliberately ruled out** — recorded here so it is not re-raised as an oversight. The perf
infrastructure itself is ready if that changes: discovery is marker-driven with pytest-split
sharding, `PerfRunType` already provides the isolation modes, and no registry edit is needed to
onboard a new op. The two things that *would* need doing first are wiring
`compare_test_and_perf.py` into CI (it exists, runs nowhere) and fixing its filename-based
pairing, which reports real pairs as unmatched.

---

## Suggested order

1. **E** — retitle the PR and fill in the body. Minutes, and nothing else gets reviewed until it
   is done. Posting the 21 drafted replies belongs here too.
2. **C5** — push `ldjurovic/compressed-mm-oob-guard` and open the PR. The work is done and
   verified; main carries the defect until someone clicks.
3. **C4** — route to an owner. A located defect in a shipping op with a minimal reproducer, which
   makes it the cheapest real fix on the list, and it decides D1.
4. **C1 / C2 / C3** — route to owners too; they are decisions rather than work, and C2 gates A5.
5. **C6** — route to an owner with the xfail. It is the last gap in the `top32_rm` family's
   coverage, and the fix is half a day once someone picks between clearing the tile and rejecting
   `num_faces < 4`.
6. **B1**, once an owner in the metal test tree exists — and note it needs CB metadata, not just
   a willing owner in tt-llk.
8. **F** — route to an owner with the triage output; no further reproduction needed, and
   repeating it costs a device reset.
