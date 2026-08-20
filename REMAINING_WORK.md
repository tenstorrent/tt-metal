# tt-llk blaze promotions — everything still to do

Single actionable index of what is left, with a plan per item. **Pruned 2026-08-20**: closed
items are no longer listed here at all. What is done, with results and the findings that came
out of it, lives in [`BLAZE_PROMOTION_TESTS_DONE.md`](BLAZE_PROMOTION_TESTS_DONE.md) — including
the four test items closed between 2026-08-18 and 2026-08-20 (`set_dst_write_addr_offset`,
`mul_reduce_scalar` re-entry, plain `custom_mm`, `top32_rm`) that earlier versions of this
document tracked as A2, A3, A4, A6, D2 and D3.

- **Forensic detail** for A5 stays in
  [`blaze_llk_promotion_test_strategy.md`](blaze_llk_promotion_test_strategy.md) §9.
- All three promotion PRs (**#52709, #52713, #52727**) have merged, so nothing below is blocked
  on a promotion landing.

Where an item says "uncalled", that means no driver under `tt_metal/tt-llk/tests/sources/`
references the symbol. State verified against the tree on 2026-08-20.

---

## At a glance

| # | Item | Type | Size | Blocked on |
|---|------|------|------|-----------|
| **A1'** | `custom_mm` (plain) — `read_transposed`, and the top of the `kt_dim` range | test | ~2 h | — |
| **A2''** | `top32_rm` — the metal wrapper layer (its other half is now C6) | test | B1-shaped | needs an owner in the metal tree |
| **A5** | `eltwise_mul_scalar` HiFi init — untested, rationale disproved | test | unknown | C2 |
| **B1** | `custom_mm` vs `compressed_custom_mm` divergence guard | test, **outside tt-llk** | ~1 d | needs an owner (interim static guard landed) |
| **C1** | `dense_packing` W-stride not format-aware | **defect** | ~0.5 d once decided | owner decision |
| **C2** | `eltwise_mul_scalar` HiFi workaround rationale does not hold | **question** | — | #52709 author |
| **C3** | `topk_xl` → `eltwise_binary` reconfig escape | **defect**, pre-existing | unknown | needs an owner |
| **C4** | `mul_reduce_scalar` re-entry needs a DEST-section boundary | **defect** | unknown | needs an owner |
| **C5** | OOB metadata read shipped to main with #52727 | **defect** | minutes | needs the fix cherry-picked |
| **C6** | `top32_rm` 32-bit unpack: partial chunk sorts against stale Dest | **defect**, NEW | ~0.5 d | needs an owner (xfail pins it) |
| **D1** | `mul_reduce_scalar_chunked_tile` ships with no caller | cleanup | — | C4 decides it |
| **E** | PR mechanics — title and body | chore | minutes | — |
| **F** | `test_matmul_custom_compressed` intermittent — host/BRISC desync | **defect**, nightly-only | unknown | needs an owner |

Nothing here is blocked on a promotion PR any more, and **only four of the thirteen are test
work** (A1', A2'', A5, B1) — two of those are hours rather than days. Six are product decisions
or defects that need an owner rather than a test (C1–C6), three of them defects in shipping ops
with a reproducer already attached (C1, C4, C6); the rest are one cleanup (D1), the PR's own
metadata (E), and one intermittent-failure investigation (F).

---

## A. Functional test gaps

### A1'. `custom_mm` (plain) — `read_transposed`, and the top of the `kt_dim` range

`transpose` and `split_acc`/`finalize` are done (64 variants passing; see the DONE document for
what the flags actually change and how each axis was shown to discriminate). Two smaller things
were never in A1's plan and are what is left of the family:

**1. `read_transposed` (~2 h, unblocked).** A *different* flag from `transpose`, and easy to
conflate with it: `transpose` transposes each in1 tile inside SrcA, while `read_transposed`
changes the ORDER the unpacker reads in1 tiles out of L1 --
`_llk_unpack_AB_custom_mm_<read_transposed, clear_src>` computes
`block_increment = kt_dim * tile_size_a` and a negative `inner_increment` instead of walking
tiles contiguously. So it reads the ct dimension with a kt stride. Both polarities are reachable
from the existing driver; the golden is a reindexed operand, not a transposed one.

**2. The top of the documented `kt_dim` range.** The sweep runs `kt_dim` 2 and 4 against tables
that claim 2..256. Nothing suggests a defect up there, but the parity constraint
(`TT_MOP(0, (kt_dim / 2) - 1, 0)`) is the only thing anyone has verified about the range.

**Watch for.** `ct_dim <= 8` is a real ceiling in this configuration, not a documentation gap:
the ct output tiles are all live in DEST and half-sync holds 8 bf16 tiles. The doc tables claim
1..16; the upper half needs `DstSync::SyncFull` or a caller that splits the block.

### A2''. `top32_rm` — the metal wrapper layer, and one shape the hardware cannot do

The mixed 1024+tail shape is done: the pre-sorted mode now finishes a row that is not a
multiple of 1024, at rows 1088 and 1152 (see the DONE document). Two things remain, and the
second is new information rather than leftover work:

**1. The metal wrapper layer (B1-shaped, needs an owner).** The 7
`llk_math_deepseek_top32_rm_*` wrappers are on main with no caller — they arrived with #52713,
not with this branch. A tt-llk test cannot reach the metal API layer; covering them needs a
metal-side test, exactly like B1.

**2. The Metal dev test's own `row=3232` shape is not reachable — see C6.** That row ends in a
32-element chunk, and a partially-filled chunk on the 32-bit branch of this family's unpack
sorts against stale Dest. The two halves either side of it are covered; the middle is pinned by
an xfail and tracked as C6.

**Not planned:** the 8-datum `bitonic_top32_load8`/`store8` helpers, which the header records as
referenced by no kernel.

**Still true before touching this area:** `_top32_rm_init_()` and `_topk_xl_init_<K, fused>()`
**cannot both be called in one kernel** — they overlap in the ADDR_MODs, the MOP and the REPLAY
buffer, and the math thread hangs (Finding 3). And see C3: a pre-existing reconfig escape lives
here, so bisect single-file-then-target before blaming your own driver.

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

**An interim guard is already in place**, and it is not a substitute for the above:
`tests/python_tests/test_custom_mm_uninit_parity.py` — a device-free static gate that runs in
the smoke job which already collects the whole `python_tests` directory. It asserts two things:

- the two compute-API uninit bodies are still byte-identical modulo comments (divergence); and
- the driver's replicated `DENSE_WSTRIDE` / `DEFAULT_WSTRIDE` expressions are still the ones
  the headers use (staleness) — the driver hardcodes them, so if a header changes the driver
  keeps asserting the old behaviour *and passes*, because it programs that stride itself.

Both mutation-checked: dropping the `restore_tile_pack_mop` branch from
`compressed_custom_mm.h` fails the first with a diff naming the missing branch; changing the
driver's `DENSE_WSTRIDE` to `* 4` fails the second.

**This does not close B1.** A text match cannot say the functions *work*, only that they still
say the same thing. The metal-side test calling the real entry points is still wanted, and is
still the item that needs an owner. What the guard buys is that divergence now fails loudly
instead of silently, which was the specific risk.

**Note.** The commit that documented `restore_tile_pack_mop` was dropped in the 2026-08-19
rebase — main's merged #52727 has no such flag at all, so there was nothing left to document.

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

The `xfail` in `test_custom_mm_uninit_restore.py` flips to XPASS when either lands — **and as of
`e94b5dd0fbe` that is actually true**; see Correction 1.

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

Located by the `mul_reduce_scalar_chunked_tile` investigation. `b59c5df50aa` adds a ~40-line driver
(`tests/sources/mul_reduce_scalar_reenter_test.cpp` +
`tests/python_tests/test_mul_reduce_scalar_reenter.py`) that runs the known-good non-chunked
sequence twice over the same input. On BH p100a:

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

**Fix:** cherry-pick `54e218ebbce` onto main, or re-apply the three-line guard. Minutes of work.
It will otherwise ride in on this branch whenever #53130 merges, which is fine but leaves main
carrying it in the meantime.

---

### C6. A partially-filled chunk on `top32_rm`'s 32-bit unpack branch sorts against stale Dest — **defect, NEW**

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
it flips to XPASS the moment the branch starts clearing its tile. It also means the sweep has a
hole with a name: the Metal dev test's `row=3232` shape ends in a 32-element chunk and is the
one shape this family cannot currently do correctly on 32-bit data.

**Fix shape, for whoever owns it:** either clear the whole tile on the 32-bit path (the ZEROACC
loop already exists — it just needs to run over all 4 faces, or the unpacker needs a -infinity
equivalent), or reject `num_faces < 4` on that branch at compile time and document that a
partial chunk is 16-bit only.

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
- **The branch wants one more rebase.** #52713 has merged, so the five commits carrying the
  `top32_rm` promotion payload should drop the way #52727's did — the branch is ~81 commits
  behind main, and after the rebase the diff should be test files plus the LLK-side cleanups
  only. Expect `helpers/test_variant_parameters.py` to be the one file both sides touch; it has
  merged cleanly twice.
- `backup/llk-tests-pre-rebase` is a local-only safety ref from the first rebase; delete it once
  you are satisfied.

---

## F. Intermittent `test_matmul_custom_compressed` failures — diagnosed, needs an owner

Was an unidentified single failure; now characterised. Six back-to-back runs of the suite on
BH p100a:

| run | outcome |
|---|---|
| 1 | 588 passed |
| 2 | **hang** (exit 5) in `test_matmul_custom_compressed_clustered` |
| 3 | 2 failed — build-tree race, see below |
| 4 | 3 failed (`TTException`) in `test_matmul_custom_compressed_single` |
| 5 | **hang** (exit 5) in `test_matmul_custom_compressed_interleaved` |
| 6 | 588 passed |

**It is a hang, not a golden mismatch.** `run_test.sh`'s triage on run 2 caught the state:

```
Unpacker/Math/Packer mailboxes = 0x0 (KERNEL_STARTED)
TRISC0/1/2  in_reset=True
BRISC       pc=0x368, unchanged  (spinning)
BriscCounter=0x118 (280)   host Python counter: 281
```

All three TRISCs sit in soft reset while BRISC spins one command behind the host — a host↔BRISC
command desync, not an LLK compute bug. `get_tensix_state` then failed to halt BRISC, so the
device was already unresponsive.

**It does not affect the PR gate.** Every failing variant reproduced —
`clustered`, `interleaved`, `single` — is `@pytest.mark.nightly`, and the gate filters
`not nightly`. It would affect a nightly run.

**Two caveats on this reproduction, both important:**

1. **Back-to-back runs are not how CI runs it,** and may be the aggravating factor rather than
   an independent trigger. Runs 1 and 6 were clean; the failures cluster in the middle.
2. **Run 3 is not a real failure.** It hit
   `test_matmul_custom_compressed_metadata_word_boundary` with
   `ld: cannot open output file .../elf/pack.elf: No such file or directory` — a `/tmp/tt-llk-build`
   tree race left behind when run 2's hang handler killed the process tree mid-compile. An
   artifact of looping, not a defect, but worth knowing since it is the one failure that landed
   on a gate-visible (non-nightly) test.

**Six runs also wedged the device** (`PcieHangError`, all devices unhealthy), needing
`tt-smi -r`. That is the sanctioned remedy here per the tt-llk notes — a runtime timeout, not a
reconfig escape — but it means this reproduction is not free, and whoever repeats it should
expect to reset.

**The symptom has two shapes, which matters for whoever looks.** On 2026-08-20 this suite also
produced a **wrong answer** rather than a hang: `shape=(1, 64, 32), formats=('bfp4',)` at
**PCC -0.033**, 587/588, in the first run after a device-contention incident. It did not
reproduce — that variant passes 17/17 in isolation and the suite then passes 588/588 — and
nothing in that day's change set is on the compressed path. Recorded as Finding 12b in the DONE
document. Looking only for a hang will miss half of it.

**Still needs an owner**, and it is a well-specified ask: a host/BRISC command-protocol desync
under repeated kernel launches, reproducible ~2 in 6 on p100a, with the triage output above.
Whether it is the same phenomenon as C3's reconfig escape is still unproven — C3 is a golden
mismatch under a specific test ordering, this is primarily a hang, so they are probably
different.

## Environment setup, for whoever picks this up next

The tt-llk suite does not run out of the box on a fresh dev box, and none of the three blockers
is documented in `tests/README.md`. All three are one-time setup, not open work — kept here
because each presents as something worse than it is:

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
2. **C5** — cherry-pick the OOB metadata guard onto main. Minutes, and main carries the defect
   until someone does.
3. **C4** — route to an owner. A located defect in a shipping op with a minimal reproducer, which
   makes it the cheapest real fix on the list, and it decides D1.
4. **C1 / C2 / C3** — route to owners too; they are decisions rather than work, and C2 gates A5.
5. **A1'** — `read_transposed`, ~2 h, unblocked and the cheapest test work left.
6. **C6** — route to an owner with the xfail; it is the only thing standing between the
   `top32_rm` sweep and the Metal dev test's own row=3232 shape.
7. **B1** and **A2''**, once an owner in the metal test tree exists.
8. **F** — route to an owner with the triage output; no further reproduction needed, and
   repeating it costs a device reset.
