# Handoff: finishing the blaze→experimental tt-llk tests on a Blackhole card

You are picking up PR **#53361** on branch **`ncvetkovic/blaze-experimental-llk-tests`**.
This branch adds tt-llk unit tests for the LLKs promoted from tt-blaze into tt-metal
`experimental/` (umbrella issue **#47554**). Most tests pass; **5 are blocked on things
that only a real Blackhole card can resolve** — that's your job. Everything below is what
the previous session (on a Wormhole n150 + the ttsim functional simulator) could and
couldn't do.

## TL;DR — what to do
Fix these 5, in priority order, on the BH card. For each: reproduce on the card, fix the
golden/driver, confirm it passes, then remove its quarantine marker (see "Un-quarantine"
per test). Commit to this branch (PR auto-updates), and re-run the BH CI gates to confirm.

| # | Test | Symptom on real BH | What's wrong (best current theory) |
|---|------|--------------------|-------------------------------------|
| 1 | `custom_mm` | 486 PCC fails (clean) | Golden/packing bug. Hand-rolled `packed_a`/`packed_b`/reorder in the test vs the **proven `compressed_utils.run_compressed`** path used by the *passing* `matmul_custom_compressed` sibling. ttsim can't run it (`UnimplementedFunctionality: bank_clr_ctrl=1`). |
| 2 | `sdpa_custom_mm` | 24 PCC fails (clean) | Golden/reorder bug. The `row=64` SrcB counter-wrap DOES run on silicon (no wedge), so this is a wrong golden, not an addressing hang. ttsim can't run it (traps `row=64`). |
| 3 | `top32_rm` | **WEDGES** (TENSIX-TIMED-OUT) | The bitonic SFPU sort hangs the Tensix on real silicon. Needs interactive debug of the sort completion path (assert-probe / debug-reg write is the prime suspect). |
| 4 | `hw_cleanup` | **WEDGES** | Teardown family. Either our driver's pack re-init after the deliberate MOP/stride poison is incomplete, or the promoted `#53296` LLK's mailbox rendezvous itself wedges. |
| 5 | `sdpa_bcast_col_srcb_reuse` | **HANGS** (3/3 on ttsim) | SrcB-reuse-from-DEST: a SrcB bank-valid / MOVD2B-from-DEST sequencing hazard. No semaphore to balance (unlike its sibling). |

## Environment setup on the card
The dev-box worktree (`/localdev/ncvetkovic/work2/blaze-llk-tests`) is **NOT shared** with the
BH card, so set up fresh:
```bash
git clone <tt-metal> && cd tt-metal
git checkout ncvetkovic/blaze-experimental-llk-tests
git submodule update --init --recursive   # tt-llk is in-tree; tracy/umd are submodules
cd tt_metal/tt-llk/tests
# venv (the harness does NOT create it; setup_testing_env.sh only installs SFPI):
python3.10 -m venv .venv && ./.venv/bin/pip install -U pip && ./.venv/bin/pip install -r requirements.txt
# requirements pin tt-exalens==0.3.29 (needs the CallstackEntry symbol) — do not use an older one.
./setup_testing_env.sh    # installs SFPI; if tests/sfpi is missing, symlink your sfpi install there:
#   ln -sfn ~/sfpi tt_metal/tt-llk/tests/sfpi   (compiler at sfpi/compiler/bin/riscv-tt-elf-g++)
```
**Run a test on the real card** (no simulator env; `CHIP_ARCH` comes from the card):
```bash
cd tt_metal/tt-llk/tests/python_tests
../.venv/bin/python -m pytest -p no:randomly test_custom_mm.py -k "M:1-kt:2-ct:2"
```
Sanity-check the card first with a known-good test: `test_sum_reduce_scalar.py` (should pass).

## Current state of every test (so you know what NOT to touch)
- **Pass on real BH (validated, leave alone):** `sdpa_reduce_row`, `sum_reduce_scalar`,
  `pack_rows_to_addr`, `rmsnorm_bcast_scalar_dest_reuse`, `sdpa_weighted_reduce`,
  `unpack_A_sdpa`, `sdpa_custom_mm_reuse_dest_srcb`.
- **Broken, YOUR job:** `custom_mm`, `sdpa_custom_mm`, `top32_rm`, `hw_cleanup`,
  `sdpa_bcast_col_srcb_reuse` (the 5 above).
- **Deliberately parked (reverted drivers — leave skipped/xfail unless you want to tackle
  the underlying LLK):** `mul_reduce_scalar_chunked` (xfail; result ~5-30x high),
  `eltwise_mul_scalar_hifi` (skip; hangs device), `eltwise_add_scalar` (skip; wedges device).

## Un-quarantine instructions (remove the guard once a test passes on the card)
- `custom_mm`: no guard — it already runs and fails; just make the golden pass.
- `sdpa_custom_mm`: has `def _skip_on_simulator(request)` — it **already runs on the card**
  (the skip only triggers under `--run-simulator`). Just fix the golden.
- `top32_rm`: `pytestmark = [skip_for_wormhole, skip_for_quasar, pytest.mark.skip(...)]` near
  the top. Once the wedge is fixed, **remove the `pytest.mark.skip(...)` entry** (keep the
  two arch skips). It also has a `_skip_on_simulator` for ttsim — leave that.
- `hw_cleanup`: `pytestmark = [blackhole_only, pytest.mark.skip(...)]`. Remove the
  `pytest.mark.skip(...)` entry once fixed.
- `sdpa_bcast_col_srcb_reuse`: module-level `pytestmark = pytest.mark.skip(...)`. Remove it
  once the deadlock is fixed. Its **golden is already correct** (per-face column-0 broadcast)
  — only the SrcB-reuse handshake needs solving. Its sibling `sdpa_custom_mm_reuse_dest_srcb`
  shows the reuse family IS unit-testable: study how that test's PACK thread supplies the
  `UNPACK_MATH_DONE` producer (init 0/KT_DIM + post KT_DIM up front, the
  `reduce_block_max_test.cpp` pattern) — though bcast_col has no semaphore, so its fix is a
  bank-valid/dest-section handshake, not a semaphore balance.

## Hard-won gotchas (don't relearn these)
- **`0.0 KHz` in ttsim output is the NORMAL summary line, printed on PASSING tests too — it
  is NOT a deadlock.** The real hang signal is a pytest **timeout with no `N passed/failed`
  summary**. (This mis-read cost the last session hours.)
- **A device wedge/hang CANNOT be `xfail`ed** — xfail still *executes* the test, so a hang
  wedges the Tensix and cascades `TENSIX TIMED OUT` into every later test. Wedgers must be
  `pytest.mark.skip` (or fixed).
- **ttsim error taxonomy:** `UnimplementedFunctionality` = sim gap, needs a real card (that's
  why 1/2/4 above need you). `UndefinedBehavior` / a 0-progress deadlock = usually a REAL
  silicon issue. But note `sdpa_custom_mm`'s `row=64` was a documented HW counter-wrap ttsim
  doesn't model, yet it STILL fails on silicon → so its golden is genuinely wrong.
- **Some tests fail alphabetically-first and cascade:** a single wedger (e.g. `top32_rm`)
  produces hundreds of collateral `TENSIX TIMED OUT` on unrelated tests. Fix/quarantine the
  *first* wedger, then re-run to see the true state of the rest.

## CI gates (this is how you validate on BH without holding the card)
Two workflows run the tt-llk python tests on a real BH runner (`bh_p150b`):
```bash
gh workflow run llk-bit-exact.yaml --ref ncvetkovic/blaze-experimental-llk-tests -f architecture=blackhole -f bit-exact-runs=100   # determinism (re-runs each variant N×)
gh workflow run pr-gate.yaml --ref ncvetkovic/blaze-experimental-llk-tests                                                          # smoke + build (runs llk-smoke)
```
- **Do NOT double-dispatch.** Both have a concurrency group keyed on the branch with
  `cancel-in-progress` — a second dispatch cancels the first, and a *cancelled* run reports
  as `failure` in the summary job (looks like a test failure but isn't). Dispatch once, wait.
- New `test_*.py` auto-collect (smoke's pytest-marker filter is exclusion-only:
  `not perf and not nightly and not quasar`). No manifest to update.
- Pull failures from a run: `gh api repos/tenstorrent/tt-metal/actions/jobs/<job-id>/logs`
  then grep for `matmul failed | Error type | TENSIX TIMED OUT`. `--log-failed` often returns
  empty for matrix jobs.

## Commit / PR workflow
- Commit to this branch; PR #53361 updates automatically. Pre-commit hooks (black,
  clang-format, autoflake, isort) **reformat files and fail the first commit** — just
  `git add` the reformatted files and commit again.
- **Edit ONLY test files** (`tests/sources/*_test.cpp`, `tests/python_tests/test_*.py`). Do
  NOT edit promoted headers, `helpers/test_variant_parameters.py`, `helpers/golden_generators.py`,
  or `conftest.py`. If the real bug is in a promoted header, report it (see below) — don't
  hack the test around it.

## Dependencies / branch state
- This branch is **stacked on 2 unmerged promotion PRs**: **#52713** (top32_rm headers) and
  **#53295** (SDPA family headers). It merges them in, so the diff is large; it shrinks when
  they land on `main`. Rebase/merge `origin/main` periodically.

## Header findings to flag to pmilenkovic (@pmilenkovicTT) — real LLK issues, not test bugs
1. **#53295 `-Werror`** (already filed as a PR comment on #53295): `llk_math_sdpa_bcast_col_srca_srcb_reuse.h`
   has 5 unused var/param symbols that break the tt-llk `-Werror` build. (That test,
   `sdpa_bcast_col_srca_srcb_reuse`, is separately skipped for this.)
2. `llk_unpack_A_rmsnorm.h`: `_llk_unpack_A_rmsnorm_init_`'s `UNP_SEL` programs unpacker-B's
   X-end for `BroadcastType::SCALAR` while the MOP streams SrcA → only lane 0 of each SrcA
   row is read (worked around in the rmsnorm test's driver).
3. The SrcB-reuse headers (`llk_math_sdpa_custom_mm_reuse_dest_srcb.h`, likely
   `llk_math_sdpa_bcast_col_srcb_reuse.h`) have no compile-time opt-out for the
   `UNPACK_MATH_DONE` handshake, so they deadlock standalone (the test must supply the
   producer half).

## Skills worth invoking for the wedge/deadlock debugging
`semaphore-handshake-audit`, `srcreg-bank-sync-audit`, `debug-kernel`, `race-audit-all`.
There is also a `run-test` skill / test-runner agent — never run pytest directly if a
project skill exists for it.

Good luck. Start with `custom_mm` (biggest clean-fail block, most tractable golden fix).
