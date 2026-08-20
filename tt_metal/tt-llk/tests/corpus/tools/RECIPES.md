# Build-infra recipes (lane DA) — killing gate-recompilation waste

Four adoption recipes.  Everything here is OPT-IN: no existing script's
behavior changed.  Evidence: `~/sfpi-uplift/laneDA-evidence-20260820/`.

## 1. Base-leg store: byte-gate WITHOUT recompiling the base leg

The base half of every pin-cycle byte-identity gate (pinned cc1plus +
reviewed flag set) is byte-identical for every lane.  Compile it ONCE
per pin, publish, and every lane gates against the published manifest.

**Populate, once per pin** — either compile-and-publish (the store's
native path; add `--keep-build` on the shared farm so sweep group
producers can also seed RUNNER_TEMPs from it):

    cd tt_metal/tt-llk/tests/corpus
    python3 corpus_leg_store.py ensure --arch bh --flags '<exact flag string>'

or ADOPT an already-completed verified leg (no recompile — the pin's
union-gate leg, a lane's crossdiff-verified base leg) while its build
trees or extracted-.text payloads still exist:

    python3 tools/leg_store_seed.py --arch bh --flags '<exact flag string>' \
        --cc1plus <the binary that compiled it> \
        --recorded-cc1plus-sha <sha256 from the leg's evidence> \
        --farm <the tt-metal checkout it compiled from> \
        --build-root <.../tt-llk-build>   # or --text-root <.../text> ... \
        --evidence '<where the leg's provenance is recorded>'

The seed REFUSES unless the named binary still hashes to the recorded
sha; payload bytes are re-hashed, never copied as recorded truth.

**Consume, per lane** — compile ONLY your edit leg, then:

    python3 tools/leg_store_gate.py --arch bh --flags '<base leg flag string>' \
        --base-compiler <shared base driver> \      # or --base-cc1plus <binary>
        --tt-metal-home <your farm> \
        --mine <your edit leg's tt-llk-build>

Exit 0 byte-identical / 2 diffs (CHANGED/MISSING/EXTRA listed) /
1 REFUSED — on any refusal (no entry, tampered manifest, wrong sha,
wrong farm/head) recompile the base leg yourself via `ensure`, which
publishes it for everyone else.  `--list` shows the inventory.

Rules the tools enforce for you (do not work around them):
- .text hashes are FARM-PATH-DEPENDENT (LLK_PROFILER path-hash
  immediates): entries only gate legs compiled from the SAME farm
  realpath at the SAME tt-metal head.
- keys are the cc1plus BINARY sha + byte-exact flag string (including
  any -B): two builds of the same source are different keys.  Seeded
  pin-14-seed keys live under `fe92c1171d0f` (gcc-build-laneCJ; exact
  flag strings in the lane DA evidence dir).  Independent-reproduction
  proof: laneCZ's separately compiled base legs gate BYTE-IDENTICAL
  (3210/3210) against the laneCN-seeded entries.

Saving: ~40-60 min per lane per pin cycle (the full mapped-corpus base
leg), multiplied by every lane that used to recompile it.

## 2. DejaGnu: parallel full rvtt.exp — IDENTITY PROVEN (and a surprise)

Proof (laneDA evidence, dejagnu/): full rvtt.exp on ONE build
(stockcfg @ pin-14 seed e0754714a5b, cc1plus 40feb0d87659…), serial vs
12 concurrent shards, sorted result-multiset diff: **IDENTICAL — 3483
result lines each (3471 PASS; FAIL set = frozen-9 + the documented
environmental 41863-consteval row on an un-installed build tree)**.

THE SURPRISE: serial full rvtt.exp took **28 seconds** on this box
(compile/scan tests, no execute boards) — it was never the hour-long
gate; the hour lives in the corpus legs (see recipe 1).  Parallel is
proven identical anyway (3s at 12 shards) for when suites grow.

Blessed serial invocation (dejagnu_gate.sh shape — scratch dir, never
the build tree's own testsuite dir):

    mkdir DEJ && cd DEJ            # site.exp: copy a known-good one and
    # sed rootme/srcdir/tmpdir/GXX_UNDER_TEST to your build + source
    runtest --tool g++ --srcdir $SFPI_GCC_SRC/gcc/testsuite rvtt.exp

Blessed parallel invocation: shard the *.C basenames of the five
rvtt.exp driver dirs (tt, tt/rv, tt/rocc, tt/sfpi, tt/tensix) into N
lists — CO-LOCATE duplicate basenames (clobber-45648-bh.C,
combine-46215-bh.C, setexp-fold-bh.C exist twice) so each file runs
exactly once — then N concurrent scratch dirs, each:

    runtest --tool g++ --srcdir $SRC/gcc/testsuite "rvtt.exp=<name1.C name2.C ...>"

and compare `grep -E '^(PASS|FAIL|XPASS|XFAIL|UNRESOLVED|UNTESTED|UNSUPPORTED|ERROR): ' */g++.sum | sort`
against the serial run's sorted lines.  Runner:
`~/sfpi-uplift/laneDA-evidence-20260820/dejagnu/run_dejagnu_parallel_proof.sh`.
Env parity matters: the sfpi/ pressure suite is gated on env(SFPI) —
set it in BOTH arms or NEITHER.

## 3. Core budget: with-cores

Long compile legs trisect each other on a shared box.  Wrap them:

    tools/with-cores 12 -- make -j12 all-gcc
    tools/with-cores 8  -- ./launch-shards.sh     # $WITH_CORES exported
    tools/with-cores status

Pool `~/sfpi-uplift/.corebudget`, nproc-4 slots (28 on quietbox0),
flock-per-slot (crashed holders auto-release — kernel-owned locks),
all-or-nothing acquisition (no partial-hold deadlock), exit status
propagated.  OPT-IN: nothing acquires a slot unless wrapped.  Selftest:
`tools/selftest_with_cores.sh`.

## 4. Lane toolchain builds: ccache + the pin-rebuild fast path

**ccache verdict: BLESSED for lane builds** (5-arm proof, laneDA
evidence fastpath/).  The stockcfg recipe tolerates `CC='ccache gcc'
CXX='ccache g++'` on configure with zero recipe changes.  Measured at
-j12, same build path, same pkgversion, sfpi-gcc e0754714a5b:

| arm | ccache | wall | cc1plus sha |
|-----|--------|------|-------------|
| 1 clean-configure | cold | 370s | 36a2b56a5f41… |
| 2 clean-configure | hot (100% direct hits) | **56s** | 36a2b56a5f41… |
| 4 clean-configure | hot (repeat) | ~56s | 36a2b56a5f41… |
| 3 clean-configure | disabled | 357s | 40feb0d87659… |
| 5 clean-configure | disabled (repeat) | 307s | 40feb0d87659… |

Both families are INTERNALLY reproducible (1=2=4, 3=5).  The
cross-family divergence is EXACTLY 36 bytes of 536MB: the 20-byte
.note.gnu.build-id + the 16-byte embedded genchecksum MD5 (PCH
validity) — ZERO code/data bytes differ, the xg++ driver is
byte-identical across all five arms, and a codegen sample produces an
identical .s from both cc1plus binaries.  Cause: host compiles use
-g and ccache's preprocessed objects carry benignly different debug
info, which genchecksum (hash over .o files) sees and the linker
mostly discards.

Recipe (lane builds):

    export CCACHE_DIR=~/sfpi-uplift/<lane>-ccache   # or a shared one
    CC='ccache gcc' CXX='ccache g++' $SRC/configure <stockcfg flags...>
    with-cores 12 -- make -j12 all-gcc              # hot rebuild: ~1 min

Policy: the INSTALLED pin rebuild stays clean-by-policy (uncached);
lane builds use ccache freely — the base-leg store and all byte gates
key on the ACTUAL binary sha, so family membership is self-consistent
by construction.  Never compare a ccache-built cc1plus sha against a
plain-built one and call it a mismatch: check the 36-byte decomposition
first (evidence: fastpath/divergence/DECOMPOSITION.txt).

**Pin-rebuild fast path (PROVEN IN PRODUCTION at pin 14)**: install the
gate-proven union binaries directly — cc1plus (+ cc1/lto1) AND the
in-tree xg++/xgcc drivers TOGETHER — with the cc1plus sha256 gate as
the sole trust anchor (verify the sha you gated is the sha you
installed, then witness_preflight on the INSTALLED binary).  Minutes,
not hours.  GOTCHA: installing cc1plus alone is INSUFFICIENT whenever
.opt options changed — the DRIVER embeds the option tables and rejects
the new flags; xgcc/xg++ must ship with cc1plus.

**Surgical stamp policy (sfpi build system)**: already exists —
scripts/build.sh pre-marks every non-stage2 newlib stamp
"Incremental" when a base txz is available (build.sh lines ~227-237),
and the standing pin practice (rm stamps/build-gcc-newlib-stage2 only)
is exactly the gcc-sources-only fast path.  The base-txz acquisition
had a SILENT-fallback-to-full-rebuild hole when the download 404s
(dev-branch versions with unpublished bases): fixed on sfpi branch
agent/build-infra (b5cd9ac) — local store ~/sfpi-uplift/build-base/
(override SFPI_BASE_STORE) tried first, loud WARNING +
BASE_FALLBACK_FULL_BUILD marker on total miss.
sfpi_7.69.0_x86_64_debian.txz is staged in the local store.

## 5. Sweep resume across pins (audit finding — fix belongs to the sweep owner)

tt-metal's JIT cache keys on `g++ --version`, so pin swaps invalidate
every compile — but sweep_2x2.py's hash-matched device resume is
DESIGNED to reuse cells whose archived .text equals this run's build
(exactly right for OFF/hand legs, byte-identical across pins,
3211/3211 proven repeatedly).  AUDIT VERDICT: cross-pin reuse NEVER
fires, one-line reason: **the resume cache is probed only under THIS
run's evidence root (`_device_job`: `work = self.ev / op / tag / sel /
label-leg`), and `--prev-run` is consumed ONLY by the scoreboard
annotator (sweep_2x2.py ~3145/~3359)** — the weekly wrapper mints a
fresh `weekly-<date>` root each run, so every OFF/hand cell
re-measures.  Fix shape (sweep owner): in the cached-cell probe, also
probe `prev_run/<op>/<tag>/<sel>/<label>-<leg>` under the SAME jobkey +
classify-hash checks (all the trust machinery already exists) and
adopt the cell on match.  Expected saving: roughly the OFF+hand half
of device work per weekly sweep.

## 6. Classify prewarm workers (measurement + recommendation)

Weekly pin-13 sweep, 6 workers: 976 classify compile events (~488
pytest compile-producer sessions) in 15.4 min sustained ≈11 s/session
≈6 cores busy — on a 32-logical-core box with a 28-slot budget and the
device phase needing ≤4 cores.  RECOMMENDATION: `SWEEP_CLASSIFY_WORKERS=12`
(expected classify wall ~8 min, ~2x), safe because sessions are
compile-dominated and the shared-/tmp build race is already avoided by
private RUNNER_TEMPs.  Caveat: log-derived, not a controlled A/B — the
harness venv/farm were the pin-14 merger's live resources tonight.
Confirmation recipe: run the classify phase of 3-4 ops twice in
private evidence roots with SWEEP_CLASSIFY_WORKERS=6 vs 12 and compare
wall clock; adopt 12 if ≥1.6x.
