# Lane recipes — do EVERYTHING the fast way (one page)

The consolidated fast-path handbook for a fresh lane: build, gate, and
measure without recompiling what somebody already proved.  Everything is
OPT-IN — no existing script's behavior changed.  Provenance: lane DA
(build infra, evidence `~/sfpi-uplift/laneDA-evidence-20260820/`) + lane DD
(wrapper surface, evidence `~/sfpi-uplift/laneDD-evidence-20260820/`).

## 0. Fresh-lane quick path (the 5-minute version)

1. **Lane toolchain build**: ccache + stockcfg (recipe 4) — hot rebuild
   ~56s instead of ~6min.  Wrap long makes in `tools/with-cores N --`
   (recipe 3) so parallel lanes don't trisect the box.
2. **Byte gates**: compile ONLY your edit leg; gate the base half against
   the published base-leg store (recipe 1) — saves 40-60 min per pin cycle.
3. **DejaGnu**: serial full rvtt.exp is ~28s (recipe 2) — never the
   bottleneck; parallel recipe exists for when suites grow.
4. **Measurement**: `headline_bh_sweep.sh` for "show me the flips fast";
   the weekly/nightly wrappers for full surface (recipe 6).  All three
   carry the evidence-root collision guard — respect its refusal.
5. **Pin cut**: `pin-install-fast.sh` installs gate-proven binaries in
   minutes, no rebuild (recipe 5).  cc1plus alone is NEVER enough when
   .opt changed — drivers ship with it.

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

## 4. Lane toolchain builds: ccache (BLESSED) + surgical stamps

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

## 5. Pin cut: install gate-proven binaries in MINUTES, no rebuild

**PROVEN IN PRODUCTION at pin 14** (owner-ratified: the trust anchor is
the sha256 of the installed binaries, never the build path).  The
productized script lives in the **sfpi repo**: `scripts/pin-install-fast.sh`
(committed with the pin-14 cut, sfpi 7e0ffd8):

    ~/sfpi-uplift/sfpi/scripts/pin-install-fast.sh \
        <gcc-build-dir> <install-root> \
        --expect-cc1plus <the sha256 the union gates blessed> \
        --flags <new-flag,comma-list>          # smoke-tests acceptance
        # add --dry-run first

It backs up everything it replaces (rollback on ANY failure), installs
cc1plus + cc1 + lto1 AND the in-tree drivers (xg++ -> riscv-tt-elf-g++,
xgcc -> riscv-tt-elf-gcc) TOGETHER, re-verifies shas, smoke-tests flag
acceptance + a default compile, and writes PIN-INSTALL-MANIFEST.txt.

GOTCHA (banked at pin 14): **installing cc1plus alone is INSUFFICIENT
whenever .opt options changed — the DRIVER embeds the option tables and
rejects the new flags**; xgcc/xg++ must ship with cc1plus.  After
install: verify `sha256sum $(g++ -print-prog-name=cc1plus)` equals the
gated sha, then run witness_preflight on the INSTALLED binary.

## 6. Measurement entry points (this directory, one level up)

- **`headline_bh_sweep.sh` — "show me the flips fast"**: ops =
  HEADLINE_ROWS + every row whose fresh body/golden/mapped test changed
  since the previous pin (git-derived by `headline_ops.py`; log in the
  evidence dir).  `--ops a,b,c` overrides.  Weekly pins/gates; no knob
  attribution, no DejaGnu suites.  Run this BEFORE a full-surface sweep
  (owner priority order 2026-08-20).
- **`weekly_bh_sweep.sh` / `nightly_bh_sweep.sh`**: full surface / nightly
  schedule, unchanged scope.
- **Evidence-root collision guard (all three)**: a wrapper REFUSES to
  write into an existing root recorded under a different toolchain pin
  (PIN_STAMP / preflight.json vs the conf's PINNED_CC1PLUS_SHA256) and
  fails closed on unknown provenance — the 2026-08-20 pin-12/pin-14
  contamination class.  Same-pin roots resume as before.  `SWEEP_DATE`
  is the sanctioned root-name override; the refusal suggests a free one.
- **`--prev-run` chain**: wrappers pass the newest N clean roots
  (default 3, `SWEEP_PREV_CHAIN=N`), skipping contaminated/quarantined
  roots — cross-pin cell reuse engages automatically once its consumer
  lands in sweep_2x2.py.
- **Live logs**: wrappers run the sweep under PYTHONUNBUFFERED + stdbuf,
  so a tee'd log shows progress immediately — a many-minutes-silent log
  now means STALLED, not buffered.
- **Classify workers**: `SWEEP_CLASSIFY_WORKERS=12` recommended on this
  box (laneDA measurement: 6 workers ≈ 15.4 min classify wall; sessions
  are compile-dominated, private RUNNER_TEMPs already avoid the /tmp
  race).  Confirm with a 3-4 op A/B in private evidence roots; adopt 12
  if ≥1.6x.
- **Detach discipline**: `setsid nohup <wrapper> ... & disown` — sweeps
  launched inside harness background tasks die when the task's process
  group is reaped.

## 7. Sweep resume across pins (audit finding — fix owned by the sweep lane)

tt-metal's JIT cache keys on `g++ --version`, so pin swaps invalidate
every compile — but sweep_2x2.py's hash-matched device resume is
DESIGNED to reuse cells whose archived .text equals this run's build
(exactly right for OFF/hand legs, byte-identical across pins,
3211/3211 proven repeatedly).  AUDIT VERDICT: cross-pin reuse NEVER
fires, one-line reason: **the resume cache is probed only under THIS
run's evidence root (`_device_job`: `work = self.ev / op / tag / sel /
label-leg`), and `--prev-run` is consumed ONLY by the scoreboard
annotator (sweep_2x2.py ~3145/~3359)** — so every OFF/hand cell
re-measures each pin.  Fix shape (sweep owner): in the cached-cell
probe, also probe `prev_run/<op>/<tag>/<sel>/<label>-<leg>` under the
SAME jobkey + classify-hash checks (all the trust machinery already
exists) and adopt the cell on match.  The wrappers already feed the
clean prev-run chain (recipe 6), so the fix engages with zero wrapper
changes.  Expected saving: roughly the OFF+hand half of device work
per weekly sweep.
