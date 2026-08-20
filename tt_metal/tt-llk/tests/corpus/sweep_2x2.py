#!/usr/bin/env python3
"""One-command {semantic, hand} x {passes OFF, ON} silicon sweep (HANDOFF §1/§3).

Encodes the silicon protocol as executable policy:
  1. changed-binary classification BEFORE any device job — a byte-identical
     OFF/ON pair is a recorded refusal, never a device run;
  2. paired CRAQ correctness (generic-path libttsim) before silicon;
  3. every device job serialized under BOTH exclusive flocks
     (/tmp/tt-device.lock outer, /tmp/tt-llk-sfpu-silicon.lock inner);
  4. per selector: correctness OFF+ON, then 3 fresh profiler processes per
     leg, alternating OFF/ON, each with a unique RUNNER_TEMP;
  5. raw+post perf CSVs copied while the lock is still held (they are
     overwritten per run); results are parsed only after the lock is released;
  6. hand OFF==ON byte-identity fills both hand cells with one physical run;
  7. per-op evidence: ELFs, .text hashes, build.h, logs, CSVs, compiler
     sha256, SHA256SUMS manifest.

Rows/markers/nodes live in sweep_2x2_ops.tsv, never in this file.  Absent
rows are machine-readable SKIPs.  Metric: post CSV mean(<metric>) at the
row's marker (KERNEL for fire-and-forget replay-launch shapes, TILE_LOOP for
eltwise suites) divided by tile_cnt = cycles/tile (per_tile=0 rows keep the
absolute scoped reading, e.g. the Reduce-SDPA REDUCE_SDPA_BODY pair).

Post-review hardening (PULL_ANALYSIS-20260817 §4):
  * the toolchain pin is the CC1PLUS binary (resolved via g++
    -print-prog-name=cc1plus): the g++ driver is byte-identical across
    cc1plus-only changes, so the driver sha alone is structurally blind and
    is kept only as a secondary check (D6);
  * resume is HASH-MATCHED: a cached device job is reused only when its
    archived .text hash set equals what THIS run's compiler produces for the
    same node/flags (stale-compiler cells re-measure); classify/CRAQ verdicts
    are keyed to the cc1plus (and simulator) sha and re-run on mismatch;
  * weekly per-knob silicon legs run the identical classify -> paired CRAQ ->
    correctness-then-perf pipeline as the main legs (D3);
  * report() is class-aware: baseline rows carry an expected class
    (win/parity/loss/refusal); a prior win row that becomes a byte-identical
    refusal is RED, refusal->changed is a flagged notice (D4;
    selftest_sweep_2x2_report.py proves win->refusal = RED);
  * rows with issue_slot_lb get the HANDOFF §1 issue-slot sanity check:
    a BODY-family reading on a macro-launch shape below the payload's
    issue-slot lower bound is INVALID_MARKER (KERNEL marker required);
  * kind=pinpair rows (Reduce-SDPA) run a paired gen-vs-hand A/B at the
    row's pinned flag set (default profitability gate), keeping the checked
    -in baseline pair and the compiler pin coherent.

Enforcement layer (ledger item 10, 2026-08-17 — the wave-6 violations were
"all one missing enforcement layer, repeated"; these convert the by-memory
rules into mechanical gates):
  * REVIEW RECORD REQUIRED (makes HANDOFF §1(4) unbypassable): a sweep whose
    phases include silicon and that authorizes hardware REFUSES in preflight
    unless <evidence-root>/../REVIEW_RECORD-<cc1plus-12hex>.md exists for the
    CURRENT cc1plus pin, names a reviewer, lists the reviewed commits and the
    gates checked, and quotes the full cc1plus sha256 (pin-match).  The
    record's sha256 is written into preflight.json and MANIFEST.txt so the
    evidence carries which review authorized it.  Template:
    corpus/REVIEW_RECORD_TEMPLATE.md.
  * SIM SHA PINNED (closes ledger 8(e): "any env-supplied libttsim.so
    satisfies the D3 gate"): --sim-bh-sha/--sim-wh-sha carry the reviewed
    libttsim sha256 pins from sweep_2x2.conf; preflight and EVERY phase
    entry (verify_toolchain) re-hash the simulators against them and refuse
    on mismatch, exactly like the compiler pins.
  * MACRO-LAUNCH ROWS REQUIRE issue_slot_lb (closes ledger 8(i)/V3: the
    check was "opt-in per row, not structural" and a headline rode an empty
    lb): the classify phase disassembles each leg's math.elf (objdump is a
    preflight-verified tool) and records a macro_scan verdict; a row whose
    ON binary contains SFPLOADMACRO launches, or fire-and-forget replay
    launches absent from the OFF leg, with an EMPTY issue_slot_lb is RED —
    named in the report with the §1 caveat — never a silent no-op.
  * issue_slot_lb units: the bound is compared against the row's RECORDED
    cell values.  For marker=TILE_LOOP rows the post CSV mean(...) is
    already per-tile (helpers/perf/core.py postprocess_tile_loop divides by
    loop_factor*tile_cnt) and _perf_value divides by tile_cnt again — a
    historical units convention kept for baseline continuity (uniform
    across cells, every booked ratio unaffected).  A TILE_LOOP row's lb
    must therefore be the true per-tile issue-slot bound divided by the
    fixture tile_cnt; each row's note records the raw arithmetic.

Sweep-hardening round 2 (adversarial review, 2026-08-16):
  * the silicon phase trusts NOTHING unkeyed: the BH CRAQ gate re-validates
    every verdict against THIS run's cc1plus+simulator+tt-metal keys, and a
    row without classify evidence keyed to this run is withheld RED (a
    `--phases silicon` resume on an old evidence root can no longer reuse a
    stale-toolchain green or skip the byte-identical refusal logic);
  * cached device jobs re-run when the classify hash reference is absent
    (expected_texts=None never reuses) and are additionally keyed on the
    pytest node id + flags + extra_env (jobkey.json);
  * tt_metal_head keys carry a +dirty.<sha> suffix when tracked tt-llk files
    are modified, so an edited kernel/TSV re-derives evidence;
  * every perf selector requires its own correctness selector (ops-load
    validation, loud failure) — no device perf cell without a correctness
    gate on the same leg;
  * report() acceptance is class- AND magnitude-aware: per-cell ABSOLUTE
    cycle drift vs baseline (uniform slowdowns, hand legs on refusal rows),
    INVALID_METRIC (unparsable metric on a row with baseline history = RED),
    WIN→PARITY = RED (unless --allow-win-to-parity), loss growth beyond
    --red-loss-growth-pct = RED; YELLOW rows show as 'YELLOW', never 'ok';
  * the toolchain the pytest HARNESS uses (tests/sfpi, an untracked
    repointable symlink — test_config.py hardcodes it) is the pinned
    subject: preflight records its realpath, refuses a divergent
    --compiler, and the harness-resolved cc1plus is re-verified against the
    pin at every phase entry.

Batched silicon execution (laneBU, weekly-20260820 forensics):
  684 silicon legs took 517 min = ~45 s wall per leg while each pytest test
  completes in ~1.4 s — >95% of device wall time was per-leg session
  overhead (fresh interpreter + conftest + device open/close per leg, one
  compile invocation per leg, all x3 by PERF_RUNS).  The batched executor
  (default; --serial-legacy reverts) preserves the §1 protocol exactly and
  removes the per-leg spin-up:
  * legs group by (flag-set, extra_env) — arch is constant (BH) for the
    silicon phase; ONE compile-producer pass per group into a shared
    RUNNER_TEMP (seeded from a verified corpus_leg_store build when the
    exact toolchain/flag-set/tree/farm matches — reuse is an optimization,
    never a trust decision);
  * ONE consumer pytest session (--compile-consumer, prebuilt tree, fresh
    process) per (group, repetition r1/r2/r3, CSV-partition) runs its legs'
    nodes inside a single dual-flock acquisition and device session — the 3
    fresh processes per repetition are preserved (determinism requirement);
    OFF/ON alternate at session granularity inside each repetition;
  * correctness-then-perf is preserved: each group's correctness nodes run
    in their own session BEFORE any perf session, and a row whose
    correctness leg fails has every perf leg withheld (exactly the legacy
    STOP semantics);
  * per-leg evidence layout is unchanged: each op still gets its
    silicon/<sel>/<label>-<leg>/ dir with node.txt, flags.txt, jobkey.json,
    log.txt, rc.txt, perf CSVs and TEXT_HASHES.txt — the consumer session's
    outputs are split back per leg (per-node outcomes from the checked-in
    corpus pytest reporter; per-leg .text manifests are the group build
    subset at the leg's classify relpaths, so the classify-vs-device
    hash-match gate keeps its exact strength);
  * perf-CSV integrity: the harness's perf report is per test MODULE, so
    two legs may share a session only if their rows in the combined module
    CSV are separable — same-module legs need distinct mathop tokens (rows
    split by the mathop column); a leg without a token must be its module's
    only leg in that session (whole-module CSV copy).  partition_perf_legs
    encodes this and is selftest-covered.
  PRE-REGISTERED SPEEDUP (weekly-20260820 shape, 759 main-phase legs):
    legacy wall ≈ 759 x 45 s ≈ 570 min.  Batched sessions ≈ per group
    (2 main groups + ~6 pinpair/extra_env groups): 1 producer (no device)
    + 1 corr session + 3 reps x ~4-13 CSV partitions ≈ 90-120 device
    sessions; wall ≈ sessions x ~45 s + legs x ~1.4 s ≈ 85-110 min, i.e.
    a >=5x reduction of the silicon phase with identical evidence.  Knob
    silicon legs (weekly, 200 legs) stay on the serial path this increment.

Batch robustness (laneCH, storm-first-silicon lesson):
  * a FAILED group producer session no longer poisons its group: pytest
    keeps compiling after a failure, so the group tree is hashed anyway
    and only the legs whose classify ELF sets are incomplete in it are
    withheld (their own rc-96 evidence, fail-closed when unprovable);
    every leg whose variants all compiled still runs and books cells
    (previously the pin-12 counted-row ICE failed 2/117 compiles and all
    33 rows of the group were withheld);
  * the classify phase batches too: pending (row, selector, leg) compiles
    group by flag set into chunk producer sessions (>=1 per worker, <=16
    nodes each, sequential inside) with per-node outcome + artefact-file
    attribution from the in-tree pytest plugin; per-leg evidence is
    byte-compatible with the solo path, unprovable legs fall back to solo
    compiles, and SWEEP_CLASSIFY_WORKERS=1 keeps the legacy sequential
    per-leg path;
  * solo classify verdicts that must NOT share a session (knob-attribution
    legs, chunk-unprovable fallbacks) still each get their own isolated
    pytest session — but the sessions run CONCURRENTLY through the same
    worker pool (laneDB): work dirs and per-leg RUNNER_TEMPs are disjoint
    per (row, selector, tag), verdict content is identical serial vs
    concurrent, and SWEEP_CLASSIFY_WORKERS=1 keeps them fully sequential.

Pipeline overhaul (laneDC, owner-ordered, 2026-08-20 — SPEED with every
trust anchor byte-identical in semantics):
  1. CLASSIFY/SILICON PIPELINING (default; --no-pipeline escape): the phase
     barrier is gone — rows admit to silicon in priority-ordered ROLLING
     WAVES as their classify (and CRAQ, when phased) verdicts complete,
     while a background gating thread keeps classifying later waves.  The
     batch planner handles incremental admission by re-planning per wave
     (session dirs silicon-batches/w<i>/); flocked device serialization,
     per-session provenance, keyed gates and refusal logic are the same
     code (_gate_one_row/_gate_rows/_batched_silicon), merely re-scheduled.
  2. ROW PRIORITY SCHEDULING: rows expected to have DIFFERING OFF/ON .text
     (something to measure) classify and measure first; expected
     byte-identical re-baseline rows last; --priority-ops jumps the queue
     entirely.  The expectation is a queue hint (prior verdicts/baseline
     class) — a wrong hint costs only position.  Results stream by value.
  3. CROSS-PIN CELL REUSE: --prev-run takes evidence root(s) (comma list,
     newest first) and the resume prober now probes them — a device leg
     whose jobkey AND archived .text hash set match THIS run's classify
     output adopts the prior silicon instead of re-running (REUSED_FROM.txt
     marker + scoreboard reused_cells: provenance visible, never silent).
     Every SOURCE ROOT is provenance-gated first (wave-12 ledger 19):
     quarantined/contaminated markers refuse, a missing pin record refuses
     fail-closed, craq-gate taint refuses unless this run carries the same
     taint (then propagated to MANIFEST); a foreign recorded pin adopts
     LOUDLY (the .text key protects the number) with the source pin
     recorded, and transitive adoption preserves the full origin chain.
  4. FIRST-CLASS ROW VERDICT STREAMING: <evidence-root>/<op>/
     ROW-VERDICT.json lands the moment a row's cells are assembled —
     cycles per leg, causal/vs_hand %, WIN/PARITY/LOSS band, baseline
     drift — via the same _row_verdict computation REPORT.md aggregates at
     the end (row lines byte-equal).

Typical one-command full sweep:
  python3 tt_metal/tt-llk/tests/corpus/sweep_2x2.py \
    --evidence-root ~/sfpi-uplift/sweep-2x2/evidence-$(date +%Y%m%d) \
    --sim-bh <libttsim-bh> --sim-wh <libttsim-wh> --allow-hardware \
    --baseline tt_metal/tt-llk/tests/corpus/sfpu_device_baseline_p150_v1.tsv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(__file__).resolve().parents[4]
LLK = ROOT / "tt_metal/tt-llk"
TESTS = LLK / "tests"
PYDIR = TESTS / "python_tests"
DEFAULT_CONFIG = HERE / "sweep_2x2_ops.tsv"

# Post-WP8 flag sets (HANDOFF §10 as amended: the -m{no-,}tt-tensix-{analyze,
# emit}-loadmacro flags were REMOVED with the quarantined exact-calendar pass
# and now error on use; the planner ON leg is -mtt-tensix-macro-planner).
OFF_FLAGS = (
    "-mno-tt-tensix-optimize-latency-schedule "
    "-mno-tt-tensix-optimize-dst-iteration-fusion "
    "-mno-tt-tensix-optimize-replay-hoist "
    "-mno-tt-tensix-optimize-invariant-loadi "
    "-mno-tt-tensix-optimize-dst-autoincr "
    "-mno-tt-tensix-optimize-dst-ownership "
    "-mno-tt-tensix-optimize-lut-select "
    "-mno-tt-tensix-optimize-setexp-fold "
    "-mno-tt-tensix-optimize-prgm-const "
    "-mno-tt-tensix-optimize-capture-rotation "
    "-mno-tt-tensix-optimize-ccmask "
    "-mno-tt-tensix-optimize-interlock-schedule "
    "-mno-tt-tensix-optimize-transp-involution "
    "-mno-tt-tensix-optimize-replay-exec-record "
    "-mno-tt-tensix-optimize-drain-schedule "
    "-mno-tt-tensix-macro-planner-residency "
    "-mno-tt-tensix-optimize-const-remat "
    "-mno-tt-tensix-optimize-const-residency "
    "-mno-tt-tensix-optimize-counted-row-formation "
    "-mno-tt-tensix-optimize-crosscall-hoist "
    "-mno-tt-tensix-optimize-crossloop-hoist "
    "-mno-tt-tensix-optimize-init-hoist"
)
ON_FLAGS = (
    "-mtt-tensix-optimize-latency-schedule "
    "-mtt-tensix-optimize-dst-iteration-fusion "
    "-mtt-tensix-optimize-replay-hoist "
    "-mtt-tensix-optimize-invariant-loadi "
    "-mtt-tensix-optimize-dst-autoincr "
    "-mtt-tensix-optimize-dst-ownership "
    "-mtt-tensix-optimize-lut-select "
    "-mtt-tensix-optimize-setexp-fold "
    "-mtt-tensix-macro-planner "
    "-mtt-tensix-macro-planner-replay "
    "-mtt-tensix-optimize-mop-form "
    "-mtt-tensix-optimize-capture-rotation "
    # Pin 10 (Lane BG, fire-witness: ccmask fire set == exactly the exp node,
    # dump-proven; interlock's 18-op pre-existing fire set CRAQ'd 42/42).
    # COUPLED with the exp semantic-source restructure merge: that source
    # without ccmask compiles to 28-slot rows.
    "-mtt-tensix-optimize-ccmask "
    "-mtt-tensix-optimize-interlock-schedule "
    # Lane BF (pin 10): transp-involution (SFPTRANSP gather fusion +
    # Dst-park elision; fire witness = welford dump, parks eliminated) and
    # replay-exec-record (exec-while-record capture; fire witness =
    # sigmoidappx 33->32 launch shape).
    "-mtt-tensix-optimize-transp-involution "
    "-mtt-tensix-optimize-replay-exec-record "
    # Lane BL (COUPLED to the next pin cycle: requires a cc1plus built
    # from sfpi-gcc agent/mopcfg-derivation or its merge — pin 10
    # refuses this flag byte-identically, so it is inert until the
    # re-pin).  M3/prgm-const returns to the ON set with the mop_cfg
    # template-effect DERIVATION (Lane BC design, implemented Lane BL):
    # the compiler PROVES the MOP run's PRGM/LaneConfig effects from
    # the TU's own template-programming stores — no markers, no
    # trusted annotations, stock harness.  Fire witness (dump-proven,
    # laneBL-evidence-20260818): the exp perf node's MATH_ISOLATE
    # math.elf — "allocated PRGM L14 for invariant immediate
    # 0x42fe0000", capture 17->16, SFPMUL+SFPADDI fused to
    # SFPMAD(...,L14); sdpa fires with the identical-immediate dedup
    # (1 register + dominated reuses, was L12+L13+L14); exp corr CRAQ
    # 7574->5630 and sdpa num_tiles:2 CRAQ 4886->4081 sim-cycles, both
    # bit-exact PASS on the pinned BH sim.
    "-mtt-tensix-optimize-prgm-const "
    # Lane AY (next pin): drain-aware boundary placement (WP13) -- per-
    # boundary drain proofs from the derived descriptor calendars; fire
    # witness = the wired minmax-max/min rows (planner dump line
    # "Macro-planner drain-schedule: run-boundary drain elided", 3 per
    # kernel; SFPNOP 12->3 per tile).  Refusals keep the full drain
    # byte-identically; unarymaxmin/sdpa proven unmoved.
    "-mtt-tensix-optimize-drain-schedule "
    # Lane BN (pin 11): descriptor-program residency — outward span extension
    # + content-equality dedup; fire witness = the where impl-1 rows (dump
    # "resident=elided", descriptor block once per kernel; ON inventory =
    # exactly the 5 where impl-1 math.elfs, CRAQ 25/25, laneBN evidence).
    "-mtt-tensix-macro-planner-residency "
    # Lane BS (pin 11): value const-remat + PRGM const-residency; fire
    # witnesses = the ICE-repro-now-compiles dg test + the 12-row designed
    # fire (where x7, sdpa-exp x2, binop-scalar x3; CRAQ 142/142, laneBS
    # evidence).  The spill-diag tier is unconditional (no flag).
    "-mtt-tensix-optimize-const-remat "
    "-mtt-tensix-optimize-const-residency "
    # Lane BM (next pin): counted-row parameterized formation (fire
    # witness = the welford perf body dump: one 5-member parameterized
    # record, 7 clones, body issue census 234 -> 218; dg fire twins in
    # sfpi-gcc counted-row-formation-*).  MERGE-ORDER: this line lands
    # only with the pin cycle whose toolchain accepts the flag.
    "-mtt-tensix-optimize-counted-row-formation "
    # QUARANTINE LIFTED AT PIN 14 (wave-9 verdict 4adac11aac1 / HANDOFF
    # item 13 quarantined crosscall-hoist, crossloop-hoist and init-hoist
    # until the shared TU-census rooting hole was fixed).  ALL THREE LIFT
    # CONDITIONS ARE MET in the pin-14 union (sfpi-gcc staging/pin14
    # 3ca94518817):
    #   (a) the census roots externally-visible/entry symbols or FAILS
    #       CLOSED — lane CG census-rooting fix 8dee5e84029 (anchor-model
    #       rooting _start>main>publics; fail-closed unrooted/expanded/
    #       entry-root paths; foreign-contract read guard; honest
    #       zero-trip argument);
    #   (b) the init-hoist ICE/segfault on non-trivially-rooted censuses
    #       is fixed (same lane CG commit);
    #   (c) the zero-trip prose matches behavior — verified against lane
    #       CJ's production-shaped census twins for the crossloop and
    #       init-hoist MOP families (e0754714a5b) with a clean lane-CJ
    #       audit verdict, and the wave-10 review verified every wrong-
    #       code item closed on cold builds (zero violations).
    # The witness rows moved back from _QUARANTINED_FIRE_WITNESSES to
    # _REVIEWED_FIRE_WITNESSES in the same commit (conf R9/R10), and this
    # ON set is INERT until the pin-14 ceremony finalizes the toolchain
    # pin: the conf carries RED-on-placeholder pin prose, and both sweep
    # wrappers refuse at conf-lint before anything runs.
    # Lane BQ (pin 12): cross-call prefix hoist -- the noinline
    # callee's call-invariant pinned-LREG coefficient prefix moves to
    # every proven caller's tile-loop preheader (BI's increment (a)).
    # Fire witness: sigmoid-tree (both perf and correctness TUs),
    # dump-proven "hoisted 6 contract materializations" + disasm: the
    # 12 coefficient SFPLOADI leave the callee for the caller's
    # preheader; with replay-exec-record the per-tile dead prefix is
    # 17 -> 4 pushes.  Pre-registered prediction (laneBQ evidence
    # PREDICTION.md): sigmoid-tree sem_on 27.7-27.8 = parity vs hand.
    "-mtt-tensix-optimize-crosscall-hoist "
    # Lane CD (pin 13): cross-loop hoist -- loop-invariant SFPU
    # immediate materializations (and the programmable-constant
    # programming points) lift across ENCLOSING loops whose bodies
    # deliver only audited-inert words (raw .ttinsn constants,
    # instruction-FIFO stores incl. PHI-joined compositions, MOPs with
    # a proven TU template census); every unaudited class refuses by
    # name (crossloop-*).  Fire witness: the exp perf node's
    # MATH_ISOLATE math.elf -- "crossloop-hoist: hoisted across loop"
    # x12 (4 loads x 3 loop levels) + "placement lifted from entry bb"
    # (prgm-const L14 programming once per kernel); per-tile Tensix
    # word count 68 -> 58 = hand parity; exp corr CRAQ 5630 -> 5589
    # sim-cycles bit-exact PASS on the pinned BH sim.
    "-mtt-tensix-optimize-crossloop-hoist "
    # Lane CA (pin 13): D2 cross-call invariant-init hoist — the noinline
    # per-tile callee's idempotent formation init prefix (descriptor
    # program + owned SETC16 + all-lanes enable, 17 issue slots on the
    # minmax shape) programs once in the caller's tile-block loop
    # preheader (stage 2 under the owned-row value-equality proof).
    # Fire witness (dump-proven, laneCA-evidence-20260819): the minmax
    # perf/corr TUs — dump line: Macro-planner init-hoist: stage=2 init
    # contract hoisted to caller loop preheader; the callee is pure payload.
    # Lane CA also completes the drain-schedule flag with loop-backedge
    # elision + exit compensation (where in-body 3-NOP tail x 4 trips ->
    # one exit block; witness dump line: loop-backedge drain elided, on
    # the where TUs).  Measured (lane CA, 3 fresh procs, BH p150): minmax
    # TILE_LOOP/tile 18.727 -> 16.72 = -5.1% vs hand 17.628 (WIN);
    # where TTNN_WHERE_BODY 167.5 -> 154.5 = -2.9% vs hand 159.17 (WIN).
    "-mtt-tensix-optimize-init-hoist"
    # M3/prgm-const is NOT in the ON set (un-shipped after pin 9's nightly):
    # its only engagement channel was the trusted TTREGION source markers in
    # the LLK headers, and trusted source annotation of the consumed library
    # is rejected at the design level (LLK-pristine rule, conf R7).  The flag
    # returns when the compiler PROVES region effects algorithmically
    # (mop_cfg template-programming dataflow derivation — Lane BC).
)
REMOVED_FLAGS = ("-mtt-tensix-emit-loadmacro", "-mtt-tensix-analyze-loadmacro")
# Weekly per-knob attribution: OFF set plus exactly one positive knob.
KNOBS = {
    "latency-schedule": "-mtt-tensix-optimize-latency-schedule",
    "dst-iteration-fusion": "-mtt-tensix-optimize-dst-iteration-fusion",
    "replay-hoist": "-mtt-tensix-optimize-replay-hoist",
    "invariant-loadi": "-mtt-tensix-optimize-invariant-loadi",
    "dst-autoincr": "-mtt-tensix-optimize-dst-autoincr",
    "macro-planner": "-mtt-tensix-macro-planner",
    "dst-ownership": "-mtt-tensix-optimize-dst-ownership",
    "lut-select": "-mtt-tensix-optimize-lut-select",
    "setexp-fold": "-mtt-tensix-optimize-setexp-fold",
    "planner-replay": "-mtt-tensix-macro-planner-replay",
    "mop-form": "-mtt-tensix-optimize-mop-form",
    "capture-rotation": "-mtt-tensix-optimize-capture-rotation",
    "ccmask": "-mtt-tensix-optimize-ccmask",
    "interlock-schedule": "-mtt-tensix-optimize-interlock-schedule",
    "transp-involution": "-mtt-tensix-optimize-transp-involution",
    "replay-exec-record": "-mtt-tensix-optimize-replay-exec-record",
    "prgm-const": "-mtt-tensix-optimize-prgm-const",
    "drain-schedule": "-mtt-tensix-optimize-drain-schedule",
    "planner-residency": "-mtt-tensix-macro-planner-residency",
    "const-remat": "-mtt-tensix-optimize-const-remat",
    "const-residency": "-mtt-tensix-optimize-const-residency",
    "counted-row-formation": "-mtt-tensix-optimize-counted-row-formation",
    # crosscall-hoist / crossloop-hoist RESTORED at the pin-14 quarantine
    # lift (lane CG census fix 8dee5e84029; see the ON_FLAGS LIFTED
    # block above for the full citation chain).
    "crosscall-hoist": "-mtt-tensix-optimize-crosscall-hoist",
    "crossloop-hoist": "-mtt-tensix-optimize-crossloop-hoist",
    # ---- pin-14 NEW default-off flags: knob legs ONLY.  Deliberately
    # NOT in the reviewed ON set — the weekly measures each solo on its
    # target rows (attribution on every changed row; silicon legs on the
    # knob-silicon row list).  Promotion into the ON set is a separate
    # reviewed step carrying its own R9 union fire witness.
    # CU (int-peephole-harvest): v_if(v<0){r=0-v} -> one integer SFPABS
    # ccmask-fold; 2^32-exhaustive EQUAL, QSR fail-closed by name.
    # Target row: absint32 (fresh row hand-EXACT 3-word stream).
    "int-abs": "-mtt-tensix-optimize-int-abs",
    # CV (loop-infra-attack): typed-census unroll request == the
    # production `#pragma GCC unroll 8` shape; fires the always-on replay
    # former to exec-record delivery.  Target rows: hardshrink-fresh /
    # hardsigmoid-fresh / softsign-fresh (CRAQ 5/5 bit-exact favorable).
    "replay-loop-unroll": "-mtt-tensix-optimize-replay-loop-unroll",
    # CY (lut-select-leaf-arity): certified non-affine LUT leaf classes +
    # below-arity duplication.  Parent lut-select is Init(1) but the knob
    # leg is built on OFF_FLAGS (parent forced off), so the leg carries
    # parent+leaf-ext; the leaf-ext-only delta reads against the plain
    # lut-select knob leg.  LICENSED-LEG-PENDING-OWNER, FAIL-CLOSED: the
    # fire additionally needs per-TU -ffinite-math-only (an UNSIGNED
    # owner decision, HANDOFF #6) which this knob leg does NOT pass —
    # until the owner signs, the leg is an honest no-fire A/B.
    # Target rows: tanhderivlut-fresh (18->6 word sfplutfp32 loop,
    # +215% -> +3.6% under the licensed 2x2 in lane CY evidence).
    "lut-select-leaf-ext": "-mtt-tensix-optimize-lut-select "
    "-mtt-tensix-optimize-lut-select-leaf-ext",
    # CN (representation-propagation): bit-involution pair cancellation
    # on audited choose-webs; corpus 0-changed at the CN gate (fire
    # evidence lives in the dg twins) — the knob leg surfaces any pin-14
    # union fire in attribution rather than assuming one.
    "repr-prop": "-mtt-tensix-optimize-repr-prop",
    # CK (typecast-planner-effects): NO new flag — it extends the
    # existing replay-hoist to planner-emitted rvtt_sfploadmacro_int
    # launches (prices-but-refuses at the current calibration).  Its A/B
    # is the existing replay-hoist knob leg on the typecast row (added
    # to the knob-silicon row list in sweep_2x2.conf).
}
HARNESS_TOOLCHAIN = TESTS / "sfpi"  # untracked symlink the harness hardcodes
DEVICE_LOCK = "/tmp/tt-device.lock"
SILICON_LOCK = "/tmp/tt-llk-sfpu-silicon.lock"
CHIP = {"bh": "blackhole", "wh": "wormhole"}
SELECTORS = ("sem-corr", "sem-perf", "hand-corr", "hand-perf")
PERF_RUNS = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- enforcement-layer gate primitives (selftest-covered) ----
# Kept as module-level pure(ish) functions so selftest_enforcement_gates.py
# can drive the REAL gate logic without a toolchain or device.


def check_review_record(record_path, cc1plus_sha256):
    """REVIEW_RECORD gate (HANDOFF §1(4) as code).

    Returns (ok, detail).  A silicon-authorized sweep refuses unless the
    record at <evidence-root>/../REVIEW_RECORD-<cc1plus-12hex>.md exists,
    quotes the FULL cc1plus sha256 (pin-match — a record for another build
    never authorizes this one), names a reviewer, and carries the reviewed
    commits and gates-checked sections.  Content honesty stays human; the
    gate makes the record's EXISTENCE and pin-binding mechanical.
    """
    record_path = pathlib.Path(record_path)
    if not record_path.is_file():
        return False, f"missing review record {record_path}"
    text = record_path.read_text(errors="replace")
    if cc1plus_sha256 not in text:
        return False, (
            f"review record {record_path} does not quote the full pinned "
            f"cc1plus sha256 {cc1plus_sha256} — pin-match failed (a record "
            "minted for another build does not authorize this one)"
        )
    checks = (
        (r"(?mi)^\s*(?:[-*]\s*)?Reviewer\s*:\s*\S", "a non-empty 'Reviewer:' line"),
        (r"(?mi)^##\s*Reviewed\b", "a '## Reviewed ...' commits/branches section"),
        (r"(?mi)^##\s*Gates\b", "a '## Gates checked' section"),
    )
    missing = [what for pat, what in checks if not re.search(pat, text)]
    if missing:
        return False, (
            f"review record {record_path} is malformed — missing "
            + "; ".join(missing)
            + " (see corpus/REVIEW_RECORD_TEMPLATE.md)"
        )
    return True, "ok"


# Fire-and-forget replay launch = TT_OP_REPLAY with execute_while_loading=0
# and load_mode=0 (a pure launch of previously recorded slots).  Record-form
# replays (…,1,1 / …,0,1) are the drain-synchronous record path and are NOT
# launches.
_SCAN_PATTERNS = {
    "sfploadmacro": re.compile(r"\bsfploadmacro\b"),
    "replay_launch": re.compile(r"\bttreplay\s+\d+,\s*\d+,\s*0,\s*0\b"),
}


def scan_disasm_text(text):
    """Mnemonic census of one disassembly relevant to macro-launch detection."""
    return {name: len(pat.findall(text)) for name, pat in _SCAN_PATTERNS.items()}


def classify_macro_launch(on_counts, off_counts=None):
    """Macro-launch classification of an ON leg census vs its OFF leg.

    Returns 'MACRO_LAUNCH' (SFPLOADMACRO launches present),
    'REPLAY_LAUNCH' (fire-and-forget replay launches beyond the OFF leg's —
    source-level lltt::record replays appear in both legs and do not count),
    or None.  With no OFF census (single-leg pinpair rows) only the
    SFPLOADMACRO criterion applies; replay counts stay informational.
    """
    if on_counts.get("sfploadmacro", 0) > 0:
        return "MACRO_LAUNCH"
    if off_counts is not None and on_counts.get("replay_launch", 0) > off_counts.get(
        "replay_launch", 0
    ):
        return "REPLAY_LAUNCH"
    return None


def macro_lb_red(op, marker, issue_slot_lb, macro_scan):
    """RED message for a macro-launch row with an EMPTY issue_slot_lb, else
    None.  Ledger 8(i)/wave-6 V3: an empty lb silently no-opped the §1
    issue-slot sanity check under exactly the shapes that need it most."""
    if not macro_scan or issue_slot_lb is not None:
        return None
    cls = macro_scan.get("classification")
    if not cls:
        return None
    return (
        f"{op}: {cls} row measured at marker {marker} with EMPTY "
        f"issue_slot_lb (ON binary: {macro_scan.get('sfploadmacro_on', 0)} "
        f"SFPLOADMACRO launches, {macro_scan.get('replay_launch_on', 0)} "
        "fire-and-forget replay launches) — HANDOFF §1 metric caveat: a "
        "BODY-family reading on a fire-and-forget shape is INVALID below the "
        "payload's issue-slot lower bound, and without issue_slot_lb that "
        "check cannot run; populate issue_slot_lb (units: the row's recorded "
        "cell units, see sweep_2x2_ops.tsv header) or move the row to the "
        "drain-inclusive KERNEL marker"
    )


def craq_gate_taint(skipped):
    """One-line CRAQ-gate disposition for the evidence trail (ledger 8(f):
    --skip-craq-gate used to leave NO taint marker in evidence dirs).  The
    line is written verbatim into preflight MANIFEST.txt and the REPORT.md
    header so a skipped gate can never be mistaken for a green one."""
    if skipped:
        return (
            "CRAQ gate: SKIPPED (--skip-craq-gate) — no paired-CRAQ evidence "
            "in this run; cells rest on device-golden gating alone"
        )
    return "CRAQ gate: ACTIVE (paired CRAQ required before silicon legs)"


def load_config(path):
    with path.open() as f:
        rows = list(
            csv.DictReader((x for x in f if not x.startswith("#")), delimiter="\t")
        )
    for row in rows:
        row["nodes"] = {
            sel: (row.get(sel.replace("-", "_")) or "").strip() for sel in SELECTORS
        }
        # Optional columns (sweep-2x2-ops-version 2); absent = v1 defaults.
        row["metric"] = (row.get("metric") or "").strip() or "MATH_ISOLATE"
        row["per_tile"] = (row.get("per_tile") or "1").strip() != "0"
        lb = (row.get("issue_slot_lb") or "").strip()
        row["issue_slot_lb"] = float(lb) if lb else None
        row["pin_flags"] = (row.get("pin_flags") or "").strip()
        env = (row.get("extra_env") or "").strip()
        row["extra_env"] = dict(kv.split("=", 1) for kv in env.split(";") if kv)
        # Optional per-selector env (corpus-expansion lane): sem_extra_env /
        # hand_extra_env apply on top of extra_env to the sem-*/hand-*
        # selectors only.  This expresses same-node A/Bs whose axis is a
        # harness compile define (e.g. mul_int: hand = production
        # SFPLOADMACRO path, "generated" = the same header's
        # -DDISABLE_SFPLOADMACRO plain-delivery arm selected via
        # TT_METAL_DISABLE_SFPLOADMACRO=1) without forking the runner.
        row["sel_extra_env"] = {}
        for side in ("sem", "hand"):
            v = (row.get(f"{side}_extra_env") or "").strip()
            row["sel_extra_env"][side] = dict(
                kv.split("=", 1) for kv in v.split(";") if kv
            )
        # Optional schedule column (device-time budget split): 'nightly'
        # (default) or 'weekly'.  Data, not code forks: the nightly wrapper
        # passes --schedule nightly; weekly/manual sweeps run every row.
        row["schedule"] = (row.get("schedule") or "").strip() or "nightly"
        if row["schedule"] not in ("nightly", "weekly"):
            sys.exit(
                f"config row {row['op']}: schedule must be 'nightly' or "
                f"'weekly' (got '{row['schedule']}')"
            )
        if row["kind"] == "pinpair" and not row["pin_flags"]:
            sys.exit(f"config row {row['op']}: kind=pinpair requires pin_flags")
        # Sweep-hardening 2: a perf leg without its own correctness leg
        # produces device cycles from a kernel nothing verified — a broken
        # hand kernel would silently keep feeding vs_hand_pct GREEN.  Loud
        # failure at ops-load; withhold the perf node until a corr node lands.
        if row["kind"] != "skip":
            for perf_sel, corr_sel in (
                ("sem-perf", "sem-corr"),
                ("hand-perf", "hand-corr"),
            ):
                if row["nodes"][perf_sel] and not row["nodes"][corr_sel]:
                    sys.exit(
                        f"config row {row['op']}: {perf_sel} has a node but "
                        f"{corr_sel} is empty — every device perf leg requires "
                        "its own correctness node (perf cycles from an "
                        "unverified kernel are not evidence); add the corr "
                        "node or withhold the perf leg in the row note"
                    )
    return rows


def row_env(row, sel):
    """Effective extra env for one selector: row extra_env overlaid with the
    selector side's sem_extra_env/hand_extra_env (see load_config)."""
    env = dict(row.get("extra_env") or {})
    env.update((row.get("sel_extra_env") or {}).get(sel.split("-")[0], {}))
    return env


def row_scope(row):
    """Baseline/scoreboard scope string for a config row (or stored result)."""
    if row["kind"] == "pinpair":
        return row["marker"]
    return f"{row['marker']}_{row.get('metric', 'MATH_ISOLATE')}_PER_TILE"


# Perf-cell naming per row kind.  pinpair rows keep the checked-in baseline's
# native selectors (e.g. Reduce-SDPA 'generated'/'handwritten_replay').
PINPAIR_CELLS = {"sem-perf": "generated", "hand-perf": "handwritten_replay"}


def cell_selector(r, cell):
    """Baseline/scoreboard selector for a result's cell."""
    if r["kind"] == "pinpair":
        return cell
    return f"{r['op']}:{cell}"


def partition_perf_legs(specs):
    """Bin-pack perf leg SPECS into consumer sessions with splittable CSVs.

    The harness's perf report is per test MODULE (module-scoped fixture,
    one combined CSV per module per session), and the CSV rows carry the
    test parameters but NOT the test id — the only reliable row-level
    discriminator between two ops of the same module is the `mathop`
    column.  A session is therefore CSV-splittable iff, per test file,
    EITHER it holds exactly one leg of that file (whole-module CSV copy)
    OR every leg of that file carries a distinct `mathop:` token in its
    node id (row filter by the mathop column).  Same-file legs that differ
    only in an axis invisible to the CSV (e.g. fresh_cpp_impl sem vs hand)
    must never share a session — the combiner would collapse their rows.

    specs: dicts with keys file, mathop (token or None), op, sel, leg.
    Deterministic (sorted greedy first-fit); selftest-covered."""
    bins = []
    for spec in sorted(
        specs,
        key=lambda s: (s["file"], s["mathop"] or "", s["op"], s["sel"], s["leg"]),
    ):
        placed = False
        for b in bins:
            same = [x for x in b if x["file"] == spec["file"]]
            if not same:
                b.append(spec)
                placed = True
                break
            if (
                spec["mathop"]
                and all(x["mathop"] for x in same)
                and spec["mathop"] not in {x["mathop"] for x in same}
            ):
                b.append(spec)
                placed = True
                break
        if not placed:
            bins.append([spec])
    return bins


class Sweep:
    # Class default keeps object.__new__-driven selftests (which bypass
    # __init__) on the legacy serial path; __init__ sets the real mode.
    exec_mode = "serial"

    def __init__(self, args):
        self.a = args
        self.ev = args.evidence_root.resolve()
        self.compiler = (
            args.compiler or TESTS / "sfpi/compiler/bin/riscv-tt-elf-g++"
        ).resolve()
        self.objcopy = self.compiler.with_name("riscv-tt-elf-objcopy")
        self.objdump = self.compiler.with_name("riscv-tt-elf-objdump")
        self.python = self._find_python(args.venv)
        self.rows = [
            r for r in load_config(args.config) if not args.ops or r["op"] in args.ops
        ]
        if args.ops:
            missing = set(args.ops) - {r["op"] for r in self.rows}
            if missing:
                sys.exit(f"unknown ops in --ops: {','.join(sorted(missing))}")
        # Schedule filter (device-time budget, data-driven): --schedule
        # nightly runs only schedule=nightly rows; --schedule weekly (and no
        # --schedule at all) runs EVERY row — the weekly sweep is the full
        # set, the nightly is the budgeted subset.  --ops overrides the
        # filter (an explicit op list is an explicit intent).
        # Batched silicon executor is the default (laneBU: ~45s/leg session
        # overhead vs ~1.4s test time); --serial-legacy is the logged escape
        # back to one pytest session per leg.
        self.exec_mode = (
            "serial" if getattr(args, "serial_legacy", False) else "batched"
        )
        if self.exec_mode == "serial":
            print(
                "sweep: --serial-legacy — batched silicon DISABLED; every "
                "device leg pays its own pytest session (the pre-batching "
                "~45s/leg overhead); cells are keyed mode=serial and never "
                "mix with batched cells"
            )
        self.deferred = []
        if getattr(args, "schedule", None) == "nightly" and not args.ops:
            self.deferred = [r for r in self.rows if r["schedule"] != "nightly"]
            self.rows = [r for r in self.rows if r["schedule"] == "nightly"]
            if self.deferred:
                print(
                    "schedule filter: nightly run defers "
                    f"{len(self.deferred)} weekly rows: "
                    + ",".join(r["op"] for r in self.deferred)
                )
        self.reds = []
        self.reused = []  # cross-run adopted device cells (provenance)
        # --priority-ops sanity: a typo'd op must fail loudly, never
        # silently deprioritize everything.  Ops deferred by the schedule
        # filter stay deferred (priority reorders, it never resurrects).
        prio = getattr(args, "priority_ops", None) or []
        known = {r["op"] for r in self.rows} | {r["op"] for r in self.deferred}
        missing = set(prio) - known
        if missing:
            sys.exit(f"unknown ops in --priority-ops: {','.join(sorted(missing))}")
        live = {r["op"] for r in self.rows}
        for op in prio:
            if op not in live:
                print(
                    f"priority: op {op} is schedule-deferred this run — "
                    "priority reorders the queue, it never resurrects a "
                    "deferred row"
                )

    # ---------------- row priority scheduling ----------------
    def _expected_identical(self, row, base_classes):
        """Best-effort hint that this row's OFF/ON legs will be
        byte-identical (nothing to measure — a re-baseline row).  Sources,
        in order: a classify verdict already on disk for this run root or
        any --prev-run root (read UNKEYED — this is a QUEUE hint, never a
        trust decision: the keyed classify verdict still decides every
        refusal and every device cell exactly as before), then the
        baseline's expected class (refusal == byte-identical history).  A
        wrong hint costs only queue position."""
        if row["kind"] == "pinpair":
            return False  # single pinned leg: always a measured A/B
        sel = "sem-perf" if row["nodes"].get("sem-perf") else "sem-corr"
        if not row["nodes"].get(sel):
            return False
        for root in [self.ev] + self._prev_roots():
            vf = pathlib.Path(root) / row["op"] / "classify" / sel / "verdict.json"
            if not vf.is_file():
                continue
            try:
                v = json.loads(vf.read_text())
            except ValueError:
                continue
            if v.get("status") == "OK" and v.get("all") in ("IDENTICAL", "CHANGED"):
                return v["all"] == "IDENTICAL"
        return base_classes.get((row["corpus_id"], row_scope(row))) == "refusal"

    def _order_rows(self):
        """ROW PRIORITY SCHEDULING (owner order, pipeline overhaul): rows
        with something to MEASURE classify and reach silicon first —
        results stream by value, not alphabetically/config order.

        Order: (0) --priority-ops, in the order given (they jump the queue
        entirely); (1) rows expected to have DIFFERING OFF/ON .text hashes
        (or unknown — never guessed identical); (2) rows expected
        byte-identical (re-baseline rows) last.  Stable within each tier
        (config order).  The expectation is a hint from prior classify
        verdicts/baseline class; actual verdicts are computed exactly as
        before and correctness is unaffected by a wrong hint."""
        prio = list(getattr(self.a, "priority_ops", None) or [])
        baseline, base_classes = self._load_baseline(getattr(self.a, "baseline", None))

        def key(i):
            row = self.rows[i]
            if row["op"] in prio:
                return (0, prio.index(row["op"]), i)
            if self._expected_identical(row, base_classes):
                return (2, 0, i)
            return (1, 0, i)

        order = sorted(range(len(self.rows)), key=key)
        self.rows = [self.rows[i] for i in order]
        tiers = {(0): "priority", (1): "measure", (2): "re-baseline"}
        desc = ",".join(
            f"{r['op']}[{tiers[key(i)[0]]}]" for i, r in enumerate(self.rows)
        )
        if self.rows:
            print(f"row priority order: {desc}")

    @staticmethod
    def _find_python(venv):
        candidates = (
            [venv]
            if venv
            else [TESTS / ".venv-laneE", TESTS / ".venv", PYDIR / ".venv"]
        )
        for c in candidates:
            if c and (pathlib.Path(c) / "bin/python").is_file():
                return pathlib.Path(c) / "bin/python"
        sys.exit(
            "no tt-llk virtualenv found (looked for tests/.venv-laneE, tests/.venv, "
            "tests/python_tests/.venv); pass --venv"
        )

    # ---------------- preflight ----------------
    @staticmethod
    def _pin_value(pin, what):
        """A pin must be a FULL 64-hex sha256.  The previous prefix
        acceptance (startswith) meant a 1-char env leak 'pinned' essentially
        nothing (adversarial finding sweep_2x2.conf:31)."""
        pin = pin.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", pin):
            sys.exit(
                f"{what} pin '{pin}' is not a full 64-hex sha256 — refusing "
                "to sweep (prefix pins accept almost anything; pass the "
                "complete sha256 from sweep_2x2.conf)"
            )
        return pin

    def _resolve_cc1plus(self):
        """cc1plus resolved through the driver (the binary that compiles)."""
        cc1 = subprocess.run(
            [str(self.compiler), "-print-prog-name=cc1plus"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not cc1 or not pathlib.Path(cc1).is_file():
            sys.exit(f"cannot resolve cc1plus via {self.compiler} (got '{cc1}')")
        return cc1, sha256(pathlib.Path(cc1))

    def verify_toolchain(self, phase):
        """Re-verify the harness toolchain identity at a phase entry.

        tests/sfpi is an UNTRACKED, repointable symlink and the pytest
        harness hardcodes it (test_config.py setup_paths) — a mid-run
        repoint would silently measure with an unpinned compiler while the
        manifest swears otherwise (adversarial finding sweep_2x2.py:160).
        """
        real = str(HARNESS_TOOLCHAIN.resolve())
        cc1, cc1_sha = self._resolve_cc1plus()
        if (
            real != self.info["harness_toolchain_realpath"]
            or cc1_sha != self.info["cc1plus_sha256"]
        ):
            sys.exit(
                f"TOOLCHAIN CHANGED MID-RUN (phase '{phase}'): tests/sfpi now "
                f"resolves to {real} with cc1plus {cc1_sha} at {cc1}; "
                f"preflight recorded {self.info['harness_toolchain_realpath']} "
                f"with cc1plus {self.info['cc1plus_sha256']} — refusing to "
                "continue (evidence already produced is keyed to the "
                "preflight identity)"
            )
        # Sim sha pin: re-verified at EVERY phase entry, not just preflight —
        # a mid-run libttsim swap must never let an unpinned oracle open the
        # CRAQ gate (ledger 8(e)).
        self._verify_sim_pins(phase)

    def _verify_sim_pins(self, phase):
        """Hash each provided simulator against its reviewed pin (and the
        preflight-recorded identity); refuse on any mismatch."""
        for arch in ("bh", "wh"):
            sim = getattr(self.a, f"sim_{arch}")
            pin = getattr(self.a, f"sim_{arch}_sha", None)
            if not pin:
                continue
            pin = self._pin_value(pin, f"libttsim {arch} (--sim-{arch}-sha)")
            if not sim or not sim.is_file():
                sys.exit(
                    f"SIM PIN {arch} set ({pin}) but no simulator file at "
                    f"'{sim}' (phase '{phase}') — a pinned CRAQ oracle that "
                    "silently degrades to SKIP_NO_SIMULATOR would withhold "
                    "silicon rows one by one instead of failing loudly; "
                    "build/point the pinned libttsim or drop the pin through "
                    "review"
                )
            found = sha256(sim)
            if found != pin:
                sys.exit(
                    f"SIM SHA MISMATCH ({arch}, phase '{phase}'): pinned "
                    f"{pin}, found {found} at {sim} — refusing (the CRAQ "
                    "oracle is pinned like the compiler: any env-supplied "
                    "libttsim.so must hash to the reviewed value; re-pin "
                    "through review, never through the environment)"
                )
            recorded = (
                self.info.get(f"sim_{arch}_sha256") if hasattr(self, "info") else None
            )
            if recorded and recorded != found:
                sys.exit(
                    f"SIMULATOR CHANGED MID-RUN ({arch}, phase '{phase}'): "
                    f"preflight recorded {recorded}, now {found} at {sim} — "
                    "refusing (evidence already produced is keyed to the "
                    "preflight identity)"
                )

    def preflight(self):
        self.ev.mkdir(parents=True, exist_ok=True)
        info = {
            "compiler": str(self.compiler),
            "off_flags": OFF_FLAGS,
            "on_flags": ON_FLAGS,
            "config": str(self.a.config),
            "evidence_root": str(self.ev),
        }
        if not self.compiler.is_file():
            sys.exit(f"missing compiler {self.compiler}")
        # The pytest harness HARDCODES its toolchain to the tests/sfpi
        # symlink (test_config.py setup_paths: TOOL_PATH = LLK_ROOT /
        # 'tests/sfpi/compiler/bin') and the sweep passes only flags/env —
        # never a compiler path.  --compiler therefore controls what
        # preflight HASHES, not what BUILDS: a divergent --compiler would
        # verify one binary and measure with another (adversarial finding
        # sweep_2x2.py:160).  Enforce that the pinned subject IS the harness
        # toolchain, and record the symlink's realpath as evidence.
        harness_gxx = (HARNESS_TOOLCHAIN / "compiler/bin/riscv-tt-elf-g++").resolve()
        info["harness_toolchain_symlink"] = str(HARNESS_TOOLCHAIN)
        info["harness_toolchain_realpath"] = (
            str(HARNESS_TOOLCHAIN.resolve()) if HARNESS_TOOLCHAIN.exists() else ""
        )
        if self.compiler != harness_gxx:
            sys.exit(
                f"--compiler {self.compiler} is NOT the harness toolchain "
                f"{harness_gxx} (tests/sfpi resolves to "
                f"{info['harness_toolchain_realpath'] or 'MISSING'}): the "
                "pytest harness hardcodes tests/sfpi/compiler/bin "
                "(test_config.py), so every build would use the harness "
                "toolchain while preflight verified a different binary — "
                "repoint tests/sfpi at the pinned build or drop --compiler"
            )
        # SECONDARY pin: the g++ driver.  Historically byte-identical across
        # cc1plus-only rebuilds (structurally blind, D6) — it can catch a
        # wrong toolchain layout but never a compiler-proper change.  Full
        # sha equality required (no prefixes).
        info["compiler_sha256"] = sha256(self.compiler)
        if self.a.compiler_sha and (
            self._pin_value(self.a.compiler_sha, "driver (--compiler-sha)")
            != info["compiler_sha256"]
        ):
            sys.exit(
                f"DRIVER SHA MISMATCH: pinned {self.a.compiler_sha}, "
                f"found {info['compiler_sha256']} — refusing to sweep"
            )
        # PRIMARY pin: cc1plus (the compiler proper), resolved through the
        # driver itself so the pin follows whatever binary actually compiles.
        cc1, cc1_sha = self._resolve_cc1plus()
        info["cc1plus"] = cc1
        info["cc1plus_sha256"] = cc1_sha
        if self.a.cc1plus_sha and (
            self._pin_value(self.a.cc1plus_sha, "cc1plus (--cc1plus-sha)")
            != info["cc1plus_sha256"]
        ):
            sys.exit(
                "CC1PLUS SHA MISMATCH (primary toolchain pin): pinned "
                f"{self.a.cc1plus_sha}, found {info['cc1plus_sha256']} at {cc1} "
                "— refusing to sweep (the g++ driver sha alone cannot detect "
                "cc1plus-only changes; rebuild/point the pinned toolchain or "
                "update the pin through review)"
            )
        ver = subprocess.run(
            [str(self.compiler), "--version"], capture_output=True, text=True
        )
        info["compiler_version"] = (
            (ver.stdout or "").splitlines()[0] if ver.stdout else ""
        )
        # The removed exact-calendar flags MUST error on use (post-WP8 pin proof).
        for flag in REMOVED_FLAGS:
            probe = subprocess.run(
                [
                    str(self.compiler),
                    "-mcpu=tt-bh-tensix",
                    flag,
                    "-fsyntax-only",
                    "-x",
                    "c++",
                    "-",
                ],
                input="int main(){return 0;}",
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                sys.exit(
                    f"pin check failed: removed flag {flag} was ACCEPTED — wrong toolchain"
                )
        info["removed_flags_error_on_use"] = True
        # Both flag sets must be accepted.
        for label, flags in (("off", OFF_FLAGS), ("on", ON_FLAGS)):
            probe = subprocess.run(
                [
                    str(self.compiler),
                    "-mcpu=tt-bh-tensix",
                    *flags.split(),
                    "-fsyntax-only",
                    "-x",
                    "c++",
                    "-",
                ],
                input="int main(){return 0;}",
                capture_output=True,
                text=True,
            )
            if probe.returncode != 0:
                sys.exit(
                    f"{label} flag set rejected by compiler:\n{probe.stdout}{probe.stderr}"
                )
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        # Sweep-hardening 2: `rev-parse HEAD` is blind to an UNCOMMITTED
        # working tree — an edited kernel header or ops TSV would resume
        # every classify/CRAQ verdict and device cell stale.  Key on the
        # tracked tt-llk diff as well (untracked files are excluded: the
        # tests/sfpi symlink and pytest __pycache__ churn would otherwise
        # invalidate every resume).
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "HEAD", "--", "tt_metal/tt-llk"],
            capture_output=True,
        ).stdout
        if dirty.strip():
            head += "+dirty." + hashlib.sha256(dirty).hexdigest()[:16]
        info["tt_metal_head"] = head
        for arch in ("bh", "wh"):
            sim = getattr(self.a, f"sim_{arch}")
            info[f"sim_{arch}"] = str(sim) if sim else ""
            info[f"sim_{arch}_sha256"] = sha256(sim) if sim and sim.is_file() else ""
            info[f"sim_{arch}_sha_pin"] = getattr(self.a, f"sim_{arch}_sha", "") or ""
        # objdump is a GATE TOOL (macro-launch classification): a missing
        # objdump would silently disable the issue_slot_lb requirement — the
        # exact silent-no-op class this enforcement layer exists to kill.
        if not self.objdump.is_file():
            sys.exit(
                f"missing objdump {self.objdump} — the classify phase "
                "disassembles every leg to detect macro-launch shapes "
                "(issue_slot_lb enforcement); a toolchain without objdump "
                "cannot run a gated sweep"
            )
        self.info = info  # phase-entry checks below read the recorded identity
        self._verify_sim_pins("preflight")
        # REVIEW RECORD gate (HANDOFF §1(4) as code): silicon-authorized
        # sweeps refuse without a pin-matched review record beside the
        # evidence root.  --phases without silicon (classify/craq-only
        # runs) and non-hardware runs stay ungated: they produce no device
        # evidence.
        info["review_record"] = ""
        info["review_record_sha256"] = ""
        if "silicon" in self.a.phases and self.a.allow_hardware:
            record = self.ev.parent / f"REVIEW_RECORD-{info['cc1plus_sha256'][:12]}.md"
            ok, detail = check_review_record(record, info["cc1plus_sha256"])
            if not ok:
                sys.exit(
                    "REVIEW RECORD REQUIRED (silicon phases authorized): "
                    + detail
                    + " — HANDOFF §1(4): independent review of compiler "
                    "mutations BEFORE silicon.  Write the record from "
                    "corpus/REVIEW_RECORD_TEMPLATE.md for the CURRENT "
                    f"cc1plus pin ({info['cc1plus_sha256']}), place it at "
                    f"{record}, then re-run.  No record, no silicon."
                )
            info["review_record"] = str(record)
            info["review_record_sha256"] = sha256(record)
        info["craq_gate_skipped"] = bool(self.a.skip_craq_gate)
        (self.ev / "preflight.json").write_text(json.dumps(info, indent=2) + "\n")
        man = [
            f"Lane sweep-2x2 evidence — {self.ev.name}",
            f"compiler driver: {self.compiler}",
            f"compiler driver sha256 (secondary pin): {info['compiler_sha256']}",
            f"cc1plus: {info['cc1plus']}",
            f"cc1plus sha256 (PRIMARY pin): {info['cc1plus_sha256']}",
            f"harness toolchain symlink: {info['harness_toolchain_symlink']}",
            "harness toolchain realpath (readlink -f, re-verified at every "
            f"phase entry): {info['harness_toolchain_realpath']}",
            f"compiler version: {info['compiler_version']}",
            f"tt-metal: {info['tt_metal_head']}",
            f"libttsim bh sha256: {info['sim_bh_sha256']}"
            + (
                f" (VERIFIED against reviewed pin {info['sim_bh_sha_pin']})"
                if info["sim_bh_sha_pin"]
                else " (UNPINNED — no --sim-bh-sha)"
            ),
            f"libttsim wh sha256: {info['sim_wh_sha256']}"
            + (
                f" (VERIFIED against reviewed pin {info['sim_wh_sha_pin']})"
                if info["sim_wh_sha_pin"]
                else " (UNPINNED — no --sim-wh-sha)"
            ),
            (
                f"review record: {info['review_record']} sha256 "
                f"{info['review_record_sha256']}"
                if info["review_record"]
                else "review record: not required (no silicon authorization this run)"
            ),
            craq_gate_taint(self.a.skip_craq_gate),
            f"OFF flags: {OFF_FLAGS}",
            f"ON flags: {ON_FLAGS}",
            "loadmacro flags: CONFIRMED error on use (removed with quarantined exact-calendar pass)",
        ]
        (self.ev / "MANIFEST.txt").write_text("\n".join(man) + "\n")
        self.info = info

    # ---------------- process helpers ----------------
    def _env(self, arch, runner_temp, flags, sim=None, extra=None):
        env = os.environ.copy()
        env.update(
            CHIP_ARCH=CHIP[arch],
            LLK_HOME=str(LLK),
            RUNNER_TEMP=str(runner_temp),
            TT_LLK_EXTRA_COMPILER_OPTIONS=flags,
        )
        if sim:
            env["TT_METAL_SIMULATOR"] = str(sim)
        if extra:
            env.update(extra)
        return env

    def _pytest(self, node, extra, env, log, timeout=1800):
        with open(log, "w") as f:
            rc = subprocess.run(
                [str(self.python), "-m", "pytest", "-q", *extra, node],
                cwd=PYDIR,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        return rc

    @staticmethod
    def _passed(log):
        log = pathlib.Path(log)
        if not log.is_file():
            return False
        text = log.read_text(errors="replace")
        return bool(re.search(r"\b[1-9]\d* passed\b", text))

    def _hash_one_elf(self, build_root, elf):
        """(relpath, .text sha256, elf sha256) for one kernel ELF."""
        rel = elf.relative_to(build_root)
        text = subprocess.run(
            [
                str(self.objcopy),
                "-O",
                "binary",
                "--only-section=.text",
                str(elf),
                "/dev/stdout",
            ],
            capture_output=True,
        ).stdout
        return (str(rel), hashlib.sha256(text).hexdigest(), sha256(elf))

    @staticmethod
    def _write_hash_file(entries, out_file):
        with open(out_file, "w") as f:
            for rel, t, e in entries:
                f.write(f"{rel}\ttext:{t}\telf:{e}\n")

    def _hash_build(self, rt, out_file):
        """Hash .text and full bytes of every kernel ELF under one RUNNER_TEMP."""
        entries = []
        for elf in sorted((rt / "tt-llk-build").rglob("*.elf")):
            if "shared" in elf.parts:
                continue  # brisc bootrom is flag-independent scaffolding
            entries.append(self._hash_one_elf(rt / "tt-llk-build", elf))
        self._write_hash_file(entries, out_file)
        return entries

    def _hash_build_subset(self, rt, rel_files, out_file):
        """_hash_build restricted to the given artefact FILES (relative to
        the build root) — one node's slice of a shared classify chunk
        build, exactly the file format the solo path writes.  File (not
        directory) granularity: sibling variants of two nodes share parent
        dirs, so any dir-level subset would swallow foreign ELFs."""
        entries = []
        build = rt / "tt-llk-build"
        for rel in sorted(rel_files):
            if not rel.endswith(".elf") or "shared" in pathlib.PurePath(rel).parts:
                continue
            elf = build / rel
            if elf.is_file():
                entries.append(self._hash_one_elf(build, elf))
        self._write_hash_file(entries, out_file)
        return entries

    def _archive_build(self, rt, dest):
        """Keep ELFs and build.h from a RUNNER_TEMP; drop the rest."""
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted((rt / "tt-llk-build").rglob("*")):
            if (
                path.is_file()
                and (path.suffix == ".elf" or path.name == "build.h")
                and "shared" not in path.parts
            ):
                out = dest / path.relative_to(rt / "tt-llk-build")
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)

    def _archive_variant_files(self, rt, rel_files, dest):
        """_archive_build restricted to the given artefact files — one
        node's slice of a shared classify chunk build."""
        dest.mkdir(parents=True, exist_ok=True)
        build = rt / "tt-llk-build"
        for rel in sorted(rel_files):
            path = build / rel
            if (
                path.is_file()
                and (path.suffix == ".elf" or path.name == "build.h")
                and "shared" not in path.parts
            ):
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)

    # ---------------- phase: classify ----------------
    def _classify_cached(self, work):
        """The (row,sel) verdict from a previous run, iff hash-keyed valid.

        Hash-matched resume: a cached classification is only valid for
        the compiler AND source tree that produced it.  Verdicts from
        another cc1plus or tt-metal head (or the pre-keying schema)
        are recompiled — kernel-source changes must re-derive hashes."""
        verdict_file = work / "verdict.json"
        if not verdict_file.is_file() or self.a.force:
            return None
        verdict = json.loads(verdict_file.read_text())
        if (
            verdict.get("cc1plus_sha256") == self.info["cc1plus_sha256"]
            and verdict.get("tt_metal_head") == self.info["tt_metal_head"]
            and (
                verdict.get("status") != "OK" or "macro_scan" in verdict
            )  # pre-enforcement-layer verdicts lack the scan: re-derive
        ):
            return verdict
        return None

    def _classify_compile_fail(self, row, sel, work, leg):
        """Shared COMPILE_FAIL verdict (legacy + batched classify paths)."""
        verdict = {
            "selector": sel,
            "status": "COMPILE_FAIL",
            "leg": leg,
            "cc1plus_sha256": self.info["cc1plus_sha256"],
            "tt_metal_head": self.info["tt_metal_head"],
        }
        (work / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        self.reds.append(f"{row['op']}/{sel}: compile {leg} failed")
        return verdict

    def _classify_verdict(self, sel, work, legnames, hashes):
        """Shared verdict tail (legacy + batched classify paths): the
        OFF-vs-ON hash-set comparison, macro-launch scan, and hash keys."""
        if len(legnames) == 1:
            verdict = {
                "selector": sel,
                "status": "OK",
                "all": "SINGLE_LEG",
                "math": "SINGLE_LEG",
            }
        else:
            a_set = sorted(h[1] for h in hashes[legnames[0]])
            b_set = sorted(h[1] for h in hashes[legnames[1]])
            math_a = sorted(
                h[1] for h in hashes[legnames[0]] if h[0].endswith("math.elf")
            )
            math_b = sorted(
                h[1] for h in hashes[legnames[1]] if h[0].endswith("math.elf")
            )
            verdict = {
                "selector": sel,
                "status": "OK",
                "all": "IDENTICAL" if a_set == b_set else "CHANGED",
                "math": "IDENTICAL" if math_a == math_b else "CHANGED",
            }
        verdict["macro_scan"] = self._macro_scan(work, legnames)
        verdict["cc1plus_sha256"] = self.info["cc1plus_sha256"]
        verdict["tt_metal_head"] = self.info["tt_metal_head"]
        (work / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        return verdict

    def classify(
        self, row, sel, legs=(("off", OFF_FLAGS), ("on", ON_FLAGS)), tag="classify"
    ):
        node = row["nodes"][sel]
        work = self.ev / row["op"] / tag / sel
        cached = self._classify_cached(work)
        if cached is not None:
            return cached
        work.mkdir(parents=True, exist_ok=True)
        (work / "node.txt").write_text(node + "\n")
        hashes = {}
        for leg, flags in legs:
            rt = work / f"rt-{leg}"
            shutil.rmtree(rt, ignore_errors=True)
            rt.mkdir(parents=True)
            (work / f"flags-{leg}.txt").write_text(flags + "\n")
            rc = self._pytest(
                node,
                ["--compile-producer"],
                self._env("bh", rt, flags, extra=row_env(row, sel)),
                work / f"compile-{leg}.log",
            )
            if rc != 0 or not self._passed(work / f"compile-{leg}.log"):
                return self._classify_compile_fail(row, sel, work, leg)
            hashes[leg] = self._hash_build(rt, work / f"hashes-{leg}.txt")
            self._archive_build(rt, work / f"elf-{leg}")
            shutil.rmtree(rt, ignore_errors=True)
        return self._classify_verdict(sel, work, [leg for leg, _ in legs], hashes)

    # ------------- batched classify producer sessions (laneCH) -------------
    # The classify phase previously paid one pytest producer session per
    # (row, selector, leg) — the 6-way prewarm pool ran those solo sessions
    # concurrently but each still paid full collection overhead.  These
    # methods batch the PENDING legs into per-(flags, extra_env) chunk
    # sessions the way the batched silicon executor batches device legs,
    # with per-node ELF attribution from the in-tree corpus pytest plugin
    # (artefact-FILE diff around each test — dir-level diffs mis-attribute
    # sibling variants under one sources/<file>/ dir; sequential only — never
    # xdist, concurrent workers would pollute the diff).  Every per-leg
    # evidence file (node.txt, flags-*.txt, hashes-*.txt, elf-*/,
    # verdict.json) is byte-compatible with the legacy solo path; anything
    # a chunk cannot PROVE (node deselected, no outcome, no created files,
    # session died) falls back to the legacy solo classify in the
    # sequential row loop — fail-open to legacy, never to a guessed
    # verdict.  SWEEP_CLASSIFY_WORKERS=1 disables all of it.

    _CLASSIFY_CHUNK_NODES = 16  # blast-radius / timeout bound per session

    def _classify_chunk_session(self, cdir, jobs, flags, extra_env):
        """ONE --compile-producer pytest session for MANY classify legs
        sharing (flags, extra_env).  Returns the parsed plugin report
        payload (possibly {}) — outcomes are per NODE, never a session
        verdict (the group-poisoning lesson applies here too)."""
        rt = cdir / "rt"
        shutil.rmtree(cdir, ignore_errors=True)
        rt.mkdir(parents=True)
        nodes = []
        for j in jobs:
            if j["node"] not in nodes:
                nodes.append(j["node"])
        (cdir / "nodes.txt").write_text("\n".join(nodes) + "\n")
        (cdir / "flags.txt").write_text(flags + "\n")
        env = self._env("bh", rt, flags, extra=dict(extra_env))
        env["SFPU_CORPUS_PYTEST_REPORT"] = str(cdir / "report.json")
        env["SFPU_CORPUS_VARIANT_MAP"] = "1"
        env["PYTHONPATH"] = f"{HERE}:{env.get('PYTHONPATH', '')}"
        rc = 0
        with open(cdir / "session.log", "w") as f:
            try:
                rc = subprocess.run(
                    [
                        str(self.python),
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "sfpu_corpus_pytest_plugin",
                        "--compile-producer",
                        *nodes,
                    ],
                    cwd=PYDIR,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=900 + 300 * len(nodes),
                ).returncode
            except subprocess.TimeoutExpired:
                rc = 124
                f.write("\nCLASSIFY CHUNK SESSION TIMED OUT\n")
        (cdir / "rc.txt").write_text(f"{rc}\n")
        rj = cdir / "report.json"
        if rj.is_file():
            try:
                return json.loads(rj.read_text())
            except ValueError:
                pass
        return {}

    def _extract_classify_leg(self, cdir, job, report):
        """Attribute one leg's outcome out of a chunk session, writing the
        same per-leg evidence files the solo path writes.  Returns
        ('ok', entries) | ('compile_fail',) | ('fallback', reason)."""
        node, leg = job["node"], job["leg"]
        phases = (report.get("reports") or {}).get(node) or {}
        vfiles = (report.get("variant_files") or {}).get(node) or []
        passed = (
            bool(phases)
            and "call" in phases
            and all(p.get("outcome") == "passed" for p in phases.values())
        )
        work = job["work"]
        work.mkdir(parents=True, exist_ok=True)
        (work / "node.txt").write_text(node + "\n")
        log = work / f"compile-{leg}.log"
        if node in (report.get("deselected") or []):
            log.write_text(
                f"batched classify chunk {cdir}: node DESELECTED (runtime-"
                "only variant collapse against another chunk node) — solo "
                "compile fallback\n"
            )
            return ("fallback", "deselected in chunk (runtime-only collapse)")
        if not phases:
            log.write_text(
                f"batched classify chunk {cdir}: node produced no outcome "
                "(session died before it ran?) — solo compile fallback\n"
            )
            return ("fallback", "no outcome in chunk session")
        if not passed:
            (work / f"flags-{leg}.txt").write_text(job["flags"] + "\n")
            log.write_text(
                f"batched classify chunk {cdir}: compile FAILED for this "
                f"node\nphases: {json.dumps(phases, sort_keys=True)}\n"
                f"(full session output: {cdir}/session.log)\n"
            )
            return ("compile_fail",)
        if not vfiles:
            log.write_text(
                f"batched classify chunk {cdir}: node passed but created no "
                "artefact files (variant shared with an earlier chunk "
                "node?) — solo compile fallback\n"
            )
            return ("fallback", "passed but no attributable artefact files")
        (work / f"flags-{leg}.txt").write_text(job["flags"] + "\n")
        entries = self._hash_build_subset(
            cdir / "rt", vfiles, work / f"hashes-{leg}.txt"
        )
        if not entries:
            (work / f"hashes-{leg}.txt").unlink(missing_ok=True)
            return ("fallback", "attributed artefact files contained no ELFs")
        shutil.rmtree(work / f"elf-{leg}", ignore_errors=True)
        self._archive_variant_files(cdir / "rt", vfiles, work / f"elf-{leg}")
        log.write_text(
            f"batched classify chunk {cdir}: 1 passed (chunk session; full "
            f"output in {cdir}/session.log)\n"
            f"artefact files: {', '.join(vfiles)}\n"
        )
        return ("ok", entries)

    def _batched_classify(self, pending):
        """Prewarm the classify caches with BATCHED producer sessions.

        pending: [(row, sel, legs_spec_or_None)].  Verdict-cached selectors
        are skipped (the row loop replays them); the rest compile in
        chunked sessions and their verdicts are written here, so the
        sequential row loop resumes every one hash-matched from cache.
        Returns the specs the chunks could NOT prove — [(row, sel, legs,
        tag)] — for the caller to dispatch through _solo_classify_pool
        (concurrent legacy-solo compiles) instead of paying them one at a
        time inside the sequential row loop."""
        jobs_by_rowsel = {}
        legjobs = []
        for row, sel, p_legs in pending:
            legs = p_legs or (("off", OFF_FLAGS), ("on", ON_FLAGS))
            work = self.ev / row["op"] / "classify" / sel
            if self._classify_cached(work) is not None:
                continue
            jobs_by_rowsel[(row["op"], sel)] = {
                "row": row,
                "sel": sel,
                "legs": legs,
                "status": {},
            }
            for leg, flags in legs:
                legjobs.append(
                    {
                        "row": row,
                        "sel": sel,
                        "leg": leg,
                        "flags": flags,
                        "node": row["nodes"][sel],
                        "extra_env": row_env(row, sel),
                        "work": work,
                    }
                )
        if not legjobs:
            print("batched classify: every pending verdict already cached")
            return []
        groups = {}
        for j in legjobs:
            key = (j["flags"], tuple(sorted(j["extra_env"].items())))
            groups.setdefault(key, []).append(j)
        broot = self.ev / "classify-batches"
        broot.mkdir(parents=True, exist_ok=True)
        chunks = []
        gkeys = sorted(groups)
        for i, key in enumerate(gkeys):
            gjobs = sorted(groups[key], key=lambda j: (j["node"], j["sel"], j["leg"]))
            gname = (
                f"g{i}-"
                + hashlib.sha256((key[0] + repr(key[1])).encode()).hexdigest()[:8]
            )
            nodes = []
            for j in gjobs:
                if j["node"] not in nodes:
                    nodes.append(j["node"])
            nchunks = min(
                len(nodes),
                max(
                    self.a.classify_workers,
                    -(-len(nodes) // self._CLASSIFY_CHUNK_NODES),
                ),
            )
            for c in range(nchunks):
                cnodes = {n for x, n in enumerate(nodes) if x % nchunks == c}
                cjobs = [j for j in gjobs if j["node"] in cnodes]
                chunks.append((broot / gname / f"c{c}", cjobs, key[0], key[1]))
        print(
            f"batched classify plan: {len(jobs_by_rowsel)} verdict(s) "
            f"pending ({len(pending) - len(jobs_by_rowsel)} cached), "
            f"{len(legjobs)} compile leg(s) across {len(gkeys)} group(s) in "
            f"{len(chunks)} chunk session(s) x {self.a.classify_workers} "
            "worker(s)"
        )

        def run_chunk(spec):
            cdir, cjobs, flags, extra_env = spec
            self.verify_toolchain("classify")
            report = self._classify_chunk_session(cdir, cjobs, flags, extra_env)
            out = [(j, self._extract_classify_leg(cdir, j, report)) for j in cjobs]
            # per-leg ELFs are archived under the classify work dirs above;
            # the shared chunk tree is disk we no longer need
            shutil.rmtree(cdir / "rt", ignore_errors=True)
            return out

        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.a.classify_workers
        ) as pool:
            futs = {pool.submit(run_chunk, spec): spec for spec in chunks}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    results.extend(fut.result())
                except Exception as exc:  # deferred: the row loop compiles solo
                    print(
                        f"classify chunk {futs[fut][0]} exception "
                        f"(deferred to legacy solo): {exc}"
                    )
        for job, status in results:
            jobs_by_rowsel[(job["row"]["op"], job["sel"])]["status"][
                job["leg"]
            ] = status
        done = failed = 0
        fell_back = []
        for entry in jobs_by_rowsel.values():
            row, sel = entry["row"], entry["sel"]
            work = self.ev / row["op"] / "classify" / sel
            legnames = [leg for leg, _ in entry["legs"]]
            hashes, verdict_done, fallback = {}, False, None
            for leg in legnames:
                st = entry["status"].get(leg) or ("fallback", "chunk never ran")
                if st[0] == "compile_fail":
                    # legacy order: the FIRST failing leg names the verdict
                    self._classify_compile_fail(row, sel, work, leg)
                    failed += 1
                    verdict_done = True
                    break
                if st[0] == "fallback":
                    fallback = st[1]
                    break
                hashes[leg] = st[1]
            if verdict_done:
                continue
            if fallback is not None:
                fell_back.append(
                    (entry["row"], entry["sel"], entry["legs"], "classify")
                )
                print(
                    f"batched classify: {row['op']}/{sel} -> legacy solo ({fallback})"
                )
                continue
            self._classify_verdict(sel, work, legnames, hashes)
            done += 1
        print(
            f"batched classify: {done} verdict(s) assembled, {failed} "
            f"COMPILE_FAIL, {len(fell_back)} deferred to the legacy solo path"
        )
        return fell_back

    def _solo_classify_pool(self, specs, what):
        """Run LEGACY SOLO classify() verdicts CONCURRENTLY (owner order
        2026-08-20, laneDB): attribution requires each leg to compile in
        its OWN isolated pytest session — it never required those sessions
        to run one after another.

        specs: [(row, sel, legs_spec_or_None, tag)].  Every dispatched
        call is self.classify() itself — byte-for-byte the sequential
        path: one --compile-producer pytest session per leg, each with
        its own RUNNER_TEMP (work/rt-<leg>, so the harness ARTEFACTS_DIR
        rmtree-at-session-start and every RUNNER_TEMP-derived dir touch
        only that session's tree) and its own byte-exact flag set; legs
        stay strictly ordered inside a verdict, later legs withheld after
        a COMPILE_FAIL exactly as before.  Only the SCHEDULING across
        independent (row, sel, tag) verdicts changes — their work dirs
        (ev/<op>/<tag>/<sel>) are disjoint, so concurrent verdicts share
        no files.  CH's batching refusal (knob/CRAQ classify legs never
        SHARE a session) is untouched: nothing here co-schedules two legs
        into one session.

        Fail-open: a spec that raises is only logged — the sequential
        caller re-runs that classify() inline, serial-legacy style.  With
        classify_workers <= 1 (SWEEP_CLASSIFY_WORKERS=1) this is a no-op
        and the sequential loop compiles everything, as documented."""
        if getattr(self.a, "classify_workers", 1) <= 1:
            return
        pending, seen = [], set()
        for row, sel, legs, tag in specs:
            work = self.ev / row["op"] / tag / sel
            if work in seen or self._classify_cached(work) is not None:
                continue
            seen.add(work)
            pending.append((row, sel, legs, tag))
        if not pending:
            return
        print(
            f"solo classify pool ({what}): {len(pending)} verdict(s) x "
            f"{self.a.classify_workers} worker(s), one isolated pytest "
            "session per leg"
        )

        def one(spec):
            row, sel, legs, tag = spec
            # same mid-run toolchain-swap guard the batched chunks carry
            self.verify_toolchain("classify")
            if legs is None:
                return self.classify(row, sel, tag=tag)
            return self.classify(row, sel, legs=legs, tag=tag)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.a.classify_workers
        ) as pool:
            futs = {pool.submit(one, s): s for s in pending}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:  # deferred: the row loop compiles solo
                    row, sel, _legs, tag = futs[fut]
                    print(
                        f"solo classify {row['op']}/{tag}/{sel} exception "
                        f"(deferred to the sequential loop): {exc}"
                    )

    def _scan_leg_disasm(self, work, leg):
        """Sum the macro-launch census over every archived math.elf of a
        classify leg (objdump -d; the classify phase already archives the
        ELFs, so the scan adds no compile work)."""
        counts = {name: 0 for name in _SCAN_PATTERNS}
        elfs = sorted((work / f"elf-{leg}").rglob("math.elf"))
        for elf in elfs:
            dis = subprocess.run(
                [str(self.objdump), "-d", str(elf)], capture_output=True, text=True
            )
            if dis.returncode != 0:
                sys.exit(
                    f"objdump failed on {elf} (macro-launch classification "
                    "is a gate; it must not silently degrade): "
                    f"{dis.stderr.strip()[:400]}"
                )
            for name, n in scan_disasm_text(dis.stdout).items():
                counts[name] += n
        counts["math_elfs"] = len(elfs)
        return counts

    def _macro_scan(self, work, legnames):
        """Macro-launch verdict for a classify evidence dir.

        Two-leg rows scan ON vs OFF (replay-launch = ON-only launches);
        single-leg rows (pinpair) scan their only leg with the
        SFPLOADMACRO criterion.  Returns the dict stored in the classify
        verdict as 'macro_scan'."""
        if len(legnames) == 1:
            on = self._scan_leg_disasm(work, legnames[0])
            off = None
        else:
            off = self._scan_leg_disasm(work, legnames[0])
            on = self._scan_leg_disasm(work, legnames[1])
        cls = classify_macro_launch(on, off)
        return {
            "classification": cls,
            "sfploadmacro_on": on.get("sfploadmacro", 0),
            "replay_launch_on": on.get("replay_launch", 0),
            "replay_launch_off": (off or {}).get("replay_launch", 0),
            "math_elfs_on": on.get("math_elfs", 0),
        }

    def _classify_texts(self, row, sel, leg, tag="classify"):
        """This run's .text hash set for (row, sel, leg) from the classify
        evidence — the reference a cached device job must hash-match."""
        path = self.ev / row["op"] / tag / sel / f"hashes-{leg}.txt"
        if not path.is_file():
            return None
        return self._texts_of(path)

    @staticmethod
    def _texts_of(hash_file):
        texts = []
        for line in pathlib.Path(hash_file).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].startswith("text:"):
                texts.append(parts[1][len("text:") :])
        return sorted(texts)

    # ---------------- phase: craq ----------------
    SOC_DESCRIPTORS = {
        "bh": "tt_metal/soc_descriptors/blackhole_140_arch.yaml",
        "wh": "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml",
    }

    def _staged_sim(self, arch):
        """ttexalens needs soc_descriptor.yaml BESIDE libttsim.so; stage it.

        The craq-sim build tree ships only the .so, so a bare --sim-* path
        would fail with 'bad file: .../soc_descriptor.yaml'.  Stage the .so
        together with tt-metal's arch descriptor under the evidence root.
        """
        sim = getattr(self.a, f"sim_{arch}")
        if not sim or not sim.is_file():
            return None
        if (sim.parent / "soc_descriptor.yaml").is_file():
            return sim
        stage = self.ev / "simstage" / arch
        stage.mkdir(parents=True, exist_ok=True)
        if not (stage / "libttsim.so").is_file():
            shutil.copy2(sim, stage / "libttsim.so")
        shutil.copy2(ROOT / self.SOC_DESCRIPTORS[arch], stage / "soc_descriptor.yaml")
        return stage / "libttsim.so"

    def craq(
        self,
        row,
        sel,
        arch,
        legs_spec=(("off", OFF_FLAGS), ("on", ON_FLAGS)),
        tag="craq",
    ):
        node = row["nodes"][sel]
        sim = self._staged_sim(arch)
        work = self.ev / row["op"] / tag / f"{sel}-{arch}"
        verdict_file = work / "verdict.json"
        sim_sha = sha256(sim) if sim and sim.is_file() else ""
        if verdict_file.is_file() and not self.a.force:
            verdict = json.loads(verdict_file.read_text())
            # Hash-matched resume: verdicts are keyed to cc1plus + simulator
            # + tt-metal head (kernel-source changes re-run the pair).
            if (
                verdict.get("cc1plus_sha256") == self.info["cc1plus_sha256"]
                and verdict.get("sim_sha256") == sim_sha
                and verdict.get("tt_metal_head") == self.info["tt_metal_head"]
                and verdict.get("status") != "SKIP_NO_SIMULATOR"
            ):
                return verdict
        if not sim or not sim.is_file():
            verdict = {"selector": sel, "arch": arch, "status": "SKIP_NO_SIMULATOR"}
            work.mkdir(parents=True, exist_ok=True)
            verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
            return verdict
        work.mkdir(parents=True, exist_ok=True)
        (work / "node.txt").write_text(node + "\n")
        legs = {}
        for leg, flags in legs_spec:
            rt = work / f"rt-{leg}"
            shutil.rmtree(rt, ignore_errors=True)
            rt.mkdir(parents=True)
            log = work / f"craq-{leg}.log"
            rc = self._pytest(
                node,
                ["--run-simulator"],
                self._env(arch, rt, flags, sim=sim, extra=row_env(row, sel)),
                log,
                timeout=2400,
            )
            text = log.read_text(errors="replace")
            if self._passed(log):
                legs[leg] = "PASS"
            elif "UnsupportedFunctionality" in text:
                legs[leg] = "UNSUPPORTED"
            elif re.search(r"\b[1-9]\d* skipped\b", text) and " failed" not in text:
                legs[leg] = "SKIPPED"
            else:
                legs[leg] = f"FAIL(rc={rc})"
            shutil.rmtree(rt, ignore_errors=True)
        verdict = {
            "selector": sel,
            "arch": arch,
            "status": "OK",
            "legs": legs,
            "cc1plus_sha256": self.info["cc1plus_sha256"],
            "sim_sha256": sim_sha,
            "tt_metal_head": self.info["tt_metal_head"],
        }
        verdict_file.write_text(json.dumps(verdict, indent=2) + "\n")
        if arch == "bh" and any(v != "PASS" for v in legs.values()):
            self.reds.append(f"{row['op']}/{sel}: CRAQ {arch} ({tag}) {legs}")
        return verdict

    # ---------------- phase: silicon ----------------
    def _bh_craq_gate(self, row):
        """Keyed silicon gate (adversarial finding sweep_2x2.py:1341).

        The gate trusts only BH CRAQ verdicts whose cc1plus + simulator +
        tt-metal keys match THIS run — legs==PASS alone would let a
        `--phases silicon` resume open the gate with greens earned by an
        older compiler/simulator/tree.  A SKIP_NO_SIMULATOR verdict (no
        legs) never opens it.
        """
        craq_dir = self.ev / row["op"] / "craq"
        if not craq_dir.is_dir():
            return False
        sim = self._staged_sim("bh")
        sim_sha = sha256(sim) if sim and sim.is_file() else ""
        verdicts = [
            json.loads(p.read_text())
            for p in sorted(craq_dir.glob("*-bh/verdict.json"))
        ]
        if not verdicts:
            return False
        for v in verdicts:
            if not (v.get("legs") and all(x == "PASS" for x in v["legs"].values())):
                return False
            if (
                v.get("cc1plus_sha256") != self.info["cc1plus_sha256"]
                or v.get("sim_sha256") != sim_sha
                or v.get("tt_metal_head") != self.info["tt_metal_head"]
            ):
                return False
        return True

    def _load_keyed_classification(self, row, sel):
        """Classification evidence for (row, sel) valid for THIS run's keys,
        or None.  Used when the classify phase was skipped: silicon must
        never run on unkeyed/stale classify evidence (the byte-identical
        refusal logic and the hash-matched device resume both depend on it)."""
        vf = self.ev / row["op"] / "classify" / sel / "verdict.json"
        if not vf.is_file():
            return None
        try:
            v = json.loads(vf.read_text())
        except ValueError:
            return None
        if (
            v.get("cc1plus_sha256") == self.info["cc1plus_sha256"]
            and v.get("tt_metal_head") == self.info["tt_metal_head"]
        ):
            return v
        return None

    def _device_job(
        self, row, sel, label, leg, flags, tag="silicon", expected_texts=None
    ):
        """One serialized device job under both flocks; CSVs copied in-lock."""
        node = row["nodes"][sel]
        work = self.ev / row["op"] / tag / sel / f"{label}-{leg}"
        # The full identity a cached cell must match before reuse: kernel
        # .text alone cannot see test parameters (node id: input ranges,
        # tolerances), flags, or extra_env (adversarial finding
        # sweep_2x2.py:572).  `mode` keys the execution context: batched
        # cells (co-scheduled consumer session) and serial cells (solo
        # session) are never mixed inside one row's samples — a mode switch
        # re-measures instead of silently blending two measurement contexts.
        batched = self.exec_mode == "batched" and tag == "silicon"
        jobkey = {
            "node": node,
            "flags": flags,
            "extra_env": row_env(row, sel),
            "tag": tag,
            "mode": "batched" if batched else "serial",
        }
        # Cross-run adoption (laneDA fix): with no local evidence for this
        # leg, probe the --prev-run root(s) and adopt a jobkey- and
        # .text-hash-matched green cell into this run's root (REUSED_FROM
        # marker written; source roots provenance-gated first — wave-12
        # ledger 19).  The adopted cell then flows through the very
        # resume validation below — adoption can never bypass it.
        if not (work / "rc.txt").is_file() and not self.a.force:
            self._adopt_prev_cell(work, jobkey, expected_texts)
        # Resume skips only GREEN jobs whose (node, flags, extra_env) jobkey
        # matches AND whose archived .text hash set equals what THIS run's
        # compiler produces for the same node/flags (from the classify
        # evidence).  ABSENT classify hashes (expected_texts=None: --phases
        # silicon without classify, or a leg whose classify stopped before
        # writing hashes) mean the cache cannot be validated: re-run, never
        # reuse (finding sweep_2x2.py:575).  A failed job, or a cell measured
        # from a stale binary, is re-run — never cached as done.
        if (work / "rc.txt").is_file() and not self.a.force:
            prior_rc = int((work / "rc.txt").read_text().strip() or 99)
            if prior_rc == 0 and self._passed(work / "log.txt"):
                cached_key = None
                if (work / "jobkey.json").is_file():
                    try:
                        cached_key = json.loads((work / "jobkey.json").read_text())
                    except (ValueError, OSError):
                        cached_key = None
                archived = (
                    self._texts_of(work / "TEXT_HASHES.txt")
                    if (work / "TEXT_HASHES.txt").is_file()
                    else None
                )
                if expected_texts is None:
                    print(
                        f"resume: {row['op']}/{sel} {label}-{leg} has no "
                        "classify hash reference for this run — cached cell "
                        "not trusted, re-measuring"
                    )
                elif cached_key != jobkey:
                    print(
                        f"resume: {row['op']}/{sel} {label}-{leg} job key "
                        "(node/flags/extra_env) changed or unrecorded — "
                        "re-measuring"
                    )
                elif archived == expected_texts:
                    return prior_rc  # keyed, hash-matched reuse
                else:
                    print(
                        f"resume: {row['op']}/{sel} {label}-{leg} .text hashes "
                        "changed — re-measuring"
                    )
        # Batched mode (non-dry): this method never executes a device job —
        # the batched executor already produced the per-leg evidence, and
        # this call is the ASSEMBLY pass reporting on it (the legacy resume
        # fast-path above already returned for green, keyed, hash-matched
        # cells).  Executing serially here would silently mix a solo
        # measurement into a batched sample set.
        if batched and not self.a.dry_run:
            return self._batched_leg_verdict(
                row, sel, label, leg, work, jobkey, expected_texts
            )
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        rt = work / "rt"
        rt.mkdir()
        (work / "node.txt").write_text(node + "\n")
        (work / "flags.txt").write_text(flags + "\n")
        (work / "jobkey.json").write_text(json.dumps(jobkey, indent=2) + "\n")
        env_prefix = " ".join(f'{k}="{v}"' for k, v in row_env(row, sel).items())
        inner = work / "inner.sh"
        # Node ids reach pytest via the line-oriented node.txt argfile
        # expanded with bash mapfile — no sh quoting layer ever parses them,
        # so single quotes, spaces, parens and angle brackets in pytest
        # parametrization reprs are safe (the pin-14 sweep killer: SdpaFwOp
        # nodes carry <DestSync.Half: 'SyncHalf'>).  The one impossible byte
        # is a newline: it would split the argfile row.  Explicit check, not
        # an assert: asserts are compiled out under `python3 -O`.
        if "\n" in node or "\r" in node:
            sys.exit(
                f"pytest node id contains a newline (breaks the node-id "
                f"argfile): {node!r}"
            )
        inner.write_text(
            f"""#!/usr/bin/env bash
rm -rf "{LLK}/perf_data"
cd "{PYDIR}" || exit 97
mapfile -t NODES < "{work}/node.txt"
env {env_prefix} CHIP_ARCH=blackhole LLK_HOME="{LLK}" RUNNER_TEMP="{rt}" \\
TT_LLK_EXTRA_COMPILER_OPTIONS="{flags}" \\
timeout 1500 "{self.python}" -m pytest -q -v "${{NODES[@]}}" > "{work}/log.txt" 2>&1
RC=$?
echo $RC > "{work}/rc.txt"
# copy raw+post perf CSVs IN-LOCK immediately (they are overwritten per run)
if [ -d "{LLK}/perf_data" ]; then cp -r "{LLK}/perf_data" "{work}/perf_data"; fi
if [ -d "{rt}/tt-llk-build/temp_perf_data" ]; then cp -r "{rt}/tt-llk-build/temp_perf_data" "{work}/raw_perf_data"; fi
exit $RC
"""
        )
        inner.chmod(0o755)
        if self.a.dry_run:
            print(f"DRY-RUN device job: {row['op']}/{sel} {label}-{leg}")
            return 0
        subprocess.run(
            [
                "flock",
                "-x",
                DEVICE_LOCK,
                "-c",
                f"flock -x {SILICON_LOCK} -c {shlex.quote(str(inner))}",
            ],
            check=False,
        )
        rc = (
            int((work / "rc.txt").read_text().strip())
            if (work / "rc.txt").is_file()
            else 99
        )
        # Post-lock archival: ELFs/.text hashes/build.h live in this job's own
        # RUNNER_TEMP, so no other process can overwrite them.
        self._hash_build(rt, work / "TEXT_HASHES.txt")
        self._archive_build(rt, work / "elf")
        shutil.rmtree(rt, ignore_errors=True)
        if rc != 0 or not self._passed(work / "log.txt"):
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: device job FAIL rc={rc}"
            )
        elif (
            expected_texts is not None
            and self._texts_of(work / "TEXT_HASHES.txt") != expected_texts
        ):
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: device job .text differs "
                "from this run's classify build (non-deterministic build?)"
            )
        return rc

    def _perf_value(self, row, sel, label, leg, tag="silicon"):
        """Parse the row's scoped metric from the copied post CSV (lock long
        released).  per_tile rows divide by tile_cnt (cycles/tile); absolute
        rows (e.g. Reduce-SDPA REDUCE_SDPA_BODY) sum the marker's rows."""
        work = self.ev / row["op"] / tag / sel / f"{label}-{leg}"
        col = f"mean({row['metric']})"
        for post in sorted(work.glob("perf_data/*/*.post.csv")):
            total, tiles, seen = 0.0, 1.0, False
            with post.open() as f:
                for rec in csv.DictReader(f):
                    if rec.get("marker") != row["marker"] or col not in rec:
                        continue
                    total += float(rec[col])
                    if not seen:
                        try:
                            tiles = float(rec.get("tile_cnt", 1) or 1)
                        except ValueError:
                            tiles = 1.0
                    seen = True
            if seen:
                return total / (tiles or 1.0) if row["per_tile"] else total
        return None

    def _result_skeleton(self, row, classifications):
        return {
            "op": row["op"],
            "corpus_id": row["corpus_id"],
            "kind": row["kind"],
            "marker": row["marker"],
            "scope": row_scope(row),
            "classify": classifications,
            "cells": {},
            "runs": {},
            "notes": [],
        }

    def _measured_flags_gate(self, row, classifications, result):
        """Contradiction lint (the welford pin-12 lesson): if classify proves
        the ON set CHANGES this row's bytes, the measured silicon legs MUST
        have carried the ON set.  A row measured at default/OFF flags while
        classify says CHANGED is benchmarking with the mechanism available
        but unrequested — RED, never silent.

        Must run AFTER measurement: on the pre-measurement skeleton
        result["runs"] is empty, which RED'd every byte-changing row
        (the s3rows-20260819 false-positive)."""
        changed = any(
            (classifications.get(sel) or {}).get("all") == "CHANGED"
            for sel in ("sem-perf", "sem-corr")
        )
        if not changed:
            return
        if self.a.dry_run:
            return  # a dry run proves wiring, never carries measured legs
        if any(str(n).startswith("STOP:") for n in result.get("notes") or []):
            return  # correctness STOP already withheld perf — that RED owns the row
        if row["kind"] == "pinpair":
            # pinpair rows measure at the row's pinned flag set: RED unless
            # that set actually requests a mechanism.
            if "-mtt-tensix-optimize" not in (row.get("pin_flags") or ""):
                msg = (
                    f"{row['op']}: classify says ON CHANGES bytes but the pinned "
                    f"flag set requests no mechanism (stale pinpair wiring): RED"
                )
                self.reds.append(msg)
                result["notes"].append(msg)
            return
        on_measured = any(
            str(k).endswith("_on_samples") and v
            for k, v in (result.get("runs") or {}).items()
        ) or any(str(k).endswith("/corr-on") for k in (result.get("runs") or {}))
        if not on_measured:
            msg = (
                f"{row['op']}: classify says ON CHANGES bytes but the row has no "
                f"ON-flag measured leg (legacy pinpair wiring): RED"
            )
            self.reds.append(msg)
            result["notes"].append(msg)

    def _macro_lb_gate(self, row, classifications, result):
        """Structural issue_slot_lb requirement (enforcement layer): a
        macro-launch row without a bound is RED, named in the report with
        the §1 caveat — never a silent no-op.  Uses the measured perf leg's
        classification when present, else the correctness leg's."""
        scan = None
        for sel in ("sem-perf", "sem-corr"):
            v = classifications.get(sel) or {}
            if v.get("macro_scan"):
                scan = v["macro_scan"]
                break
        msg = macro_lb_red(row["op"], row["marker"], row["issue_slot_lb"], scan)
        if msg:
            self.reds.append(msg)
            result["notes"].append(msg)

    def _issue_slot_check(self, row, result):
        """HANDOFF §1 metric caveat as code: a BODY-family reading on a
        macro-launch shape must be >= the payload's issue-slot lower bound
        (issue_slot_lb, cycles/tile), else the marker reading is INVALID and
        the KERNEL marker is required.  The check result is recorded either
        way so every measured cell carries its validity evidence."""
        lb = row["issue_slot_lb"]
        if lb is None:
            return
        cells = result["cells"]
        checked, invalid = [], []
        for cell, val in list(cells.items()):
            if not isinstance(val, (int, float)):
                continue
            if val < lb:
                cells[cell] = "INVALID_MARKER"
                invalid.append(f"{cell}={val:.2f}")
                self.reds.append(
                    f"{row['op']}/{cell}: INVALID_MARKER — {row['marker']} "
                    f"reading {val:.2f} < issue-slot lower bound {lb:g}; "
                    "KERNEL marker required"
                )
            else:
                checked.append(f"{cell}={val:.2f}")
        if invalid:
            result["notes"].append(
                f"issue-slot check FAIL ({', '.join(invalid)} < {lb:g}): "
                f"{row['marker']} is not a valid metric zone for this "
                "macro-launch shape — re-measure with the KERNEL marker"
            )
        elif checked:
            result["notes"].append(
                f"issue-slot check PASS: {', '.join(checked)} all >= payload "
                f"issue-slot lower bound {lb:g} cycles/tile "
                f"({row['marker']} reading valid for this macro-launch shape)"
            )

    # ---------------- batched silicon executor (laneBU) ----------------
    # See the module docstring "Batched silicon execution" for the protocol
    # mapping and the pre-registered speedup arithmetic.  The executor runs
    # BEFORE the per-row assembly pass; assembly (silicon()/_device_job) then
    # consumes the per-leg evidence through the keyed hash-matched resume
    # path, so every cell/ratio/STOP decision is computed by the exact
    # legacy code.

    @staticmethod
    def _node_file(node):
        return node.split("::", 1)[0]

    @staticmethod
    def _node_mathop(node):
        """The node id's mathop token (the only CSV-visible row
        discriminator between two ops of one perf module), or None."""
        m = re.search(r"[\[-]mathop:([A-Za-z0-9_]+)", node)
        return m.group(1) if m else None

    def _classify_entries(self, row, sel, leg, tag="classify"):
        """(relpath, text_sha, elf_sha) triples from the classify hash file
        — the node->ELF map a batched leg's TEXT_HASHES subset uses."""
        path = self.ev / row["op"] / tag / sel / f"hashes-{leg}.txt"
        if not path.is_file():
            return None
        out = []
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[1].startswith("text:"):
                out.append((parts[0], parts[1][5:], parts[2][4:]))
        return out

    def _mk_job(self, row, sel, label, leg, flags, kind):
        node = row["nodes"][sel]
        return {
            "row": row,
            "op": row["op"],
            "sel": sel,
            "label": label,
            "leg": leg,
            "flags": flags,
            "kind": kind,
            "node": node,
            "extra_env": row_env(row, sel),
            "file": self._node_file(node),
            "mathop": self._node_mathop(node),
            "rep": 0 if kind == "corr" else int(label[1:]),
            "work": self.ev / row["op"] / "silicon" / sel / f"{label}-{leg}",
        }

    def _silicon_jobs(self, row, cls):
        """Enumerate the row's device legs exactly as silicon()/
        silicon_pinpair() would execute them (byte-identity leg folding,
        sem-perf refusals, COMPILE_FAIL blocks all mirrored)."""
        jobs = []
        if row["kind"] == "pinpair":
            flags = row["pin_flags"]
            for sel in ("sem-corr", "hand-corr"):
                if row["nodes"][sel]:
                    jobs.append(
                        self._mk_job(row, sel, "corr", "default", flags, "corr")
                    )
            for r in range(1, PERF_RUNS + 1):
                for sel in ("sem-perf", "hand-perf"):
                    if row["nodes"][sel]:
                        jobs.append(
                            self._mk_job(row, sel, f"r{r}", "default", flags, "perf")
                        )
            return jobs
        for sel in ("sem-corr", "hand-corr"):
            if not row["nodes"][sel]:
                continue
            c = cls.get(sel, {})
            legs = ["off"] if c.get("all") == "IDENTICAL" else ["off", "on"]
            for leg in legs:
                jobs.append(
                    self._mk_job(
                        row,
                        sel,
                        "corr",
                        leg,
                        OFF_FLAGS if leg == "off" else ON_FLAGS,
                        "corr",
                    )
                )
        for sel in ("sem-perf", "hand-perf"):
            if not row["nodes"][sel]:
                continue
            c = cls.get(sel, {})
            if c.get("status") == "COMPILE_FAIL":
                continue
            identical = c.get("all") == "IDENTICAL"
            if identical and sel == "sem-perf":
                continue  # recorded refusal: zero device jobs
            legs = ["off"] if identical else ["off", "on"]
            for r in range(1, PERF_RUNS + 1):
                for leg in legs:
                    jobs.append(
                        self._mk_job(
                            row,
                            sel,
                            f"r{r}",
                            leg,
                            OFF_FLAGS if leg == "off" else ON_FLAGS,
                            "perf",
                        )
                    )
        return jobs

    def _job_key(self, job):
        return {
            "node": job["node"],
            "flags": job["flags"],
            "extra_env": job["extra_env"],
            "tag": "silicon",
            "mode": "batched",
        }

    def _prev_roots(self):
        """--prev-run evidence roots as a list (newest first).  Accepts the
        legacy single-Path form (older wrappers/selftests) and None."""
        prev = getattr(self.a, "prev_run", None)
        if not prev:
            return []
        return list(prev) if isinstance(prev, (list, tuple)) else [prev]

    def _cell_green(self, work, jobkey, expected_texts):
        """The keyed hash-matched cell validity check (one evidence dir):
        green rc + 'passed' log, jobkey equality, and archived .text set ==
        THIS run's classify hashes.  expected_texts=None never validates.
        Shared verbatim by the local resume and the cross-run adoption
        probe so reuse can never be weaker than resume."""
        if not (work / "rc.txt").is_file():
            return False
        try:
            rc = int((work / "rc.txt").read_text().strip() or 99)
        except ValueError:
            return False
        if rc != 0 or not self._passed(work / "log.txt"):
            return False
        try:
            cached_key = json.loads((work / "jobkey.json").read_text())
        except (ValueError, OSError):
            return False
        if cached_key != jobkey:
            return False
        if expected_texts is None or not (work / "TEXT_HASHES.txt").is_file():
            return False
        return self._texts_of(work / "TEXT_HASHES.txt") == expected_texts

    def _prev_root_provenance(self, prev):
        """Source-ROOT provenance gate for cross-run adoption, evaluated
        once per --prev-run root (cached).  The cell-level checks
        (_cell_green) protect the NUMBER; this protects where it came from
        (wave-12 ledger 19: _adopt_prev_cell consumed cells from
        quarantined roots, roots with no pin record and craq-gate-tainted
        roots — the cell checks never looked at the source root).
        Verdicts:
          - contamination/quarantine markers (the same set
            newest_clean_runs skips) ......................... REFUSE;
          - no readable pin record (PIN_STAMP first line, then
            preflight.json cc1plus_sha256 — the wrapper guard's own
            order) ........... REFUSE (fail closed: unknown provenance
            is how contamination starts);
          - preflight records craq_gate_skipped taint and THIS run
            gates on CRAQ .................................... REFUSE
            (a taint-matched run adopts, and _adopt_prev_cell
            propagates the taint line into THIS run's MANIFEST);
          - recorded pin != this run's pin ......... ALLOW with a loud
            CROSS-PIN ADOPTION line (the per-cell .text key against
            THIS run's classify hashes protects the number; the pin is
            recorded per adoption so provenance stays visible).
        Returns {"pin": ..., "taint": ...} on ACCEPT, None on REFUSE
        (refusals print once per root)."""
        if not hasattr(self, "_prev_prov"):
            self._prev_prov = {}
        key = str(prev.resolve())
        if key in self._prev_prov:
            return self._prev_prov[key]

        def refuse(why):
            print(f"reuse REFUSED: prev root {prev}: {why}")
            self._prev_prov[key] = None
            return None

        name = prev.name.lower()
        if "contaminated" in name or "quarantine" in name:
            return refuse(
                "root NAME marks it dirty (*CONTAMINATED*/*quarantine*) — "
                "never a reuse source"
            )
        if (prev / "QUARANTINED").exists():
            return refuse(
                "QUARANTINED marker file — quarantined evidence is never a "
                "reuse source"
            )
        if (prev / "CONTAMINATION-NOTE.md").exists():
            return refuse(
                "CONTAMINATION-NOTE.md — known-mixed evidence is never a "
                "reuse source"
            )
        pf = {}
        if (prev / "preflight.json").is_file():
            try:
                pf = json.loads((prev / "preflight.json").read_text())
            except (ValueError, OSError):
                pf = {}
        pin = ""
        if (prev / "PIN_STAMP").is_file():
            try:
                stamp = (prev / "PIN_STAMP").read_text().splitlines()
                pin = stamp[0].strip() if stamp else ""
            except OSError:
                pin = ""
        if not pin:
            pin = str(pf.get("cc1plus_sha256", "") or "")
        if not pin:
            return refuse(
                "no readable pin record (no PIN_STAMP, no preflight.json "
                "cc1plus_sha256) — fail closed: unknown provenance is how "
                "contamination starts"
            )
        taint = bool(pf.get("craq_gate_skipped", False))
        if taint and not getattr(self.a, "skip_craq_gate", False):
            return refuse(
                "its preflight records craq_gate_skipped taint but THIS run "
                "gates on CRAQ — a tainted cell cannot satisfy an untainted "
                "run"
            )
        ours = (getattr(self, "info", None) or {}).get("cc1plus_sha256", "")
        if pin != ours:
            print(
                f"CROSS-PIN ADOPTION: prev root {prev} recorded pin "
                f"{pin[:12]} != this run's {ours[:12] or '(unrecorded)'} — "
                "adoption allowed ONLY because every adopted cell is keyed "
                "on THIS run's classify .text hashes; source_pin recorded "
                "in REUSED_FROM.txt and scoreboard reused_cells"
            )
        prov = {"pin": pin, "taint": taint}
        self._prev_prov[key] = prov
        return prov

    @staticmethod
    def _reuse_chain(src, prev, pin):
        """Full adoption chain for a source cell, OLDEST FIRST — entry 0 is
        always the run that touched silicon.  If the source cell was itself
        adopted (carries a REUSED_FROM.txt), its recorded chain is EXTENDED
        with this hop, never overwritten (wave-12 ledger 19: the
        copytree-then-overwrite laundered the origin on transitive
        adoption).  Entries are (root, recorded_pin) pairs."""
        chain = []
        marker = src / "REUSED_FROM.txt"
        if marker.is_file():
            try:
                lines = marker.read_text().splitlines()
            except OSError:
                lines = []
            in_chain = False
            for ln in lines:
                if ln.startswith("chain-oldest-first:"):
                    in_chain = True
                    continue
                if in_chain:
                    m = re.match(r"^  (.+) pin:(\S+)$", ln)
                    if m:
                        chain.append((m.group(1), m.group(2)))
                        continue
                    in_chain = False
            if not chain:
                # legacy pre-gate marker (single reused-from line, no
                # chain): keep the origin hop; its pin was not recorded
                # back then
                for ln in lines:
                    if ln.startswith("reused-from:"):
                        chain.append((ln.split(":", 1)[1].strip(), "unrecorded"))
        chain.append((str(prev), pin))
        return chain

    def _adopt_prev_cell(self, work, jobkey, expected_texts):
        """Cross-run/cross-pin silicon cell reuse (laneDA root cause:
        --prev-run fed ONLY the scoreboard annotator while the resume prober
        looked at the current run root alone, so byte-identical OFF/hand
        cells re-ran every pin).  Probe each --prev-run root (newest first)
        for this leg's evidence at the SAME relative path and adopt it iff
        (1) the SOURCE ROOT passes the provenance gate
        (_prev_root_provenance: clean markers + readable pin record +
        craq-gate taint parity — wave-12 ledger 19) and (2) the cell passes
        the EXACT checks the local resume applies (_cell_green: green rc +
        jobkey equality + .text set == THIS run's classify hashes — a
        cc1plus bump that changes the bytes can never reuse).  Adoption
        COPIES the evidence into this run's root (self-contained,
        SHA256SUMS-covered) and writes a REUSED_FROM.txt marker beside it
        carrying the FULL adoption chain oldest-first with each hop's
        recorded pin, so the run that touched silicon is always entry 0 —
        provenance visible, never silent, never laundered.  Returns the
        source root or None."""
        if expected_texts is None or getattr(self.a, "force", False):
            return None
        try:
            rel = work.relative_to(self.ev)
        except ValueError:
            return None
        for root in self._prev_roots():
            prev = pathlib.Path(root)
            if prev.resolve() == self.ev.resolve():
                continue
            prov = self._prev_root_provenance(prev)
            if prov is None:
                continue
            src = prev / rel
            if not self._cell_green(src, jobkey, expected_texts):
                continue
            chain = self._reuse_chain(src, prev, prov["pin"])
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(src, work)
            (work / "REUSED_FROM.txt").write_text(
                f"reused-from:{prev}\nleg:{rel}\n"
                "chain-oldest-first: (entry 0 = the run that touched silicon)\n"
                + "".join(f"  {r} pin:{p}\n" for r, p in chain)
                + "checks: source-root provenance (clean markers + readable "
                "pin record + craq-gate taint parity) + jobkey equality "
                "(node/flags/extra_env/tag/mode) + archived .text hash set "
                "== this run's classify hashes + green rc/'passed' log\n"
            )
            if prov["taint"]:
                # Taint-matched adoption (this run also runs
                # --skip-craq-gate): the source root's taint line must
                # follow the cells it produced (ledger 8(f): a skipped
                # gate can never be mistaken for a green one).
                with (self.ev / "MANIFEST.txt").open("a") as f:
                    f.write(
                        f"CRAQ gate: ADOPTED-CELL TAINT — {rel} reused-from "
                        f"{prev} whose preflight records craq_gate_skipped; "
                        "this run runs --skip-craq-gate itself, taint "
                        "propagated\n"
                    )
            if not hasattr(self, "reused"):
                self.reused = []
            self.reused.append(
                {
                    "leg": str(rel),
                    "reused_from": str(prev),
                    "source_pin": prov["pin"][:12],
                    "source_taint": prov["taint"],
                    "origin_root": chain[0][0],
                }
            )
            print(
                f"reuse: {rel} reused-from:{prev} "
                f"source_pin:{prov['pin'][:12]} source_taint:{prov['taint']} "
                f"origin:{chain[0][0]} (root provenance + jobkey + .text "
                "hash-matched)"
            )
            return prev
        return None

    def _job_cached(self, job):
        """Quiet twin of _device_job's keyed hash-matched resume check,
        extended with the --prev-run cross-run adoption probe."""
        if self.a.force:
            return False
        work = job["work"]
        jobkey = self._job_key(job)
        exp = self._classify_texts(job["row"], job["sel"], job["leg"])
        if self._cell_green(work, jobkey, exp):
            return True
        return self._adopt_prev_cell(work, jobkey, exp) is not None

    def _batched_leg_verdict(self, row, sel, label, leg, work, jobkey, expected_texts):
        """Assembly-side verdict on evidence the batched executor produced
        (never executes; mirrors the legacy post-run RED semantics)."""
        rcf = work / "rc.txt"
        if not rcf.is_file():
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: batched executor produced "
                "no evidence for this leg (session failed before it ran, or "
                "executor/assembly leg enumeration diverged)"
            )
            return 99
        try:
            rc = int(rcf.read_text().strip() or 99)
        except ValueError:
            rc = 99
        if rc != 0 or not self._passed(work / "log.txt"):
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: device job FAIL rc={rc}"
            )
            return rc or 99
        try:
            cached_key = json.loads((work / "jobkey.json").read_text())
        except (ValueError, OSError):
            cached_key = None
        if cached_key != jobkey:
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: batched evidence jobkey "
                "mismatch (executor/assembly skew) — cell not trusted"
            )
            return 99
        archived = (
            self._texts_of(work / "TEXT_HASHES.txt")
            if (work / "TEXT_HASHES.txt").is_file()
            else None
        )
        if expected_texts is not None and archived != expected_texts:
            self.reds.append(
                f"{row['op']}/{sel} {label}-{leg}: device job .text differs "
                "from this run's classify build (non-deterministic build?)"
            )
        return rc

    def _write_failed_leg(self, job, rc, msg):
        """Executor-side failure evidence for a leg that never ran (group
        producer failure, unmappable ELFs).  No 'passed' text: assembly
        raises the RED through the legacy path."""
        work = job["work"]
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        (work / "node.txt").write_text(job["node"] + "\n")
        (work / "flags.txt").write_text(job["flags"] + "\n")
        (work / "jobkey.json").write_text(
            json.dumps(self._job_key(job), indent=2) + "\n"
        )
        (work / "log.txt").write_text(f"batched leg NOT RUN: {msg}\n")
        (work / "rc.txt").write_text(f"{rc}\n")

    def _xdist_available(self):
        if not hasattr(self, "_xdist_ok"):
            self._xdist_ok = (
                subprocess.run(
                    [str(self.python), "-c", "import xdist"], capture_output=True
                ).returncode
                == 0
            )
        return self._xdist_ok

    def _group_build(self, gdir, flags, extra_env, jobs):
        """ONE compile pass for a group's distinct nodes into a shared
        RUNNER_TEMP.  Seeded from a verified corpus_leg_store build when the
        exact (cc1plus, arch, flags, tt-metal head, farm path) matches and
        it covers every leg's classify relpaths; otherwise one
        --compile-producer session (parallel when xdist is available).

        Returns {'rt', 'manifest', 'seeded', 'producer_rc'}.  A non-zero
        producer_rc NEVER discards the build tree: pytest without -x keeps
        compiling the other variants after a failure, so the manifest still
        names every ELF that DID build — the caller attributes the failure
        to the specific legs whose ELFs are missing (storm-first-silicon
        all-withheld lesson: one ICE variant must not poison its group)."""
        rt = gdir / "rt"
        nodes = sorted({j["node"] for j in jobs})
        needed = set()
        for j in jobs:
            entries = self._classify_entries(j["row"], j["sel"], j["leg"])
            if entries:
                needed.update(rel for rel, _, _ in entries)
        seeded = False
        try:
            import importlib.util as _ilu

            spec = _ilu.spec_from_file_location(
                "corpus_leg_store", HERE / "corpus_leg_store.py"
            )
            store = _ilu.module_from_spec(spec)
            spec.loader.exec_module(store)
            build = store.find_build(
                store.DEFAULT_STORE,
                self.info["cc1plus_sha256"],
                "bh",
                flags,
                self.info["tt_metal_head"],
                str(ROOT.resolve()),
            )
            if build and needed and all((build / rel).is_file() for rel in needed):
                shutil.rmtree(rt, ignore_errors=True)
                rt.mkdir(parents=True)
                rc = subprocess.run(
                    ["cp", "-al", str(build), str(rt / "tt-llk-build")],
                    capture_output=True,
                ).returncode
                seeded = rc == 0
                if seeded:
                    print(
                        f"batched: group build seeded from store {build} "
                        f"({len(needed)} ELFs verified present)"
                    )
        except Exception as exc:  # seeding is opportunistic, never load-bearing
            print(f"batched: store seeding unavailable ({exc}); compiling")
        producer_rc = 0
        if not seeded:
            shutil.rmtree(rt, ignore_errors=True)
            rt.mkdir(parents=True)
            cmd = [str(self.python), "-m", "pytest", "-q", "--compile-producer"]
            if self._xdist_available():
                cmd += ["-n", "8"]
            cmd += nodes
            log = gdir / "producer.log"
            with open(log, "w") as f:
                try:
                    producer_rc = subprocess.run(
                        cmd,
                        cwd=PYDIR,
                        env=self._env("bh", rt, flags, extra=dict(extra_env)),
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        timeout=900 + 90 * len(nodes),
                    ).returncode
                except subprocess.TimeoutExpired:
                    producer_rc = 124
                    f.write("\nGROUP PRODUCER SESSION TIMED OUT\n")
            if producer_rc != 0:
                print(
                    f"batched: group producer session FAILED rc={producer_rc} "
                    f"({log}); hashing the tree anyway — only legs whose "
                    "variants are missing will be withheld"
                )
        entries = self._hash_build(rt, gdir / "TEXT_HASHES-group.txt")
        return {
            "rt": rt,
            "manifest": {rel: (t, e) for rel, t, e in entries},
            "seeded": seeded,
            "producer_rc": producer_rc,
        }

    def _producer_coverage(self, gjobs, gctx, gdir):
        """Per-leg failure attribution after a FAILED group producer
        session: a leg proceeds iff EVERY classify relpath it needs is
        present in the group build manifest (its variant provably
        compiled); a leg with missing ELFs — or with no classify map to
        prove coverage — fails CLOSED with its OWN per-leg evidence.  The
        whole group is never failed as a unit (storm-first-silicon lesson:
        the pin-12 counted-row ICE failed 2/117 producer compiles and the
        old rc-gate withheld all 33 rows).  Returns the runnable subset."""
        ok = []
        rc = gctx["producer_rc"]
        for j in gjobs:
            entries = self._classify_entries(j["row"], j["sel"], j["leg"])
            if entries is None:
                self._write_failed_leg(
                    j,
                    96,
                    f"group producer session failed (rc={rc}) and this leg "
                    "has no classify hash file to prove its variant "
                    f"compiled — failing closed (see {gdir}/producer.log)",
                )
                continue
            missing = [rel for rel, _, _ in entries if rel not in gctx["manifest"]]
            if missing:
                self._write_failed_leg(
                    j,
                    96,
                    "this leg's variant did not compile in the group "
                    f"producer session (rc={rc}; {len(missing)} of "
                    f"{len(entries)} ELFs missing, first: {missing[0]}; "
                    f"see {gdir}/producer.log)",
                )
                continue
            ok.append(j)
        return ok

    def _archive_group_subset(self, rt, rels, dest):
        """Per-leg ELF archive: the leg's classify relpaths (plus their
        variant dirs' build.h) copied out of the shared group build."""
        build = rt / "tt-llk-build"
        seen_vdirs = set()
        for rel in rels:
            src = build / rel
            if not src.is_file():
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            vdir = src.parent.parent
            if vdir in seen_vdirs:
                continue
            seen_vdirs.add(vdir)
            for bh in vdir.rglob("build.h"):
                bout = dest / bh.relative_to(build)
                bout.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bh, bout)

    @staticmethod
    def _filter_perf_csv(src, dst, mathop):
        """Write dst with only the rows whose mathop column names this
        leg's op (token match on the enum suffix).  A CSV without a mathop
        column cannot be split: returns False (caller records the leg
        failure) — never guesses row ownership."""
        with src.open() as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "mathop" not in reader.fieldnames:
                return False
            rows = [
                r for r in reader if (r.get("mathop") or "").split(".")[-1] == mathop
            ]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=reader.fieldnames)
            w.writeheader()
            w.writerows(rows)
        return True

    def _split_batch_session(self, sdir, jobs, session_rc, gctx):
        """Split one consumer session's outputs back into the per-leg
        evidence layout (unchanged vs legacy: node.txt, flags.txt,
        jobkey.json, log.txt, rc.txt, TEXT_HASHES.txt, elf/, perf CSVs)."""
        reports = {}
        rj = sdir / "report.json"
        if rj.is_file():
            try:
                reports = json.loads(rj.read_text()).get("reports", {})
            except ValueError:
                reports = {}
        file_count = {}
        for j in jobs:
            file_count[j["file"]] = file_count.get(j["file"], 0) + 1
        for job in jobs:
            work = job["work"]
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            (work / "node.txt").write_text(job["node"] + "\n")
            (work / "flags.txt").write_text(job["flags"] + "\n")
            (work / "jobkey.json").write_text(
                json.dumps(self._job_key(job), indent=2) + "\n"
            )
            (work / "session.txt").write_text(str(sdir) + "\n")
            phases = reports.get(job["node"]) or {}
            passed = (
                bool(phases)
                and "call" in phases
                and all(p.get("outcome") == "passed" for p in phases.values())
            )
            rc = 0 if passed else (1 if phases else 98)
            notes = []
            entries = self._classify_entries(job["row"], job["sel"], job["leg"])
            if entries is None:
                rc = rc or 97
                notes.append(
                    "classify hash file missing — cannot map this node's "
                    "ELFs out of the group build"
                )
            else:
                lines, missing = [], []
                for rel, _t, _e in entries:
                    got = gctx["manifest"].get(rel)
                    if got is None:
                        missing.append(rel)
                    else:
                        lines.append(f"{rel}\ttext:{got[0]}\telf:{got[1]}")
                (work / "TEXT_HASHES.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
                if missing:
                    rc = rc or 97
                    notes.append(
                        f"group build lacks {len(missing)} of this leg's "
                        f"classify ELFs (first: {missing[0]})"
                    )
                self._archive_group_subset(
                    gctx["rt"], [rel for rel, _, _ in entries], work / "elf"
                )
            if job["kind"] == "perf":
                base = pathlib.Path(job["file"]).stem
                src = sdir / "perf_data" / base
                solo = file_count[job["file"]] == 1
                if src.is_dir():
                    if solo:
                        shutil.copytree(src, work / "perf_data" / base)
                        raw = sdir / "raw_perf_data"
                        if raw.is_dir():
                            (work / "raw_perf_data").mkdir(exist_ok=True)
                            for f in sorted(raw.glob(f"{base}.*")):
                                shutil.copy2(f, work / "raw_perf_data" / f.name)
                    else:
                        ok_all = True
                        for f in sorted(src.glob("*.csv")):
                            ok_all &= self._filter_perf_csv(
                                f, work / "perf_data" / base / f.name, job["mathop"]
                            )
                        raw = sdir / "raw_perf_data"
                        if raw.is_dir():
                            for f in sorted(raw.glob(f"{base}.*.csv")):
                                (work / "raw_perf_data").mkdir(exist_ok=True)
                                self._filter_perf_csv(
                                    f, work / "raw_perf_data" / f.name, job["mathop"]
                                )
                        if not ok_all:
                            rc = rc or 97
                            notes.append(
                                "module CSV lacks a mathop column — rows "
                                "cannot be attributed to this leg (partition "
                                "bug: same-file legs without tokens shared a "
                                "session?)"
                            )
                elif passed:
                    notes.append(
                        "session produced no perf_data for this module "
                        "(session killed before module teardown?)"
                    )
            log_lines = [
                f"batched consumer session: {sdir}",
                f"session rc: {session_rc}",
                f"node: {job['node']}",
                f"phases: {json.dumps(phases, sort_keys=True)}",
                *notes,
            ]
            if rc == 0:
                log_lines.append(
                    "1 passed (batched split; session log has the full run)"
                )
            else:
                log_lines.append(
                    f"batched leg outcome rc={rc} "
                    f"({'node ran' if phases else 'node produced no outcome'}; "
                    "see session log)"
                )
            (work / "log.txt").write_text("\n".join(log_lines) + "\n")
            (work / "rc.txt").write_text(f"{rc}\n")

    def _run_batch_session(self, gctx, gdir, name, jobs, flags, extra_env):
        """ONE consumer pytest session for a set of legs: single dual-flock
        acquisition, prebuilt tree (--compile-consumer), per-node outcomes
        via the checked-in corpus pytest reporter, CSVs copied in-lock,
        outputs split per leg afterwards."""
        sdir = gdir / name
        shutil.rmtree(sdir, ignore_errors=True)
        sdir.mkdir(parents=True)
        jobs = sorted(jobs, key=lambda j: (j["file"], j["node"], j["sel"]))
        # Module-contiguous node order: the perf report fixture is module-
        # scoped and re-entering a module would unlink+rewrite its CSV.
        nodes = []
        for j in jobs:
            if j["node"] not in nodes:
                nodes.append(j["node"])
        # Node ids reach pytest via the line-oriented nodes.txt argfile
        # expanded with bash mapfile — no sh quoting layer ever parses them
        # (the pin-14 sweep killer: SdpaFwOp parametrization reprs carry
        # single quotes, spaces and parens).  The one impossible byte is a
        # newline: it would split an argfile row.
        for n in nodes:
            if "\n" in n or "\r" in n:
                sys.exit(
                    f"pytest node id contains a newline (breaks the node-id "
                    f"argfile): {n!r}"
                )
        (sdir / "nodes.txt").write_text("\n".join(nodes) + "\n")
        env_prefix = " ".join(f'{k}="{v}"' for k, v in sorted(extra_env))
        timeout_s = 600 + 300 * len(nodes)
        script = f"""#!/usr/bin/env bash
rm -rf "{LLK}/perf_data" "{gctx['rt']}/tt-llk-build/temp_perf_data"
cd "{PYDIR}" || exit 97
mapfile -t NODES < "{sdir}/nodes.txt"
env {env_prefix} CHIP_ARCH=blackhole LLK_HOME="{LLK}" RUNNER_TEMP="{gctx['rt']}" \\
TT_LLK_EXTRA_COMPILER_OPTIONS="{flags}" \\
SFPU_CORPUS_PYTEST_REPORT="{sdir}/report.json" \\
PYTHONPATH="{HERE}:$PYTHONPATH" \\
timeout {timeout_s} "{self.python}" -m pytest -q -v -p sfpu_corpus_pytest_plugin --compile-consumer "${{NODES[@]}}" > "{sdir}/log.txt" 2>&1
RC=$?
echo $RC > "{sdir}/rc.txt"
# copy raw+post perf CSVs IN-LOCK immediately (they are overwritten per run)
if [ -d "{LLK}/perf_data" ]; then cp -r "{LLK}/perf_data" "{sdir}/perf_data"; fi
if [ -d "{gctx['rt']}/tt-llk-build/temp_perf_data" ]; then cp -r "{gctx['rt']}/tt-llk-build/temp_perf_data" "{sdir}/raw_perf_data"; fi
exit $RC
"""
        session_sh = sdir / "session.sh"
        session_sh.write_text(script)
        session_sh.chmod(0o755)
        print(
            f"batched session {gdir.name}/{name}: {len(nodes)} node(s), "
            f"{len(jobs)} leg(s)"
        )
        subprocess.run(
            [
                "flock",
                "-x",
                DEVICE_LOCK,
                "-c",
                f"flock -x {SILICON_LOCK} -c {shlex.quote(str(session_sh))}",
            ],
            check=False,
        )
        session_rc = (
            int((sdir / "rc.txt").read_text().strip())
            if (sdir / "rc.txt").is_file()
            else 99
        )
        self._split_batch_session(sdir, jobs, session_rc, gctx)

    def _batched_silicon(self, gated, wave=None):
        """Executor entry: gated = [(row, classifications)] rows the silicon
        gates admitted.  Produces every pending leg's evidence; assembly
        (silicon()) then consumes it via the keyed resume path.

        `wave` scopes the session dirs (silicon-batches/<wave>/...) under
        pipelined rolling admission: each admitted wave is re-planned and
        executed independently, so group dir names never collide across
        waves while per-LEG evidence stays at its one canonical path."""
        jobs = []
        for row, cls in gated:
            jobs.extend(self._silicon_jobs(row, cls))
        if not jobs:
            return
        pending = [j for j in jobs if not self._job_cached(j)]
        groups = {}
        for j in pending:
            key = (j["flags"], tuple(sorted(j["extra_env"].items())))
            groups.setdefault(key, []).append(j)

        def gorder(key):
            rank = 0 if key[0] == OFF_FLAGS else (1 if key[0] == ON_FLAGS else 2)
            return (rank, key[0], key[1])

        gkeys = sorted(groups, key=gorder)
        broot = self.ev / "silicon-batches"
        if wave is not None:
            broot = broot / str(wave)
        broot.mkdir(parents=True, exist_ok=True)
        gnames = {
            key: f"g{i}-{hashlib.sha256((key[0] + repr(key[1])).encode()).hexdigest()[:8]}"
            for i, key in enumerate(gkeys)
        }
        plan = [
            f"batched silicon plan: {len(jobs)} legs total, "
            f"{len(jobs) - len(pending)} cached (keyed+hash-matched), "
            f"{len(pending)} to run across {len(gkeys)} group(s)"
        ]
        gspecs = {}
        for key in gkeys:
            gjobs = groups[key]
            specs = {}
            for j in gjobs:
                if j["kind"] == "perf":
                    specs[(j["op"], j["sel"], j["leg"])] = {
                        "file": j["file"],
                        "mathop": j["mathop"],
                        "op": j["op"],
                        "sel": j["sel"],
                        "leg": j["leg"],
                    }
            gspecs[key] = specs
            parts = partition_perf_legs(list(specs.values()))
            corr_n = sum(1 for j in gjobs if j["kind"] == "corr")
            plan.append(
                f"  group {gnames[key]}: {len({j['node'] for j in gjobs})} "
                f"node(s), {corr_n} corr leg(s), {len(specs)} perf leg-spec(s) "
                f"in {len(parts)} CSV partition(s) x {PERF_RUNS} reps; "
                f"extra_env={dict(key[1])}; flags: {key[0][:80]}..."
            )
        (broot / "PLAN.txt").write_text("\n".join(plan) + "\n")
        for line in plan:
            print(line)
        if self.a.dry_run:
            print("DRY-RUN: batched sessions planned, not executed")
            return
        # 1. one compile pass per group (no device, no flocks)
        gctxs = {}
        for key in gkeys:
            self.verify_toolchain("silicon-batch-build")
            gdir = broot / gnames[key]
            gdir.mkdir(parents=True, exist_ok=True)
            gctx = self._group_build(gdir, key[0], key[1], groups[key])
            if gctx["producer_rc"] != 0:
                # Per-leg attribution, never a group verdict: only the legs
                # whose variants provably failed to compile are withheld;
                # every leg whose classify ELF set is fully present in the
                # group build still runs and books its cells.
                survivors = self._producer_coverage(groups[key], gctx, gdir)
                print(
                    f"batched: group {gnames[key]} producer failed "
                    f"(rc={gctx['producer_rc']}): "
                    f"{len(groups[key]) - len(survivors)} leg(s) withheld "
                    f"individually, {len(survivors)} leg(s) verified "
                    "compiled and proceeding"
                )
                groups[key] = survivors
                if not survivors:
                    continue
            gctxs[key] = gctx
        # 2. correctness sessions (one per group, before ANY perf session)
        for key in gkeys:
            if key not in gctxs:
                continue
            corr = [j for j in groups[key] if j["kind"] == "corr"]
            if corr:
                self.verify_toolchain("silicon-batch-corr")
                self._run_batch_session(
                    gctxs[key], broot / gnames[key], "corr", corr, key[0], key[1]
                )
        # 3. rows whose correctness failed get every perf leg withheld
        #    (assembly reproduces the legacy STOP note from the corr rc)
        failed_ops = set()
        for row, cls in gated:
            for j in jobs:
                if j["op"] != row["op"] or j["kind"] != "corr":
                    continue
                rcf = j["work"] / "rc.txt"
                try:
                    rc = int(rcf.read_text().strip() or 99) if rcf.is_file() else 99
                except ValueError:
                    rc = 99
                if rc != 0 or not self._passed(j["work"] / "log.txt"):
                    failed_ops.add(row["op"])
                    break
        # 4. perf sessions: reps x (OFF, ON, ... alternating) x partitions;
        #    partitions computed once per group => identical composition in
        #    every repetition (determinism), 3 fresh processes per rep.
        gparts = {}
        for key in gkeys:
            if key not in gctxs:
                continue
            live = [s for s in gspecs[key].values() if s["op"] not in failed_ops]
            gparts[key] = partition_perf_legs(live)
        maxp = max((len(p) for p in gparts.values()), default=0)
        for r in range(1, PERF_RUNS + 1):
            for p in range(maxp):
                for key in gkeys:
                    parts = gparts.get(key)
                    if not parts or p >= len(parts):
                        continue
                    part_ids = {(s["op"], s["sel"], s["leg"]) for s in parts[p]}
                    legs = [
                        j
                        for j in groups[key]
                        if j["kind"] == "perf"
                        and j["rep"] == r
                        and (j["op"], j["sel"], j["leg"]) in part_ids
                    ]
                    if legs:
                        self.verify_toolchain("silicon-batch-perf")
                        self._run_batch_session(
                            gctxs[key],
                            broot / gnames[key],
                            f"r{r}-p{p}",
                            legs,
                            key[0],
                            key[1],
                        )

    def silicon_pinpair(self, row, classifications):
        """kind=pinpair: paired gen-vs-hand A/B at the row's pinned flag set
        (e.g. Reduce-SDPA at the default profitability gate).  Same pipeline
        discipline as the 2x2: correctness first, then 3 fresh processes per
        selector alternating gen/hand, hash-matched resume per job."""
        result = self._result_skeleton(row, classifications)
        self._macro_lb_gate(row, classifications, result)
        if self.a.dry_run:
            result["notes"].append(
                "DRY-RUN: device jobs printed, not executed; no cells expected"
            )
        flags = row["pin_flags"]
        result["notes"].append(f"pinpair leg flags: {flags}")
        for sel in ("sem-corr", "hand-corr"):
            if not row["nodes"][sel]:
                continue
            rc = self._device_job(
                row,
                sel,
                "corr",
                "default",
                flags,
                expected_texts=self._classify_texts(row, sel, "default"),
            )
            result["runs"][f"{sel}/corr-default"] = (
                "PASS" if rc == 0 else f"FAIL(rc={rc})"
            )
            if rc != 0:
                result["notes"].append(f"STOP: {sel} correctness failed; perf withheld")
                return result
        samples = {sel: [] for sel in ("sem-perf", "hand-perf")}
        for r in range(1, PERF_RUNS + 1):
            for sel in ("sem-perf", "hand-perf"):  # alternating gen/hand
                if not row["nodes"][sel]:
                    continue
                self._device_job(
                    row,
                    sel,
                    f"r{r}",
                    "default",
                    flags,
                    expected_texts=self._classify_texts(row, sel, "default"),
                )
                val = self._perf_value(row, sel, f"r{r}", "default")
                if val is not None:
                    samples[sel].append(val)
        for sel, cell in PINPAIR_CELLS.items():
            src = samples[sel]
            result["runs"][f"{sel}/{cell}_samples"] = src
            result["cells"][cell] = (sum(src) / len(src)) if src else None
        self._issue_slot_check(row, result)
        c = result["cells"]
        gen, hand = c.get("generated"), c.get("handwritten_replay")
        if isinstance(gen, (int, float)) and isinstance(hand, (int, float)) and hand:
            result["vs_hand_pct"] = 100.0 * (gen - hand) / hand
        self._measured_flags_gate(row, classifications, result)
        return result

    def silicon(self, row, classifications):
        if row["kind"] == "pinpair":
            return self.silicon_pinpair(row, classifications)
        result = self._result_skeleton(row, classifications)
        self._macro_lb_gate(row, classifications, result)
        if self.a.dry_run:
            # A dry run proves gate wiring, never metrics: mark the row so
            # report() treats its empty cells as blocked-by-design instead
            # of INVALID_METRIC RED.
            result["notes"].append(
                "DRY-RUN: device jobs printed, not executed; no cells expected"
            )
        # correctness first, OFF then ON; byte-identical pair => one run fills both
        for sel in ("sem-corr", "hand-corr"):
            if not row["nodes"][sel]:
                continue
            cls = classifications.get(sel, {})
            legs = ["off"] if cls.get("all") == "IDENTICAL" else ["off", "on"]
            if len(legs) == 1:
                result["notes"].append(
                    f"{sel}: OFF==ON byte-identical — one correctness run fills both legs"
                )
            for leg in legs:
                rc = self._device_job(
                    row,
                    sel,
                    "corr",
                    leg,
                    OFF_FLAGS if leg == "off" else ON_FLAGS,
                    expected_texts=self._classify_texts(row, sel, leg),
                )
                result["runs"][f"{sel}/corr-{leg}"] = (
                    "PASS" if rc == 0 else f"FAIL(rc={rc})"
                )
                if rc != 0:
                    result["notes"].append(
                        f"STOP: {sel} correctness {leg} failed; perf withheld"
                    )
                    return result
        # perf: 3 fresh processes per leg, alternating OFF/ON
        for sel, cells in (
            ("sem-perf", ("sem_off", "sem_on")),
            ("hand-perf", ("hand_off", "hand_on")),
        ):
            if not row["nodes"][sel]:
                continue
            cls = classifications.get(sel, {})
            if cls.get("status") == "COMPILE_FAIL":
                result["cells"][cells[0]] = result["cells"][cells[1]] = None
                result["notes"].append(f"{sel}: COMPILE_FAIL — perf blocked")
                continue
            identical = cls.get("all") == "IDENTICAL"
            if identical and sel == "sem-perf":
                result["notes"].append(
                    "sem-perf OFF/ON byte-identical: recorded refusal, no device run"
                )
                result["cells"]["sem_off"] = result["cells"]["sem_on"] = (
                    "REFUSAL_BYTE_IDENTICAL"
                )
                continue
            legs = ["off"] if identical else ["off", "on"]
            if identical:
                result["notes"].append(
                    f"{sel}: OFF==ON byte-identical — one physical leg fills both cells"
                )
            samples = {leg: [] for leg in legs}
            for r in range(1, PERF_RUNS + 1):
                for leg in legs:  # alternating OFF/ON inside each round
                    self._device_job(
                        row,
                        sel,
                        f"r{r}",
                        leg,
                        OFF_FLAGS if leg == "off" else ON_FLAGS,
                        expected_texts=self._classify_texts(row, sel, leg),
                    )
                    val = self._perf_value(row, sel, f"r{r}", leg)
                    if val is not None:
                        samples[leg].append(val)
            for leg, cell in zip(("off", "on"), cells):
                src = samples[leg] if leg in samples else samples["off"]
                result["runs"][f"{sel}/{cell}_samples"] = src
                result["cells"][cell] = (sum(src) / len(src)) if src else None
        # marker validity first: an INVALID_MARKER cell must not feed a ratio
        self._issue_slot_check(row, result)
        # derived ratios
        c = result["cells"]
        num = lambda x: isinstance(x, (int, float))
        if num(c.get("sem_off")) and num(c.get("sem_on")) and c["sem_off"]:
            result["causal_pct"] = 100.0 * (c["sem_on"] - c["sem_off"]) / c["sem_off"]
        if num(c.get("sem_on")) and num(c.get("hand_on")) and c["hand_on"]:
            result["vs_hand_pct"] = 100.0 * (c["sem_on"] - c["hand_on"]) / c["hand_on"]
        self._measured_flags_gate(row, classifications, result)
        return result

    # ---------------- weekly: per-knob attribution ----------------
    def attribute_knobs(self, row, classifications):
        if row["kind"] == "pinpair":
            return {"op": row["op"], "status": "SKIP_PINPAIR"}
        sel = "sem-perf" if row["nodes"]["sem-perf"] else "sem-corr"
        if (
            not row["nodes"][sel]
            or classifications.get(sel, {}).get("all") != "CHANGED"
        ):
            return {"op": row["op"], "status": "SKIP_NOT_CHANGED"}
        firing = []
        for knob, flag in KNOBS.items():
            verdict = self.classify(
                row,
                sel,
                legs=(("off", OFF_FLAGS), ("knob", f"{OFF_FLAGS} {flag}")),
                tag=f"knobs/{knob}",
            )
            if verdict.get("all") == "CHANGED":
                firing.append(knob)
        out = {
            "op": row["op"],
            "selector": sel,
            "status": "OK",
            "firing_knobs": firing,
            "single_knob_attribution": firing[0] if len(firing) == 1 else None,
        }
        (self.ev / row["op"] / "knob-attribution.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )
        return out

    def knob_silicon(self, row, attribution):
        """Per-knob silicon legs (weekly, headline rows only): OFF vs OFF+knob.

        D3 fix (PULL_ANALYSIS-20260817): these legs run the IDENTICAL
        classify -> paired CRAQ -> correctness-then-perf pipeline as the main
        legs.  Per firing knob: the perf selector's OFF-vs-knob classification
        must be CHANGED (byte-identical => recorded refusal, no device run);
        the correctness selector is classified and paired-CRAQ'd with the same
        OFF-vs-knob legs and the BH gate must be green; device correctness
        runs OFF then knob BEFORE any perf leg; only then 3 fresh perf
        processes per leg, alternating, hash-matched like every device job.
        Callers must invoke this only for rows whose MAIN BH CRAQ gate is
        already green (enforced in run())."""
        sel = attribution.get("selector")
        if attribution.get("status") != "OK" or not sel:
            return
        corr_sel = "sem-corr" if row["nodes"]["sem-corr"] else None
        out = {}
        for knob in attribution.get("firing_knobs", []):
            knob_flags = f"{OFF_FLAGS} {KNOBS[knob]}"
            legs_spec = (("off", OFF_FLAGS), ("knob", knob_flags))
            entry = {"selector": sel, "flags": knob_flags}
            out[knob] = entry
            # 1. classification (perf selector; already produced by
            #    attribute_knobs — classify() resumes hash-matched).
            cls = self.classify(row, sel, legs=legs_spec, tag=f"knobs/{knob}")
            entry["classify"] = cls
            if cls.get("status") != "OK":
                entry["status"] = "CLASSIFY_FAIL"
                continue
            if cls.get("all") == "IDENTICAL":
                entry["status"] = "REFUSAL_BYTE_IDENTICAL"  # no device run
                continue
            if not corr_sel:
                entry["status"] = "WITHHELD_NO_CORR_NODE"
                self.reds.append(
                    f"{row['op']}/{knob}: knob silicon withheld — no correctness node"
                )
                continue
            # 2. correctness-selector classification (for its own byte-identity
            #    handling and the hash-matched device resume below).
            corr_cls = self.classify(
                row, corr_sel, legs=legs_spec, tag=f"knobs/{knob}-corr"
            )
            entry["classify_corr"] = corr_cls
            if corr_cls.get("status") != "OK":
                entry["status"] = "CLASSIFY_FAIL"
                continue
            # 3. paired CRAQ on the correctness node, same OFF-vs-knob legs;
            #    the BH gate must be green (SKIP_NO_SIMULATOR never opens it).
            bh_verdict = None
            for arch in row["craq_archs"].split(","):
                arch = arch.strip()
                v = self.craq(
                    row, corr_sel, arch, legs_spec=legs_spec, tag=f"knobs-craq/{knob}"
                )
                if arch == "bh":
                    bh_verdict = v
            gate = bool(
                bh_verdict
                and bh_verdict.get("legs")
                and all(x == "PASS" for x in bh_verdict["legs"].values())
            )
            entry["craq_bh"] = bh_verdict
            if not gate and not self.a.skip_craq_gate:
                entry["status"] = "WITHHELD_CRAQ_NOT_GREEN"
                self.reds.append(
                    f"{row['op']}/{knob}: knob silicon withheld — paired BH CRAQ not green"
                )
                continue
            # 4. device correctness FIRST (OFF then knob; byte-identical corr
            #    pair => one run fills both legs, like the main pipeline).
            tag = f"knobs-silicon/{knob}"
            corr_legs = (
                [("off", OFF_FLAGS)]
                if corr_cls.get("all") == "IDENTICAL"
                else list(legs_spec)
            )
            corr_fail = False
            for leg, flags in corr_legs:
                rc = self._device_job(
                    row,
                    corr_sel,
                    "corr",
                    leg,
                    flags,
                    tag=tag,
                    expected_texts=self._classify_texts(
                        row, corr_sel, leg, tag=f"knobs/{knob}-corr"
                    ),
                )
                entry[f"corr_{leg}"] = "PASS" if rc == 0 else f"FAIL(rc={rc})"
                if rc != 0:
                    corr_fail = True
                    break
            if corr_fail:
                entry["status"] = "STOP_CORRECTNESS_FAILED"
                self.reds.append(
                    f"{row['op']}/{knob}: knob correctness failed; perf withheld"
                )
                continue
            # 5. perf: 3 fresh processes per leg, alternating OFF/knob.
            samples = {"off": [], "knob": []}
            for r in range(1, PERF_RUNS + 1):
                for leg, flags in legs_spec:
                    self._device_job(
                        row,
                        sel,
                        f"r{r}",
                        leg,
                        flags,
                        tag=tag,
                        expected_texts=self._classify_texts(
                            row, sel, leg, tag=f"knobs/{knob}"
                        ),
                    )
                    val = self._perf_value(row, sel, f"r{r}", leg, tag=tag)
                    if val is not None:
                        samples[leg].append(val)
            cell = {leg: (sum(v) / len(v)) if v else None for leg, v in samples.items()}
            if cell["off"] and cell["knob"]:
                cell["delta_pct"] = 100.0 * (cell["knob"] - cell["off"]) / cell["off"]
            entry["cells"] = cell
            entry["status"] = "OK"
        (self.ev / row["op"] / "knob-silicon.json").write_text(
            json.dumps(out, indent=2) + "\n"
        )

    # ---------------- scoreboard / manifest ----------------
    def emit_scoreboard(self, results, skips):
        payload = {
            "provenance": self.info,
            "results": results,
            "skips": skips,
            "reds": self.reds,
            # cross-run adopted device cells (also marked per-leg by
            # REUSED_FROM.txt in each adopted evidence dir)
            "reused_cells": getattr(self, "reused", []),
        }
        (self.ev / "scoreboard.json").write_text(json.dumps(payload, indent=2) + "\n")
        cc1 = self.info["cc1plus_sha256"]
        sim_sha = self.info.get("sim_bh_sha256", "")
        with (self.ev / "scoreboard.tsv").open("w") as f:
            f.write(
                "# schema=2; chip-class silicon cells from sweep_2x2.py; "
                "compiler_sha = cc1plus binary sha256 (PRIMARY toolchain pin), "
                "craq_sim_sha = BH libttsim sha256\n"
            )
            f.write(
                "id\tarch\tmetric\tscope\tselector\tcycles\tstatus\t"
                "compiler_sha\tcraq_sim_sha\tprovenance\n"
            )
            for r in results:
                for cell, val in r.get("cells", {}).items():
                    status = (
                        "measured"
                        if isinstance(val, (int, float))
                        else (val or "missing")
                    )
                    cyc = f"{val}" if isinstance(val, (int, float)) else ""
                    f.write(
                        f"{r['corpus_id']}\tbh\tdevice_cycles\t{r['scope']}\t"
                        f"{cell_selector(r, cell)}\t{cyc}\t{status}\t"
                        f"{cc1}\t{sim_sha}\t{self.ev.name}\n"
                    )
        lines = [
            "# 2x2 sweep scoreboard",
            "",
            f"- evidence: `{self.ev}`",
            f"- cc1plus sha256 (primary pin): `{self.info['cc1plus_sha256']}`",
            f"- driver sha256 (secondary): `{self.info['compiler_sha256']}`",
            "",
            "| op | marker | sem OFF | sem ON | causal | hand | vs hand | notes |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else (v or "—")
        for r in results:
            c = r.get("cells", {})
            if r["kind"] == "pinpair":
                # gen-vs-hand pair at the row's pinned flag set: the generated
                # cell rides the "sem ON" column, hand is hand.
                so, sn = "—", fmt(c.get("generated"))
                h = fmt(c.get("handwritten_replay"))
            else:
                so, sn = fmt(c.get("sem_off")), fmt(c.get("sem_on"))
                h = fmt(c.get("hand_on", c.get("hand_off")))
            lines.append(
                "| {op} | {m} | {so} | {sn} | {cz} | {h} | {vh} | {n} |".format(
                    op=r["op"],
                    m=r["marker"],
                    so=so,
                    sn=sn,
                    cz=f"{r['causal_pct']:+.2f}%" if "causal_pct" in r else "—",
                    h=h,
                    vh=f"{r['vs_hand_pct']:+.2f}%" if "vs_hand_pct" in r else "—",
                    n="; ".join(r.get("notes", [])),
                )
            )
        for s in skips:
            lines.append(f"| {s['op']} | — | — | — | — | — | — | {s['reason']} |")
        (self.ev / "SCOREBOARD.md").write_text("\n".join(lines) + "\n")

    def emit_sha256sums(self):
        out = self.ev / "SHA256SUMS"
        entries = []
        for path in sorted(self.ev.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                entries.append(f"{sha256(path)}  {path.relative_to(self.ev)}")
        out.write_text("\n".join(entries) + "\n")

    # ---------------- report ----------------
    @staticmethod
    def _load_baseline(path):
        """Baseline TSV -> (cycles map, expected-class map).

        cycles:  (id, scope, selector) -> [floats]  (min = the established
                 three-process convention when aggregating repeats)
        classes: (id, scope) -> expected class from the schema-2
                 expected_class column (win/parity/loss/refusal); rows with
                 status 'refusal'/'expected_refusal' also declare refusal.
        Falls back to deriving win/parity/loss from the measured sem cells
        for schema-1 baselines without the column.
        """
        cycles, classes = {}, {}
        if not (path and path.is_file()):
            return cycles, classes
        with path.open() as f:
            for rec in csv.DictReader(
                (x for x in f if not x.startswith("#")), delimiter="\t"
            ):
                key2 = (rec["id"], rec["scope"])
                cls = (rec.get("expected_class") or "").strip()
                if cls:
                    classes.setdefault(key2, cls)
                if (rec.get("status") or "").strip() in (
                    "refusal",
                    "expected_refusal",
                    "refusal_byte_identical",
                ):
                    classes[key2] = "refusal"
                try:
                    cyc = float(rec.get("cycles", ""))
                except (TypeError, ValueError):
                    continue
                cycles.setdefault(
                    (rec["id"], rec["scope"], rec["selector"]), []
                ).append(cyc)
                # Legacy selector aliases (pin-12 retype): rows measured under
                # generated/handwritten_replay history keep their flip/drift
                # tripwires when re-keyed to the uniform sem_/hand_ cells.
                alias = {
                    "generated": "sem_on",
                    "handwritten_replay": "hand_on",
                    "handwritten_direct": "hand_off",
                }.get(rec["selector"])
                if alias:
                    op = rec["selector"]  # placeholder to keep black quiet
                    cycles.setdefault((rec["id"], rec["scope"], alias), []).append(cyc)
        return cycles, classes

    @staticmethod
    def _derived_class(baseline, r):
        """win/parity/loss from the baseline's measured sem cells (schema-1
        fallback when no expected_class column exists)."""
        off = baseline.get((r["corpus_id"], r["scope"], cell_selector(r, "sem_off")))
        on = baseline.get((r["corpus_id"], r["scope"], cell_selector(r, "sem_on")))
        if not (off and on and min(off)):
            return None
        pct = 100.0 * (min(on) - min(off)) / min(off)
        if pct < -0.5:
            return "win"
        if pct <= 0.5:
            return "parity"
        return "loss"

    _RAG_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}

    @classmethod
    def _worst_rag(cls, a, b):
        return a if cls._RAG_ORDER[a] >= cls._RAG_ORDER[b] else b

    @staticmethod
    def _row_class(r):
        """WIN/PARITY/LOSS band for a row's streamed verdict (the sem-vs-hand
        sign convention the scoreboard uses: <-0.5% win, <=+0.5% parity)."""
        c = r.get("cells") or {}
        if any(v == "REFUSAL_BYTE_IDENTICAL" for v in c.values()):
            return "REFUSAL"
        v = r.get("vs_hand_pct")
        if isinstance(v, (int, float)):
            return "WIN" if v < -0.5 else ("PARITY" if v <= 0.5 else "LOSS")
        return "UNMEASURED"

    def _row_verdict(self, r, baseline, base_classes, prev):
        """The per-row acceptance computation of report(), factored out so
        the silicon phase can STREAM each row's verdict the moment its
        cells complete (ROW-VERDICT.json) and the final REPORT.md is an
        aggregation of the IDENTICAL logic — the row line is byte-equal
        whether computed at completion time or at the end.  Pure over
        (result row, baseline, prev): no self.reds side effects, so
        streaming + final aggregation never double-book a RED.

        Returns {'scope', 'verdicts', 'col', 'rag'} (rag is row-local;
        report() folds it into the run verdict with _worst_rag)."""
        max_abs_drift_pct = getattr(self.a, "max_abs_drift_pct", 10.0)
        red_loss_growth_pct = getattr(self.a, "red_loss_growth_pct", 5.0)
        max_drift_pct = getattr(self.a, "max_drift_pct", 5.0)
        allow_win_to_parity = getattr(self.a, "allow_win_to_parity", False)
        verdicts = []
        rag = "GREEN"
        c = r.get("cells", {})
        scope = r.get("scope") or f"{r['marker']}_MATH_ISOLATE_PER_TILE"
        r = dict(r, scope=scope)
        expected = base_classes.get((r["corpus_id"], scope)) or self._derived_class(
            baseline, r
        )
        # acceptance 1 (class-aware, D4): a refusal is GREEN only when the
        # baseline class is refusal (or the row has no baseline history).
        # A row whose baseline carries a measured WIN that now collapses
        # to a byte-identical refusal is a total-refusal regression: RED.
        if c.get("sem_off") == "REFUSAL_BYTE_IDENTICAL":
            if expected == "win":
                verdicts.append(
                    "WIN→REFUSAL FLIP (baseline class win, now "
                    "byte-identical refusal — planner stopped firing): RED"
                )
                rag = "RED"
            elif expected in ("parity", "loss"):
                verdicts.append(
                    f"{expected.upper()}→REFUSAL: flagged notice "
                    "(measured baseline row now refuses): YELLOW"
                )
                if rag == "GREEN":
                    rag = "YELLOW"
            elif expected == "refusal":
                verdicts.append(
                    "refusal byte-identical (baseline class refusal): GREEN"
                )
            else:
                verdicts.append("refusal byte-identical (no baseline history): GREEN")
        elif expected == "refusal" and any(
            isinstance(v, (int, float)) for v in c.values()
        ):
            # refusal -> changed: not a regression by itself, but a class
            # transition that must be surfaced, never silently blessed.
            verdicts.append(
                "REFUSAL→CHANGED: flagged notice (baseline expects a "
                "byte-identical refusal, OFF/ON now differs and was "
                "measured): YELLOW"
            )
            if rag == "GREEN":
                rag = "YELLOW"
        # acceptance 1b: INVALID_MARKER cells can never gate GREEN.
        if any(v == "INVALID_MARKER" for v in c.values()):
            verdicts.append(
                "INVALID_MARKER cell(s) — reading below the payload "
                "issue-slot lower bound (KERNEL marker required): RED"
            )
            rag = "RED"
        # acceptance 1d (enforcement layer): a macro-launch row with an
        # EMPTY issue_slot_lb is RED — the §1 issue-slot check silently
        # no-ops on empty lb, under exactly the fire-and-forget shapes
        # that need it (wave-6 V3).
        if any("EMPTY issue_slot_lb" in n for n in r.get("notes", [])):
            verdicts.append(
                "MACRO-LAUNCH ROW WITHOUT issue_slot_lb — HANDOFF §1 "
                "metric caveat unenforceable on this row's cells: RED"
            )
            rag = "RED"
        # acceptance 1c (finding sweep_2x2.py:1276): a cell with baseline
        # history that produced NO parsable metric this run is
        # INVALID_METRIC RED — a profiler/post-CSV or marker rename must
        # never turn the nightly permanently GREEN while measuring
        # nothing.  Withheld/blocked rows already carry their own RED.
        blocked = any(
            "STOP" in n or "COMPILE_FAIL" in n or "withheld" in n or "DRY-RUN" in n
            for n in r.get("notes", [])
        )
        if not blocked:
            dead = sorted(
                cell
                for cell, v in c.items()
                if v is None
                and baseline.get((r["corpus_id"], scope, cell_selector(r, cell)))
            )
            numeric_any = any(isinstance(v, (int, float)) for v in c.values())
            refused = any(v == "REFUSAL_BYTE_IDENTICAL" for v in c.values())
            if dead:
                verdicts.append(
                    f"INVALID_METRIC — cell(s) {', '.join(dead)} have "
                    "baseline history but produced no parsable metric "
                    "(marker/post-CSV drift?): RED"
                )
                rag = "RED"
            elif expected and c and not numeric_any and not refused:
                verdicts.append(
                    "INVALID_METRIC — row has baseline class history but "
                    "every cell is unparsable/None: RED"
                )
                rag = "RED"
        # acceptance 2a (findings sweep_2x2.py:1222/:1181): per-cell
        # ABSOLUTE cycle drift vs the baseline's min-aggregated cycles.
        # Ratio-only acceptance is blind to uniform slowdowns (both legs
        # +50% keeps every ratio) and never checks the hand leg on
        # refusal rows.  Slowdowns beyond --max-abs-drift-pct are RED;
        # improvements beyond it are YELLOW (stale baseline — reviewed
        # update needed), never silently blessed.
        for cell in sorted(c):
            val = c[cell]
            if not isinstance(val, (int, float)):
                continue
            base = baseline.get((r["corpus_id"], scope, cell_selector(r, cell)))
            if not base or not min(base):
                continue
            abs_pct = 100.0 * (val - min(base)) / min(base)
            if abs_pct > max_abs_drift_pct:
                verdicts.append(
                    f"{cell} ABS CYCLES {min(base):g}→{val:g} "
                    f"({abs_pct:+.2f}% > {max_abs_drift_pct:g}%): RED"
                )
                rag = "RED"
            elif abs_pct < -max_abs_drift_pct:
                verdicts.append(
                    f"{cell} abs cycles improved {min(base):g}→{val:g} "
                    f"({abs_pct:+.2f}%; baseline stale — reviewed update "
                    "needed): YELLOW"
                )
                if rag == "GREEN":
                    rag = "YELLOW"
        # acceptance 2: win-sign preservation vs baseline
        for name, key in (("causal", "causal_pct"), ("vs_hand", "vs_hand_pct")):
            if key not in r:
                continue
            base_pair = None
            if r["kind"] == "pinpair":
                if name == "causal":
                    continue
                on = baseline.get((r["corpus_id"], scope, "generated"))
                hand = baseline.get((r["corpus_id"], scope, "handwritten_replay"))
                base_pair = (min(hand), min(on)) if on and hand else None
            elif name == "causal":
                off = baseline.get((r["corpus_id"], scope, cell_selector(r, "sem_off")))
                on = baseline.get((r["corpus_id"], scope, cell_selector(r, "sem_on")))
                base_pair = (min(off), min(on)) if off and on else None
            else:
                on = baseline.get((r["corpus_id"], scope, cell_selector(r, "sem_on")))
                hand = baseline.get(
                    (r["corpus_id"], scope, cell_selector(r, "hand_on"))
                )
                base_pair = (min(hand), min(on)) if on and hand else None
            if not base_pair or not base_pair[0]:
                verdicts.append(f"{name} {r[key]:+.2f}% (no baseline row)")
                continue
            base_pct = 100.0 * (base_pair[1] - base_pair[0]) / base_pair[0]
            drift = abs(r[key] - base_pct)
            if base_pct < 0 <= r[key]:  # win-sign flip (causal or vs-hand)
                verdicts.append(
                    f"{name} WIN→LOSS FLIP {base_pct:+.2f}%→{r[key]:+.2f}%: RED"
                )
                rag = "RED"
            elif base_pct <= -0.5 and r[key] > -0.5:
                # Finding sweep_2x2.py:1259 (fixture C): a real win
                # (class band <= -0.5%) eroding into the parity band is a
                # regression, not drift — RED by default; a full flip to
                # >= 0 is caught above.
                tag = "YELLOW" if allow_win_to_parity else "RED"
                verdicts.append(
                    f"{name} WIN→PARITY {base_pct:+.2f}%→{r[key]:+.2f}%: {tag}"
                )
                if tag == "RED":
                    rag = "RED"
                elif rag == "GREEN":
                    rag = "YELLOW"
            elif base_pct > 0.5 and (r[key] - base_pct) > red_loss_growth_pct:
                # Finding sweep_2x2.py:1259 (fixture D): an existing loss
                # growing beyond --red-loss-growth-pct percentage points
                # is RED (exit 1), not an unalertable YELLOW.
                verdicts.append(
                    f"{name} LOSS GREW {base_pct:+.2f}%→{r[key]:+.2f}% "
                    f"(+{r[key] - base_pct:.2f}pp > "
                    f"{red_loss_growth_pct:g}pp): RED"
                )
                rag = "RED"
            elif drift > max_drift_pct:
                verdicts.append(f"{name} drift {base_pct:+.2f}%→{r[key]:+.2f}%: YELLOW")
                if rag == "GREEN":
                    rag = "YELLOW"
            else:
                verdicts.append(
                    f"{name} {r[key]:+.2f}% vs baseline {base_pct:+.2f}%: GREEN"
                )
        if r["op"] in prev and "causal_pct" in r and "causal_pct" in prev[r["op"]]:
            verdicts.append(
                f"prev-run causal {prev[r['op']]['causal_pct']:+.2f}%→{r['causal_pct']:+.2f}%"
            )
        if any("STOP" in n or "COMPILE_FAIL" in n for n in r.get("notes", [])):
            verdicts.append("correctness/compile failure: RED")
            rag = "RED"
        # Verdict column carries YELLOW too (adversarial missed item:
        # a YELLOW row displaying 'ok' hid the one channel YELLOW has).
        col = (
            "RED"
            if any("RED" in v for v in verdicts)
            else ("YELLOW" if any("YELLOW" in v for v in verdicts) else "ok")
        )
        return {"scope": scope, "verdicts": verdicts, "col": col, "rag": rag}

    def _report_row_line(self, r, row):
        """The REPORT.md table line for one row verdict (shared by the
        streamed ROW-VERDICT.json and the final aggregation — byte-equal)."""
        return (
            f"| {r['op']} | {row['col']} | "
            f"{'; '.join(row['verdicts']) or 'no silicon cells this run'} |"
        )

    def _emit_row_verdict(self, result):
        """FIRST-CLASS ROW VERDICT STREAMING: write
        <evidence-root>/<op>/ROW-VERDICT.json the moment the row's silicon
        cells complete — the same computation report() applies at the end
        (cycles per leg, vs_hand %, WIN/PARITY/LOSS class, baseline drift
        verdicts), per row, at completion time.  The final REPORT.md is an
        aggregation of the identical _row_verdict logic."""
        if not hasattr(self, "_stream_ctx"):
            baseline, base_classes = self._load_baseline(
                getattr(self.a, "baseline", None)
            )
            self._stream_ctx = (baseline, base_classes, self._load_prev_results())
        baseline, base_classes, prev = self._stream_ctx
        row = self._row_verdict(result, baseline, base_classes, prev)
        payload = {
            "op": result["op"],
            "corpus_id": result.get("corpus_id"),
            "scope": row["scope"],
            "cells": result.get("cells", {}),
            "runs": result.get("runs", {}),
            "notes": result.get("notes", []),
            "causal_pct": result.get("causal_pct"),
            "vs_hand_pct": result.get("vs_hand_pct"),
            "class": self._row_class(result),
            "verdict": row["col"],
            "rag": row["rag"],
            "details": row["verdicts"],
            "baseline": str(getattr(self.a, "baseline", "") or ""),
            "report_row": self._report_row_line(result, row),
        }
        out = self.ev / result["op"] / "ROW-VERDICT.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    def _load_prev_results(self):
        """Previous-run results for the report's drift annotation: the
        NEWEST --prev-run root that carries a scoreboard.json (roots are
        given newest first)."""
        prev = {}
        for root in self._prev_roots():
            sb = pathlib.Path(root) / "scoreboard.json"
            if not sb.is_file():
                continue
            for r in json.loads(sb.read_text()).get("results", []):
                prev[r["op"]] = r
            break
        return prev

    def report(self, results, skips):
        baseline, base_classes = self._load_baseline(self.a.baseline)
        prev = self._load_prev_results()
        prev_desc = ",".join(str(p) for p in self._prev_roots()) or "none"
        lines = [
            "# 2x2 sweep report",
            "",
            f"- run: `{self.ev}`",
            f"- baseline: `{self.a.baseline or 'none'}`",
            f"- previous run: `{prev_desc}`",
            f"- {craq_gate_taint(getattr(self.a, 'skip_craq_gate', False))}",
            "",
            "| op | verdict | detail |",
            "|---|---|---|",
        ]
        rag = "GREEN"
        for r in results:
            # Per-row verdict via the SAME factored computation the silicon
            # phase streams into ROW-VERDICT.json at row completion — the
            # final report is an aggregation of those verdicts, byte-equal.
            row = self._row_verdict(r, baseline, base_classes, prev)
            rag = self._worst_rag(rag, row["rag"])
            lines.append(self._report_row_line(r, row))
        for s in skips:
            lines.append(f"| {s['op']} | SKIP | {s['reason']} |")
        if self.reds:
            rag = "RED"
            lines += ["", "## RED events", ""] + [f"- {x}" for x in self.reds]
        lines += ["", f"## Overall: {rag}"]
        (self.ev / "REPORT.md").write_text("\n".join(lines) + "\n")
        print(f"REPORT: {rag} -> {self.ev / 'REPORT.md'}")
        return rag

    # ---------------- main flow ----------------
    def _silicon_phase(self, slots, wave=None):
        """Execute + assemble the silicon phase for the gated rows, in row
        order.  slots entries: ("withheld", <result>) pass straight through
        to the results list; ("go", row, classifications, attribution) rows
        run through the batched executor (unless --serial-legacy) and then
        the legacy per-row assembly — silicon() computes every cell, ratio,
        note and STOP decision from the per-leg evidence via the keyed
        hash-matched resume path, so batched and serial runs share one
        assembly code path.  Factored out of run() so the batched-vs-legacy
        layout selftest can drive it without a toolchain.  `wave` scopes
        the batched session dirs under pipelined rolling admission."""
        gated = [(s[1], s[2]) for s in slots if s[0] == "go"]
        if self.exec_mode == "batched" and gated:
            self._batched_silicon(gated, wave=wave)
        results = []
        for s in slots:
            if s[0] == "withheld":
                results.append(s[1])
                self._emit_row_verdict(s[1])
                continue
            _tag, row, classifications, attribution = s
            result = self.silicon(row, classifications)
            results.append(result)
            # FIRST-CLASS ROW VERDICT STREAMING: the row's verdict lands in
            # <op>/ROW-VERDICT.json the moment its cells are assembled —
            # REPORT.md later aggregates the identical computation.
            self._emit_row_verdict(result)
            # Weekly per-knob silicon legs run BEHIND the main BH CRAQ
            # gate (D3) and add their own per-knob classify/CRAQ/
            # correctness pipeline inside knob_silicon().  They stay on
            # the serial per-leg path this increment (jobkeys mode=serial).
            if attribution and row["op"] in (self.a.knob_silicon_rows or []):
                self.knob_silicon(row, attribution)
        return results

    def _classify_prewarm(self, rows, phases):
        """Batched classify prewarm (owner order 2026-08-19; laneCH session
        batching) for the given rows: every (row, selector) classify is
        independent — its work dir, logs, and verdict file are disjoint —
        so the pending ones compile here in BATCHED producer sessions
        (chunked per flag set, per-node outcome attribution via the
        in-tree pytest plugin); the sequential gating loop then resumes
        every verdict hash-matched from cache.  Errors surface exactly as
        before: a cached COMPILE_FAIL verdict replays identically, and any
        leg the batch could not prove falls back to the legacy solo
        compile inside classify()."""
        if "classify" not in phases or getattr(self.a, "classify_workers", 1) <= 1:
            return
        self.verify_toolchain("classify")
        prewarm = []
        for row in rows:
            if row["kind"] == "skip":
                continue
            p_legs = (
                (("default", row["pin_flags"]),) if row["kind"] == "pinpair" else None
            )
            for sel in SELECTORS:
                if row["nodes"][sel]:
                    prewarm.append((row, sel, p_legs))
        if prewarm:
            unproven = self._batched_classify(prewarm)
            # Legs the chunks could not prove used to compile one at a
            # time inside the gating loop; dispatch them through the pool
            # as CONCURRENT legacy-solo sessions instead (laneDB: identical
            # sessions, identical verdicts — only the scheduling changes).
            self._solo_classify_pool(unproven, "batched-classify fallback")
        if self.a.knob_attribution:
            # Knob-attribution prewarm (owner order 2026-08-20, laneDB):
            # attribute_knobs runs len(KNOBS) solo classify verdicts per
            # CHANGED row inside the sequential gating loop — the
            # serialized stretch that dominated weekly classify
            # wall-clock.  Mirror its gating EXACTLY (non-pinpair,
            # perf-else-corr selector, main verdict CHANGED) and compile
            # the same verdict set concurrently; the gating loop then
            # resumes every one hash-matched from cache.  Rows whose main
            # verdict is still unwritten (solo-pool spec raised) simply
            # stay serial-legacy in the loop.  Under pipelined rolling
            # admission this runs per WAVE, so knob prewarm overlaps the
            # earlier waves' silicon exactly like the main classify.
            knob_specs = []
            for row in rows:
                if row["kind"] in ("skip", "pinpair"):
                    continue
                sel = "sem-perf" if row["nodes"]["sem-perf"] else "sem-corr"
                if not row["nodes"][sel]:
                    continue
                cached = self._classify_cached(
                    self.ev / row["op"] / "classify" / sel
                )
                if cached is None or cached.get("all") != "CHANGED":
                    continue
                for knob, flag in KNOBS.items():
                    knob_specs.append(
                        (
                            row,
                            sel,
                            (("off", OFF_FLAGS), ("knob", f"{OFF_FLAGS} {flag}")),
                            f"knobs/{knob}",
                        )
                    )
            self._solo_classify_pool(knob_specs, "knob attribution")

    def _gate_one_row(self, row, phases):
        """Classify/CRAQ/attribution for ONE row — verbatim the legacy
        pass-1 per-row body (unchanged semantics), factored so the
        pipelined gating thread and the legacy loop share one code path.
        Returns the prelim triple (row, classifications, attribution)."""
        # pinpair rows classify/CRAQ a single pinned-flag leg per selector.
        pin_legs = (
            (("default", row["pin_flags"]),) if row["kind"] == "pinpair" else None
        )
        classifications = {}
        if "classify" in phases:
            self.verify_toolchain("classify")
            for sel in SELECTORS:
                if row["nodes"][sel]:
                    classifications[sel] = (
                        self.classify(row, sel, legs=pin_legs)
                        if pin_legs
                        else self.classify(row, sel)
                    )
        if "craq" in phases:
            self.verify_toolchain("craq")
            for arch in row["craq_archs"].split(","):
                for sel in ("sem-corr", "hand-corr"):
                    if row["nodes"][sel]:
                        if pin_legs:
                            self.craq(row, sel, arch.strip(), legs_spec=pin_legs)
                        else:
                            self.craq(row, sel, arch.strip())
        attribution = None
        if self.a.knob_attribution and "classify" in phases:
            attribution = self.attribute_knobs(row, classifications)
        return (row, classifications, attribution)

    def _gate_rows(self, prelim):
        """Gate every row BEFORE any device work (same rules, same REDs as
        the legacy flow): keyed classify evidence required, keyed BH CRAQ
        gate required.  `slots` keeps the row order so the results list is
        unchanged vs the legacy loop.  Returns ordered slots:
        ("withheld", result) | ("go", row, classifications, attribution)."""
        slots = []
        for row, classifications, attribution in prelim:
            # Silicon runs only on classify evidence KEYED to this run
            # (finding sweep_2x2.py:1341: with classify skipped,
            # classifications={} disabled the byte-identical refusal
            # logic and every hash-match).  A resumed evidence root
            # supplies verdicts only if their cc1plus/tt-metal keys
            # match; otherwise the row is withheld RED.
            missing_cls = []
            for sel in SELECTORS:
                if row["nodes"][sel] and sel not in classifications:
                    keyed = self._load_keyed_classification(row, sel)
                    if keyed is None:
                        missing_cls.append(sel)
                    else:
                        classifications[sel] = keyed
            if missing_cls:
                self.reds.append(
                    f"{row['op']}: silicon withheld — no classify evidence "
                    f"keyed to this toolchain/tree for "
                    f"{','.join(missing_cls)} (run the classify phase)"
                )
                slots.append(
                    (
                        "withheld",
                        dict(
                            self._result_skeleton(row, classifications),
                            notes=[
                                "silicon withheld: classify evidence missing "
                                "or keyed to another toolchain/tree"
                            ],
                        ),
                    )
                )
                continue
            # Keyed BH CRAQ gate: stale-toolchain greens never open it.
            gate = self._bh_craq_gate(row)
            if not gate and not self.a.skip_craq_gate:
                self.reds.append(
                    f"{row['op']}: silicon withheld — paired BH CRAQ not green"
                )
                if attribution and row["op"] in (self.a.knob_silicon_rows or []):
                    self.reds.append(
                        f"{row['op']}: knob silicon withheld — main BH CRAQ gate not green"
                    )
                slots.append(
                    (
                        "withheld",
                        dict(
                            self._result_skeleton(row, classifications),
                            notes=["silicon withheld: BH CRAQ gate not green"],
                        ),
                    )
                )
                continue
            slots.append(("go", row, classifications, attribution))
        return slots

    # ------------- classify/silicon pipelining (rolling admission) -------
    _FIRST_WAVE_ROWS = 3

    def _admission_waves(self, rows):
        """Rolling admission groups over the priority-ordered rows.  Wave 0
        is deliberately small — the --priority-ops rows when given (they
        are a prefix of the priority order), else the first
        _FIRST_WAVE_ROWS rows — so device work begins minutes after
        launch; later waves take --admit-wave-rows rows each."""
        if not rows:
            return []
        prio = set(getattr(self.a, "priority_ops", None) or [])
        first_n = sum(1 for r in rows if r["op"] in prio) or min(
            self._FIRST_WAVE_ROWS, len(rows)
        )
        step = max(1, int(getattr(self.a, "admit_wave_rows", 8) or 8))
        waves = [rows[:first_n]]
        i = first_n
        while i < len(rows):
            waves.append(rows[i : i + step])
            i += step
        return waves

    def _pipeline_run(self, phases, rows):
        """CLASSIFY/SILICON PIPELINING (owner order — kill the phase
        barrier): silicon starts on rows whose classify (and CRAQ, when in
        phases) verdicts are complete while classify continues on the
        rest, so device work begins minutes after launch instead of hours.

        A background GATING thread processes admission waves in priority
        order — batched classify prewarm per wave, then the verbatim
        per-row classify/CRAQ/attribution (_gate_one_row) and the verbatim
        keyed gates (_gate_rows) — and publishes each wave's slots.  The
        MAIN thread consumes admitted waves as they land: the batch
        planner handles incremental row admission by RE-PLANNING per wave
        (rolling groups; session dirs scoped silicon-batches/w<i>/), and
        per-row assembly + ROW-VERDICT streaming run immediately.  Trust
        anchors are untouched in semantics: same classify/CRAQ code, same
        keyed gates, same flocked device serialization and per-session
        provenance inside _batched_silicon, same assembly path.  Evidence
        writes are disjoint between the threads (classify/craq dirs of
        not-yet-admitted rows vs silicon dirs of admitted rows).  A gating
        failure — including SystemExit from a mid-run toolchain swap check
        — is forwarded to the main thread and re-raised, never swallowed.
        """
        import queue as queue_mod
        import threading

        waves = self._admission_waves(rows)
        q = queue_mod.Queue()
        print(
            f"pipeline: {sum(len(w) for w in waves)} row(s) in "
            f"{len(waves)} admission wave(s): "
            + " | ".join(",".join(r["op"] for r in w) for w in waves)
        )

        def gate():
            try:
                for wi, wave in enumerate(waves):
                    self._classify_prewarm(wave, phases)
                    prelim = [self._gate_one_row(row, phases) for row in wave]
                    q.put(("wave", wi, self._gate_rows(prelim)))
                q.put(("done", None, None))
            except BaseException as exc:  # incl. SystemExit: forward, re-raise
                q.put(("fatal", None, exc))

        t = threading.Thread(target=gate, name="sweep-gating", daemon=True)
        t.start()
        results = []
        while True:
            kind, wi, payload = q.get()
            if kind == "fatal":
                raise payload
            if kind == "done":
                break
            print(
                f"pipeline: wave w{wi} admitted ({len(payload)} row "
                "slot(s)) — silicon executes now while later waves keep "
                "classifying"
            )
            results.extend(self._silicon_phase(payload, wave=f"w{wi}"))
        t.join()
        return results

    def run(self):
        phases = self.a.phases
        self.preflight()
        # ROW PRIORITY SCHEDULING: measure-first order (priority ops, then
        # expected-changed, then expected-byte-identical re-baseline rows).
        # Every downstream list (results, scoreboard, report rows, streamed
        # ROW-VERDICTs) follows this order — results stream by value.
        self._order_rows()
        results, skips = [], []
        # Schedule deferrals are machine-readable skips in every scoreboard —
        # a weekly row absent from the nightly report would be the silent-
        # omission class the corpus discipline forbids.
        for row in getattr(self, "deferred", []):
            skips.append(
                {
                    "op": row["op"],
                    "corpus_id": row["corpus_id"],
                    "status": "SKIP_SCHEDULE_WEEKLY",
                    "reason": "schedule=weekly row deferred by --schedule "
                    "nightly (device-time budget split; runs in the weekly/"
                    "full sweep)",
                }
            )
        live = []
        for row in self.rows:
            if row["kind"] == "skip":
                skips.append(
                    {
                        "op": row["op"],
                        "corpus_id": row["corpus_id"],
                        "status": "SKIP_ABSENT_NODE",
                        "reason": row["note"],
                    }
                )
            else:
                live.append(row)
        # CLASSIFY/SILICON PIPELINING is the default whenever this run both
        # classifies and executes silicon with the batched executor: the
        # phase barrier (classify EVERY row, then device work) is replaced
        # by rolling admission.  Resume-shaped runs (--phases without
        # classify), --serial-legacy, --no-pipeline and non-hardware runs
        # keep the legacy phase-barrier flow (semantics identical either
        # way — the pipeline only reorders WHEN gated rows reach the
        # executor, never what gates them).
        pipelined = (
            "silicon" in phases
            and self.a.allow_hardware
            and "classify" in phases
            and self.exec_mode == "batched"
            and not getattr(self.a, "no_pipeline", False)
            and bool(live)
        )
        if pipelined:
            self.verify_toolchain("silicon")
            results.extend(self._pipeline_run(phases, live))
        else:
            if "silicon" in phases and self.a.allow_hardware and live:
                print(
                    "pipeline: legacy phase-barrier flow (--no-pipeline, "
                    "--serial-legacy, or a resume without the classify phase)"
                )
            self._classify_prewarm(live, phases)
            prelim = [self._gate_one_row(row, phases) for row in live]
            if "silicon" not in phases:
                for row, classifications, _attr in prelim:
                    results.append(self._result_skeleton(row, classifications))
            elif not self.a.allow_hardware:
                for row, _cls, _attr in prelim:
                    skips.append(
                        {
                            "op": row["op"],
                            "corpus_id": row["corpus_id"],
                            "status": "SKIP_HARDWARE_NOT_AUTHORIZED",
                            "reason": "silicon phase requires --allow-hardware",
                        }
                    )
            else:
                self.verify_toolchain("silicon")
                results.extend(self._silicon_phase(self._gate_rows(prelim)))
        self.emit_scoreboard(results, skips)
        rag = "GREEN"
        if "report" in phases:
            rag = self.report(results, skips)
        self.emit_sha256sums()
        if self.reds:
            print("RED events:\n  " + "\n  ".join(self.reds))
        return 1 if (self.reds or rag == "RED") else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--evidence-root", type=pathlib.Path, required=True)
    ap.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--ops",
        type=lambda s: s.split(","),
        default=None,
        help="comma list of op rows (default: all config rows)",
    )
    ap.add_argument(
        "--priority-ops",
        type=lambda s: s.split(","),
        default=None,
        help="comma list of op rows that JUMP the queue entirely (classified "
        "and measured first, in the order given); remaining rows order by "
        "expected value — rows whose OFF/ON .text hashes are expected to "
        "DIFFER first, expected byte-identical re-baseline rows last "
        "(hint from prior classify verdicts/baseline class; a wrong hint "
        "costs only queue position, never correctness)",
    )
    ap.add_argument(
        "--phases",
        type=lambda s: s.split(","),
        default=["classify", "craq", "silicon", "report"],
        help="subset of classify,craq,silicon,report",
    )
    ap.add_argument(
        "--sim-bh", type=pathlib.Path, help="BH libttsim.so (generic-path CRAQ oracle)"
    )
    ap.add_argument("--sim-wh", type=pathlib.Path, help="WH libttsim.so")
    ap.add_argument(
        "--sim-bh-sha",
        help="required FULL sha256 of the BH libttsim.so — the reviewed CRAQ "
        "oracle pin (sweep_2x2.conf PINNED_SIM_BH_SHA256); verified at "
        "preflight and every phase entry; prefixes are rejected",
    )
    ap.add_argument(
        "--sim-wh-sha",
        help="required FULL sha256 of the WH libttsim.so — the reviewed CRAQ "
        "oracle pin (sweep_2x2.conf PINNED_SIM_WH_SHA256); verified at "
        "preflight and every phase entry; prefixes are rejected",
    )
    ap.add_argument("--venv", type=pathlib.Path, help="tt-llk virtualenv root")
    ap.add_argument(
        "--compiler",
        type=pathlib.Path,
        help="riscv-tt-elf-g++ (default: tests/sfpi symlink)",
    )
    ap.add_argument(
        "--compiler-sha",
        help="required FULL sha256 of the g++ DRIVER — secondary pin only "
        "(historically byte-identical across cc1plus-only changes); "
        "prefixes are rejected",
    )
    ap.add_argument(
        "--cc1plus-sha",
        help="required FULL sha256 of cc1plus, resolved via "
        "g++ -print-prog-name=cc1plus — the PRIMARY toolchain pin; "
        "prefixes are rejected",
    )
    ap.add_argument(
        "--baseline",
        type=pathlib.Path,
        help="chip-class device baseline TSV for --phases report",
    )
    ap.add_argument(
        "--prev-run",
        type=lambda s: [pathlib.Path(x) for x in s.split(",") if x],
        help="previous evidence root(s), comma list NEWEST FIRST: the newest "
        "feeds the report's drift comparison, and EVERY root is probed for "
        "cross-pin silicon cell reuse — a leg whose jobkey (node/flags/"
        "extra_env/tag/mode) matches and whose archived .text hash set "
        "equals THIS run's classify hashes is adopted instead of re-run, "
        "copied into this run's evidence with a REUSED_FROM.txt marker "
        "carrying the full origin chain (provenance visible, never silent). "
        "Source roots are provenance-gated: quarantined/contaminated or "
        "pin-record-less roots refuse (fail closed), craq-gate-tainted "
        "roots refuse unless this run runs --skip-craq-gate too, and a "
        "foreign-pin root adopts loudly with its pin recorded",
    )
    ap.add_argument("--max-drift-pct", type=float, default=5.0)
    ap.add_argument(
        "--max-abs-drift-pct",
        type=float,
        default=10.0,
        help="per-cell ABSOLUTE cycle drift vs baseline: slowdowns beyond "
        "this are RED (uniform slowdowns preserve every ratio); "
        "improvements beyond it are YELLOW (stale baseline)",
    )
    ap.add_argument(
        "--red-loss-growth-pct",
        type=float,
        default=5.0,
        help="a baseline loss growing by more than this many percentage "
        "points is RED (exit 1), not YELLOW",
    )
    ap.add_argument(
        "--allow-win-to-parity",
        action="store_true",
        help="downgrade WIN→PARITY erosion from RED (default) to YELLOW",
    )
    ap.add_argument(
        "--allow-hardware",
        action="store_true",
        help="authorize serialized device jobs (both flocks)",
    )
    ap.add_argument(
        "--knob-attribution",
        action="store_true",
        help="weekly: classify each changed row against each single optimization knob",
    )
    ap.add_argument(
        "--knob-silicon-rows",
        type=lambda s: s.split(","),
        default=None,
        help="weekly: comma list of headline rows that also get per-knob silicon legs",
    )
    ap.add_argument(
        "--schedule",
        choices=("nightly", "weekly"),
        default=None,
        help="row schedule filter (ops TSV 'schedule' column): 'nightly' runs "
        "only schedule=nightly rows; 'weekly' or omitted runs every row",
    )
    ap.add_argument(
        "--skip-craq-gate",
        action="store_true",
        help="run silicon without a green CRAQ gate — the RATIFIED default for scheduled sweeps (owner, 2026-08-20): per-cell device-golden legs gate correctness; the run is taint-marked in evidence either way",
    )
    ap.add_argument(
        "--classify-workers",
        type=int,
        default=int(os.environ.get("SWEEP_CLASSIFY_WORKERS", "6")),
        help="concurrent classify chunk sessions in the prewarm pool "
        "(pending (row,selector) legs are BATCHED into per-flag-set "
        "producer sessions with per-node outcome/ELF attribution; work "
        "dirs are disjoint per (row,selector) and verdicts are hash-keyed "
        "so the sequential loop replays them from cache; unprovable legs "
        "and knob-attribution legs keep one isolated pytest session per "
        "leg but those solo sessions run CONCURRENTLY through the same "
        "pool). 1 disables the prewarm (legacy sequential per-leg "
        "classify).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run steps whose evidence already exists",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print device jobs instead of running them",
    )
    ap.add_argument(
        "--no-pipeline",
        action="store_true",
        help="ESCAPE: disable classify/silicon pipelining (rolling wave "
        "admission) and run the legacy phase barrier — classify EVERY row, "
        "then all device work; gating/refusal/provenance semantics are "
        "identical either way",
    )
    ap.add_argument(
        "--admit-wave-rows",
        type=int,
        default=8,
        help="pipelined rolling admission: rows per gating wave after the "
        "first (the first wave is the --priority-ops list, else "
        f"{Sweep._FIRST_WAVE_ROWS} rows, so device work starts early)",
    )
    ap.add_argument(
        "--serial-legacy",
        action="store_true",
        help="ESCAPE: revert the silicon phase to one pytest session per "
        "device leg (the pre-batching ~45s/leg overhead path); logged "
        "loudly; serial and batched cells are jobkey-separated and never "
        "mix inside a row's samples",
    )
    args = ap.parse_args()
    return Sweep(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
