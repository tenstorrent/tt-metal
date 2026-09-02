# REVIEW_RECORD — pin 53 (FABLE_GOES_BURR wave 5 + sim re-pin)

cc1plus sha256: 46d3116c469d59d5b64be411413d3573024bcfc2dcadb2da2578d9dbea0d564d
driver (g++) sha256: dded27bb7d1d726cfc9f1d07a92c00c3aee2493797caf341d10bd9ba28eb1626
sim pins: bh 1d162f0adf67... -> 7f90adaae35d... / wh f22bc917a4ef... ->
41cf8996455a... (craq-sim agent/laneKR-sim-fidelity fa016347; laneKS
full-bar validation — see the conf SIM PINS section for the complete
re-pin record)
source: sfpi-gcc nkapre/sfpi ae333a1e145 = pin-52 61a98b80944 + five
wave-5 merges (union 418bd5d5bcf, ZERO conflicts) + the KH #15 deletion
commit: KR agent/laneKR-wordfact-table 6ae03c78b3b (ff), KQ
agent/laneKQ-cc-region-rtl-view 46dd60086d6, KP
agent/laneKP-addrsqrt-gimple-rename 816239d787a, KO
agent/laneKO-r3-mad-restructure 58ca0fcb247, KN
agent/laneKN-mve-stage2-realization bb538dda62b.  tt-metal canon: + KR
harness-fixes b8427d8be7 (ff) + KQ 389bac4a59 + KP 1fd3739e15 + KO
69782f09e6 (two keep-both KNOB-dict conflicts resolved; final KNOB_MODES
54).  Built in gcc-build-laneFR (make all-gcc, pinned auto-host.h, rc=0
twice: union + deletion); OPTCHECK + installed-driver smokes for all
four new knobs; installed via pin-install-fast with loud
--expect-cc1plus.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

WAVE 5 — the roadmap-closure wave: all 15 FABLE_GOES_BURR items now
shipped at >= stage A and all 3 residual attacks executed.  Full lane
records in laneK{N,O,P,Q,R,S}-evidence dirs; headline facts:

KN #5 stage 2: MVE kmin=2 realization PERFORMED (marshaller in the one
timing engine, renames through the service + temporal tier,
producer-lockstep belt with in-tree sabotage red/green; twin fire
36->28).  Corpus: ZERO fires in every composition (mve adds no bytes);
crp parity proven; TRIG floor RE-CERTIFIED with realized-kernel
evidence (capture-budget / mve-rename-exhausted 39/44 /
not-counted-kernel).  Flag unregistered in KNOB_MODES pending a
promotion lane.

KO R3: LICENSED MAD restructure (both-keys wall) — immediate-fold veto
where the product dies into an add; the single-use MAD contract fuses.
TRIGONOMETRY-FRESH FLIPPED +2.82 LOSS -> -1.51 WIN (the campaign's
first trig win): 375354 x3 cycle-identical on chip 1, modeled II 95->91
within 0.01pp, attribution pure (assoc-keys-alone byte+cycle inert),
BOOKED.  Honest residual: the licensed keys speed the hand row -7.52 on
its own leg; comparator stays booked-hand per the II/IG convention.
Divergence cert: double->single rounding + SFPMULI sign-of-zero.

KP R1(b): the addrsqrt cert's GIMPLE escape hatch CLOSED BY
MACHINERY+MEASUREMENT — blocker named (a previously-SILENT gap), the
merge-rename class reaches it and loses (+126cy; word-neutral refusal
by construction; composition census: a GIMPLE rename SPENDS 8-LREG
slack).  LOSS +0.98 re-certified on a fresh both-arms chip-2
re-measure.  Both R1(b) rename tiers now closed.

KQ: the CC-region RTL view (post-RA frame classifier + all-lanes entry
proof); the 8,251-refusal cc-span wall FULLY CLASSIFIED, 18.0% opens,
committed renames 2,446 -> 2,614; KE's successor note discharged.
CAMPAIGN LESSON (third independent hit): gcc_checking_assert compiles
out of release-checking builds — belt idiom is
`if (flag_checking) gcc_assert (...)`.

KR closures x4: JO-F1 isclose harness init fixed + re-booked taint-free
on chip 3 (WIN -18.19 STANDS; pre-fix anchors reproduced exact first);
the craq-sim silicon-fidelity fixes (INDIRECT_VD write-mask root cause
— a scheduling defect, not an ARECIP model error; sign-preserving
packer NaN->Inf) = this pin's SIM RE-PIN; item-#4 Deliverable B
wordfact table (4 legacy classifiers deleted; one-pin legacy_* assert
DELETE PIN 54).

laneKS sim re-pin validation (the HU bar, met in full): both libs
deterministically rebuilt; 439/439 CRAQ keys verdict-identical
leg-for-leg both sims x2 passes; GW/HT/HS probe anchors cycle-identical;
recip + tanhlut byte-exact vs the laneKC silicon archives; laneJO
formal_equiv ledger proven untouched; CI + diff-fuzz green on the
staged binaries.  NAMED: addint/sem-corr-wh ON leg fails
bit-identically on both oracles — a compiler-era WH-ON class from pins
35..52, standing item for an owning lane.

KH #15 deletion (ae333a1e145): -552 lines; every site blame-verified to
b49ec40c5ed; 0 dg checks removed; KR's pin-54 markers left.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin53): 7708 PASS = 7578 +
  KN 35 + KQ 24 + KP 34 + KO 37 + KR 0 EXACT; FAIL set 16
  LINE-IDENTICAL; dg ERROR 0.
- UNION CORPUS GATES (sweep-2x2/pin53-ceremony/corpus/): pinned-52 vs
  union ON-37 3300/3300 BYTE-IDENTICAL; union chkon rc=0, ZERO assert
  ICEs, .text == union-ON.  No canon-tree edits during legs (the pin-52
  tripwire lesson held).
- conf_lint GREEN across the full re-pin chain (cc1plus + driver + BOTH
  sim shas + baseline anchors); witness_preflight at ON-37 on the
  installed binary (result below).
- BOARD: 84W/35P/15L -> 85W/35P/14L (the trig flip; KO + KR re-books).
- ON set UNCHANGED at 37; KNOB_MODES 51 -> 54.
- Install: sha-verified 2e2df8e9151a... -> 46d3116c469d...
- PIN-54 DELETION LIST: KR wordfact legacy_* assert phase.
- Push state: pending at record time; pushed tt-metal + sfpi-gcc all
  hops + craq mirror + the craq-sim branch verified (before pin close).
