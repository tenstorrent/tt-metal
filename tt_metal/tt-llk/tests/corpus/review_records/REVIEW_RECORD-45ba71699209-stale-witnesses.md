# Pin-29 stale union fire-witness review

Status: **BLOCKED (proof-backed 9/12)**.  This is a review record, not a
promotion.  It does not change `ON_FLAGS`, `KNOBS`, `KNOB_MODES`, or either
fire-witness table.

Date: `2026-08-25`

Subject compiler:

- cc1plus SHA-256:
  `45ba7169920924fd6ebeb6eeb3766156b413dbf895e091b53603bed1e35e7d79`
- driver SHA-256:
  `a04de6aad4c29aa222e7b5f2e9d699b8bb89fec6accfd38dcf4a78e72e47e720`
- sfpi-gcc source:
  `075e9f2f4b22dd08342be730d42e34060da10d4a`
- comparison source (pin 28): `fd2bb4a481d`

## Exact-pin result

The installed-pin command was:

```sh
python3 witness_preflight.py \
  --work /tmp/pin29-witness-confirm.GDDW8C \
  --tt-metal-home /home/ttuser/sfpi-uplift/tt-metal
```

The explicit `--tt-metal-home` supplies the canonical read-only harness because
this isolated worktree deliberately has no `tests/sfpi` symlink; the command
did not repoint that symlink.  It resolved the exact cc1plus SHA-256 above.  All
eight production nodes compiled successfully with the complete 29-flag
reviewed ON set.  The machine-readable `verdicts.json` result was 9 present and
3 absent:

| flag | reviewed node | required dump line | exact-pin result |
|---|---|---|---|
| `-mtt-tensix-optimize-init-hoist` | minmax-max fresh impl 1 | `Macro-planner init-hoist: stage=2 init contract hoisted` | absent; `init-hoist: closure (callee-external-entry)` then `drain-init-callers-unproven` |
| `-mtt-tensix-optimize-crossloop-hoist` | exp fresh impl 1 | `crossloop-hoist: hoisted across loop` | absent; `mop-template-alias-unproven` then `crossloop-mop-slot-unproven` |
| `-mtt-tensix-optimize-crosscall-hoist` | sigmoid-appx tree impl 2 | `hoisted 6 contract materializations` | absent; `crosscall-callee-external-entry` in `calculate_sigmoid_appx_tree_cpp` |

The regexes are still exact descriptions of genuine pass fires.  They were
not weakened, and no refusal line was substituted as a witness.

## Exhaustive search of the prior production fire set

The checked-in PASS-OP matrix records the complete pre-hardening production
fire inventory: crossloop fired on addcmul, exp, exp2, expm1cw, hardmish,
hardtanh, i0, lerp, and sdpa; crosscall fired on sigmoid-appx tree; init-hoist
fired on the minmax/where family.  This set is an upper bound at the final
source because the implementing-source diff from pin 28 to pin 29 is only the
monotone safety hardening described below: crossloop and macro-planner did not
change, and the shared crosscall/TU analysis only added roots and refusals.

Every member not already compiled by the witness preflight was compiled with
the exact installed compiler, complete reviewed ON set, and all three pass
dumps.  The additional production selectors were addcmul fresh impl 1; exp2,
expm1cw, hardmish, hardtanh, and i0 generic perf rows; lerp; sdpa unclamped;
and minmax-min fresh impl 1.  All nine compiles passed.  Across 333 relevant
dump files under `/tmp/pin29-stale-search.tRRWf5`, there was no match for any
of the three fire expressions.  The former crossloop set now uniformly fails
the MOP-template proof with `mop-template-alias-unproven` /
`crossloop-mop-slot-unproven`; minmax-min has the same
`callee-external-entry` / `drain-init-callers-unproven` init refusal as
minmax-max.  The preflight's where compile produced no replacement init fire.

This proves **registry-wide production inertia at the final full ON set** for
these three flags.  It does not say the compiler passes are dead in every
legal program: the GCC testsuite still has structural firing fixtures.  In
particular, crosscall and init firing fixtures give the transformed callee
internal linkage, and the crossloop fixture supplies an auditable MOP-template
shape.  Those are sound legal-domain tests, but they are not production
registry witnesses.

## Root cause and monotonicity check

The mechanical source check was:

```sh
git diff --stat fd2bb4a481d..075e9f2f4b22dd08342be730d42e34060da10d4a \
  -- gcc/config/riscv/tt/gimple-rvtt-crosscall.cc \
     gcc/config/riscv/tt/gimple-rvtt-crossloop.cc \
     gcc/config/riscv/tt/rtl-rvtt-macro-planner.cc
```

It reports only `gimple-rvtt-crosscall.cc`, from
`1740e2ac312 riscv: harden RVTT entry and CFG proofs`.  That change:

1. replaces the guessed unique `_start`/`main` entry anchor with every public
   definition (including COMDAT) as a conservative executable-closure root;
2. refuses crosscall transformation when its callee is such an external
   entry, because in-TU callgraph edges cannot prove a complete caller set;
3. applies the same external-entry rule to init-hoist; and
4. makes crossloop consult the now-conservative shared MOP-template census.

Those changes only remove previously admitted cases.  Combined with the
complete pre-hardening production fire inventory and its exact-pin replay,
there is no structurally honest replacement row in the current registry.

## Required resolution and evidence consequences

Do not land pin 29 with the present 12-row reviewed table.  The defensible
resolution is to re-review all three flags out of `ON_FLAGS` (29 -> 26), move
their existing rows verbatim to `_QUARANTINED_FIRE_WITNESSES`, and remove the
three drop-one knob entries/modes until a real production-shaped legal-domain
fire exists.  The negative OFF spellings may remain.  This recommendation is
deliberately not applied by this record: it changes the reviewed experiment
and requires owner acceptance.

Removing the flags is expected to be byte-neutral because the final full-ON
compiler refuses every previously firing production case, but that expectation
is not provenance.  It has these mechanical implications:

1. the existing 854-row classifier and 263-row completion-guard census name
   the 29-flag ON baseline and cannot be relabelled as 26-flag evidence;
2. rerun the complete 854-row classifier with the 26-flag ON set and rerun the
   strict 263-row guard census against that same baseline; expected counts or
   byte identity must be measured, not copied;
3. revise CURRENT/PIN HISTORY prose and the manifest flag identity, and make
   R9/R10 green with the reviewed-to-quarantined witness moves;
4. any silicon campaign already launched with 29 flags remains evidence for
   that exact experiment, not the proposed 26-flag experiment.  A final
   promotion needs correctness/CRAQ and full silicon keyed to the accepted
   ON set (or an explicit, mechanically proven adoption procedure); and
5. historical minmax/where, exp-family, and sigmoid-tree attribution to these
   three flags is not a final-pin attribution.  The final refusals mean any
   remaining changed bytes or wins on those selectors come from other passes
   and must be reported that way.

Until that re-review is performed, exact-pin witness preflight correctly
remains RED at 9/12.  Fabricating three replacement rows would hide the safety
hardening and violate the witness gate's purpose.
