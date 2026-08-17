# Typecast descriptor-sharing discriminator

## Result

The first same-function experiment is a precise **NO-GO**, not a performance
candidate.  Inlining the clean typed eight-row face body into the existing RC
wrapper preserves the architectural face transitions, but the current GCC
pass emits no `SFPLOADMACRO` at all.  It therefore does not reach the required
one-configuration, 32-launch dynamic tile schedule and was not sent to CRAQ or
Blackhole.

This negative is narrower than the earlier offset-substitution failure.  The
generated final ELF preserves the wrapper's four face iterations and its exact
`TTSETRWC` transitions.  The body instead falls back to an explicit four-word
load/cast/round/store replay for each face.

## Reproducible discriminator

Starting from TT-Metal `4d055cb3`, temporarily changing only
`calculate_typecast_uint16_to_fp16b_semantic<8>` from `noinline` to
`always_inline` and compiling Blackhole with SFPI-GCC `e4b974208` plus
`-mtt-tensix-emit-loadmacro` produces:

- zero `SFPLOADMACRO` instructions;
- one four-word `TTREPLAY` capture and seven playbacks in the face-loop body;
- the original four-iteration face loop;
- the original two `TTSETRWC(0,4,8,0,0,4)` instructions at its face boundary;
- the final `TTSETRWC(0,0,0,0,0,4)` reset.

The rejected experiment ELF is
`/tmp/tt-llk-build/sources/eltwise_unary_typecast_test.cpp/de5ae8fb667c68ae637a0d8c505c35fd95dfa56b5bc9d5fae23e2c782cb7e83c/elf/math.elf`.
Its SHA-256 is recorded below.  The source change itself is intentionally not
retained: without the compiler mechanism it regresses the already-correct
macro formation.

`25ad54c336b2d765f2bf2d2300f8ef8a928f4eeb554872693f8a047aba050f02`

## Compiler root cause

This final ELF exposes the exact missing proof in
`rtl-rvtt-loadmacro.cc`.  Candidate discovery happens in the caller after
inlining, but emission is guarded by the function-wide
`config_available = !source_config_access_p(fn)`.  The shared unary operation
initialization earlier in `run_kernel` writes load-macro-owned configuration
destinations.  That single earlier owner rejects every later cast/round row,
even though no configuration write occurs between the proposed compiler-owned
materialization point and the final face-loop drain.

Independent review found two architectural proof gaps below that first guard,
so merely making the guard path-sensitive is not a safe patch:

- The RC wrapper's `TTSETRWC` face transitions reach this compiler revision as
  opaque raw instructions.  There is no typed RTL pattern carrying their Dst
  counter effects.  A correct ownership pass must therefore reject them; it
  cannot whitelist their numeric instruction words while claiming to preserve
  the RWC state they hide.
- The accepted cast/round emitter selects address-modifier slot 6 on Blackhole.
  On Wormhole its encoded selector 2 maps to physical slot 6 only while the
  high-bank `ADDR_MOD_SET_Base=1` ABI is active.  The emitter deletes every
  explicit typed `TTINCRWC`, but it does not synthesize or prove either the
  corresponding physical-slot configuration or that bank-base precondition.
  The TT integration happens to run after shared operation initialization, so
  this hidden caller state made the focused CRAQ test pass.  That is not a
  generic compiler ownership proof and cannot be extended across calls or
  faces.

The compiler must not decode those earlier raw configuration writes or assume
that they happen to describe the same macro.  The sound general fix starts by
representing those effects, then adds path-sensitive descriptor ownership:

1. Add a typed `TTSETRWC` RTL boundary with explicit volatile RWC/Dst effects,
   and admit only that pattern across a descriptor region.
2. Give Blackhole slot 6 and Wormhole selector 2 -> physical slot 6 an explicit
   compiler-owned initializer, effect model, and Wormhole bank-base proof, or
   retain the typed increments.  Never depend on shared caller initialization
   that the pass has not proved.
3. Find a compiler-owned configuration insertion point after the last
   reachable call, asm, or owned configuration access.
4. Prove that point dominates every admitted face region and that no path to
   the final drain contains another owner.
5. Preserve every typed `TTSETRWC`/`TTINCRWC` transition as an architectural
   barrier; never replace a face transition with a source-address offset.
6. Group identical cast/round descriptors across the four face regions and
   materialize the descriptor once at the dominating point.
7. Emit eight alternating L0/L1 launches per face (32 dynamically), retain the
   exact face transitions, and drain before an exit or a new owner.
8. Refuse calls, asm, config owners, exceptional CFG, QSR, or a path that does
   not satisfy the dominance and post-dominance proof.  Rejected/default-off
   output must stay byte-identical.

Only after that final ELF exists should the lane run WH/BH compilation, QSR
refusal, paired CRAQ correctness, and then Blackhole silicon.  The current
failure occurs before functional validation, so it is not a correctness or
performance claim.

## Status update — impl-1 form adoption (Lane AV, 2026-08-17)

The Lane AK probe structure (`-DTYPECAST_TYPED_RWC_BOUNDARY`: always_inline
semantic body + typed `lltt::setrwc` face boundaries inside one
compiler-visible region) was ADOPTED as the impl-1 default in
`tests/sources/eltwise_unary_typecast_test.cpp` (REDUCE_SDPA typed-boundary
precedent; silicon 273.33 vs hand 264.67 = +3.27%,
`~/sfpi-uplift/silicon-promotions-20260817/item2-typecast-probe`).  The
cross-function descriptor-sharing gap this document describes therefore no
longer gates the measured sweep row; it remains a real mechanism gap for any
kernel whose faces are separate out-of-line calls.  The measurable probe for
it is now the opt-in legacy form `-DTYPECAST_NOINLINE_FACE_BOUNDARY`
(the old default; the old probe define was retired with the inversion).
Gates and provenance: `~/sfpi-uplift/laneAV-evidence-20260817`.
