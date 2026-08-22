# Cross-lane acceptance arsenal (lane FB, 2026-08-21)

Independent acceptance gate (X3 of the cross-lane vocabulary program,
`CROSSLANE-DESIGN-INPUT.md` lane EY-R) for the builder lane's typed
shuffle/transpose/reduce/sort surface.  Built BEFORE and INDEPENDENT of the
builder: the oracle derives only from the tt-isa-documentation functional
models; the sim legs validate against the PINNED simulators; nothing here
shares code with the builder.

## Components

| file | role |
|---|---|
| `helpers/crosslane_oracle.py` | host oracle: instruction models (SFPSHFT2 all modes, SFPTRANSP, SFPSWAP mods 0-9 + ENABLE_DEST_INDEX + EXCHANGE_SRCB_SRCC + lane-mask SFPCONFIG, partial 16b Dst modes, lane predication) + composed consensus primitives (rotr/broadcast/zip/butterfly/reduce/scan/sort2/sort2_kv/bitonic networks/top-k) with PINNED fold orders |
| `test_crosslane_oracle_identities.py` | host identity battery (38): permutation-hood, rotate/transpose/zip/butterfly inverses, copy4 nilpotency, swap mask table + quirk + tie-divergence witness, reduce reassociation licence witness, scan relations, network sortedness/permutation on adversarial sign-magnitude stimuli, softmax_k mask==tileid-predicate equivalence |
| `../sources/sfpu_crosslane_probe.cpp` | lane-tracer probe kernel, PROBE_MODE 0-18 + 30 (DS ladder-probe skeleton; raw builtins + minimal raw-TTI regions) |
| `test_crosslane_lane_tracer.py` | sim battery (17): empirical tensor<->(row,lane) calibration (rowtag/lanetag/identity), per-primitive exhaustive 32-lane compare vs the oracle on sentinel + varied (genericity twin) stimuli, empirical operand-role adjudication, partial-16b aliasing + BF16-RMW probes |
| `helpers/crosslane_demand_goldens.py` + `crosslane_fixtures/` | CRAQ-checkable demand-kernel goldens for X7: softmax_k (both formulations), moe_gate_topk, ema (BOTH arithmetic contracts), cumsum (int + fp32), bitonic per-stage traces (8 & 32, values + KV, tie-free), sfpswap tie-behavior (both variants of the unresolved divergence) |
| `test_crosslane_demand_goldens.py` | fixture byte-identity + consistency gate |
| `../../../../tools/crosslane_arsenal_gate.py` | the whole battery against an arbitrary compiler + sim; VERDICTS.tsv; `--mode future` = builder acceptance |

## Running

```
cd tt_metal/tt-llk/tests/python_tests
export CHIP_ARCH=blackhole LLK_HOME=<repo>/tt_metal/tt-llk \
       TT_METAL_SIMULATOR=<simstage>/bh/libttsim.so RUNNER_TEMP=<fresh>
.venv/bin/python -m pytest -q test_crosslane_oracle_identities.py \
                              test_crosslane_demand_goldens.py
.venv/bin/python -m pytest -q --run-simulator test_crosslane_lane_tracer.py
# or everything at once:
tools/crosslane_arsenal_gate.py --mode today --llk-home tt_metal/tt-llk \
    --sim <simstage>/bh/libttsim.so
```

Pinned sim: craq-sim `9f324140`, release_bh libttsim sha256 `32489dda4fd6...`
(soc_descriptor.yaml beside the .so).  The lane tracer is BH; the oracle
carries the WH arms (SHFLSHR1 lane-0 UnpredictableValue etc.) for a future
WH leg.

## Builder (lane FA) acceptance — `--mode future`

1. Build the branch's toolchain; wire it per the lane-DT recipe: cc1plus-only
   changes ride `--gcc-exec-prefix <hybrid>/compiler/lib/gcc/`; new `-m`
   flags need the full hybrid (driver + cc1plus) at `tests/sfpi` and the
   flags via `--extra-flags`.
2. `tools/crosslane_arsenal_gate.py --mode future ...` must be ALL-PASS and
   reproduce the FACT rows byte-identically (swap-role, tie-divergence,
   aliasing, BF16-RMW).  Any deviation is a gate FAIL to adjudicate, never
   patch around.
3. Typed-surface probes: when the graduated wrappers (sort2/sort2_rows/
   sort2_kv/transp8/subvec_rotr/reduce/...) land, add a typed twin of each
   probe mode; the twin must be word-identical (or better) to the raw mode
   and bit-identical on the sim.  The oracle already defines the semantics
   the wrappers must meet.
4. Demand fixtures: X7 migrations CRAQ against `crosslane_fixtures/*.json`
   (per-stage bitonic traces localize the first diverging stage).  The ema
   fixture carries BOTH arithmetic contracts — the lowering must match one
   and DOCUMENT which; a third value is a finding.

## Empirically pinned facts (arsenal contract)

- SWAP-ROLE: `__builtin_rvtt_sfpswap(a, b, mod)` / `sfpswap_indexed` —
  operand 0 plays VD (select index 0 = the VD-role result, min for mod 1);
  adjudicated on-sim via row-group mods + companion movement, consistent
  across all modes.
- vConstTileId == 2*lane re-proven on-sim (calibration asserts it).
- SFPCONFIG lane-mask form (Mod1=8, mask bit `(lane&7)*2`) decodes per the
  doc; per-column EXCHANGE flip behaves as the value-form state-equivalent.
- Dst 16b-view (fp32-acc config): RAW 16b modes (6/14/15) ride the Dst.md
  Adj16 bank cells — loads at rows<8 return the BF16-swizzled high half,
  stores never clobber the same-address 32b datum; the in-view
  store->load roundtrip is exact.

## FINDINGS (doc/sim/toolchain divergences — the gold)

1. **SFPSWAP tie decision, doc vs pinned sim** (consequential for
   argmin/argmax ties): SFPSWAP.md keys tie-swaps on SIGN (min lanes swap
   equal negatives, max lanes equal positives); craq-sim 9f324140
   (`sfpswap_vd_gets_c`) uses `min: c<d / max: c>=d` — no sign arm.
   Invisible for plain min/max; VISIBLE via ENABLE_DEST_INDEX companions.
   Silicon adjudication pending; fixtures are tie-free or carry both
   variants (`tie_behavior.json`); oracle exposes `tie="doc"|"sim"`.
2. **BF16-format SFPSTORE onto 32-bit Dst — three-way divergence**:
   SFPSTORE.md writes only its 16b cell (paired low half PRESERVED); the
   pinned sim ZEROES the paired low half, except under ENABLE_DEST_INDEX
   or the TopK LCONST0 special case where it RMW-preserves (the
   hardcoding-audit open question at tensix.cpp `preserve_dst32_low_half`,
   now pinned by probe mode 30); tt-blaze #2475 claims silicon
   BF16-CANONICALIZES the paired half.  Silicon adjudication pending.
3. **`rvtt_sfpshft2_subvec_copy4` (SUBVEC_CHAINED_COPY4) is unusable on
   BH at the pinned toolchain**: rvtt.md `rvtt_sfpshft2_subvec_copy4_int`
   emits `SFPSHFT2\t%x0 %x0, 0, %8` (missing comma) — the BH assembler
   rejects it ("extension xtttensixqsr required").  The design doc lists
   CHAINED_COPY4 as a lowering ingredient (P-table row); the builder must
   fix the md template (one comma) before using it.  The INSTRUCTION
   itself is fine (probe mode 6 exercises it via raw TTI words and matches
   the doc model exactly).
4. Minor: craq-sim SFPLOAD FP16 arm has `// XXX no ENABLE_FP16A_INF`
   (doc models the infinity remap) — untested here (FP16A out of scope),
   recorded for the sim-coverage ledger.

## Probe-mode inventory (kernel <-> tests)

0-2 calibration (identity/rowtag/lanetag) | 3 transp8 both banks |
4 ror1/shr1/rotr^3/ror1^8 | 5 COPY4 | 6 CHAINED_COPY4 (raw TTI) |
7 ROR1_AND_COPY4 | 8/9 swap mods 1-4/5-8 | 10 swap mod0 + mod1 |
11 EXCHANGE flip global | 12 SFPCONFIG lane-mask flip |
13 indexed swap (ENABLE_DEST_INDEX window) | 14 SFPCONFIG L11/L12
vertical broadcast | 15 int reduce composition (row fold + transp
cross-row + config broadcast) | 16/17 register-axis sort4 (+ transp
sandwich) | 18 partial LO16/HI16 companion roundtrip | 30 BF16 RMW probe
(plain + ENABLE_DEST_INDEX arms).
