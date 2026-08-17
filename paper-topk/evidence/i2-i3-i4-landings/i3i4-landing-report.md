# I3 + I4 landing report (2026-08-17, p150a, branch nkapre/sorting @ 482de67d779)

All device runs under `flock /tmp/tt-device.lock` with outer timeouts, serial, full tee'd logs.
Tree left UNCOMMITTED with exactly the kept wins; nothing pushed, nothing committed.
Durable copies of every table/curve cited below: `storm/i3i4-landing/artifacts/`.

## MISSION A — I3 MoE-gate route extension: **LANDED**

Both patches applied CLEANLY on the post-envelope tree (no hand rebase; gate arm verified
to coexist with the landed TILE-out/u16 route policy). py_compile green. Rebuilt
`_ttnn.so` (log /tmp/logs/i3_build.log; provenance uniform `so_md5=608eaf2c` across all
A/B cells, `head=482de67d7796`).

### Decision A/B (charter: land iff routed*1.05 <= stocknow at BOTH cells) — PASS at both

Sweep dir `generated/canonical_sweep/i3_gate_ab/` (scenarios_table.csv/.md; copies in
artifacts/), log /tmp/logs/i3_gate_ab.log, 5 iters/cell:

| cell | shape | routed_us | stocknow_us | rule (x1.05) | speedup |
|---|---|---|---|---|---|
| gate_gptoss_k4 | 32x128 k=4 | **8.20** | 24.22 | 8.61 <= 24.22 PASS | 2.95x |
| gate_qwen35_k10 | 32x512 k=10 | **8.29** | 77.49 | 8.70 <= 77.49 PASS | 9.34x |
| gate_gptoss_op16 (diag) | bare op k=16 | 4.05 | — | anchor OK (~4.07 exp) | — |
| gate_qwen35_op16 (diag) | bare op k=16 | 4.05 | — | anchor OK (~4.04 exp) | — |

Sanity anchors matched plan (stocknow 24.22/77.49 vs ~24.2/77.5). Composite envelope on
the post-TILE-native tree is ~4.2 us over the bare op — the concurrent envelope landing is
what makes these tiny shapes winnable.

### Guards (I3)

| suite | result | log |
|---|---|---|
| contract default | 71P/1F/1S — all 9 new GATE_CELLS + rows-cap PASSED; the 1F is the documented pre-existing intermittent `nan_bf16[routed-W10000-k32-largest]` (passed solo 1/1 right after; k=32 > gate_route_max_k=16, the new arm provably cannot touch it) | /tmp/logs/i3_contract.log, i3_nan_solo.log |
| contract FULL | 81P/1F — same known cell, same signature | /tmp/logs/i3_contract_full.log |
| reduce/test_topk | 220P, 8 skipped, 80 xfailed | /tmp/logs/i3_reduce_topk.log |
| tli nightly | 181P/2F — both env IOMMU perf-check cells ("Real-time profiler must be active... needs IOMMU"), known box limitation, unrelated | /tmp/logs/i3_tli.log |
| deepseek-v3 test_topk (optional) | UNCOLLECTABLE on this box as anticipated ("Invalid system name: None" at collection); contract cell gate-dsv3class-w256-k8 (PASSED) stands in | /tmp/logs/i3_dsv3_topk.log |

### Post-land follow-up done in-tree

`_canonical_topk_sweep.py` built-in MODEL_SCENARIOS `gate_gptoss_k4`/`gate_qwen35_k10`:
dropped the now-false `[NO-CHANGE CONTROL]` tags/notes; engines -> ["routed","stocknow"],
today_engine -> routed, notes carry the measured numbers. py_compile green.

### Follow-ups left for orchestrator (AB_PLAN post-LAND list)

1. Regenerate + republish ledger artifact 252f8c5a-... with the new gate cells.
2. Watch gpt-oss / qwen3.5 e2e-perf bands (thresholds fail both directions).

## MISSION B — I4 chunk-skip telemetry + gate-constant A/B

Patches applied cleanly (disjoint from I3). JIT-only, no rebuild. Log root
/tmp/logs/i4_20260817/.

### Phase 0 — disabled-path byte-identity: **PASS** (recipe refinement needed)

Baseline (I3 tree, no I4): adversarial 36/36, 435 hash rows. Patched (telemetry OFF,
divisor 4): 36/36, 435 rows. Raw diff flagged 2 rows (one MATH config each in compute /
compute_with_values). Root-caused via direct ELF disassembly diff: instruction bytes and
addresses IDENTICAL; the only delta is objdump's comment annotation of the compiler's
file-local symbol `trisck.cc.<source-hash>` (name embeds a source-text hash — changes by
construction; address moved 3 bytes in a non-text section). Comments stripped
(`sed 's/[[:space:]]*#.*$//'`): **0 diff lines both pairs**. Other 433 rows raw-identical.
Evidence: artifacts/phase0_verdict.txt, /tmp/logs/i4_20260817/{phase0_*,diffdirs/}.
RUN_PLAN recipe gap (tail -n +4 strips path headers, not objdump comments), not a code change.

### Phase 1 — telemetry validation + G4 curve (divisor 4): **GREEN — telemetry LANDS**

- Adversarial 36/36 with telemetry ON under DPRINT (TR1), 316 CSTL records.
- Parser hard validation (popcount==s, word order/count, bit ranges) passes over all
  records — after fixing a real parser gap found on silicon: multi-core row-parallel
  launches interleave per-core DPRINT streams in one file; the as-prepared parser attached
  one core's CSTLM words to another core's CSTL header. Fixed in
  `_topk_large_indices_skip_telemetry_parse.py` by demuxing on the `<dev>:(x,y):<RISC>:`
  prefix (test-only file from patch 0001; the fix is part of the landing).
- Tracer cross-check vs proven CHUNK_SKIP_DEBUG: RUN_PLAN's TOPK_K=32 cell is vacuous
  (4-chunk row, first_tested=8 -> zero tested chunks; CSTL correctly reports n=4 f=8 s=0,
  itself pinning the gate arithmetic; k=8 attempt correctly rejected by the op's k>=16
  floor). Real cross-check at divisor=8, k=16 (first_tested=2 -> chunks 2,3 tested):
  CSD max/thr/skip match host-computed expectations bit-exactly in asc (skip 0,0) and
  desc (skip 1,1); CSTLM mask 0b1100 (chunks 2,3) matches CSD decisions exactly; outputs
  EXACT both modes. Header restored after. Logs phase1_xchk_k16d8_*.
- Hang battery telemetry ON: 20/20.
- Telemetry disabled + caches cleared after collection (shipping default in tree).

### G4 telemetry curve — **MATCHES THE LAW (paper evidence: YES)**

40 bench runs (rows {2,8} x seeds 0..19, k=32, W=65536, 128 chunks), 40/40 value-correct,
400 row records. Per-position P_obs tracks P_law = e^(-32/(c+1)) across all 120 tested
positions (c=8: 0.030 obs vs 0.029 law; c=127: 0.730 vs 0.779); aggregate E[#skips]/row
= 65.11 observed vs 66.62 law (-2.3%, finite-sample/bf16-tie direction). **Curve data:
/tmp/logs/i4_20260817/g4_curve.csv (+ .txt table; raw logs cstl_r{2,8}_s{0..19}.log);
durable copy storm/i3i4-landing/artifacts/g4_curve.csv.** The gate A/B telemetry runs
reproduce the law independently per arm (skiprate_d*.csv).

### Phase 2 — gate-constant A/B: **KEEP /4** (charter stop rule; table is the deliverable)

Harness dir /tmp/logs/i4_gate_ab_20260817_120230 (console log
/tmp/logs/i4_20260817/gate_ab_console.log; durable copies artifacts/gate_ab_*.csv/txt).
3 trials x 5 iters Tracy medians, W=65536:

| cell | div | gate | med_us | spread% | vs /4 | skips obs | skips law |
|---|---|---|---|---|---|---|---|
| rows=2 k=32 | 2 | 16 | 151.85 | 0.0 | -0.8% | 64.60 | 65.99 |
| rows=2 k=32 | 4 | 8 | 153.00 | 0.0 | +0.0% | 65.40 | 66.62 |
| rows=2 k=32 | 8 | 4 | 153.49 | 0.0 | +0.3% | 65.50 | 66.66 |
| rows=2 k=512 | 2 | 256 | 280.47 | 0.0 | -0.0% | 0.00 | 0.00 |
| rows=2 k=512 | 4 | 128 | 280.48 | 0.0 | +0.0% | 0.00 | 0.00 |
| rows=2 k=512 | 8 | 64 | 289.37 | 0.0 | +3.2% | 0.20 | 0.42 |
| rows=8 k=32 | 2 | 16 | 176.78 | 0.0 | -0.7% | 65.50 | 65.99 |
| rows=8 k=32 | 4 | 8 | 177.97 | 0.1 | +0.0% | 66.15 | 66.62 |
| rows=8 k=32 | 8 | 4 | 178.51 | 0.0 | +0.3% | 66.20 | 66.66 |

Decision rule requires a >=5% k=32 win (best observed: -0.8% via /2) AND <=0.5% k=512
regression (/8 regresses +3.2%). Law-predicted ties confirmed -> **default stays
CHUNK_SKIP_GATE_DIVISOR 4**; table files as the paper's gate-sensitivity ablation.
Header restored by the harness trap (verified: divisor 4, telemetry + debug commented).

### Guards (I4, shipping config, after Phase 2)

| suite | result | log (all under /tmp/logs/i4_20260817/) |
|---|---|---|
| contract FULL | 81P/1F — same known intermittent nan_bf16 cell (solo re-run: 1/1 PASS). Note: it failed in all 3 full-suite runs today vs the documented 2-in-5 — consistent with "state-dependent after heavy device activity" (device was saturated all day), still 100% solo-pass; flagged as a watch item, not attributable to I4 (disabled path byte-identical to baseline) | guard_contract.log, guard_nan_solo.log |
| tli nightly | 181P/2F — identical env IOMMU cells as pre-I4 | guard_tli.log |
| reduce/test_topk | 220P, 8 skipped, 80 xfailed | guard_topk.log |
| adversarial (shipping) | 36/36 (phase 0b run; divisor default unchanged, no rerun needed per RUN_PLAN) | phase0_adv_patched.log |
| hang battery (shipping, no DPRINT) | HANGBATTERY OK: 20 launches | guard_hang.log |
| column-parallel pin | 13.211 us median (anchor ~13.2 us) — unchanged; `git diff --stat -- '*compute_tree*'` empty | guard_colpin.log |

### Landing decision (RUN_PLAN section 7)

- **0001 telemetry: LANDS** (byte-identity PASS, phase 1 items 2-4 green, guard set green;
  ships with telemetry commented out).
- **0002 knob + harness: LANDS** as measurement scaffolding (default 4 == identical code;
  phase 0 hashes taken with 0002 applied, so the identity claim covers it).
- **Gate default: unchanged (/4).**

## What was reverted / temporary (and why)

- Nothing reverted permanently in either mission (both missions landed).
- Temporary: I4 patches reverted + re-applied once during Phase 0 root-cause (to
  regenerate baseline ELFs for the disassembly diff); header knobs flipped during
  Phase 1/2 (telemetry ON for collection, CHUNK_SKIP_DEBUG ON + divisor 8 for the tracer
  cross-check, divisor sweep by the harness) — ALL restored; final header state verified:
  `// #define CHUNK_SKIP_DEBUG 1`, `// #define CHUNK_SKIP_TELEMETRY 1`,
  `#define CHUNK_SKIP_GATE_DIVISOR 4`.

## Final tree state (uncommitted, for orchestrator review)

Kept wins (9 modified + 2 new):
- I3: ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp; tests/.../reduction/test_topk_contract.py;
  models/common/sampling/_utils.py; tests/.../reduction/_canonical_topk_sweep.py (follow-up edit)
- I4: .../topk_large_indices/device/kernels/{topk_large_indices_chunk_skip.hpp,compute.cpp,
  compute_with_values.cpp}; tests/.../experimental/_topk_large_indices_bench.py;
  NEW tests/.../experimental/_topk_large_indices_skip_telemetry_parse.py (with the
  multi-core demux fix); NEW tests/.../experimental/_topk_large_indices_gate_ab.sh

Pre-existing dirt NOT mine, left untouched: `M .github/workflows/package-and-release.yaml`;
untracked lx-reset, tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.yaml,
tt_metal/programming_examples/eltwise_poly/, ttnn/ttnn/_ttnn.so.release.

## G4 law question (explicit ask)

**YES — the telemetry curve matched the amortization law** e^(-K/(c+1)): per-position
agreement across all 120 tested positions and E[#skips]/row within 2.3% of the law sum
(65.11 vs 66.62) on 400 iid rows; independently reproduced per divisor arm in the gate A/B
(e.g. d4 r2: 65.40 obs vs 66.62 law; d8 k512: 0.20 obs vs 0.42 law on a 0.2-expected-skip
tail). Curve data: /tmp/logs/i4_20260817/g4_curve.csv (durable copy:
storm/i3i4-landing/artifacts/g4_curve.csv; per-arm curves artifacts/skiprate_d*.csv).
