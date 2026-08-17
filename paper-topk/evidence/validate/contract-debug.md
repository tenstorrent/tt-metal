# ttnn.topk contract-suite debug: bf16 specials T1 failures + reported data-dependent hang

Date: 2026-08-16 · Branch: nkapre/sorting · HW: single Blackhole p150a · Author: hang/correctness debug subagent

## Executive summary

| Failure class | Classification | Outcome |
|---|---|---|
| T1 `value_multiset` on bf16 NaN/±0 cells (all 3 engines) | **(ii) DATAPATH VALUE SEMANTIC** | Exact mutation rule measured on silicon via a pure SFPU identity op; contract suite now models it (`canonicalize_bf16_datapath()`), bit-preservation demoted T1→T3 documented divergence. Suite **52 passed / 1 FULL-gated skip**. |
| subnormal `NONE_OF_3_MODELS` hard-fails | same (ii) root cause | Canonical model is the missing 4th variant; now candidate #1 (`exact`). All subnormal cells green. |
| Reported data-dependent hang (W=10000 k32 bf16, 8 specials @100..107) | **NOT REPRODUCIBLE** (not classifiable as (i)) | 0 hangs in ~90 properly-launched cells covering the exact trigger, every subset, all engines, ±Watcher, stress reps, cache-hit paths. Evidence points to a session-environment artifact, with precedent of rare branch-wide intermittents. |

No kernel/LLK/host-op code was touched. Only `test_topk_contract.py` + `TOPK_CONTRACT_RUNBOOK.md` were edited (suite-side), per the (ii) prescription.

## 1. The value-semantic bug class: bf16 datapath canonicalization

### 1.1 The mutation rule (silicon ground truth)

Probe: `ttnn.identity` (pure SFPU pass-through, zero topk logic) on a bf16 TILE tensor with all special
bit patterns planted (`probe_semantics.py`, log `semantics.log`, this directory):

```
in : 7fff 7fc0 7fc1 ffc0 ffc1 ffff 7f80 ff80 0000 8000 0001 007f 8001 807f 0080 8080 3f80 bf80
out: 7f80 7f80 7f80 ff80 ff80 ff80 7f80 ff80 0000 0000 0000 0000 0000 0000 0080 8080 3f80 bf80
```

Rule (bf16 only; fp32 is bit-exact — fp32 contract cells pass unmodified):

- **NaN (any payload, either sign) → Inf of the same sign** (mantissa fully cleared)
- **−0 → +0**
- **±subnormal → +0** (sign dropped, not signed-zero flush)
- ±Inf, ±min-normal (0x0080/0x8080), all normals: bit-preserved

Because `ttnn.identity` shows the identical mutation, this is the **generic bf16 compute datapath**
(unpack → Dst → SFPU → pack), NOT topk/SFPSWAP logic and NOT the campaign's LLK edits.
Corroborating ISA documentation: `tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/Dst.md:69`
("The representation of infinity differs from IEEE 754, and NaN is not supported") and `SFPSTORE.md:48`
(datapath conversions flush denormals). Note the sign-drop on flushed zeros distinguishes this from the
SFPSTORE `ToBF16` model (which keeps zero sign) — the stage is upstream of SFPSTORE, most plausibly the
unpacker/gasket or the FPU move into Dst; the observable rule above is exact regardless of stage.

### 1.2 How this produced every contract failure

Canonicalization happens **before** the sort: the SFPSWAP network ranks canonicalized values; the values
output is canonicalized; **indices keep the original input positions** (ties among canonicalized-equal
lanes resolve in implementation-defined positional order — observed orders varied with layout).

- **nan cells**: `0x7FFF/0x7FC0 → 0x7F80` tie with the planted +Inf (largest); `0xFFC0 → 0xFF80` ties with
  planted −Inf (smallest). Reference multiset (built on true input bits) can't match. Topk probe readback:
  top-3 values `7f80,7f80,7f80` at indices {100,101,103} = the two NaN positions + the Inf position.
- **zeros cells**: every −0 lane reads +0. Ledger decode showed expected `0x8000` lanes returned as `0x0000`
  on all 3 engines, both directions. Zeros probe: 16 planted ±0 → 16 lanes of `0000`, indices covering all
  16 original ± positions.
- **subnormal cells**: min-normals survive; subnormals AND ±0 all collapse to key 32768 (+0) — matches none
  of the suite's 3 models (exact / flush_keep_sign / flush_to_pos0) because −0→+0 and sign-dropping flush
  weren't modeled. The canonical model is the exact 4th variant.
- **fp32 mirrors pass** → dtype-specific, as expected for a bf16 conversion stage.

Earlier campaign probe (from_torch→device→to_torch bit-exact) is consistent: pure DM round-trips never
enter the compute datapath.

### 1.3 Suite changes applied (classification (ii) prescription)

`tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py`:

1. New `canonicalize_bf16_datapath()` (module-docstring section documents the rule + probe provenance).
   Unit-checked host-side against the silicon identity-op table (exact match, fp32 passthrough).
2. `verify_topk_cell` builds ALL reference models (T1 multiset, T2 bit-sequence, gather, subnormal
   variants) on the canonicalized input; emits a
   `T3 documented_divergence:bf16_datapath_canonicalization` ledger row whenever any lane mutates.
3. `test_contract_nan_bf16` hard asserts re-pinned post-canon (exactly 3 +Inf-class lanes on top for
   largest; exactly 2 −Inf-class lanes at the bottom for smallest) + new
   `T3 documented_divergence:nan_payload_canonicalized` row. bf16 NaN payload order is unobservable in
   values; fp32 cells still pin payload preservation.
4. `test_contract_zeros_bf16` re-pinned: all `k−n_win` zero lanes read +0, −0 never appears;
   `signed_zero_order` T3 row rewritten (boundary split among ±0 positions is implementation-defined).
5. Subnormal candidates: `exact` now means "canonical model bit-exact" (fp32 semantics unchanged).

`TOPK_CONTRACT_RUNBOOK.md`: T3 row list + a "bf16 datapath canonicalization" contract note added.

### 1.4 Verification

- `pytest ... -k "nan or zeros or subnormal"`: **18/18 passed** (was 15 failed / 3 passed).
  Log: `/tmp/logs/topk_contract_after_canon.log`, ledger `/tmp/logs/topk_contract_ledger3.jsonl`.
- Full default suite: **52 passed, 1 skipped** (FULL-gated cell). Log:
  `/tmp/logs/topk_contract_full_after_canon.log`.
- A single mutation rule closes every T1 failure across all three engines including the T1 gather check
  (`canonicalized_input[index] == value` bit-exact) and T2 exact-sequence pins — strong evidence the model
  is complete, not curve-fit.

## 2. The reported hang: not reproducible

Reported trigger: `torch.linspace(-1,1,10000)` bf16, 8 specials `[7FC0,7FC1,FFC1,8000,0000,7F80,FF80,0001]`
at indices 100..107, single-core W=10000 k=32 TILE; dispatch returned, `to_torch(vals)` never completed
(killed 480 s).

### 2.1 Trials (every run: `flock /tmp/tt-device.lock` + `timeout 120`, `tt-smi -r` before each isolated trial)

| Trial | Config | Outcome |
|---|---|---|
| watcher repro | exact trigger, rows=32, TT_METAL_WATCHER=10 | PASS (and exposed NaN→Inf values live) |
| no-watcher | exact trigger, rows=32 | PASS |
| rows=1 ± watcher (3 runs) | exact trigger | PASS |
| largest=0 (2 runs) | exact trigger | PASS |
| rows=1 + largest=0 | exact trigger | PASS |
| repeat×10 in one process (rows 32 and 1) | program-cache-hit path | PASS |
| no-reset 5-process chain | watcher→nowatch→r1→L0→r1L0 | PASS/PASS then 3× STARTUP_FAIL (see 2.2) |
| bisection sweep (one process, `sweep_specials.py`) | 8 singles ×{largest,smallest}; 7 pairs ×2 (incl. +NaN+Inf, −NaN−Inf, ±Inf-in-row vs −Inf padding, ±NaN, ±0, subnormal+−0); all-8 at W=10000/W=8192-multicore(L0/L1)/W=8192-k96-routed/W=16384; 20× stress + 20× rows=1 stress | **76/76 CELL_OK, SWEEP_DONE, 0 hangs** |

Full table `trials.tsv`; per-trial logs `trial_*.log`, `sweep.log` (this directory).

### 2.2 The false hang scent in this session

An early 3-run loop appeared to hang (`NO_DONE ×3`). Replaying it with full logs showed
`probe_topk.py: error: unrecognized arguments: --rows 1` (rc=2) — a shell arg-tokenization artifact in my
loop (args reached python as one token; the `"$@"`-based runner never failed). **Every apparent hang this
session was a startup failure, not a device hang.** No watcher dump of a genuine hang exists because no
genuine hang occurred; the device never needed recovery (all resets were prophylactic).

### 2.3 Assessment

- The exact reported input passes deterministically and repeatedly (>25 executions of the trigger row
  pattern), on cold JIT, warm cache, cache-hit repeats, both row counts, both directions, all engines.
- Data-dependence is further refuted structurally: the single-core sort kernel's control flow is
  phase-count-static; the value-mutation happens in the conversion datapath and cannot alter loop trip
  counts.
- Precedent: the campaign ledger records rare monolithic-run hangs "never in per-cell isolation" (~3040
  clean isolated calls) on this branch. The 480 s observation is most consistent with a session-state /
  environment artifact (e.g. prior device activity without reset, or the known rare intermittent), not a
  reproducible data-dependent kernel bug.
- Consequently **no minimal repro exists to commit**; the parametrized probe
  (`probe_topk.py`/`sweep_specials.py`, this directory) stands ready if it ever recurs — first action
  then: rerun under `TT_METAL_WATCHER=10` and read `generated/watcher/` core states before reset.

## 3. Artifacts

- Suite: `/home/nachiket/tt-metal/tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py` (edited)
- Runbook: `/home/nachiket/tt-metal/tests/ttnn/unit_tests/operations/reduction/TOPK_CONTRACT_RUNBOOK.md` (edited)
- Probes + logs + trial table: this directory (`probe_topk.py`, `probe_semantics.py`, `sweep_specials.py`,
  `semantics.log`, `sweep.log`, `trials.tsv`, `trial_*.log`)
- Pytest logs: `/tmp/logs/topk_contract_after_canon.log`, `/tmp/logs/topk_contract_full_after_canon.log`
- Ledger with the new T3 rows: `/tmp/logs/topk_contract_ledger3.jsonl`

## 4. Follow-ups (not in this task's scope)

- The bf16 NaN→Inf / −0→+0 / subnormal-sign-drop rule is a **hardware-generation datapath fact** worth a
  Wormhole comparison run (same identity-op probe) before porting the contract suite there.
- Any consumer relying on `ttnn.topk` bf16 values to carry NaN (e.g. NaN-propagation checks in sampling
  chains) silently receives ±Inf instead — worth a note in the op docs.
