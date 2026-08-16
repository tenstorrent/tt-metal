# RUNBOOK — Gate-1 ttnn.topk differential contract suite

File under test: `tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py`
Spec: scratchpad `storm/research/contract.md` (Gate-1 report, 2026-08-16).
No build required — pure pytest against the existing `_ttnn.so`; no tracked file was modified.

## Prerequisites

- Run from `$TT_METAL_HOME` (`/home/nachiket/tt-metal`) with the repo venv active.
- Device must be free (the Gate-1 baseline sweep must have finished). All device
  commands below are wrapped in `flock /tmp/tt-device.lock`.
- `env | grep -E '^TT_METAL_(DPRINT|WATCHER)'` must be empty (leftover DPRINT
  routing slows the run).

## 1. Decisive subset (default, target <= ~5 min warm cache; cold JIT may add a few min)

```bash
cd /home/nachiket/tt-metal && source python_env/bin/activate
flock /tmp/tt-device.lock timeout 1800 pytest \
    tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py \
    --timeout=300 --timeout-method=thread -v \
    2>&1 | tee /tmp/logs/topk_contract_default.log
```

Expected: ~53 tests collected on Blackhole (routed cells auto-skip on other
arches). Read the verdict with:

```bash
grep -E '^=+ .*(passed|failed|error|skipped).*=+' /tmp/logs/topk_contract_default.log | tail -1
```

## 2. Full matrix (adds fp32/largest=False mirrors, routed width ceiling 2^19,
## dim=1 transpose, extra k-alignment cells; the 2^19 cells move real data — budget ~15-25 min)

```bash
flock /tmp/tt-device.lock timeout 3600 env TOPK_CONTRACT_FULL=1 pytest \
    tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py \
    --timeout=600 --timeout-method=thread -v \
    2>&1 | tee /tmp/logs/topk_contract_full.log
```

## 3. Per-class runs (-k on the group names)

```bash
# groups: nan / zeros / subnormal / ties / infleak / determinism / gates
flock /tmp/tt-device.lock timeout 900 pytest \
    tests/ttnn/unit_tests/operations/reduction/test_topk_contract.py \
    -k subnormal --timeout=300 --timeout-method=thread -v \
    2>&1 | tee /tmp/logs/topk_contract_subnormal.log
```

The single most valuable first run is `-k subnormal` — I6 is the report's one
true unknown; the `subnormal_survival` ledger rows are the answer either way.

## 4. Artifacts

- **Divergence ledger** (JSON lines, appended, one file across runs):
  `$TOPK_CONTRACT_LEDGER` if set, else
  `$TT_METAL_HOME/generated/topk_contract_ledger.jsonl`.
  Rows: `{ts, test, tier, check, engine, cell, expected, actual, note}`.
  - `tier=info, check=engine_predicted` — one per cell (predicted engine; the
    factory choice is not observable from Python, prediction mirrors
    topk.cpp:247-295 + topk_device_operation.cpp:59-115 and was verified
    against 27 gate cells host-side).
  - `tier=T3` rows are expected and healthy: `torch_value_order_diff` on
    NaN/±0 cells (hw sign-magnitude order vs torch — note torch.topk itself
    canonicalizes bf16 NaN payloads in its values output, which inflates
    lane-diff counts), `subnormal_survival` (exact / flush_keep_sign /
    flush_to_pos0), `padding_index_leak_observed`,
    `documented_divergence:{neg_nan,signed_zero_order,`
    `bf16_datapath_canonicalization,nan_payload_canonicalized}`.
  - **bf16 datapath canonicalization (silicon-measured 2026-08-16, p150a,
    identity-op probe — see the module docstring):** the bf16 compute
    datapath mutates NaN(any payload)→same-sign Inf, −0→+0, ±subnormal→+0
    BEFORE the sort; values output is canonicalized, indices keep original
    positions; fp32 is bit-exact.  All reference models apply
    `canonicalize_bf16_datapath()` first; bit-preservation of NaN payloads /
    −0 / subnormal sign on bf16 is a documented T3 divergence, not a T1
    invariant.  Verified green across all three engines post-model
    (2026-08-16: 52 passed / 1 FULL-gated skip).
  - Any `tier=T1` or `tier=T2` row means a test FAILED — those are contract
    violations (T1: universal topk invariants; T2: incumbent pins — value
    bit-sequence under the sign-magnitude model, index-dtype boundary, routed
    sentinel, sorted-flag no-op, 3-launch determinism).
  Recommend `export TOPK_CONTRACT_LEDGER=/tmp/logs/topk_contract_ledger.jsonl`
  per campaign run so ledgers do not interleave.
- Pytest logs in `/tmp/logs/` as tee'd above.

## 5. Pass/fail interpretation

- All green: the incumbent contract is pinned; the ledger's T3 rows BECOME the
  contract text for NaN/±0/subnormal/leak semantics (feed back into
  THRESHOLD_SELECT_DESIGN.md §2 / RADIX gate-1 sign-off).
- `test_contract_subnormal_*` hard-fails only if the datapath matches neither
  exact nor either flush model — that is new information; capture the printed
  key rows.
- `test_contract_gates*` failure on the engine assert means the routing
  predicate changed (or the Python mirror drifted) — re-check
  topk.cpp:247-295 / topk_device_operation.cpp:59-115 before touching tests.
- Known environment hazards: parallel device processes (`CHIP_IN_USE_*_PCIe`
  lock) — never run two pytest invocations at once; `tt-smi -r` before rerun
  after any kill.

## 6. Validation already performed (host-only, no device)

- `python3 -m py_compile` clean.
- Engine predictor checked against all 27 report boundary cells (BH grid
  13x10) + clause spot-checks (fp32/stable/dim/WH never route) — all match.
- Full suite mock-executed against a compliant sign-magnitude oracle with
  routed-sentinel emulation: 52/52 default cells, 62/62 FULL cells green,
  91 well-formed ledger rows. (Mock harness:
  scratchpad/check_predictor.py + scratchpad/mock_run_suite.py.)

## Notes / limitations

- bfp8_b is intentionally out of scope (values re-quantize through the bf16
  sort, so bit-exact gather does not apply; existing test_topk.py bfp8+inf
  tests keep covering it).
- Engine identity is *predicted*, not observed; if the campaign later wants
  ground truth, a tracy op-report run per cell can confirm factory selection.
- CPU-torch gotcha baked into the suite design: multi-dim bf16 `torch.gather`
  and `torch.topk` values canonicalize NaN payloads — all bit-exact checks
  therefore run in an int64 bits domain, never through bf16 torch ops.
