# GPT-OSS Trace 15 — Local Sweep Validation Session

## Goal

Run the full sweep validation loop locally (matching the CI workflow `ttnn-model-trace-sweep-validation-impl.yaml`) for gpt_oss trace 15, and reach 100% exact-match validation.

## Setup

- **Trace:** 15 (`gpt_oss`, tt-galaxy-wh, 32 cards, 130 configs, status `active`)
- **Hardware:** Galaxy 4×8 (`UF-EV-B8-GWH02`, 32 devices confirmed via `ttnn.get_num_devices()`)
- **Mode:** "pinned trace" path — bypasses the manifest, uses `reconstruct-trace 15` directly.

Local script equivalent of the CI workflow:

```bash
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-trace 15 model_tracer/traced_operations/ttnn_operations_master.json
python3 model_tracer/config_hash_preflight.py model_tracer/traced_operations/ttnn_operations_wan.json --allow-partial --report validation_artifacts/config_hash_preflight.md
python3 tests/sweep_framework/sweeps_parameter_generator.py --model-traced all --suite-name model_traced --group-by hw --master-trace model_tracer/traced_operations/ttnn_operations_master.json --tag ci-validate-pinned
LEAD_MODELS_RUN=1 python3 model_tracer/generic_ops_tracer.py tests/sweep_framework/sweeps_runner.py -o model_tracer/traced_operations/sweep_trace_gpt_oss.json -- --suite-name model_traced --vector-source vectors_export --result-dest results_export --main-proc-verbose
python3 tests/sweep_framework/validate_sweep_trace.py --master-trace model_tracer/traced_operations/ttnn_operations_master.json --sweep-trace model_tracer/traced_operations/sweep_trace_gpt_oss.json --output-report validation_summary.md
```

## Issues hit en route

| Issue | Resolution |
|---|---|
| `--module-name "all"` rejected by sweeps_runner | Drop the flag entirely (runs every module by default) |
| `DISPATCH_D_SHUTDOWN_SEM_ID was not declared in this scope` during dispatch JIT compile | Stale build — kernel sources newer than `libtt_metal.so`. Rebuild required. |
| Need full log output despite mid-run segfault | Use `script -q -c '<cmd>' run.log` (survives crashes) or `> >(tee logfile) 2>&1` |

## Current validation state

```
Total master configs: 130
Exact matches:        18
With diffs:            6
Hash mismatch:        63   (args match, hash differs)
Not exercised:        43
Coverage:             66.9%
```

`validate_sweep_trace.py` PASS condition requires:
1. 0 argument diffs (or all in `--ignore-categories`)
2. 0 hash mismatches
3. Coverage above `--pass-threshold` (none set by default, so "Not exercised" is informational)

## The 6 argument diffs — root cause confirmed

All 6 are `ttnn.linear` (2) and `ttnn.matmul` (4), and all show the same structural diff:

```
arg0.tensor_placement.distribution_shape   master: [4, 8]                                        sweep: [32]
arg0.tensor_placement.placement            master: [Replicate, Shard(-1)]                        sweep: [Replicate]
arg0.memory_config.buffer_type             master: L1                                            sweep: DRAM
memory_config (top-level kwarg)            master: {L1, INTERLEAVED}                             sweep: <missing>
```

### Why this happens

For master config 290 (linear, hash `083fe675988306b9...`):
- `arg0` (hidden_states): global `(1, 2880)`, sharded along K on cluster_axis=1 → **per-chip K = 360**
- `arg1` (wqkv): global `(2880, 128)`, replicated → **per-chip K = 2880**

`ttnn.linear`'s C++ validate (`matmul_device_operation.cpp:122`) compares per-chip `a_shape[-1]` vs `b_shape[-2]` and emits:

```
TT_FATAL: a_shape[-1] == b_shape[-2]
Mismatch: width=360 height=2880
```

The `try/except` in `linear_model_traced.py` catches this and recreates A and B via `_make_dram_tensors()` — which produces 1D `[32] [Replicate]` DRAM tensors. The retried call succeeds, and **that** is what gets traced. Hence the diff structure above is the fingerprint of the silent fallback firing, not a real metadata mismatch.

### Verified with hardware reproducers

1. `_restore_topology` works correctly — after `create_tensor_on_mesh`, the tensor's `tensor_topology()` reports `MeshShape([4, 8])` with `[Replicate, Shard(-1)]`.
2. `ttnn.linear` with that exact A and a K-replicated B fails with the TT_FATAL above.
3. The actual gpt_oss `apply_qkv_projection` in `models/demos/gpt_oss/tt/attention/operations.py` is a one-liner `ttnn.linear(hidden_states, weights.wqkv, bias=...)` with no upstream `all_gather` in the visible call chain — the trace must be capturing context that's not directly reproducible by calling `ttnn.linear` standalone.

### What this means

The master JSON records the *logical intent* of these matmul calls (TP-aware K-sharded inputs), but **standalone `ttnn.linear` doesn't accept this combination** — the model only works because of surrounding context (likely an upstream movement or a TP-aware wrapper) that the per-op tracer doesn't capture as a single replayable unit.

## The 63 hash mismatches

`validate_sweep_trace.py` flags them as "args match, hash differs". From `.cursor/rules/diagnose-config-hash-mismatch.mdc`, this is hash-function drift:
- The master JSON was reconstructed from trace 15 loaded into the DB on Apr 28 with that day's hash code.
- The sweep trace's `config_hash` is computed today by `generic_ops_tracer.py`.
- The most common cause is the `memory_config.hash` field (a device-specific runtime allocation hash that's never stable). Other categories: `shard_spec` `"None"` (string) vs `null` (JSON null) serialization, etc.

These are **not** functional mismatches; they're hash-computation drift. The validator still treats them as failing.

## The 43 "Not exercised"

Configs with no matching execution in the sweep trace. Concentrated in:

| Op | Missing | Likely cause |
|---|---:|---|
| `ttnn.reshape` | 15 | Sweep parameter generator filters out shape patterns |
| `ttnn.experimental.rotary_embedding_llama` | 4 | Sweep module exists but coverage gap |
| `ttnn.add` / `ttnn.multiply` | 3 each | Specific shape/dtype combos filtered |
| `ttnn.slice` / `ttnn.rms_norm` | 3 each | Coverage gap |
| `ttnn.scatter`, `ttnn.softmax`, `ttnn.transformer.{paged_,}scaled_dot_product_attention*` | 2 each | Coverage gap or filtered out |
| Other experimental ops | 1 each | Mixed — some may need module work |

## Path to 100% — ranked by viability

| Option | Effort | Reaches 100%? | Notes |
|---|---|---|---|
| **A.** Add upstream `all_gather` wrapper in matmul/linear sweep before the op | Medium | No | Fixes the TT_FATAL but trace still diffs (A becomes fully replicated, master expects sharded) |
| **B.** Run validator with `--ignore-categories tensor_placement memory_config` | Trivial | No | Diffs become PASS but 63 hash mismatches + 43 missing remain |
| **C.** Reload trace 15 into the DB with current code; regenerate master JSON | Low | Clears 63 hash mismatches | One-shot DB op; doesn't help diffs |
| **D.** Re-trace gpt_oss demo from scratch with the latest tracer; load fresh trace | High | Most accurate | Captures the *real* current op stream including any all_gather context |
| **E.** Write/extend sweep modules for the 43 missing ops | Very high | Necessary for full coverage | Days of work; doesn't help diffs/hashes |
| **F.** Fix `generic_ops_tracer.py` to strip `memory_config.hash` and normalize `shard_spec` before hashing | Medium | Clears most hash mismatches across all traces | Per `.cursor/rules/diagnose-config-hash-mismatch.mdc` Category 1 + 2 |

**Realistic shortest path to "validator PASS":**
1. **C** (reload trace) → clears 63 hash mismatches.
2. **D** (re-trace) → makes the 6 diffs match the actual current code.
3. Coverage will still be < 100% until **E**.

**Quickest path to a green validator with caveats:**
1. **B** + **C** — diffs ignored by category, hash mismatches fixed by reload.

## Files referenced

- `validation_summary.md` — full diff/mismatch tables
- `model_tracer/traced_operations/ttnn_operations_master.json` — reconstructed master from trace 15 (130 configs)
- `model_tracer/traced_operations/sweep_trace_gpt_oss.json` — output of the local sweep run (28 ops, 87 configs)
- `tests/sweep_framework/sweeps/model_traced/{matmul,linear}_model_traced.py` — sweep modules where the silent DRAM fallback lives (lines ~439–447 in `linear_model_traced.py`, lines ~381–401 in `matmul_model_traced.py`)
- `.cursor/rules/diagnose-config-hash-mismatch.mdc` — guide for the 63 hash mismatches
- `.cursor/rules/fix-sweep-trace-match.mdc` — guide for sweep module fixes
