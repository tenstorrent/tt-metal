# Held local source-regression benchmark handoff

Private prepared helpers: `bringup/artifacts/stage12-expanded-pool-benchmark/{server.py,client.py,guard.py,commands.json,source-frozen.json}`. No launch performed. Root owns queue registration, capacity-server clean closure, benchmark device accounting and graceful idle shutdown. Helpers do not edit central docs or queue records.

Forecast0.07605deviceh, box0.15h within expanded-pool effort2.5h. Execution source3a7704d65abc05d4d9d9b5c84e73fcc0136f22f3; publication4b7d9a3b4142dff5e8dd804438d6c42d9626bdcc. Frozen runtime includes14Python files plus selected precision JSON, launcher and unchanged stage11 runtime-model JSON. Actual TTI client SHA frozen as af1ef17abfecb059da13d0a1287c00eace0f210e with client/config source hashes. Guard fails before hardware if any bound file or client HEAD changes. HEAD may later advance only if coordinator deliberately refreshes this handoff/source binding before launch; do not bypass guard.

After capacity has closed and root verifies owners empty, from execution checkout in separately managed sessions:

```bash
python -u bringup/artifacts/stage12-expanded-pool-benchmark/server.py \
  > bringup/artifacts/stage12-expanded-pool-benchmark/device-driver.log 2>&1
```

```bash
python -u bringup/artifacts/stage12-expanded-pool-benchmark/client.py \
  > bringup/artifacts/stage12-expanded-pool-benchmark/client-driver.log 2>&1
```

Uses prior successful phase-archive benchmark protocol with PYTHONUNBUFFERED=1 for every subprocess. B1/C1/maxseq1/context262144/APC, full60, stock async and operator top_k32 default unchanged. Optional GEMMA4_B8_KV_POOL_IDS explicitly unset; GEMMA4_SERVING_TEST_LAYERS unset. Response debug explicitly false; direct readiness launcher has no request-logging flag and does not consume the new TTI model YAML, so stock request logging remains disabled. Preserve periodic phase/token diagnostics. Confirm actual startup arguments and source identity when running.

Order: separate4096/128 warmup1, then exact4096input/128output headline8 at C1, then unchanged stage11 TTI matrix128/128 n8 at C1. Same random seeds/ignore_eos/tokenizer pin/benchmark settings as prior successful commands; warmup is not a headline measurement. TTI uses existing serialized stage11 spec and ONLY_BENCHMARK_TARGETS=1. It does not dispatch normal evals or change their caps. Client health ceiling450s is a fail-safe, not forecast. Server outer safety timeout540s equalsbox; root should close promptly after client completion and fresh running0/waiting0, not rely on timeout to terminate active work. No new shutdown helper or forced termination prepared.

No results claimed before measurement. Save actual generated lengths, native source/full60 identity, durations, phase accounting and safe closure to coordinator run evidence. This is normal new-source regression measurement while CI image builds, not telemetry repair. Guard and Python syntax checks passed host-only.
