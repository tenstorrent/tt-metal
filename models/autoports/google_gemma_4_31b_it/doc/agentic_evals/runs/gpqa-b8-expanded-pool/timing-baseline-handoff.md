# Held C4/B8 matching cadence baseline

Private helpers under `bringup/artifacts/gpqa-b8-c4-timing-baseline/`. Forecast0.20deviceh/box0.30 within coordinator efforttotal3.5h. Prepared only; root owns queue, hardware launch, health wait and closure. No central files changed.

Driver differs from completedC8 only in output path and client-count-dependent values8→4: independent histories/workers, required active rows, expected cache queries and planned_clients. Server remains B8/maxseq8/full60/context262144 with APC/chunking/sampling unchanged; optional GEMMA4_B8_KV_POOL_IDS explicitly unset both before launch and in shell. Debug response loggingfalse, retained phases enabled,512MiB private phasefilecap. Same five lengths4096/32768/65536/98304/131072, full max_tokens126976/requesttimeout14400. Own streams close after all4 receive>=128 and active4 phase residency proof, then wait for idle. No repeated native exactness or B1. Fresh nonce avoids stale cache history; synthetic outputs are not scored.

Coordinator launch from execution checkout after queue registration and /proc owners empty (server independently verifies owner absence and frozen runtime hashes):

```bash
python -u bringup/artifacts/gpqa-b8-c4-timing-baseline/server.py \
  > bringup/artifacts/gpqa-b8-c4-timing-baseline/device-driver.log 2>&1
```

After normal `/health` readiness200, in another managed session:

```bash
PYTHONUNBUFFERED=1 python_env/bin/python \
  bringup/artifacts/gpqa-b8-c4-timing-baseline/capacity.py \
  > bringup/artifacts/gpqa-b8-c4-timing-baseline/capacity.log 2>&1
```

On successful client completion and fresh idle, close owned readiness process using reused guard:

```bash
python -u bringup/artifacts/gpqa-b8-c4-timing-baseline/idle-close.py
```

The launcher and client preserve original long per-request/safety limits; the0.30h effort box is coordinator-managed, not a shortened model cap. Do not kill active TT work to force box compliance. Client’s own stream-disconnect/idle protocol is unchanged. Any nonidle/error requires preserving evidence and coordinator handling.

After completion derive full-active4 completion cadence at each context from retained model phase boundaries; compare descriptively with existingC8 same protocol and actual cache reuse. Distinguish per-user cadence, aggregate tokens/s, prefill time and setup. This supplies matching shape timing, not a causal GPQA speed result, new capacity acceptance or scored cohort. Final evidence will be separate `timing-baseline.json` after measurement. Python syntax and shell syntax passed; source tree unchanged.
