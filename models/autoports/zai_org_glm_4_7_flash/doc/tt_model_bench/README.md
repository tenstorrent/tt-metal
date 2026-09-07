# Container benchmarks, GLM-4.7-Flash on one Blackhole p150

Measured against the v5.1 container package (`stisiTT/glm-4-7-flash-p150`), not the host
serving path, through tt-inference-server's benchmark workflow targeting the already-running
container.

Run on 2026-09-07. 23 sweep points, 0 failed requests. Acceptance PASS.

    cd tt-inference-server
    ./venv/bin/python run.py --model GLM-4.7-Flash --tt-device p150 \
      --workflow benchmarks --service-port 8000 --dev-mode --no-auth \
      --skip-system-sw-validation --disable-trace-capture

Note there is no `--server-url`. Passing it marks the connection remote, which drops the
port and makes the client poll port 80 until its 1200 s health timeout expires.

## Headline

| Scenario | TTFT | TPOT | Throughput |
|---|---|---|---|
| 1 user, ISL/OSL 128/128 | 296 ms | 29.5 ms | 31.7 tok/s |
| 32 users, ISL/OSL 128/128 | 9.3 s | 91.6 ms | 391 tok/s total |
| 1 user, 128K context | 1033 s | 276.8 ms | 0.1 tok/s |

TPOT of 29.5 ms at the reference point is identical to the figure the host bring-up recorded
in `doc/tti_release/RUN_NOTES.md`, so the container reproduces the validated serving path
rather than approximating it.

## Two things a reader should not misread

**High TTFT at concurrency 32 is the design, not a defect.** Chunked prefill is disabled, so
simultaneous prompts prefill serially and TTFT is measured from submission. This package is a
throughput profile. At 128/128 it delivers 391 tok/s in aggregate.

**The one graded point misses both its targets.** TTFT 296.2 ms against a 274.07 ms target
(8.1% over) and 31.7 tok/s/user against 33.903 (6.5% under). Reproduced across two
independent runs at 8 and 256 requests, so this is real and not measurement noise. The other
22 sweep points have no targets defined and are reported NA.

Full report and raw JSON are in this directory.
