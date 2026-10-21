# GPT-OSS-120B 4,352-byte P/D measurement record

Measurement status: all 29 cases completed. The final saved-evidence audit at 2026-10-21 13:39:04 UTC reports zero errors, failed cases or pending cases. This comprises 29 warmups and 87 measured requests. No numerical-accuracy validation was performed. Publication and allocation cleanup are recorded separately in REPORT.md and the accompanying evidence.

## Scope

The user selected a 4,352-byte prefill fabric packet payload and the original six chunk sizes. The historical P/D sweep used 2,880 bytes. No golden generation, golden comparison, PCC check, or new standalone sweep is part of this experiment.

- ISL: 1, 2, 4, 5, 8, 16, 32, 55 and 64 Ki tokens.
- Chunk: 1, 4, 8, 16, 32 and 64 Ki tokens; chunk must not exceed ISL.
- `K = 1024`; 55K therefore means 56,320 tokens.
- 29 cases, one warmup followed by three measured requests per case.
- Output cap 32, with EOS enabled. Report actual output tokens and finish reason.
- Full 36-layer GPT-OSS-120B, original quantized model weights.
- Prefill SP4/TP8 on one 32-chip Galaxy. Decode has four 4x2 reload stages on a second Galaxy.
- Capacity 131,072 tokens, two allocated slots, one active request.
- One retained decoder across the sweep. Prefill changes only between chunk groups.

## Source pins and change

| Component | Pin / experiment branch |
|---|---|
| Metal base, including the already validated measurement harness | `aa93f1b795694a2cea1dd8ceb952147dce620463` |
| Tested packet-size commit | `2898a1b47b7411611e9ab41b2873e16cf5aa9ac7` |
| New Metal branch | `divanovic/mg-gptoss-pd-payload4352-20261021-metal` |
| Blaze, unchanged | `d6d0b09f22f60a161d66fe4db90b9abed81ef82d` |
| Blaze's nested Metal | `44d66500520fda9f2c7060c0f6b41ec48f7ab37e` |
| D-gen, unchanged from the final historical repair | `5ca79e3076f193c092962d34a3b69731e39aff91` |

The sole model-source delta is in
`models/demos/deepseek_v3_d_p/reference/gpt_oss_120b_config.py`:

```python
FABRIC_PAYLOAD_SIZE = 4352
```

The preceding version used `EMB_SIZE`, whose value is 2880. The common prefill mesh builder consumes this configuration as its maximum fabric payload. It is not the size of a network KV migration message or a KV tensor chunk.

These measurements use fixed source pins rather than tracking a moving PR head. On October 21, PR [#60485](https://github.com/tenstorrent/tt-metal/pull/60485) had head `b1b537740f5206e47cdbef3ecd5c880bbd197f06`, whose GPT-OSS configuration sets `FABRIC_PAYLOAD_SIZE = 8704`. The historical measured commits `3441c663200ca87f7deafe7c48d39ba61c40d5c0` and `aa93f1b795694a2cea1dd8ceb952147dce620463` both set `FABRIC_PAYLOAD_SIZE = EMB_SIZE` with `EMB_SIZE = 2880`. No sweep in this comparison measures 8,704 bytes; the user selected 4,352 bytes for the new run.

`runtime/run-prefill-verified.py` asserts the effective imported value and saves each group's `prefill/fabric-config.json`. The audit checks both payload and chunk. The experiment has separate runtime and evidence directories while reusing the completed project's repositories, weight caches, Python environments and native-library staging. Preserve the historical results and Git pins when reproducing this arrangement.

## Hosts, access and prerequisites

Successful measurement hosts:

| Role | Host | IP | Historical reservation |
|---|---|---|---|
| Prefill | `bh-glx-120-c05u02` | `10.33.24.209` | `141422`, ends 2026-10-21 16:57 UTC |
| Decode | `bh-glx-120-b03u02` | `10.32.16.65` | `141554`, ends 2026-10-21 16:53:25 UTC |

These reservations authorize this recorded run only. A later reproduction needs fresh verified reservations, updated host/IP/service identifiers and new deadlines. Never reuse an expired allocation or reset another user's owner. The initial decoder `c08u02` was released after firmware recovery failed; none of the accepted request measurements use it.

Access from the laptop uses SSH ProxyJump:

```bash
ssh divanovic@slurm-login-01.exabox.tenstorrent.com
ssh -J divanovic@slurm-login-01.exabox.tenstorrent.com divanovic@bh-glx-120-c05u02
ssh -J divanovic@slurm-login-01.exabox.tenstorrent.com divanovic@bh-glx-120-b03u02
```

Direct login-node-to-compute SSH did not work in this session. The serial coordinator runs on the laptop, which must remain awake and connected. Model owners and services are detached remotely and bounded by deadlines.

Shared experiment root:

```text
/data/divanovic/gptoss-payload4352-20261021
```

Local coordinator/report root:

```text
/Users/divanovic/gptoss-payload4352-20261021
```

Prerequisites reused from the prior completed reproduction include the model at `/mnt/weka/model-weights/llm/openai/gpt-oss-120b`, separate prefill/decode Python environments, the native KV manager and Python bindings, Dynamo frontend/workers, etcd, OpenMPI 5.0.7 ULFM and staged native libraries. Metal builds are node-local and must match the corresponding Python imports. The complete dependency recipe is the existing Metal repro `PERFORMANCE_MEASUREMENTS.md`; this record documents the additional packet experiment.

## Exact recorded execution flow

Inspect `experiment.json`, `runtime/current-pd-run.json`, `runtime/setup-coordinator-status.json`, launch records and actual PIDs before any launch. The scripts refuse duplicate records and active device holders. Historical preparation scripts contain fixed hosts and deadlines and must not be blindly rerun.

1. `prepare_remote.py` created the new root, branch and independent evidence/runtime, preserving the completed baseline. Native builds ran through `runtime/build-role.sh` for prefill and the original decoder, then `runtime/build-role-b03.sh` for the replacement decoder. Build commands and environment are preserved in these scripts and per-role launch records.
2. Prefill starts through each group's `runtime/sweep-pd-c<chunk>/start-prefill-owner.sh`. Its environment sets full36, two slots, SP4/TP8, capacity131072, asynchronous layer completion, real file-exported KV tables, linear topology with the 1D fabric, and the single-Galaxy torus XY mesh descriptor. Tracing is off. The launch executes the verified wrapper over the unchanged timing wrapper and common prefill runner.
3. The retained decode driver is `runtime/start-decode-owner-b03.sh`. Readiness requires `runtime/repro-full36/decode/external.ready.json`, `decode.pb` and `decode-device-map.txt`; completed capture count alone is insufficient.
4. `runtime/start-service-v4.py` starts etcd/frontend once and native manager/worker pairs for each group. Group configurations are `worker-prefill.json`, `worker-decode.json`, `manager-prefill.env.json` and `manager-decode.env.json`. Service launch records preserve arguments, environment and deadlines.
5. The completed coordinator was `resume-after-prefill-recovery.py`, with `ops_b03_graceful.py`, `ops_b03.py`, `case_ops_b03.py` and an exclusive `run-sweep.lock`. It resumes after the preserved initialization failure, keeps DONE evidence, and runs remaining groups serially. Do not run an earlier coordinator alongside it.

Representative exact case invocation, run on the prefill host by that coordinator:

```bash
ROOT=/data/divanovic/gptoss-payload4352-20261021
RUN="$ROOT/runtime/sweep-pd-c32768"
timeout --signal=TERM --kill-after=20s 2100s \
  "$ROOT/tt-gen/adapters/dynamo/.venv/bin/python" \
  "$ROOT/tt-metal/models/demos/gpt_oss_d_p/repro/run_pd_case.py" \
  --run "$RUN" \
  --output "$ROOT/evidence/sweep/pd/55k-c32k" \
  --model /mnt/weka/model-weights/llm/openai/gpt-oss-120b \
  --isl 56320 --chunk 32768 \
  --prefill-host bh-glx-120-c05u02 \
  --prefill-ip 10.33.24.209 --decode-ip 10.32.16.65 \
  --deadline 1792600380
```

This is a recorded invocation, not a command to submit again into an existing output directory. The harness retains exact request bodies, responses and one warmup plus three measurements. Prompt generation and prefix-reuse accounting match the historical harness; prompts are synthetic with fresh nonces, not the realistic-text prompts in the motivating packet-size note.

## Graceful group changes

After every case in a chunk group is DONE and the saved-evidence audit has no errors, stop only that group's workers and managers. The guard in `runtime/shutdown-prefill.py` verifies completed cases, exclusive stream ownership, service identity and the expected live prefill owner. It connects to the existing H2D service and sends the runner's supported `(-1,-1,-1)` shutdown metadata. It does not open a second mesh or run an extra model request.

Before launching the next prefill, require all of:

- The runner logged `SHUTDOWN sentinel received` and `shutdown complete`.
- The owner exit marker is zero.
- No process holds the prefill devices.

Keep the decode owner alive. The first verified transition closed 32K at 12:48:18 UTC and initialized the subsequent 16K fabric on all 32 devices without recovery. Socket prior-destructor and nanobind teardown warnings remain in the raw logs. A failed close must halt the coordinator for diagnosis; it must not silently launch a second owner.

## Measurements and acceptance

TTFT is client-observed first-token latency. Prefill throughput uses actual newly prefilled tokens divided by the host-observed prefill service span; separately retain requested, reused, logical processed and fixed-width padded counts. Decode t/s/user uses the native host-commit interval and actual generated-token count minus one, not the whole request time.

Each case preserves transfer wire bytes; summed transfer durations; the union of active transfer intervals; transfer span; overlap with active prefill; non-overlapped prefill and transfer; the initial exposed prefill segment; and the final transfer tail. Interval sums, active unions and spans are distinct. Neither overlapping transfers nor concurrent reload-stage timings may be added to wall time or TTFT.

The table uses means of three measured requests and sample standard deviations; variance and raw runs are retained. A warmup is reported separately. A DONE performance case is not a numerical-accuracy pass.

Saved-evidence audit, with no device traffic:

```bash
python3 /data/divanovic/gptoss-payload4352-20261021/runtime/audit-pd.py \
  /data/divanovic/gptoss-payload4352-20261021 \
  --output /data/divanovic/gptoss-payload4352-20261021/evidence/report/sweep-audit.json
```

Final acceptance requires `complete: true`, all 29 expected pairs, no failed/pending cases and no audit errors. The audit checks payload configuration, request count, actual token accounting, output cap/EOS, migration counts, overlap conservation, native TSU formula, means and sample variances. Failed infrastructure attempts remain separate evidence, even when their replacement succeeds.

All applicable pre-commit hooks for the changed model configuration passed in the disposable `runtime/payload-hook-copy`. Both the tested copy and executing source retained SHA256 `69aa219abb010fbeacf12236600eea7453d43d39ed2e993e797b254ab7bd1a91`. Guard checks also verify that the shutdown client refuses incomplete groups and live workers. Applicable report/document hooks are recorded with the published result artifacts.

## Setup and obstacles

- Original decoder initialization failed after approximately 302 s waiting for ARC firmware, before model weights or reload capture. The bounded site recovery timed out at 900 s; the node was released.
- Replacement decoder readiness took 2340.198 s (39m00s) from driver launch. It completed 41 images:36 decoder layers, four head/bypass images and embedding. Four stages prepare concurrently.
- The first 64K prefill startup took 603.0 s. A 32K prefill restart failed during mesh opening with an Ethernet heartbeat timeout before model loading. Failure evidence was preserved; prefill-only recovery passed in 64.578 s, with 32 devices and healthy links. Decoder stayed loaded.
- Fresh 32K and 16K prefill readiness took 222.565 s and 230.700 s, respectively. Further group times are in the report.
- The earlier signal-based owner shutdown lacked explicit mesh-close logs. The new sentinel transition provides that evidence; this does not prove the old shutdown caused the firmware fault.

`setup_summary.py` reports per-host native compilation/import checks, each failed/successful startup, recovery, and per-stage reload load/build, weight staging, tap, configure and readback totals. Do not sum concurrent stage totals to estimate bringup wall time.

## Final shutdown and resource release

After all 29 cases passed, the final prefill owner closed through its native sentinel with exit 0. The decoder owner-stop path in the unchanged Blaze showcase calls `os._exit(0)`; the MPI launcher reported improper rank 0 termination and returned 1. This is not a clean decoder shutdown pass. All ranks, driver and task services had exited; both hosts had no remaining device holders. Allocations 141422 and 141554 were cancelled at 13:49:08 UTC. Raw shutdown logs and explicit cleanup/release records are preserved. No further model traffic was run.

## Comparison limitations and artifacts

The old hosts were c02u02/c06u14; new measurements use c05u02/b03u02. Historical D-gen cases span documented repair cohorts; this sweep consistently uses the final pinned D-gen commit. The graphs show observed matched-case differences, not isolated packet-size causality. No datatype/model-math change was introduced, but no PCC result is claimed.

The live page is `http://127.0.0.1:8768/payload4352-comparison.html`. Both plots share axes, colors and right-side chunk toggles. Missing cases stay blank. Improvement is `(new mean / old mean - 1) * 100%`, alongside the absolute difference. During the run, `publish.py` read saved remote evidence every 20 s; after completion the publisher stopped and the final snapshot remains available. `make-report.py` produces REPORT.md from the fresh local snapshot. Neither submits traffic.

Retain the final JSON/CSV/report, per-case evidence, effective fabric configurations, launch commands, source pins, failed attempts, recovery logs and coordinator scripts. Archive executing scripts only after they exit. After the final audit and publication, stop verified task-owned services and models, release both allocations and verify release. Exact historical allocation IDs must never be used to cancel an unrelated future job.
