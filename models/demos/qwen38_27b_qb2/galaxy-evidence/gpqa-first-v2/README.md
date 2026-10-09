# GPQA-first eight-hour queue, Oct 9 2026

Active on the allocated `10.228.203.98` Galaxy as
`qwen38-gpqa-first-v2-20261009.service`, invocation
`a6b386c9e3734c849202bb266bf4ff23`, launched 16:26:38 UTC.
After a locked Galaxy reset, the endpoint passed startup/API checks and entered
corrected full GPQA at 16:37:22 UTC. All eight workers report the G0-qualified
BFP8 policy. At the 16:43 UTC observation, 48 responses had completed; this is
not a final accuracy result.

## Order and bounds

1. Full 198-question GPQA with the scientific-notation fix, existing T=1,
   top-p=.95, top-k=20, seed=42, 65536 output tokens and unchanged 177/198 gate.
2. Full 198-question GPQA using unmodified OpenBench task/prompt/scorer at
   `d03922ccbbd9004e802c556ef9c2a40ed870659c`, T=.5. Explicit differences:
   one epoch rather than upstream's ten, a 65536-token output cap, and our local
   endpoint. OpenRouter's private overrides are unknown. Results stay separate;
   no better-score selection, merged answers or silent rescoring.
3. Native BFP8 one-TP4 perf: 32K then 16K, B8/B16/B32 plus B1, prefill token
   budget 32768, one warmup and three measurements, 128 generated tokens.
4. Longer-context checks: 128K/B4 and B8, near-256K/B4, within the existing KV
   capacity guard. These are not claims of maximum supported concurrency.
5. Matched 16384 and 65536 prefill-token-budget sweeps at 32K/16K, B8/B16/B32.

The systemd unit owns all descendants, has an eight-hour runtime limit and
210-second stop grace, and survives SSH/session disconnection with user linger
enabled. It does not resume after a host reboot. Stages reserve cleanup time;
the last experiments may remain unrun if the budget expires. Every hardware
stage uses `/tmp/tt-device.lock`. A hardware/process failure aborts the queue;
an accuracy miss is retained and does not prevent independent perf measurement.
The queue does not change serving defaults or promote precision policies.

## Validation and recovery

- 47 CPU tests passed; original model/kernel/precision bytes match the prior
  passing eight-replica G0 receipt. Only evaluator/controller code changed.
- The OpenBench mirror matches all 198 original GPQA question/answer records.
  Its pinned dataset revision is `c63e9ba02dc3da4c698e2a8485551b35041c3900`.
- A local synthetic HTTP test verifies the exact model ID, T=.5, 65536-token
  request, final-answer scoring independently of reasoning, and terminal log
  extraction. It opens no device and is not benchmark-quality evidence.
- Attempt v1 exited before inference: DP5 could not map its four-chip fabric
  graph onto discovered connectivity. All owned workers stopped. v2 preserves
  that failure, resets under the lock and has progressed to real GPQA inference.
  Reset recovery does not establish whether the initiating fault was hardware
  or software.
- Collected frozen scripts are AST-identical to the formatted branch versions.
  `collection.json` records exact evidence hashes and that comparison.

## Observe and reproduce

Task-owned remote roots:

```text
/home/ttuser/qwen38-artifacts-20261007/gpqa-first-v2
/home/ttuser/qwen38-artifacts-20261007/gpqa-first-control-v2
/home/ttuser/qwen38-artifacts-20261007/gpqa-first-source-v2
/home/ttuser/qwen38-artifacts-20261007/openbench-overnight-v1
```

```bash
systemctl --user status qwen38-gpqa-first-v2-20261009.service
cat /home/ttuser/qwen38-artifacts-20261007/gpqa-first-v2/queue.json
cat /home/ttuser/qwen38-artifacts-20261007/gpqa-first-v2/qualification/evaluation/gpqa/gpqa-progress.json
cat /home/ttuser/qwen38-artifacts-20261007/openbench-overnight-v1/summary.json
```

`launch.json` contains the exact argv, environment and limits. Reproduction
requires new result/control paths and a fresh unit name, the pinned checkpoint,
the frozen source manifest, prior exact-runtime G0 receipt, and the OpenBench
environment installed with its `uv.lock` (`uv sync --frozen --no-default-groups`).
Run `demo/run_openbench_gpqa.py --prepare` before dispatch; it verifies dataset
identity and caches inputs. Never overwrite this run's outputs.

Private prompts and raw model completions stay on host disk. Public evidence
contains configuration, scores, hashes, timings and synthetic transport data.

## Completed earlier overnight work

The original BFP8 GPQA finished 171/198 (86.36%), zero truncations, 59m51s.
That result predates the preprocessing repair and is retained unchanged.
Chunked-state continuation/interleaving/slot-remapping passed; scheduler and
sampling qualification remain open. The completed B16 precision comparison:

| Policy | 16K output tok/s per TP4 | 32K output tok/s per TP4 | 16K input tok/s | 32K input tok/s |
|---|---:|---:|---:|---:|
| BFP4/LoFi, BFP8 head | 225.47 | 207.93 | 6560.50 | 6002.79 |
| BFP4/HiFi2, BFP8 head | 221.12 | 204.22 | 5883.79 | 5432.48 |
| BFP8/HiFi2 | 202.29 | 188.03 | 5746.89 | 5313.62 |

These are native measured TP4 rates, not Galaxy HTTP throughput. BFP8 costs
8.52%/7.93% output throughput versus BFP4/HiFi2 at 16K/32K respectively.
