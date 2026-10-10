# Compact GDN: completed full-model candidate, qualification pending

October 10, 2026 UTC. The candidate finished both contexts at 19:06:43 with
clean teardown. Each cell has one warmup and three measured repetitions of
the full 64-layer model on one TP4 replica, B16, 128 output tokens (127 timed
decode steps). BFP8 weights/KV, BF16 activations, FP32 recurrence and no
speculative decoding are unchanged. HTTP overhead is outside this timing.

| Context | Before TSU | Compact TSU | Gain | Before step | Compact step | Compact output tok/s/TP4 |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 16.5837 | 20.0350 | 20.81% | 60.3001 ms | 49.9128 ms | 320.5593 |
| 16K | 18.4059 | 22.7506 | 23.60% | 54.3303 ms | 43.9549 ms | 364.0094 |

All before/candidate warmup and measured output hashes match at each context.
Both raw arms match the frozen model/policy manifest. No measured repetition
includes a trace capture. The shared L1 scratch pool clears the previously
failing full-model prefill/head boundary in this run. This is evidence of
successful recovery; it is not an instrumented allocator-address proof.

The after-control is still running. Stable before/after controls and compact
GPQA qualification remain pending. The previous qualified fusion policy
scored 177/198 (89.39%); that score does not qualify compact GDN.

Compact prefill remains approximately unchanged: 95.8414 s at 32K and
44.3187 s at 16K. Median TTFT is 95.9684 s and 44.4720 s respectively.
These are fresh full prefills without prefix reuse. The isolated decode
improvement must not be applied to prefill or whole-request throughput.

Eight times the measured 32K TP4 output rate is 2564.47 tok/s/Galaxy at
128 active users. This is an unverified scaling projection. Thirty native
TSU still needs 16.5794 ms less per step, or 49.74% more output throughput.

## Persistent qualification

`qwen38-compact-followup-v3-20261010.service` was verified live with PID
244905, invocation `28c688b7780445999483523a9cbc75fc`. It waits for the exact
compact-v3 invocation and clean after-control. It recomputes the raw
comparison, requires matching generated tokens, stable controls and a >=1%
32K gain, then runs eight-replica G0 and full 198-question GPQA with 65536
output-token budget. All questions, including truncations, stay in the
denominator. Matching compact whole-model profiling follows GPQA.

The follower uses the same source directory and manifest as the timing run.
No duplicate hardware job was launched and no serving policy was promoted.
The user systemd unit survives client disconnect, not reboot. Its configured
limits are 36 hours and 256 GiB host RAM; the existing hardware lock serializes
execution. Source and launch arguments are retained below.

## Receipts

- `sweep.json`: original first-measured-repeat snapshot, still running.
- Root `compact-sweep.json`, queue and capture receipts: the 19:06:23 snapshot,
  with completed 32K and two measured 16K repeats.
- `completed-candidate/`: 19:07:48 capture, both candidate cells completed,
  clean teardown, after-control running and qualification follower waiting.
- `measured-progress.json`: values recomputed from completed raw arms;
  explicitly not an after-control or GPQA pass.
- `collect-progress.py.gz`: read-only collector. Raw logs are gzip-compressed
  without editing their contents. Captures are sequential reads, not atomic
  snapshots of every remote file.

The published-versus-frozen convolution reader has the already documented
[formatting-only difference](../compact-long-horizon-v2/source-format-difference.json).
The live timing and qualification sources match each other exactly; do not
describe the published checkout as byte-identical to that frozen tree.
