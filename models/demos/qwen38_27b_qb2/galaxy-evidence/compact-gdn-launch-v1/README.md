# Compact GDN implementation and persistent launch

Snapshot: October 10, 2026, approximately 06:55 UTC. **Hardware qualification
has not started.** User target is 30 native tokens/s/user at B16/32K, with no
speculative decoding. This opt-in experiment is not a serving promotion.

## Change and expected benefit

`single_step_compact_gdn` connects the packed DRAM projection, compact
convolution/history update, direct Q/K/V preparation, FP32 recurrence and gated
output into one model boundary. QKV/z and normalized output remain compact in
L1; only 64 scalar-gate channels expand. Native weight matmuls and TP reduction
remain in the measured boundary. Prefill and small decode buckets retain their
existing paths. No weight/KV/state precision changes.

The epilogue reads z from the packed projection at channel 2560. Aligned
64-byte paired reads handle odd users on Blackhole. Disjoint 32-byte face-row
writes preserve neighboring users; the final live user of each head zeros
only unused rows. Reader and writer scratch are separate.

Target is another 4-6 ms off the measured 60.422 ms B16/32K step: approximately
17.7-18.4 TSU if achieved. This is a subset of broader compact/fused decoder
work, not enough to reach 30 TSU. No gain is yet measured for this change.

## Validation and queue

- Local descriptor validation: 4 unittest cases with layout/boundary subcases.
- Allocated-host CPU preflight: 512 passed, 69 subtests passed, 1 unrelated skip.
  Three physical test entry points collected without opening the accelerator.
- Physical plan: 26 epilogue cases across public/compact/packed gates, output
  layouts, L1/DRAM and B16/B32 plus B1/B17/B31 boundaries. Rank-distinct inputs,
  poisoned padding, allocation rebinding and A/B/A changed-input trace replay
  compare against native arithmetic and the public epilogue.
- Real-weight layer: B16/B32/B8/B1 control/compact/control, 64 FP32 reference
  updates, exact logical operand/state/projected-output hashes. Additional
  B16/B32 sessions change inputs and compare recurrent state, history and
  projected output on all ranks at seven checkpoints through 64 steps.
- Only after those pass: B16 full-model before/compact/after sweeps at 32K and
  16K, one warmup and three measured 128-token generations per cell. Report
  input throughput, TTFT, decode TSU and all-in output throughput. Require
  stable controls and identical generated tokens for a qualified comparison.
  Full GPQA and serving promotion remain separate gates.

No active hardware run was interrupted. Only the exact waiting projection
follower was stopped before hardware acquisition, and its frozen source is
reused. Current ordering:

1. `qwen38-gdn-fusion-full-v2-20261010.service`
2. `qwen38-b16-priority-v2-20261010.service`
3. `qwen38-compact-gdn-v1-20261010.service`, PID 3704023 at verification,
   invocation `436e57932b7349868ee2d8042b959830`.
4. `qwen38-projection-sweep-v2-20261010.service`, PID 3704026 at verification,
   invocation `2a7ff1771ed945398736788e16613aa9`.

The controller waits for the exact predecessor and a clean terminal receipt,
verifies source hashes before every stage and uses the shared hardware lock.
Physical stage limits are 1 hour, 1 hour and three 90-minute sweeps. Estimated
physical elapsed time is 90-180 minutes, excluding predecessor wait. Unit caps:
192 GiB RAM, 8 CPUs, 38 hours including waiting. Survives disconnect, not reboot.

Host directories under `/home/ttuser/qwen38-artifacts-20261007`:
`compact-gdn-source-v1`, `compact-gdn-control-v1`, `compact-gdn-v1`,
`projection-sweep-control-v2`, `projection-sweep-v2`.

## Evidence boundaries

`control/source-manifest.json` is authoritative. The snapshot uses the previous
projection snapshot plus this source overlay; the parent Git label alone does
not identify every inherited file. Twelve inherited files differ from the
checkout. The active convolution reader diff is formatting only; the bounded
runner adds an unused optional artifact-budget argument. Other differences are
older evaluation utilities. `inherited-source-differences.json` records hashes.
No frozen running snapshot was edited to match the later checkout.

The native-after snapshot reproduced B16/32K at 14.8715 TSU (67.2427 ms), versus
the existing fusion candidate's 16.5502 TSU (60.4222 ms). This supports that
earlier timing gain, not the new compact policy's correctness or performance.

Initial SSH staging and clang-format temporary-index access hit sandbox limits;
explicit escalated retries succeeded. Formatting corrections preceded host
preflight. No native installation, firmware, NFS or serving-default changes.
