# Compact L1 scratch recovery, October 10, 2026

Compact v2 loaded all 64 layers, then failed in the first B16/32K prefill's
vocabulary-head matmul. Its static dataflow-buffer region ended at **850944**,
while a live L1 allocation began at **825088**. The native allocator rejected
the overlap before candidate decode timing. The sweep confirmed clean device
teardown; all three followers stopped without starting hardware. No reset or
reboot was needed. The original failed run and receipts are retained.

The compact implementation allocated four persistent L1 tensors per GDN layer,
so the 48-layer stack retained 192 tensors per batch shape. The isolated
one-layer test did not expose that accumulation. This is the leading cause of
the full-model collision; an exact allocator-address attribution is not yet
instrumented.

`CompactScratch` now owns one four-tensor pool per replica and batch shape.
The ordered CQ0 layer stack consumes Q/K/V and the fused epilogue output before
the next layer overwrites them. Per-layer FP32 recurrent state, convolution
history, DRAM workspaces and independent replicas remain separate. The pool
rejects cross-mesh reuse and retains buffer identity across batch changes.
This changes allocation ownership, not kernel arithmetic, precision or native
installation. A CPU regression test models 48 layers: four compact L1 tensors
at B16, eight with B32 also prepared, rather than 192/384.

Frozen preflight passed **557 tests, 69 subtests, one skip** and collected the
real-weight and full-trace hardware tests. The new source also includes the
default-off batched-prefill prototype; `QWEN_PREFILL_BATCHED_ATTENTION` is not
enabled. Fresh full-model before/compact/after runs will compare the same source
and workload, followed by G0/GPQA only on a measured win. Successful small
tests do not establish that the full-model collision is resolved.

The persistent recovery queue is:

1. `qwen38-compact-gdn-v3-20261010.service`, invocation
   `ddc627b6d8ff4f80bcee38abd1294378`.
2. `qwen38-compact-followup-v3-20261010.service`, invocation
   `28c688b7780445999483523a9cbc75fc`.
3. `qwen38-projection-sweep-v5-20261010.service`, invocation
   `6f7c63955e144aa79693e4d8260cef7c`.
4. `qwen38-prefill-attention-v2-20261010.service`, invocation
   `258b685967ee458f84d0d0c809700fad`.

Recovery verified every old unit's exact terminal invocation, the failed
sweep's clean teardown and hardware_started=false on all followers. Fresh
source/control/output directories preserve the failed attempts. Existing
hardware-lock, runtime and memory bounds remain. Follow-up uses the exact
compact-v3 source and manifest; the profile directory is a new bounded path
under `/dev/shm`. Services survive disconnects, not reboot. Projection and
prefill followers retain their existing frozen sources.

Measured B16/32K decode remains **16.55 TSU**, with prior full GPQA **177/198**.
Compact's approximately **20-TSU** expectation comes only from its isolated
block. The 30-TSU target and a compact full-model speedup remain unproven.

At 18:34 UTC, the recovered physical epilogue and real-weight block tests had
passed, including all-rank changing-input equivalence through64 updates at
B16/B32. The new before-control model was loading. Full-model recovery remains
pending. Follow-up launch metadata was rebound to the actual compact manifest;
the original copied metadata is retained beside the corrected record.
