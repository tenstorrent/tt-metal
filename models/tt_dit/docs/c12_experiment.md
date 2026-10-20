# C12: one main-vocoder convolution, four CPU-prepared cells

This default-off experiment is based on historical source
`b9f8587ce6c681f380e341753c6f0ea7f5ee441d`. It has no native acceptance.
It changes only `vocoder.resblocks.16.convs2.0`, after verifying explicit checkpoint
configuration, the main-vocoder module structure, the original DilatedConv1d type,
C24/C24/K7/dilation1/stride1/zero padding and checkpoint weight/bias identities.
Absent checkpoint fields are an eligibility gap; defaults are not evidence.
BWE, activations, upsampling and the global `_FP32_BLOCKINGS` remain unchanged.

`LTX_C12_MODE` is `off` by default. Explicit modes are:

| Mode | Pack | T block | Purpose |
| --- | ---: | ---: | --- |
| A | 1 | 4 | Pinned effective baseline blocking |
| B | 1 | 32 | Blocking-only control |
| C | 4 | 4 | Packing with A blocking |
| D | 4 | 32 | Packing with B blocking |

`auto:A` through `auto:D` permit recorded baseline fallback. Explicit cells reject
incompatible construction/input before the experimental operation; upload preflight
rejects incompatible batch/local lengths before device upload. Auto cells retain a
separately prepared original module for shape fallback, increasing resident weights;
do not mix their capacity/timing cohort with explicit cells. Missing checkpoint evidence
in automatic mode keeps the original module and emits a selection reason. All inputs
must be logical batch1/C24 ROW_MAJOR FP32, interleaved DRAM, channel-factor1 and a
single sharded time axis. Pack4 also needs localT divisible by4, localT/4 >=32, and
aligned shard starts derived from the actual upload's contiguous zero-origin partition.
Direct wrapper calls without upload/partition evidence fail closed (or auto-fallback).

All four cells keep C in/out blocks32, H/W blocks1, FP32 input/weight/output, HiFi4,
approximate math false, FP32 destination and requested packer accumulation true,
split mode off. Effective native packer behavior remains to be attested. Mode and
precision/configuration are fixed at construction before weight loading; use a new
process to change them. The original k7 blocking is4 because the later BWE dictionary
entry overwrites64. No global-table repair or inherited packed block8 is used here.

Use a **separate empty owned `TT_DIT_CACHE_DIR` for each process/cell**, including a
separate ordinary baseline process. Experiment startup writes an exclusive
`c12-session.json`; reuse of any populated root is rejected. The experiment-only
vocoder suffix binds transform schema, mode, original geometry, phase-major weight
layout, blocking/precision, checkpoint config/tensor hashes and source identity,
mesh and partition configuration. Default/off keeps the original namespace. Existing
cache content manifests continue to apply. Checkpoint source replacement after
construction rejects reload; actual prepared tensors must match the checkpoint
hashes. The default source identity may be a stat identity for a non-content-addressed
checkpoint: native acceptance must additionally pin the complete checkpoint hash.

For pack p, `X[q,r*24+i]=x[p*q+r,i]`. For each original tap j and output phase r,
`delta,s=divmod(r+j-3,p)` and
`V[r*24+o,s*24+i,delta+ceil(3/p)]=W[o,i,j]`, all other entries zero.
Bias repeats `[b0..b23]` per phase. At pack4 this is C96/K3.
Logical strided phase slices exclude physical C24->C32 padding, followed by C concat,
a one-row neighbor halo on each side, 5D reshape, ordinary conv3d with no internal
pad, trim, unpack to C24 and an owned output clone. Reshapes may copy. The caller's
original-sample zero-tail update still happens before the wrapper, and subsequent
replicate-tail activation operates on unpacked original samples. No caller/alias or
persistent CCL allocation is force-deallocated. Intermediate Python references remain
live through the call; actual buffer ownership, page sizes and trace allocations need
native evidence, not inference from these references.

The execution JSON log and `execution_manifests` bind selection/fallback, checkpoint,
shape/padded shape, original tail length, global starts, placement, finite time-block
counts and source-derived tile-product estimates. Vocoder and complete audio-chain
trace keys bind the input shape and experiment identity. Native collection must retain
those records together with the trace IDs and executed build/source identities; they
are not a replacement for the project's acceptance manifest/raw-log binding.
At localT132, A/B/C/D estimated tile products are231/35/243/54. The long-shape27/28
MAC ratio does not describe every finite length. No value here is measured work,
latency or power.

CPU checks, without TTNN import/device access:

```sh
python3 -B models/tt_dit/tests/unit/test_audio_c12_cpu.py -v
python3 -B models/tt_dit/tests/unit/test_audio_c12_torch_cpu.py \
  --upstream /home/smarton/ltx-rt/tt-metal/tt-project/state/runs/106/evidence/upstream-audio_pack.py
```

The first runs the actual analytic helper with a tensor-shaped standard-library shim,
independent scatter/impulse oracles, changed/restored inputs, zero/bias cases, odd/short
lengths, CPU-reference batch2, simulated2/4/8 shards, tails, wrong seams, cache/selector
and operation/ownership stubs. The second requires existing Torch and verifies the
pinned PR56922 impulse construction (SHA256 checked) and direct Torch convolution;
it fails rather than skips when Torch is missing. Neither is native FP32 validation.
The original run106 oracle and independent task53/run112 design audit are separate
artifacts; their passing results do not approve this implementation.

Before native acceptance, obtain independent committed-code review and close the Torch
and real checkpoint gaps. Existing task4/task9 own Galaxy/cs04 validation, task38 owns
combined measurement-adapter acceptance, and sulphur owns serving/deployment.
Keep the run29 profiling protocol and task49/run106 design gates:

1. Attest exact experimental commit/dirty state, native/data build IDs, ABI/library
   hashes, checkpoint and workload, cache root/JIT/capture state, broker/locks/pauses
   and non-disruptive access. Resolve or isolate C03 replay instability. No g15blx02
   device operation under its current prohibition.
2. Establish unmodified baseline and standalone A/B/C/D output correctness, all-chip
   halo/seam/global-edge and original-tail behavior, capacity with all simultaneous
   tensors/weights/CBs/traces, and at least32 changed/restored replay cycles.
3. Alternate separate explicit A/B/C/D processes and baseline repeats, at least three
   truly warm samples per cell/workload. Include slice/concat/copy/layout/halo/conv/
   trim/unpack/deallocation/completion synchronization in layer time, then AMP,
   complete audio and completed real request. Cold transforms/load/upload/JIT/capture
   are separate. Compare B-A for blocking, C-A and D-B for packing, plus interaction;
   require improvement beyond repeat noise and no representative request regression.
4. Preserve audio PSNR>=28dB, paired-baseline maxabs<5e-3, finite/nonflat48kHz output,
   sample/seam/spectral checks and the retained nine lengths. Keep full AV gates and
   Galaxy/cs04 real renders before any promotion. Unknown chip/operation/replay
   coverage or positive final-drain loss invalidates resource metrics. Power stays
   unmeasured without an actual meter.

This CPU task authorizes no device job, publication or deployment.
