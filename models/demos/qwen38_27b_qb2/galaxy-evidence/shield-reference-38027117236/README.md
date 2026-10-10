# Shield Qwen reference audit, October 10, 2026

The useful new transfers are long-prompt prefill traces, fused prefill
communication/MLP, scheduler admission, and cheaper recurrent-slot remapping.
No matched result in this run demonstrates 30 native TSU at B16/32K with our
BFP8 precision policy. No running source, serving policy, or hardware queue
was changed during this audit.

## Provenance and comparison

- [Shield run](https://github.com/tenstorrent/tt-shield/actions/runs/38027117236):
  `5815afe0e2186c11798d7c1a26986cee23c36b3f`.
- The linked job `114140207683` validates dispatch only. Hardware benchmark job
  `114140267161` ran on `g11blx02`; the run was **cancelled** and its report is
  explicitly partial (22 benchmark cells). This is not an eval pass.
- Metal: `e6b4fe334de4b008ca2cd800f2b26130417b3aa0`, branch
  `atupe/qwen38-optimizations-main-merge`, implementation
  `models/demos/blackhole/qwen36`.
- Inference server: `a44268316063ebf0d25a92f70d769965e17e5942`, branch
  `atupe/qwen38-p150x8-qwen36-impl`. SHA verified in checkout log.
- vLLM report revision: `c62035d`. Image tag is retained in
  `reference-config.json`; an image digest was not established by this audit.
- Reference: eight chips, one TP8 replica, max 32 requests, 262144 context,
  525312-token KV pool/admission budget, 64-token pages, 1-GiB trace region.
- Actual server logs confirm BFP4 MLP gate/up, BFP8 MLP down/GDN input/attention,
  and MLP LoFi. Our selected policy is BFP8 weights/KV and HiFi2 projections.

| 32K input / 128 output comparison | Reference | Our current fusion candidate |
|---|---:|---:|
| Chips and concurrency | TP8, C15 | TP4, B16 |
| Measurement boundary | HTTP client | Native model |
| Decode tokens/s/user | 16.844 | 16.550 |
| Step / mean client TPOT | 59.370 ms | 60.422 ms |
| TTFT | 68.877 s | 98.631 s |
| All-in output tokens/s | 25.122 | 19.265 |

Reference TSU here is `1000 / mean_tpot_ms`; it is not a separately measured
device step or the mean of per-request reciprocal TPOT. Context, precision,
chip count, concurrency and timing boundary differ. No normalized speedup is
claimed. Report input throughput includes decode and queueing; it is not an
isolated prefill kernel rate. Our measurements come from the completed
`../roofline-reconciliation-v1/candidate-sweep.json` and are not yet an eval
qualification of the new fusion. Do not multiply either row into a measured
full-Galaxy result.

## Transfers ranked for our path

### 1. Trace long-prompt prefill with a device-side chunk offset

Reference `tt/model.py:3536` replays a 2048-token trace, updates token/page/rope
buffers on the same command queue and synchronizes every eight chunks. It
retains host DMA source tensors through the final drain. Its attention path
uses `chunk_start_idx_tensor` instead of baking each offset into a program.

Our `tt/generator.py:390` submits long grouped prefill eagerly. Its short
single-slot trace is restricted to <=4096 tokens; the serving launcher requests
decode-only traces. This is a real missing coverage area. Our pinned native
checkout already exposes the device-offset API, so it need not start with a
native reinstall. Preserve slot-specific GDN/conv carry, page ownership,
continuation offsets, unaligned tails, and trace lifetime before enabling it.

Expected impact: remove repeated Python/dispatch overhead and per-offset
program specialization from long prefill. No honest percent gain is established
yet: measure eager versus traced B16/16K and B16/32K, separating host enqueue,
device work and first-token publication. This affects TTFT, not steady decode.

### 2. Fuse prefill all-gather, projection and SwiGLU

Reference `tt/tp_common.py:514,572` and `tt/mlp.py:266` use two-link
`all_gather_minimal_matmul_async`, including `fuse_swiglu=True` with tile-pair
interleaved gate/up weights. They tune M/K/N blocks for prefill and retain
persistent gather storage where required. Our selected prefill path gathers a
replicated normalization input and subsequently performs the projections and
gating; generic experimental AGMM code elsewhere is not proof this path uses
the same fusion.

The pinned native checkout already contains `fuse_swiglu`. Port the prefill
shape/layout contract at BFP8/HiFi2 and compare complete layer boundaries,
including communication and reshaping. Do not copy the reference's BFP4/LoFi
policy. Its extra prefill weight packing also consumes DRAM; recalculate KV
capacity before retaining a second packed copy.

Expected impact: fewer communication/activation round trips and intermediate
gate/up tensors across all 64 MLPs. End-to-end TTFT savings require a matched
prefill profile; the reference's aggregate TTFT is not an A/B of this fusion.

### 3. Fix scheduler admission separately from device chunk size

The [exact inference-server commit](https://github.com/tenstorrent/tt-inference-server/commit/a44268316063ebf0d25a92f70d769965e17e5942)
raises `max_num_batched_tokens` from 262144 to 525312. Its author reports a
15 x 32K burst previously split 1/8/6, where later synchronous prefills stalled
requests already decoding. Reported client TPOT changes from 190 to 59 ms and
long inter-token stalls disappear. This A/B is described in the commit; this
run independently retains the final 59.37-ms cell, not the prior 190-ms raw run.

Our `demo/galaxy_serving.py` and release launcher still set admission to 262144,
despite a 1050592-token per-replica KV pool. Sixteen 32K inputs alone are 524288
tokens. Test a pool-bounded admission budget with simultaneous and staggered
arrivals. Keep `QWEN_PREFILL_MAX_BATCH_TOKENS=32768` as the separate internal
device activation/chunk budget; raising admission does not mean materializing
a million-token activation. Check plugin host-buffer allocations before launch.

Expected impact: potentially large client TPOT/p95 improvements when this
admission split occurs, **zero improvement to isolated steady decode**. Longer
prefill admission can increase other requests' waiting time; measure TTFT and
stall distribution, not just average TPOT. No 3.2x device-speedup claim follows.

### 4. Coalesce recurrent-state row moves

Reference `tt/gdn/tp.py:1634` groups consecutive source rows into runs and moves
only the valid convolution representation. Our
`tt/generator.py:566` still slices every row of every conv/recurrent tensor and
releases traces on a nonidentity remap.

The reference log reports approximately 285-294 to 22-23 ms per full-model
condense/small swap, with bit-identical live state. Its arbitrary reverse case
has little benefit when both representations are live. These are reference
microbenchmarks, not measurements of our differently shaped state.

Expected impact: fewer eager operations at request turnover and better tail
latency. Port run coalescing first, retain permutation/alias safety and the
current trace-release rule. Removing recapture is a separate change requiring
bucket-state ownership and program/trace-lifetime validation. Fixed B16 decode
with no turnover does not improve from remap alone.

## Already covered or unsuitable for direct adoption

- Their handoff explicitly describes ports from `qwen38_27b_qb2`: replicated
  TP4 residuals, reduced collectives, fused GDN conv/recurrence/norm, compact
  attention and device-resident sampled-token/position feedback. Our current
  direct-preparation/epilogue candidate further changes this decode path.
- Full-grid SDPA and 128/128 prefill attention chunks are already our defaults.
  Their B1 attention/cache-fill loop is not a new batched attention solution.
- They reverted the multi-reader QKV experiment after end-to-end regressions;
  this reinforces measuring layout/padding costs with the matmul.
- The exact Metal tip disables replicated-residual all-reduce at TP>4 after
  a TP8 decode hang. Do not apply TP4 collective settings blindly to TP8.
- Their saved Qwen3.8 teacher-forced study reports 85.42% top-1 agreement over
  1536 positions for default mixed precision versus 95.31% with BFP8 gate/up
  and HiFi2. These are **not GPQA scores**; the handoff lists task evals as not
  run there. Reuse the quantized Torch reference methodology for diagnostics,
  not as permission to change our qualified precision.

## Next validation order

Finish the currently running fusion qualification. Keep compact GDN and
projection sweeps intact: they address the native 30-TSU objective. Add a
matched BFP8 prefill profile, then a long-prefill trace A/B and fused prefill
MLP A/B. Test scheduler admission and remapping under real request turnover
as separate serving experiments so host improvements are not mislabeled as
kernel throughput. No new reference-inspired hardware experiment was launched
or inserted ahead of the existing jobs by this audit.

Immutable source references:
[prefill implementation](https://github.com/tenstorrent/tt-metal/blob/e6b4fe334de4b008ca2cd800f2b26130417b3aa0/models/demos/blackhole/qwen36/tt/model.py#L3536),
[fused prefill MLP](https://github.com/tenstorrent/tt-metal/blob/e6b4fe334de4b008ca2cd800f2b26130417b3aa0/models/demos/blackhole/qwen36/tt/tp_common.py#L572),
[remap implementation](https://github.com/tenstorrent/tt-metal/blob/e6b4fe334de4b008ca2cd800f2b26130417b3aa0/models/demos/blackhole/qwen36/tt/gdn/tp.py#L1634),
[measured optimization log](https://github.com/tenstorrent/tt-metal/blob/e6b4fe334de4b008ca2cd800f2b26130417b3aa0/models/demos/blackhole/qwen36/QWEN38_OPTIMIZATION_LOG.md#L655).
