# Independent LTX audio replay candidate

This CPU-tested candidate decouples audio trace requests from DiT tracing. It is
opt-in for an eager pipeline. It does not establish native capture, memory fit,
quality, latency, power savings, or deployment readiness.

## Configuration and compatibility

`LTXPipeline(..., traced=False, audio_traced=True)` requests independent audio
replay. The shared factory forwards `audio_traced` through its pipeline keyword
arguments, including the LTX2.3 and LTX2.5 subclasses. Without an explicit API
argument, `LTX_AUDIO_TRACE=1` opts in and `LTX_AUDIO_TRACE=0` opts out. Without either,
audio follows `traced` as before. The API argument takes precedence over the
environment. Invalid environment booleans raise. The existing `LTX_VAE_TRACE`,
`LTX_VOC_TRACE` and `LTX_BWE_TRACE` per-stage opt-outs remain effective.

An eager DiT stays eager; independent audio does not force whole-pipeline warmup.
The first independent decode performs one eager audio warmup before capturing the
selected stages. Existing fully traced warmup and default stage selection remain
unchanged. The checkpoint parsing, prepared-weight precision, host bridges,
trimming, and existing `LTX_AUDIO_DEVICE_CHAIN` selection are unchanged. In chain
mode, either vocoder trace flag enables the single owning chain trace, as before.
The legacy fully traced path retains its existing capacity policy.

Independent mode requires `dynamic_load=False` and a final video decoder that does
not require exclusive residency. This is checked after subclass module
construction, including a possible LTX2.5 decoder replacement. Paging or replacing
weights under live captures can invalidate recorded buffer addresses. A weight
reload through the audio adapter releases captures before loading new weights.
Checkpoint, precision, partitioning and module configuration remain fixed for the
pipeline lifetime; construct a new pipeline to change them.

## Bounded capacity contract for sulphur

The owner must provision the **existing parent mesh** with a positive
`trace_region_size=B` when opening it. If sulphur uses `LTX_TRACE_REGION`, it must
verify that the launch path forwards that value into this device-open argument.
This candidate neither opens a device nor changes a reservation. Setting only
`LTX_AUDIO_TRACE=1` while the region remains zero is rejected before decoder
construction. In this source, zero can select dynamic top-down DRAM allocation
(`tt_metal/distributed/mesh_trace.cpp`); that mode is intentionally excluded for
independent replay.

`B` is the total trace budget across DRAM banks **per device**, not a per-bank or
whole-Galaxy budget. The allocator rounds per-bank reservations to trace-page
alignment, so the observed `reservation_bytes` can exceed the requested strict
native cap. The adapter verifies a positive TRACE allocator region and checks for
remaining contiguous capacity before warmup. These checks do not prove that all
captures fit. Native trace-size/allocation checks still enforce the bound; any
capture or execution error propagates, with partial independent audio captures
released, including on cancellation (KeyboardInterrupt, SystemExit or GeneratorExit).
Cancellation is re-raised after cleanup. There is no silent eager fallback and no automatic increase of `B`.

No defensible numeric `B` is established by CPU mocks. Before admission sulphur
must select and record a finite cap from actual trace sizes for the exact audio
length, checkpoint, topology and selected stages, including any other resident
traces. Verify every participating device's reserved/free/contiguous bytes and
DRAM/L1 high-water alongside all resident weights, I/O, intermediates and CCL
buffers. Confirm page alignment and safety headroom under the longest accepted
job. A zero region, unavailable memory-view API, exhaustion, or resident-weight
incompatibility must fail the opt-in validation rather than be interpreted as
replay. Disabling the candidate restores the previous eager policy.

## Capture identity, ownership and observability

The adapter permits one live independent input/configuration family: spatial
shape, dtype, stage flags and chain selection. Changing any while captured raises
before warming or capturing another family. Call `pipeline.release_traces()`
before changing shape or re-warming; it also clears the adapter's warm identity.
Repeated identical inputs/configuration reuse the existing stage trace keys.
Existing C12 experiment keys are neither removed nor replaced by this guard.
Host bridges remain owned host tensors, avoiding borrowed outputs between traces.
No donor packing, fusion or padding changes are included.

`Audio trace status` logs contain the requested policy, observed reservation, and
per-stage `enabled`, `captured`, `captures`, and `replays`. `captured` counts live
traces, `captures` counts successfully recorded captures, and `replays` counts
subsequent execute calls that returned successfully (audio uses blocking execution).
The initial capture's execution is not a replay. Capture-only kernel prewarm and
eager calls increment neither counter. Released tracers retain lifetime counters;
compare deltas and live counts. A failed first execution may follow a successful
capture, so a capture count alone is not execution evidence. Exceptions release
partial independent captures and their held input/output references. These
counters are control-flow evidence, not output correctness or hardware timing.
If all stage flags opt out, `requested` remains true but every stage stays eager
with zero captures/replays. A retry after failed capture eagerly warms again;
its first successful capture still has zero replays.

## CPU regression checks

Run directly without device-aware conftest imports:

```sh
python3 models/tt_dit/tests/unit/test_ltx_audio_trace_control.py
python3 models/tt_dit/tests/unit/test_ltx_audio_submesh_lifecycle.py
python3 models/tt_dit/tests/unit/test_ltx_euler_tail.py  # requires CPU PyTorch
```

The focused test executes production adapter, pipeline and Tracer methods with
native/checkpoint shells replaced. It covers the original propagation failure,
enablement/disablement, API/environment precedence, zero/unknown/exhausted
capacity, legacy fully traced/eager operation, stage opt-outs, warmup, capture
reuse, owned host results, shape/dtype/mode rejection, chain ownership, weight
reload, release/recapture, capture/prewarm/replay counters and partial failures.
The fake does not model allocator aliasing, native address reuse, CCL scheduling,
real checkpoint construction, numerical kernels or asynchronous device failures.

## Admitted-device acceptance plan (not performed by this task)

1. **Resources and owners:** sulphur coordinates an existing broker/tt-device
   admission on authorized Galaxy blx04 after operational admission and compatible
   build attestation. cs04 supplies the required eight-chip girl-singing real
   render. blx06 remains unavailable under user hold11; blx01, blx03 and
   g15blx02 are prohibited. Preserve locks, STOP, non-disruptive
   access, attested compatible builds and all pause/recovery restrictions.
2. Bind receipts to the exact candidate commit, native build/loaded-library
   hashes, source slot, audio checkpoint content/config and prepared-weight
   identity, LTX2.3/LTX2.5 variant, real convolutional-video-VAE checkpoint/class
   and executed decoder path. Record mesh/submesh, precision, trace flags, `B`,
   lengths, FPS, prompt, seed and cache/compile state. Source defaults alone are
   insufficient. Coordinate application solely through task73/sulphur.
3. Use unchanged-source eager control and independent-audio candidate with the
   same finite reservation to isolate replay. Also report the total resource
   cost of adding the reservation versus the existing zero-reservation launch.
   Preserve fully traced controls and stage opt-outs. Log before/after status:
   first real capture, at least two warm replays, release and recapture. Run
   repeated A/B/restored-A and at least 32 replay cycles; include short, longest
   supported and boundary audio lengths. Test shape/configuration rejection,
   capacity failure, cleanup and parent/submesh shutdown without skip paths.
4. Compare identical-latent eager/replay waveforms and existing deterministic
   exactness contracts, CPU waveform PSNR >=28 dB, output duration/trimming and
   video/audio quality in real prompts. Preserve unresolved C03 and historical
   C07 controls; passing a new short replay does not resolve the old defect.
   Require the Galaxy and cs04 girl-singing end-to-end acceptance on exact heads.
5. Measure synchronized latent-in to waveform-out wall time including host
   bridges, transfers, synchronization and trim. Separately report cold compile,
   eager warmup, capture and warm replay; the existing stage timer starts after
   preparation and is not inclusive cold latency. Include complete real-prompt
   render latency, queue/submit time separately, compile misses, trace bytes,
   DRAM/L1/host high-water, transfers and utilization. Use segmented profiler
   capture when needed. Power remains unknown without an actual meter.

Retain task296/82 qualification, original 8+67 mandatory checks, task38/274
combined candidate acceptance, task89 actual zero-skip lifecycle qualification,
independent review and task73-only application. These candidate gates do not add
requirements to task9's separately authorized unchanged-source reference path.
Sulphur owns serving/deployment; this candidate authorizes neither publication to
`origin/ltx-rt` nor push-queue approval. A later PR remains draft until independent
review and explicit user approval; no auto-merge is authorized.

## Scope reconciliation — November 6, 2026, 18:09 PST

The retained LTX candidate is based on a103d2ae08a7dbdb2038b180b38aa5b8bf6fda85;
this is an experimental source identity, not a serving/build attestation. Task387's
completed audit remains the source for donor applicability; no fusion, packing,
padding or prepared-weight precision changes are included here.

At this source, MiniMaxH3Pipeline._decode_audio calls audio_decoder(latents)
without the supported traced argument. H3 replay, including the requested BWE
and vocoder coverage, is not implemented or accepted by this LTX candidate.
FastH3/H3 Turbo coverage and canonical descriptor/session-epoch integration remain
unverified. The separately reviewed leased_yuv_v1 design is also unimplemented;
none of its eight acceptance cells is claimed complete. Preserve these explicit
remaining deliverables when splitting the next implementation part.

Before native acceptance of H3 t2va/ref2va, the existing validation owner must bind
the exact 4x8 Blackhole source/build/checkpoint and executed decoder stages, prove
first capture and subsequent replay (including BWE/vocoder where applicable),
and verify target-row exclusion, waveform correctness under a verified checker,
release/recapture, bounded reservation, and inclusive latency/memory. Eager warmup
and source inspection are not replay evidence. No device validation ran here.
