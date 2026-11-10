# Experimental H3 reference-feedback controller

`streaming.Session` is a synchronous controller for one H3 reference-feedback chain.
It is not a validated infinite streamer. No realtime, quality, cache-bounding or
device-utilization result is implied by the CPU contract tests.

`streaming_pipeline.PipelineRenderer` wraps an already loaded `ref2va` pipeline
with the base/Turbo inherited call interface. The caller owns device admission,
pipeline initialization, profiler callbacks, job timeouts and mesh teardown.
Use it only through an authorized broker or owner-approved per-box device lock.
The reviewed bridge was developed against H3 source
`15d5b0028220bb994f6d3652fd9948bea04e34c6`. On integration target
`522ef26916bdd78e2446fd1ff19a4f33e40ecfb6`, the base ref2va call, reference
and output interfaces match. That target lacks the Turbo/HyperFlow runtime and
adapter loader. The optional speed routes remain unavailable there until a
separately reviewed runtime integration supplies them; their mock contract tests
do not establish runtime availability. Use the actual loaded source/build
identity, never the original source receipt for this different target.

The base route remains 50 grid points. Explicit Turbo4 and HyperFlow8 routes use
5 and 9 points, validate adapter bytes and schedule-family/shift settings, and
retain distinct complete source/build/config/checkpoint/adapter identities.
The bridge requires float output and does not silently convert the serving YUV
path. This is a separate experimental configuration; it changes no model defaults.
Checkpoint/build/config identities are owner attestations, not independently
verified by the controller; include the actual loaded artifact receipts.

One call to `step(prompt, seed)` produces one fresh chunk, appending at most one
copied video/audio context reference to the original ordered references. The
caller must refer to that last reference appropriately in the prompt when needed.
Reference feedback is not temporal prefix pinning. This bridge reports zero
overlap; the controller only trims a backend-declared repeated prefix after an
exact video **and** audio comparison with the previous tail. Seam smoothness,
reference adherence and long-run quality drift still need natural-media evidence.
No cross-request latent tensor or decoder cache is retained by this controller.

The sink receives a `Chunk` containing owned media, integer video/audio PTS,
their rational time bases, trim count and model identity. It must encode/mux
using these timestamps, transport the output, and wait until the client can
decode it before returning `Acknowledgement(index, frames, samples, played_frames)`.
Writing an MP4 alone is not client acknowledgement. The sink must release its
input buffers before returning and bound any buffers it owns. A failed or partial
delivery closes the chain; retry requires explicit caller recovery, not reuse of
potentially emitted timestamps. A receipt callback must append/fsync metadata
rows and must not retain media objects.

`streaming_encode.NutEncoder` supplies an optional CPU export stage. It requires
an explicit `pixel_format="rgb8"` choice and positive frame, width, height, sample
and encoded-byte limits. Each call creates and closes its own NUT muxer, exports
FFV1 RGB8 and float32 stereo PCM, and returns immutable bytes with a SHA256 and
the chunk's frame/sample counts and global PTS. RGB8 conversion rounds the
decoded floats; FFV1 preserves those quantized pixels, not the original floats.
Invalid/nonfinite input is rejected without clipping or padding. NUT is for an
FFmpeg/PyAV-capable client, not browser-native playback. No model/export default
changes and no speed or quality equivalence is asserted.

The byte cap applies before mux writes or seeks; live export storage is bounded
by a decoded chunk, per-frame conversion/codec working memory and at most two
capped byte buffers. A cancelled, over-limit or failed export returns no media.
Cancellation and deadlines are checked between frames and after mux close; the
caller must still impose a process/job timeout for native codec hangs. The
encoder retains no cross-call codec state. The bounded transport protocol below
supplies acknowledgement and retry deduplication contracts; the caller still
provides authenticated wire transport and client playback. Do not pass
`EncodedChunk` as a `Session` acknowledgement. Record its
`encode_s` within the existing request-to-client-ready span, not as additional
wall time. Real encoder tests use PyAV 16.0.1; their inputs are synthetic and
provide no generated-video acceptance evidence.

No render runs concurrently with a sink. Admission also reserves a whole chunk
against `max_buffer_s` using the client's confirmed playhead. When `Backpressure`
is raised before admission, wait/poll outside the device allocation as appropriate
for the serving owner, get a fresh client playhead, call `update_playout`, and
retry the same unadmitted request. Advancing wall time alone never frees capacity:
a paused consumer cannot produce an unlimited queue. The reported ideal playout
buffer/underruns are a model; actual client underrun telemetry is still required.

The request timer includes rendering, context copy, sink encoding, transport and
acknowledgement. `elapsed_rtf` also includes gaps between requests, including
intentional pacing; `service_rtf` sums admitted service time only. They are
different metrics. Bounded 1x playback is expected to approach elapsed RTF 1;
report unpaced capacity/headroom separately without hiding latency or growing
an unbounded playback queue. Startup is reported separately in the first row.
Per-row data supports p50/p95/p99/max analysis outside the reservation.
Select `measurement_mode="capacity"` explicitly for an unpaced client that
decodes and then discards output; its acknowledgement counts decoded/discarded
frames in the playhead field. Never mix that mode with paced playback evidence.

Rows carry the existing receipt's chunk/session identity, `request_s`, `ready_s`,
`media_start_s`, `media_end_s`, and emitted/discarded frame fields. Missing actual
stage, memory, quality and continuity measurements are deliberately null/false;
`measurement_kind="control_plane_incomplete"` prevents this controller alone
from qualifying as a complete client-stream receipt. The owner measurement
producer must enrich those fields from real instrumentation, add the run/input
manifest and client playback telemetry, and preserve the explicit paced/capacity
distinction. Do not fill missing measurements with zero or infer quality from
successful transport.

Controller-owned state is limited to fixed references, a copied context tail,
and a bounded number of full-chunk buffers while a step runs. Real RSS/device
high-water, pipeline trace/reference cache key counts, export queues, clocks,
stage profiles and energy are not inferred from this structural bound.

Run the lightweight controls without importing torch/ttnn or opening devices:

```sh
python3 models/tt_dit/tests/models/minimax_h3/test_streaming_control.py -v
python3 models/tt_dit/tests/models/minimax_h3/test_streaming_pipeline_contract.py -v
```

The lightweight bridge tests stub media tensors. `test_streaming_tensors.py`
exercises actual CPU Torch layout, detached context copies, exact video/audio
overlap, decoded-reference preparation and 100 synthetic tensor chunks. It never
loads a model or claims generated-video continuity, device cache bounds or speed.
Run that test on a validated remote CPU host, followed by a device matched fixture
A/B, before use. Then measure at least 60 chunks and 300 new playable seconds
and the owner's 30-minute playback window per claimed variant, adapting both to
concurrent production. Qualification requires sustained unique playable video
received by the API caller / elapsed wall time >1 across participating galaxies,
including queue, generation, export and delivery, excluding overlap and padding.
Per-clip latency, startup buffer, concurrency and participating galaxies are
reported separately, as are unpaced capacity and deliberately paced playback.
Keep continuity, bounded memory, quality and zero underruns after the declared
buffer. All four H3/LTX 2.5 IC-LoRA FL2V/R2V cases require TYO2 at enforced
160 W per participating chip, each limit read back and recorded, with host/mesh,
actual AICLK and throttling. CPU tests qualify none of them. The former per-clip
RTF <=0.8 target is historical, not a current gate. Combined-source review,
validated delivery to ltx-rt and the sulphur-mac integration receipt remain open.
# Bounded transport protocol

`streaming_transport.TransportSink` wraps the opt-in encoder and can be passed
directly as `Session`'s sink. Use the same unique `StreamConfig.session_id` for the
sink and its `ReadyClient`. Supply an authenticated wire adapter
`exchange(delivery, deadline)` that sends the immutable delivery and returns the
client's `ReadyReceipt`. Bound wire framing before allocation and enforce the
absolute monotonic deadline; the enclosing process/job timeout remains required.
This module supplies the protocol endpoints, not a network server or player.

The client `accept(encoded)` callback must decode and commit playable media before
returning its actual played-frame count. Keep the client endpoint alive across
connection retries. The sender encodes once and retransmits the same bytes on
connection/ack loss; the client caches only its last receipt and never appends
that sequence twice. A conflicting/stale receipt, deadline, retry exhaustion,
abort or ambiguous partial append closes the affected endpoint. Restart with a
fresh session identity; never re-render or resume a failed sequence implicitly.

Only one capped encoded payload is in flight. Set the encoder's `max_bytes` no
higher than the sink/client caps; bound decoder/playout memory in the client.
Feed later playhead events to `Session.update_playout`, which gates generation.
Abort the session and endpoints together on shutdown. The existing Session rows
include export/transport/retry time through acknowledgement and retain
`control_plane_incomplete`: codec/protocol tests are not network measurements or
proof of temporal continuity, quality or faster-than-realtime generation.
