# LTX 2.5 Ingredients 0.9 + convolutional VAE

`fixture.example.json` is deliberately **unresolved**. It is not a real-weight
fixture and cannot load a pipeline. CPU tests use synthetic files and a pipeline
double; they establish source behavior only.

The public entry point is:

```python
from models.tt_dit.pipelines.ltx.pipeline_ltx25_iclora import LTX25ICLoRAPipeline

pipe = LTX25ICLoRAPipeline.create_pipeline(mesh_device, fixture_path="resolved.json", traced=True)
pipe.warmup_buffers()
frames, audio = pipe.generate_r2v("The bound subject walks through a garden", seed=10)
# Or export exactly once through the existing AV exporter:
path = pipe.generate_r2v("The bound subject walks through a garden", seed=11,
                         first_frame="previous-last.png", output_path="continuation.mp4")
```

Resolve every null, set `resolved` to true, and retain the manifest alongside the
source/build identity and outputs. Paths resolve relative to the manifest. All
five split model files and the reference still must already exist locally; this
consumer never downloads, converts or fuses weights. Each model binding needs its
actual SHA256, immutable source revision, `model_version: "2.5"` and declared kind.
The native 2.5 loader additionally checks that the video VAE contains conv decoder
blocks. A filename alone is insufficient.

The primary `transformer` must be an **already fused** Ingredients transformer.
`fusion.fused_sha256` must equal its verified content hash and differ from the
recorded base checkpoint hash. Record the base hash/revision and the existing
fusion tool/revision/recipe, as well as adapter scale. The supported adapter is
`Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`, revision
`5f83f6f7e5e6cd6eadfade02a57f03b9212d6434`, Ingredients 0.9 LFS SHA256
`ff873a5beada3c579a8137c7c53343916f78bcc9a529ba910143073fe8715e95`,
1,308,787,472 bytes. That public metadata does **not** establish local access,
correct fusion, numerical equivalence or usable identity conditioning. No real
fused artifact, split-model hashes or quality baseline is supplied here.

The session binds geometry, FPS, output/reference lengths, reference and pin
strengths, and a named quality variant. The reference is a single still sheet
repeated over the reference sequence; animated images/videos are rejected.
Endpoints must be still images at the bound output resolution. An R2V call always
retains the sheet, including with both endpoints. `mode: "fl2v"` requires both
endpoints on every call. Endpoints supplied to calls can change for continuation;
the initial acceptance endpoints are separately digest-bound in the manifest.
Do not modify an input file concurrently with a call.

Version 1 supports the existing full 8/3 sigma schedules and deterministic
`s1_ref`/`s2_ref` Euler behavior. The generic 2.5 ancestral override applies to
`s1` only; the wrapper explicitly disables it. The default full-reference example
is 768×512, 121 output/reference frames, 24 FPS; this is a reproducibility recipe,
not a quality or speed claim. Shorter reference lengths or alternate pin strengths
require a separately named, opt-in fixture. Unsupported schedules and inherited
recipe overrides fail before native loading. `environment` records the supported
baseline switches, including `LTX_REF_IMG0=s2` and still encoding. All inherited
`LTX_`/`LTX25_` variables must match that map; unset variables use the recorded
upstream defaults. No environment is silently changed. Changes after construction
also fail. Use a dedicated worker process with this environment. Already imported
native sigma/pin constants are checked too. Runtime keyword options are limited
to mesh/parallel/loading/tracing choices; they cannot replace fixture bindings.

Without `output_path`, the existing decoder returns `(frames, audio)`: default
RGB uint8 `(B,3,F,H,W)`, `output_type="float"` for float `[-1,1]`, or explicit
`"yuv"` for the existing planar decoder output. Audio retains the existing `Audio`
object and sample rate. With `output_path`, the existing AV export returns the
path. The wrapper adds no extra encode/export. All `num_frames` at the bound FPS
are retained, including the first pinned overlap frame; unique playable-video
accounting and overlap removal belong to the serving consumer.

Changed bytes at an existing endpoint path invalidate its two cached resolutions
**before** generation. After success only the active first/last endpoints remain
(at most four entries); failure/interruption removes continuation-owned endpoint
state. The fixed digest-keyed reference latent pair is left unchanged. This host
cache policy does **not** prove native memory release or safe fresh encoding after
trace replay. Existing source warns about that native hazard.

For owner-operated native acceptance, fill `acceptance` with permitted first/last
images, one fixed prompt, three seeds, pinned quality-reference recipe or baseline,
and explicit criteria decided before rendering. Under the existing device broker,
select only this consumer (normal project device fixture setup is required):

```sh
TT_ICLORA_FIXTURE=/absolute/resolved.json TT_ICLORA_OUTPUT=/absolute/retained-output \
  python3 -m pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx25_iclora.py -v
```

It retains three R2V AV outputs with the preceding decoded final frame feeding the
next call, then a separate both-endpoints FL2V output with the sheet retained.
Output media and per-call receipts are retained; native completion alone is not
quality acceptance. The owner must compare reference influence, identity,
continuity, sheet-copy cuts, blocks/noise and audio/playback to the pinned baseline.
Native asset absence is an explicit missing prerequisite, never a passing skip.
A fresh-encode failure remains an acceptance gap; it does not authorize shared
trace/audio repairs or any stopped workstream.

Performance and sustained-streaming qualification requires TYO2 with enforced
160 W read back and recorded for **every participating chip**, about one minute
of offered load plus bounded drain, unique caller-playable video with overlap,
failures/retries/admission/queue/export/delivery accounted for, and identity,
continuity, quality, clocks/throttling, memory and playback evidence. This CPU
change and native correctness consumer establish none of those claims.
