# Local Wan 2.2 I2V server

Single-process FastAPI backend that serves Wan2.2-I2V-A14B on Tenstorrent
hardware. One pipeline is built at startup and shared by all requests.

## 1. Setup

```bash
cd /home/vkovinic/tt-metal
git checkout vkovinic/wan_demo
source python_env/bin/activate
```

## 2. Start the server

From the tt-metal repo root:

```bash
python -m server.launch
```

Startup takes **~30-60 s** (mesh open + pipeline warmup). **Wait for the banner**:

```
===========================
  SERVER IS READY
  backend=wan  pinned=1280x720  num_frames=81
===========================
```

Or poll:

```bash
curl -s http://127.0.0.1:8000/healthz      # /health is an alias
# {"status":"ok","backend":"wan","ready":true, ...}
```

Only send requests once `"ready": true`. Ctrl-C shuts down cleanly.

## 3. Configuration

Three things are **baked in at startup** and require a restart to change.
Edit the constants at the top of [server.py](server.py):

| Constant                | Default          | Effect                                            |
|-------------------------|------------------|---------------------------------------------------|
| `WAN_BACKEND_DEFAULT`   | `"wan"`          | `"wan"` (real) or `"stub"` (ffmpeg crossfade).    |
| `WAN_SERVER_CONFIG`     | `"bh_4x8sp1tp0"` | Mesh preset from `wan_i2v_core.CONFIGS`.          |
| `WAN_SERVER_RESOLUTION` | `"720p"`         | Inference resolution: `"480p"` or `"720p"`.       |
| `WAN_SERVER_NUM_FRAMES` | `81`             | Pinned frame count. Must satisfy `(n-1) % 4 == 0`. |
| `WAN_SERVER_USE_LOCK`   | `True`           | `flock` `/tmp/tt_device.lock` on startup.         |

Each can also be overridden by an env var of the same name for one-off runs:

```bash
WAN_SERVER_RESOLUTION=480p WAN_SERVER_NUM_FRAMES=121 python -m server.launch
```

Video duration = `WAN_SERVER_NUM_FRAMES / frames_per_second` (default 81/16 ≈ 5 s).
Valid frame counts: `1, 5, 9, 13, ..., 81, 85, 89, ..., 121`. Invalid values are
rounded UP to the next valid one and logged.

## 4. Making a request

`POST /predictions` is multipart. Only `prompt` and `image` are required.

### Minimal (first frame only)

```bash
curl -s -X POST http://127.0.0.1:8000/predictions \
  -F "prompt=slow push-in on a sunlit garden" \
  -F "image=@./start.png" \
  -o response.json
```

### First + last frame

```bash
curl -s -X POST http://127.0.0.1:8000/predictions \
  -F "prompt=two people run across the field and embrace" \
  -F "image=@./start.png" \
  -F "last_image=@./end.png" \
  -o response.json
```

Use two **different** images — same first/last collapses to a static clip.

### All parameters

Copy-paste ready — all fields optional except `prompt` and `image`:

```bash
curl -s -X POST http://127.0.0.1:8000/predictions \
  -F "prompt=..." \
  -F "image=@./start.png" \
  -F "last_image=@./end.png" \
  -F "negative_prompt=blurry, low quality" \
  -F "frames_per_second=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=3.5" \
  -F "guidance_scale_2=3.5" \
  -F "seed=42" \
  -F "resolution=720p" \
  -F "height=720" \
  -F "width=1280" \
  -F "num_frames=81" \
  -o response.json
```

**Per-request, no restart:**

| Field                 | Required | Default              | What it does |
|-----------------------|----------|----------------------|--------------|
| `prompt`              | yes      | —                    | Positive text describing the scene. Include an explicit motion cue ("the camera slowly dollies forward", "leaves rustle in the wind") — Wan tends toward static without one. |
| `image`               | yes      | —                    | First-frame conditioning image (PNG/JPEG, any aspect). The server resizes internally. |
| `last_image`          | no       | —                    | Last-frame conditioning image. |
| `negative_prompt`     | no       | Chinese Wan defaults | Things to suppress. Default is Wan's recommended negatives (blur, low quality, static, extra fingers, …). Override to add domain-specific suppressions. |
| `frames_per_second`   | no       | `16`                 | Playback rate of the output mp4. Does **not** re-run inference — only changes duration (and thus perceived motion speed). Wan is trained at 16; deviating looks fast-forward / slow-mo. |
| `num_inference_steps` | no       | `40`                 | Diffusion denoising steps. Higher = slower but potentially better adherence/quality. Useful range ≈30-50; Wan's reference recipe uses 40. |
| `guidance_scale`      | no       | `3.5`                | Classifier-free guidance (CFG) for the **high-noise expert** (first half of the schedule, responsible for overall composition and motion). Higher → stronger prompt adherence but can oversaturate / distort; lower → more natural motion but may ignore the prompt. Useful range **3.0–5.0**. |
| `guidance_scale_2`    | no       | `3.5`                | Same idea for the **low-noise expert** (second half, responsible for fine detail / late-stage refinement). Usually matches `guidance_scale`. Bump it higher (e.g. `4.0–4.5`) if late frames look blurry or drift off-prompt. |
| `seed`                | no       | `42`                 | Random seed. Same `prompt + image + seed` → identical output (handy for A/B comparing prompt tweaks). Change it to explore variations. |

**Tied to server-pinned constants (see §3):**

| Field               | Required | Default              | What happens if it differs from pinned |
|---------------------|----------|----------------------|----------------------------------------|
| `resolution`        | no       | server pinned        | `"480p"` or `"720p"`. Inference always runs at `WAN_SERVER_RESOLUTION`; a different request value triggers an in-process PIL resize of the output frames. No restart needed. |
| `height`, `width`   | no       | from `resolution`    | Explicit dimensions. Overrides `resolution` when both are set. Same resize rules as above. |
| `num_frames`        | no       | server pinned        | Frame count. Must satisfy `(n-1) % 4 == 0`; server rounds UP automatically. If the rounded value **≠** `WAN_SERVER_NUM_FRAMES` → **HTTP 400**, and a **restart** is required with a new `WAN_SERVER_NUM_FRAMES`. |

## 5. Getting the video

Response shape on success:

```json
{
  "id": "pred_abcd1234",
  "status": "succeeded",
  "output": "http://127.0.0.1:8000/files/pred_abcd1234.mp4",
  "urls": {"get": "http://127.0.0.1:8000/predictions/pred_abcd1234"},
  "input": { "...": "echoed back" },
  "created_at": "...",
  "completed_at": "..."
}
```

Download the mp4:

```bash
curl -s -o out.mp4 "$(jq -r .output response.json)"
# play:   ffplay out.mp4
# scp:    scp out.mp4 user@laptop:~/
```

No `jq`? Paste the URL from `response.json` manually into a second `curl -o`.

Failure still returns HTTP 200, with `"status":"failed"` and an `"error"` string.

## 6. Troubleshooting

- **`"ready":false` in healthz**: warmup not done yet. Wait for the banner.
- **HTTP 400 `num_frames=... != server pinned ...`**: frame count is fixed
  per process. Restart with a different `WAN_SERVER_NUM_FRAMES`.
- **Connection hangs / drops mid-request**: a client-side timeout, not the
  server. Use `curl --max-time 600` or equivalent for 720p/81-frame runs.
- **`ImportError: wan_i2v_core`**: `python -m server.launch` must run from
  the tt-metal repo root (where `wan_i2v_core.py` lives).
