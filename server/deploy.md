# Deploying `server.py` on a Wan-capable machine

This guide walks through running the local Wan 2.2 I2V HTTP backend (`server.py`)
on a Tenstorrent TT-QuietBox (or any machine where `tenstorrent/tt-metal`'s
`WanPipelineI2V` can execute), and hitting it from a client machine with
`generate_clips.py --backend local`.

## 1. Prerequisites on the server machine

- A working `tenstorrent/tt-metal` checkout and a Python environment where you
  can already run the Wan I2V test:
  `models/tt_dit/tests/models/wan2_2/test_pipeline_wan_i2v.py`.
- `ffmpeg` and `ffprobe` on `PATH` (used both by the stub and for encoding
  the final mp4 from Wan's output).
- Python 3.10+.

## 2. Copy the two files over

Only two artifacts need to move to the Wan machine:

```bash
# On the client, from the repo root:
scp server.py requirements-server.txt user@wan-host:/opt/wan-server/
```

(Path is arbitrary — pick anything writable. The server creates `inputs/` and
`outputs/` relative to its working directory.)

## 3. Install server dependencies

```bash
# On the Wan machine:
cd /opt/wan-server
python3 -m pip install -r requirements-server.txt
```

This installs `fastapi`, `uvicorn[standard]`, and `python-multipart`. These
are independent of the tt-metal environment — install them into the same
virtualenv your Wan pipeline runs in so one `python3` sees both.

## 4. Wire in the real Wan pipeline

`server.py` ships with a stub `run_wan(...)` that runs an ffmpeg crossfade
when `WAN_BACKEND=stub` (the default) and raises `NotImplementedError`
otherwise. On the Wan machine you replace that second branch with an actual
`WanPipelineI2V` call.

Open `server.py` and find `run_wan(...)`. Add a `wan` branch:

```python
# Module-level (top of server.py), near the other imports:
from pathlib import Path
import PIL.Image

_PIPELINE = None  # lazy-initialized on first call

def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
        # Mesh / sharding args depend on your hardware. Copy these from the
        # values that work in test_pipeline_wan_i2v.py on this machine.
        _PIPELINE = WanPipelineI2V.create_pipeline(
            mesh_device=...,
            sp_axis=...,
            tp_axis=...,
            num_links=...,
            dynamic_load=...,
            topology=...,
            is_fsdp=...,
            target_height=480,
            target_width=832,
            num_frames=81,
        )
    return _PIPELINE


def _run_wan_real(prompt, negative_prompt, first_image_path, last_image_path,
                  num_frames, fps, height, width,
                  num_inference_steps, guidance_scale, guidance_scale_2,
                  seed, out_path):
    from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt

    pipeline = _get_pipeline()

    image_prompt = [ImagePrompt(image=PIL.Image.open(first_image_path), frame_pos=0)]
    if last_image_path is not None:
        image_prompt.append(
            ImagePrompt(image=PIL.Image.open(last_image_path), frame_pos=num_frames - 1)
        )

    kwargs = dict(
        prompt=prompt,
        image_prompt=image_prompt,
        negative_prompt=negative_prompt,
        height=height, width=width, num_frames=num_frames,
        seed=seed,
    )
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        kwargs["guidance_scale"] = guidance_scale
    if guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = guidance_scale_2

    result = pipeline(**kwargs)

    # `result` is a list/tensor of frames. Encode to mp4 at the requested fps.
    # Adapt this to whatever shape the pipeline returns — typical path:
    #   frames = result.frames[0]  # list[PIL.Image]
    #   use ffmpeg -r {fps} -i frame_%04d.png -vframes {num_frames} ... out.mp4
    _encode_frames_to_mp4(result, num_frames, fps, out_path)
    return out_path
```

Then in `run_wan(...)` replace the existing `raise NotImplementedError(...)`
line with:

```python
    if WAN_BACKEND == "wan":
        return _run_wan_real(
            prompt=prompt, negative_prompt=negative_prompt,
            first_image_path=first_image_path, last_image_path=last_image_path,
            num_frames=num_frames, fps=fps, height=height, width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, guidance_scale_2=guidance_scale_2,
            seed=seed, out_path=out_path,
        )
    raise NotImplementedError(f"backend '{WAN_BACKEND}' not wired up")
```

The mesh-device setup and the exact `result → mp4` encoding step depend on
the specific hardware and the shape `WanPipelineI2V.__call__` returns — copy
those bits from the working `test_pipeline_wan_i2v.py` on your machine.

## 5. Open the port in the firewall

Pick a port (8000 is a fine default). Open it for inbound TCP from the
client IP (or the whole LAN if that's acceptable).

**Linux (ufw):**
```bash
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
sudo ufw status
```

**Linux (iptables, no ufw):**
```bash
sudo iptables -A INPUT -p tcp --dport 8000 -s 192.168.1.0/24 -j ACCEPT
# Persist per your distro (iptables-save / netfilter-persistent)
```

**Cloud / managed host:** add an inbound TCP rule for port 8000 from the
client's IP range in the provider's security-group UI.

If the server sits behind a second NAT (home router), forward port 8000 to
the Wan machine's LAN IP in the router admin.

Leave port 8000 bound to `0.0.0.0` in uvicorn (step 6) so it accepts
connections on all interfaces — the firewall rule above is what gates who
can reach it.

## 6. Start the server

```bash
cd /opt/wan-server
export WAN_BACKEND=wan
uvicorn server:app --host 0.0.0.0 --port 8000
```

Expected log line: `Uvicorn running on http://0.0.0.0:8000`. Leave it
running (or wrap in `systemd`/`tmux` if you want it to survive logout).

Health check from the Wan machine itself:
```bash
curl -s http://127.0.0.1:8000/healthz
# {"status":"ok","backend":"wan"}
```

If the response says `"backend":"stub"`, you forgot `WAN_BACKEND=wan` in
the environment of the uvicorn process — stop, export, and restart.

## 7. Verify from the client machine

```bash
# Replace 192.168.1.42 with the Wan machine's LAN IP:
curl -s http://192.168.1.42:8000/healthz
# {"status":"ok","backend":"wan"}
```

If this hangs or times out: the firewall rule isn't right, or uvicorn is
bound to `127.0.0.1` instead of `0.0.0.0`.

## 8. Run generate_clips.py against it

On the client (from the repo root):

```bash
python3 generate_clips.py \
    --scenario scenario.md \
    --backend local \
    --local-server http://192.168.1.42:8000 \
    --shot 5
```

This will POST the shot's first/last keyframes + prompt to
`/predictions`, block until the Wan pipeline returns (can take minutes
— the server's request handler is synchronous), then download the mp4
to `clips/clip_05-06.mp4`.

To run the whole scenario, drop `--shot 5`. To override resolution:
`--resolution 720p`.

## 9. Troubleshooting

- **`{"backend":"stub"}` in healthz on the Wan machine**: `WAN_BACKEND`
  not exported before `uvicorn`.
- **Connection refused from the client**: uvicorn is bound to
  `127.0.0.1`. Restart with `--host 0.0.0.0`.
- **Connection times out**: firewall. Re-check the ufw / iptables /
  cloud security-group rule.
- **`ImportError: models.tt_dit.pipelines.wan...`**: the Python path
  that `uvicorn` sees doesn't include the tt-metal repo. Run uvicorn
  from the tt-metal root (so `models/` is importable), or set
  `PYTHONPATH=/path/to/tt-metal` before starting uvicorn.
- **Mesh-device init fails**: verify `test_pipeline_wan_i2v.py` still
  passes on this machine with the same mesh args.
- **Client times out during poll/download**: the server is synchronous;
  `generate_clip_local` uses `timeout=3600` for the POST and
  `timeout=120` for the download. If a single Wan run exceeds an hour,
  bump the POST timeout in `generate_clips.py:generate_clip_local`.

## 10. What the contract looks like

Nothing else changes on the client side. The wire format the server
speaks (under `POST /predictions`) is documented in
`docs/superpowers/specs/2026-04-17-local-wan-server-design.md` §
"API contract".
