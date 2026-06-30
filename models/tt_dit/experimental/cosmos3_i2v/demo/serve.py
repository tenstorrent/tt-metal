# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Persistent HTTP server for interactive Cosmos3-I2V generation.

Builds the pipeline once at startup (paying the ~20s weights+build cost only
once), then exposes a tiny web UI:

    GET  /            → HTML form: prompt textarea + Generate button
    POST /generate    → run pipe(); redirect to result page
    GET  /videos/<id> → serve the generated MP4
    GET  /history     → JSON of past runs

Device is single-tenant; requests are serialized with an internal lock.

Launch via ttq so the broker holds the device lock for the server's lifetime:

    ttq -H g03blx04 run-bg "cd ~/tt-metal && \\
      TT_DIT_VAE_DUPUP_CHUNKS=2 TT_DIT_CACHE_DIR=~/.tt-dit-cache \\
      TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
      python -m models.tt_dit.experimental.cosmos3_i2v.demo.serve \\
        --image ~/ref.jpg --port 8080 --mesh-shape 4x8 \\
        --pipeline native-cfg --vae-encoder-t-chunk-size 4"

From your Mac, port-forward and browse:

    ssh -N -L 8080:localhost:8080 g03blx04
    open http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cosmos3-I2V interactive HTTP generator")
    p.add_argument("--image", required=True, type=Path, help="Reference image (JPEG/PNG).")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--out-dir", type=Path, default=Path.home() / "cosmos3_serve")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1072)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--mesh-shape", default="auto")
    p.add_argument("--weight-dtype", default="auto", choices=["auto", "bfloat16", "bfloat8_b"])
    p.add_argument("--pipeline", default="native-cfg", choices=["native", "native-cfg"])
    p.add_argument("--num-links", type=int, default=None)
    p.add_argument("--flow-shift", type=float, default=10.0)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--vae-encoder-t-chunk-size", type=int, default=4)
    p.add_argument("--vae-decoder-t-chunk-size", type=int, default=None)
    p.add_argument("--cfg-serial-dispatch", action="store_true")
    return p.parse_args(argv)


def _resolve_mesh_shape(spec: str, available: int) -> tuple[int, int]:
    if spec == "auto":
        if available >= 32:
            return (4, 8)
        if available >= 8:
            return (1, 8)
        if available >= 4:
            return (1, 4)
        if available >= 2:
            return (1, 2)
        return (1, 1)
    r, c = (int(x) for x in spec.lower().split("x"))
    return (r, c)


def _resolve_weight_dtype(spec: str, mesh_shape: tuple[int, int]):
    import ttnn

    if spec == "auto":
        return ttnn.bfloat8_b if mesh_shape == (1, 8) else ttnn.bfloat16
    if spec == "bfloat16":
        return ttnn.bfloat16
    return ttnn.bfloat8_b


class State:
    def __init__(self, args, mesh, pipe, ref_image) -> None:
        self.args = args
        self.mesh = mesh
        self.pipe = pipe
        self.ref_image = ref_image
        self.lock = threading.Lock()
        self.history: list[dict] = []

    def generate(self, prompt: str, seed: int | None = None) -> dict:
        import torch
        from diffusers.utils import export_to_video

        from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_omni_native import _make_release_callback

        run_id = uuid.uuid4().hex[:12]
        out_path = self.args.out_dir / f"{run_id}.mp4"

        with self.lock:
            t_wall0 = time.time()
            generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
            kwargs = dict(
                image=self.ref_image,
                prompt=prompt,
                num_frames=self.args.frames,
                height=self.args.height,
                width=self.args.width,
                num_inference_steps=self.args.steps,
                guidance_scale=self.args.guidance_scale,
                output_type="pil",
                callback_on_step_end=_make_release_callback(self.args.steps),
                callback_on_step_end_tensor_inputs=[],
            )
            if generator is not None:
                kwargs["generator"] = generator
            t_gen0 = time.time()
            result = self.pipe(**kwargs)
            gen_s = time.time() - t_gen0
            export_to_video(result.video, str(out_path), fps=self.args.fps)
            wall_s = time.time() - t_wall0

        entry = {
            "id": run_id,
            "prompt": prompt,
            "seed": seed,
            "wall_s": round(wall_s, 1),
            "gen_s": round(gen_s, 1),
            "mp4": f"/videos/{run_id}.mp4",
            "ts": time.time(),
        }
        self.history.append(entry)
        return entry


def build(state_args):
    import torch
    from PIL import Image

    import ttnn
    from models.tt_dit.experimental.cosmos3_i2v.demo.generate import close_mesh, open_mesh

    available = ttnn.get_num_devices()
    mesh_shape = _resolve_mesh_shape(state_args.mesh_shape, available)
    weight_dtype = _resolve_weight_dtype(state_args.weight_dtype, mesh_shape)
    print(
        f"[serve] mesh={mesh_shape}/{available} weight_dtype={weight_dtype} pipeline={state_args.pipeline}", flush=True
    )

    ref_image = Image.open(state_args.image).convert("RGB").resize((state_args.width, state_args.height))
    mesh = open_mesh(mesh_shape)
    try:
        t0 = time.time()
        if state_args.pipeline == "native":
            from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v_native import (
                build_cosmos3_i2v_native_pipeline,
            )

            pipe = build_cosmos3_i2v_native_pipeline(
                mesh,
                dtype=torch.bfloat16,
                use_tt_vae=True,
                num_links=state_args.num_links,
                flow_shift=state_args.flow_shift,
                trunk_weight_dtype=weight_dtype,
                vae_decoder_t_chunk_size=state_args.vae_decoder_t_chunk_size,
                vae_encoder_t_chunk_size=state_args.vae_encoder_t_chunk_size,
            )
        else:
            from models.tt_dit.experimental.cosmos3_i2v.pipelines.pipeline_cosmos3_i2v_native_cfg import (
                build_cosmos3_i2v_native_cfg_pipeline,
            )

            pipe = build_cosmos3_i2v_native_cfg_pipeline(
                mesh,
                dtype=torch.bfloat16,
                use_tt_vae=True,
                num_links=state_args.num_links,
                flow_shift=state_args.flow_shift,
                trunk_weight_dtype=weight_dtype,
                vae_decoder_t_chunk_size=state_args.vae_decoder_t_chunk_size,
                vae_encoder_t_chunk_size=state_args.vae_encoder_t_chunk_size,
                serial_dispatch=state_args.cfg_serial_dispatch,
            )
        print(f"[serve] pipeline built in {time.time() - t0:.1f}s", flush=True)
        state_args.out_dir.mkdir(parents=True, exist_ok=True)
        return mesh, pipe, ref_image
    except Exception:
        close_mesh(mesh)
        raise


_INDEX_HTML = """<!doctype html><html><head><title>Cosmos3-I2V</title>
<style>
body{font-family:system-ui,sans-serif;max-width:780px;margin:32px auto;padding:0 16px;color:#222}
textarea{width:100%;height:90px;font-size:15px;padding:8px;box-sizing:border-box}
button{padding:10px 18px;font-size:15px;cursor:pointer}
.hist{margin-top:32px;border-top:1px solid #ccc;padding-top:16px}
.row{padding:8px 0;border-bottom:1px solid #eee}
.t{color:#666;font-size:13px}
.busy{color:#a00}
</style></head><body>
<h2>Cosmos3-I2V generator</h2>
<form method="POST" action="/generate">
<label>Prompt:</label>
<textarea name="prompt" autofocus required>__PROMPT__</textarea>
<div style="margin-top:8px"><label>Seed (optional): <input name="seed" size="10"></label>
<button type="submit">Generate</button></div>
</form>
<p class="t">Steps=__STEPS__ · frames=__FRAMES__ · __WIDTH__×__HEIGHT__ · pipeline=__PIPELINE__</p>
<div class="hist"><h3>History</h3>__HISTORY__</div>
</body></html>
"""


def _render_history(history):
    if not history:
        return '<p class="t">(none yet)</p>'
    rows = []
    for h in reversed(history[-20:]):
        rows.append(
            f'<div class="row"><div><a href="{h["mp4"]}">{h["id"]}.mp4</a> '
            f'<span class="t">wall {h["wall_s"]}s · gen {h["gen_s"]}s</span></div>'
            f'<div class="t">prompt: {h["prompt"]}</div></div>'
        )
    return "\n".join(rows)


def make_handler(state: State):
    import urllib.parse
    from http.server import BaseHTTPRequestHandler

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # noqa: ARG002
            sys.stderr.write(f"[serve] {self.address_string()} - {fmt % a}\n")

        def _send_html(self, body: str, status: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _index(self, prompt: str = "") -> str:
            return (
                _INDEX_HTML.replace("__PROMPT__", prompt)
                .replace("__STEPS__", str(state.args.steps))
                .replace("__FRAMES__", str(state.args.frames))
                .replace("__WIDTH__", str(state.args.width))
                .replace("__HEIGHT__", str(state.args.height))
                .replace("__PIPELINE__", state.args.pipeline)
                .replace("__HISTORY__", _render_history(state.history))
            )

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send_html(self._index())
            elif self.path == "/history":
                data = json.dumps(state.history, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path.startswith("/videos/"):
                name = Path(self.path).name
                path = state.args.out_dir / name
                if not path.exists() or path.suffix != ".mp4":
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/generate":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            form = urllib.parse.parse_qs(raw)
            prompt = (form.get("prompt", [""])[0] or "").strip()
            seed_str = (form.get("seed", [""])[0] or "").strip()
            seed = int(seed_str) if seed_str else None
            if not prompt:
                self._send_html(self._index(), status=400)
                return
            try:
                entry = state.generate(prompt, seed=seed)
                print(
                    f"[serve] {entry['id']} wall={entry['wall_s']}s gen={entry['gen_s']}s "
                    f"prompt={entry['prompt']!r}",
                    flush=True,
                )
            except Exception as e:
                self._send_html(f"<pre>generation failed: {e}</pre><p><a href='/'>back</a></p>", status=500)
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    return H


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")

    mesh, pipe, ref_image = build(args)
    state = State(args, mesh, pipe, ref_image)

    from http.server import ThreadingHTTPServer

    from models.tt_dit.experimental.cosmos3_i2v.demo.generate import close_mesh

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[serve] listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        close_mesh(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
