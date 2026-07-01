# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Tiny Flask web UI to try Ideogram 4.0 prompts interactively at 2048x2048,
# UNTRACED, with a user-selectable sampler preset.
#
# A background worker thread runs generations one-at-a-time (the mesh does one at
# a time); HTTP requests enqueue jobs and return immediately. The page polls
# /status every second so the QUEUE DEPTH and the running job's LIVE ELAPSED time
# are always visible, plus a history of recent generations with their durations.
#
# NOT a CI test: the file is intentionally NOT named test_*.py, so default pytest
# collection skips it. It defines a `test_serve` function only so it can borrow the
# `mesh_device` fixture (which does the correct (4,2)+FABRIC_1D mesh setup/teardown).
#
# Launch (serves on 0.0.0.0:7860 until killed):
#   source container_python_env/bin/activate
#   pytest models/tt_dit/tests/models/ideogram4/serve_ideogram4.py::test_serve \
#     -s -q -p no:cacheprovider --timeout=0
# =============================================================================

import collections
import io
import itertools
import threading
import time

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.ideogram4.pipeline import Ideogram4Pipeline
from .test_pipeline_class_ideogram4 import PROMPT as DEFAULT_PROMPT

PRESETS = ["V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48"]
PORT = 7860
HEIGHT = WIDTH = 2048

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Ideogram 4.0 — TT Blackhole</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:880px;margin:1.5rem auto;padding:0 1rem;color:#1b2430}}
 h1{{font-size:1.25rem}} textarea{{width:100%;height:9rem;font-family:ui-monospace,monospace;font-size:.82rem}}
 select,input,button{{font-size:1rem;padding:.4rem}} .row{{margin:.55rem 0}}
 button{{background:#1b3a4b;color:#fff;border:0;border-radius:6px;padding:.6rem 1.2rem;cursor:pointer}}
 #status{{background:#f3f5f7;border:1px solid #dde2e6;border-radius:8px;padding:.6rem .9rem;margin:.8rem 0;font-size:.92rem}}
 #status b{{color:#1b3a4b}} .run{{color:#1b7f3b;font-weight:600}} .idle{{color:#888}}
 ul.hist{{list-style:none;padding-left:0;margin:.3rem 0;font-size:.85rem;color:#555}}
 ul.hist li{{padding:.1rem 0}} .err{{color:#bc4749}} img{{max-width:100%;border:1px solid #ddd;border-radius:8px;margin-top:1rem}}
 .note{{color:#888;font-size:.85rem}}
</style></head><body>
<h1>Ideogram 4.0 — 2048×2048 (untraced) on Blackhole SP4×TP2</h1>
<form method="post" action="/generate">
 <div class="row"><label>Prompt (structured JSON works best; plain prose may wash out at 2k):</label>
  <textarea name="prompt">{prompt}</textarea></div>
 <div class="row">Preset: <select name="preset">{preset_opts}</select>
  &nbsp; Seed: <input name="seed" value="{seed}" size="10">
  &nbsp; <button type="submit">Generate</button></div>
 <div class="note">~55s (TURBO_12) · ~70s (DEFAULT_20) · ~165s (QUALITY_48). One at a time; new shapes recompile (slower first time).</div>
</form>
<div id="status">starting…</div>
<div id="result"></div>
<script>
let shown = null;
async function poll() {{
  try {{
    const j = await (await fetch('/status')).json();
    let h = '<b>Queued:</b> ' + j.queued + ' &nbsp;|&nbsp; ';
    if (j.running) {{
      h += '<span class="run">Running:</span> ' + j.running.preset + ' · seed ' + j.running.seed +
           ' · <b>' + j.running.elapsed.toFixed(0) + 's</b> elapsed';
    }} else {{
      h += '<span class="idle">Idle</span>';
    }}
    if (j.history && j.history.length) {{
      h += '<ul class="hist">';
      for (const e of j.history) {{
        h += '<li>' + (e.ok ? '✓' : '<span class="err">✗</span>') + ' ' + e.preset +
             ' · seed ' + e.seed + ' · ' + e.dur.toFixed(0) + 's' +
             (e.ok ? '' : ' — <span class="err">' + (e.err||'error') + '</span>') + '</li>';
      }}
      h += '</ul>';
    }}
    document.getElementById('status').innerHTML = h;
    if (j.latest && j.latest !== shown) {{
      shown = j.latest;
      document.getElementById('result').innerHTML = '<img src="/image/' + j.latest + '?t=' + Date.now() + '">';
    }}
  }} catch (e) {{ document.getElementById('status').innerText = 'status unavailable: ' + e; }}
}}
setInterval(poll, 1000); poll();
</script>
</body></html>"""


def _page(prompt=DEFAULT_PROMPT, preset="V4_TURBO_12", seed=1234):
    opts = "".join(f'<option value="{p}"{" selected" if p == preset else ""}>{p}</option>' for p in PRESETS)
    return _PAGE.format(prompt=prompt, preset_opts=opts, seed=seed)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [pytest.param((4, 2), (4, 2), 1, id="sp4tp2")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}], indirect=True
)
def test_serve(*, mesh_device, submesh_shape, tp_axis) -> None:
    from flask import Flask, abort, request

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    logger.info("Building Ideogram4Pipeline (device-resident, SP4xTP2)... first build can take a few minutes.")
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    logger.info("Pipeline ready.")

    lock = threading.Lock()
    pending = collections.deque()  # queued jobs: {id, prompt, preset, seed}
    history = collections.deque(maxlen=12)  # finished (newest first): {id, preset, seed, dur, ok, err}
    images: dict[str, bytes] = {}  # job id -> PNG bytes (kept for finished jobs)
    state = {"running": None}  # {id, preset, seed, t_start} or None
    ids = itertools.count(1)

    def worker():
        while True:
            with lock:
                job = pending.popleft() if pending else None
            if job is None:
                time.sleep(0.2)
                continue
            with lock:
                state["running"] = {**job, "t_start": time.time()}
            ok, err, png = True, None, None
            try:
                img = pipe(
                    job["prompt"], height=HEIGHT, width=WIDTH, preset=job["preset"], seed=job["seed"], traced=False
                )
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                png = buf.getvalue()
            except Exception as e:  # noqa: BLE001 — surface to the page (e.g. >2048 tokens)
                ok, err = False, str(e)
                logger.exception("generation failed")
            dur = time.time() - state["running"]["t_start"]
            with lock:
                if png is not None:
                    images[job["id"]] = png
                history.appendleft(
                    {"id": job["id"], "preset": job["preset"], "seed": job["seed"], "dur": dur, "ok": ok, "err": err}
                )
                state["running"] = None
            logger.info(f"job {job['id']} {job['preset']} seed={job['seed']} ok={ok} in {dur:.0f}s")

    threading.Thread(target=worker, daemon=True).start()

    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return _page()

    @app.route("/generate", methods=["POST"])
    def generate():
        prompt = (request.form.get("prompt") or "").strip()
        preset = request.form.get("preset") if request.form.get("preset") in PRESETS else "V4_TURBO_12"
        try:
            seed = int(request.form.get("seed") or 1234)
        except ValueError:
            seed = 1234
        if prompt:
            with lock:
                pending.append({"id": str(next(ids)), "prompt": prompt, "preset": preset, "seed": seed})
        return _page(prompt=prompt or DEFAULT_PROMPT, preset=preset, seed=seed)

    @app.route("/status", methods=["GET"])
    def status():
        with lock:
            run = state["running"]
            running = (
                None
                if run is None
                else {"preset": run["preset"], "seed": run["seed"], "elapsed": time.time() - run["t_start"]}
            )
            hist = list(history)
            latest = next((e["id"] for e in hist if e["ok"]), None)
            return {"queued": len(pending), "running": running, "history": hist, "latest": latest}

    @app.route("/image/<jid>", methods=["GET"])
    def image(jid):
        with lock:
            png = images.get(jid)
        if png is None:
            abort(404)
        return app.response_class(png, mimetype="image/png")

    logger.info(f"Ideogram4 web UI listening on http://0.0.0.0:{PORT}  (2048px, untraced, queue+live-status)")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
