# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Web UI + inference server for Ideogram 4.0 at 2048x2048 on Blackhole SP4xTP2.
#
# A background worker runs generations one-at-a-time (the mesh does one at a time);
# HTTP requests enqueue jobs and return immediately. Features:
#   * INPUT VALIDATION with proper HTTP error codes (never crashes the server):
#     empty prompt, unknown preset, non-integer seed, malformed JSON prompt, and
#     over-length prompts (pre-flight token count vs MAX_TEXT_TOKENS) are all
#     rejected with 400 + a structured {"error", "message"} body.
#   * JOB HISTORY: every job records its prompt, settings (preset/seed/size),
#     token count, and — once finished — its QUEUE TIME and PROCESSING TIME, plus
#     the output image. Served paginated so the page can lazy-load on scroll.
#   * LIVE QUEUE STATUS: how many jobs are queued and how long the running job has
#     been going, polled every second.
#   * The page shows the form + live status up top, and an infinite-scroll history
#     below whose images use loading="lazy" so nothing downloads until scrolled to.
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

import io
import itertools
import json
import os
import tempfile
import threading
import time

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.ideogram4.pipeline import MAX_TEXT_TOKENS, Ideogram4Pipeline
from .test_pipeline_class_ideogram4 import PROMPT as DEFAULT_PROMPT

PRESETS = ["V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48"]
PORT = 7860
HEIGHT = WIDTH = 2048
# The traced denoise path (device-resident latent) is the fast one and 2048px is a
# constant shape here, so there is no per-length recompile-transition risk. Flip to
# False if the board shows trace-capture flakiness across back-to-back generations.
TRACED = True

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Ideogram 4.0 — TT Blackhole</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:900px;margin:1.4rem auto;padding:0 1rem;color:#1b2430}}
 h1{{font-size:1.25rem}} h2{{font-size:1.05rem;margin-top:1.6rem;border-top:1px solid #e3e7ea;padding-top:1rem}}
 textarea{{width:100%;height:9rem;font-family:ui-monospace,monospace;font-size:.82rem}}
 select,input,button{{font-size:1rem;padding:.4rem}} .row{{margin:.55rem 0}}
 button{{background:#1b3a4b;color:#fff;border:0;border-radius:6px;padding:.6rem 1.2rem;cursor:pointer}}
 #status{{background:#f3f5f7;border:1px solid #dde2e6;border-radius:8px;padding:.6rem .9rem;margin:.8rem 0;font-size:.92rem}}
 #status b{{color:#1b3a4b}} .run{{color:#1b7f3b;font-weight:600}} .idle{{color:#888}}
 #submitmsg{{font-size:.9rem;margin:.4rem 0;min-height:1.1rem}} .ok{{color:#1b7f3b}} .err{{color:#bc4749}}
 .card{{border:1px solid #dde2e6;border-radius:8px;padding:.7rem .9rem;margin:.7rem 0;font-size:.88rem}}
 .card .meta{{color:#556;margin-bottom:.35rem}} .card .st{{font-weight:600}}
 .card details{{margin:.3rem 0}} .card summary{{cursor:pointer;color:#1b3a4b}}
 .card pre{{white-space:pre-wrap;word-break:break-word;background:#f7f9fa;border-radius:6px;padding:.5rem;font-size:.78rem;max-height:16rem;overflow:auto}}
 .card img{{max-width:100%;border:1px solid #ddd;border-radius:8px;margin-top:.5rem}}
 .note{{color:#888;font-size:.85rem}} #sentinel{{height:2rem}}
</style></head><body>
<h1>Ideogram 4.0 — 2048×2048 on Blackhole SP4×TP2</h1>
<form id="genform" onsubmit="return submitGen(event)">
 <div class="row"><label>Prompt (structured JSON works best; plain prose may wash out at 2k):</label>
  <textarea name="prompt" id="prompt">{prompt}</textarea></div>
 <div class="row">Preset: <select name="preset" id="preset">{preset_opts}</select>
  &nbsp; Seed: <input name="seed" id="seed" value="{seed}" size="10">
  &nbsp; <button type="submit">Generate</button></div>
 <div class="note">~55s (TURBO_12) · ~70s (DEFAULT_20) · ~165s (QUALITY_48). One at a time; new shapes recompile (slower first time).</div>
</form>
<div id="submitmsg"></div>
<div id="status">starting…</div>
<h2>History <span class="note" id="histcount"></span></h2>
<div id="history"></div>
<div id="sentinel"></div>
<script>
const esc = s => {{ const d=document.createElement('div'); d.textContent=s==null?'':String(s); return d.innerHTML; }};
const stIcon = j => j.status==='done'&&j.ok ? '✓' : (j.status==='error'||j.ok===false ? '<span class="err">✗</span>'
                 : (j.status==='running' ? '<span class="run">▶</span>' : '⏳'));
function timing(j) {{
  let t=[];
  if(j.queue_time!=null) t.push('queued '+j.queue_time.toFixed(1)+'s');
  if(j.proc_time!=null)  t.push('ran '+j.proc_time.toFixed(1)+'s');
  return t.join(' · ');
}}
function cardEl(j) {{
  const el=document.createElement('div'); el.className='card'; el.dataset.id=j.id;
  let h = '<div class="meta"><span class="st">'+stIcon(j)+' #'+j.id+'</span> · '+esc(j.preset)+
          ' · seed '+esc(j.seed)+' · '+esc(j.tokens)+' tok · '+j.width+'×'+j.height;
  const tm=timing(j); if(tm) h+=' · '+tm;
  h+='</div>';
  if(j.status==='error'||j.ok===false) h+='<div class="err">'+esc(j.err||'error')+'</div>';
  h+='<details><summary>prompt</summary><pre>'+esc(j.prompt)+'</pre></details>';
  if(j.has_image) h+='<img loading="lazy" src="/image/'+j.id+'" alt="job '+j.id+'">';
  el.innerHTML=h; return el;
}}

const hist=document.getElementById('history');
let oldestId=null, newestId=null, loadingOlder=false, exhausted=false;

async function loadOlder() {{
  if(loadingOlder||exhausted) return; loadingOlder=true;
  const q = oldestId==null ? '/history?limit=8' : '/history?limit=8&before='+oldestId;
  try {{
    const j = await (await fetch(q)).json();
    document.getElementById('histcount').textContent = '('+j.total+' total)';
    if(!j.items.length) {{ exhausted=true; }}
    for(const it of j.items) {{ hist.appendChild(cardEl(it)); oldestId=it.id; if(newestId==null) newestId=it.id; }}
    if(j.items.length) newestId=Math.max(newestId, j.items[0].id);
  }} catch(e) {{}} finally {{ loadingOlder=false; }}
}}
async function loadNewer() {{
  if(newestId==null) return;
  try {{
    const j = await (await fetch('/history?after='+newestId)).json();
    document.getElementById('histcount').textContent = '('+j.total+' total)';
    // items are newest-first; insert reversed so the newest ends up on top.
    for(const it of j.items.slice().reverse()) {{ hist.insertBefore(cardEl(it), hist.firstChild); newestId=Math.max(newestId,it.id); }}
  }} catch(e) {{}}
}}

async function poll() {{
  try {{
    const j = await (await fetch('/status')).json();
    let h = '<b>Queued:</b> ' + j.queued + ' &nbsp;|&nbsp; ';
    if (j.running) {{
      h += '<span class="run">Running:</span> #'+j.running.id+' '+esc(j.running.preset)+' · seed '+esc(j.running.seed)+
           ' · <b>'+j.running.elapsed.toFixed(0)+'s</b> elapsed';
    }} else {{ h += '<span class="idle">Idle</span>'; }}
    document.getElementById('status').innerHTML = h;
  }} catch (e) {{ document.getElementById('status').innerText = 'status unavailable: ' + e; }}
  loadNewer();  // pick up any newly-finished jobs (metadata only; images lazy-load on scroll)
}}

async function submitGen(ev) {{
  ev.preventDefault();
  const msg=document.getElementById('submitmsg'); msg.className=''; msg.textContent='submitting…';
  try {{
    const body={{prompt:document.getElementById('prompt').value, preset:document.getElementById('preset').value, seed:document.getElementById('seed').value}};
    const r=await fetch('/generate',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});
    const j=await r.json();
    if(!r.ok) {{ msg.className='err'; msg.textContent='✗ '+(j.message||j.error||('HTTP '+r.status)); }}
    else {{ msg.className='ok'; msg.textContent='✓ queued as job #'+j.id+' ('+j.tokens+' tokens)'; poll(); }}
  }} catch(e) {{ msg.className='err'; msg.textContent='✗ request failed: '+e; }}
  return false;
}}

new IntersectionObserver(es => {{ if(es[0].isIntersecting) loadOlder(); }}).observe(document.getElementById('sentinel'));
loadOlder(); setInterval(poll, 1000); poll();
</script>
</body></html>"""


def _page(prompt=DEFAULT_PROMPT, preset="V4_TURBO_12", seed=1234):
    opts = "".join(f'<option value="{p}"{" selected" if p == preset else ""}>{p}</option>' for p in PRESETS)
    return _PAGE.format(prompt=prompt, preset_opts=opts, seed=seed)


def make_server(pipe, img_dir: str | None = None):
    """Build the Flask app + start the background worker for a ready pipeline.

    Split out from ``test_serve`` (which builds the real device pipeline) so the request
    handling — input validation, queue, history pagination — can be exercised with a mock
    ``pipe`` (needs only ``count_text_tokens`` + ``__call__``) via Flask's test client, with
    no device. ``pipe`` runs one generation at a time on the mesh; the worker serializes jobs.
    """
    from flask import Flask, abort, request, send_file

    if img_dir is None:
        img_dir = tempfile.mkdtemp(prefix="ideogram4_serve_")
    logger.info(f"Serving generated images from {img_dir}")

    lock = threading.Lock()
    jobs: dict[int, dict] = {}  # id -> full job record
    pending: list[int] = []  # queued job ids (FIFO)
    state = {"running": None}  # running job id or None
    ids = itertools.count(1)

    def _public(r: dict) -> dict:
        """JSON-safe view of a job record with derived queue/processing times."""
        qt = (r["start_ts"] - r["submit_ts"]) if r.get("start_ts") else None
        pt = (r["end_ts"] - r["start_ts"]) if (r.get("end_ts") and r.get("start_ts")) else None
        return {
            "id": r["id"],
            "status": r["status"],
            "ok": r.get("ok"),
            "preset": r["preset"],
            "seed": r["seed"],
            "height": r["height"],
            "width": r["width"],
            "tokens": r["tokens"],
            "prompt": r["prompt"],
            "err": r.get("err"),
            "submit_ts": r["submit_ts"],
            "queue_time": qt,
            "proc_time": pt,
            "has_image": r.get("has_image", False),
        }

    def worker():
        while True:
            with lock:
                jid = pending.pop(0) if pending else None
            if jid is None:
                time.sleep(0.15)
                continue
            with lock:
                rec = jobs[jid]
                rec["status"] = "running"
                rec["start_ts"] = time.time()
                state["running"] = jid
            ok, err, png = True, None, None
            try:
                img = pipe(
                    rec["prompt"],
                    height=rec["height"],
                    width=rec["width"],
                    preset=rec["preset"],
                    seed=rec["seed"],
                    traced=TRACED,
                )
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")
                png = buf.getvalue()
            except Exception as e:  # noqa: BLE001 — never let a bad job kill the worker
                ok, err = False, str(e)
                logger.exception("generation failed")
            end = time.time()
            with lock:
                if png is not None:
                    with open(os.path.join(img_dir, f"{jid}.png"), "wb") as f:
                        f.write(png)
                    rec["has_image"] = True
                rec["status"] = "done" if ok else "error"
                rec["ok"] = ok
                rec["err"] = err
                rec["end_ts"] = end
                state["running"] = None
            logger.info(
                f"job #{jid} {rec['preset']} seed={rec['seed']} ok={ok} "
                f"queue={rec['start_ts'] - rec['submit_ts']:.1f}s proc={end - rec['start_ts']:.1f}s"
            )

    threading.Thread(target=worker, daemon=True).start()

    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return _page()

    @app.route("/generate", methods=["POST"])
    def generate():
        # Accept JSON (the page's fetch) or form-encoded. Validate everything and return a
        # structured 400 on any bad input — the server never 500s / crashes on a bad request.
        data = request.get_json(silent=True) or request.form
        prompt = (data.get("prompt") or "").strip()
        preset = data.get("preset") or "V4_TURBO_12"
        seed_raw = data.get("seed")

        if not prompt:
            return {"error": "empty_prompt", "message": "Prompt is empty."}, 400
        if preset not in PRESETS:
            return {"error": "invalid_preset", "message": f"Unknown preset {preset!r}. Choose one of {PRESETS}."}, 400
        try:
            seed = int(seed_raw) if seed_raw not in (None, "") else 1234
        except (ValueError, TypeError):
            return {"error": "invalid_seed", "message": f"Seed must be an integer, got {seed_raw!r}."}, 400
        # If the prompt is meant to be JSON (structured prompts start with { or [), it must parse.
        if prompt.lstrip()[:1] in "{[":
            try:
                json.loads(prompt)
            except json.JSONDecodeError as e:
                return {"error": "malformed_json", "message": f"Prompt looks like JSON but failed to parse: {e}"}, 400
        # Pre-flight token count so an over-length prompt is rejected here, not mid-generation.
        try:
            n_tok = pipe.count_text_tokens(prompt)
        except Exception as e:  # noqa: BLE001 — tokenizer failure is a client-visible 400, not a crash
            return {"error": "tokenize_failed", "message": f"Could not tokenize prompt: {e}"}, 400
        if n_tok > MAX_TEXT_TOKENS:
            return {
                "error": "too_many_tokens",
                "message": f"Prompt is {n_tok} tokens; the maximum is {MAX_TEXT_TOKENS}.",
                "tokens": n_tok,
                "max_tokens": MAX_TEXT_TOKENS,
            }, 400

        with lock:
            jid = next(ids)
            jobs[jid] = {
                "id": jid,
                "prompt": prompt,
                "preset": preset,
                "seed": seed,
                "tokens": n_tok,
                "height": HEIGHT,
                "width": WIDTH,
                "status": "queued",
                "submit_ts": time.time(),
                "start_ts": None,
                "end_ts": None,
                "ok": None,
                "err": None,
                "has_image": False,
            }
            pending.append(jid)
            depth = len(pending)
        return {"id": jid, "queued": depth, "tokens": n_tok}, 202

    @app.route("/status", methods=["GET"])
    def status():
        with lock:
            run = state["running"]
            running = None
            if run is not None:
                r = jobs[run]
                running = {
                    "id": run,
                    "preset": r["preset"],
                    "seed": r["seed"],
                    "elapsed": time.time() - r["start_ts"],
                    "prompt": r["prompt"][:120],
                }
            return {"queued": len(pending), "running": running, "total": len(jobs)}

    @app.route("/history", methods=["GET"])
    def history():
        # Paginated, newest-first. ?before=<id> for older jobs (infinite scroll down);
        # ?after=<id> for jobs newer than what the page has (live prepend); default = newest page.
        try:
            limit = min(max(int(request.args.get("limit", 8)), 1), 50)
        except (ValueError, TypeError):
            limit = 8
        before = request.args.get("before")
        after = request.args.get("after")
        with lock:
            ids_desc = sorted(jobs.keys(), reverse=True)
            if after not in (None, ""):
                try:
                    a = int(after)
                except ValueError:
                    return {"error": "bad_after", "message": "after must be an integer"}, 400
                sel = [i for i in ids_desc if i > a]
            elif before not in (None, ""):
                try:
                    b = int(before)
                except ValueError:
                    return {"error": "bad_before", "message": "before must be an integer"}, 400
                sel = [i for i in ids_desc if i < b][:limit]
            else:
                sel = ids_desc[:limit]
            items = [_public(jobs[i]) for i in sel]
            total = len(jobs)
        return {"items": items, "total": total}

    @app.route("/job/<int:jid>", methods=["GET"])
    def job(jid):
        with lock:
            r = jobs.get(jid)
            if r is None:
                abort(404)
            return _public(r)

    @app.route("/image/<int:jid>", methods=["GET"])
    def image(jid):
        path = os.path.join(img_dir, f"{jid}.png")
        if not os.path.exists(path):
            abort(404)
        return send_file(path, mimetype="image/png")

    return app


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [pytest.param((4, 2), (4, 2), 1, id="sp4tp2")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}], indirect=True
)
def test_serve(*, mesh_device, submesh_shape, tp_axis) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    logger.info("Building Ideogram4Pipeline (device-resident, SP4xTP2)... first build can take a few minutes.")
    pipe = Ideogram4Pipeline.from_pretrained(submesh, tp_axis=tp_axis)
    logger.info("Pipeline ready.")

    app = make_server(pipe)
    logger.info(f"Ideogram4 web UI listening on http://0.0.0.0:{PORT}  (2048px, traced={TRACED}, queue+history)")
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
