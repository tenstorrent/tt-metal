# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Read-only browser viewer for Qwen Image 2.1 validation images.

The page polls the artifact directory, so a newly written PNG appears without
restarting the server. Only PNGs under --artifacts-dir and the explicit CUDA
reference image are served.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen Image 2.1 · CUDA / TT viewer</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#0d1520;color:#e8eef7}
body{margin:0}main{max-width:1200px;margin:0 auto;padding:28px 24px 60px}
h1{font-size:1.9rem;margin:.2rem 0}.sub{color:#aebfd1;line-height:1.5;margin:.4rem 0 1.4rem}
.status{display:flex;gap:16px;align-items:center;color:#9bb0c6;font-size:.9rem;margin-bottom:20px}
.live{color:#7bdfb2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
.card{background:#172536;border:1px solid #2a4056;border-radius:12px;padding:16px;min-width:0}
.card h2{font-size:1.1rem;margin:0 0 12px}.frame{aspect-ratio:1;background:#101b29;display:grid;place-items:center;border-radius:8px;overflow:hidden}
.frame img{display:block;width:100%;height:100%;object-fit:contain;image-rendering:auto}
.placeholder{color:#8397aa}.meta{font-size:.85rem;color:#9bb0c6;margin-top:12px;overflow-wrap:anywhere}
.metrics{margin:20px 0;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.metric{background:#172536;border:1px solid #2a4056;border-radius:10px;padding:14px}
.metric strong{font-size:1.25rem;display:block;color:#fff}.metric span{font-size:.8rem;color:#9bb0c6}
details{background:#172536;border:1px solid #2a4056;border-radius:10px;padding:14px;margin-top:22px}
summary{cursor:pointer;font-weight:600}.archive{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
.archive a{color:#a9d7ff;background:#22364b;border-radius:7px;padding:8px 10px;text-decoration:none}
.note{color:#aebfd1;line-height:1.5;margin-top:18px}
</style></head><body><main>
<h1>Qwen Image 2.1 · CUDA / TT viewer</h1>
<p class="sub" id="description">Loading capture details…</p>
<div class="status"><span class="live" id="live">Connecting…</span><span id="updated"></span><span id="progress"></span></div>
<div class="grid">
  <section class="card"><h2>CUDA reference · same step</h2><div class="frame" id="cuda-frame"><span class="placeholder">Waiting for image</span></div><div class="meta" id="cuda-meta"></div></section>
  <section class="card"><h2>TT denoiser → CUDA VAE</h2><div class="frame" id="tt-frame"><span class="placeholder">Waiting for image</span></div><div class="meta" id="tt-meta"></div></section>
</div>
<div class="metrics" id="metrics"></div>
<p class="note">The TT image uses CUDA VAE decoding for validation. Prompt encoding and initial noise also come from the pinned CUDA reference; denoising steps run on TT.</p>
<details><summary>Saved images</summary><div class="archive" id="archive"></div></details>
</main><script>
const shown = new Map();
function putImage(frameId, metaId, item) {
  const frame = document.getElementById(frameId), meta = document.getElementById(metaId);
  if (!item) { frame.replaceChildren(Object.assign(document.createElement('span'), {className:'placeholder', textContent:'Waiting for image'})); meta.textContent=''; return; }
  const key = item.url + '?' + item.mtime_ns;
  if (shown.get(frameId) !== key) {
    const img = document.createElement('img'); img.alt = item.name; img.src = key;
    frame.replaceChildren(img); shown.set(frameId, key);
  }
  meta.textContent = item.name + ' · ' + new Date(item.mtime * 1000).toLocaleString();
}
function metric(label, value) {
  const box=document.createElement('div'); box.className='metric';
  const strong=document.createElement('strong'); strong.textContent=value;
  const span=document.createElement('span'); span.textContent=label;
  box.append(strong,span); return box;
}
async function refresh() {
  try {
    const response=await fetch('/api/state', {cache:'no-store'});
    if (!response.ok) throw new Error('HTTP '+response.status);
    const state=await response.json();
    const m=state.metadata;
    document.getElementById('description').textContent='“'+m.prompt+'” · seed '+m.seed+' · '+m.width+' × '+m.height+' · '+m.steps+' denoising steps. Images and reports refresh every 5 seconds.';
    for (const frame of document.querySelectorAll('.frame')) frame.style.aspectRatio=m.width+'/'+m.height;
    document.getElementById('live').textContent='Live · read-only';
    document.getElementById('updated').textContent='Checked '+new Date().toLocaleTimeString();
    const p=state.progress || {};
    document.getElementById('progress').textContent=p.total_steps ? ('TT '+p.status+' · '+p.completed_steps+'/'+p.total_steps+' steps') : (p.stage ? ('TT VAE '+p.status+' · '+p.stage) : 'TT pending');
    putImage('cuda-frame','cuda-meta',state.cuda);
    putImage('tt-frame','tt-meta',state.tt);
    const metrics=document.getElementById('metrics'); metrics.replaceChildren();
    const r=state.report || {};
    if (p.latest && Number.isFinite(p.latest.latent_relative_rms_error)) metrics.append(metric('Latest latent relative RMS', (100*p.latest.latent_relative_rms_error).toFixed(2)+'%'));
    if (p.latest && Number.isFinite(p.latest.velocity_relative_rms_error)) metrics.append(metric('Latest velocity relative RMS', (100*p.latest.velocity_relative_rms_error).toFixed(2)+'%'));
    if (Number.isFinite(r.tt_vs_cuda_latent_relative_rms_error)) metrics.append(metric(r.step===m.steps-1?'Final latent relative RMS':'Preview latent relative RMS', (100*r.tt_vs_cuda_latent_relative_rms_error).toFixed(2)+'%'));
    if (Number.isFinite(r.tt_vs_cuda_pixel_mean_abs_error)) metrics.append(metric('Mean pixel difference / 255', r.tt_vs_cuda_pixel_mean_abs_error.toFixed(2)));
    if (Number.isFinite(r.cuda_redecode_exact_fraction)) metrics.append(metric('CUDA re-decode exact pixels', (100*r.cuda_redecode_exact_fraction).toFixed(1)+'%'));
    if (Number.isFinite(r.pcc)) metrics.append(metric('VAE output PCC vs CUDA', r.pcc.toFixed(6)));
    if (Number.isFinite(r.relative_rms_error)) metrics.append(metric('VAE output relative RMS', (100*r.relative_rms_error).toFixed(2)+'%'));
    const archive=document.getElementById('archive'); archive.replaceChildren();
    for (const item of state.images) {
      const link=document.createElement('a'); link.href=item.url; link.target='_blank'; link.rel='noopener'; link.textContent=item.name;
      archive.append(link);
    }
  } catch (error) { document.getElementById('live').textContent='Viewer error: '+error.message; }
}
refresh(); setInterval(refresh,5000);
</script></body></html>"""


def make_handler(artifacts: Path, reference: Path, tt_vae: bool = False, integrated: bool = False):
    artifacts = artifacts.resolve()
    reference = reference.resolve()
    manifest = json.loads((reference.parent / "manifest.json").read_text())
    metadata = {key: manifest[key] for key in ("prompt", "seed", "height", "width", "steps")}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, data: bytes, kind: str, *, cache: str = "no-store") -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(data)

        @staticmethod
        def _item(path: Path, name: str, url: str) -> dict[str, object]:
            stat = path.stat()
            return {"name": name, "url": url, "mtime": stat.st_mtime, "mtime_ns": stat.st_mtime_ns}

        def _state(self) -> dict[str, object]:
            images = []
            if reference.is_file():
                images.append(self._item(reference, "CUDA pipeline reference", "/image/reference"))
            for path in artifacts.rglob("*.png"):
                if path.is_file():
                    relative = path.relative_to(artifacts).as_posix()
                    images.append(self._item(path, relative, "/image/" + quote(relative)))
            images.sort(key=lambda item: item["mtime_ns"], reverse=True)
            tt_name = "tt_vae.png" if tt_vae or integrated else "tt_latents_cuda_vae.png"
            tt = next((item for item in images if item["name"].endswith(tt_name)), None)
            cuda = next((item for item in images if item["name"] == "CUDA pipeline reference"), None)
            report = None
            if tt:
                preview_dir = Path(str(tt["name"])).parent
                paired_name = "cuda_vae.png" if tt_vae else "cuda_redecoded.png"
                paired = artifacts / preview_dir / paired_name
                if tt_vae and not integrated and not paired.is_file():
                    cuda = None
                if paired.is_file() and not integrated:
                    cuda = self._item(
                        paired,
                        (preview_dir / paired_name).as_posix(),
                        "/image/" + quote((preview_dir / paired_name).as_posix()),
                    )
                report_path = artifacts / preview_dir / "report.json"
                if report_path.is_file():
                    report = json.loads(report_path.read_text())
                    if isinstance(report, list):
                        report = report[-1] if report else None
            progress_path = artifacts / "progress.json"
            progress = json.loads(progress_path.read_text()) if progress_path.is_file() else None
            return {
                "images": images,
                "tt": tt,
                "cuda": cuda,
                "report": report,
                "progress": progress,
                "metadata": metadata,
            }

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                page = PAGE
                if integrated:
                    page = page.replace("CUDA reference · same step", "CUDA pipeline · native inputs")
                    page = page.replace("TT denoiser → CUDA VAE", "TT pipeline · native inputs")
                    page = page.replace(
                        "The TT image uses CUDA VAE decoding for validation. Prompt encoding and initial noise also come from the pinned CUDA reference; denoising steps run on TT.",
                        "Both pipelines start from the same raw prompt and seed and compute their own prompt embeddings, initial noise, all denoising steps, and VAE decode. No captured activations are injected. Prompt expansion is off. Backend-native random generators produce different initial noise, so this compares independent generation rather than pixel accuracy.",
                    )
                elif tt_vae:
                    page = page.replace("CUDA reference · same step", "CUDA VAE · same TT latent")
                    page = page.replace("TT denoiser → CUDA VAE", "TT denoiser → TT VAE")
                    page = page.replace(
                        "The TT image uses CUDA VAE decoding for validation. Prompt encoding and initial noise also come from the pinned CUDA reference; denoising steps run on TT.",
                        "The saved final latent was denoised on TT and decoded with the new TT VAE. The CUDA VAE comparison uses that same TT latent to isolate decoder accuracy. Prompt encoding and initial noise came from the pinned reference.",
                    )
                self._send(page.encode(), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                try:
                    self._send(json.dumps(self._state()).encode(), "application/json")
                except (OSError, ValueError) as error:
                    self.send_error(500, str(error))
                return
            if path.startswith("/image/"):
                name = unquote(path.removeprefix("/image/"))
                if name == "reference":
                    source = reference
                else:
                    source = (artifacts / name).resolve()
                    if not source.is_relative_to(artifacts):
                        self.send_error(403)
                        return
                if source.suffix.lower() != ".png" or not source.is_file():
                    self.send_error(404)
                    return
                try:
                    self._send(source.read_bytes(), mimetypes.guess_type(source.name)[0] or "image/png")
                except OSError as error:
                    self.send_error(500, str(error))
                return
            self.send_error(404)

    return Handler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument(
        "--tt-vae", action="store_true", help="show TT VAE output against CUDA decoding of the same latent"
    )
    parser.add_argument(
        "--integrated", action="store_true", help="compare independently generated native CUDA and TT images"
    )
    args = parser.parse_args(argv)
    if not args.artifacts_dir.is_dir():
        parser.error(f"artifact directory does not exist: {args.artifacts_dir}")
    if not args.reference.is_file():
        parser.error(f"reference image does not exist: {args.reference}")
    server = ThreadingHTTPServer(
        (args.bind, args.port), make_handler(args.artifacts_dir, args.reference, args.tt_vae, args.integrated)
    )
    print(f"Serving Qwen Image 2.1 viewer on http://{args.bind}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
