# SPDX-License-Identifier: Apache-2.0
"""run-demo: run a model's OWN on-device demo on real inputs and collect the batch of real outputs.

After a model has been brought up / optimized, this runs the exact demo the model ships (DISCOVERED,
never hardcoded) on the SAME device the run used, capturing whatever the demo produces -- a batch of
audio, a batch of images, or a batch of generated text -- into an out dir with a ``manifest.json`` the
dashboard renders.

Everything is derived from the model itself, so it works for any model with no per-model code:
  * the demo entry script is found by scanning ``<demo_dir>/demo/*.py`` for a ``main``/``__main__`` and
    picking the one whose name best matches the run's OWN task (taken from the run's discovered perf/pcc
    test), so a model with several demos (e.g. continuation vs text-to-speech) runs the right one;
  * its output-dir / batch CLI flags are read STATICALLY from the demo's ``add_argument`` calls (no
    import, no device needed to introspect);
  * the modality is inferred from the files the demo actually WRITES (audio/image extensions) or, when it
    writes none, from its captured stdout -- never from a model-name lookup.
"""

from __future__ import annotations

import ast
import datetime as _dt
import glob as _glob
import json
import os
import re
import subprocess
from pathlib import Path

from .optimize import _repo_root, _resolve_target
from .publish_hf import _checkout_of, _discover_test_node

# Extension -> modality. This is the ONLY "type" knowledge, and it keys off what the demo writes, so a
# new model that emits the same media renders with no change here (no model/stage names baked in).
_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
_TEXT_EXTS = (".txt", ".json", ".md", ".jsonl")


def _kind_of(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _TEXT_EXTS:
        return "text"
    return None


def _demo_scripts(demo_dir: Path) -> list[Path]:
    """Every runnable demo script under the model's own ``demo/`` dir (has a ``main`` / ``__main__``)."""
    out = []
    for f in sorted(_glob.glob(str(Path(demo_dir) / "demo" / "*.py"))):
        name = Path(f).name
        if name.startswith("__"):
            continue
        try:
            src = Path(f).read_text(errors="replace")
        except Exception:
            continue
        if "__main__" in src or "def main" in src:
            out.append(Path(f))
    return out


def _task_tokens(checkout: Path, demo_dir: Path) -> list[str]:
    """Tokens describing the run's OWN task, from its discovered perf/pcc test's file name (e.g.
    ``test_e2e_text_to_speech.py`` -> {text, speech}). Never a hardcoded task name."""
    node = _discover_test_node(Path(checkout), Path(demo_dir), "perf") or _discover_test_node(
        Path(checkout), Path(demo_dir), "pcc"
    )
    if not node:
        return []
    fname = Path(node.split("::")[0]).name.lower()
    drop = {"test", "e2e", "py", "perf", "pcc", "gate", "gates", "main", "component"}
    return [t for t in re.split(r"[^a-z0-9]+", fname) if len(t) > 2 and t not in drop]


def _pick_demo(demo_dir: Path, slug: str, task_toks: list[str]) -> tuple[Path | None, str | None]:
    """(demo_file, ``python -m`` module) for the model's runnable demo whose name best matches the run's
    task; falls back to the first runnable demo. Returns (None, None) if the model ships no demo."""
    scripts = _demo_scripts(demo_dir)
    if not scripts:
        return None, None
    chosen = scripts[0]
    if task_toks:
        best_score = -1
        for f in scripts:
            low = f.stem.lower()
            score = sum(1 for t in task_toks if t in low)
            if score > best_score:
                best_score, chosen = score, f
    return chosen, f"models.demos.{slug}.demo.{chosen.stem}"


def _discover_flags(demo_file: Path) -> tuple[str | None, str | None, str | None]:
    """Statically read the demo's argparse options (no import / no device) and pick its output-dir and
    batch flags, plus the out flag's default. Generic: keys off the ROLE of the option name, not a fixed
    string -- an out flag is a long option whose name mentions out(+dir/path); a batch flag mentions
    batch/users/samples. Returns (out_flag, out_default, batch_flag)."""
    try:
        tree = ast.parse(Path(demo_file).read_text(errors="replace"))
    except Exception:
        return None, None, None
    opts: list[tuple[str, object]] = []  # (long_flag, default)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        flag = None
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("--"):
                flag = a.value
                break
        if not flag:
            continue
        default = None
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                default = kw.value.value
        opts.append((flag, default))

    def _match(preds) -> tuple[str | None, object]:
        for want_all in preds:  # each pred is a set of substrings that must ALL be present
            for flag, dflt in opts:
                name = flag.lstrip("-").lower()
                if all(s in name for s in want_all):
                    return flag, dflt
        return None, None

    out_flag, out_default = _match([{"out", "dir"}, {"out", "path"}, {"output"}, {"out"}])
    batch_flag, _ = _match([{"batch"}, {"users"}, {"samples"}, {"num"}])
    return out_flag, (out_default if isinstance(out_default, str) else None), batch_flag


def _collect(out_dir: Path, stdout: str) -> tuple[str, list[dict]]:
    """(modality, items) from what the demo produced. Media files win; otherwise the stdout is the
    batch of text answers. Items carry absolute box paths the dashboard streams back per item."""
    files = [Path(p) for p in _glob.glob(str(Path(out_dir) / "**" / "*"), recursive=True) if Path(p).is_file()]
    media = sorted(f for f in files if _kind_of(f) in ("audio", "image"))
    if media:
        kinds = {_kind_of(f) for f in media}
        modality = "audio" if "audio" in kinds else "image"
        items = [
            {"kind": _kind_of(f), "path": str(f.resolve()), "name": f.name} for f in media if _kind_of(f) == modality
        ]
        return modality, items
    # No media -> text. Prefer per-sample text files if the demo wrote them, else the captured stdout.
    text_files = sorted(f for f in files if _kind_of(f) == "text")
    items = []
    for f in text_files:
        try:
            items.append({"kind": "text", "name": f.name, "text": f.read_text(errors="replace")[:20000]})
        except Exception:
            pass
    if not items:
        items = [{"kind": "text", "name": "output", "text": (stdout or "").strip()[-20000:]}]
    return "text", items


def cmd_run_demo(args) -> int:
    from .optimize_dashboard import find_run_dir, run_slug
    from ..bringup_loop import find_demo_dir

    repo_root = _repo_root()

    # Resolve the run -> model demo dir, the SAME way publish-hf does (explicit target/run, else newest).
    demo_dir = None
    target = getattr(args, "target", None)
    if target:
        demo_dir = _resolve_target(target, repo_root)
    slug = demo_dir.name if demo_dir is not None else None
    run_dir = None
    if demo_dir is None:
        run_dir = find_run_dir(repo_root, slug=slug, run_ref=getattr(args, "run", None))
        if run_dir is not None:
            slug = slug or run_slug(run_dir)
    if demo_dir is None and slug:
        d = find_demo_dir(slug, repo_root)
        demo_dir = d.resolve() if d else None
    if (demo_dir is None or not Path(demo_dir).is_dir()) and slug:
        cand = Path(repo_root) / "models" / "demos" / slug
        if cand.is_dir():
            demo_dir = cand
    if demo_dir is None or not Path(demo_dir).is_dir():
        print(f"  [run-demo] could not locate a model demo dir (slug={slug}). Pass a target or --run.")
        return 2
    slug = demo_dir.name
    checkout = _checkout_of(Path(demo_dir))

    task_toks = _task_tokens(checkout, demo_dir)
    demo_file, demo_mod = _pick_demo(Path(demo_dir), slug, task_toks)
    if not demo_mod:
        print(f"  [run-demo] {slug} ships no runnable demo under demo/. Nothing to run.")
        return 2
    out_flag, out_default, batch_flag = _discover_flags(demo_file)

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    default_out = Path(os.path.expanduser("~/.cc-dashboard/demo-out")) / f"{slug}-{stamp}"
    out_dir = Path(os.path.expanduser(getattr(args, "out", None) or str(default_out))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = getattr(args, "batch", None)
    plan = {
        "slug": slug,
        "demo": demo_mod,
        "demo_file": str(Path(demo_file).resolve()),
        "checkout": str(checkout),
        "task_tokens": task_toks,
        "out_flag": out_flag,
        "batch_flag": batch_flag,
        "out_dir": str(out_dir),
        "batch": batch,
    }
    if getattr(args, "plan", False):
        print(json.dumps({"ok": True, "plan": plan}, indent=2))
        return 0

    py = str(Path(checkout) / "python_env" / "bin" / "python")
    cmd = [py, "-m", demo_mod]
    if out_flag:
        cmd += [out_flag, str(out_dir)]
    if batch and batch_flag:
        cmd += [batch_flag, str(batch)]
    env = dict(os.environ, TT_METAL_HOME=str(checkout))
    # Run in out_dir so a demo without an out flag still drops its files somewhere we collect.
    cwd = str(checkout)
    print(f"  [run-demo] {slug}: {' '.join(cmd)}  (out={out_dir})", flush=True)
    log_path = out_dir / "run.log"
    rc = None
    out_text = ""
    try:
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(getattr(args, "timeout", 0) or 7200),
            )
        out_text = proc.stdout or ""
        log_path.write_text(out_text)
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        out_text = (e.stdout or "") if isinstance(e.stdout, str) else ""
        log_path.write_text(out_text + "\n[run-demo] TIMEOUT")
        rc = 124

    modality, items = _collect(out_dir, out_text)
    manifest = {
        "ok": rc == 0 and bool(items),
        "rc": rc,
        "slug": slug,
        "demo": demo_mod,
        "modality": modality,
        "batch": batch or len(items),
        "count": len(items),
        "out_dir": str(out_dir),
        "log": str(log_path),
        "items": items,
        "stdout_tail": (out_text or "").strip()[-4000:],
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "items"} | {"items": len(items)}, indent=2))
    print(f"  [run-demo] wrote manifest: {out_dir / 'manifest.json'}")
    return 0 if manifest["ok"] else 1
