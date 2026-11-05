from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

_PINNED_FIRST = "1.0.1"


def ttl_importable() -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec("ttl") is not None


def ttnn_version() -> str | None:
    try:
        import ttnn

        return getattr(ttnn, "__version__", None)
    except Exception:
        return None


def _pip(args: list[str], timeout_s: int = 600) -> bool:
    from .pkgtools import run_pip

    try:
        return run_pip(args, timeout_s=timeout_s).returncode == 0
    except Exception:
        return False


def _install(version: str) -> bool:
    return _pip(["install", f"tt-lang=={version}", "--no-deps", "--quiet"])


def _uninstall() -> bool:
    return _pip(["uninstall", "-y", "tt-lang", "--quiet"], timeout_s=120)


def _available_versions() -> list[str]:
    from .pkgtools import pip_cmd

    try:
        r = subprocess.run(
            pip_cmd(["index", "versions", "tt-lang"]),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return []
    m = re.search(r"[Aa]vailable versions:\s*([^\n]+)", (r.stdout or "") + (r.stderr or ""))
    if not m:
        return []
    return [v.strip() for v in m.group(1).split(",") if v.strip()]


def _candidates(version_lister) -> list[str]:
    listed = version_lister() if version_lister else _available_versions()
    ordered, seen = [], set()
    for v in [_PINNED_FIRST, *listed]:
        if v and v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _cache_path(cache_dir) -> Path:
    return Path(cache_dir) / ".ttlang_resolved.json"


def _cache_get(cache_dir, ttnn_v):
    if not cache_dir or not ttnn_v:
        return None
    try:
        return json.loads(_cache_path(cache_dir).read_text()).get(ttnn_v)
    except Exception:
        return None


def _cache_put(cache_dir, ttnn_v, version):
    if not cache_dir or not ttnn_v:
        return
    try:
        f = _cache_path(cache_dir)
        d = json.loads(f.read_text()) if f.exists() else {}
        d[ttnn_v] = version
        f.write_text(json.dumps(d))
    except Exception:
        pass


def ensure_ttl(installer=None, uninstaller=None, version_lister=None, cache_dir=None) -> dict:
    if ttl_importable():
        return {"available": True, "version": "preinstalled", "action": "none"}
    inst = installer or _install
    uninst = uninstaller or _uninstall
    ttnn_v = ttnn_version()
    cached = _cache_get(cache_dir, ttnn_v)
    order, seen, tried = [], set(), []
    for v in ([cached] if cached else []) + _candidates(version_lister):
        if v and v not in seen:
            seen.add(v)
            order.append(v)
    for v in order:
        tried.append(v)
        if not inst(v):
            continue
        if ttl_importable() and ttnn_version() == ttnn_v:
            _cache_put(cache_dir, ttnn_v, v)
            return {"available": True, "version": v, "action": "installed"}
        uninst()
    return {"available": False, "version": None, "action": "failed", "tried": tried}


def main() -> int:
    """Install tt-lang into THIS interpreter's env, the way the optimize run does (same resolver,
    same cache), so a provisioning script can call one entry point instead of a bare `pip install`
    that, in a pip-less uv venv, silently targets the system Python instead.

    Never fatal to the caller's intent: tt-lang only feeds optimize's tt-lang kernel rung, so a
    miss is reported (with this interpreter's Python, the usual reason) and exits 1 for the caller
    to treat as a warning."""
    import sys

    result = ensure_ttl(cache_dir=str(Path(__file__).resolve().parent.parent))
    py = "%d.%d" % sys.version_info[:2]
    if result.get("available"):
        print(f"tt-lang available ({result.get('version')}, {result.get('action')}) for Python {py}")
        return 0
    print(
        f"tt-lang could not be installed for this env's Python {py} (tried: {result.get('tried') or 'none'}). "
        "Usually no tt-lang build exists for this Python version. Only optimize's tt-lang kernel rung "
        "needs it; bring-up and the other optimize rungs are unaffected."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
