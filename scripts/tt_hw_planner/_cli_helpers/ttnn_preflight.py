"""Pre-flight: ensure `import ttnn` works before any on-device test runs.

On failure: print a clean diagnosis, auto-rebuild (submodules + build_metal.sh),
re-check, then fall back to the pip-only LLM env-fix agent, else abort with the
runbook. Dormant when ttnn already imports.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def _check_import() -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import ttnn"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return (proc.returncode == 0), (proc.stderr or proc.stdout or "")


def _diagnose(stderr: str) -> str:
    s = stderr.strip()
    last = s.splitlines()[-1] if s else ""
    m = re.search(r"cannot import name '([^']+)' from '(ttnn[^']*)'", last)
    if m:
        sym, mod = m.group(1), m.group(2)
        return (
            f"ttnn's compiled extension is STALE — `{mod}` does not provide "
            f"`{sym}`, but the checked-out source needs it.\n"
            f"    Cause: _ttnn.so was built from OLDER source (git carries source, "
            f"not the compiled binary), or for a different checkout/arch.\n"
            f"    → It must be rebuilt to match this source."
        )
    if "No module named 'ttnn'" in last or "No module named ttnn" in last:
        return (
            "ttnn is NOT installed/built in this Python env.\n"
            "    → tt-metal needs to be built and the venv created for this checkout."
        )
    if "_ttnn" in last and ("undefined symbol" in last or "ImportError" in last):
        return (
            "ttnn's compiled extension failed to load (ABI/build mismatch between "
            "_ttnn.so and the current source/toolchain).\n"
            "    → It must be rebuilt."
        )
    return f"`import ttnn` failed:\n    {last or '(no error text)'}\n    → likely a stale/missing ttnn build."


def _repo_root() -> Path:
    try:
        from ..discovery import REPO_ROOT

        return Path(REPO_ROOT)
    except Exception:
        return Path(os.environ.get("TT_METAL_HOME", str(Path.cwd())))


def _run_streaming(cmd: list, cwd: Path, timeout_s: int) -> int:
    print(f"    $ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
    except FileNotFoundError as exc:
        print(f"    (could not run: {exc})")
        return 127
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write("    | " + line)
            sys.stdout.flush()
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"    (build exceeded {timeout_s}s; killed)")
        return 124


def _attempt_rebuild() -> Tuple[bool, str]:
    root = _repo_root()
    build = root / "build_metal.sh"
    if not build.is_file():
        return False, f"no build_metal.sh at {root} (set TT_METAL_HOME to the tt-metal checkout)"
    print("  [ttnn-preflight] syncing submodules…")
    _run_streaming(["git", "submodule", "update", "--init", "--recursive"], root, timeout_s=1800)
    print("  [ttnn-preflight] rebuilding ttnn (./build_metal.sh) — this can take 10–30 min…")
    rc = _run_streaming(["bash", str(build)], root, timeout_s=5400)
    ok, _ = _check_import()
    if ok:
        return True, "rebuild succeeded; import ttnn now works"
    return False, f"rebuild finished (rc={rc}) but `import ttnn` still fails"


def ensure_ttnn_ready(*, agent_bin: str = "claude", agent_model: str = "sonnet", allow_llm: bool = True) -> bool:
    """Return True if `import ttnn` works (proceed), else False (caller aborts)."""
    ok, err = _check_import()
    if ok:
        return True

    sep = "=" * 78
    print("\n" + sep)
    print("  ✗ PRE-FLIGHT: ttnn is not usable for this checkout")
    print(sep)
    print("  " + _diagnose(err).replace("\n", "\n  "))
    print(sep)

    print("  [ttnn-preflight] attempting automatic rebuild…")
    fixed, note = _attempt_rebuild()
    if fixed:
        print(f"  ✓ [ttnn-preflight] {note} — continuing.")
        return True
    print(f"  [ttnn-preflight] {note}")

    if allow_llm:
        print("\n  [ttnn-preflight] rebuild did not resolve it; consulting the LLM env-fix agent…")
        try:
            if _llm_env_fix(err=err, agent_bin=agent_bin, agent_model=agent_model):
                ok2, _ = _check_import()
                if ok2:
                    print("  ✓ [ttnn-preflight] LLM env-fix resolved the import — continuing.")
                    return True
        except Exception as exc:
            print(
                f"  [ttnn-preflight] LLM env-fix step errored (non-fatal): {type(exc).__name__}: {exc}", file=sys.stderr
            )

    print("\n" + sep)
    print("  ✗ ttnn still cannot be imported — cannot run any on-device PCC test.")
    print("    Fix it manually, then re-run:")
    print("      git submodule update --init --recursive")
    print("      ./build_metal.sh")
    print('      python -c "import ttnn"   # must succeed')
    print("  (Not running iterations: no component can graduate until import ttnn works.)")
    print(sep)
    return False


def _llm_env_fix(*, err: str, agent_bin: str, agent_model: str) -> bool:
    try:
        from .agent import _bringup_cwd, _invoke_agent
        from .env_fix import build_env_fix_prompt, parse_env_fix_verdict
    except Exception:
        return False
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60
        ).stdout
    except Exception:
        freeze = ""
    cwd = _bringup_cwd()
    verdict_path = Path(cwd) / "_handoff" / "ttnn_env_fix.json"
    try:
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    last = (err.strip().splitlines()[-1] if err.strip() else err) or "import ttnn failed"
    prompt = build_env_fix_prompt(
        env_problems=[f"`import ttnn` fails: {last}"],
        installed_packages_summary=freeze[:4000],
        python_version=sys.version.split()[0],
        verdict_path=verdict_path,
    )
    _invoke_agent(
        prompt,
        provider=("claude" if "claude" in agent_bin else "cursor"),
        agent_bin=agent_bin,
        cwd=cwd,
        model=agent_model,
        timeout_s=600,
    )
    proposal = parse_env_fix_verdict(verdict_path)
    if not proposal:
        print("  [ttnn-preflight] LLM proposed no actionable pip fix.")
        return False
    print(f"  [ttnn-preflight] applying LLM-proposed fix: {proposal.pip_command_str()}")
    return subprocess.run([sys.executable, "-m", "pip", "install", *proposal.pip_args]).returncode == 0
