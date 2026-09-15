from __future__ import annotations

import importlib.util
import subprocess
import sys


def have_pip() -> bool:
    return importlib.util.find_spec("pip") is not None


def pip_cmd(args: list[str]) -> list[str]:
    if have_pip():
        return [sys.executable, "-m", "pip", *args]
    return ["uv", "pip", args[0], "--python", sys.executable, *args[1:]]


def run_pip(args: list[str], timeout_s: int = 600, check: bool = False):
    return subprocess.run(pip_cmd(args), capture_output=True, text=True, timeout=timeout_s, check=check)


def installer_hint() -> str:
    return "pip install" if have_pip() else "uv pip install"
