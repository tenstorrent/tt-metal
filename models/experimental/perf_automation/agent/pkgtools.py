from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys

# Set to 1 to forbid ensure_system_tool from installing anything (it then only reports presence).
NO_SYSTEM_INSTALL_ENV = "TT_HW_PLANNER_NO_SYSTEM_INSTALL"


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


_SYSTEM_TOOL_TRIED: dict = {}


def ensure_system_tool(binary: str, package: str | None = None, timeout_s: int = 600) -> bool:
    """True when `binary` is on PATH, installing its distro package first when it is missing.

    The system-package counterpart of the tt-lang auto-install: a host tool the run depends on is
    installed rather than reported. NON-INTERACTIVE only (`sudo -n`, DEBIAN_FRONTEND=noninteractive),
    so it never waits on a password prompt; a host without apt-get or passwordless sudo just gets
    False and the caller carries on. Tried at most once per process per binary, so a host where the
    install cannot work is not asked again on every reset. NO_SYSTEM_INSTALL_ENV=1 turns it off.
    """
    if shutil.which(binary):
        return True
    if os.environ.get(NO_SYSTEM_INSTALL_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if binary in _SYSTEM_TOOL_TRIED:
        return _SYSTEM_TOOL_TRIED[binary]
    _SYSTEM_TOOL_TRIED[binary] = False
    apt = shutil.which("apt-get")
    if not apt:
        return False
    cmd = [apt, "install", "-y", "-q", package or binary]
    if os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if not sudo:
            return False
        cmd = [sudo, "-n", *cmd]
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"},
        )
    except Exception:  # noqa: BLE001 -- an install that cannot run is reported as absent, never raised
        return False
    _SYSTEM_TOOL_TRIED[binary] = shutil.which(binary) is not None
    return _SYSTEM_TOOL_TRIED[binary]
