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


# The install command, as config: set it to the host's own ("<command> <package>" is run). Unset, the
# first package manager of _PACKAGE_MANAGERS present on PATH is used.
SYSTEM_INSTALL_CMD_ENV = "TT_HW_PLANNER_SYSTEM_INSTALL_CMD"
# Known package managers, in order of preference: (executable, non-interactive install arguments).
_PACKAGE_MANAGERS = (
    ("apt-get", ("install", "-y", "-q")),
    ("dnf", ("install", "-y")),
    ("yum", ("install", "-y")),
    ("zypper", ("--non-interactive", "install")),
)
# How a non-root process elevates without ever blocking on a password prompt.
_ELEVATE = ("sudo", "-n")
_NONINTERACTIVE_ENV = {"DEBIAN_FRONTEND": "noninteractive"}

_SYSTEM_TOOL_TRIED: dict = {}


def _install_command(package: str) -> list | None:
    """The command that installs `package` on this host, or None when there is no way to."""
    configured = os.environ.get(SYSTEM_INSTALL_CMD_ENV, "").split()
    if configured:
        cmd = [*configured, package]
    else:
        found = next(((shutil.which(pm), args) for pm, args in _PACKAGE_MANAGERS if shutil.which(pm)), None)
        if found is None:
            return None
        cmd = [found[0], *found[1], package]
    if os.geteuid() != 0:
        elevate = shutil.which(_ELEVATE[0])
        if not elevate:
            return None
        cmd = [elevate, *_ELEVATE[1:], *cmd]
    return cmd


def ensure_system_tool(binary: str, package: str | None = None, timeout_s: int = 600) -> bool:
    """True when `binary` is on PATH, installing its distro package first when it is missing.

    The system-package counterpart of the tt-lang auto-install: a host tool the run depends on is
    installed rather than reported. NON-INTERACTIVE only (see _ELEVATE, _NONINTERACTIVE_ENV), so it
    never waits on a password prompt; a host with no known package manager (and no
    SYSTEM_INSTALL_CMD_ENV) or no passwordless elevation just gets False and the caller carries on.
    Tried at most once per process per binary, so a host where the install cannot work is not asked
    again on every reset. NO_SYSTEM_INSTALL_ENV=1 turns it off.
    """
    if shutil.which(binary):
        return True
    if os.environ.get(NO_SYSTEM_INSTALL_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    if binary in _SYSTEM_TOOL_TRIED:
        return _SYSTEM_TOOL_TRIED[binary]
    _SYSTEM_TOOL_TRIED[binary] = False
    cmd = _install_command(package or binary)
    if cmd is None:
        return False
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, **_NONINTERACTIVE_ENV},
        )
    except Exception:  # noqa: BLE001 -- an install that cannot run is reported as absent, never raised
        return False
    _SYSTEM_TOOL_TRIED[binary] = shutil.which(binary) is not None
    return _SYSTEM_TOOL_TRIED[binary]
