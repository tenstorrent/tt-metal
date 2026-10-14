#!/usr/bin/env python3
"""Install/manage five-minute JIT source-update polling for this checkout."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import update_llk_server as updater

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
UNIT = "tt-jit-update-" + hashlib.sha256(os.fsencode(ROOT)).hexdigest()[:12]


def state_dir():
    path = Path(updater.git(ROOT, "rev-parse", "--path-format=absolute", "--git-path", "jit-server-watchdog"))
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", opener=lambda name, flags: os.open(name, flags, 0o600)) as output:
        json.dump(data, output)
    temporary.replace(path)


def systemctl(*args, check=True):
    return subprocess.run(["systemctl", "--user", *args], check=check, text=True, capture_output=True, timeout=15)


def has_systemd():
    try:
        return systemctl("show-environment", check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def worker_pid(state):
    try:
        record = json.loads((state / "worker.json").read_text())
        proc = Path("/proc") / str(record["pid"])
        if updater.alive(proc, record["identity"]):
            return record["pid"]
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def tick(state):
    config = json.loads((state / "config.json").read_text())
    env = dict(os.environ, **config["environment"])
    # A scheduled check must fail promptly rather than wait for a password prompt.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_SSH_COMMAND"] = env.get("GIT_SSH_COMMAND", "ssh") + " -oBatchMode=yes -oConnectTimeout=15"
    with open(state / "watchdog.log", "ab", buffering=0) as log:
        stamp = datetime.now(timezone.utc).isoformat()
        log.write(f"\n[{stamp}] Checking llk_helper_library\n".encode())
        result = subprocess.run(
            [config["python"], str(ROOT / "scripts/update_llk_server.py"), "--", *config["build_args"]],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
        log.write(f"[{datetime.now(timezone.utc).isoformat()}] Update exit status: {result.returncode}\n".encode())
    write_json(
        state / "last-check.json", {"finished": datetime.now(timezone.utc).isoformat(), "exit_code": result.returncode}
    )
    return result.returncode


def run_loop(state):
    # Separate from the updater lock: one poller per checkout, one build at a time.
    with open(state / "worker.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        stopping = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stopping.set())
        identity = (Path("/proc") / str(os.getpid()) / "stat").read_text().rsplit(")", 1)[1].split()[19]
        write_json(state / "worker.json", {"pid": os.getpid(), "identity": identity})
        try:
            while not stopping.is_set():
                config = json.loads((state / "config.json").read_text())
                # Wait before the first check too, allowing setup to finish.
                if stopping.wait(config["interval"]):
                    break
                try:
                    tick(state)
                except Exception as error:
                    with open(state / "watchdog.log", "a") as log:
                        log.write(f"[{datetime.now(timezone.utc).isoformat()}] Check failed: {error}\n")
        finally:
            (state / "worker.json").unlink(missing_ok=True)


def unit_quote(value):
    # systemd has its own quoting, percent specifiers, and ExecStart dollar expansion.
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def install(state, interval, build_args):
    updater.validate_build_args(build_args)
    if updater.git(ROOT, "branch", "--show-current") != "llk_helper_library":
        raise RuntimeError("Automatic updates require branch llk_helper_library")
    server = updater.find_server(ROOT)
    server[4].close()
    server[5].close()
    existing = state / "config.json"
    if existing.exists():
        config = json.loads(existing.read_text())
        if worker_pid(state) or (
            config["backend"] == "systemd"
            and has_systemd()
            and systemctl("is-active", UNIT + ".timer", check=False).returncode == 0
        ):
            print("Watchdog already installed; stop it before changing its settings.")
            status(state)
            return
    backend = "systemd" if has_systemd() else "loop"
    # Keep build/Git settings, without persisting the shell's entire environment.
    keys = (
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "SSH_AUTH_SOCK",
        "GIT_SSH_COMMAND",
        "TT_METAL_HOME",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "TMPDIR",
    )
    environment = {k: v for k, v in os.environ.items() if k in keys or k.startswith(("CCACHE_", "CMAKE_", "SFPI_"))}
    write_json(
        existing,
        {
            "backend": backend,
            "interval": interval,
            "build_args": build_args,
            "python": sys.executable,
            "environment": environment,
        },
    )
    if backend == "systemd":
        units = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "systemd/user"
        units.mkdir(parents=True, exist_ok=True)
        service = f"""[Unit]
Description=Update LLK sources and restart the JIT compile server

[Service]
Type=oneshot
ExecStart={unit_quote(sys.executable)} {unit_quote(SELF)} check
TimeoutStartSec=infinity
# The replacement JIT server must outlive this oneshot updater.
KillMode=process
"""
        timer = f"""[Unit]
Description=Check for LLK source updates every {interval} seconds

[Timer]
OnActiveSec={interval}s
OnUnitInactiveSec={interval}s
AccuracySec=1s
Unit={UNIT}.service

[Install]
WantedBy=timers.target
"""
        (units / (UNIT + ".service")).write_text(service)
        (units / (UNIT + ".timer")).write_text(timer)
        systemctl("daemon-reload")
        systemctl("enable", "--now", UNIT + ".timer")
        print(f"Installed user timer {UNIT}.timer. User-manager lifetime controls persistence across logout/reboot.")
    else:
        with open(state / "watchdog.log", "ab") as log:
            child = subprocess.Popen(
                [sys.executable, str(SELF), "run"],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        deadline = time.monotonic() + 5
        while not worker_pid(state):
            if child.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(f'Watchdog failed to start; see {state / "watchdog.log"}')
            time.sleep(0.05)
        print("No user systemd available; started a detached polling loop. Rerun setup after container restart.")
    status(state)


def status(state):
    config_file = state / "config.json"
    if not config_file.exists():
        print("Watchdog is not installed.")
        return
    config = json.loads(config_file.read_text())
    print(f"Backend: {config['backend']}; interval: {config['interval']} seconds")
    if config["backend"] == "systemd":
        result = systemctl("is-active", UNIT + ".timer", check=False)
        print(f"Timer: {UNIT}.timer ({result.stdout.strip()})")
    else:
        pid = worker_pid(state)
        print(f"Worker PID: {pid}" if pid else "Worker is stopped.")
    print(f'Log: {state / "watchdog.log"}')
    if (state / "last-check.json").exists():
        print("Last check: " + (state / "last-check.json").read_text())


def stop(state):
    if not (state / "config.json").exists():
        print("Watchdog is not installed.")
        return
    config = json.loads((state / "config.json").read_text())
    if config["backend"] == "systemd":
        systemctl("disable", "--now", UNIT + ".timer")
    else:
        pid = worker_pid(state)
        if pid:
            os.kill(pid, signal.SIGTERM)
    print("Polling stopped; any current update is allowed to finish. The JIT server is left running.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "status", "stop", "run", "check"))
    parser.add_argument("--interval", type=int, default=300)
    # Keep build options separate from watchdog options.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    options = parser.parse_args(argv[:split])
    build_args = argv[split + 1 :]
    if options.interval < 1:
        parser.error("--interval must be positive")
    if build_args and options.command != "install":
        parser.error("Build arguments are only accepted by install")
    state = state_dir()
    # Serialize installation and management without blocking a running check.
    if options.command in ("install", "stop"):
        with open(state / "manage.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if options.command == "install":
                install(state, options.interval, build_args)
            else:
                stop(state)
    elif options.command == "run":
        run_loop(state)
    elif options.command == "check":
        return tick(state)
    else:
        status(state)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
