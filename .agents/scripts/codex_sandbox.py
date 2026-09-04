"""Check Codex sandbox startup without a model call; print the permitted mode."""

import os
import signal
import subprocess
import sys


# Observed bubblewrap startup failures on restricted Linux hosts/containers.
# Do not treat arbitrary permission, configuration, or application errors as these.
SANDBOX_STARTUP_ERRORS = {
    "bwrap: Failed to make / slave: Permission denied",
    "bwrap: setting up uid map: Permission denied",
    "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted",
    "bwrap: Creating new namespace failed: Operation not permitted",
}


def select_sandbox(timeout=15):
    allow_unsandboxed = os.environ.get("AUTODEBUG_ALLOW_UNSANDBOXED", "0")
    if allow_unsandboxed not in {"0", "1"}:
        raise SystemExit("autodebug: AUTODEBUG_ALLOW_UNSANDBOXED must be 0 or 1")

    # Inherit the caller's cwd, environment and Codex configuration. The shell
    # exercises sandbox initialization, not the target code or a paid model.
    try:
        with subprocess.Popen(
            ["codex", "sandbox", "-c", 'sandbox_mode="workspace-write"', "--", "/bin/sh", "-c", "exit 0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        ) as probe:
            try:
                output, _ = probe.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(probe.pid, signal.SIGKILL)
                probe.communicate()
                raise SystemExit("autodebug: sandbox preflight timed out; no model was started")
    except OSError as error:
        raise SystemExit(f"autodebug: cannot run sandbox preflight: {error}") from error

    if probe.returncode == 0:
        return "workspace-write"

    print(output.rstrip(), file=sys.stderr)
    if not SANDBOX_STARTUP_ERRORS.intersection(line.strip() for line in output.splitlines()):
        raise SystemExit("autodebug: sandbox preflight failed; fix the error before retrying")
    if allow_unsandboxed != "1":
        raise SystemExit(
            "autodebug: sandbox initialization failed; no model was started. "
            "Fix the host sandbox, or have the user/operator authorize this environment "
            "with AUTODEBUG_ALLOW_UNSANDBOXED=1."
        )

    print(
        "autodebug: WARNING: sandbox initialization failed. "
        "AUTODEBUG_ALLOW_UNSANDBOXED=1 permits this fresh session to run with "
        "danger-full-access and no approval prompts. It can access files and the network "
        "with your account's permissions, including mounted/shared data.",
        file=sys.stderr,
    )
    return "danger-full-access"


if __name__ == "__main__":
    print(select_sandbox())
