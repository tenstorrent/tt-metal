import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Self-wrapping clean-screen + full-log layer.
#
# Goal: running the tool the normal way —
#     python -m scripts.tt_hw_planner emit-e2e <model>
# — gives a CLEAN terminal AND a single COMPLETE log file, with no separate
# wrapper command. It does this by re-running itself as a child with full
# output, tee-ing the ENTIRE stream to one log file, and printing only a
# filtered (clean) view on the terminal.
#
# Safety:
#   * Only activates for interactive (TTY) runs of the noisy user-facing
#     commands — redirected/piped/CI runs are untouched (so `> run.log 2>&1`
#     still works exactly as before).
#   * `_TT_HW_PLANNER_WRAPPED=1` stops the child (and the tool's own worktree
#     re-exec) from wrapping again.
#   * Opt out entirely with `TT_HW_PLANNER_NO_WRAP=1`.
#   * If wrap setup fails it falls through to a normal run — never double-runs.
# ─────────────────────────────────────────────────────────────────────────────

# emit-e2e is intentionally NOT here: its output is free-form agent narration
# that a regex filter can't clean. emit-e2e does its own clean-screen + single
# full-log internally (see commands/emit_e2e.py). This wrapper is only for the
# marker-tagged-noise commands.
_WRAPPABLE_COMMANDS = {"auto-up", "up", "bringup", "promote"}


def _should_wrap() -> bool:
    if os.environ.get("_TT_HW_PLANNER_WRAPPED"):
        return False
    if os.environ.get("TT_HW_PLANNER_NO_WRAP", "") not in ("", "0", "false", "False"):
        return False
    cmd = next((a for a in sys.argv[1:] if not a.startswith("-")), "")
    if cmd not in _WRAPPABLE_COMMANDS:
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _run_wrapped() -> int:
    """Re-run this command as a child with full output; tee the complete
    stream to one log file and show a clean filtered view on the terminal."""
    import re
    import subprocess

    argv = sys.argv[1:]
    positionals = [a for a in argv if not a.startswith("-")]
    model = positionals[1] if len(positionals) > 1 else "run"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    log_path = os.path.join("generated", f"{safe}_full.log")
    try:
        os.makedirs("generated", exist_ok=True)
    except Exception:
        pass

    # Lines hidden from the SCREEN (the file keeps everything, unfiltered).
    noise = re.compile(
        r"\[auto:"
        r"|Loading checkpoint shards"
        r"|^\d{4}-\d\d-\d\d "
        r"|\| (DEBUG|INFO|TRACE|debug|info|trace) "
        r"|^Config\{"
        r"|Initial ttnn\.CONFIG"
        r"|\[prompt-block\]"
        r"|\[exemplar\]"
        r"|^\[agent\] "
    )

    child_env = dict(os.environ)
    child_env["_TT_HW_PLANNER_WRAPPED"] = "1"
    child_env["TT_HW_PLANNER_VERBOSE"] = "1"  # full output so the log is complete

    # Spawn BEFORE risking anything else; if Popen fails the caller falls
    # through to a normal in-process run (no double execution).
    proc = subprocess.Popen(
        [sys.executable, "-m", "scripts.tt_hw_planner", *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=sys.stdin,
        text=True,
        bufsize=1,
        env=child_env,
    )

    logf = None
    try:
        logf = open(log_path, "w", buffering=1, errors="ignore")
    except Exception:
        logf = None

    sys.stdout.write(f"  full log → {log_path}\n")
    sys.stdout.flush()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if logf is not None:
                try:
                    logf.write(line)
                except Exception:
                    pass
            if not noise.search(line):
                sys.stdout.write(line)
                sys.stdout.flush()
    except Exception:
        pass
    finally:
        if logf is not None:
            try:
                logf.close()
            except Exception:
                pass
    return proc.wait()


def _main() -> int:
    if _should_wrap():
        try:
            return _run_wrapped()
        except Exception:
            # Setup failed before the child ran — fall through to a normal run.
            pass
    from .cli import main

    return main()


if __name__ == "__main__":
    sys.exit(_main())
