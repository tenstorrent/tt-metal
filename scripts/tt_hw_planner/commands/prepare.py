from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def cmd_prepare(args) -> int:
    from ..cli import (
        BringupError,
        REPO_ROOT,
        _download_model_snapshot,
        _parse_mesh,
        prepare_bringup,
        render_bringup_json,
        render_bringup_script,
        render_bringup_text,
    )

    mesh_override: Optional[Tuple[int, int]] = None
    if args.mesh:
        try:
            mesh_override = _parse_mesh(args.mesh)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    try:
        plan = prepare_bringup(
            model_id=args.model_id,
            box_override=args.box,
            mesh_override=mesh_override,
            dtype_override=args.dtype,
            batch=args.batch,
            max_seq_len=args.max_seq_len,
            max_generated_tokens=args.max_generated_tokens,
            accuracy=args.accuracy,
            trace=not args.no_trace,
            paged_attention=not args.no_paged_attention,
            instruct=not args.no_instruct,
        )
    except BringupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(render_bringup_json(plan))
    elif args.format == "script":
        print(render_bringup_script(plan))
    else:
        print(render_bringup_text(plan))

    if args.write_script:
        path = Path(args.write_script).expanduser().resolve()
        _SYSTEM_PREFIXES = ("/etc", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc", "/dev")
        s = str(path)
        if any(s == d or s.startswith(d + "/") for d in _SYSTEM_PREFIXES):
            print(f"\nERROR: refusing to write bring-up script to system path: {path}", file=sys.stderr)
            return 1
        path.write_text(render_bringup_script(plan))
        path.chmod(0o700)
        print(f"\nWrote bring-up script: {path}", file=sys.stderr)

    if args.execute:
        if plan.invocation is None:
            print("\nERROR: no executable command (see blockers above).", file=sys.stderr)
            return 2
        if args.strict and plan.compat_overall not in {"ALREADY SUPPORTED", "READY"}:
            print(
                f"\nERROR: --strict refused — compat verdict is '{plan.compat_overall}'. "
                "Remove --strict to execute despite PARTIAL blocks.",
                file=sys.stderr,
            )
            return 2
        import subprocess

        full_env = {**os.environ, **plan.invocation.env}
        if getattr(args, "download_first", False):
            download_target = plan.invocation.env.get("HF_MODEL", args.model_id)
            print(f"\nPre-downloading Hugging Face weights for {download_target} …", file=sys.stderr)
            try:
                _download_model_snapshot(download_target)
            except Exception as exc:
                print(f"\nERROR: pre-download failed: {exc}", file=sys.stderr)
                return 2
        _bringup_cwd_env = os.environ.get("TT_HW_PLANNER_BRINGUP_CWD")
        bringup_cwd = Path(_bringup_cwd_env) if _bringup_cwd_env else REPO_ROOT
        print(f"\nExecuting in {bringup_cwd} …", file=sys.stderr)

        _cap_path = globals().get("_pytest_capture_sink", None)

        from ..bringup import PytestInvocation as _PI

        _per_test = _PI.per_test_timeout_s()
        _wall_default = max(2700, int(_per_test * 1.5))
        try:
            pytest_timeout_s = int(os.environ.get("TT_PLANNER_PYTEST_TIMEOUT_S", str(_wall_default)))
        except (TypeError, ValueError):
            pytest_timeout_s = _wall_default
        if pytest_timeout_s < _per_test + 60:
            print(
                f"WARNING: TT_PLANNER_PYTEST_TIMEOUT_S={pytest_timeout_s}s is "
                f"smaller than --timeout={_per_test}s + 60s; pytest will be "
                f"killed before its own per-test timeout can fire. Either "
                f"unset TT_PLANNER_PYTEST_TIMEOUT_S or raise it to "
                f">={_per_test + 60}s.",
                file=sys.stderr,
            )
        import signal as _signal

        if _cap_path:
            import threading

            _cap_fh = open(_cap_path, "w", buffering=1)
            proc = subprocess.Popen(
                plan.invocation.argv(),
                cwd=bringup_cwd,
                env=full_env,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )

            def _pump():
                try:
                    assert proc.stdout is not None
                    for _line in proc.stdout:
                        sys.stdout.write(_line)
                        sys.stdout.flush()
                        _cap_fh.write(_line)
                except Exception:
                    pass

            _pump_t = threading.Thread(target=_pump, daemon=True)
            _pump_t.start()
        else:
            _cap_fh = None
            _pump_t = None
            proc = subprocess.Popen(
                plan.invocation.argv(),
                cwd=bringup_cwd,
                env=full_env,
                start_new_session=True,
            )
        try:
            _rc_pytest = proc.wait(timeout=pytest_timeout_s)
            if _pump_t is not None:
                _pump_t.join(timeout=5)
            if _cap_fh is not None:
                try:
                    _cap_fh.flush()
                    _cap_fh.close()
                except Exception:
                    pass
            return _rc_pytest
        except subprocess.TimeoutExpired:
            print(
                f"\nERROR: pytest exceeded wall-clock budget of {pytest_timeout_s}s "
                f"(set TT_PLANNER_PYTEST_TIMEOUT_S to override). A ttnn op likely "
                f"deadlocked on the device — pytest's own signal-based timeout "
                f"cannot interrupt C++ device calls. Killing the process tree.\n"
                f"\n"
                f"RECOVERY:\n"
                f"  - If a NEW component's native ttnn stub is the culprit, restore\n"
                f"    its CPU fallback so the baseline can converge:\n"
                f"      python -m scripts.tt_hw_planner up <model> --force-fallback ...\n"
                f"  - Or re-run with LLM credentials + --auto so the loop can fix\n"
                f"    the broken stub iteratively:\n"
                f"      export ANTHROPIC_API_KEY=...\n"
                f"      python -m scripts.tt_hw_planner up <model> --auto --auto-agent claude ...",
                file=sys.stderr,
            )
            try:
                os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                    proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            if _pump_t is not None:
                _pump_t.join(timeout=5)
            if _cap_fh is not None:
                try:
                    _cap_fh.flush()
                    _cap_fh.close()
                except Exception:
                    pass
            return 124

    if plan.invocation is None:
        return 2
    return 0
