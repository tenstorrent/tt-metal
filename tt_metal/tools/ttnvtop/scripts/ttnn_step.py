#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# ttnn step-debugger. Wraps a configurable set of ttnn ops with a synchronize +
# interactive prompt. Pauses between op invocations so you can:
#   - inspect the chip state in ttnvtop (run it in another terminal)
#   - inspect tensor values via the wrapped op's return
#   - manually advance one op at a time, or run to a named op
#
# Pauses happen on the *host* between Python ops. The chip is fully drained
# (synchronize_device) before each pause, so what ttnvtop shows during the
# pause is the steady-state right after the just-completed op.
#
# Example use:
#
#   import ttnn
#   from ttnvtop.scripts.ttnn_step import Stepper
#
#   device = ttnn.open_device(device_id=0)
#   step = Stepper(device, watch=["matmul", "softmax", "rms_norm", "linear"])
#   step.install()
#   try:
#       my_decode_loop(device)
#   finally:
#       step.uninstall()
#       ttnn.close_device(device)
#
# Once paused, the prompt accepts:
#   <Enter> or s   — step to next op
#   c              — continue (no more pauses unless you press Ctrl-C)
#   c <op-name>    — continue until next call to ttnn.<op-name>
#   n <count>      — run <count> more ops without pausing, then break
#   l              — list of recent ops and their elapsed time
#   q              — quit (raises SystemExit; calls synchronize first)

import functools
import signal
import sys
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple

try:
    import ttnn
except ImportError as e:
    print(f"[ttnn_step] ttnn not importable: {e}", file=sys.stderr)
    raise


# ───────────────────────────────────────────────────────────────────────────
# History entry
# ───────────────────────────────────────────────────────────────────────────
class _Call:
    __slots__ = ("name", "elapsed_us", "tag")

    def __init__(self, name: str, elapsed_us: int, tag: str = ""):
        self.name = name
        self.elapsed_us = elapsed_us
        self.tag = tag


def _shape_summary(t: Any) -> str:
    """Best-effort short string for a ttnn tensor / Python obj."""
    try:
        # ttnn.Tensor has .shape and .dtype attributes.
        s = list(t.shape)
        d = str(getattr(t, "dtype", "")).split(".")[-1]
        return f"{tuple(s)}{':' + d if d else ''}"
    except Exception:
        return type(t).__name__


def _arg_summary(args: Tuple[Any, ...], kwargs: dict) -> str:
    parts: List[str] = []
    for a in args[:3]:
        parts.append(_shape_summary(a))
    if len(args) > 3:
        parts.append(f"+{len(args) - 3} more")
    for k in list(kwargs)[:3]:
        parts.append(f"{k}={_shape_summary(kwargs[k])}")
    return ", ".join(parts)


# ───────────────────────────────────────────────────────────────────────────
# Stepper
# ───────────────────────────────────────────────────────────────────────────
class Stepper:
    DEFAULT_WATCH = (
        "matmul",
        "linear",
        "softmax",
        "rms_norm",
        "layer_norm",
        "exp",
        "gelu",
        "silu",
        "add",
        "mul",
        "sub",
        "div",
        "concat",
        "reshape",
        "transpose",
        "embedding",
    )

    def __init__(
        self,
        device,
        watch: Optional[Iterable[str]] = None,
        history_size: int = 50,
    ):
        self.device = device
        self.watch = set(watch) if watch is not None else set(self.DEFAULT_WATCH)
        self.history: List[_Call] = []
        self.history_size = history_size
        self._originals: dict = {}
        self._installed = False
        # Run-control state
        self._continue = False  # run forever, no pauses
        self._continue_until: Optional[str] = None  # break when next op == this name
        self._skip = 0  # silently skip N pauses
        # Allow Ctrl-C during continue to drop back into prompt.
        self._interrupted = False

    # ── install / uninstall ────────────────────────────────────────────
    def install(self):
        if self._installed:
            return
        for name in self.watch:
            fn = getattr(ttnn, name, None)
            if not callable(fn):
                continue
            self._originals[name] = fn
            setattr(ttnn, name, self._wrap(name, fn))
        self._installed = True
        signal.signal(signal.SIGINT, self._on_sigint)
        print(
            f"[ttnn_step] watching {len(self._originals)} ops: " f"{', '.join(sorted(self._originals))}",
            file=sys.stderr,
        )
        print("[ttnn_step] press <Enter> after each call; type 'h' for help", file=sys.stderr)

    def uninstall(self):
        for name, fn in self._originals.items():
            setattr(ttnn, name, fn)
        self._originals.clear()
        self._installed = False

    # ── wrapper ────────────────────────────────────────────────────────
    def _wrap(self, name: str, fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            arg_str = _arg_summary(args, kwargs)
            t0 = time.monotonic_ns()
            result = fn(*args, **kwargs)
            # Drain so what we see in ttnvtop reflects this op's effect.
            try:
                ttnn.synchronize_device(self.device)
            except Exception:
                pass
            elapsed_us = (time.monotonic_ns() - t0) // 1000
            self._record(name, elapsed_us, arg_str)
            self._maybe_pause(name, arg_str, elapsed_us, result)
            return result

        return wrapper

    def _record(self, name: str, elapsed_us: int, tag: str):
        self.history.append(_Call(name, elapsed_us, tag))
        if len(self.history) > self.history_size:
            self.history.pop(0)

    # ── pause / prompt ────────────────────────────────────────────────
    def _maybe_pause(self, name: str, arg_str: str, elapsed_us: int, result):
        # Skip semantics
        if self._interrupted:
            self._continue = False
            self._continue_until = None
            self._skip = 0
            self._interrupted = False
            print("[ttnn_step] interrupt — dropping back into stepper", file=sys.stderr)
        elif self._continue and self._continue_until is None:
            return
        elif self._continue_until is not None:
            if name == self._continue_until:
                self._continue = False
                self._continue_until = None
            else:
                return
        elif self._skip > 0:
            self._skip -= 1
            return

        # Pause prompt
        while True:
            try:
                line = input(
                    f"\n[ttnn_step] ttnn.{name}({arg_str})  →  {_shape_summary(result)}  "
                    f"({elapsed_us / 1000:.2f} ms) — (s/c/n/l/h/q)> "
                )
            except EOFError:
                self._continue = True
                return
            cmd = line.strip()
            if cmd == "" or cmd == "s":
                return
            if cmd == "c":
                self._continue = True
                return
            if cmd.startswith("c "):
                self._continue = True
                self._continue_until = cmd[2:].strip()
                print(f"[ttnn_step] running until next ttnn.{self._continue_until} ...")
                return
            if cmd.startswith("n "):
                try:
                    self._skip = int(cmd[2:].strip())
                    print(f"[ttnn_step] running {self._skip} ops, then break ...")
                    return
                except ValueError:
                    print("[ttnn_step] usage: n <count>")
                    continue
            if cmd == "l":
                self._dump_history()
                continue
            if cmd == "h" or cmd == "?":
                print(
                    "[ttnn_step] commands:\n"
                    "  <Enter> | s     — step one op\n"
                    "  c               — continue (no more pauses)\n"
                    "  c <op-name>     — continue until next ttnn.<op-name>\n"
                    "  n <count>       — run <count> ops then break\n"
                    "  l               — list recent ops with timings\n"
                    "  q               — quit (synchronize + exit)\n"
                    "  h | ?           — this help"
                )
                continue
            if cmd == "q":
                try:
                    ttnn.synchronize_device(self.device)
                except Exception:
                    pass
                raise SystemExit("[ttnn_step] quit")
            print(f"[ttnn_step] unknown command: {cmd!r}  (try 'h')")

    def _dump_history(self):
        print("\n[ttnn_step] recent ops (most recent last):")
        for c in self.history:
            print(f"  ttnn.{c.name:<14} {c.tag:<40} {c.elapsed_us / 1000:8.2f} ms")
        print()

    def _on_sigint(self, *_):
        # Set a flag so the next pause point drops back into prompt instead of
        # continuing. Don't raise — we don't want to abort mid-op.
        self._interrupted = True
        # Reinstall so further Ctrl-C still works.
        signal.signal(signal.SIGINT, self._on_sigint)


# ───────────────────────────────────────────────────────────────────────────
# Convenience CLI: step-debug a python script
# ───────────────────────────────────────────────────────────────────────────
def _cli():
    """Allow `python -m ttnn_step my_workload.py` to run a script under stepping.
    Note: this requires ttnn.open_device to be callable from the wrapped script.
    """
    import argparse
    import runpy

    ap = argparse.ArgumentParser()
    ap.add_argument("script", help="Python file to execute under the stepper")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument(
        "--watch",
        nargs="+",
        default=None,
        help="ttnn op names to wrap. Default: a built-in list of common ops.",
    )
    args = ap.parse_args()

    device = ttnn.open_device(device_id=args.device_id)
    step = Stepper(device, watch=args.watch)
    step.install()
    try:
        runpy.run_path(args.script, run_name="__main__")
    finally:
        step.uninstall()
        try:
            ttnn.close_device(device)
        except Exception:
            pass


if __name__ == "__main__":
    _cli()
