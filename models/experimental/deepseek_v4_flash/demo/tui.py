# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Live split-screen console for ``demo/server.py``.

A resident server produces two very different kinds of output: a *state* that is only
interesting as its latest value (who is connected, how full the KV pool is, how fast
each user is decoding) and a *log* that is only interesting as a history. Printing both
to one stream makes each harder to read, so this splits them:

* a **status header** with the served model, uptime, active turns, throughput and the
  KV block pool, repainted a few times a second;
* a **per-user table**, one row per KV-cache slot: its ``user`` key, tokens used of its
  context, and the phase and tok/s of any turn it is running;
* the **log**, timestamped, with the time in a narrow left column and the message on the
  right, scrolling under the status.

Keys, while it runs: ``d`` toggles debug lines, ``p`` pauses scrolling (the status keeps
updating), ``c`` clears the log, ``q`` quits the server.

Everything degrades gracefully: without ``rich`` installed, or when stdout is not a
terminal (a pipe, ``nohup``, a CI log), :func:`console` yields ``None`` and the server
just logs to stderr as usual, so nothing here can stop it from serving.
"""

from __future__ import annotations

import contextlib
import queue
import shutil
import sys
import threading
import time
from collections import deque
from datetime import datetime

_LEVEL_STYLE = {
    "TRACE": "dim",
    "DEBUG": "cyan",
    "INFO": "white",
    "SUCCESS": "green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}
# Square brackets are rich markup, so the key hints spell the keys out instead.
_HELP = "keys: d debug · p pause · c clear · q quit"


def available() -> tuple[bool, str]:
    """Whether a live console can be drawn, and why not when it cannot."""
    try:
        import rich  # noqa: F401
    except ImportError:
        return False, "rich is not installed (pip install rich)"
    if not sys.stdout.isatty():
        return False, "stdout is not a terminal"
    return True, ""


class ServerConsole:
    """The live view: a loguru sink plus a repainting status pane.

    The server's threads only ever call :meth:`write` (via loguru), which appends to a
    bounded deque; a single painter thread turns that plus ``stats()`` into frames. So no
    HTTP thread and, more importantly, no decode step ever blocks on the terminal.
    """

    def __init__(self, stats, on_quit=None, max_lines: int = 2000, fps: float = 6.0, debug: bool = False):
        from rich.console import Console

        self.stats = stats
        self.on_quit = on_quit
        self.debug = debug
        self.fps = fps
        self.console = Console(highlight=False, soft_wrap=False)
        self._lines: deque[tuple[str, str, str]] = deque(maxlen=max_lines)  # (time, level, message)
        self._incoming: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.paused = False
        self._stop = threading.Event()
        self._painter = threading.Thread(target=self._paint_loop, name="tui-painter", daemon=True)
        self._keys = threading.Thread(target=self._key_loop, name="tui-keys", daemon=True)
        self._dropped = 0

    # -- loguru sink ------------------------------------------------------------ #
    def write(self, message) -> None:
        """Loguru sink: queue one record. Never blocks and never raises, because this
        runs on whichever thread logged -- including the decode scheduler."""
        try:
            record = message.record
            stamp = record["time"].strftime("%H:%M:%S.%f")[:-3]
            self._incoming.put_nowait((stamp, record["level"].name, record["message"]))
        except Exception:  # noqa: BLE001 - logging must never take the server down
            self._dropped += 1

    def _drain(self) -> None:
        while True:
            try:
                stamp, level, text = self._incoming.get_nowait()
            except queue.Empty:
                return
            if level == "DEBUG" and not self.debug:
                continue
            for i, line in enumerate(text.splitlines() or [""]):
                self._lines.append((stamp if i == 0 else "", level, line))

    # -- lifecycle -------------------------------------------------------------- #
    def start(self) -> None:
        self._painter.start()
        self._keys.start()

    def stop(self) -> None:
        self._stop.set()
        self._painter.join(timeout=2)

    def __enter__(self) -> "ServerConsole":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- rendering -------------------------------------------------------------- #
    def _status_panel(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        stats = self.stats()
        active = stats.get("active", [])
        uptime = int(stats.get("uptime", 0))
        head = Text()
        head.append(f"{stats.get('model_id', '?')}  ", style="bold cyan")
        head.append(f"up {uptime // 3600:d}h{uptime // 60 % 60:02d}m{uptime % 60:02d}s   ")
        head.append(f"turns {len(active)}/{stats.get('slots', 0)}", style="bold")
        head.append(
            f"   {stats.get('step_rate', 0.0):.1f} tok/s total"
            f"   {stats.get('per_user_rate', 0.0):.1f} tok/s/user"
            f"   {stats.get('inflight', 0)} in flight"
            f"   {stats.get('rounds', 0)} rounds"
        )
        if stats.get("broken"):
            head.append(f"\nDECODE WEDGED: {stats['broken']}", style="bold red")

        pool = stats.get("pool") or {}
        if pool:
            parts = []
            for name, (used, total) in pool.items():
                share = used / total if total else 0.0
                style = "red" if share > 0.9 else "yellow" if share > 0.7 else "green"
                parts.append(f"[{style}]{name} {used}/{total}[/{style}]")
            head.append("\nKV pages: ")
            head.append(Text.from_markup(" ".join(parts)))
            head.append(f"  ~{stats.get('tokens_left', 0)} tokens free")

        # Not expanded: with seven short columns, stretching to the panel width scatters
        # them across the screen and makes the rows hard to read across.
        table = Table(box=None, pad_edge=False, expand=False, header_style="dim", padding=(0, 2, 0, 0))
        for name, justify, width in (
            ("slot", "right", 4),
            ("user", "left", 16),
            ("context", "right", 13),
            ("msgs", "right", 4),
            ("phase", "left", 14),
            ("progress", "right", 11),
            ("tok/s", "right", 6),
        ):
            table.add_column(name, justify=justify, no_wrap=True, width=width)
        by_slot = {turn["slot"]: turn for turn in active}
        for row in stats.get("users", []):
            turn = by_slot.get(row["index"])
            if turn is None:
                phase, progress, rate = ("busy" if row.get("busy") else "idle"), "", ""
                style = "dim" if not row.get("busy") else ""
            elif turn["phase"] == "prefill":
                phase = "prefill"
                progress = f"{turn['prefilled']}/{turn['prompt_tokens']}"
                rate = ""
                style = "yellow"
            else:
                phase = "decode" + (" (gone)" if turn["cancelled"] else "")
                progress = f"{turn['generated']}/{turn['max_tokens']}"
                rate = f"{turn['decode_rate']:.1f}"
                style = "green"
            table.add_row(
                str(row["index"]),
                (str(row["id"])[:14] or "-"),  # a slot nobody has used yet has no owner
                f"{row['tokens']}/{stats.get('max_seq', 0)}",
                str(row["messages"]),
                phase,
                progress,
                rate,
                style=style,
            )
        if not stats.get("users"):
            table.add_row("-", "(no users yet)", "", "", "", "", "", style="dim")

        flags = []
        if self.debug:
            flags.append("[cyan]debug[/cyan]")
        if self.paused:
            flags.append("[yellow]paused[/yellow]")
        if self._dropped:
            flags.append(f"[red]{self._dropped} log lines dropped[/red]")
        subtitle = "  ".join(flags + [f"[dim]{_HELP}[/dim]"])
        return Panel(Group(head, table), title="DeepSeek-V4-Flash server", subtitle=subtitle, border_style="cyan")

    def _log_lines(self, height: int):
        """The tail of the log, as ``time | message`` rows sized to what is left of the
        screen. Rendered as a table so the time column stays aligned and the message
        column wraps under itself rather than under the timestamps."""
        from rich.table import Table

        table = Table(box=None, pad_edge=False, expand=True, show_header=False)
        table.add_column("time", justify="left", no_wrap=True, style="dim", width=12)
        table.add_column("message", justify="left", overflow="fold", ratio=1)
        rows = list(self._lines)[-max(height, 1) :]
        for stamp, level, text in rows:
            table.add_row(stamp, text, style=_LEVEL_STYLE.get(level, ""))
        return table

    def _frame(self):
        from rich.console import Group

        status = self._status_panel()
        rows = shutil.get_terminal_size((100, 30)).lines
        # Measure the status so the log fills exactly the rest of the screen: a frame
        # taller than the terminal would scroll the status off the top.
        used = len(self.console.render_lines(status, self.console.options, pad=False))
        return Group(status, self._log_lines(rows - used - 1))

    def _paint_loop(self) -> None:
        from rich.live import Live

        interval = 1.0 / max(self.fps, 1.0)
        try:
            with Live(
                self._frame(),
                console=self.console,
                refresh_per_second=self.fps,
                screen=True,
                transient=False,
            ) as live:
                while not self._stop.is_set():
                    if not self.paused:
                        self._drain()
                    with contextlib.suppress(Exception):  # a resize mid-render, etc.
                        live.update(self._frame())
                    time.sleep(interval)
        except Exception as e:  # noqa: BLE001 - the server keeps running without the view
            self._stop.set()
            print(f"[console stopped: {e}]", file=sys.stderr, flush=True)

    # -- keys ------------------------------------------------------------------- #
    def _key_loop(self) -> None:
        """Read single keys without echo, if the terminal allows it.

        A no-op where stdin is not a tty (``--tui`` under ``nohup``), which just leaves
        the view read-only."""
        try:
            import termios
            import tty
        except ImportError:  # not POSIX
            return
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                key = sys.stdin.read(1)
                if not key:
                    return
                self._on_key(key.lower())
        except Exception:  # noqa: BLE001 - losing the keys is not worth a crash
            return
        finally:
            with contextlib.suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _on_key(self, key: str) -> None:
        if key == "q":
            self._stop.set()
            if self.on_quit is not None:
                self.on_quit()
        elif key == "d":
            self.debug = not self.debug
            self._note(f"debug lines {'on' if self.debug else 'off'}")
        elif key == "p":
            self.paused = not self.paused
            self._note(f"log {'paused' if self.paused else 'resumed'}")
        elif key == "c":
            self._lines.clear()
            self._dropped = 0

    def _note(self, text: str) -> None:
        self._lines.append((datetime.now().strftime("%H:%M:%S.%f")[:-3], "SUCCESS", f"[console] {text}"))


@contextlib.contextmanager
def console(logger, stats, on_quit=None, enabled: bool = True, debug: bool = False):
    """Route ``logger`` into a live :class:`ServerConsole` for the duration of the block.

    Yields the console, or ``None`` when one cannot be drawn (no ``rich``, not a
    terminal, or ``enabled=False``) -- in which case the logger is left exactly as it
    was, so the caller needs no fallback of its own.
    """
    ok, why = available()
    if not enabled or not ok:
        if enabled and not ok:
            logger.info(f"live console disabled: {why}")
        yield None
        return

    view = ServerConsole(stats, on_quit=on_quit, debug=debug)
    logger.remove()
    # Every level reaches the sink; the ``d`` key filters DEBUG at display time, so
    # toggling it shows the lines logged while it was off.
    sink_id = logger.add(view.write, level="DEBUG", format="{message}", enqueue=False)
    try:
        with view:
            yield view
    finally:
        logger.remove(sink_id)
        logger.add(sys.stderr, level="DEBUG" if debug else "INFO")
        # Replay what the pane held so a scrollback of the session survives the exit.
        for stamp, level, text in list(view._lines)[-200:]:
            print(f"{stamp:<12} {level:<8} {text}" if stamp else f"{'':<12} {level:<8} {text}", file=sys.stderr)


def _demo() -> int:
    """Render the console against fake stats, to check the layout without a device::

    python models/experimental/deepseek_v4_flash/demo/tui.py
    """
    import random

    from loguru import logger

    users = [
        {"id": f"user{i}", "index": i, "tokens": 0, "messages": 1, "thinking": False, "busy": False} for i in range(4)
    ]
    state = {"t0": time.time(), "rounds": 0, "steps": 0}

    def stats():
        state["rounds"] += 3
        state["steps"] += 7
        active = []
        for i, u in enumerate(users):
            u["tokens"] = min(u["tokens"] + random.randint(0, 30), 4096)
            if u["tokens"] % 7 < 4:
                active.append(
                    {
                        "user": u["id"],
                        "slot": i,
                        "phase": "prefill" if u["tokens"] < 300 else "decode",
                        "prompt_tokens": 300,
                        "prefilled": min(u["tokens"], 300),
                        "generated": max(u["tokens"] - 300, 0),
                        "max_tokens": 2048,
                        "prefill_seconds": 1.4,
                        "decode_rate": random.uniform(4, 12),
                        "cancelled": False,
                    }
                )
        return {
            "model_id": "deepseek-v4-flash",
            "uptime": time.time() - state["t0"],
            "slots": len(users),
            "max_seq": 4096,
            "users": users,
            "active": active,
            "rounds": state["rounds"],
            "steps": state["steps"],
            "step_rate": random.uniform(20, 40),
            "per_user_rate": random.uniform(4, 12),
            "inflight": random.randint(0, 8),
            "prefill_chunk": 16,
            "pool": {"sliding": (random.randint(1, 90), 100), "compress": (12, 64)},
            "tokens_left": random.randint(100, 9000),
            "broken": None,
        }

    stop = threading.Event()
    with console(logger, stats, on_quit=stop.set, debug=True) as view:
        if view is None:
            print("no live console available here", file=sys.stderr)
            return 1
        i = 0
        while not stop.is_set() and i < 400:
            i += 1
            logger.debug(f"round {i}: scheduled {random.randint(1, 4)} steps, pool moved")
            if i % 10 == 0:
                logger.info(f"user 'user{i % 4}': prefill of 312 tokens done in 1.42s (219 tok/s)")
            if i % 37 == 0:
                logger.warning("shared KV-cache pool is nearly full")
            time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(_demo())
