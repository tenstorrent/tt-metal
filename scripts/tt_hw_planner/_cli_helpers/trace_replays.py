# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--trace-replays: how many times the trace speed test replays each captured stage.

trace_replay's default (16, cut only when the TIME budget runs short) suits a step costing
milliseconds. A step costing tens of seconds needs few replays to average, and 16 back to back are
what heat the board: Devstral's encode stage, 34.3 s a replay, reached the 95C abort limit on every
attempt on a 4-chip p300c (2026-10-12). The count was reachable only through trace_replay's env var.
"""

from __future__ import annotations

import argparse
import os


def _replay_count(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1 (got %s)" % text)
    return n


def add_trace_replays_arg(parser: argparse.ArgumentParser) -> None:
    """Single definition of ``--trace-replays``, shared by the subcommands that run trace replay."""
    parser.add_argument(
        "--trace-replays",
        dest="trace_replays",
        type=_replay_count,
        default=None,
        help=(
            "How many times the trace speed test replays each captured stage (default: trace_replay's "
            "own, 16, cut to what the time budget affords). Lower it for a stage that takes seconds per "
            "replay: back-to-back replays are what heat the board, and a long step needs few to average."
        ),
    )


def apply_trace_replays(args) -> None:
    """Put --trace-replays into this process's environment, which every device child inherits (the
    emit-e2e gate server, whose env is built from scratch, forwards it by the same name)."""
    n = getattr(args, "trace_replays", None)
    if n is None:
        return
    from models.experimental.perf_automation.agent.trace_replay import REPLAY_ITERS_ENV

    os.environ[REPLAY_ITERS_ENV] = str(n)
