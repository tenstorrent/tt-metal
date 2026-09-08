# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import random
import resource
import sys
import sysconfig

import seaborn as sns

import ttnn

# Frames from the standard library or installed packages are never what a profiling run wants to
# see -- pytest's own collection alone touches 40K+ distinct (file, line) locations before a model
# is even built, blowing past tracy's 32K static-source-location ceiling and, on a real decode loop,
# accumulating hundreds of millions of zones (measured: 241M zones, ~185GB host RSS, OOM-killed) for
# instrumentation of code nobody asked to profile. Filtered by path rather than by package name so it
# needs no maintenance as dependencies change; the model's own code (never under either prefix) keeps
# its full per-function zones exactly as before.
_STDLIB_PREFIX = sysconfig.get_paths()["stdlib"]
_SITE_PACKAGES_PREFIX = sysconfig.get_paths()["purelib"]


def _is_library_frame(filename: str) -> bool:
    return filename.startswith(_STDLIB_PREFIX) or filename.startswith(_SITE_PACKAGES_PREFIX)


# The stdlib/site-packages filter above only reduces the rate of growth of this process's own
# Tracy client zone queue, not its ceiling -- a model with enough distinct Python call/line volume
# (e.g. many MoE-routed layers) can still exhaust host memory before the run finishes. Since that
# volume scales with the model rather than any name we can hardcode, the cutoff is derived from the
# machine's own memory (not a fixed constant) and checked cheaply (once per _RSS_CHECK_INTERVAL
# calls) so any model self-limits before OOM instead of tracing unconditionally to the end.
_RSS_CHECK_INTERVAL = 50_000
_RSS_LIMIT_KB = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") * 0.5 / 1024)
_profile_call_count = [0]
_profiling_disengaged = [False]
_zone_open_stack = []


def _rss_limit_exceeded() -> bool:
    _profile_call_count[0] += 1
    if _profile_call_count[0] % _RSS_CHECK_INTERVAL != 0:
        return False
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > _RSS_LIMIT_KB


def hex_to_int(color):
    return int(color[1:], 16)


plotColors = sns.color_palette("deep").as_hex()
plotColorOne = random.choice(plotColors)
plotColors.remove(plotColorOne)
plotColorTwo = random.choice(plotColors)
plotColors.remove(plotColorTwo)
plotColorThree = random.choice(plotColors)
plotColors.remove(plotColorThree)
plotColorFour = random.choice(plotColors)

plotColorOne = hex_to_int(plotColorOne)
plotColorTwo = hex_to_int(plotColorTwo)
plotColorThree = hex_to_int(plotColorThree)
plotColorFour = hex_to_int(plotColorFour)

callStack = []


def tracy_send_message(message):
    ttnn.tracy_message(message)


def tracy_marker_line(frame, event, args):
    global callStack
    if event == "call":
        callStack.append("call")
        ttnn.start_tracy_zone(f"{frame.f_code.co_filename}", f"PY_FUNC_{frame.f_code.co_name}", frame.f_lineno)
    elif event == "return":
        while callStack and callStack.pop() == "line":
            ttnn.stop_tracy_zone(color=plotColorThree)
        if (
            "ttnn_profiler_wrapper.py" in f"{frame.f_code.co_filename}"
            and frame.f_locals
            and "local_name" in frame.f_locals.keys()
        ):
            ttnn.stop_tracy_zone(f"PY_TT_LIB_{frame.f_locals['local_name']}", plotColorTwo)
        else:
            ttnn.stop_tracy_zone(color=plotColorOne)
    elif event == "line":
        if "ttnn_profiler_wrapper.py" not in f"{frame.f_code.co_filename}":
            if callStack and callStack[-1] == "line":
                ttnn.stop_tracy_zone(color=plotColorThree)
            else:
                callStack.append("line")
            ttnn.start_tracy_zone(f"{frame.f_code.co_filename}", f"PY_LINE_{frame.f_code.co_name}", frame.f_lineno)

    return tracy_marker_line


def tracy_marker_func(frame, event, args):
    # Per-frame record of whether a zone was actually opened for it, mirroring the call stack --
    # a global counter can't tell a later "return" whether ITS frame opened a zone, since a frame
    # that called in before the RSS trip (zone opened) can return after it, while a nested frame
    # that called in after the trip (zone skipped) returns first: only an explicit per-frame stack,
    # popped in the same order frames return, tells each return what its own call decided.
    if event in ["call", "c_call"]:
        if _profiling_disengaged[0] or _is_library_frame(frame.f_code.co_filename):
            _zone_open_stack.append(False)
            return
        if _rss_limit_exceeded():
            _profiling_disengaged[0] = True
            _zone_open_stack.append(False)
            return
        _zone_open_stack.append(True)
        ttnn.start_tracy_zone(f"{frame.f_code.co_filename}", f"PY_FUNC_{frame.f_code.co_name}", frame.f_lineno)
    elif event in ["return", "c_return", "c_exception"]:
        opened = _zone_open_stack.pop() if _zone_open_stack else False
        if opened:
            if (
                "ttnn_profiler_wrapper.py" in f"{frame.f_code.co_filename}"
                and frame.f_locals
                and "local_name" in frame.f_locals.keys()
            ):
                ttnn.stop_tracy_zone(f"PY_TT_LIB_{frame.f_locals['local_name']}", plotColorTwo)
            else:
                ttnn.stop_tracy_zone(color=plotColorOne)
        # Only fully unregister once every zone opened before the trip has returned (stack empty) --
        # doing it earlier would silence "return" events for frames still on the stack, leaving their
        # already-opened zones with no matching stop.
        if _profiling_disengaged[0] and not _zone_open_stack:
            sys.setprofile(None)


def finish_all_zones():
    while not ttnn.stop_tracy_zone(color=plotColorFour):
        pass
