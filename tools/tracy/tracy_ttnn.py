# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import random
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
    # Same filename check on both branches: a given frame's co_filename cannot change between its
    # own call and return, so start/stop stay paired -- skipping one side but not the other would
    # leave stop_tracy_zone popping a zone a filtered-out call never pushed (the orphan-marker bug
    # profiler.cpp already has to tolerate elsewhere, worth not adding a Python-side source of it).
    if _is_library_frame(frame.f_code.co_filename):
        return
    if event in ["call", "c_call"]:
        ttnn.start_tracy_zone(f"{frame.f_code.co_filename}", f"PY_FUNC_{frame.f_code.co_name}", frame.f_lineno)
    elif event in ["return", "c_return", "c_exception"]:
        if (
            "ttnn_profiler_wrapper.py" in f"{frame.f_code.co_filename}"
            and frame.f_locals
            and "local_name" in frame.f_locals.keys()
        ):
            ttnn.stop_tracy_zone(f"PY_TT_LIB_{frame.f_locals['local_name']}", plotColorTwo)
        else:
            ttnn.stop_tracy_zone(color=plotColorOne)


def finish_all_zones():
    while not ttnn.stop_tracy_zone(color=plotColorFour):
        pass
