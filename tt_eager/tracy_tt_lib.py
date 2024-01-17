# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random

import seaborn as sns

from tt_lib.profiler import start_tracy_zone, stop_tracy_zone


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


def tracy_marker_line(frame, event, args):
    global callStack
    if event == "call":
        callStack.append("call")
        start_tracy_zone(f"{frame.f_code.co_filename}", f"PY_FUNC_{frame.f_code.co_name}", frame.f_lineno)
    elif event == "return":
        while callStack and callStack.pop() == "line":
            stop_tracy_zone(color=plotColorThree)
        if (
            "tt_lib_profiler_wrapper.py" in f"{frame.f_code.co_filename}"
            and frame.f_locals
            and "local_name" in frame.f_locals.keys()
        ):
            stop_tracy_zone(f"PY_TT_LIB_{frame.f_locals['local_name']}", plotColorTwo)
        else:
            stop_tracy_zone(color=plotColorOne)
    elif event == "line":
        if "tt_lib_profiler_wrapper.py" not in f"{frame.f_code.co_filename}":
            if callStack and callStack[-1] == "line":
                stop_tracy_zone(color=plotColorThree)
            else:
                callStack.append("line")
            start_tracy_zone(f"{frame.f_code.co_filename}", f"PY_LINE_{frame.f_code.co_name}", frame.f_lineno)

    return tracy_marker_line


def tracy_marker_func(frame, event, args):
    if event in ["call", "c_call"]:
        start_tracy_zone(f"{frame.f_code.co_filename}", f"PY_FUNC_{frame.f_code.co_name}", frame.f_lineno)
    elif event in ["return", "c_return", "c_exception"]:
        if (
            "tt_lib_profiler_wrapper.py" in f"{frame.f_code.co_filename}"
            and frame.f_locals
            and "local_name" in frame.f_locals.keys()
        ):
            stop_tracy_zone(f"PY_TT_LIB_{frame.f_locals['local_name']}", plotColorTwo)
        else:
            stop_tracy_zone(color=plotColorOne)


def finish_all_zones():
    while not stop_tracy_zone(color=plotColorFour):
        pass
