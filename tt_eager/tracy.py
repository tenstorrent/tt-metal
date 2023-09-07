# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib.machinery
import sys
import io
import random

from loguru import logger
import seaborn as sns

from tt_lib.profiler import start_tracy_zone, stop_tracy_zone

import tracy_state


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


class Profiler:
    def __init__(self):
        self.doProfile = tracy_state.doPartial and sys.gettrace() is None and sys.getprofile() is None
        self.doLine = tracy_state.doLine

    def enable(self):
        if self.doProfile:
            if self.doLine:
                sys.settrace(tracy_marker_line)
            else:
                sys.setprofile(tracy_marker_func)

    def disable(self):
        if self.doProfile:
            sys.settrace(None)
            sys.setprofile(None)
            while not stop_tracy_zone(color=plotColorFour):
                pass


def runctx(cmd, globals, locals, partialProfile):
    if not partialProfile:
        sys.setprofile(tracy_marker_func)

    try:
        exec(cmd, globals, locals)
    finally:
        sys.setprofile(None)
        while not stop_tracy_zone(color=plotColorFour):
            pass


def main():
    import os
    from optparse import OptionParser

    usage = "tracy_python.py [-m module | scriptfile] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option("-m", dest="module", action="store_true", help="Profile a library module.", default=False)
    parser.add_option("-p", dest="partial", action="store_true", help="Only profile enabled zones", default=False)
    parser.add_option("-l", dest="lines", action="store_true", help="Profile every line of python code", default=False)

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    if len(args) > 0:
        if options.module:
            import runpy

            code = "run_module(modname, run_name='__main__')"
            globs = {
                "run_module": runpy.run_module,
                "modname": args[0],
            }
        else:
            progname = args[0]
            sys.path.insert(0, os.path.dirname(progname))
            with io.open_code(progname) as fp:
                code = compile(fp.read(), progname, "exec")
            spec = importlib.machinery.ModuleSpec(name="__main__", loader=None, origin=progname)
            globs = {
                "__spec__": spec,
                "__file__": spec.origin,
                "__name__": spec.name,
                "__package__": None,
                "__cached__": None,
            }

        if options.partial:
            tracy_state.doPartial = True

        if options.lines:
            tracy_state.doLine = True

        try:
            runctx(code, globs, None, options.partial)
        except BrokenPipeError as exc:
            # Prevent "Exception ignored" during interpreter shutdown.
            sys.stdout = None
            sys.exit(exc.errno)
    else:
        parser.print_usage()
    return parser


# When invoked as main program, invoke the profiler on a script
if __name__ == "__main__":
    main()
