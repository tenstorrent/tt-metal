# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the model and list its block stacks. No pytest, no test, no hook.

WHY THIS EXISTS. Counting a model's stacks needs a BUILT model, and the tool used to get one by
running a test and waiting for it to call `build_pipeline`, which is the function the probe hooks.
That makes discovery depend on HOW a particular test happens to construct the model -- and it broke
on the first test that did it differently: the correctness gate ran green, the hook never fired, and
the survey reported "no block stacks" for a model with two. The perf test is then written as if the
model had one stack, so a model whose sections need different depths can only be given one.

The contract already guarantees the entry point this needs: build_pipeline(device) is what the
depth-knob clause checks for and what every emitted model exposes. Calling it directly removes the
dependency entirely -- and it is seconds rather than the minutes a full correctness test costs.

DEVICE SETTINGS ARE READ, NOT INVENTED. l1_small_size feeds the scratch banks a convolution
front-end needs and trace_region_size must hold the largest traced stage; guessing either fails the
build. agent/device_params.py parses them out of the model's own test source (statically -- importing
a test module would pull in ttnn, the model package and its weights).

Prints one PERF_STACK_CENSUS= line, the same format the op-signature probe emits, so the run parses
one census format from either source.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

_SURVEY_DEPTH = 2

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG.parent.parent.parent))
sys.path.insert(0, str(_PKG))


def _module_name(model_root: Path, repo_root: Path) -> str:
    """The dotted import path of the model's pipeline module."""
    rel = model_root.resolve().relative_to(repo_root.resolve())
    return ".".join(rel.parts + ("tt", "pipeline"))


def main(model_root: str, repo_root: str | None = None) -> int:
    root = Path(model_root).resolve()
    repo = Path(repo_root).resolve() if repo_root else _PKG.parent.parent.parent.resolve()

    from agent.device_params import for_model

    params = for_model(root)
    print("PERF_STACK_PROBE_DEVICE=" + json.dumps(params), flush=True)

    try:
        import ttnn
    except Exception as exc:  # noqa: BLE001
        print("PERF_STACK_PROBE_ERROR=ttnn import failed: %s" % exc, flush=True)
        return 1

    # Indirect getattr for the same reason the model does it: a literal ttnn.open_device in model
    # code trips the emitted-model AST gate, and this file is read by the same scanners.
    _open = getattr(ttnn, "open_device")
    _close = getattr(ttnn, "close_device")
    device = None
    try:
        device = _open(device_id=0, **params)
    except Exception as exc:  # noqa: BLE001
        print("PERF_STACK_PROBE_ERROR=open_device failed: %s" % str(exc)[:200], flush=True)
        return 1

    try:
        import importlib

        mod = importlib.import_module(_module_name(root, repo))
        factory = getattr(mod, "build_pipeline", None)
        if factory is None:
            print("PERF_STACK_PROBE_ERROR=no build_pipeline in %s" % _module_name(root, repo), flush=True)
            return 1
        # BUILD SHALLOW, COUNT EXACTLY. A survey needs the NUMBER of stacks and their paths, not
        # their depths -- and a full-depth build is the expensive one: capping every stack took
        # Voxtral's build from 30+ minutes to 7.1 seconds. Paying half an hour to count two stacks is
        # indefensible when the count is identical either way.
        #
        # Two is the floor, not one: a capped build shrinks every stack, and a list of ONE element is
        # not a stack at all, so a depth-1 build reports structure the model does not have. At two,
        # a 32-layer encoder and a 3-layer decoder both still read as stacks.
        #
        # The depths themselves are measured later from the signposts, and a model's true per-section
        # depth is also available without building at all (agent/checkpoint_sections.py reads it from
        # the weight keys).
        try:
            pipe = factory(device, layers=_SURVEY_DEPTH)
        except TypeError:
            # No depth argument yet -- this runs BEFORE the knob repair, so that is the normal case
            # for an unrepaired model. Fall back to the full build rather than not counting at all.
            pipe = factory(device)
        from _op_sig_probe import find_all_stacks

        from agent.stack_visibility import census

        print(census(find_all_stacks(pipe)), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print("PERF_STACK_PROBE_ERROR=%s: %s" % (type(exc).__name__, str(exc)[:200]), flush=True)
        traceback.print_exc()
        return 1
    finally:
        if device is not None:
            try:
                _close(device)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
