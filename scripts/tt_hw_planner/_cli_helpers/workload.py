# SPDX-License-Identifier: Apache-2.0
"""The WORKLOAD a run drives: how many samples, and -- for a model whose cost follows an input SIZE
rather than a token count -- how large one of them is.

Two commands ask for the size and mean different things by it. `optimize` profiles AT it; `emit-e2e`
BUILDS for it, which matters more than it sounds: the size a pipeline is built for decides how it lays
its stages out over the mesh, so one built for a smaller default does not merely run slower at the
larger size, it aborts. Neither can discover the size from the checkpoint -- a vision or diffusion model
runs at whatever resolution it is handed -- so it is the operator's to state, and a default picked
inside the tool is a decision the operator never saw.

The flag and its validation live here because both commands take them; the variable the tests read is
named once in perf_adapter (AREA_ENV), the way the batch already was.
"""

from __future__ import annotations


def add_area_arg(parser) -> None:
    """--area on a parser."""
    from models.experimental.perf_automation.agent.perf_adapter import AREA_ENV

    parser.add_argument(
        "--area",
        type=int,
        default=None,
        dest="area",
        help="input side for a model whose cost follows an input SIZE rather than a token count -- a "
        "diffusion pipeline's condition image, say -- as the side of the equivalent square, the model's "
        "own resizing rule deciding the exact shape (sets %s). A model with no such axis ignores it." % AREA_ENV,
    )


def workload_value(args, name):
    """(value, error) for one workload flag.

    (None, None) when it was not passed: a flag nobody gave must leave things as they are, so exporting
    the variable still works and a supervised run keeps the environment it inherited."""
    v = getattr(args, name, None)
    if v is None:
        return None, None
    if int(v) < 1:
        return None, "--%s must be 1 or more" % name
    return int(v), None
