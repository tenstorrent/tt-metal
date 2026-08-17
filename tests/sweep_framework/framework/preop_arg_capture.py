# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-operation argument capture for sweep tracing.

``ttnn.operation_tracer`` serializes an operation's arguments *after* the op runs
(its wrapper executes the op, then serializes args + return). For arguments the op
mutates in place — notably ``all_gather_async``'s ``persistent_output_buffer``,
whose tensor topology becomes the gathered/output layout after the op — that
records the *post-op* state. The model trace (master) records the *input* topology,
so a post-op sweep capture diffs against the master on ``tensor_placement``.

This module captures arguments *before* the op executes, matching how the master
was recorded, by registering a ttnn pre-operation hook that reuses
operation_tracer's own serializer. It also disables operation_tracer's post-op
auto-capture so traces aren't written twice. The op's return value is not needed
for validation (config_hash is derived from arguments; master configs carry no
return_value), so the pre-op hook records arguments only.

Lives in the sweep framework — ttnn core (operation_tracer.py) is unchanged.
"""

import os
import pathlib
import sys

from loguru import logger


def _trace_log_dir():
    """Mirror operation_tracer's log-dir resolution so collected files land together."""
    import ttnn

    if os.environ.get("TTNN_OPERATION_TRACE_DIR"):
        return pathlib.Path(os.environ["TTNN_OPERATION_TRACE_DIR"])
    log_dir = getattr(ttnn.CONFIG, "operation_parameter_log_dir", None)
    if log_dir:
        return pathlib.Path(log_dir)
    root = getattr(ttnn.CONFIG, "root_report_path", None)
    if root:
        return pathlib.Path(root) / "operation_parameters"
    return pathlib.Path("generated/ttnn/operation_parameters")


def _preop_hook(operation, function_args, function_kwargs):
    """Serialize the operation's arguments BEFORE it executes (return omitted)."""
    import ttnn.operation_tracer as ot

    try:
        op_name = getattr(operation, "python_fully_qualified_name", None) or str(operation)
        ot.serialize_operation_parameters(
            op_name,
            function_args,
            function_kwargs,
            _trace_log_dir(),
            return_value=None,
            serialize_tensor_values=ot._SERIALIZE_TENSOR_VALUES,
        )
    except Exception as e:  # tracing must never break the op under test
        logger.debug(f"pre-op arg capture failed for {operation}: {e}")
    return None


def enable_preop_capture():
    """Capture op arguments pre-op (instead of operation_tracer's post-op capture).

    Registers the pre-op hook and disables operation_tracer's own auto-capture so
    every op is traced exactly once, with arguments as they were passed in.
    """
    import ttnn
    import ttnn.decorators as decorators
    import ttnn.operation_tracer as ot

    # Turn OFF operation_tracer's post-op auto-capture: it triggers on
    # ``--trace-params`` in argv (or enable_tracing). Drop the flag and clear the
    # tracer's lazily-cached argv check so only our pre-op hook records traces.
    sys.argv = [a for a in sys.argv if a != "--trace-params"]
    ot._TRACE_PARAMS_IN_ARGV = None  # force recompute from the cleaned argv
    ot.enable_tracing(False)

    if _preop_hook not in decorators.PRE_OPERATION_HOOKS:
        decorators.PRE_OPERATION_HOOKS.append(_preop_hook)
    logger.info("Operation tracing enabled (pre-op argument capture)")
