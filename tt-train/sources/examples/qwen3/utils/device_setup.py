# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared Tenstorrent device setup/teardown for single-device and distributed modes.

Pair every ``setup_device`` with :func:`teardown_device` — ``ctx.close_device()``
alone is not the inverse of ``ttml.open_device_mesh`` (see teardown_device).
"""

import os
import sys

import ttml

_MGD_WARNING = """
================================================================================
  WARNING: TT_MESH_GRAPH_DESC_PATH is NOT set!

  Distributed mode requires a Mesh Graph Descriptor (MGD) file.
  enable_fabric() will attempt automatic selection for 8 or 32 devices,
  but this may not match your hardware topology.

  For reliable operation, set the environment variable explicitly:

      export TT_MESH_GRAPH_DESC_PATH="/path/to/your/mesh_graph_descriptor.textproto"

  Common MGD files (relative to $TT_METAL_HOME):
      tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto
      tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto

  See: https://github.com/tenstorrent/tt-metal/blob/main/tt-train/docs/DISTRIBUTED_TRAINING.md#setting-mgd-files-via-environment-variable
================================================================================
"""


def setup_device(dp_size: int, tp_size: int, seed: int = 42):
    """Open a Tenstorrent device (single or mesh) and return ``(ctx, device)``.

    Handles:
      - Fabric enablement and MGD validation via ``ttml.open_device_mesh``
      - Registration of the named ``("dp", "tp")`` mesh
      - Parallelism-context initialisation

    The distributed path goes through ``ttml.open_device_mesh`` rather than
    ``enable_fabric`` + ``AutoContext.open_device`` so that ``ttml.mesh()`` is
    populated: the ``ttml.modules`` parallel layers (``VocabParallelEmbedding``,
    ``FeatureParallelEmbedding``, ...) resolve their cluster axis by *name*, and
    raise if no mesh has been registered. It performs the same fabric
    enablement, and additionally validates the mesh shape against the MGD file.
    """
    distributed = dp_size > 1 or tp_size > 1
    total_devices = dp_size * tp_size

    ctx = ttml.autograd.AutoContext.get_instance()
    if distributed:
        if "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
            print(_MGD_WARNING, file=sys.stderr)

        print(
            f"\nEnabling distributed mode: DP={dp_size}, TP={tp_size} "
            f"({total_devices} devices, mesh [{dp_size}, {tp_size}])"
        )
        ttml.open_device_mesh(ttml.Mesh((dp_size, tp_size), ("dp", "tp")))
        # ParallelismContext is a one-shot snapshot on the AutoContext singleton:
        # ``initialize_parallelism_context`` throws if one already exists and there is
        # no reset hook, so ``teardown_device`` cannot clear it. It stores only axis
        # indices and device counts taken from the mesh shape (no device handle), so an
        # existing context is still accurate for a mesh of the same shape -- which is
        # what a repeated setup_device() in one process (tests, notebooks) opens. Reuse
        # it in that case; a different shape genuinely cannot be served in-process.
        if ctx.is_parallelism_context_initialized():
            pctx = ctx.get_parallelism_context()
            existing = (pctx.get_ddp_size(), pctx.get_tp_size())
            if existing != (dp_size, tp_size):
                raise RuntimeError(
                    f"This process already initialized a ParallelismContext for "
                    f"DP={existing[0]}, TP={existing[1]}, and it cannot be reset, so "
                    f"DP={dp_size}, TP={tp_size} cannot be served. Use a fresh process "
                    f"for a different mesh shape."
                )
        else:
            ctx.initialize_parallelism_context(
                ttml.autograd.DistributedConfig(enable_ddp=dp_size > 1, enable_tp=tp_size > 1)
            )
    else:
        ctx.open_device()
    ctx.set_seed(seed)
    return ctx, ctx.get_device()


def teardown_device():
    """Inverse of :func:`setup_device`. Use this, not ``ctx.close_device()``.

    ``ttml.open_device_mesh`` installs *two* pieces of state: the ``MeshDevice`` on
    the AutoContext, and the process-global named mesh that ``ttml.mesh()`` reads.
    ``ctx.close_device()`` only drops the first (it also disables fabric), leaving
    ``ttml.mesh()`` answering with the shape and axis names of a mesh whose devices
    are gone. The next implicit ``get_device()`` then opens the default 1x1 device
    while ``ttml.mesh()`` still claims the old shape, so anything resolving a TP
    axis by name builds for the wrong device count and fails somewhere unrelated.
    ``ttml.close_device_mesh()`` does both, in the right order.

    Safe on the single-device path (no mesh was registered), and idempotent — safe
    to call when nothing is open and safe to call twice, so callers can put it in a
    ``finally`` on top of a normal-path call without guarding.

    Does NOT reset the AutoContext's ParallelismContext: that is a one-shot snapshot
    with no reset hook exposed (see ``setup_device``, which reuses a matching one).
    So a process can reopen the *same* mesh shape after teardown, but not a
    different one.
    """
    ttml.close_device_mesh()
