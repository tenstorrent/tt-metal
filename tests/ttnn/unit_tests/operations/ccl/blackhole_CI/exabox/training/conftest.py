# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for DistributedContext.send/recv exabox tests on QUAD_BH.

Background:
- The C++ primitive under test is
  tt::tt_metal::distributed::multihost::DistributedContext::send/recv —
  byte-level host-side MPI point-to-point. This is the underlying op used
  by the multi-host socket descriptor handshake in mesh_socket_utils.cpp
  and by MPISocket::send/recv (ttnn/core/distributed/mpi_socket.cpp:51,62).
- We exercise it directly through Python bindings on
  ttml.core.distributed.DistributedContext (added in nb_core.cpp). This
  bypasses MeshSocket entirely, so the test is independent of mesh
  layout: a single-mesh rank-binding suffices.

Build prerequisite:
- $TT_METAL_HOME built with --build-tt-train (creates _ttml*.so).
- The conftest auto-symlinks the artifact into the source ttml package and
  pre-loads it into sys.modules so `import ttml` resolves both as a package
  and as the bare `_ttml` module that several ttml submodules import.
"""

from __future__ import annotations

import glob
import os
import sys

import pytest


def _bootstrap_ttml_extension() -> None:
    """Make ttml's C++ extension (_ttml*.so) importable in this process.

    See the previous conftest revision history for the full reasoning. In
    short: load _ttml*.so directly via importlib + register as `_ttml` in
    sys.modules; symlink it into the source ttml/ttml/ package so the
    fallback `from . import _ttml` branch in ttml/__init__.py also works;
    add build_Release/lib to LD_LIBRARY_PATH for transitive C++ deps.

    Idempotent; no-op if the build artifact is missing or already loaded.
    """
    if "_ttml" in sys.modules:
        return

    tt_metal_home = os.environ.get("TT_METAL_HOME")
    if not tt_metal_home:
        raise RuntimeError("TT_METAL_HOME must be set to the root of the tt-metal checkout.")
    build_dir = os.path.join(tt_metal_home, "build_Release", "ttml")
    lib_dir = os.path.join(tt_metal_home, "build_Release", "lib")
    src_pkg = os.path.join(tt_metal_home, "tt-train", "sources", "ttml", "ttml")

    candidates = glob.glob(os.path.join(build_dir, "_ttml*.so"))
    if not candidates:
        return  # not built; tests will skip with a clear message
    so_path = candidates[0]

    if os.path.isdir(lib_dir):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if lib_dir not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld}" if ld else lib_dir

    if os.path.isdir(src_pkg) and not glob.glob(os.path.join(src_pkg, "_ttml*.so")):
        try:
            os.symlink(so_path, os.path.join(src_pkg, os.path.basename(so_path)))
        except FileExistsError:
            # Concurrent pytest workers may race on this symlink; the operation
            # is idempotent so the loser of the race can safely continue.
            pass

    # ttnn pre-loads the shared libs that _ttml depends on
    import ttnn  # noqa: F401

    import importlib.util

    spec = importlib.util.spec_from_file_location("_ttml", so_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ttml"] = module
    spec.loader.exec_module(module)


_bootstrap_ttml_extension()


# QUAD_BH topology: 4 hosts × 1 MPI rank/host.
#
# Rank-binding used:
#   tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml
# (stock — all 4 ranks share mesh_id: 0). The byte-level primitive doesn't
# construct a MeshSocket and doesn't care about mesh layout.
QUAD_BH_NUM_RANKS = 4


@pytest.fixture(scope="session")
def distributed_runtime():
    """Initialize tt-train AutoContext + DistributedContext once per run.

    Yields a namespace exposing:
      - distributed_ctx : ttml.core.distributed.DistributedContext
      - rank, world_size : ints
      - ttml : the imported ttml module

    No device is opened. No fabric is enabled. We only need MPI bootstrap.

    Skips the test session if the run is single-process (world_size <= 1)
    or the world size doesn't match QUAD_BH (4).
    """
    try:
        import ttml
    except ImportError as e:
        pytest.skip(f"tt-train Python module unavailable: {e}. Build with --build-tt-train.")

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)
    distributed_ctx = autograd_ctx.get_distributed_context()
    rank = distributed_ctx.rank()
    world_size = distributed_ctx.size()

    if world_size <= 1:
        pytest.skip(f"send/recv tests require multi-process launch (world_size={world_size}); use tt-run.")
    if world_size != QUAD_BH_NUM_RANKS:
        pytest.skip(f"This test is parametrized for QUAD_BH ({QUAD_BH_NUM_RANKS} ranks); got world_size={world_size}.")

    class _Runtime:
        pass

    rt = _Runtime()
    rt.distributed_ctx = distributed_ctx
    rt.rank = rank
    rt.world_size = world_size
    rt.ttml = ttml

    yield rt

    distributed_ctx.barrier()
