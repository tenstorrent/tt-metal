# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package-level pytest configuration for ``mistral_3_5_d_p``.

Sets the mesh-graph descriptor before ttnn initialises the cluster, because the right one depends on
how the pod is wired and getting it wrong is a TT_FATAL during collection rather than a test failure:

  * A Galaxy wired as a plain 8x4 GRID (no wrap-around links) can only run ``FABRIC_1D``, and its
    descriptor is ``single_bh_galaxy_mesh_graph_descriptor``. Asking for the torus descriptor there
    fails with "Graph specified in MGD could not fit in the discovered physical topology ... STRICT".
  * A Galaxy wired as a 2D TORUS runs ``FABRIC_1D_RING`` with
    ``single_bh_galaxy_torus_xy_graph_descriptor``, which is faster for every ring-gather collective.

``MISTRAL_LINEAR_FABRIC`` selects between them and defaults to the linear/grid case, which is what
this bring-up's pod is. An explicit ``TT_MESH_GRAPH_DESC_PATH`` always wins.
"""

import os
from pathlib import Path

import pytest

_DESCRIPTORS = {
    True: "single_bh_galaxy_mesh_graph_descriptor.textproto",
    False: "single_bh_galaxy_torus_xy_graph_descriptor.textproto",
}


def _linear_fabric() -> bool:
    return os.getenv("MISTRAL_LINEAR_FABRIC", "1").strip().lower() in ("1", "true", "yes", "on")


def _set_default_mesh_graph_descriptor() -> None:
    if os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        return  # explicit override wins
    root = os.environ.get("TT_METAL_HOME")
    if not root:
        return
    path = Path(root) / "tt_metal" / "fabric" / "mesh_graph_descriptors" / _DESCRIPTORS[_linear_fabric()]
    if path.is_file():
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(path)


_set_default_mesh_graph_descriptor()


def pytest_addoption(parser):
    parser.addoption(
        "--skip-model-load",
        action="store_true",
        default=False,
        help="Skip loading the real checkpoint state dict (every pre-P1 test uses random weights anyway)",
    )


@pytest.fixture(scope="session")
def state_dict(request):
    """The real checkpoint, or an empty dict. Every test up to P1 uses random weights and must not
    depend on this — it exists for the P1+ tests that do."""
    import os as _os

    load_model = not request.config.getoption("--skip-model-load")
    model_path = _os.getenv("HF_MODEL")
    if model_path is None or not load_model:
        return {}
    from models.demos.mistral_3_5_d_p.tt.model_config import ModelArgs

    return ModelArgs.load_state_dict(model_path)
