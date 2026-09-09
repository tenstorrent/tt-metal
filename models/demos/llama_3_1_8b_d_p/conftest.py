# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package-level fixtures, and the mesh-graph descriptor choice.

**The descriptor is chosen HERE, before the cluster initialises.** Setting it later has no effect,
which is why the recipe puts it in the package conftest rather than in a test body.

This Galaxy is a plain 8x4 GRID with no wrap-around links, so the torus descriptor cannot map on it
(the cluster fails STRICT at init: "Graph specified in MGD could not fit in the discovered physical
topology"). The default is therefore the plain mesh descriptor, which maps on ANY galaxy, torus-wired
or not, paired with `FABRIC_1D` + `ttnn.Topology.Linear`.

The torus sits behind one env knob, `LLAMA31_8B_FABRIC_TORUS=1`: it is the single significant perf
lever a bring-up has, so take it where the pod offers it, but it is never a correctness gate —
correctness is topology-independent — and a bring-up must not block for the want of it. Collective
COST does differ between the two, so linear and torus numbers are not comparable and every
measurement records which it was taken on.

**On this pod the knob does nothing usable, and that is a logged `env` finding, not a blocker.**
Every torus descriptor shipped for a Blackhole galaxy spans MULTIPLE galaxies or hosts
(`dual_bh_galaxy_torus_xy`, `32x4_quad_bh_galaxy_torus_xy`, `bh_galaxy_sp4_torus_xy`, ...), so
selecting one here fails at control-plane init with `Mesh 0 has 4 host ranks, expected 1` — a
single-host 8x4 torus descriptor does not exist. Setting the knob therefore raises rather than
silently running; all measurements in this package are LINEAR.
"""

import os

import pytest

_DESCRIPTORS = {
    "linear": "single_bh_galaxy_mesh_graph_descriptor.textproto",
    "torus": "bh_galaxy_sp4_torus_xy_graph_descriptor.textproto",
}


def _set_mesh_graph_descriptor():
    """Point TT_MESH_GRAPH_DESC_PATH at the right descriptor, unless the caller already has."""
    if os.getenv("TT_MESH_GRAPH_DESC_PATH"):
        return
    tt_metal_home = os.getenv("TT_METAL_HOME")
    if not tt_metal_home:
        return
    which = "torus" if os.getenv("LLAMA31_8B_FABRIC_TORUS") == "1" else "linear"
    if which == "torus":
        raise RuntimeError(
            "LLAMA31_8B_FABRIC_TORUS=1 is set, but every shipped Blackhole-galaxy torus mesh-graph "
            "descriptor is multi-galaxy/multi-host and fails control-plane init on this single 8x4 "
            "pod ('Mesh 0 has 4 host ranks, expected 1'). Unset it to run linear, or point "
            "TT_MESH_GRAPH_DESC_PATH at a single-host torus descriptor if one becomes available."
        )
    path = os.path.join(tt_metal_home, "tt_metal", "fabric", "mesh_graph_descriptors", _DESCRIPTORS[which])
    if os.path.exists(path):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = path


_set_mesh_graph_descriptor()


def pytest_addoption(parser):
    parser.addoption(
        "--skip-model-load",
        action="store_true",
        default=False,
        help="Skip loading the checkpoint state dict (random-weight tests do not need it)",
    )


@pytest.fixture(scope="session")
def state_dict(request):
    """The real checkpoint, or `{}`.

    Every PCC test up to P1 runs on RANDOM weights, identical on both sides, so the default for a
    unit-test session is to load nothing: real checkpoint loading is not a dependency of any module
    test and is deferred to P1.
    """
    from models.demos.llama_3_1_8b_d_p.tt.model_config import ModelArgs, resolve_weights_path

    if request.config.getoption("--skip-model-load"):
        return {}
    path = resolve_weights_path(required=False)
    return ModelArgs.load_state_dict(path) if path else {}
