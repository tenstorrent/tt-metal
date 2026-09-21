# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package-level pytest configuration.

The **mesh-graph descriptor is chosen before the cluster initialises**, so it is set here at import
time rather than in a fixture. Default is the plain single-galaxy MESH descriptor, which maps on
any galaxy whether or not it is torus-wired; ``QWEN35_TORUS=1`` selects the torus descriptor where
the pod offers it. The torus is the one significant perf lever bring-up has, but it is never a
correctness gate — if it will not map, the run falls back to linear and logs an ``env`` finding
(recipe section 4, "run on whatever machine you get").
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_DESCRIPTORS = _REPO / "tt_metal" / "fabric" / "mesh_graph_descriptors"
_MESH = _DESCRIPTORS / "single_bh_galaxy_mesh_graph_descriptor.textproto"
_TORUS = _DESCRIPTORS / "single_bh_galaxy_torus_xy_graph_descriptor.textproto"


def _select_mesh_graph_descriptor() -> None:
    if os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        return  # an explicit override always wins
    want_torus = os.environ.get("QWEN35_TORUS") == "1"
    chosen = _TORUS if want_torus else _MESH
    if chosen.exists():
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(chosen)


_select_mesh_graph_descriptor()


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--skip-model-load",
        action="store_true",
        default=False,
        help="Skip loading the real checkpoint (random-weight tests do not need it).",
    )


@pytest.fixture(scope="session")
def hf_state_dict(request):
    """The real Qwen3.8-27B text-tower state dict, or ``{}`` when the checkpoint is not wanted.

    Session-scoped because loading 52 GB of safetensors once per test is not an option.
    """
    if request.config.getoption("--skip-model-load"):
        return {}
    from models.demos.qwen_3_8_27b_d_p.tt.weights import load_text_backbone_state_dict, resolve_checkpoint_path

    path = resolve_checkpoint_path(required=False)
    if path is None:
        return {}
    return load_text_backbone_state_dict(path)
