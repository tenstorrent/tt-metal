# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks for the FabricType binding and the get_all_mgd_fabric_types export.

The dim_types -> FabricType inference itself is covered by the C++ offline test
(test_mesh_graph_descriptor.cpp). These guard the Python surface: that both symbols are
exported from ttnn and that the enum keeps its expected members and bitflag relationship,
so a binding/enum regression is caught before the Blaze consumer relies on it.
"""

import ttnn


def test_fabric_type_exported():
    for name in ("MESH", "TORUS_X", "TORUS_Y", "TORUS_XY"):
        assert hasattr(ttnn.FabricType, name), f"ttnn.FabricType missing {name}"


def test_fabric_type_is_bitflag():
    # Bound with nb::is_arithmetic(); TORUS_XY is the OR of the single-axis torus flags.
    assert int(ttnn.FabricType.TORUS_XY) == int(ttnn.FabricType.TORUS_X) | int(ttnn.FabricType.TORUS_Y)


def test_get_all_mgd_fabric_types_exported():
    assert callable(ttnn.get_all_mgd_fabric_types)
