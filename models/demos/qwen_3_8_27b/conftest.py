# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Package fixtures. The mesh graph descriptor is selected here, at import, before any cluster init."""

import pytest

from models.demos.qwen_3_8_27b.tt.mesh import select_fabric_env

select_fabric_env()


@pytest.fixture(scope="session")
def spec():
    from models.demos.qwen_3_8_27b.config import PrefillSpec

    return PrefillSpec.load().validate()


@pytest.fixture(scope="session")
def mesh(spec):
    """The spec's target mesh (8x4 BH galaxy), opened once per session."""
    import ttnn
    from models.demos.qwen_3_8_27b.tt.mesh import close_mesh, open_mesh

    if ttnn.get_num_devices() < spec.sp * spec.tp:
        pytest.skip(f"needs {spec.sp * spec.tp} devices")
    m = open_mesh(spec.mesh_shape)
    yield m
    close_mesh(m)


@pytest.fixture(scope="session")
def mesh_config(mesh, spec):
    from models.demos.qwen_3_8_27b.tt.mesh import MeshConfig

    return MeshConfig(mesh, sp=spec.sp, tp=spec.tp)


@pytest.fixture(scope="session")
def ccl_manager(mesh):
    from models.demos.qwen_3_8_27b.tt.mesh import make_ccl_manager

    return make_ccl_manager(mesh)
