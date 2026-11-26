# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the all_to_all_matmul pre-migration suite.

``mesh_device`` / ``_fabric_mesh_cache`` (the golden suite's 4-device focus ring) come from the parent directory's
conftest.py. ``all_to_all_matmul_op`` (route selection + generic_op guard) is defined once, in test_acceptance.py;
it is re-exported here so test_golden_cut.py calls the operation through the very same fixture.
"""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "a2amm_pre_migration_route", pathlib.Path(__file__).with_name("test_acceptance.py")
)
_route = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_route)

all_to_all_matmul_op = _route.all_to_all_matmul_op
