# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter_average test package.

Deliberately NO ``use_module_device`` marker here: these are multi-device tests
built on the root ``mesh_device`` fixture with a parametrized ``device_params``
(fabric_config), and the root conftest forbids combining the module-device
marker with parametrized device_params. The mesh must be opened per test so
``set_fabric(FABRIC_1D)`` runs before every ``open_mesh_device``. Matches the
point_to_point / all_gather / all_reduce / reduce_scatter reference suites.

Run via scripts/run_multidevice_sim_pytest.py --op reduce_scatter_average
(never run_safe_pytest.sh — it forces slow dispatch on sim and has no multichip
awareness).
"""
