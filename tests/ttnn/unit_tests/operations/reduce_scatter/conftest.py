# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration for the reduce_scatter acceptance tests.

Deliberately does NOT apply @pytest.mark.use_module_device: reduce_scatter is a
multi-device CCL op whose tests parametrize the root `mesh_device` +
`device_params` fixtures (mesh shape + fabric_config) per test function. The
module-device hook targets the single-device `device` fixture and the root
conftest hard-errors when the marker is combined with a `device_params`
parametrize; the reference CCL ops (all_reduce, all_gather, point_to_point)
likewise carry no such hook. This file exists (instead of being deleted) so the
harness's conftest injector — which only skips directories that already have a
conftest — does not re-add the hook.

Do not define local `device` / `mesh_device` fixtures here; the root conftest
owns them (fabric setup/teardown included).
"""
