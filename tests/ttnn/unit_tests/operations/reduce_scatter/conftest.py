# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""reduce_scatter acceptance-test conftest — deliberately empty.

This is a MULTI-DEVICE (mesh_device) suite. The usual per-op
``@pytest.mark.use_module_device`` marker targets the root single-device
``device`` fixture, which these tests never request — applying it here is at
best inert and at worst misleading during triage. The other CCL op suites
(all_reduce, all_gather, point_to_point) likewise carry no conftest; the root
conftest's function-scoped ``mesh_device`` fixture (parametrized indirectly
with the topology's mesh shape + ``device_params={"fabric_config":
ttnn.FabricConfig.FABRIC_1D}``) is the whole device-management story.

Do not add a local ``device`` or ``mesh_device`` fixture here — it would shadow
the root fixtures.
"""
