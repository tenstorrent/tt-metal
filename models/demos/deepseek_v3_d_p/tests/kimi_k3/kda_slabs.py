# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The engine-owned KDA state slabs for a transformer built directly by a test.

The runner allocates these from the config alone and hands them to the model at ``compile()``; a
test that builds ``TtKimiK3Transformer`` itself does the same here, since the slabs are the KDA
carries' only store and a KDA forward has nowhere to run without them.
"""

from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry


def attach_kda_states(mesh_device, model, *, sp_axis: int, tp_axis: int, num_slots: int = 1) -> KdaStates | None:
    """Allocate this rank's slabs and put them on ``model``; ``None`` when the rank has no KDA layer."""
    layer_ids = model.schedule.rank_kda_layer_ids()
    if not layer_ids:
        return None
    geometry = kimi_k3_kda_geometry(mesh_device.shape, sp_axis, tp_axis)
    model.kda_states = KdaStates.allocate(mesh_device, geometry, layer_ids=layer_ids, num_slots=num_slots)
    return model.kda_states
