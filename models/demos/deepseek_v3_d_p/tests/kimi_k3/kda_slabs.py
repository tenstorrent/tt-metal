# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The engine-owned KDA state slabs for a transformer or KDA layer built directly by a test.

The runner allocates these from the config alone and hands them to the model at ``compile()``; a
test that builds ``TtKimiK3Transformer`` itself does the same here, since the slabs are the KDA
carries' only store and a KDA forward has nowhere to run without them.
"""

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.kda.utils import upload_slab
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, TtK3KdaAttention, build_kda_scalars
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry


def attach_kda_states(mesh_device, model, *, sp_axis: int, tp_axis: int, num_slots: int = 1) -> KdaStates | None:
    """Allocate this rank's slabs and put them on ``model``; ``None`` when the rank has no KDA layer."""
    layer_ids = model.schedule.rank_kda_layer_ids()
    if not layer_ids:
        return None
    geometry = kimi_k3_kda_geometry(mesh_device.shape, sp_axis, tp_axis)
    model.kda_states = KdaStates.allocate(mesh_device, geometry, layer_ids=layer_ids, num_slots=num_slots)
    return model.kda_states


def run_kda_chunk(
    attention: TtK3KdaAttention,
    hidden: ttnn.Tensor,
    slabs: KdaStates,
    metadata: tuple[ttnn.Tensor, ...] | None,
    cache_user_id: int = 0,
    actual_start: int | None = None,
    actual_end: int | None = None,
) -> ttnn.Tensor:
    """One K3 KDA layer on a chunk, with its scalars built as ``TtKimiK3Transformer.forward`` builds them:
    from the device scalars ``metadata = (slot, actual_start, actual_end)`` when given, so a capture of this call
    decides on device, per replay, which slot it reads and whether the chunk is a request head; otherwise from
    the host ints, as an eager chunk does. OWNED output."""
    scalars, owned = build_kda_scalars(hidden.device(), metadata, cache_user_id, actual_start, actual_end)
    try:
        return attention.forward(hidden, K3AttnContext(kda_scalars=scalars, kda_states=slabs))
    finally:
        for scalar in owned:
            ttnn.deallocate(scalar)


def fill_slabs_with_noise(mesh_device, slabs: KdaStates, *, seed: int) -> None:
    """Store random carries into every slot and layer, so a stale slot holds neither zeros nor a real carry."""
    generator = torch.Generator().manual_seed(seed)
    for slab in (slabs.recurrent, slabs.convolution):
        upload_slab(slab, [torch.randn(tuple(slab.shape), generator=generator)] * mesh_device.get_num_devices())
