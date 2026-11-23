# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The cross-stage activation of a pipeline-parallel Hy4 prefill under iHC, over the real D2D socket service.

Under iHC (#65891) a stage hands the next one the 4 packed fp32 residual streams, so the runner sizes the D2D
activation from the adapter (``pipeline_activation_streams`` / ``pipeline_activation_dtype``):
``activation_global_spec(rows, d2d_activation_width(hidden, streams=4), dtype=float32)``. This pushes one such
activation (a 4K chunk at the 4x4 stage shape: SP 4 on the rows, the 4 * 6144 width over the 4 columns, fp32 TILE)
from the top half of the 8x4 Galaxy to the bottom half with ``outbound_socket_service_sync`` /
``inbound_socket_service_sync`` and checks every shard bit-exact, twice (program cache), as ``test_d2d_socket_sync.py``
does for its uint32 stand-in. The model-side handoff is ``test_hy4_prefill_ihc_pp_split.py``.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.runner_utils import activation_global_spec, d2d_activation_width
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_d2d_socket_sync import _assert_shards_equal, _shard_mapper
from models.demos.deepseek_v3_d_p.tt.hy4.ihc import residual_layout
from models.demos.deepseek_v3_d_p.tt.runners.adapters.hy4_preview import Hy4Adapter

_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))
_FIFO_SIZE_BYTES = 16384
_CHUNK = 4096


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((8, 4), torus_xy_device_params(), id="torus-xy-8x4")],
    indirect=["mesh_device", "device_params"],
)
def test_hy4_ihc_d2d_activation(mesh_device):
    if not is_blackhole():
        pytest.skip("D2DStreamService needs Blackhole service cores")
    adapter = Hy4Adapter()
    streams, dtype = residual_layout(Hy4Config)
    # The adapter (runner sizing) and the model (what a stage's forward returns) must agree.
    assert (adapter.pipeline_activation_streams, getattr(ttnn, adapter.pipeline_activation_dtype)) == (streams, dtype)

    half, cols = mesh_device.shape[0] // 2, mesh_device.shape[1]
    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(0, 0))
    receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(half, 0))
    sender_mesh.enable_program_cache()
    receiver_mesh.enable_program_cache()

    width = d2d_activation_width(Hy4Config.EMB_SIZE, streams=streams)
    spec = activation_global_spec(_CHUNK, width, 1, dtype)
    assert tuple(spec.shape) == (1, 1, _CHUNK, 4 * 6144) and spec.dtype == ttnn.float32
    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=spec,
        mapper=_shard_mapper(sender_mesh),
        fifo_size_bytes=_FIFO_SIZE_BYTES,
        sender_worker_cores=_WORKER_CORES,
        receiver_worker_cores=_WORKER_CORES,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=0,
        share_fabric_links=False,
    )
    try:
        for it in range(2):
            # fp32 values bf16 cannot hold: a bf16 wire would round them.
            x = torch.randn(1, 1, _CHUNK, width, generator=torch.Generator().manual_seed(it)) * 37.0 + 1e-3
            tt_in = ttnn.from_torch(
                x,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=sender_mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_shard_mapper(sender_mesh),
            )
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(sender, tt_in)
            recv = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(receiver, metadata_size_bytes=0)[0]
            assert recv.dtype == ttnn.float32
            _assert_shards_equal(tt_in, recv, f"iHC fp32 activation, iter {it}")
            ttnn.deallocate(tt_in)
    finally:
        del receiver
        del sender
