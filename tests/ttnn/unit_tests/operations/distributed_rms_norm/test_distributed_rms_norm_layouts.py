# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2: ROW_MAJOR γ sticks + non-tile-aligned shapes (W and H), interleaved and sharded.

- RM γ: per device (1, 1, ceil(W_local/32), 32) sticks (golden helpers.create_ttnn_gamma); the writer reads each stick
  into row 0 of faces 0 / 1 of a cb_gamma_tiles page.
- w_non_aligned: the reader zeroes the W-pad columns of each row's last tile before Σx². The `dirty_pad` cases fill the
  input's tile padding with a large value first (ttnn.fill_implicit_tile_padding, test-side input preparation only), so
  a pad column that reached Σx² would show as a gross mismatch.
- h_non_aligned: pad rows are independent rows; rank-3 / rank-4 with batch > 1 exercise the per-image ceil geometry.

Mesh: the root conftest `mesh_device` (system mesh), FABRIC_2D_TORUS_XY, group along cluster_axis 1.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import (
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

AXIS = 1
INTERLEAVED = ttnn.TensorMemoryLayout.INTERLEAVED
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
PAD_FILL = 1000.0


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


def _run(mesh, shape, placement, *, dtype, gamma_dtype, gamma_layout, dirty_pad=False, seed=11):
    x = random_stacked(mesh, shape, _TORCH[dtype], seed=seed)
    g = None
    if gamma_dtype is not None:
        g = random_stacked(mesh, (1, 1, 1, shape[-1]), _TORCH[gamma_dtype], seed=seed + 1)
    tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=TILE, memory_layout=placement)
    if dirty_pad:
        tx = ttnn.fill_implicit_tile_padding(tx, PAD_FILL, memory_config=tx.memory_config())
    tg = None if g is None else create_ttnn_gamma(g, mesh, dtype=gamma_dtype, layout=gamma_layout)
    kwargs = {} if placement == INTERLEAVED else {"memory_config": tx.memory_config()}
    out = distributed_rms_norm(tx, cluster_axis=AXIS, gamma=tg, **kwargs)
    expected = pytorch_distributed_rms_norm(x, g, AXIS, 1e-6)
    check_output(out, expected, shape=shape, dtype=dtype, fp32_dest_acc_en=True, expected_layout=TILE)


SHAPES = [
    (1, 1, 32, 50),  # W non-aligned, one tile-row
    (1, 1, 64, 1000),  # W non-aligned, R3-X decode-ish width
    (2, 1, 128, 100),  # W non-aligned, batch > 1
    (4, 128, 47),  # rank 3, W non-aligned
    (32, 17),  # rank 2, W < 16
    (1, 1, 17, 50),  # both non-aligned
    (1, 1, 100, 512),  # H non-aligned
    (1, 1, 1, 1792),  # H = 1 decode
    (1, 17, 128),  # rank 3, H non-aligned
    (17, 64),  # rank 2, H non-aligned
    (2, 3, 47, 96),  # rank 4, batch > 1, H non-aligned per image
]
_id = lambda s: "x".join(map(str, s))


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("placement", [INTERLEAVED, HEIGHT, WIDTH], ids=["interleaved", "height", "width"])
@pytest.mark.parametrize("shape", SHAPES, ids=_id)
def test_non_aligned(mesh_device, shape, placement):
    _run(mesh_device, shape, placement, dtype=ttnn.bfloat16, gamma_dtype=ttnn.bfloat16, gamma_layout=RM)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("placement", [INTERLEAVED, HEIGHT, WIDTH], ids=["interleaved", "height", "width"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 50), (1, 1, 64, 1000), (1, 1, 17, 50), (4, 128, 47)], ids=_id)
def test_dirty_w_padding(mesh_device, shape, dtype, placement):
    if placement != INTERLEAVED and len(shape) == 3:
        pytest.skip("test-side input prep: fill_implicit_tile_padding refuses a multi-slice shard (N_slices == 1)")
    _run(mesh_device, shape, placement, dtype=dtype, gamma_dtype=None, gamma_layout=None, dirty_pad=True)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("placement", [INTERLEAVED, HEIGHT, WIDTH], ids=["interleaved", "height", "width"])
@pytest.mark.parametrize("dtype,gamma_dtype", [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16)])
@pytest.mark.parametrize("shape", [(1, 1, 32, 1792), (1, 1, 640, 1792), (1, 1, 32, 8192), (1, 1, 64, 1000)], ids=_id)
def test_rm_gamma(mesh_device, shape, dtype, gamma_dtype, placement):
    if placement == HEIGHT and dtype == ttnn.float32 and shape[-1] == 8192:
        pytest.skip("infeasible L1: 1 MB fp32 input + output shards on one core (golden: INFEASIBLE_L1)")
    _run(mesh_device, shape, placement, dtype=dtype, gamma_dtype=gamma_dtype, gamma_layout=RM)
