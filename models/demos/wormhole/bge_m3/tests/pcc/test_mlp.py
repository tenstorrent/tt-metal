# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.device import is_blackhole as ttnn_is_blackhole

import ttnn
from models.demos.wormhole.bge_m3.reference.hf_reference import PositionwiseFeedForward
from models.demos.wormhole.bge_m3.tests.test_utils import (
    SEQUENCE_LENGTHS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)
from models.demos.wormhole.bge_m3.tt.mlp import BgeM3MLP
from models.demos.wormhole.bge_m3.tt.optimizations import (
    _WORMHOLE_L1_UNRESERVED_BASE_BYTES,
    Optimizations,
    _minimal_matmul_fits_wormhole_l1,
    _minimal_matmul_l1_bytes,
)

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
BATCH_SIZE = 1


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_mlp_vs_pytorch(device, seq_len):
    require_single_device(device)
    torch.manual_seed(42)

    reference_layer = PositionwiseFeedForward(
        hidden_size=HIDDEN_SIZE,
        mlp_size=INTERMEDIATE_SIZE,
        drop_prob=0.0,
    ).eval()
    with torch.no_grad():
        reference_layer.proj1.weight.copy_(torch.randn_like(reference_layer.proj1.weight) * 0.02)
        reference_layer.proj1.bias.copy_(torch.randn_like(reference_layer.proj1.bias) * 0.01)
        reference_layer.proj2.weight.copy_(torch.randn_like(reference_layer.proj2.weight) * 0.02)
        reference_layer.proj2.bias.copy_(torch.randn_like(reference_layer.proj2.bias) * 0.01)

    x = torch.randn((BATCH_SIZE, 1, seq_len, HIDDEN_SIZE), dtype=torch.float32)

    tt_model = BgeM3MLP(
        wi_weight=make_lazy_weight(
            reference_layer.proj1.weight.detach().clone().transpose(-1, -2).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        wo_weight=make_lazy_weight(
            reference_layer.proj2.weight.detach().clone().transpose(-1, -2).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        wi_bias=make_lazy_weight(
            reference_layer.proj1.bias.detach().clone().reshape(1, -1).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        wo_bias=make_lazy_weight(
            reference_layer.proj2.bias.detach().clone().reshape(1, -1).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        activation="gelu",
    )

    tt_output = tt_model.forward(to_ttnn_tensor(x, device))
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, HIDDEN_SIZE))

    reference_output = reference_layer(x.squeeze(1)).unsqueeze(1).to(torch.float32)
    assert_pcc(reference_output, tt_output_torch, 0.999)


# The two S8192 minimal_matmul configs Optimizations resolves on Wormhole (optimizations.py).
_WI_S8192 = ttnn.MinimalMatmulConfig(
    M_block_size=16,
    K_block_size=16,
    N_block_size=8,
    subblock_h=4,
    subblock_w=2,
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
)
_WO_S8192 = ttnn.MinimalMatmulConfig(
    M_block_size=8,
    K_block_size=32,
    N_block_size=4,
    subblock_h=4,
    subblock_w=2,
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
)
_BF16 = ttnn.bfloat16
_BF8 = ttnn.bfloat8_b


@pytest.mark.parametrize(
    "config, in0_dtype, weight_dtype, out_dtype, region_end, fits",
    [
        # The error the bf16 pcc build hit after #58149: "grow to 2481376 B which is beyond max L1 size of 1499136 B".
        pytest.param(_WI_S8192, _BF16, _BF16, _BF16, 2_481_376, False, id="wi-bf16"),
        pytest.param(_WI_S8192, _BF8, _BF8, _BF8, 1_490_656, True, id="wi-bf8"),
        # bf8 model with the mlp_wi_output_dtype override: the wider output alone breaks the fit.
        pytest.param(_WI_S8192, _BF8, _BF8, _BF16, 1_736_416, False, id="wi-bf8-bf16-out"),
        pytest.param(_WO_S8192, _BF16, _BF16, _BF16, 1_883_360, False, id="wo-bf16"),
        pytest.param(_WO_S8192, _BF8, _BF8, _BF8, 1_080_800, True, id="wo-bf8"),
    ],
)
def test_s8192_minimal_matmul_l1_footprint(config, in0_dtype, weight_dtype, out_dtype, region_end, fits):
    """Pins the footprint arithmetic to the numbers minimal_matmul_program_descriptor.cpp produces
    (in0 / in1 / out double buffered, one Float16_b intermediate block, one bias block)."""
    dtypes = dict(
        in0_dtype=in0_dtype,
        in1_dtype=weight_dtype,
        out_dtype=out_dtype,
        bias_dtype=weight_dtype,
        fp32_dest_acc_en=False,
    )
    assert _WORMHOLE_L1_UNRESERVED_BASE_BYTES + _minimal_matmul_l1_bytes(config, **dtypes) == region_end
    assert _minimal_matmul_fits_wormhole_l1(config, **dtypes) is fits


def test_s8192_minimal_matmul_configs_follow_the_l1_fit(device):
    """Optimizations keeps the S8192 minimal_matmul configs only for dtype mixes whose buffers fit."""
    require_single_device(device)
    if ttnn_is_blackhole(device):
        pytest.skip("the S8192 minimal_matmul configs are Wormhole-only")
    grid = device.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 8:
        pytest.skip("the S8192 minimal_matmul configs need an 8x8 grid")

    bf8 = Optimizations.build(device, max_batch_size=1, max_seq_len=8192, dtype=_BF8)
    assert bf8.mlp.wi_minimal_config is not None and bf8.mlp.wo_minimal_config is not None
    assert bf8.mlp.wi_prg_config is None and bf8.mlp.wo_prg_config is None

    bf16 = Optimizations.build(device, max_batch_size=1, max_seq_len=8192, dtype=_BF16)
    assert bf16.mlp.wi_minimal_config is None and bf16.mlp.wo_minimal_config is None

    mixed = Optimizations.build(device, max_batch_size=1, max_seq_len=8192, dtype=_BF8, mlp_wi_output_dtype=_BF16)
    assert mixed.mlp.wi_minimal_config is None and mixed.mlp.wo_minimal_config is None
