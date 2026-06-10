# TRM+pin verification matrix (Phase 4). Kept on the feature branch for the profiling follow-on;
# drop before upstream PR.
# Eligible config: vanilla-UNet 288<-288 60x80 HS 3x3 s1 p1, no bias, packer_l1_acc,
# LoFi bf16 weights, act_block_h=128 (per_core_M=4). ROW_MAJOR engages Phase 2;
# TILE engages Phase 3. Non-eligible bias variant must stay SubblockMajor.

import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map  # noqa: F401


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_hw, config_override",
    [
        (288, (60, 80), {"act_block_h": 128}),
        (288, (60, 80), None),  # production config
        (288, (96, 96), None),  # per_core_M=5 -> relaxed 5x1
        # NOTE: multi-act-block 288-ch eligible configs (e.g. 60x80 abh=64, 120x160) exceed n150's
        # 1.5MB usable L1 (core count drops; per-core CBs grow to ~1.8-2.3MB) for ALL paths incl.
        # SubblockMajor+bias — physical limit, not TRM. Multi-act-block eligible coverage instead
        # comes from the small-K variant below (in=64, out=288 keeps per_core_N=9 awkward).
        (64, (120, 160), {"act_block_h": 64}),  # multi-act-block per core, K=2 blocks
    ],
)
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("has_bias", [False, True])
# fp32 weights at 288-in OOM n150 L1 for ALL paths (weight CB doubles); fp32 coverage lives in
# test_trm_pin_other_awkward_n below (64-in keeps weights ~660KB)
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16])
def test_trm_pin_eligible_288(
    device,
    torch_tensor_map,
    input_channels,
    config_override,
    input_hw,
    output_layout,
    has_bias,
    weights_dtype,
    output_dtype,
):
    if input_hw == (96, 96) and output_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("96x96 ROW_MAJOR (bias or not) clashes CBs with L1 buffers on n150 — physical, not TRM")
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        output_dtype,
        weights_dtype,
        1,  # batch
        288,  # out channels
        input_channels,
        input_hw[0],
        input_hw[1],
        3,
        3,
        1,
        1,
        (1, 1),
        config_override,
        packer_l1_acc=True,
        has_bias=has_bias,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=output_layout,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("output_channels", [416, 352])  # per_core_N = 13, 11 (awkward, prime)
def test_trm_pin_other_awkward_n(device, torch_tensor_map, output_layout, weights_dtype, output_channels):
    if weights_dtype == ttnn.float32 and output_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("fp32 weights + ROW_MAJOR untilize shard exceeds n150 L1 — physical, not TRM")
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,
        weights_dtype,
        1,  # batch
        output_channels,
        64,  # in channels (small K so weights fit)
        120,
        160,
        3,
        3,
        1,
        1,
        (1, 1),
        {"act_block_h": 64},  # multi-act-block per core
        packer_l1_acc=True,
        has_bias=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=output_layout,
    )
