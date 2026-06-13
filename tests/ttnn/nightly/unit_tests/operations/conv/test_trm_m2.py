# M2 TileRowMajor (TRM) verification matrix — NON-pin dedicated-partials base.
# Re-enables TRM for HEIGHT_SHARDED no-bias convs whose per_core_N is stranded by the
# SubblockMajor out_subblock_w == per_core_N constraint, completing the matrix with NO pin
# and NO caveats: l1_acc ON and OFF, ROW_MAJOR and TILE output.
#
# Eligible TRM canaries:
#   • vanilla-UNet 288<-288 60x80 HS 3x3 s1 p1 (per_core_N=9 -> relaxed 2x3 or similar)
#   • 64-in awkward-N: out=416 (per_core_N=13), out=352 (per_core_N=11), act_block_h=64 (K=2 blocks)
# The full {l1_acc on/off} x {ROW_MAJOR, TILE} cross product engages TRM (confirm via the
# CONV_TILE_PACK_ROW_MAJOR define in the fresh JIT kernel_args.csv and the host "ELIGIBLE" log).
# Bias variant must stay SubblockMajor (defensive degrade) — covered by has_bias=True rows.

import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map  # noqa: F401


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, output_channels, input_hw, config_override",
    [
        (288, 288, (60, 80), {"act_block_h": 128}),  # per_core_M=4, per_core_N=9 awkward
        (64, 416, (120, 160), {"act_block_h": 64}),  # per_core_N=13, multi-act-block K=2
        (64, 352, (120, 160), {"act_block_h": 64}),  # per_core_N=11, multi-act-block K=2
    ],
)
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("packer_l1_acc", [True, False])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat16, ttnn.float32])
def test_trm_eligible_m2(
    device,
    torch_tensor_map,
    input_channels,
    output_channels,
    input_hw,
    config_override,
    output_layout,
    packer_l1_acc,
    weights_dtype,
):
    if weights_dtype == ttnn.float32 and input_channels == 288:
        # fp32 weights at 288-in double the weight CB; statically-allocated CBs grow to ~1.62MB,
        # beyond the 1.5MB L1 cap — for ALL paths (SubblockMajor too). Physical limit, not TRM.
        pytest.skip("fp32 weights + 288-in OOM L1 for all paths — physical, not TRM")
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,  # output_dtype
        weights_dtype,
        1,  # batch
        output_channels,
        input_channels,
        input_hw[0],
        input_hw[1],
        3,
        3,
        1,
        1,
        (1, 1),
        config_override,
        packer_l1_acc=packer_l1_acc,
        has_bias=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=output_layout,
    )


# Non-eligible: same shapes WITH bias must stay SubblockMajor (TRM forbids bias). PCC-only sanity
# that the defensive degrade path is correct under both l1_acc settings and both layouts.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("packer_l1_acc", [True, False])
def test_trm_bias_stays_sbm_m2(device, torch_tensor_map, output_layout, packer_l1_acc):
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat16,
        ttnn.bfloat16,
        1,
        288,
        288,
        60,
        80,
        3,
        3,
        1,
        1,
        (1, 1),
        {"act_block_h": 128},
        packer_l1_acc=packer_l1_acc,
        has_bias=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        output_layout=output_layout,
    )
