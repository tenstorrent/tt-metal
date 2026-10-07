# Round 3 matmul (#58714): the conv2d layers of the Blackhole ResNet50 and vgg_unet models whose conv_bmm_tilize blocks reach
# the row MOP (8-tile sub blocks, in0_block_w above 1, packer ReLU, 16-bit DEST), with the models' per-core block parameters
# on this card: height-sharded input on N cores so each core holds the model's rows, act_block_h as the model's (the
# input is fed row major in bf16, where the model passes the previous conv's bfp8 tiles). Through the
# nightly test's run_conv (golden check included), so the device runs exactly that op.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS  # noqa: F401

# (name, batch, cin, cout, h, w, k, stride, pad, cores, act_block_h, in dtype, out dtype, out layout, relu, packer_l1_acc)
CASES = [
    ("rn50_l2m2_conv2", 16, 128, 128, 28, 28, 3, 1, 1, 98, 128, "bf16", "bfp8", "tile", True, True),
    ("rn50_l2m1_conv2", 16, 128, 128, 56, 56, 3, 2, 1, 98, 128, "bf16", "bfp8", "tile", True, True),
    ("rn50_l2m1_ds", 16, 256, 512, 56, 56, 1, 2, 0, 98, 128, "bf16", "bfp8", "tile", False, True),
    ("rn50_l3m2_conv2", 16, 256, 256, 14, 14, 3, 1, 1, 98, 32, "bf16", "bfp8", "tile", True, True),
    ("vgg_s3_10", 1, 128, 256, 64, 64, 3, 1, 1, 64, 32, "bf16", "bf16", "rm", True, False),
    ("vgg_d4_conv2", 1, 64, 64, 256, 256, 3, 1, 1, 64, 512, "bf16", "bf16", "rm", True, False),
]
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_rn50m_prof(device, torch_tensor_map, case):
    _, b, ci, co, h, w, k, s, p, n, abh, din, dout, lay, relu, l1acc = case
    grid = device.compute_with_storage_grid_size()
    sharded_cfg = ttnn.create_sharded_memory_config(
        (b * h * w // n, ci),
        core_grid=ttnn.num_cores_to_corerangeset(n, grid, True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        DT[dout],
        ttnn.bfloat8_b,
        b,
        co,
        ci,
        h,
        w,
        k,
        k,
        s,
        s,
        (p, p),
        {"act_block_h": abh},
        fp32_accum=False,
        packer_l1_acc=l1acc,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_dtype=DT[din],
        output_layout=ttnn.TILE_LAYOUT if lay == "tile" else ttnn.ROW_MAJOR_LAYOUT,
        has_bias=True,
        shard_layout=HS,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
        enable_act_double_buffer=dout == "bfp8",
        enable_weights_double_buffer=dout == "bfp8",
        sharded_cfg=sharded_cfg,
    )
