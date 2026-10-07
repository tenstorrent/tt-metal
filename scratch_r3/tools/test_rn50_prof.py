# Round 3 matmul r14 (#58714 verdict): three ResNet50 conv2d layers of the nightly test_resnet50_conv_wh (LoFi, bfp8 weights and
# output, packer L1 accumulation, bias, no auto shard) through its run_conv, so the profiler runs see the same op as the nightly.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, BS  # noqa: F401

CASES = [
    ("rn50_l1_3x3_64", 16, 64, 64, 56, 56, 3, 3, 1, 1, 1, 1, HS),
    ("rn50_l3_3x3_256", 16, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, BS),
    ("rn50_ds3_1x1_1024", 20, 1024, 512, 28, 28, 1, 1, 2, 2, 0, 0, BS),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_rn50_prof(device, torch_tensor_map, case):
    _, b, co, ci, h, w, fh, fw, sh, sw, ph, pw, layout = case
    run_conv(
        device,
        torch_tensor_map,
        ttnn.MathFidelity.LoFi,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        b,
        co,
        ci,
        h,
        w,
        fh,
        fw,
        sh,
        sw,
        (ph, pw),
        config_override=None,
        packer_l1_acc=True,
        fp32_accum=False,
        has_bias=True,
        auto_shard=False,
        shard_layout=layout,
        input_layout=ttnn.TILE_LAYOUT,
    )
