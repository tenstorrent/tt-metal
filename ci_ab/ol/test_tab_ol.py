# Scratch op-level cases (not for merge): ttnn max_pool2d, avg_pool2d and bilinear upsample, the tilizeA_B block users, on height sharded inputs so
# in_ntiles_c = C / 32 per core. Every call appends "<case> <sha256 of the output>" to $TT_METAL_CACHE/sha.txt (one cache per arm).
import hashlib
import os
import sys

import pytest
import torch
import ttnn

# name: (kind, (N, C, H, W), kernel or scale, stride, padding)
CASES = {
    "max3x3s2_1x256x56x56": ("max", (1, 256, 56, 56), (3, 3), (2, 2), (1, 1)),
    "max2x2s2_1x128x112x112": ("max", (1, 128, 112, 112), (2, 2), (2, 2), (0, 0)),
    "avg7x7_1x512x7x7": ("avg", (1, 512, 7, 7), (7, 7), (1, 1), (0, 0)),
    "max3x3s2_1x512x28x28": ("max", (1, 512, 28, 28), (3, 3), (2, 2), (1, 1)),
    "bil2x_1x256x32x32": ("bil", (1, 256, 32, 32), (2, 2), None, None),
    "bil2x_1x512x14x14": ("bil", (1, 512, 14, 14), (2, 2), None, None),
    "ctrl_max3x3s2_1x64x112x112": ("max", (1, 64, 112, 112), (3, 3), (2, 2), (1, 1)),
    "ctrl_avg3x3s2_1x96x56x56": ("avg", (1, 96, 56, 56), (3, 3), (2, 2), (1, 1)),
}


def _record(case, out):
    p = os.path.join(os.environ.get("TT_METAL_CACHE", "/tmp"), "sha.txt")
    with open(p, "a") as f:
        f.write(f"{case} {hashlib.sha256(out.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]}\n")


@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 24576}, id="l1s")], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_tab(device, case):
    kind, (n, c, h, w), k, s, p = CASES[case]
    if kind == "bil":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../tests/ttnn/unit_tests/operations/pool"))
        sys.path.insert(0, "/work/tests/ttnn/unit_tests/operations/pool")
        import test_upsample

        _, out = test_upsample.upsample_multicore_common(
            device, [n, c, h, w], k[0], k[1], ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, mode="bilinear",
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, run_twice=False)
        _record(case, out.to(torch.bfloat16))
        return
    g = torch.Generator().manual_seed(7)
    x = torch.randn((n, c, h, w), generator=g).to(torch.bfloat16)
    x = x.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)
    tx = ttnn.to_device(ttnn.from_torch(x, ttnn.bfloat16), device)
    fn = ttnn.max_pool2d if kind == "max" else ttnn.avg_pool2d
    kw = dict(dilation=[1, 1]) if kind == "max" else {}
    y = fn(input_tensor=tx, batch_size=n, input_h=h, input_w=w, channels=c, kernel_size=list(k), stride=list(s),
           padding=list(p), ceil_mode=False, memory_config=None,
           applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, deallocate_input=False,
           reallocate_halo_output=True, dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
    _record(case, ttnn.to_torch(y).to(torch.bfloat16))
