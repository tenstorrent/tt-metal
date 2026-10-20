# Scratch op-level cases (not for merge): ttnn max_pool2d and avg_pool2d, the tilizeA_B block users. Every call appends
# "<case> <sha256 of the output>" to $TT_METAL_CACHE/sha.txt (one cache per arm).
import hashlib
import os

import pytest
import torch
import ttnn

# name: (pool, (N, C, H, W), kernel, stride, padding)
CASES = {
    "max3x3s2_1x64x112x112": ("max", (1, 64, 112, 112), (3, 3), (2, 2), (1, 1)),
    "max2x2s2_4x256x40x40": ("max", (4, 256, 40, 40), (2, 2), (2, 2), (0, 0)),
    "max5x5s1_1x128x32x32": ("max", (1, 128, 32, 32), (5, 5), (1, 1), (2, 2)),
    "avg2x2s1_4x256x40x40": ("avg", (4, 256, 40, 40), (2, 2), (1, 1), (0, 0)),
    "avg3x3s2_1x64x112x112": ("avg", (1, 64, 112, 112), (3, 3), (2, 2), (1, 1)),
}


def _record(case, out):
    p = os.path.join(os.environ.get("TT_METAL_CACHE", "/tmp"), "sha.txt")
    with open(p, "a") as f:
        f.write(f"{case} {hashlib.sha256(out.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]}\n")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize("case", list(CASES))
def test_tab(device, case):
    pool, (n, c, h, w), k, s, p = CASES[case]
    g = torch.Generator().manual_seed(7)
    x = torch.randn((n, c, h, w), generator=g).to(torch.bfloat16)
    x = x.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)
    tx = ttnn.to_device(ttnn.from_torch(x, ttnn.bfloat16), device)
    fn = ttnn.max_pool2d if pool == "max" else ttnn.avg_pool2d
    kw = dict(dilation=[1, 1]) if pool == "max" else {}
    y = fn(input_tensor=tx, batch_size=n, input_h=h, input_w=w, channels=c, kernel_size=list(k), stride=list(s),
           padding=list(p), ceil_mode=False, memory_config=None,
           applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED, deallocate_input=False,
           reallocate_halo_output=True, dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
    _record(case, ttnn.to_torch(y).to(torch.bfloat16))
