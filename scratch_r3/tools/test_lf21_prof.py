# Round 3 matmul, fifth look of #58713 (2026-10-08): ttnn auto-config matmuls at their default LoFi (an 8-bit in1) whose
# sub block is 2x1 at in0_block_w 1 with packer L1 accumulation, with a 16-bit and a 32-bit DEST, under the device
# profiler; the block class where the harness reads the Auto TTSync unpack 0.4 to 0.6 percent slower. Shapes picked by
# their compiled sub blocks on the P100 and P150 mock clusters.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

BF, B8 = ttnn.bfloat16, ttnn.bfloat8_b

# (name, M, K, N, in0 dtype, compute config: None = the op's default, False = LoFi with a 16-bit DEST, True = LoFi with a
# 32-bit DEST; packer L1 accumulation on)
CASES = [
    ("lf_512_4k_256_default", 512, 4096, 256, BF, None),
    ("lf_512_4k_256_fp32", 512, 4096, 256, BF, True),
    ("lf_512_4k_256_lofi", 512, 4096, 256, BF, False),
    ("lf_2k_4k_352_lofi", 2048, 4096, 352, BF, False),
    ("lf_1k_8k_224_lofi", 1024, 8192, 224, BF, False),
    ("lf_512_4k_256_b8in0_default", 512, 4096, 256, B8, None),
    ("lf_512_4k_256_b8in0_fp32", 512, 4096, 256, B8, True),
    ("lf_2k_4k_352_default", 2048, 4096, 352, BF, None),
    ("lf_2k_4k_352_fp32", 2048, 4096, 352, BF, True),
    ("lf_2k_4k_416_default", 2048, 4096, 416, BF, None),
    ("lf_2k_4k_416_fp32", 2048, 4096, 416, BF, True),
    ("lf_4k_2k_352_default", 4096, 2048, 352, BF, None),
    ("lf_4k_2k_416_default", 4096, 2048, 416, BF, None),
    ("lf_1k_8k_224_default", 1024, 8192, 224, BF, None),
    ("lf_1k_8k_224_fp32", 1024, 8192, 224, BF, True),
    ("lf_2k_2k_96_default", 2048, 2048, 96, BF, None),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_lf21_mm(device, case):
    name, m, k, n, d0, fp32 = case
    torch.manual_seed(0)
    D = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=d0, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=B8, layout=ttnn.TILE_LAYOUT, device=device, memory_config=D)
    kw = {}
    if fp32 is not None:
        kw["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=fp32,
            packer_l1_acc=True,
        )
    out = ttnn.matmul(a, b, memory_config=D, dtype=BF, **kw)
    ttnn.synchronize_device(device)
    if _os.environ.get("V12_OUT"):
        torch.save(ttnn.to_torch(out).contiguous().view(torch.int16), _os.path.join(_os.environ["V12_OUT"], f"lf21_{name}.pt"))
    out.deallocate()
    a.deallocate()
    b.deallocate()
