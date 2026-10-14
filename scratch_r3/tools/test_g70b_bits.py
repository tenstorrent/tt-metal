# #58817: the Llama 3.3 70B Galaxy FF1/FF3 prefill cases with the DRAM-sharded w1 (test_sync_prof.py's g70b_*_real) that
# read different from main on the merge with 3c202f9944d; outputs and the dequantized inputs saved for a golden compare.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

from test_sync_prof import MM_CASES, _ff13_pc_2d, _w1_sharded

OUT = _os.environ.get("BITID_OUT")
CASES = [c for c in MM_CASES if c[0].endswith("_real")]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("sync", ["half", "full", "half2"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_g70b(device, case, sync):
    name, s0, s1, d0, d1, do, fid, approx, fp32, pc, mem1 = case
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(s0) * 0.1, dtype=d0, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b = ttnn.from_torch(torch.randn(s1) * 0.02, dtype=d1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem1(device))
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=approx, fp32_dest_acc_en=fp32,
        packer_l1_acc=True, dst_full_sync_en=(sync == "full"),
    )
    out = ttnn.matmul(a, b, compute_kernel_config=cfg, dtype=do, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc())
    ttnn.synchronize_device(device)
    if OUT:
        torch.save(ttnn.to_torch(out).contiguous().view(torch.int16), _os.path.join(OUT, f"g70b_{name}_{sync}.pt"))
        torch.save((ttnn.to_torch(a).float(), ttnn.to_torch(b).float()), _os.path.join(OUT, f"in_{name}_{sync}.ptin"))
