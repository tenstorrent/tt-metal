# Round 3 matmul (07 evidence rule 3): every bf16 bit pattern (65536, NaN, inf, denormals and signed zero included) through the
# kernels #58817 changes, on the paths it changes: the bmm kernel with an 8-tile sub block at in0_block_w above 1 (the row
# MOP), the same with a bfp8 in1 (the 8-bit stream body), minimal_matmul with 2x4 sub blocks (its row MOP). The outputs are
# saved as raw bits (BITID_OUT) so main's and the branch's can be compared bit for bit.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import os

import pytest
import torch
import ttnn

FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi4": ttnn.MathFidelity.HiFi4}


def all_bf16(rows, cols):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    return bits.view(torch.bfloat16).reshape(rows, cols)


def rnd(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 0.5).to(torch.bfloat16)


CASES = [
    # (name, kernel, A, B, in1 dtype, fidelity)
    ("bmm_allA_lofi", "bmm", "all_64x1024", "rnd_1024x256", "bf16", "LoFi"),
    ("bmm_allA_hifi4", "bmm", "all_64x1024", "rnd_1024x256", "bf16", "HiFi4"),
    ("bmm_allB_lofi", "bmm", "rnd_64x256", "all_256x256", "bf16", "LoFi"),
    ("bmm_allB_hifi4", "bmm", "rnd_64x256", "all_256x256", "bf16", "HiFi4"),
    ("bmm_allA_bfp8_lofi", "bmm", "all_64x1024", "rnd_1024x256", "bfp8", "LoFi"),
    ("bmm_allA_bfp8_hifi4", "bmm", "all_64x1024", "rnd_1024x256", "bfp8", "HiFi4"),
    ("mm_allA_lofi", "minimal", "all_64x1024", "rnd_1024x256", "bf16", "LoFi"),
    ("mm_allA_hifi4", "minimal", "all_64x1024", "rnd_1024x256", "bf16", "HiFi4"),
    ("mm_allB_lofi", "minimal", "rnd_64x256", "all_256x256", "bf16", "LoFi"),
    ("mm_allA_bfp8_lofi", "minimal", "all_64x1024", "rnd_1024x256", "bfp8", "LoFi"),
]


def make(spec, seed):
    kind, shape = spec.split("_")
    r, c = (int(x) for x in shape.split("x"))
    return all_bf16(r, c) if kind == "all" else rnd(r, c, seed)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_bitid(device, case):
    name, kernel, a_spec, b_spec, d1, fid = case
    a = make(a_spec, 1)
    b = make(b_spec, 2)
    m, k = a.shape
    n = b.shape[1]
    ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.bfloat8_b if d1 == "bfp8" else ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    if kernel == "bmm":
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=m // 32,
            out_block_w=n // 32,
            per_core_M=m // 32,
            per_core_N=n // 32,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        out = ttnn.matmul(ta, tb, program_config=cfg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    else:
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=2, K_block_size=8, N_block_size=8, subblock_h=2, subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
        )
        out = ttnn.experimental.minimal_matmul(ta, tb, config=cfg, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    got = ttnn.to_torch(out).reshape(m, n).contiguous().view(torch.int16)
    if os.environ.get("BITID_OUT"):
        torch.save(got, os.path.join(os.environ["BITID_OUT"], f"{name}.pt"))
    assert got.shape == (m, n)
