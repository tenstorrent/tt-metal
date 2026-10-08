# Round 3 matmul r14 (#58714 verdict): functional check of ttnn.experimental.minimal_matmul with 2x4 sub blocks and K blocks
# of 2 and 8 (the kernel's row MOP set with a 16-bit DEST): plain, padded M/K/N (narrowed edge sub blocks), bias, a fused
# activation (which keeps the tile MOP), fused SwiGLU, bf16/bfp8/bfp4 weights, LoFi to HiFi4. PCC against torch.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import os
import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
PCC = {"bf16": 0.99, "bfp8": 0.98, "bfp4": 0.95}

CASES = [
    # (name, M, K, N, blocks (M, K, N, sbh, sbw), weight dtype, fidelity, bias, activation, swiglu)
    ("plain_lofi_bfp8", 512, 1024, 1024, (4, 8, 16, 2, 4), "bfp8", "LoFi", False, None, False),
    ("plain_hifi2_bf16", 512, 1024, 1024, (4, 8, 16, 2, 4), "bf16", "HiFi2", False, None, False),
    ("plain_hifi4_bf16_k2", 256, 512, 512, (2, 2, 8, 2, 4), "bf16", "HiFi4", False, None, False),
    ("plain_lofi_bfp4", 512, 1024, 1024, (4, 8, 16, 2, 4), "bfp4", "LoFi", False, None, False),
    ("padded_lofi_bfp8", 320, 704, 352, (4, 8, 8, 2, 4), "bfp8", "LoFi", False, None, False),
    ("bias_hifi2_bf16", 512, 1024, 1024, (4, 8, 16, 2, 4), "bf16", "HiFi2", True, None, False),
    ("gelu_lofi_bf16", 512, 1024, 1024, (4, 8, 16, 2, 4), "bf16", "LoFi", False, "gelu", False),
    ("swiglu_lofi_bfp8", 512, 1024, 2048, (4, 8, 16, 2, 4), "bfp8", "LoFi", False, None, True),
    ("swiglu_bias_hifi2_bf16", 256, 512, 1024, (4, 8, 8, 2, 4), "bf16", "HiFi2", True, None, True),
]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mm_func(device, case):
    name, m, k, n, (mb, kb, nb, sbh, sbw), wd, fid, use_bias, act, swiglu = case
    torch.manual_seed(0)
    x = torch.randn(m, k)
    w = torch.randn(k, n)
    bias = torch.randn(1, n) if use_bias else None
    ref = x @ w + (bias if use_bias else 0)
    if act == "gelu":
        ref = torch.nn.functional.gelu(ref)
    tt_w_src, tt_b_src = w, bias
    if swiglu:
        gate, up = ref[:, : n // 2], ref[:, n // 2 :]
        ref = torch.nn.functional.silu(gate) * up
        tt_w_src = prepare_for_fused_swiglu(w, ndev=1, gate_is_first=True)
        if use_bias:
            tt_b_src = prepare_for_fused_swiglu(bias, ndev=1, gate_is_first=True)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(tt_w_src, dtype=DT[wd], layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(tt_b_src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if use_bias else None
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sbh,
        subblock_w=sbw,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )
    out = ttnn.experimental.minimal_matmul(
        tx,
        tw,
        bias_tensor=tb,
        fused_activation=(ttnn.UnaryOpType.GELU, False) if act == "gelu" else None,
        config=cfg,
        compute_kernel_config=ckc,
        fuse_swiglu=swiglu,
    )
    got = ttnn.to_torch(out).float().reshape(ref.shape)
    if os.environ.get("R14_OUT"):
        torch.save(got, os.path.join(os.environ["R14_OUT"], f"{name}.pt"))
    ok, pcc = comp_pcc(ref, got, PCC[wd])
    assert ok, f"{name}: PCC {pcc}"
