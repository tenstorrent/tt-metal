# #58817 review: minimal_matmul's row MOP with a Float32 output from a 16-bit DEST. The Llama 3.1 8B prefill matmuls per
# device of the 1x4 ring (models/demos/llama31_8b_qb2: 2048 rows, 2x4 sub blocks, K block 8, LoFi, 16-bit DEST, bf16
# activations, bfp8 or bfp4 weights; gate_up with fuse_swiglu and the MLP config) with a Float32 output, plus the qkv
# shape with the model's bf16 output as a control that no variant changes.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

CASES = [
    # (name, M, K, N, in1 dtype, config kind, fuse_swiglu, output dtype)
    ("qkv_f32", 2048, 4096, 1536, "bfp8", "attn", False, "f32"),
    ("o_f32", 2048, 1024, 4096, "bfp8", "attn", False, "f32"),
    ("gate_up_swiglu_f32", 2048, 4096, 7168, "bfp4", "mlp", True, "f32"),
    ("qkv_bf16_ctl", 2048, 4096, 1536, "bfp8", "attn", False, "bf16"),
]
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "f32": ttnn.float32}
_INPUTS = {}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


def _config(device, kind):
    grid = device.compute_with_storage_grid_size()
    if kind == "attn":
        return ttnn.MinimalMatmulConfig(
            M_block_size=4, K_block_size=8, N_block_size=16, subblock_h=2, subblock_w=4, compute_with_storage_grid_size=grid
        )
    return ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=8,
        N_block_size=24,
        subblock_h=2,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(min(11, grid.x), min(8, grid.y)),
    )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mm_prof(device, case):
    name, m, k, n, d1, kind, swiglu, dout = case
    if name not in _INPUTS:
        torch.manual_seed(0)
        _INPUTS.clear()
        _INPUTS[name] = (
            ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(torch.randn(k, n) * 0.1, dtype=DT[d1], layout=ttnn.TILE_LAYOUT, device=device),
        )
    a, b = _INPUTS[name]
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    out = ttnn.experimental.minimal_matmul(
        a,
        b,
        config=_config(device, kind),
        compute_kernel_config=ckc,
        dtype=DT[dout],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        fuse_swiglu=swiglu,
    )
    ttnn.synchronize_device(device)
    od = _os.environ.get("R14_OUT")
    if od:
        torch.save(ttnn.to_torch(out), _os.path.join(od, f"{name}.pt"))
    out.deallocate()
