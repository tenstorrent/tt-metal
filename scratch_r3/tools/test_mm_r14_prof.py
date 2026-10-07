# Round 3 matmul r14 (#58714 verdict): ttnn.experimental.minimal_matmul under the device profiler, the per-device prefill
# matmuls of models/demos/llama31_8b_qb2 (Llama 3.1 8B on a 1x4 ring: qkv, o, down with the model's prefill config, 2x4 sub
# blocks, K block 8, LoFi, 16-bit DEST, bf16 activations, bfp8 or bfp4 weights; gate_up with fuse_swiglu and the MLP config),
# plus a default-config Float32-accumulate control (2x2 sub blocks, which no form changes).
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

CASES = [
    # (name, M, K, N, in1 dtype, config kind, fuse_swiglu, fidelity, fp32 dest)
    ("llama_qkv_2k", 2048, 4096, 1536, "bfp8", "attn", False, "LoFi", False),
    ("llama_o_2k", 2048, 1024, 4096, "bfp8", "attn", False, "LoFi", False),
    ("llama_down_2k", 2048, 3584, 4096, "bfp8", "attn", False, "LoFi", False),
    ("llama_gate_up_swiglu_2k", 2048, 4096, 7168, "bfp4", "mlp", True, "LoFi", False),
]
DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
_INPUTS = {}
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


@pytest.fixture(scope="module")
def device():
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    yield dev
    ttnn.close_device(dev)


def _config(device, kind):
    grid = device.compute_with_storage_grid_size()
    if kind == "attn":
        return ttnn.MinimalMatmulConfig(
            M_block_size=4, K_block_size=8, N_block_size=16, subblock_h=2, subblock_w=4, compute_with_storage_grid_size=grid
        )
    if kind == "mlp":
        return ttnn.MinimalMatmulConfig(
            M_block_size=4,
            K_block_size=8,
            N_block_size=24,
            subblock_h=2,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(min(11, grid.x), min(8, grid.y)),
        )
    return None


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_mm_prof(device, case):
    name, m, k, n, d1, kind, swiglu, fid, fp32 = case
    if name not in _INPUTS:  # built once per case: host-side tilizing of the big weights would dominate the repetitions
        torch.manual_seed(0)
        _INPUTS.clear()
        _INPUTS[name] = (
            ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(torch.randn(k, n) * 0.1, dtype=DT[d1], layout=ttnn.TILE_LAYOUT, device=device),
        )
    a, b = _INPUTS[name]
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=FID[fid], math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
    )
    out = ttnn.experimental.minimal_matmul(
        a,
        b,
        config=_config(device, kind),
        compute_kernel_config=ckc,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        fuse_swiglu=swiglu,
    )
    ttnn.synchronize_device(device)
    out.deallocate()
