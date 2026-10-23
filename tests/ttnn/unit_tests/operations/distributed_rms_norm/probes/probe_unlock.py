# Verifier probe (not collected: run with `scripts/run_safe_pytest.sh --dev <this file>`; pytest collects it when named explicitly).
"""Probe: do h_non_aligned / bfloat8_b already work when validate() is bypassed? (verifier drift check)"""
import pytest
import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401
from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    quantize_like_device,
    check_output,
)
from ttnn.operations.distributed_rms_norm import default_compute_kernel_config
from ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor import run_distributed_rms_norm

CASES = [
    ((1, 1, 1, 1792), ttnn.bfloat16),
    ((1, 1, 17, 1280), ttnn.bfloat16),
    ((1, 1, 100, 512), ttnn.float32),
    ((1, 17, 128), ttnn.bfloat16),
    ((17, 64), ttnn.float32),
    ((1, 1, 32, 1792), ttnn.bfloat8_b),
    ((1, 1, 640, 1792), ttnn.bfloat8_b),
    ((2, 4, 128, 512), ttnn.bfloat8_b),
    ((1, 1, 100, 512), ttnn.bfloat8_b),
]


@pytest.mark.parametrize("use_gamma", [True, False])
@pytest.mark.parametrize("shape,dtype", CASES, ids=lambda v: str(v))
def test_unlock(shape, dtype, use_gamma, mesh_device):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    td = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = quantize_like_device(torch.randn((rows, cols, *shape)).to(td), dtype)
    g = torch.randn((rows, cols, 1, 1, 1, shape[-1])).to(torch.bfloat16) if use_gamma else None
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if use_gamma else None
    out = run_distributed_rms_norm(
        tx,
        gamma=tg,
        cluster_axis=1,
        epsilon=1e-6,
        ring=False,
        num_links=1,
        compute_kernel_config=default_compute_kernel_config(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    expected = pytorch_distributed_rms_norm(x, g, 1, 1e-6)
    check_output(out, expected, shape=shape, dtype=dtype, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT)
