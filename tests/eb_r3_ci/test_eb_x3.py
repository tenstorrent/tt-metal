# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723): single-device calls of two kernels no single-chip test reaches.
- layernorm_post_allgather_welford.cpp: the distributed layer norm with use_welford=True, its all-gather simulated by
  computing each width slice's statistics with layer_norm_pre_all_gather and concatenating them on the host (the shapes of
  test_distributed_layernorm_exhaustive.py, 8 devices).
- moreh_layer_norm_large_kernel.cpp: moreh layer_norm with a normalized width that does not fit L1 (the large algorithm)."""
import importlib.util

import pytest
import torch
import ttnn


@pytest.mark.parametrize("seq_len, hidden_dim", [(1024, 4096), (2048, 8192), (512, 2048)])
def test_ln_post_allgather_welford(device, seq_len, hidden_dim):
    n = 8
    torch.manual_seed(42)
    x = torch.randn(1, 1, seq_len, hidden_dim)
    w = torch.randn(hidden_dim)
    b = torch.randn(hidden_dim)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    pc = ttnn.LayerNormDefaultProgramConfig(legacy_reduction=False, use_welford=True)
    width = hidden_dim // n
    grid = device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    recip = ttnn.create_layer_norm_reciprocals(device, crs, width)
    xs = [ttnn.from_torch(x[..., i * width : (i + 1) * width], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16) for i in range(n)]
    stats = [
        ttnn.to_torch(ttnn.layer_norm_pre_all_gather(t, compute_kernel_config=ckc, dtype=ttnn.bfloat16, program_config=pc, recip_tensor=recip))
        for t in xs
    ]
    gathered = ttnn.from_torch(torch.cat(stats, dim=-1), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    outs = []
    for i in range(n):
        wi = ttnn.from_torch(w[i * width : (i + 1) * width].reshape(1, 1, 1, width), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        bi = ttnn.from_torch(b[i * width : (i + 1) * width].reshape(1, 1, 1, width), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        o = ttnn.layer_norm_post_all_gather(xs[i], gathered, epsilon=1e-5, weight=wi, bias=bi, compute_kernel_config=ckc, program_config=pc)
        outs.append(ttnn.to_torch(o))
    got = torch.cat(outs, dim=-1).float()
    ref = torch.nn.functional.layer_norm(x, (hidden_dim,), w, b, 1e-5)
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.999


_P = "tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py"
_spec = importlib.util.spec_from_file_location("eb_moreh_ln", _P)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


@pytest.mark.parametrize("input_shape_normalized_dims", [([1, 64, 32768], 1), ([2, 32, 65536], 1)], ids=["64x32768", "2x32x65536"])
@pytest.mark.parametrize("elementwise_affine", [False, True], ids=["affine=False", "affine=True"])
def test_moreh_layer_norm_large(device, input_shape_normalized_dims, elementwise_affine):
    torch.manual_seed(2024)
    _m.run_moreh_layer_norm(input_shape_normalized_dims, elementwise_affine, 1e-5, ttnn.bfloat16, device)
