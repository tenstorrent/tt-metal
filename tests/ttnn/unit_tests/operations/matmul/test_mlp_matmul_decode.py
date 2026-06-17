# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone GeGLU MLP test: baseline (ttnn.linear) vs optimized (matmul_decode).

The MLP is the pi0.5 Gemma-300M action-expert block:

    out = down_proj( gelu(gate_proj(x)) * up_proj(x) )

with width=1024, mlp_dim=4096 (so gate/up are 1024->4096, down is 4096->1024),
bfloat8_b weights and bfloat16 activations. M (sequence) defaults to 32 (a 32x32
tile; e.g. a 10-action decode chunk padded to one tile).

Two implementations, both checked against the same torch reference with PCC:

  * test_mlp_baseline         -- ttnn.linear for all three projections.
  * test_mlp_matmul_decode    -- OPTIMIZED: gate/up via matmul_decode partial-WS
      with N-packing (single call each, Nc=128, 32 output cores); gelu + multiply;
      then RESHARD the (M, mlp_dim) activation from 32 cores -> 2 cores before the
      `down` matmul_decode. The reshard is the key step: matmul_decode multicasts
      the full activation to every output core, so feeding `down` a few-core
      activation (2 cores) instead of the 32-core gate/up output cuts `down` from
      ~35us to ~9us -- bringing it near gate/up parity. (matmul_decode output is
      hardcoded to N/Nc cores, which is why gate/up land on 32 cores and the
      reshard is needed before the K-heavy `down`.)

Run:
    pytest tests/ttnn/unit_tests/operations/matmul/test_mlp_matmul_decode.py -x -s
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

WIDTH = 1024
MLP_DIM = 4096
K_BLOCKS = 2  # partial-width-sharded K split (>=2 required for the cross-core K reduction)
PCC = 0.99

# Weights are bfloat8_b, activations bfloat16 -- the production dtype policy.
_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)


def _make_weights(seed=0):
    """Random GeGLU weights. Stored as the F.linear weight (out, in); the ttnn ops
    use the transpose (in, out) = (K, N)."""
    torch.manual_seed(seed)
    return {
        "gate": torch.randn(MLP_DIM, WIDTH, dtype=torch.float32) * 0.02,  # (mlp_dim, width)
        "up": torch.randn(MLP_DIM, WIDTH, dtype=torch.float32) * 0.02,
        "down": torch.randn(WIDTH, MLP_DIM, dtype=torch.float32) * 0.02,  # (width, mlp_dim)
    }


def _reference(x, w):
    """Torch GeGLU MLP reference: down(gelu_tanh(gate(x)) * up(x))."""
    gate = F.linear(x, w["gate"])
    up = F.linear(x, w["up"])
    hidden = F.gelu(gate, approximate="tanh") * up
    return F.linear(hidden, w["down"])


# --------------------------------------------------------------------------- helpers
def _crs(device, n):
    return ttnn.num_cores_to_corerangeset(n, device.compute_with_storage_grid_size(), True)


def _width_sharded_A(device, x_torch, k, ncores, tile):
    """(M, K) activation, width-sharded across `ncores` cores, with `tile` geometry."""
    m = x_torch.shape[0]
    mc = ttnn.create_sharded_memory_config(
        (m, k // ncores),
        core_grid=_crs(device, ncores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        x_torch, layout=ttnn.TILE_LAYOUT, tile=tile, device=device, memory_config=mc, dtype=ttnn.bfloat16
    )


def _partial_width_sharded_B(device, w_kn, n_blocks):
    """Reshape/permute a (K, N) weight into the partial-width-sharded inputB layout
    (each core holds a [Kc, Nc] block) and shard it across K_BLOCKS*n_blocks cores."""
    k, n = w_kn.shape
    kc, nc = k // K_BLOCKS, n // n_blocks
    br = w_kn.reshape(K_BLOCKS, kc, n).permute(1, 0, 2).reshape(kc, n * K_BLOCKS)
    mc = ttnn.create_sharded_memory_config(
        (kc, nc),
        core_grid=_crs(device, K_BLOCKS * n_blocks),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(br, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc, dtype=ttnn.bfloat8_b)


# --------------------------------------------------------------------------- MLPs
def mlp_baseline(device, x_torch, w):
    """ttnn.linear GeGLU MLP (bf8_b weights, bf16 activations)."""
    x = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    gw = ttnn.from_torch(w["gate"].t().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    uw = ttnn.from_torch(w["up"].t().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    dw = ttnn.from_torch(w["down"].t().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    gate = ttnn.linear(x, gw, compute_kernel_config=_HIFI2)
    gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
    up = ttnn.linear(x, uw, compute_kernel_config=_HIFI2)
    hidden = ttnn.multiply(gate, up)
    return ttnn.linear(hidden, dw, compute_kernel_config=_LOFI)


def mlp_matmul_decode(device, x_torch, w):
    """Optimized matmul_decode GeGLU MLP with the reshard-before-down recipe."""
    m = x_torch.shape[0]
    tile = ttnn.Tile((32, 32))
    x = _width_sharded_A(device, x_torch, WIDTH, 2, tile)
    gate_b = _partial_width_sharded_B(device, w["gate"].t().contiguous(), 32)  # K=1024,N=4096 -> Nc=128
    up_b = _partial_width_sharded_B(device, w["up"].t().contiguous(), 32)
    down_b = _partial_width_sharded_B(device, w["down"].t().contiguous(), 32)  # K=4096,N=1024 -> Nc=32

    gate = ttnn.matmul_decode(x, gate_b, partial_width_sharded=True, compute_kernel_config=_HIFI2)
    up = ttnn.matmul_decode(x, up_b, partial_width_sharded=True, compute_kernel_config=_HIFI2)
    gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
    hidden = ttnn.multiply(gate, up, memory_config=gate.memory_config())  # (M, mlp_dim) width-sharded on 32 cores

    # RESHARD 32 -> 2 cores: give `down` a few-core activation (the key optimization).
    mc2 = ttnn.create_sharded_memory_config(
        (m, MLP_DIM // 2),
        core_grid=_crs(device, 2),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    hidden_2c = ttnn.to_memory_config(hidden, mc2)
    return ttnn.matmul_decode(hidden_2c, down_b, partial_width_sharded=True, compute_kernel_config=_LOFI)


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("m", [32], ids=["m32"])
def test_mlp_baseline(device, m):
    w = _make_weights()
    x = torch.randn(m, WIDTH, dtype=torch.float32)
    ref = _reference(x, w)
    out = ttnn.to_torch(mlp_baseline(device, x.to(torch.bfloat16), w)).float()
    assert_with_pcc(ref, out, PCC)


@pytest.mark.parametrize("m", [32], ids=["m32"])
def test_mlp_matmul_decode(device, m):
    w = _make_weights()
    x = torch.randn(m, WIDTH, dtype=torch.float32)
    ref = _reference(x, w)
    out = ttnn.to_torch(mlp_matmul_decode(device, x.to(torch.bfloat16), w)).float()
    assert_with_pcc(ref, out, PCC)
