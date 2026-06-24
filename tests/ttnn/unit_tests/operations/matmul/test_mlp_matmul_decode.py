# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-layer GeGLU MLP: WITH vs WITHOUT matmul_decode.

The MLP is the pi0.5 Gemma-300M action-expert block:

    out = down_proj( gelu(gate_proj(x)) * up_proj(x) )

with width=1024, mlp_dim=4096 (gate/up: 1024->4096, down: 4096->1024), bfloat8_b weights
and bfloat16 activations. M (sequence) = 32 -- one 32x32 tile, e.g. a 10-action decode
chunk padded to a tile.

One parametrized test runs the layer's MLP three ways and asserts each against the SAME
torch reference (PCC >= 0.99), then prints each path's warm latency so with/without is
directly comparable:

  * ``linear``        -- WITHOUT matmul_decode: ttnn.linear for all three projections
                         (the production default, interleaved).
  * ``decode``        -- WITH matmul_decode: partial-width-sharded gate/up/down with
                         N-packing (Nc=128, 32 output cores) and the reshard-before-down
                         recipe; separate ttnn.gelu. matmul_decode multicasts the full
                         activation to every output core, so feeding ``down`` a 2-core
                         activation (vs the 32-core gate/up output) cuts ``down`` from
                         ~35us to ~9us.
  * ``decode_fused``  -- WITH matmul_decode + ``fused_gelu=True``: the gate activation is
                         applied inside the matmul's output pack, so the gate projection
                         needs no separate elementwise gelu op.

Run:
    pytest tests/ttnn/unit_tests/operations/matmul/test_mlp_matmul_decode.py -x -s
"""

import time

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

WIDTH = 1024
MLP_DIM = 4096
M = 32
K_BLOCKS = 2  # partial-width-sharded K split (>=2 required for the cross-core K reduction)
N_BLOCKS = 32  # gate/up N split -> Nc=128 (4 tiles), output width-sharded on 32 cores
RESHARD_CORES = 2  # feed `down` a 2-core activation (cheap multicast gather) -- the key recipe
PCC = 0.99

# Weights are bfloat8_b, activations bfloat16 -- the production dtype policy.
_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)
_L1 = ttnn.L1_MEMORY_CONFIG


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


# --------------------------------------------------------------------------- MLP builders
# Each builder uploads weights ONCE and returns a `run()` closure that does only the
# forward, so the latency measurement excludes weight upload.
def _build_linear(device, x_torch, w):
    """WITHOUT matmul_decode -- ttnn.linear GeGLU (bf8_b weights, bf16 activations)."""
    x = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=_L1)
    gw = ttnn.from_torch(w["gate"].t().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    uw = ttnn.from_torch(w["up"].t().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    dw = ttnn.from_torch(w["down"].t().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)

    def run():
        # bf16 activations (matmul_decode also outputs bf16) for a fair, equal-precision compare.
        gate = ttnn.linear(x, gw, memory_config=_L1, compute_kernel_config=_HIFI2)
        up = ttnn.linear(x, uw, memory_config=_L1, compute_kernel_config=_HIFI2)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
        hidden = ttnn.multiply(gate, up, memory_config=_L1)
        out = ttnn.linear(hidden, dw, memory_config=_L1, compute_kernel_config=_LOFI)
        for t in (gate, up, hidden):
            ttnn.deallocate(t)
        return out

    return run


def _build_matmul_decode(device, x_torch, w, fused_gelu):
    """WITH matmul_decode -- partial-width-sharded gate/up/down + reshard-before-down.
    fused_gelu=True folds the gate activation into the gate matmul (no separate gelu op)."""
    tile = ttnn.Tile((32, 32))
    x = _width_sharded_A(device, x_torch, WIDTH, RESHARD_CORES, tile)
    gate_b = _partial_width_sharded_B(device, w["gate"].t().contiguous(), N_BLOCKS)  # K=1024,N=4096 -> Nc=128
    up_b = _partial_width_sharded_B(device, w["up"].t().contiguous(), N_BLOCKS)
    down_b = _partial_width_sharded_B(device, w["down"].t().contiguous(), N_BLOCKS)  # K=4096,N=1024
    mc2 = ttnn.create_sharded_memory_config(
        (M, MLP_DIM // RESHARD_CORES),
        core_grid=_crs(device, RESHARD_CORES),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    def run():
        gate = ttnn.matmul_decode(
            x, gate_b, partial_width_sharded=True, compute_kernel_config=_HIFI2, fused_gelu=fused_gelu
        )
        up = ttnn.matmul_decode(x, up_b, partial_width_sharded=True, compute_kernel_config=_HIFI2)
        if not fused_gelu:
            gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
        hidden = ttnn.multiply(gate, up, memory_config=gate.memory_config())  # (M, mlp_dim) sharded on 32 cores
        # RESHARD 32 -> 2 cores: give `down` a few-core activation (the key optimization).
        hidden_2c = ttnn.to_memory_config(hidden, mc2)
        out = ttnn.matmul_decode(hidden_2c, down_b, partial_width_sharded=True, compute_kernel_config=_LOFI)
        for t in (gate, up, hidden, hidden_2c):
            ttnn.deallocate(t)
        return out

    return run


def _latency_ms(device, run, reps=20):
    for _ in range(5):  # warm-up / kernel compile
        ttnn.deallocate(run())
        ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        o = run()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2]


# --------------------------------------------------------------------------- test
@pytest.mark.parametrize("mode", ["linear", "decode", "decode_fused"])
def test_mlp_with_without_matmul_decode(device, mode):
    """One denoise-layer MLP, with vs without matmul_decode; each must match torch (PCC >= 0.99)."""
    w = _make_weights()
    x = torch.randn(M, WIDTH, dtype=torch.float32)
    ref = _reference(x, w)

    if mode == "linear":
        run = _build_linear(device, x.to(torch.bfloat16), w)
    else:
        run = _build_matmul_decode(device, x.to(torch.bfloat16), w, fused_gelu=(mode == "decode_fused"))

    out = ttnn.to_torch(run()).float()
    lat = _latency_ms(device, run)
    print(f"\n[MLP {mode:13s}] latency median = {lat:.3f} ms  (PCC asserted >= {PCC})")
    assert_with_pcc(ref, out, PCC)
