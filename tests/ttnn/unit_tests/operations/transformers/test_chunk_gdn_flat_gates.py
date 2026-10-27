# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flat token-major q/k/v + token-major g/beta ([B,T,HV], no head-major relayout) vs the torch golden.

With flat q/k/v at chunk 32 the host wrapper hands g / beta to the prep reader as they are; the reader
picks head hv's column (hv % 32) out of tile page (b*NC + c)*HVt + hv/32 and moves it to column 0. This
checks the page / column arithmetic for B = 1 and B = 2, HV < 32 (HVt = 1), HV == 32, HV > 32 with a
partially filled second tile column (HV = 40) and HV = 48 (hv%32 spans both faces), on the fused
and phased paths the cost model picks (BH = B * HV up to 96), with fp32 and bf16 g / beta inputs.
The num_key_heads cases run K != V (the sequence-parallel GDN fold: H = 2, HV = 6, K = 128, V = 256) on the auto,
fused and phased paths and check that the host wrapper rejects a num_key_heads that does not fit the inputs.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(
    not is_blackhole(), reason="the fused and phased chunk_gated_delta_rule paths are Blackhole-only"
)

# The simulator runs much slower than silicon, so it covers the fp32 gates only.
_hw_only = pytest.mark.skipif(bool(os.environ.get("TT_METAL_SIMULATOR")), reason="hardware test only")

CHUNK = 32
DIM = 128  # K == V: the flat q/k path assumes it unless num_key_heads is given


def _pcc(golden, actual):
    g = golden.to(torch.float64).flatten()
    a = actual.to(torch.float64).flatten()
    assert torch.isfinite(a).all(), "device output contains non-finite values"
    vg = g - g.mean()
    va = a - a.mean()
    denom = vg.norm() * va.norm()
    return 0.0 if denom == 0 else float((vg @ va) / denom)


def _const_tiles(device):
    eye = torch.eye(CHUNK, dtype=torch.float32)
    tril = torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.float32))
    ones = torch.ones(CHUNK, CHUNK, dtype=torch.float32)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

    def _up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return tuple(_up(t) for t in (eye, tril, ones, masks))


def _check_flat(device, B, T, H, HV, K, V, gate_dtype, seed, num_key_heads=None, program_config=None):
    """Flat q/k/v + token-major g/beta through the op (pinned to program_config, if given) vs the torch golden."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H * K).to(torch.bfloat16)
    k = torch.randn(B, T, H * K).to(torch.bfloat16)
    v = (0.5 * torch.randn(B, T, HV * V)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, HV))
    g = -torch.nn.functional.softplus(torch.randn(B, T, HV)) * 0.5
    if gate_dtype == ttnn.bfloat16:  # the golden sees exactly what the device sees
        beta = beta.to(torch.bfloat16).float()
        g = g.to(torch.bfloat16).float()
    s0 = 0.05 * torch.randn(B, HV, K, V)

    def up(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    optional = {}
    if num_key_heads is not None:
        optional["num_key_heads"] = num_key_heads
    if program_config is not None:
        optional["program_config"] = program_config
    eye, tril, ones, masks = _const_tiles(device)
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        up(q, ttnn.bfloat16),
        up(k, ttnn.bfloat16),
        up(v, ttnn.bfloat16),
        up(g, gate_dtype),
        up(beta, gate_dtype),
        initial_state=up(s0, ttnn.float32),
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
        **optional,
    )
    o_t = ttnn.to_torch(o).float().reshape(B, T, HV, V)
    fs_t = ttnn.to_torch(fs).float().reshape(B, HV, K, V)

    golden_fn = ttnn.get_golden_function(ttnn.transformer.chunk_gated_delta_rule)
    o_ref, fs_ref = golden_fn(
        q.float().reshape(B, T, H, K),
        k.float().reshape(B, T, H, K),
        v.float().reshape(B, T, HV, V),
        g,
        beta,
        scale=K**-0.5,
        initial_state=s0,
        output_final_state=True,
        chunk_size=64,
        use_qk_l2norm=True,
    )
    pcc_o = _pcc(o_ref.float().reshape(B, T, HV, V), o_t)
    pcc_s = _pcc(fs_ref.float().reshape(B, HV, K, V), fs_t)
    print(f"flat gates B={B} H={H} HV={HV} T={T} {gate_dtype}: pcc_o={pcc_o:.6f} pcc_state={pcc_s:.6f}")
    assert pcc_o > 0.999, f"o PCC {pcc_o}"
    assert pcc_s > 0.999, f"final state PCC {pcc_s}"


@pytest.mark.parametrize(
    "gate_dtype", [ttnn.float32, pytest.param(ttnn.bfloat16, marks=_hw_only)], ids=["g_fp32", "g_bf16"]
)
@pytest.mark.parametrize(
    "batch, num_k_heads, num_v_heads, seq",
    [
        pytest.param(1, 2, 6, 256, id="B1-HV6"),
        pytest.param(2, 2, 6, 128, id="B2-HV6"),
        pytest.param(1, 8, 32, 128, id="B1-HV32"),
        pytest.param(2, 8, 32, 64, id="B2-HV32"),
        pytest.param(1, 4, 40, 128, id="B1-HV40"),
        pytest.param(1, 16, 48, 128, id="B1-HV48"),
        pytest.param(2, 16, 48, 64, id="B2-HV48"),
    ],
)
def test_flat_gates_vs_golden(device, batch, num_k_heads, num_v_heads, seq, gate_dtype):
    B, T, H, HV = batch, seq, num_k_heads, num_v_heads
    grid = device.compute_with_storage_grid_size()
    if B * HV > grid.x * grid.y:
        pytest.skip(f"BH={B * HV} exceeds the {grid.x}x{grid.y} grid")
    _check_flat(device, B, T, H, HV, DIM, DIM, gate_dtype, seed=20261013 + B * 1000 + HV)


@pytest.mark.parametrize(
    "path",
    ["auto", pytest.param("fused", marks=_hw_only), pytest.param("phased", marks=_hw_only)],
)
def test_flat_key_heads_k_ne_v(device, path):
    """K != V through the flat path (H = 2, HV = 6, K = 128, V = 256: one TP8 device of the sequence-parallel fold)."""
    config_classes = {
        "auto": None,
        "fused": ttnn.ChunkGdnFusedProgramConfig,
        "phased": ttnn.ChunkGdnPhasedProgramConfig,
    }
    program_config = config_classes[path]() if config_classes[path] else None
    # Twice with fresh inputs: the second run replays the cached program.
    for seed in (20261101, 20261102):
        _check_flat(
            device,
            B=1,
            T=128,
            H=2,
            HV=6,
            K=128,
            V=256,
            gate_dtype=ttnn.float32,
            seed=seed,
            num_key_heads=2,
            program_config=program_config,
        )


@pytest.mark.parametrize(
    "rank4, num_key_heads, message",
    [
        pytest.param(False, 0, "and num_key_heads > 0", id="zero"),
        pytest.param(False, 3, "divisible by num_key_heads", id="not-dividing-width"),
        pytest.param(False, 4, "must be divisible by H", id="HV-not-multiple-of-H"),
        pytest.param(True, 3, "must equal q's head count", id="rank4-mismatch"),
    ],
)
def test_flat_key_heads_contract(device, expect_error, rank4, num_key_heads, message):
    """num_key_heads that does not fit the inputs is rejected by the host wrapper, before any device work."""
    B, T, HV, V = 1, 64, 6, 128
    q_shape = (B, T, 2, DIM) if rank4 else (B, T, 2 * DIM)  # 2 key heads, or the flat width of 2 * DIM
    v_shape = (B, T, HV, V) if rank4 else (B, T, HV * V)

    def up(shape, dtype):
        return ttnn.from_torch(torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    q, k, v = up(q_shape, ttnn.bfloat16), up(q_shape, ttnn.bfloat16), up(v_shape, ttnn.bfloat16)
    g, beta = up((B, T, HV), ttnn.float32), up((B, T, HV), ttnn.float32)
    with expect_error(RuntimeError, message):
        ttnn.transformer.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=CHUNK, num_key_heads=num_key_heads)
