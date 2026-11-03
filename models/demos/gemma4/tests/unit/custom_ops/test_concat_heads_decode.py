# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ConcatHeadsDecode (custom op) vs the shipped reshard + nlp_concat_heads_decode + sharded_to_interleaved."""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.concat_heads_decode import ConcatHeadsDecode
from models.demos.gemma4.tt.attention.operations import concat_heads

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
CI = os.environ.get("CI") == "true"
SHAPES = [
    (32, 4, 256),
    (1, 4, 256),
    (8, 16, 128),
    (32, 2, 256),
    (32, 8, 256),
    (32, 1, 512),
    (32, 4, 512),
    # dense per-chip head counts: 31B tp4 (8 heads, D 256 / 512), 12B tp8 (2 heads), batch 1 and 8 rows
    (32, 8, 512),
    (1, 8, 256),
    (1, 8, 512),
    (8, 8, 256),
    (32, 2, 512),
    (1, 2, 256),
]


def _cases():
    """(B, H, D, heads-per-unit override, forced grid): the full sweep locally; under CI=true (the ~60 s budget of a
    one-chip unit test) the automatic split on the device grid for every shape plus one forced-8x8 case."""
    out = []
    for B, H, D in SHAPES:
        for hg, hg_id in ((None, "hg_auto"), (1, "hg1"), (2, "hg2"), (99, "hg_all")):
            for grid, g_id in ((None, "device_grid"), ("8x8", "grid8x8")):
                in_ci = hg is None and (grid is None or (B, H, D) == (32, 8, 256))
                marks = [] if in_ci or not CI else [pytest.mark.skip(reason="CI subset: hg_auto on the device grid")]
                out.append(pytest.param(B, H, D, hg, grid, id=f"{B}x{H}x{D}-{hg_id}-{g_id}", marks=marks))
    return out


@pytest.mark.parametrize("B, H, D, hg, grid", _cases())
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_concat_heads_decode_matches_composite(mesh_device, device_params, B, H, D, hg, grid, monkeypatch):
    if grid is not None:
        if hg is not None or (B, H, D) not in ((32, 8, 256), (32, 8, 512)):
            pytest.skip("one 8x8 case per shape family")
        monkeypatch.setenv("GEMMA4_TEST_GRID", grid)  # the Wormhole-sized unit split on the same chip
    torch.manual_seed(0)
    x = torch.randn(1, B, 32, D).to(torch.bfloat16)  # heads padded to 32 rows (rows >= H: padding)
    golden = ConcatHeadsDecode.golden(x.float(), H)
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ref = ttnn.to_torch(concat_heads(tx, is_decode_mode=True, num_heads=H, head_dim=D, mesh_device=mesh_device)).float()
    out = ttnn.to_torch(ConcatHeadsDecode.op(tx, H, heads_per_unit=hg)).float()
    assert list(out.shape) == [1, 1, B, H * D]
    assert torch.equal(out, golden), f"max|diff| vs golden {(out - golden).abs().max()}"
    assert torch.equal(out, ref[:, :, :B, :]), f"max|diff| vs composite {(out - ref[:, :, :B, :]).abs().max()}"
    tx2 = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    assert torch.equal(ttnn.to_torch(ConcatHeadsDecode.op(tx2, H, heads_per_unit=hg)).float(), out)


@pytest.mark.parametrize(
    "B, H, D, K_N",
    [
        (32, 8, 256, (2048, 5376)),
        (1, 8, 256, (2048, 5376)),
        (32, 4, 256, (1024, 3840)),
        (32, 2, 256, (512, 3840)),
        (32, 8, 512, (4096, 5376)),
    ],
    ids=["31b_tp4", "31b_tp4_b1", "12b_tp4", "12b_tp8", "31b_tp4_global"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_concat_heads_decode_sharded_output(mesh_device, device_params, B, H, D, K_N):
    """The output written straight into the DRAM-sharded o_proj decode matmul's L1 activation layout
    (activation_memcfg(K, N)): bit-identical to the interleaved form, one program, and the matmul takes it
    without its interleaved -> sharded copy (GEMMA4_CONCAT_HEADS_SHARDED=1)."""
    from models.demos.gemma4.tt.dram_sharded import activation_memcfg, can_dram_shard

    K, N = K_N
    assert K == H * D
    if not can_dram_shard(K, N, dtype=ttnn.bfloat8_b):
        pytest.skip(f"[{K}, {N}] has no DRAM-sharded decode config on this grid")
    torch.manual_seed(0)
    x = torch.randn(1, B, 32, D).to(torch.bfloat16)
    golden = ConcatHeadsDecode.golden(x.float(), H)
    tx = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ref = ttnn.to_torch(ConcatHeadsDecode.op(tx, H)).float()
    mem = activation_memcfg(K, N)
    out = ConcatHeadsDecode.op(tx, H, out_memory_config=mem)
    assert out.is_sharded() and out.memory_config() == mem
    got = ttnn.to_torch(out).float()
    assert list(got.shape) == [1, 1, B, H * D]
    assert torch.equal(got, golden), f"sharded output vs golden max|diff| {(got - golden).abs().max()}"
    assert torch.equal(got, ref), "sharded output differs from the interleaved form"
    # the DRAM-sharded matmul consumes it as is: same result as from the interleaved input
    from models.demos.gemma4.tt.dram_sharded import DramShardedLinear

    w = (torch.randn(1, 1, K, N) * 0.02).to(torch.bfloat16)
    lin = DramShardedLinear(w, mesh_device, None, k=K, n=N, dtype=ttnn.bfloat8_b, cache_file_name=None)
    assert lin.accepts_sharded_rows() and lin.act_memory_config == mem
    y_ref = ttnn.to_torch(lin(ConcatHeadsDecode.op(tx, H))).float()
    y_sh = ttnn.to_torch(lin(ConcatHeadsDecode.op(tx, H, out_memory_config=mem))).float()
    assert list(y_sh.shape) == [1, 1, B, N]
    assert torch.equal(y_sh, y_ref), f"matmul on the sharded input differs: max|diff| {(y_sh - y_ref).abs().max()}"
