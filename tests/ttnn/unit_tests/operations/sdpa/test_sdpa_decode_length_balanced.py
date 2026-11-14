# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SDPA decode with users at very different positions.

Causal decode with a cur_pos tensor splits the (user, KV head, K chunk) work evenly over all cores, so one long user is
spread over the whole grid instead of the cores assigned to its batch slot. Every user must still match attention over
exactly its own first cur_pos + 1 positions; skipped users (cur_pos -1) are not checked.
"""

import random

import pytest
import torch

import ttnn

NH, D, BLOCK = 32, 128, 64


def _positions(kind, b, s):
    rng = random.Random(11)
    top = s - 1
    if kind == "equal":
        return [s // 2 - 1] * b
    if kind == "one_long":
        return [100] * (b - 1) + [top]
    if kind == "long_first":
        return [top] + [5] * (b - 1)
    if kind == "mixed":
        return [rng.randint(0, top) for _ in range(b)]
    if kind == "edges":
        base = [0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 63, 64, 65, 1000]
        return [base[i % len(base)] for i in range(b)]
    if kind == "skip":
        pos = [rng.randint(0, top) for _ in range(b)]
        for i in range(0, b, 3):
            pos[i] = -1
        return pos
    raise ValueError(kind)


@pytest.mark.parametrize(
    "b, nkv, s, kind, layout",
    [
        (32, 8, 2048, "equal", "paged"),
        (32, 8, 2048, "one_long", "paged"),
        (32, 8, 2048, "long_first", "paged"),
        (32, 8, 2048, "mixed", "paged"),
        (32, 8, 2048, "edges", "paged"),
        (32, 8, 2048, "skip", "paged"),
        (5, 8, 2048, "mixed", "paged"),
        (32, 4, 2048, "mixed", "paged"),
        (8, 8, 8192, "one_long", "paged"),
        (32, 8, 2048, "mixed", "non_paged"),
        (32, 8, 2048, "one_long", "non_paged"),
        (32, 8, 2048, "mixed", "sharded_q"),
        (32, 8, 2048, "skip", "sharded_q"),
    ],
)
@pytest.mark.parametrize("fp32_dest", [True, False], ids=["fp32_dest", "bf16_dest"])
def test_sdpa_decode_length_balanced(device, b, nkv, s, kind, layout, fp32_dest):
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < b:
        pytest.skip("fewer cores than users")
    torch.manual_seed(5)
    pos = _positions(kind, b, s)
    group = NH // nkv
    k = torch.randn(b, nkv, s, D)
    v = torch.randn(b, nkv, s, D)
    q = torch.randn(1, b, NH, D).bfloat16()

    if layout == "paged":
        blocks = s // BLOCK
        perm = torch.randperm(b * blocks, generator=torch.Generator().manual_seed(3))
        to_blocks = (
            lambda t: t.reshape(b, nkv, blocks, BLOCK, D).permute(0, 2, 1, 3, 4).reshape(b * blocks, nkv, BLOCK, D)
        )
        k_paged, v_paged = torch.empty(b * blocks, nkv, BLOCK, D), torch.empty(b * blocks, nkv, BLOCK, D)
        k_paged[perm], v_paged[perm] = to_blocks(k), to_blocks(v)
        tk = ttnn.from_torch(k_paged, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tv = ttnn.from_torch(v_paged, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        from_blocks = (
            lambda t: ttnn.to_torch(t)
            .float()[perm]
            .reshape(b, blocks, nkv, BLOCK, D)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, nkv, s, D)
        )
        k_dev, v_dev = from_blocks(tk), from_blocks(tv)
        page_table = ttnn.from_torch(
            perm.reshape(b, blocks).int(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
    else:
        tk = ttnn.from_torch(k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        tv = ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        k_dev, v_dev = ttnn.to_torch(tk).float(), ttnn.to_torch(tv).float()

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if layout == "sharded_q":
        grid = ttnn.num_cores_to_corerangeset(b, device.compute_with_storage_grid_size(), True)
        shard = ttnn.ShardSpec(grid, (32, D), ttnn.ShardOrientation.ROW_MAJOR)
        tq = ttnn.to_memory_config(
            tq, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
        )
    cur_pos = ttnn.from_torch(torch.tensor(pos, dtype=torch.int32), dtype=ttnn.int32, device=device)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2 if fp32_dest else ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest,
        packer_l1_acc=fp32_dest,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )
    kwargs = dict(
        cur_pos_tensor=cur_pos,
        scale=D**-0.5,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if layout == "paged":
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tq, tk, tv, page_table_tensor=page_table, **kwargs
        )
    else:
        out = ttnn.transformer.scaled_dot_product_attention_decode(tq, tk, tv, **kwargs)
    got = ttnn.to_torch(out).float()[0, :, :NH, :]

    qf = q.float()[0]
    for u in range(b):
        if pos[u] < 0:
            continue
        length = pos[u] + 1
        ref = torch.empty(NH, D)
        for h in range(nkv):
            scores = (qf[u, h * group : (h + 1) * group] @ k_dev[u, h, :length].T) * D**-0.5
            ref[h * group : (h + 1) * group] = torch.softmax(scores, -1) @ v_dev[u, h, :length]
        pcc = torch.corrcoef(torch.stack([got[u].flatten(), ref.flatten()]))[0, 1].item()
        assert torch.isfinite(got[u]).all() and pcc >= 0.999, f"user {u} at position {pos[u]}: PCC {pcc}"
