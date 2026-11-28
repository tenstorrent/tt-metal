# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.device_utils import is_blackhole

BLOCK, NBLK, KVH, QH, DH = 32, 1024, 8, 32, 128
SINGLE_POS = [14, 31, 77, 95, 129, 520, 1000, 2047, 4186]
MULTI_CASES = [
    (2, 2, 77),
    (3, 3, 94),
    (4, 4, 93),
    (4, 3, 125),
    (4, 4, 157),
    (5, 5, 77),
    (6, 6, 77),
    (7, 7, 77),
    (8, 8, 77),
    (8, 8, 93),
    (8, 8, 2100),
    (8, 8, 4186),
]


def _require_p100(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Blackhole P100 only")


def _configs(device):
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    return program_config, compute_kernel_config


def _cache(device, nblk=NBLK):
    torch.manual_seed(0)
    kv = [
        ttnn.from_torch(
            torch.randn(nblk, KVH, BLOCK, DH),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    page = torch.argsort(torch.randperm(nblk)).reshape(1, nblk).to(torch.int32)
    return kv[0], kv[1], page


def _i32(device, t):
    return ttnn.from_torch(
        t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _q(device, q):
    rows = q.shape[1]
    cores = ttnn.num_cores_to_corerangeset(rows, ttnn.CoreCoord(8, 8), row_wise=True)
    mem = ttnn.create_sharded_memory_config(
        shape=(QH, DH),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


def _single(device, q1, k, v, pt, pos, program_config, compute_kernel_config):
    return ttnn.to_torch(
        ttnn.experimental.llama31_8b_v3.sdpa_decode_bh(
            q1,
            k,
            v,
            pt,
            _i32(device, torch.tensor([pos], dtype=torch.int32)),
            program_config,
            scale=DH**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
    )


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.parametrize("nblk", [NBLK, 2048])
def test_sdpa_decode_bh_matches_generic(device, nblk):
    _require_p100(device)
    program_config, compute_kernel_config = _configs(device)
    k, v, page = _cache(device, nblk)
    pt = _i32(device, page)
    for pos in SINGLE_POS:
        q1 = _q(device, torch.randn(1, 1, QH, DH, generator=torch.Generator().manual_seed(pos)))
        cur = _i32(device, torch.tensor([pos], dtype=torch.int32))
        actual = ttnn.experimental.llama31_8b_v3.sdpa_decode_bh(
            q1,
            k,
            v,
            pt,
            cur,
            program_config,
            scale=DH**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        expected = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q1,
            k,
            v,
            page_table_tensor=pt,
            cur_pos_tensor=cur,
            scale=DH**-0.5,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert torch.equal(ttnn.to_torch(actual), ttnn.to_torch(expected)), pos


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.parametrize("case", ["dram_q", "q_off_origin", "hifi4"])
def test_sdpa_decode_bh_rejects_unsupported_config(device, case, expect_error):
    _require_p100(device)
    program_config, compute_kernel_config = _configs(device)
    k, v, page = _cache(device)
    pt = _i32(device, page)
    q = torch.randn(1, 4, QH, DH, generator=torch.Generator().manual_seed(7))
    if case == "dram_q":
        q4 = ttnn.from_torch(
            q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q1 = ttnn.slice(q4, [0, 1, 0, 0], [1, 2, QH, DH])
    elif case == "q_off_origin":
        mem = ttnn.create_sharded_memory_config(
            shape=(QH, DH),
            core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        q1 = ttnn.from_torch(q[:, :1], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    else:
        q1 = _q(device, q[:, :1])
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    with expect_error(RuntimeError, r"llama31_8b_v3\.sdpa_decode_bh: "):
        ttnn.experimental.llama31_8b_v3.sdpa_decode_bh(
            q1,
            k,
            v,
            pt,
            _i32(device, torch.tensor([77], dtype=torch.int32)),
            program_config,
            scale=DH**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.parametrize("rows, valid, p0", MULTI_CASES)
def test_sdpa_decode_bh_multi_pos_rows_match_single(device, rows, valid, p0):
    _require_p100(device)
    program_config, compute_kernel_config = _configs(device)
    k, v, page = _cache(device, 2048 if p0 >= 4096 else NBLK)
    pt1 = _i32(device, page)
    ptr = _i32(device, page.repeat(rows, 1))
    q = torch.randn(1, rows, QH, DH, generator=torch.Generator().manual_seed(1000 * rows + p0))
    pos = [p0 + b if b < valid else -1 for b in range(rows)]
    block = 128 if p0 >= 64 else (64 if p0 >= 32 else 32)
    out = ttnn.to_torch(
        ttnn.experimental.llama31_8b_v3.sdpa_decode_bh(
            _q(device, q),
            k,
            v,
            ptr,
            _i32(device, torch.tensor(pos, dtype=torch.int32)),
            program_config,
            scale=DH**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
            multi_pos=True,
        )
    )
    checked = 0
    for b in range(rows):
        if pos[b] < 0 or pos[b] // block != p0 // block:
            continue
        single = _single(device, _q(device, q[:, b : b + 1]), k, v, pt1, pos[b], program_config, compute_kernel_config)
        assert torch.equal(out[:, b : b + 1], single), (b, pos[b])
        checked += 1
    assert checked == valid
