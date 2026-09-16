# SPDX-License-Identifier: Apache-2.0
"""R1 paged MLA decode at the geometry of tests/ttnn/unit_tests/operations/sdpa/test_mla_decode.py row 1
("DeepSeek V3 TG full DP": batch 4, nh 128, nkv 1, kv_lora 512, d_rope 64, Q height-sharded on 64 cores, Q bf16, KV bfp8_b,
paged block 64, reuse_k True so V is None), following run_flash_mla_decode_impl in mla_test_utils.py: q_chunk 0, k_chunk 128,
exp_approx False, max_cores_per_head_batch 16, HiFi4 math_approx False fp32 off; output height-sharded like Q.
Env: MLAD_POS (comma list of cache positions, cur_pos = pos for every user), MLAD_CACHE (cache length per user, default
2 x max pos, multiple of 64), MLAD_B 4, MLAD_NH 128, MLAD_KVLORA 512, MLAD_DROPE 64, MLAD_QCORES 64, MLAD_ITERS 3, MLAD_MCPHB 16 (max_cores_per_head_batch, the cores-per-group cap).
"""
import math
import os

import torch


def test_r1_mla_decode(device):
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import page_table_setup, to_paged_cache
    from models.common.utility_functions import nearest_y

    batch = int(os.environ.get("MLAD_B", "4"))
    nh = int(os.environ.get("MLAD_NH", "128"))
    nkv = 1
    kv_lora = int(os.environ.get("MLAD_KVLORA", "512"))
    d_rope = int(os.environ.get("MLAD_DROPE", "64"))
    q_num_cores = int(os.environ.get("MLAD_QCORES", "64"))
    iters = int(os.environ.get("MLAD_ITERS", "3"))
    mcphb = int(os.environ.get("MLAD_MCPHB", "16"))
    positions = [int(x) for x in os.environ.get("MLAD_POS", "1024").split(",")]
    block_size = 64
    cache = int(os.environ.get("MLAD_CACHE", str(max(2 * max(positions), 2048))))
    cache = ((cache + block_size - 1) // block_size) * block_size
    d_qk = kv_lora + d_rope
    max_num_blocks = cache // block_size * batch
    cfg = (
        ttnn.PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
        if hasattr(ttnn, "PagedAttentionConfig")
        else None
    )
    if cfg is None:
        from models.tt_transformers.tt.common import PagedAttentionConfig

        cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    torch.manual_seed(0)
    q = torch.randn(batch, nh, 1, d_qk).float()
    k = torch.randn(batch, nkv, cache, d_qk).float()
    page_table = page_table_setup(batch, cfg)
    k_paged = to_paged_cache(k, page_table, cfg)
    tt_page_table = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    q_for_tt = q.permute(2, 0, 1, 3)
    q_num_cores = min(batch * nh, q_num_cores)
    block_height = nearest_y(int(torch.tensor(q.shape[:-1]).prod().item()) // q_num_cores, ttnn.TILE_SIZE)
    q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    q_mem = ttnn.create_sharded_memory_config(
        shape=(block_height, q.shape[-1]),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_mem = ttnn.create_sharded_memory_config(
        shape=(block_height, kv_lora),
        core_grid=q_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    tt_q = ttnn.from_torch(q_for_tt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=q_mem)
    tt_k = ttnn.from_torch(
        k_paged, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=mcphb,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    scale = d_qk**-0.5
    print(
        f"\n[r1_mla_decode] batch={batch} nh={nh} kv_lora={kv_lora} d_rope={d_rope} cache={cache} block={block_size} q_cores={q_num_cores} mcphb={mcphb} positions={positions} iters={iters}",
        flush=True,
    )
    for pos in positions:
        cur = ttnn.from_torch(torch.tensor([pos for _ in range(batch)]), device=device, dtype=ttnn.int32)
        for it in range(iters):
            out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                tt_q,
                tt_k,
                None,
                head_dim_v=kv_lora,
                page_table_tensor=tt_page_table,
                cur_pos_tensor=cur,
                scale=scale,
                program_config=pc,
                compute_kernel_config=ck,
                memory_config=out_mem,
            )
            ttnn.synchronize_device(device)
            out.deallocate()
            print(f"[r1_mla_decode] pos={pos} iter {it} done", flush=True)
        cur.deallocate()
    for t in (tt_q, tt_k, tt_page_table):
        t.deallocate()
