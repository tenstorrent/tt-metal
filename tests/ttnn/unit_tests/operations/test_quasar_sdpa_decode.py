# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Standalone repro for the Quasar flash-decode SDPA op, seen in llama32_1b decode.

The llama32_1b decode attention ends in a flash-decode SDPA:

    ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        q_heads, keys, values, page_table_tensor=..., cur_pos_tensor=..., scale=..., ...)

The e2e run reaches this op (after QKV matmul -> create_qkv_heads -> RoPE -> paged_update_cache)
and appears to stall/fault inside it on the Quasar simulator. This file exercises JUST that op, with
the model's exact decode shapes and program config, so it can be run under watcher in isolation.

Shapes (llama-3.2-1B, matches the captured graph case):
    q        : [1, 1, n_q_heads=32, head_dim=64]  bf16 TILE, HEIGHT_SHARDED L1 (1 core/batch)
    keys/vals: [max_num_blocks=128, n_kv_heads=8, block_size=32, head_dim=64]  bf16 TILE, DRAM interleaved  (paged)
               [batch, n_kv_heads=8, max_seq_len, head_dim=64]                 bf16 TILE, DRAM interleaved  (non-paged)
    page_table: [batch, max_num_blocks]  int32 ROW_MAJOR DRAM   (paged only)
    cur_pos   : [batch]                  int32 ROW_MAJOR DRAM
    program_config: SDPAProgramConfig(compute_with_storage_grid_size=[8,4], exp_approx_mode=True,
                                      q_chunk_size=0, k_chunk_size=0)

Inputs are built via a bf16 row-major upload + quasar.tilize (NOT from_torch(TILE), which hangs on the
Quasar sim). bf16 throughout (Quasar dropped bf8_b -> MX formats).

NOT marked xfail -- craq-sim tooling drives off a real FAIL/hang. On WH/BH this passes.

Run (Quasar sim, with watcher):
    MESH_DEVICE=<qsr> TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_WATCHER=1 \
        pytest tests/ttnn/unit_tests/operations/test_quasar_sdpa_decode.py -k paged
"""

import pytest
import torch
from loguru import logger

import ttnn

# llama-3.2-1B decode attention dims
N_Q_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 32  # paged KV cache page size (tile height)
# Keep the KV cache small: craq-sim tilizes every tile in software, so a big cache dominates runtime
# (128 blocks -> 4 MB / 2048 tiles per cache -> minutes just to upload). cur_pos=64 needs only ~3 pages;
# 8 blocks (256 positions) reproduces the SDPA geometry (kv_heads/head_dim/block_size/grid) ~16x faster.
MAX_NUM_BLOCKS = 8
GRID_X, GRID_Y = 8, 4  # the model's decode SDPA grid -- used ONLY as _skip_if_small's minimum (the tree
# reduction needs num_cores_per_head>1, i.e. >= a full 8x4). The program config itself now uses the DEVICE
# grid (see _prog_cfg), never a hardcoded 8x4, so it never exceeds the device.
SCALE = HEAD_DIM**-0.5


def _tile_bf16_dram(t_bf16, mesh_device):
    """bf16 TILE, DRAM-interleaved without from_torch(TILE) (hangs on the Quasar sim): upload row-major,
    then tilize via the Gen2-native quasar op where available."""
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        return ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    except (AttributeError, RuntimeError) as e:
        logger.info(f"[sdpa-repro] quasar.tilize unavailable ({e}); using mainline ttnn.tilize")
        return ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _int32_rm_dram(t_int, mesh_device):
    return ttnn.from_torch(
        t_int,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )


def _q_height_sharded(t_bf16, mesh_device, batch):
    """q -> bf16 TILE, HEIGHT_SHARDED on L1: one core per batch (grid [batch,1]), shard [32, n_q_heads*... ]."""
    qt = _tile_bf16_dram(t_bf16, mesh_device)
    # q logical shape [1, batch, n_q_heads, head_dim] flattens to [batch, n_q_heads*head_dim] worth of a
    # 32-row tile per batch; the captured case shards [32, 64] on a single core for batch=1.
    core_rs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, max(batch - 1, 0)))})
    shard_spec = ttnn.ShardSpec(core_rs, (BLOCK_SIZE, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR)
    memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    qi2s = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    return (qi2s or ttnn.interleaved_to_sharded)(qt, memcfg)


def _prog_cfg(mesh_device, max_cores_per_head_batch=16, k_chunk_size=0, grid_xy=None):
    # The decode-SDPA factory uses program_config.compute_with_storage_grid_size DIRECTLY (it builds the core
    # grid from it and FATALs if it exceeds the device -- sdpa_decode_program_factory.cpp:174/191/195). It does
    # NOT clamp to the device. So pass a grid <= device: an explicit grid_xy, else the DEVICE grid (never a
    # hardcoded 8x4, which would FATAL on a smaller device or silently run the full 8x4 tree on a larger one).
    if grid_xy is not None:
        gx, gy = grid_xy
    else:
        dg = mesh_device.compute_with_storage_grid_size()
        gx, gy = int(dg.x), int(dg.y)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        exp_approx_mode=True,
        q_chunk_size=0,
        k_chunk_size=k_chunk_size,
        max_cores_per_head_batch=max_cores_per_head_batch,
    )


def _compute_cfg():
    # fp32_dest_acc_en=False on Quasar (bf16->Tf32 unpack gap); HiFi2 matches the model.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _skip_if_small(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x < GRID_X or grid.y < GRID_Y:
        pytest.skip(f"needs an {GRID_X}x{GRID_Y} grid; device has {grid.x}x{grid.y}")


def _run_paged(
    mesh_device, cur_pos, max_cores_per_head_batch=16, k_chunk_size=0, grid_xy=None, nq=N_Q_HEADS, nkv=N_KV_HEADS
):
    """Paged flash-decode SDPA at the llama shapes. num_cores_per_head is grid-derived (8x4 / 8 KV heads = 4),
    so a multi-core TREE reduction is configured; whether it actually runs (and deadlocks on Quasar) depends
    on k_num_chunks>1 (children get K data). max_cores_per_head_batch=1 collapses to 1 core/head (no tree).
    grid_xy overrides the program-config compute grid (the e2e clamps it to 2 nodes). nq/nkv override the head
    counts -- nkv=1 tests the ATOMIC UNIT of the 'split into per-kv-head SDPAs' idea (1 kv-head -> 1 core, the
    config that maps like the passing 8x4-1/head case; if this passes on a small grid, the split fits 2 cores)."""
    if grid_xy is None:
        _skip_if_small(mesh_device)
    else:
        gx, gy = grid_xy
        dev = mesh_device.compute_with_storage_grid_size()
        if dev.x < gx or dev.y < gy:
            pytest.skip(f"grid {gx}x{gy} needs a device >= that; device is {dev.x}x{dev.y}")
    batch = 1
    torch.manual_seed(0)
    q = torch.randn(1, batch, nq, HEAD_DIM, dtype=torch.bfloat16)
    keys = torch.randn(MAX_NUM_BLOCKS, nkv, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    values = torch.randn(MAX_NUM_BLOCKS, nkv, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).reshape(1, MAX_NUM_BLOCKS).repeat(batch, 1)
    cur_pos_t = torch.full((batch,), cur_pos, dtype=torch.int32)

    q_t = _q_height_sharded(q, mesh_device, batch)
    k_t = _tile_bf16_dram(keys, mesh_device)
    v_t = _tile_bf16_dram(values, mesh_device)
    pt_t = _int32_rm_dram(page_table, mesh_device)
    cp_t = _int32_rm_dram(cur_pos_t, mesh_device)

    if grid_xy is not None:
        gx, gy = grid_xy
    else:
        _dg = mesh_device.compute_with_storage_grid_size()
        gx, gy = int(_dg.x), int(_dg.y)
    logger.info(
        f"[sdpa-repro] paged decode cur_pos={cur_pos} max_cores_per_head_batch={max_cores_per_head_batch} "
        f"k_chunk_size={k_chunk_size} grid {gx}x{gy} (config grid = device grid unless grid_xy given) "
        f"-> {'NO tree' if max_cores_per_head_batch == 1 else 'TREE reduction (if cores/head > 1)'}"
    )
    out = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        q_t,
        k_t,
        v_t,
        page_table_tensor=pt_t,
        cur_pos_tensor=cp_t,
        scale=SCALE,
        program_config=_prog_cfg(
            mesh_device, max_cores_per_head_batch=max_cores_per_head_batch, k_chunk_size=k_chunk_size, grid_xy=grid_xy
        ),
        compute_kernel_config=_compute_cfg(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)
    logger.info(f"[sdpa-repro] paged decode out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()}")
    assert torch.isfinite(o).all(), "SDPA decode produced non-finite output"


def test_paged_sdpa_decode(mesh_device):
    """Baseline: short KV (cur_pos=64), k_num_chunks=1 -> tree children have no data -> single-core finalize
    path. Passes on Quasar (this is the case that already worked)."""
    _run_paged(mesh_device, cur_pos=64)


def test_paged_sdpa_decode_tree_reduction_hang(mesh_device):
    """REPRO: long KV + small k_chunk forces k_num_chunks>1, so all 4 cores/head get K data and the MULTI-CORE
    TREE reduction runs. This DEADLOCKS on the Quasar sim (workers stall at waypoint WFW, cross-core
    mcast/DFB handshake). Run under watcher to observe the hang. On WH/BH it passes."""
    _run_paged(mesh_device, cur_pos=200, max_cores_per_head_batch=16, k_chunk_size=BLOCK_SIZE)


def test_paged_sdpa_decode_single_core(mesh_device):
    """FIX validation: same long-KV case as the hang repro, but max_cores_per_head_batch=1 -> num_cores_per_head=1
    -> num_tree_reduction_rounds=0 (no cross-core reduction). Should PASS on Quasar. This is what the e2e
    _install_quasar_sdpa_single_core monkeypatch does."""
    _run_paged(mesh_device, cur_pos=200, max_cores_per_head_batch=1, k_chunk_size=BLOCK_SIZE)


@pytest.mark.parametrize("grid_xy", [(1, 1), (2, 1)], ids=["1node", "2node"])
def test_paged_sdpa_decode_single_core_grid(mesh_device, grid_xy):
    """Grid sweep of the SINGLE-CORE MULTI-CHUNK decode (max_cores_per_head_batch=1, k_num_chunks>1). This
    mirrors the e2e, where _install_quasar_sdpa_single_core clamped the program-config grid to 2 nodes and the
    on-device decode SDPA then HUNG (2026-09-26: grid=2-1, max_cores=1, cur_pos~512, no op progress). The
    passing test_paged_sdpa_decode_single_core above used the (8,4) grid config, so the compute grid -- not
    multi-chunk per se -- is the suspect. Compares 1 node vs 2 nodes at the e2e's single-core config:
      - if 1node PASSES and 2node HANGS -> the 2-node clamp is the bug; the e2e decode SDPA should use 1 node.
      - if both hang -> single-core multi-chunk decode is broken on device regardless of grid (keep host SDPA).
    Run under watcher; alias=0 in the env (matches the e2e)."""
    _run_paged(mesh_device, cur_pos=200, max_cores_per_head_batch=1, k_chunk_size=BLOCK_SIZE, grid_xy=grid_xy)


@pytest.mark.parametrize("grid_xy", [(1, 1), (2, 1)], ids=["1node", "2node"])
def test_paged_sdpa_decode_single_kv_head(mesh_device, grid_xy):
    """ATOMIC UNIT of the 'split decode SDPA into per-kv-head calls' idea (for fitting a 2-core device).
    The full 8-kv-head decode HANGS when a core owns >1 kv-head (1x1 / 2-node); it PASSES at 8x4 (1 kv-head/core).
    So split it into calls each carrying <= num_cores kv-heads (1 kv-head/core). This tests the smallest such
    call: nkv=1, multi-chunk (cur_pos=200, k_chunk=32), on 1 and 2 cores, max_cores=1.
    nq=32 (not the real split's 4) because a sub-tile q height (4) fails the intermediate tilize and
    from_torch(TILE) hangs on the sim; nq=32 is tile-aligned AND a STRONGER proxy -- 32 q-heads/core is more
    than the real split's 4, so if 1 kv-head/core passes here it passes there. It isolates kv-heads/core.
    If it PASSES, the split is viable -- run 4 calls (2 kv-heads each) or 8 (1 each) on 2 cores and concat.
    If it HANGS even at nkv=1, the hang is intrinsic to single-core multi-chunk (not kv-head count) -> keep host."""
    gx, gy = grid_xy
    dev = mesh_device.compute_with_storage_grid_size()
    if dev.x < gx or dev.y < gy:
        pytest.skip(f"grid {gx}x{gy} needs a device >= that; device is {dev.x}x{dev.y}")
    _run_paged(
        mesh_device,
        cur_pos=200,
        max_cores_per_head_batch=1,
        k_chunk_size=BLOCK_SIZE,
        grid_xy=grid_xy,
        nq=N_Q_HEADS,
        nkv=1,
    )


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _torch_gqa_decode(q, keys, values, cur_pos, scale):
    """Reference for the paged decode with an identity page table (block b -> logical positions [b*BS, (b+1)*BS)).
    q [1,1,nq,hd]; keys/values [nb, nkv, BS, hd]. GQA: q-head h attends kv-head h // (nq//nkv). Causal decode
    attends positions 0..cur_pos inclusive. Returns [nq, hd]."""
    nb, nkv, bs, hd = keys.shape
    nq = q.shape[2]
    qpk = nq // nkv
    seqlen = cur_pos + 1
    k = keys.permute(1, 0, 2, 3).reshape(nkv, nb * bs, hd)[:, :seqlen, :].float()  # [nkv, seq, hd]
    v = values.permute(1, 0, 2, 3).reshape(nkv, nb * bs, hd)[:, :seqlen, :].float()
    out = torch.zeros(nq, hd)
    for h in range(nq):
        kvh = h // qpk
        qh = q[0, 0, h].float()
        scores = (k[kvh] @ qh) * scale  # [seq]
        w = torch.softmax(scores, dim=0)
        out[h] = w @ v[kvh]
    return out


def _run_split(mesh_device, num_cores):
    """Split the 8-kv-head decode SDPA into per-group calls that each map ONE kv-head per core, so no core ever
    owns >1 kv-head (the config that HANGS on Quasar -- see test_paged_sdpa_decode_single_kv_head). The full
    tensors are built once (DRAM-interleaved TILE); each group is carved out ON DEVICE with ttnn.slice (q sliced
    on the head axis, k/v on the kv-head axis) so we never host-tilize a sub-tile q height (that FATALs). Each
    group runs paged decode SDPA with grid (num_cores,1), max_cores_per_head_batch=1 (no tree reduction), then
    the group outputs are concatenated back on the head axis.
      - num_cores=1: group size 1 -> 8 sequential calls, 1 kv-head on 1 core.
      - num_cores=2: group size 2 -> 4 sequential calls, 2 kv-heads on 2 cores (still 1 kv-head/core).
    Validates finiteness AND PCC vs a torch GQA reference."""
    dev = mesh_device.compute_with_storage_grid_size()
    if int(dev.x) < num_cores:
        pytest.skip(f"split needs >= {num_cores} cores in a row; device is {dev.x}x{dev.y}")

    batch = 1
    nq, nkv = N_Q_HEADS, N_KV_HEADS
    qpk = nq // nkv  # 4 q-heads per kv-head
    gs = min(num_cores, nkv)  # kv-heads per call = cores (1 kv-head/core)
    assert nkv % gs == 0, f"nkv={nkv} must split into groups of {gs}"
    cur_pos = 200

    torch.manual_seed(0)
    q = torch.randn(1, batch, nq, HEAD_DIM, dtype=torch.bfloat16)
    keys = torch.randn(MAX_NUM_BLOCKS, nkv, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    values = torch.randn(MAX_NUM_BLOCKS, nkv, BLOCK_SIZE, HEAD_DIM, dtype=torch.bfloat16)
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).reshape(1, MAX_NUM_BLOCKS).repeat(batch, 1)
    cur_pos_t = torch.full((batch,), cur_pos, dtype=torch.int32)

    # q DRAM-interleaved TILE (the op accepts DRAM-interleaved Q, not just height-sharded -- so we can device-slice
    # the head axis without a host tilize of a sub-tile height).
    q_t = _tile_bf16_dram(q, mesh_device)
    k_t = _tile_bf16_dram(keys, mesh_device)
    v_t = _tile_bf16_dram(values, mesh_device)
    pt_t = _int32_rm_dram(page_table, mesh_device)
    cp_t = _int32_rm_dram(cur_pos_t, mesh_device)

    logger.info(
        f"[sdpa-repro] SPLIT decode: nq={nq} nkv={nkv} qpk={qpk} group_size={gs} "
        f"({nkv // gs} calls) grid {num_cores}x1 max_cores=1 cur_pos={cur_pos}"
    )

    group_outs = []
    for g0 in range(0, nkv, gs):
        # q-heads [g0*qpk : (g0+gs)*qpk] on the head axis (dim 2); kv-heads [g0 : g0+gs] on the kv-head axis (dim 1).
        q_g = ttnn.slice(q_t, [0, 0, g0 * qpk, 0], [1, batch, (g0 + gs) * qpk, HEAD_DIM])
        k_g = ttnn.slice(k_t, [0, g0, 0, 0], [MAX_NUM_BLOCKS, g0 + gs, BLOCK_SIZE, HEAD_DIM])
        v_g = ttnn.slice(v_t, [0, g0, 0, 0], [MAX_NUM_BLOCKS, g0 + gs, BLOCK_SIZE, HEAD_DIM])
        out_g = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
            q_g,
            k_g,
            v_g,
            page_table_tensor=pt_t,
            cur_pos_tensor=cp_t,
            scale=SCALE,
            program_config=_prog_cfg(
                mesh_device, max_cores_per_head_batch=1, k_chunk_size=BLOCK_SIZE, grid_xy=(num_cores, 1)
            ),
            compute_kernel_config=_compute_cfg(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(mesh_device)
        group_outs.append(out_g)
        logger.info(f"[sdpa-repro] SPLIT group kv[{g0}:{g0 + gs}] done")

    # Concatenate the per-group outputs back along the q-head axis (dim 2).
    try:
        out = ttnn.concat(group_outs, dim=2)
        o = ttnn.to_torch(out)
    except (RuntimeError, AttributeError) as e:
        logger.info(f"[sdpa-repro] device concat unavailable ({e}); concatenating on host")
        o = torch.cat([ttnn.to_torch(g) for g in group_outs], dim=2)

    o = o.reshape(-1, HEAD_DIM)[:nq]  # [nq, hd]
    ref = _torch_gqa_decode(q, keys, values, cur_pos, SCALE)
    pcc = _pcc(o, ref)
    logger.info(f"[sdpa-repro] SPLIT out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()} PCC={pcc:.5f}")
    assert torch.isfinite(o).all(), "split SDPA decode produced non-finite output"
    assert pcc > 0.99, f"split SDPA decode PCC too low: {pcc}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("num_cores", [1, 2], ids=["1core", "2core"])
def test_paged_sdpa_decode_split(mesh_device, num_cores):
    """Per-kv-head SPLIT of the decode SDPA (the e2e fix for a 2-core Quasar device). Runs nkv/num_cores
    sequential paged-decode calls, each mapping exactly 1 kv-head per core (the ONLY config that doesn't hang),
    then concatenates. 1core = 8 calls (1 kv-head each); 2core = 4 calls (2 kv-heads, 1/core). If these PASS,
    the e2e decode SDPA can run on device via this split instead of the host fallback."""
    _run_split(mesh_device, num_cores)


def test_non_paged_sdpa_decode(mesh_device):
    """Non-paged flash-decode SDPA (simpler: full [batch,kv_heads,seq,head_dim] cache, no page table)."""
    _skip_if_small(mesh_device)
    batch = 1
    max_seq = MAX_NUM_BLOCKS * BLOCK_SIZE  # 4096
    cur_pos = 64

    torch.manual_seed(0)
    q = torch.randn(1, batch, N_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    keys = torch.randn(batch, N_KV_HEADS, max_seq, HEAD_DIM, dtype=torch.bfloat16)
    values = torch.randn(batch, N_KV_HEADS, max_seq, HEAD_DIM, dtype=torch.bfloat16)
    cur_pos_t = torch.full((batch,), cur_pos, dtype=torch.int32)

    q_t = _q_height_sharded(q, mesh_device, batch)
    k_t = _tile_bf16_dram(keys, mesh_device)
    v_t = _tile_bf16_dram(values, mesh_device)
    cp_t = _int32_rm_dram(cur_pos_t, mesh_device)

    _dg = mesh_device.compute_with_storage_grid_size()
    logger.info(
        f"[sdpa-repro] non-paged decode: q[1,{batch},{N_Q_HEADS},{HEAD_DIM}] "
        f"kv[{batch},{N_KV_HEADS},{max_seq},{HEAD_DIM}] grid {int(_dg.x)}x{int(_dg.y)} (device grid)"
    )
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        q_t,
        k_t,
        v_t,
        cur_pos_tensor=cp_t,
        scale=SCALE,
        program_config=_prog_cfg(mesh_device),
        compute_kernel_config=_compute_cfg(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)
    o = ttnn.to_torch(out)
    logger.info(f"[sdpa-repro] non-paged decode out shape {tuple(o.shape)} finite={torch.isfinite(o).all().item()}")
    assert torch.isfinite(o).all(), "SDPA decode produced non-finite output"
