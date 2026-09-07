# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (untracked) microbenchmark for the chunked paged SDPA at the Qwen3.6-27B TP=4 shape.

Per device: Q [1, 6, 2048, 256] bf16, paged K/V [num_blocks, 1, 64, 256] (bf16 or bf8), 1 kv head,
FLEXIBLE chunk_start_idx_tensor path (as used by the traced model), HiFi2 fp32-dest compute.
Runs each chunk index in SDPA_BENCH_CHUNKS several times between tracy signposts and checks PCC
against a torch reference for that chunk.

  source profiles/env.sh
  SDPA_BENCH_CHUNKS=0,1,3,7,15 pytest models/demos/blackhole/qwen36/tests/test_sdpa_chunked_bench_scratch.py -k bench
  # or under the device profiler: profiles/prof_sdpa.sh <name>

Env knobs:
  SDPA_BENCH_CHUNKS   comma list of chunk indices (prior 2048-token chunks)          default 0,1,3,7,15
  SDPA_BENCH_QK       q/k chunk size                                                default 128
  SDPA_BENCH_KV       bf16 | bf8                                                    default bf16
  SDPA_BENCH_REPEATS  timed repeats per chunk index                                 default 5
  SDPA_BENCH_NH       local q heads                                                 default 6
  SDPA_BENCH_PCC      run torch reference PCC check (1/0)                           default 1
  SDPA_BENCH_FULLSYNC 1 -> dst_full_sync_en (fp32 dest 8 tiles, 2x4 matmul subblocks)     default 0
  SDPA_BENCH_KCHUNK   k chunk size (default = SDPA_BENCH_QK)
  SDPA_BENCH_MESH     1 -> open the (1,4) mesh and replicate (model env); 0 -> single device   default 1
"""
import os
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import comp_pcc

CHUNK = 2048
BLOCK = 64
HD = 256


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _run_bench(device, replicate):
    chunks = [int(c) for c in os.environ.get("SDPA_BENCH_CHUNKS", "0,1,3,7,15").split(",")]
    qk = _env_int("SDPA_BENCH_QK", 128)
    kchunk = _env_int("SDPA_BENCH_KCHUNK", qk)
    kv_dtype = ttnn.bfloat8_b if os.environ.get("SDPA_BENCH_KV", "bf16") == "bf8" else ttnn.bfloat16
    repeats = _env_int("SDPA_BENCH_REPEATS", 5)
    nh = _env_int("SDPA_BENCH_NH", 6)
    do_pcc = os.environ.get("SDPA_BENCH_PCC", "1") == "1"
    nkv = 1
    max_chunk = max(chunks)
    total_len = (max_chunk + 1) * CHUNK
    num_blocks = ((total_len // BLOCK) + 31) // 32 * 32

    compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=os.environ.get("SDPA_BENCH_FP32DEST", "1") == "1",  # 0 -> bf16 DEST (8 tiles, 2x4 subblocks)
        packer_l1_acc=True,
        dst_full_sync_en=os.environ.get("SDPA_BENCH_FULLSYNC", "0") == "1",  # 8 fp32 dest tiles -> 2x4 subblocks
    )
    grid = device.compute_with_storage_grid_size()
    sdpa_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        exp_approx_mode=False,
        q_chunk_size=qk,
        k_chunk_size=kchunk,
    )
    logger.info(
        f"[SDPA_BENCH] grid={grid.x}x{grid.y} q={qk} k={kchunk} kv={kv_dtype} nh={nh} chunks={chunks} blocks={num_blocks}"
    )

    torch.manual_seed(0)
    Q = torch.randn(1, nh, total_len, HD) * 1.5
    K = torch.randn(1, nkv, total_len, HD)
    V = torch.randn(1, nkv, total_len, HD)
    # Identity page table (model uses arange page tables as well); pad blocks beyond total_len with zeros.
    paged_k = torch.zeros(num_blocks, nkv, BLOCK, HD)
    paged_v = torch.zeros(num_blocks, nkv, BLOCK, HD)
    nb_used = total_len // BLOCK
    paged_k[:nb_used] = K.reshape(1, nkv, nb_used, BLOCK, HD).transpose(1, 2).reshape(nb_used, nkv, BLOCK, HD)
    paged_v[:nb_used] = V.reshape(1, nkv, nb_used, BLOCK, HD).transpose(1, 2).reshape(nb_used, nkv, BLOCK, HD)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    mesh_kw = {}
    if replicate:
        mesh_kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)}
    tt_k = ttnn.from_torch(paged_k, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device, **mesh_kw)
    tt_v = ttnn.from_torch(paged_v, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device, **mesh_kw)
    tt_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, **mesh_kw)
    tt_q = ttnn.from_torch(Q[:, :, :CHUNK], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, **mesh_kw)
    tt_csi = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, **mesh_kw
    )

    def run():
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=tt_q,
            input_tensor_k=tt_k,
            input_tensor_v=tt_v,
            page_table_tensor=tt_pt,
            chunk_start_idx_tensor=tt_csi,
            compute_kernel_config=compute_cfg,
            program_config=sdpa_cfg,
        )

    # KV/Q dtype-rounded reference inputs
    if kv_dtype == ttnn.bfloat8_b:
        K_ref = ttnn.to_torch(ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        V_ref = ttnn.to_torch(ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    else:
        K_ref, V_ref = K.bfloat16().float(), V.bfloat16().float()

    results = {}
    for c in chunks:
        start = c * CHUNK
        q_host = Q[:, :, start : start + CHUNK].bfloat16()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(q_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **mesh_kw), tt_q
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mesh_kw
            ),
            tt_csi,
        )
        out = run()  # compile / warm
        ttnn.synchronize_device(device)
        times = []
        det = os.environ.get("SDPA_BENCH_DETERMINISM", "0") == "1"
        _comp = ttnn.ConcatMeshToTensor(device, dim=0) if replicate else None
        _fetch = lambda t: (ttnn.to_torch(t, mesh_composer=_comp) if replicate else ttnn.to_torch(t))
        first = _fetch(out) if det else None
        mismatches = 0
        maxd = 0.0
        signpost(f"start_c{c}")
        for _ in range(repeats):
            t0 = time.perf_counter()
            o = run()
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
            if det:
                cur = _fetch(o)
                if not torch.equal(cur, first):
                    mismatches += 1
                    maxd = max(maxd, float((cur.float() - first.float()).abs().max()))
            ttnn.deallocate(o)
        signpost(f"stop_c{c}")
        if det:
            print(
                f"SDPA_BENCH_DETERMINISM chunk={c} repeats={repeats} mismatching_runs={mismatches} max|d|={maxd:.4e}",
                flush=True,
            )
        host_ms = min(times) * 1000
        pcc = None
        if do_pcc:
            if replicate:
                got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:1]
            else:
                got = ttnn.to_torch(out)
            got = got.float()
            L = start + CHUNK
            qf = q_host.float()
            kf = K_ref[:, :, :L].expand(1, nh, L, HD)
            vf = V_ref[:, :, :L].expand(1, nh, L, HD)
            mask = torch.ones(CHUNK, L, dtype=torch.bool).tril(diagonal=start)
            ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf, attn_mask=mask, scale=HD**-0.5)
            _, pcc = comp_pcc(ref, got, 0.99)
        ttnn.deallocate(out)
        results[c] = (host_ms, pcc)
        logger.info(f"[SDPA_BENCH] chunk={c} start={start} host_min_ms={host_ms:.3f} pcc={pcc}")
    for c, (ms, pcc) in results.items():
        print(f"SDPA_BENCH_RESULT chunk={c} host_min_ms={ms:.3f} pcc={pcc}")
    return results


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_sdpa_chunked_bench_mesh(mesh_device):
    mesh_device.enable_program_cache()
    _run_bench(mesh_device, replicate=True)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_sdpa_chunked_bench_single(device):
    device.enable_program_cache()
    _run_bench(device, replicate=False)
