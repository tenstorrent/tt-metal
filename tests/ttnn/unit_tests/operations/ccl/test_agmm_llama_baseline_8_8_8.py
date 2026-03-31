# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Baseline fused `all_gather_minimal_matmul_async` aligned with **your** Galaxy `TtModelArgs`
`prefill_ff2_minimal_matmul_config` when **`USE_FUSED_AG_MM=1`** (pasted policy):

  - Blocks: **8×8×8** (tile units) for all listed ISLs.
  - **seq_len ≤ 4096:** subblock **(2,2)**, grid **6×8**
  - **4096 < seq_len ≤ 16384 (8k, 16k):** **(2,2)**, **6×8**
  - **16384 < seq_len ≤ 32768 (32k):** **(4,2)**, **6×8**
  - **32768 < seq_len ≤ 65536 (64k):** **(2,4)**, **6×8**
  - **seq_len ≥ 131072 (128k):** **(2,4)**, **6×9**

**Fabric / links:** `wh_galaxy` — **3 links**, **num_workers_per_link=2**, same as fused sweep.

**Note:** Some tt-metal builds validate `subblock_h * subblock_w <= max_dest_volume` (4) inside
`all_gather_minimal_matmul_async` — configs with product **8** ((4,2) or (2,4)) may **TT_FATAL** on
your checkout. If they pass on your machine, your stack matches what the model uses for fused FF2.

Usage (tt-metal root, Galaxy):
  pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_llama_baseline_8_8_8.py -x -s
  pytest tests/ttnn/unit_tests/operations/ccl/test_agmm_llama_baseline_8_8_8.py -k 64k -x -s
"""

import pytest
import torch
from loguru import logger

import ttnn

DEVICE_CONFIG = {
    "mesh_shape": (8, 4),
    "fabric_config": "FABRIC_1D_RING",
    "fabric_router_config_payload": 7168,
    "topology": "Ring",
    "num_links": 3,
    "num_workers_per_link": 2,
    "cluster_axis": 1,
    "ring_size": 4,
}

K = 3584
N = 2048

M_BLOCK = 8
K_BLOCK = 8
N_BLOCK = 8

WARMUP_ITERS = 1
MEASURED_ITERS = 2

# (M, sub_h, sub_w, core_x, core_y, pytest id) — matches USE_FUSED_AG_MM=1 branch in shared model_args
LLAMA_FUSED_FF2_BASELINE_CASES = [
    (4096, 2, 2, 6, 8, "4k_888_sb22_g68"),
    (8192, 2, 2, 6, 8, "8k_888_sb22_g68"),
    (16384, 2, 2, 6, 8, "16k_888_sb22_g68"),
    (32768, 4, 2, 6, 8, "32k_888_sb42_g68"),
    (65536, 2, 4, 6, 8, "64k_888_sb24_g68"),
    (131072, 2, 4, 6, 9, "128k_888_sb24_g69"),
]

# Alternative cases using (1,4) subblocks for 32k+ to avoid TT_FATAL validation
LLAMA_FUSED_FF2_BASELINE_CASES_1x4 = [
    (32768, 1, 4, 6, 8, "32k_888_sb14_g68"),
    (65536, 1, 4, 6, 8, "64k_888_sb14_g68"),
    (131072, 1, 4, 6, 9, "128k_888_sb14_g69"),
]


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def create_global_semaphores(mesh_device, num_devices, core_range_set, initial_value=0):
    return [ttnn.create_global_semaphore(mesh_device, core_range_set, initial_value) for _ in range(num_devices)]


def open_mesh(cfg):
    fabric_kwargs = [
        getattr(ttnn.FabricConfig, cfg["fabric_config"]),
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    ]
    if cfg["fabric_router_config_payload"] is not None:
        fabric_kwargs.append(create_fabric_router_config(cfg["fabric_router_config_payload"]))
    ttnn.set_fabric_config(*fabric_kwargs)
    rows, cols = cfg["mesh_shape"]
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))


def close_mesh(mesh_device):
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _run_fused_baseline(M: int, sub_h: int, sub_w: int, core_x: int, core_y: int, case_id: str):
    cfg = DEVICE_CONFIG
    cluster_axis = cfg["cluster_axis"]
    topology = getattr(ttnn.Topology, cfg["topology"])

    logger.info(
        f"Llama fused FF2 baseline [{case_id}]: M={M} K={K} N={N} grid={core_x}x{core_y} "
        f"blocks=(8,8,8) subblock=({sub_h},{sub_w}) links={cfg['num_links']} "
        f"(USE_FUSED_AG_MM=1-style policy)"
    )

    mesh_device = open_mesh(cfg)
    try:
        dtype = ttnn.bfloat8_b
        core_grid = ttnn.CoreCoord(core_x, core_y)

        tt_input = ttnn.from_torch(
            torch.randn((M, K), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, cluster_axis]
            ),
        )
        tt_weight = ttnn.from_torch(
            torch.randn((K, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_bias = ttnn.from_torch(
            torch.randn((1, N), dtype=torch.float32),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
        )
        persistent_output_buffer = ttnn.from_torch(
            torch.zeros((M, K), dtype=torch.float32),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None]),
        )
        ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
        )
        ccl_semaphore_handles = create_global_semaphores(mesh_device, mesh_device.get_num_devices(), ccl_cores, 0)
        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=M_BLOCK,
            K_block_size=K_BLOCK,
            N_block_size=N_BLOCK,
            subblock_h=sub_h,
            subblock_w=sub_w,
            compute_with_storage_grid_size=core_grid,
        )

        def run_op():
            ttnn.experimental.all_gather_minimal_matmul_async(
                tt_input,
                tt_weight,
                bias_tensor=tt_bias,
                fused_activation=None,
                compute_kernel_config=compute_config,
                config=matmul_config,
                persistent_output_buffer=persistent_output_buffer,
                multi_device_global_semaphore=ccl_semaphore_handles,
                num_links=cfg["num_links"],
                topology=topology,
                cluster_axis=cluster_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=cfg["num_workers_per_link"],
                num_buffers_per_channel=8,
                scalar=None,
                addcmul_input_tensor1=None,
                addcmul_input_tensor2=None,
                chunks=1,
            )
            ttnn.synchronize_device(mesh_device)

        for _ in range(WARMUP_ITERS):
            run_op()
        logger.info(f"[{case_id}] warmup done")

        from tracy import signpost

        signpost("start")
        for _ in range(MEASURED_ITERS):
            run_op()
        signpost("stop")

        logger.info(f"[{case_id}] measured {MEASURED_ITERS} iters between signposts")
    finally:
        close_mesh(mesh_device)


def _baseline_params():
    out = []
    for M, sh, sw, gx, gy, pid in LLAMA_FUSED_FF2_BASELINE_CASES:
        out.append(pytest.param(M, sh, sw, gx, gy, id=pid))
    return out


def _baseline_1x4_params():
    out = []
    for M, sh, sw, gx, gy, pid in LLAMA_FUSED_FF2_BASELINE_CASES_1x4:
        out.append(pytest.param(M, sh, sw, gx, gy, id=pid))
    return out


@pytest.mark.timeout(21600)
@pytest.mark.parametrize("M,sub_h,sub_w,core_x,core_y", _baseline_params())
def test_agmm_llama_baseline_blocks_8_8_8(M, sub_h, sub_w, core_x, core_y):
    """Fused AGMM baseline matching USE_FUSED_AG_MM=1 FF2 minimal config (see module doc)."""
    case_id = f"M{M}_g{core_x}x{core_y}_sb{sub_h}x{sub_w}"
    _run_fused_baseline(M, sub_h, sub_w, core_x, core_y, case_id)


@pytest.mark.timeout(21600)
@pytest.mark.parametrize("M,sub_h,sub_w,core_x,core_y", _baseline_1x4_params())
def test_agmm_llama_baseline_blocks_8_8_8_subblock_1x4(M, sub_h, sub_w, core_x, core_y):
    """Fused AGMM baseline for 32k+ using (1,4) subblocks to avoid TT_FATAL validation."""
    case_id = f"M{M}_g{core_x}x{core_y}_sb{sub_h}x{sub_w}_1x4"
    _run_fused_baseline(M, sub_h, sub_w, core_x, core_y, case_id)
