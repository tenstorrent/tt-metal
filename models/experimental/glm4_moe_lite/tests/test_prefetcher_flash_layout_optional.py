# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end on-device test of Flash's OWN prefetcher layout.

test_prefetcher_ring_shapes_optional.py validates Flash's weight shapes through the
canonical upstream harness, but that harness carries llama's core layout, which is
hardcoded for the 7x10 COL-dispatch grid. Flash runs ROW dispatch on Galaxy and gets
an 8x9 grid, so it needs its own layout -- this test exercises that layout:

    get_glm_core_ranges (8 senders col 6, 16 receivers cols 4-5, workers cols 0-5)
      -> SubDevice manager create + load
      -> GlobalCB at the computed 640-tile size
      -> DRAM-WIDTH_SHARDED w_o-shaped weights over the 8 prefetcher DRAM banks
      -> dram_prefetcher
      -> gather_in0 ring matmul consuming the GlobalCB
      -> PCC against a torch reference

This is the last piece that was unproven before model integration. Everything here
runs outside the model, so a deadlock costs one Galaxy reset instead of a hung
47-layer run that is hard to attribute.

Grid facts (measured, not assumed):
    ROW dispatch -> 8x9 = 72 cores   <- Flash on Galaxy
    COL dispatch -> 7x10 = 70 cores  <- upstream prefetcher harness
    DRAM grid    -> 12x1 in both
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_HW_TESTS") != "1",
        reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
    ),
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_MULTI_DEVICE_TESTS") != "1",
        reason="Enable with TT_ENABLE_MULTI_DEVICE_TESTS=1 (opens a 4x8 Galaxy mesh).",
    ),
]

# Flash o_proj. 160 x 64 tiles -> the 16-core ring; 640-tile GlobalCB.
K, N = 5120, 2048
M = 32  # one tile of padded decode batch
NUM_LAYERS = 4  # enough for the prefetcher to cycle addresses; keeps the test short


@pytest.mark.parametrize("mesh_device", [pytest.param((4, 8), id="4x8_galaxy")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 23887872}],
    indirect=True,
)
def test_flash_layout_prefetch_and_ring_matmul(mesh_device, device_params, function_level_defaults) -> None:
    from models.experimental.glm4_moe_lite.tt.prefetcher_setup import (
        Glm4MoeLitePrefetcherSetup,
        global_cb_tiles_for,
    )

    device = mesh_device
    grid = device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (8, 9):
        pytest.skip(f"Flash prefetcher layout targets an 8x9 grid, got {grid.x}x{grid.y}")

    setup = Glm4MoeLitePrefetcherSetup(device, n_tensors_per_layer=1, n_layers=NUM_LAYERS)
    ring = setup.RING_CORES
    assert len(setup.receiver_cores) == ring
    assert setup.global_cb_size == global_cb_tiles_for(K, N, ring) * 1088

    # --- DRAM-WIDTH_SHARDED weights, one per layer, over the prefetcher's DRAM banks.
    dram_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in setup.dram_cores])
    assert N % len(setup.dram_cores) == 0, f"N={N} must divide across {len(setup.dram_cores)} banks"
    weight_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_core_range_set, [K, N // len(setup.dram_cores)], ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch.manual_seed(0)
    torch_weights = [torch.randn(K, N) for _ in range(NUM_LAYERS)]
    for w in torch_weights:
        tt_w = ttnn.as_tensor(
            w,
            device=device,
            dtype=ttnn.bfloat8_b,
            memory_config=weight_mem_cfg,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        setup.insert_tensor(tt_w)

    # --- Activation, WIDTH_SHARDED on exactly the ring cores.
    torch_act = torch.randn(1, 1, M, K)
    act = ttnn.as_tensor(
        torch_act,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=setup.oproj_input_mem_cfg,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def run_once():
        """One dram_prefetcher immediately followed by its consumers.

        CRITICAL ORDERING: dram_prefetcher must be issued exactly once per
        consumption sequence, with the consuming matmuls right behind it. It fills the
        GlobalCB and then STALLS until a consumer drains it, so two prefetcher calls
        with no consumer in between deadlock the device -- verified the hard way
        (2935% CPU spin, Galaxy reset required).

        That is why compile_prefetch() is not used here. It exists for the traced
        model path, where the second issue happens inside trace capture (which records
        rather than executes) and the compile-time stall drains when the traced
        consumers run. Outside a trace there is nothing to drain it.

        The canonical upstream harness has the same shape: one run_op() containing
        prefetcher + consumers, called twice (compile, then measure).
        """
        garbage = setup.start_prefetch()
        outs = []
        for layer in range(NUM_LAYERS):
            outs.append(
                ttnn.matmul(
                    act,
                    setup.tensors[layer],
                    program_config=setup.oproj_program_config,
                    memory_config=setup.oproj_output_mem_cfg,
                    compute_kernel_config=ckc,
                    global_cb=setup.global_circular_buffer,
                    sub_device_id=setup.worker_sub_device_id,
                    dtype=ttnn.bfloat16,
                )
            )
        setup.stop_prefetch(garbage)
        return outs

    try:
        setup.ensure_ready()
        outputs = run_once()

        # --- Compare against torch. Compose only device 0's shard: the weights are
        # replicated, so every device computes the same product.
        from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

        for layer, out in enumerate(outputs):
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[:1]
            got = got.reshape(M, N)
            want = (torch_act.reshape(M, K).to(torch.float32)) @ torch_weights[layer].to(torch.float32)
            passing, msg = comp_pcc(want, got.to(torch.float32), pcc=0.99)
            assert passing, f"layer {layer} ring matmul PCC failed: {msg}"
    finally:
        setup.teardown()
