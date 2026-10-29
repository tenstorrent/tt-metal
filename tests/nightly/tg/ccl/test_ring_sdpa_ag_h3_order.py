# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Synthetic repro of tt-metal #62923 (H3 hangs "bug b" and "bug c"), sharing semaphores exactly as H3 does.

H3 (models/tt_dit, minimax_h3 on a 4x8 Blackhole Galaxy) runs, per attention block on the SP axis (mesh axis 1),
ring joint SDPA and five FSDP weight all-gathers. All of them take their out_ready semaphores from one CCLManager: the
FSDP gathers through all_gather_persistent_buffer -> get_ag_ping_pong_semaphore(1) (models/tt_dit/layers/linear.py),
and before PR #62456 ring SDPA through the same getter (attention_minimax_h3.py). H3's captured axis-1 getter order
(stock build, 153 ring calls, ring always pp1) is
    ring SDPA pp1 -> W pp0 -> W pp1 -> W pp0 -> W pp1 -> W pp0 -> ring SDPA pp1
so ring SDPA and the second weight gather after it wait on the same out_ready semaphores (pp1) on the same CCL core.
This test builds that CCLManager (2 links, Linear, full-grid CCL cores, sub-device 0) and calls the same ops with H3's
arguments in that order: per iteration W_a pp0 -> ring SDPA pp1 -> W_b pp0 -> W_c pp1 -> W_d pp0 -> W_e pp1.
BUGBC_BLOCK=short drops W_d and W_e (the 4-op order of the first device runs). Ring SDPA takes its semaphores from the
getter attention_minimax_h3.py uses in the tree under test: get_ag_ping_pong_semaphore before PR #62456,
get_ring_sdpa_ping_pong_semaphore with it. Axis-0 ops are left out: they draw from the axis-0 pools, which no axis-1
op touches. Absolute semaphore addresses differ from H3's by a fixed offset (H3 allocates other CCLManagers first);
BUGBC_POOL logs both.

The stock ring all-gather reader ends with noc_semaphore_set(out_ready_sem, 0). If a neighbour's increments for the
next op on the same address land before that reset, they are erased and that op waits forever, 2 credits short.
The race window is widened with a test-only spin just before the reset (head: before its subtract), compiled in only
with the watcher's debug delay and run only on the chip/core the atomic delay mask selects:
  TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_DEBUG_DELAY=<cycles>
  TT_METAL_ATOMIC_DEBUG_DELAY_CHIPS=<chip id> TT_METAL_ATOMIC_DEBUG_DELAY_CORES="(11,0)"
  TT_METAL_ATOMIC_DEBUG_DELAY_RISCVS=TR0
(TR0 = TRISC0, which issues no NOC atomics, so the mask slows nothing else.) With no delay set the test is its own
control. Set TT_METAL_OPERATION_TIMEOUT_SECONDS so a hang raises; the test then reads every axis-1 out_ready semaphore
with direct L1 reads (no dispatch), again 30 s later, and fails with BUGBC_RESULT HANG.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal

ITERS = int(os.environ.get("BUGBC_ITERS", "20"))
# H3 768P 5 s: 4736 rows per device, measured chunks (320, 384); 14 local heads of 128 (TP 4).
SEQ_LOCAL = int(os.environ.get("BUGBC_SEQ_LOCAL", "4736"))
N_LOCAL_HEADS, HEAD_DIM = 14, 128
# FSDP weight shard gathered on dim 2 over axis 1; weight-sized so AllGatherAsync picks 4 workers per direction and
# puts a reader on the CCL column core (11,0), as in H3.
W_ROWS, W_COLS = 5376, 5376
# H3's five FSDP weight gathers per block (default), or the earlier 3-gather order.
BLOCK = os.environ.get("BUGBC_BLOCK", "h3")
assert BLOCK in ("h3", "short"), f"BUGBC_BLOCK={BLOCK}: use h3 or short"
# H3 allocates its ring K/V gather buffers with kv_gather_capacity rows (bucketed denoise); unset = exact size.
KV_CAPACITY = int(os.environ["BUGBC_KV_CAPACITY"]) if os.environ.get("BUGBC_KV_CAPACITY") else None
# H3's axis-1 ag pool addresses (getter log of a stock-build H3 soak on one 4x8 BH Galaxy), for the side-by-side log.
H3_ADDRS = {"pp0": (0x17B8C0, 0x17B880), "pp1": (0x17B840, 0x17B800)}

GETS = []  # (iter, caller, getter, pp, addrs)


def chunk_sizes(seq_local):
    measured = {4736: (320, 384), 9184: (256, 512), 13664: (256, 512)}
    if seq_local in measured:
        return measured[seq_local]
    tile = ttnn.TILE_SIZE
    return max(tile, min(256, (seq_local // tile) * tile)), max(tile, min(512, (seq_local // tile) * tile))


def read_sems(mesh_device, named_sems, cores):
    """Nonzero out_ready values: (name, row, col, x, y) -> value. Direct L1 reads, no dispatch."""
    values = {}
    for name, sem in named_sems:
        addr = ttnn.get_global_semaphore_address(sem)
        for row in range(mesh_device.shape[0]):
            for col in range(mesh_device.shape[1]):
                coord = ttnn.MeshCoordinate(row, col)
                for core in cores:
                    word = list(ttnn.experimental.reload.read_core_l1(mesh_device, core, addr, 4, coord))[0]
                    if word:
                        values[(name, row, col, core.x, core.y)] = word
    return values


def log_sems(tag, values, device_ids):
    if not values:
        logger.info(f"BUGBC_SEMS {tag}: all axis-1 out_ready semaphores 0")
        return
    for (name, row, col, x, y), v in sorted(values.items()):
        logger.info(f"BUGBC_SEMS {tag}: {name} mesh({row},{col}) dev{device_ids[(row, col)]} core({x},{y}) = {v}")


@pytest.mark.timeout(int(os.environ.get("BUGBC_PYTEST_TIMEOUT", "1500")))
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}],
    ids=["fabric_1d"],
    indirect=True,
)
def test_ring_sdpa_then_fsdp_gathers_h3_order(mesh_device):
    sp_axis, tp_axis = 1, 0
    rows, cols = tuple(mesh_device.shape)
    device_ids = {
        (r, c): mesh_device.get_device_id(ttnn.MeshCoordinate(r, c)) for r in range(rows) for c in range(cols)
    }
    logger.info(f"BUGBC_MESH device ids by (row,col): {device_ids}")
    slow = {k: v for k, v in os.environ.items() if "DEBUG_DELAY" in k or k == "TT_METAL_WATCHER"}
    logger.info(f"BUGBC_SLOW {slow if slow else 'none (control)'}")

    # H3's BH 4x8 preset: 2 links; prod runs the fabric 1D, so Linear.
    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    has_ring_pool = hasattr(ccl, "get_ring_sdpa_ping_pong_semaphore")
    caller = {"iter": -1, "op": "?"}

    def logged(getter_name):
        orig = getattr(ccl, getter_name)

        def get(mesh_axis):
            idx = ccl.ag_ping_pong_idx[mesh_axis] if getter_name == "get_ag_ping_pong_semaphore" else None
            sems = orig(mesh_axis)
            addrs = tuple(ttnn.get_global_semaphore_address(s) for s in sems)
            GETS.append((caller["iter"], caller["op"], getter_name, idx, addrs))
            if caller["iter"] < 2:
                logger.info(
                    f"BUGBC_GET iter {caller['iter']} {caller['op']} axis {mesh_axis} {getter_name} "
                    f"pp{idx if idx is not None else '-'} addrs {[hex(a) for a in addrs]}"
                )
            return sems

        setattr(ccl, getter_name, get)

    logged("get_ag_ping_pong_semaphore")
    if has_ring_pool:
        logged("get_ring_sdpa_ping_pong_semaphore")
    # The getter attention_minimax_h3.py calls for ring SDPA in this tree.
    ring_sem_getter = ccl.get_ring_sdpa_ping_pong_semaphore if has_ring_pool else ccl.get_ag_ping_pong_semaphore

    pools = [(f"ag{k}", s) for k, s in enumerate(ccl.ag_ping_pong_semaphores[sp_axis])]
    if has_ring_pool:
        pools += [(f"ring{k}", s) for k, s in enumerate(ccl.ring_sdpa_ping_pong_semaphores[sp_axis])]
    for name, s in pools:
        logger.info(f"BUGBC_POOL axis {sp_axis} {name} addr {hex(ttnn.get_global_semaphore_address(s))}")
    logger.info(
        f"BUGBC_POOL H3 axis 1 ag pool: pp0 {[hex(a) for a in H3_ADDRS['pp0']]} pp1 {[hex(a) for a in H3_ADDRS['pp1']]}"
    )

    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_worker_grid = (grid.x - 1, grid.y)  # H3 reserves the last column for CCL
    # AllGatherAsync workers fill rows 0-1 from (0,0); ring SDPA's CCL readers run column-major from (grid.x-1, 0).
    sem_cores = [ttnn.CoreCoord(x, y) for y in range(2) for x in range(grid.x)]
    sem_cores += [ttnn.CoreCoord(grid.x - 1, y) for y in range(2, grid.y)]

    q_chunk, k_chunk = chunk_sizes(SEQ_LOCAL)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*sdpa_worker_grid),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    logger.info(
        f"BUGBC_CONFIG seq_local {SEQ_LOCAL} chunks ({q_chunk},{k_chunk}) heads {N_LOCAL_HEADS}x{HEAD_DIM} "
        f"links {ccl.num_links} {ccl.topology} grid {grid.x}x{grid.y} block {BLOCK} kv_capacity {KV_CAPACITY} ring SDPA getter "
        f"{'get_ring_sdpa_ping_pong_semaphore' if has_ring_pool else 'get_ag_ping_pong_semaphore'}"
    )

    def put(t, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims),
        )

    # q/k/v: heads over the TP axis (rows), sequence over the SP axis (columns), as q_BHNE in H3.
    qkv_shape = [1, N_LOCAL_HEADS * rows, SEQ_LOCAL * cols, HEAD_DIM]
    q, k, v = (put(torch.randn(qkv_shape).bfloat16(), (1, 2)) for _ in range(3))
    dummy_joint_input = bf16_tensor(torch.zeros((1, N_LOCAL_HEADS, 0, HEAD_DIM)), device=mesh_device)
    # FSDP weight: 2D, rows sharded over the SP axis (gathered back by the FSDP all-gather), columns over TP.
    w_torch = torch.randn(W_ROWS, W_COLS).bfloat16()
    w = put(w_torch, (1, 0))

    def weight_gather(op):
        caller["op"] = op
        return ccl.all_gather_persistent_buffer(ttnn.unsqueeze_to_4D(w), dim=2, mesh_axis=sp_axis)

    def ring_sdpa():
        caller["op"] = "ring_sdpa"
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            dummy_joint_input,
            dummy_joint_input,
            dummy_joint_input,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(
                k.shape, 2, sp_axis, dtype=k.dtype, capacity=KV_CAPACITY
            ),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(
                v.shape, 2, sp_axis, dtype=v.dtype, capacity=KV_CAPACITY
            ),
            joint_strategy="rear",
            logical_n=SEQ_LOCAL * cols,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=ring_sem_getter(sp_axis),
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ccl.topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(sdpa_worker_grid[0], 0),
            use_column_major_ccl=True,
        )

    sem_list = list(pools)
    ttnn.synchronize_device(mesh_device)
    t_start = time.time()
    for i in range(ITERS):
        caller["iter"] = i
        t0 = time.time()
        try:
            # No host sync inside the block: as in the model, a fast rank moves on to the next op.
            weight_gather("W_a")
            ring_sdpa()
            weight_gather("W_b")
            out = weight_gather("W_c")
            if BLOCK == "h3":
                weight_gather("W_d")
                out = weight_gather("W_e")
            ttnn.synchronize_device(mesh_device)
        except Exception as e:
            dt = time.time() - t0
            logger.error(
                f"BUGBC_HANG iter {i} after {dt:.0f} s ({time.time() - t_start:.0f} s into the loop): "
                f"{type(e).__name__}: {str(e)[:300]}"
            )
            for g in GETS[-6:]:
                logger.error(f"BUGBC_HANG getter {g[1]} {g[2]} pp{g[3]} {[hex(a) for a in g[4]]}")
            log_sems(f"hang iter {i}", read_sems(mesh_device, sem_list, sem_cores), device_ids)
            # A slow rank still working would move the counters; an erased credit leaves them stuck.
            time.sleep(30)
            log_sems(f"hang iter {i} +30 s", read_sems(mesh_device, sem_list, sem_cores), device_ids)
            logger.error(f"BUGBC_RESULT HANG at iter {i} of {ITERS}, {time.time() - t_start:.0f} s into the loop")
            pytest.fail(f"BUGBC_RESULT HANG at iter {i} of {ITERS}")

        # Every op consumed exactly the credits it waited for: every axis-1 out_ready reads 0.
        left = read_sems(mesh_device, sem_list, sem_cores)
        log_sems(f"after iter {i} ({time.time() - t0:.2f} s)", left, device_ids)
        assert not left, f"iter {i}: out_ready credits left over: {left}"

    coords = [(r, c) for r in range(rows) for c in range(cols)]
    shard = W_COLS // rows
    for (r, c), dev_out in zip(coords, ttnn.get_device_tensors(out)):
        eq, msg = comp_equal(ttnn.to_torch(dev_out).reshape(W_ROWS, shard), w_torch[:, r * shard : (r + 1) * shard])
        assert eq, f"last weight gather wrong on mesh({r},{c}): {msg}"

    # Which pool slot each op drew, for the comparison with H3's getter table.
    order = [
        f"{op}:{getter.replace('get_', '').replace('_ping_pong_semaphore', '')}/pp{pp}"
        for it, op, getter, pp, _ in GETS
        if it == 1
    ]
    logger.info(f"BUGBC_ORDER iter 1: {' -> '.join(order)}")
    logger.info(f"BUGBC_RESULT PASS {ITERS} iters in {time.time() - t_start:.0f} s")
