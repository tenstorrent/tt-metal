# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the 8x4 mesh tells the attention layer program's builder: the worker grid, L1 per core, the fabric payload,
whether the one-chip pieces build on a mesh device, and the circular-buffer bytes every core of each piece takes
(the L1 budget of PLAN.md section 4). Mock or board; nothing is launched. Log lines start with HY4_MK8A."""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.fused.projections import program as P
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.op import SparseMla
from models.experimental.hy4_preview.tt import prefill_bridge as pb

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(pb.device_params(), (8, 4))], indirect=True, ids=["8x4"]
)


def l1(mesh_device):
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    return v.total_bytes_per_bank, v.total_bytes_allocated_per_bank


def cb_bytes_per_core(desc, grid):
    """{(x, y): bytes of the program's own circular buffers on that core}, as the allocator lays them out: a
    buffer over several core ranges starts at the highest end of any of them (program.cpp allocate_dfb)."""
    end = {(x, y): 0 for x in range(grid[0]) for y in range(grid[1])}
    for cb in desc.cbs:
        cores = [
            (x, y)
            for r in cb.core_ranges.ranges()
            for x in range(r.start.x, r.end.x + 1)
            for y in range(r.start.y, r.end.y + 1)
        ]
        try:
            if cb.buffer is not None or getattr(cb, "global_circular_buffer", None) is not None:
                continue
        except Exception:
            pass
        start = max(end[c] for c in cores)
        for c in cores:
            end[c] = start + cb.total_size
    return end


@galaxy
def test_probe_mesh(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    g = (grid.x, grid.y)
    logger.info(
        f"HY4_MK8A probe grid {g}, L1 per core (total, allocated) {l1(mesh_device)}, fabric max payload {ttnn.get_tt_fabric_max_payload_size_bytes()}"
    )
    logger.info(f"HY4_MK8A probe CBDescriptor attrs {[a for a in dir(ttnn.CBDescriptor) if not a.startswith('_')]}")

    # sparse_mla on the mesh
    cfg = SL.SparseMlaConfig(rows_per_slot=4096, num_layers=1)
    before = l1(mesh_device)[1]
    op = SparseMla(cfg, mesh_device)
    logger.info(
        f"HY4_MK8A probe sparse_mla on a mesh: global semaphores at {[hex(a) for a in op.sem]}, L1 tensors {l1(mesh_device)[1] - before} B"
    )
    chips = mesh_device.get_num_devices()
    kv = ttnn.from_torch(
        torch.zeros(cfg.users, 1, cfg.rows_per_slot, cfg.row_bytes),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs

    kv = ttnn.to_memory_config(kv, hs.kv_memory_config(mesh_device, cfg))
    sel = hs.make_selection(mesh_device, cfg, [torch.arange(4)] * 8)
    q = hs.to_dev(torch.zeros(1, 1, 64, op.q_page_values), mesh_device, ttnn.bfloat16)
    out = hs.to_dev(torch.zeros(1, 1, 64, op.out_page_values), mesh_device, ttnn.bfloat16)
    desc = op.attend_program(kv, sel, q, out, 0)
    per = cb_bytes_per_core(desc, g)
    logger.info(
        f"HY4_MK8A probe sparse_mla attend CB bytes: attention core (1, 0) {per[(1, 0)]}, reader core (0, 0) {per[(0, 0)]}, max {max(per.values())}"
    )
    sizes = sorted(
        ((cb.total_size, cb.format_descriptors[0].buffer_index, cb.core_ranges.num_cores()) for cb in desc.cbs),
        reverse=True,
    )
    logger.info(f"HY4_MK8A probe sparse_mla attend CBs (bytes, id, cores) {sizes}")
    logger.info(
        f"HY4_MK8A probe sparse_mla kernels {[(str(k.kernel_source).split('/')[-1], k.core_ranges.num_cores()) for k in desc.kernels]}"
    )
    ttnn.experimental.prepare_generic_op([kv, sel, q, out], desc)
    logger.info("HY4_MK8A probe sparse_mla attend compiles on the mesh")

    # projections on the mesh
    for name, build, kw in (
        ("pre", P.pre_program, {}),
        ("pre gate_ct=4", P.pre_program, {"gate_ct": 4}),
        ("query", P.query_program, {}),
        ("post", P.post_program, {}),
        ("post o_ct=12", P.post_program, {"o_ct": 12}),
    ):
        before = l1(mesh_device)[1]
        w = PP.placeholder_share(0, False)
        prog = build(mesh_device, w, **kw)
        d, io = prog.descriptor()
        per = cb_bytes_per_core(d, g)
        logger.info(
            f"HY4_MK8A probe proj {name}: {len(prog.units)} units, hub {(prog.hub.x, prog.hub.y)}, cores "
            f"{sorted({(c.x, c.y) for c in prog.core_of})}, L1 tensors {l1(mesh_device)[1] - before} B, CB bytes max "
            f"{max(per.values())} min {min(per.values())}, kernels {len(d.kernels)}, weight bytes {prog.weight_bytes}"
        )
        ttnn.experimental.prepare_generic_op(io, d)
    logger.info(f"HY4_MK8A probe proj compiles on the mesh; L1 allocated now {l1(mesh_device)}")
