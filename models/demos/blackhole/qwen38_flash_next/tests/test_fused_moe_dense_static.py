# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The MoE dense composite (``ttnn/fused/moe_dense``) and the core placement helper without a device: the registry
entry, the placement on both p150 compute grids (the top-k's rectangle, the down workers and the untilize cores
disjoint from each other and from the reserved set), the down kernel's argument layout and its arithmetic pins (the
gr_read DOWN kernel's spill/reload sequence, the DRAM-sharded down linear's one-tile K block), the composite's CB and
semaphore tables, and the model's switch sites."""

import inspect
import re
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import dram_sharded_matmul_configs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_dense as md
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import placement
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import shared_expert as se

MODEL_DIR = fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next"
DOWN_KERNEL = (MODEL_DIR / "ttnn/fused/moe_dense/kernels/down_compute.cpp").read_text()
GR_DOWN_KERNEL = (MODEL_DIR / "ttnn/fused/gr_read/kernels/down_compute.cpp").read_text()
MOE = (MODEL_DIR / "ttnn/moe.py").read_text()
GRIDS = {"p150 harvested (120 Tensix)": (11, 10), "p150 unharvested (130 Tensix)": (12, 10)}
# The census die's exclusion set as read on 2026-09-25 (an 11 x 10 p150): the five storage cores, the eight primary
# DRAM reader cores in columns 0 and 6, the drain core (10, 9); the real set is read from the mesh by
# placement.reserved_cores.  The far corner's 4 x 5 holds the drain core, so the top-k lands one row up (y 4..8).
READERS = frozenset({(0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4)})
EXAMPLE_RESERVED = frozenset(set(placement.STORAGE_CORES) | READERS | {(10, 9)})


def _rect_cores(rect):
    return set(placement.rectangle_cores(rect))


def test_registered_bitwise_with_a_gate():
    entry = fused.kernel(md.NAME)
    assert entry.tolerance == fused.BITWISE and entry.fused is md.moe_dense and entry.composed is md.moe_dense_composed
    assert entry.gate is not None and entry.gate.layers == tuple(range(48)) and entry.gate.reference is None
    assert entry.admits is None
    assert (
        entry.default_on
    )  # default since 2026-09-26: bitwise on the device (2026-09-25) at rows 1/5/32, both weight formats and placements
    # the two forms take the union of each other's keyword arguments
    fused_params = inspect.signature(md.moe_dense).parameters
    composed_params = inspect.signature(md.moe_dense_composed).parameters
    for name in (
        "logits",
        "gate_up_scalar_ws",
        "down",
        "full_hidden",
        "scores",
        "indices",
        "top_k",
        "compute_kernel_config",
    ):
        assert name in fused_params and name in composed_params
    assert "partial_memory_config" in fused_params and "_composed_only" in fused_params
    assert {"down_program_config", "intermediate_memory_config", "_fused_only"} <= set(composed_params)


@pytest.mark.parametrize("grid", list(GRIDS.values()), ids=list(GRIDS))
def test_placement_scans_from_the_far_corner_and_keeps_the_groups_disjoint(grid):
    x, y = grid
    topk = placement.free_rectangle_in(x, y, EXAMPLE_RESERVED, *rt.RECTANGLE)
    if x == 11:
        assert topk == (7, 4, 10, 8)  # the far corner, one row up from the drain core (the device readout's rectangle)
    assert (10, 9) not in _rect_cores(topk) and not _rect_cores(topk) & READERS
    taken = set(EXAMPLE_RESERVED) | _rect_cores(topk)
    workers = placement.free_rectangle_in(x, y, taken, *md.DOWN_SHAPES[0])
    taken |= _rect_cores(workers)
    untilize = placement.free_rectangle_in(x, y, taken, *md.UNTILIZE_SHAPE)
    groups = [_rect_cores(topk), _rect_cores(workers), _rect_cores(untilize), set(placement.STORAGE_CORES)]
    for i, a in enumerate(groups):
        assert all(0 <= cx < x and 0 <= cy < y for cx, cy in a)
        for b in groups[i + 1 :]:
            assert not (a & b)
    assert len(_rect_cores(workers)) == md.DOWN_WORKERS and len(_rect_cores(untilize)) * 10 == md.N_TILES
    assert not (_rect_cores(topk) | _rect_cores(workers) | _rect_cores(untilize)) & EXAMPLE_RESERVED


def test_placement_refuses_what_does_not_fit_and_honors_accept(expect_error):
    with expect_error(RuntimeError, match="no free 4x5 rectangle"):
        placement.free_rectangle_in(4, 5, {(0, 0)}, 4, 5)
    with expect_error(ValueError):
        placement.free_rectangle_in(3, 3, (), 4, 1)
    # accept refuses the far-corner candidate: the scan moves on (y first, then x)
    first = placement.free_rectangle_in(11, 10, (), 2, 2)
    second = placement.free_rectangle_in(11, 10, (), 2, 2, accept=lambda rect: rect != first)
    assert first == (9, 8, 10, 9) and second == (9, 7, 10, 8)
    assert placement.rectangle_cores((1, 2, 2, 3)) == [(1, 2), (2, 2), (1, 3), (2, 3)]  # row-major: y outer
    rect = placement.core_range((1, 2, 2, 3))
    assert placement.range_rect(rect) == (1, 2, 2, 3)
    assert [(c.x, c.y) for c in placement.cores_of(rect)] == [(1, 2), (2, 2), (1, 3), (2, 3)]


def test_router_tail_program_parts_take_the_rectangle_and_the_lane_row(expect_error):
    params = inspect.signature(rt.program_parts).parameters
    assert list(params) == ["logits", "index_template", "scores", "indices", "rows", "top_k", "rectangle", "sem_base"]
    assert params["rectangle"].default is None and params["sem_base"].default == 0
    assert rt.RECTANGLE == (4, 5)
    rect = placement.core_range((7, 5, 10, 9))
    cores = rt.lane_cores(rect, 4)
    assert [(c.x, c.y) for c in cores] == [(7, 5), (8, 5), (9, 5), (10, 5)]
    with expect_error(ValueError):
        rt.lane_cores(rect, 5)
    source = inspect.getsource(rt.router_tail_program)
    assert "program_parts(" in source and "rectangle=rectangle" in source
    prepare = inspect.getsource(rt.router_tail_prepare)
    assert "placement.free_rectangle(mesh, *RECTANGLE)" in prepare  # the placement's device reads before any capture


def test_down_kernel_argument_layout_matches_the_python_side():
    args = re.findall(r"get_compile_time_arg_val\((\d+)\)", DOWN_KERNEL)
    assert sorted(int(a) for a in args) == list(range(8))
    source = inspect.getsource(md.moe_dense_program)
    assert "[K_TILES, DOWN_BLK, CB_IN0, CB_IN1, CB_OUT, DOWN_SPILL, CB_INTERM, N_PER_WORKER]" in source
    assert md.K_TILES == 5 and md.N_TILES == 80 and md.DOWN_WORKERS * md.N_PER_WORKER == md.N_TILES
    assert md.DOWN_BLK == 1 and md.K_TILES % md.DOWN_BLK == 0


def test_down_kernel_is_the_gr_read_down_kernel_with_an_output_tile_loop():
    """The spill/reload sequence (pack the fp32 partial, reload it through SrcA, resume the matmul) is gr_read's,
    verbatim; the loop reconfigures the packer to the intermediate format before each tile."""

    def block(text):
        start = text.index("if constexpr (spill > 0) {\n")
        start = text.index("const uint32_t next = k + blk;", start)
        end = text.index("matmul_block_init(c_in0, c_in1, 0, 1, 1, 1);\n", start)
        return re.sub(r"\s+", " ", text[start:end]).strip()

    assert block(DOWN_KERNEL) == block(GR_DOWN_KERNEL)
    assert "for (uint32_t n = 0; n < n_tiles; ++n) {" in DOWN_KERNEL
    assert "pack_reconfig_data_format(c_interm);" in re.sub(r"\s+", "", DOWN_KERNEL)  # before each tile's spills
    assert DOWN_KERNEL.index("in0.wait_front(Kt);") < DOWN_KERNEL.index("for (uint32_t n = 0")
    assert DOWN_KERNEL.index("in0.pop_front(Kt);") > DOWN_KERNEL.rindex("out.push_back(1);")
    for call in (
        "compute_kernel_hw_startup<SrcOrder::Reverse>(c_in0, c_in1, c_interm);",
        "matmul_block(c_in0, c_in1, k + kk, kk, 0, 0, 1, 1, 1);",
        "pack_reconfig_data_format(c_out);",
        "copy_tile(c_interm, 0, 0);",
    ):
        assert call in DOWN_KERNEL and call in GR_DOWN_KERNEL


def test_down_spill_is_the_dram_sharded_down_linears_k_block():
    """The stock down linear (K = 160 over five storage cores) runs in0_block_w = 1: one K tile per block, so its
    fp32 partial spills and reloads after every K tile; the composite's kernel does the same (spill = 1)."""

    mesh = SimpleNamespace(dram_grid_size=lambda: SimpleNamespace(x=8, y=1))
    _, config = dram_sharded_matmul_configs(mesh, se.LOCAL_INTERMEDIATE, md.HIDDEN, num_cores=se.STORAGE_CORES)
    assert config.in0_block_w == md.DOWN_SPILL == 1
    assert config.per_core_N == 16  # the partial's five-core width shard: 16 tiles per storage core


def test_composite_cb_and_semaphore_tables():
    eltwise_indices = {index for _n, index, _d, _p in se.CBS}
    assert md.CB_IN0 not in eltwise_indices  # declared on the storage cores as the multicast source view too
    assert len({md.CB_IN0, md.CB_IN1, md.CB_INTERM, md.CB_OUT}) == 4
    assert not {md.CB_IN1, md.CB_INTERM, md.CB_OUT} & eltwise_indices
    assert md.UNTILIZE_CB == 0  # the stock interleaved reader and writer_rows.cpp pin cb 0
    assert md.SEM_DOWN == md.TOPK_SEMAPHORES == 5  # ids 0..4 stay the top-k's (its exact form's five)
    source = inspect.getsource(md.moe_dense_program)
    assert "sem_base=0" in source and "fp.semaphore_descriptor(SEM_DOWN, all_set)" in source
    assert "senders=se.STORAGE_CORES" in source and "recv_tiles=K_TILES" in source
    assert 'src_cb=se.CB_INDEX["cb_inter"]' in source and "dst_cb=CB_IN0" in source
    assert 'extra=(sigmoid, se.CB_INDEX["cb_sig_bcast"])' in source
    # the eltwise kernels are shared_expert's by path, the untilize kernels untilize_rows', the top-k router_tail's
    assert 'se.KERNELS["reader"]' in source and 'se.KERNELS["compute"]' in source
    assert "ur.READER" in source and "ur.WRITER" in source and "rt.program_parts(" in source


def test_model_switch_sites():
    init = MOE[MOE.index("    def __init__(") : MOE.index("    def release_owned_buffers(")]
    assert (
        "self.moe_dense_fused = (\n            self.routing_in_l1\n            and self.shared_expert_fused\n"
        '            and fused.resolve("moe_dense") is fused.kernel("moe_dense").fused\n        )' in init
    )
    assert "fused.moe_dense.plan_for(mesh_device)" in init
    assert "    moe_dense_fused = False\n" in MOE  # the class default: fakes and the composed chains take the old path
    composite = MOE[MOE.index("    def _dense_composite(") : MOE.index("    def _check_routing_shards(")]
    assert "ttnn.to_memory_config(logits_ws, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)" in composite
    assert (
        "fused.moe_dense.moe_dense(" in composite and "partial_memory_config=self.hidden_act_memory_config" in composite
    )
    assert "compute_kernel_config=self.shared_compute_config" in composite
    assert composite.index("self.weights.shared_gate_up_scalar") < composite.index("fused.moe_dense.moe_dense(")
    forward = MOE[MOE.index("    def forward(") :]
    assert "routing, shared, sparse_rows = self._dense_composite(" in forward
    assert (
        'sparse_rows=temporaries["sparse_rows"],' in re.sub(r"\s+", "", forward) and 'release("sparse_rows")' in forward
    )
    assert (
        '"shared_partial",\n                "sparse_rows",\n                "hidden_tiles",' in forward
    )  # the failure cleanup
    assert "self.moe_dense_fused = False" in init  # a mesh the composite cannot be placed on keeps the four programs
    local_sum = MOE[MOE.index("    def _routed_local_sum(") : MOE.index("    def _dense_composite(")]
    assert "if sparse_rows is None:\n            sparse_input = ttnn.to_layout(" in local_sum
    assert (
        "if sparse_rows is None or not same_buffer(sparse_input, sparse_rows):" in local_sum
    )  # a view's buffer is the caller's
    stamp = inspect.getsource(md.moe_dense)
    assert (
        "for tensor in (scores, indices, partial, sigmoid, sparse_rows):\n        fp.stamp_topology(tensor, full_hidden)"
        in stamp
    )
    assert '"fp32_dest_acc_en": True' in inspect.getsource(
        md._fidelity_of
    ) and '"packer_l1_acc": False' in inspect.getsource(md._fidelity_of)
