# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused MoE glue (``moe_post``) and shared expert (``shared_expert``) without a device: registry entries, the
CB / argument contracts between the Python side and the kernels, the exactness pins in the compute kernels (the
replaced ops' instruction sequences), the tile-face addressing of the reader, the owner rows and the weight concat,
and the model's switch sites in ``ttnn/moe.py`` (read as text: the module needs the runtime)."""

from __future__ import annotations

import inspect
import re

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_post as mp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail as rt
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import shared_expert as se

POST = {name: (fp.REPO_ROOT / path).read_text() for name, path in mp.KERNELS.items()}
SHARED = {name: (fp.REPO_ROOT / path).read_text() for name, path in se.KERNELS.items()}
MOE = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/moe.py").read_text()


def _named(source: str) -> set[str]:
    return set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', source))


def _runtime_indices(source: str) -> list[int]:
    return [int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", source)]


def test_registered_bitwise_with_gates():
    for name, module, fused_fn, composed_fn in (
        ("moe_post", mp, mp.moe_post, mp.moe_post_composed),
        ("shared_expert", se, se.shared_expert, se.shared_expert_composed),
    ):
        entry = fused.kernel(name)
        assert entry.tolerance == fused.BITWISE
        assert entry.fused is fused_fn and entry.composed is composed_fn
        assert entry.gate is not None and entry.gate.layers == tuple(range(48)) and entry.gate.reference is None
        assert fused.resolve(name, {}) is (fused_fn if name in fused.DEFAULT_ON else composed_fn)  # the list decides
        assert fused.resolve(name, {fused.OFF_ENV: name}) is composed_fn
        assert fused.resolve(name, {fused.ENV: name}) is fused_fn
    assert inspect.signature(mp.moe_post).parameters.keys() == inspect.signature(mp.moe_post_composed).parameters.keys()


def test_post_cb_table_and_named_args():
    indices = [index for _name, index, _dtype, _bytes, _pages in mp.CBS]
    assert len(set(indices)) == len(indices) and max(indices) < 32 and mp.CB_INDEX["cb_out"] == 16
    for name in ("cb_act", "cb_scores"):
        assert dict((n, p) for n, _i, _d, _b, p in mp.CBS)[name] == mp.TOP_K
    extra = {"top_k", "has_sig", "stage_pages", "owner_bytes", "route_contiguous"}
    for kernel in ("reader", "compute", "writer"):
        names = _named(POST[kernel])
        assert names, kernel
        assert names <= set(mp.CB_INDEX) | extra, names - set(mp.CB_INDEX)
    assert _runtime_indices(POST["reader"]) == list(range(len(mp.READER_ARGS)))
    assert _runtime_indices(POST["writer"]) == list(range(len(mp.WRITER_ARGS)))
    assert POST["reader"].count("next_compile_time_args_offset()") == 5  # six accessors chained
    program = inspect.getsource(mp.moe_post_program)
    for tensor in ("pages", "scores", "indices", "owner", "shared", "sig"):
        assert f"fp.accessor_args({tensor})" in program
    assert "sig = shared if sigmoid is None else sigmoid" in program


def test_post_compute_pins_the_replaced_ops_instruction_sequences():
    compute = POST["compute"]
    # deepseek_moe_fast_reduce_nc_fused_compute.cpp: COL-broadcast ELWMUL MAC into DST 0, acc_to_dest, slots in order
    assert "bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_act, cb_scores)" in compute
    assert "llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>" in compute
    assert "1 /*acc_to_dest*/" in compute
    assert re.search(
        r"for \(uint32_t k = 0; k < top_k; \+\+k\) \{\s*mul_tiles_bcast_cols\(cb_act, cb_scores, k, k, 0\);", compute
    )
    # binary_ng: the SFPU multiply in the 16-bit-dest rounding (ttnn.multiply defaults fast_and_approximate_mode
    # False); the FPU add_tiles (ttnn.add defaults True) -- never an SFPU add
    assert "mul_binary_tile<false>(0, 1, 0)" in compute
    assert "binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_routed, cb_rhs)" in compute
    assert "add_tiles(cb_routed, cb_rhs, 0, 0, 0)" in compute and "add_binary_tile" not in compute
    mac, mul, add = (
        compute.index(call)
        for call in ("mul_tiles_bcast_cols(cb_act", "mul_binary_tile<false>(0, 1, 0)", "add_tiles(cb_routed")
    )
    assert mac < mul < add
    program = inspect.getsource(mp.moe_post_program)
    assert "fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True" in program


def test_post_reader_face_addressing_and_score_tiles():
    reader = POST["reader"]
    # the chain's column-0 score tile: face 0 rows 0..15 at j*16, face 2 rows 16..31 at 512 + (j-16)*16
    assert "(j < 16) ? j * 16 : 512 + (j - 16) * 16" in reader
    assert "stile[k * 1024 + col0] = sc_u16[j * route_words + k]" in reader
    assert "owner_u16[idx_u16[j * route_words + k]] != 0" in reader and "route_pitch = 64" in reader
    # the score tiles are zero-seeded like the activation pages (only owned scores are written); routing rows that are
    # one L1 shard on one core come in one read per tensor at the shard's page pitch
    assert "noc.async_write_zeros(CoreLocalMem<uint32_t>(stile_base), top_k * tile_bytes, {})" in reader
    assert "route_words = indices.get_aligned_page_size() / 2" in reader
    assert "noc_async_read(indices.get_noc_addr(0), idx_base, rows * 2 * route_words)" in reader
    assert "noc_async_read(scores.get_noc_addr(0), sc_base, rows * 2 * route_words)" in reader
    # a token row's 32 columns split into faces (j/16)*2 (columns 0..15) and +1 (columns 16..31), row j%16
    assert "(j >> 4) * 2 * face_bytes + (j & 15) * face_row_bytes" in reader
    assert "dst + face_bytes, face_row_bytes" in reader
    assert ".page_id = k * rows + j, .offset_bytes = tile_col * fragment_bytes" in reader
    assert "& ~(fragment_bytes - 1)" in reader  # 64-byte alignment of the staging area
    assert "noc.async_write_zeros(CoreLocalMem<uint32_t>(act_base), top_k * tile_bytes, {})" in reader
    assert reader.index("async_write_zeros") < reader.index("noc.async_read_barrier();  // the zero pages")


def test_owner_rows_are_the_four_disjoint_expert_shards():
    rows = mp.owner_rows()
    assert rows.shape == (4, 512) and rows.dtype == torch.int16
    assert torch.equal(rows.sum(dim=0), torch.ones(512, dtype=torch.int16))
    for d in range(4):
        assert rows[d, 128 * d : 128 * (d + 1)].all() and rows[d].sum() == 128


def test_concat_shared_weights_layout():
    k, local = 64, 16
    gate = torch.arange(k * 64, dtype=torch.float32).reshape(k, 64)
    up = -gate
    scalar = torch.full((k, 1), 7.0)
    cat = se.concat_shared_weights(gate, up, scalar)
    width = 2 * local + fp.TILE
    assert cat.shape == (k, 4 * width)
    for d in range(4):
        block = cat[:, d * width : (d + 1) * width]
        assert torch.equal(block[:, :local], gate[:, d * local : (d + 1) * local])
        assert torch.equal(block[:, local : 2 * local], up[:, d * local : (d + 1) * local])
        assert torch.equal(block[:, 2 * local], scalar[:, 0]) and not block[:, 2 * local + 1 :].any()
    assert se.CAT_WIDTH == 352 and se.SCALAR_COLUMN == 320 and se.GATE_TILES == 5


def test_shared_eltwise_contract_and_pins():
    for kernel in ("reader", "compute", "writer"):
        names = _named(SHARED[kernel])
        assert names <= set(se.CB_INDEX) | {"gate_tiles"}, names - set(se.CB_INDEX)
    assert _runtime_indices(SHARED["reader"]) == list(range(len(se.READER_ARGS)))
    assert _runtime_indices(SHARED["writer"]) == list(range(len(se.WRITER_ARGS)))
    compute = SHARED["compute"]
    # unary eltwise_sfpu.cpp: copy_tile then the op chain inside the register window; binary_ng SFPU multiply
    assert "copy_tile(cb_gate, 0, 0);\n    silu_tile_init();\n    silu_tile(0);" in compute
    assert "mul_binary_tile(0, 1, 0)" in compute and "mul_binary_tile<" not in compute
    assert "sigmoid_tile_init<false>();\n        sigmoid_tile<VectorMode::RC, false>(0);" in compute
    assert (
        "unary_bcast_init<BroadcastType::COL>(cb_sig)" in compute
        and "unary_bcast<BroadcastType::COL>(cb_sig, 0, 0)" in compute
    )
    assert "fp32_dest=False" in inspect.getsource(se.shared_eltwise_program)
    assert "{.page_id = 2 * gate_tiles}" in SHARED["reader"] and "{.page_id = gate_tiles + tile}" in SHARED["reader"]


def test_router_tail_accepts_the_bf16_shard_and_preallocated_outputs():
    assert "fp32 or bf16 TILE" in inspect.getsource(rt._rows_of)
    assert 'logits.dtype if name == "cb_in0" else dtype' in inspect.getsource(rt.program_parts)
    assert inspect.signature(rt.router_tail_into).parameters.keys() == {"logits", "scores", "indices", "top_k"}
    assert "return router_tail_into(logits, scores, indices, top_k=top_k)" in inspect.getsource(rt.router_tail)


def test_model_switch_sites():
    init = MOE[MOE.index("    def __init__(") : MOE.index("    def release_owned_buffers(")]
    assert 'self.moe_post_fused = one_tile and fused.resolve("moe_post") is fused.kernel("moe_post").fused' in init
    assert (
        'self.shared_expert_fused = one_tile and fused.resolve("shared_expert") is fused.kernel("shared_expert").fused'
        in init
    )
    assert 'self.routing_in_l1 = self.moe_post_fused and self._route_tail is fused.kernel("router_tail").fused' in init
    assert "mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 0))" in init
    route = MOE[MOE.index("    def _route(") : MOE.index("    def _routed_partial(")]
    assert "if self.routing_in_l1:\n                return self._route_into_shards(logits_ws, phase_observer)" in route
    assert route.index("if self.routing_in_l1:") < route.index(
        "ttnn.to_memory_config(logits_ws, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)"
    )
    into = MOE[MOE.index("    def _route_into_shards(") : MOE.index("    def _routing_rows(")]
    assert "fused.router_tail.router_tail_into(logits, scores_l1, indices_l1, top_k=TOP_K)" in into
    assert "ttnn.to_memory_config(logits_ws, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)" in into  # the proven drain
    local_sum = MOE[MOE.index("    def _routed_local_sum(") : MOE.index("    def _check_routing_shards(")]
    assert (
        "ttnn.fill(" not in local_sum
        and "deepseek_moe_fast_reduce_nc_fused" not in local_sum
        and "to_layout(" in local_sum
    )
    # the composed routed partial is untouched (its AST is pinned by the layer-0 diagnostics)
    routed = MOE[MOE.index("    def _routed_partial(") : MOE.index("    def _routed_local_sum(")]
    assert "outputs = ttnn.experimental.moe_compute(" in routed and "ttnn.fill(self.local_combine_output" in routed
    assert (
        "fused.moe_post.moe_post(\n            outputs[5],\n            routing.scores,\n            routing.indices,\n            self.expert_owner,"
        in local_sum
    )
    shared = MOE[MOE.index("    def _shared_partial(") : MOE.index("    def forward(")]
    assert "if self.shared_expert_fused:" in shared and "return Qwen38TTNNSharedPartial(partial, sigmoid)" in shared
    forward = MOE[MOE.index("    def forward(") :]
    assert 'temporaries["local_sum"] = self._routed_local_sum(' in forward
    assert forward.count('self._synchronize_stage("routed-partial")') == 1  # the six fences stay six
    assert (
        forward.index("self._routed_local_sum(")
        < forward.index("self._routed_partial(")
        < forward.index("tt_all_reduce(")
    )
    loader = MOE[MOE.index("    def from_checkpoint(") : MOE.index("@dataclass(frozen=True)\nclass Qwen38TTNNRouting")]
    assert '"shared_gate_up_scalar_dram_sharded"' in loader and "fused.shared_expert.concat_shared_weights(" in loader
    assert MOE.count("class Qwen38TTNNSharedPartial") == 1


@pytest.mark.parametrize("bad", [(9, 1, 2560), (10, 1, 2048)])
def test_post_rejects_wrong_page_shapes(bad, expect_error):
    class T:
        def __init__(self, shape, dtype, layout):
            self.shape, self.dtype, self.layout = shape, dtype, layout

    scores = T((1, 1, 1, 10), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    indices = T((1, 1, 1, 10), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
    assert mp.routing_rows_of(scores, indices) == 1
    with expect_error(ValueError, "moe_post pages must be"):
        mp._check_inputs(T(bad, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), scores, indices, None, None, None)
