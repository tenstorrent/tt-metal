# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The QSA projection merge without a device.  The served fused decode path runs one token's five K = 2560 linears
(index query, index key, query/gate, K, V) as one DRAM-sharded linear against a merged weight, and the fused tails read
their column windows of the one L1 shard by first tile.  Pinned here: the column layout and its first tiles against
the tail programs' widths, the merged weight's host assembly (device d's block is the five device blocks side by side),
the one-reader program config (15 output tiles per storage core, in0_block_w 5, eight cores, no padding), the window
check, and the source of the served path: one linear before the index tail, both tails and the post-attention on the
shard, the separate linears kept for the composed chain and the prefill chunk path, the readers' offset arguments."""

import inspect
import re
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

TILE = ttnn.TILE_SIZE
KERNELS = fp.REPO_ROOT / fp.KERNEL_ROOT / "qsa_block" / "kernels"


def _mesh(banks: int = 8):
    return SimpleNamespace(dram_grid_size=lambda: ttnn.CoreCoord(banks, 1))


def _flat(source) -> str:
    return re.sub(r"\s+", "", inspect.getsource(source))


class _Row:
    """A [1, 1, rows, width] row tile of whole tiles, as the window check reads it."""

    def __init__(self, width: int) -> None:
        self.shape = (1, 1, 1, width)
        self.padded_shape = (1, 1, TILE, width)


def test_the_merged_columns_are_the_five_projections_in_order_with_their_first_tiles() -> None:
    assert [name for name, _ in qsa.PROJECTION_COLUMNS] == ["index_q", "index_k", "qg", "k", "v"]
    widths = dict(qsa.PROJECTION_COLUMNS)
    assert widths == {
        "index_q": qsa_block.INDEX_HEAD_DIM,
        "index_k": qsa_block.INDEX_HEAD_DIM,
        "qg": qsa_block.QG_WIDTH,
        "k": qsa_block.HEAD_DIM,
        "v": qsa_block.HEAD_DIM,
    }
    assert widths == {"index_q": 128, "index_k": 128, "qg": 3072, "k": 256, "v": 256}
    assert qsa.PROJECTIONS_WIDTH == sum(widths.values()) == 3840
    assert qsa.PROJECTION_FIRST_TILE == {"index_q": 0, "index_k": 4, "qg": 8, "k": 104, "v": 112}
    # every window starts on a tile and the windows tile the shard without gaps
    edge = 0
    for name, width in qsa.PROJECTION_COLUMNS:
        assert width % TILE == 0 and qsa.PROJECTION_FIRST_TILE[name] * TILE == edge
        edge += width
    assert edge == qsa.PROJECTIONS_WIDTH
    # the tail programs' tile counts the windows feed: 4 index tiles, 8 per head half, 6 heads of 16 in qg
    assert qsa_block.INDEX_TILES == 4 and qsa_block.HEAD_TILES == 8
    assert 2 * qsa_block.HEAD_TILES * qsa_block.LOCAL_HEADS == widths["qg"] // TILE == 96


def test_the_merged_program_config_is_one_reader_with_fifteen_tiles_per_core(expect_error) -> None:
    mesh = _mesh()
    activation, config = dm.dram_sharded_matmul_configs(mesh, qsa.HIDDEN_SIZE, qsa.PROJECTIONS_WIDTH, num_cores=8)
    assert (config.per_core_M, config.per_core_N, config.in0_block_w) == (1, 15, 5)
    assert config.num_workers_per_dram_bank == 1 and config.fused_activation is None
    # no padding: eight banks x 15 tiles x 32 columns is the width, the weight's bank shard [2560, 480]
    assert dm.bank_tiles(mesh, qsa.HIDDEN_SIZE, qsa.PROJECTIONS_WIDTH) == 15
    weight = dm.dram_sharded_weight_memory_config(mesh, qsa.HIDDEN_SIZE, qsa.PROJECTIONS_WIDTH)
    assert tuple(weight.shard_spec.shape) == (qsa.HIDDEN_SIZE, 480)
    assert weight.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and weight.buffer_type == ttnn.BufferType.DRAM
    # the activation shard and the K blocking are the separate K = 2560 linears'
    assert tuple(activation.shard_spec.shape) == (TILE, qsa.HIDDEN_SIZE // 8)
    for n in (2 * qsa.LOCAL_QUERY_WIDTH, qsa.HEAD_DIM, qsa.INDEX_HEAD_DIM):
        separate_activation, separate = dm.dram_sharded_matmul_configs(mesh, qsa.HIDDEN_SIZE, n, num_cores=8)
        assert separate_activation == activation and separate.in0_block_w == config.in0_block_w
    # the merged width is not in the two-reader table
    with expect_error(ValueError, match="not qualified"):
        dm.bank_tiles(mesh, qsa.HIDDEN_SIZE, qsa.PROJECTIONS_WIDTH, 2)
    assert (qsa.HIDDEN_SIZE, qsa.PROJECTIONS_WIDTH) not in dm.TWO_WORKER_PROJECTIONS


def test_the_merged_host_weight_is_the_five_device_blocks_side_by_side(expect_error) -> None:
    g = torch.Generator().manual_seed(3840)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    qg = rand(2 * qsa.QUERY_WIDTH, qsa.HIDDEN_SIZE)
    k = rand(qsa.KV_HEADS * qsa.HEAD_DIM, qsa.HIDDEN_SIZE)
    v = rand(qsa.KV_HEADS * qsa.HEAD_DIM, qsa.HIDDEN_SIZE)
    index_q = rand(qsa.INDEX_QUERY_HEADS * qsa.INDEX_HEAD_DIM, qsa.HIDDEN_SIZE)
    index_k = rand(qsa.INDEX_HEAD_DIM, qsa.HIDDEN_SIZE)
    merged = qsa._merged_projections(qg, k, v, index_q, index_k)
    assert tuple(merged.shape) == (1, 1, qsa.HIDDEN_SIZE, qsa.TP_SIZE * qsa.PROJECTIONS_WIDTH)
    assert merged.dtype == torch.bfloat16
    # the separate weights' host tensors as from_checkpoint uploads them (dim 3 sharded over the four devices)
    separate = {
        "index_q": index_q.transpose(0, 1),
        "index_k": index_k.transpose(0, 1),
        "qg": qg.transpose(0, 1),
        "k": qsa._expanded_pair_kv(k).transpose(0, 1),
        "v": qsa._expanded_pair_kv(v).transpose(0, 1),
    }
    width = qsa.PROJECTIONS_WIDTH
    blocks = [merged[0, 0, :, d * width : (d + 1) * width] for d in range(qsa.TP_SIZE)]
    for d, block in enumerate(blocks):
        for name, columns in qsa.PROJECTION_COLUMNS:
            start = qsa.PROJECTION_FIRST_TILE[name] * TILE
            window = block[:, start : start + columns]
            if name == "index_k":
                expected = separate[name]  # replicated: whole on every device
            else:
                expected = separate[name][:, d * columns : (d + 1) * columns]
            assert torch.equal(window, expected), (d, name)
    # the K/V pair grouping survives the merge: devices 0/1 hold KV head 0, devices 2/3 head 1
    for name in ("k", "v"):
        start = qsa.PROJECTION_FIRST_TILE[name] * TILE
        stop = start + qsa.HEAD_DIM
        assert torch.equal(blocks[0][:, start:stop], blocks[1][:, start:stop])
        assert torch.equal(blocks[2][:, start:stop], blocks[3][:, start:stop])
        assert not torch.equal(blocks[0][:, start:stop], blocks[2][:, start:stop])
    with expect_error(ValueError, match="merged projection block index_k"):
        qsa._merged_projections(qg, k, v, index_q, index_k[: qsa.INDEX_HEAD_DIM // 2])


def test_the_window_check_admits_the_separate_shard_and_the_merged_windows_only(expect_error) -> None:
    merged = _Row(qsa.PROJECTIONS_WIDTH)
    for name, width in qsa.PROJECTION_COLUMNS:
        qsa_block._window(_Row(width), 0, width, name)  # the separate projection shard
        qsa_block._window(merged, qsa.PROJECTION_FIRST_TILE[name], width, name)  # its window of the merged shard
    with expect_error(ValueError, match="outside"):
        qsa_block._window(_Row(qsa_block.QG_WIDTH), qsa.PROJECTION_FIRST_TILE["qg"], qsa_block.QG_WIDTH, "qg")
    with expect_error(ValueError, match="outside"):
        qsa_block._window(merged, qsa.PROJECTION_FIRST_TILE["v"] + 1, qsa_block.HEAD_DIM, "v")
    with expect_error(ValueError, match="outside"):
        qsa_block._window(merged, -1, qsa_block.HEAD_DIM, "v")
    with expect_error(ValueError):
        qsa_block._window(_Row(100), 0, 100, "ragged")
    one, two = _Row(1), _Row(2)
    assert qsa_block._io(one, two, one, two) == [one, two]


def test_the_weights_carry_the_merged_tensor_beside_the_five() -> None:
    fields = qsa.Qwen38TTNNQSAWeights.__dataclass_fields__
    assert list(fields)[-1] == "projections" and fields["projections"].default is None
    loader = _flat(qsa.Qwen38TTNNQSAWeights.from_checkpoint)
    assert (
        'projections_tt=upload(_merged_projections(qg,k,v,index_q,index_k),"qsa_proj_dram_sharded",output_mapper,'
        "dram_sharded_weight_memory_config(mesh_device,HIDDEN_SIZE,PROJECTIONS_WIDTH),dtype=weight_dtype,)"
    ) in loader
    for name in (
        "qg_dram_sharded",
        "k_pair_grouped_dram_sharded",
        "v_pair_grouped_dram_sharded",
        "index_q_dram_sharded",
        "index_k_dram_sharded",
    ):
        assert f'"{name}"' in loader  # the five separate uploads stay for the chain and the prefill chunk path
    assert "mesh_contract.validate_tensor(projections_tt,placement=TensorPlacement.HEAD_SHARDED,shard_dim=3)" in loader
    assert "projections=projections_tt," in loader
    validate = _flat(qsa.Qwen38TTNNQSAWeights.validate)
    assert "ifself.projectionsisnotNone:" in validate
    assert (
        '("projections",self.projections,TensorPlacement.HEAD_SHARDED,3,(1,1,HIDDEN_SIZE,PROJECTIONS_WIDTH),)'
        in validate
    )
    assert "self.projections," in _flat(qsa.Qwen38TTNNQSAWeights.deallocate)


def test_the_served_path_runs_one_linear_and_the_tails_read_its_windows() -> None:
    forward = _flat(qsa.Qwen38TTNNQSA._forward_decode_generic_fused)
    assert (
        "full_hidden=self._all_gather_hidden(hidden_sharded)merged=self._project_merged(full_hidden)"
        "ifself.merged_projectionselseNone"
    ) in forward
    assert forward.count("_project_merged(") == 1
    assert (
        "self._index_tail_step(full_hidden,cos,sin,block_start_cos,block_start_sin,state,position,projections=merged)"
    ) in forward
    assert "self._main_tail_step(full_hidden,cos,sin,state,position,projections=merged)" in forward
    assert (
        "self._sparse_value_attention_fused(sparse_query,qg_ws,sparse_indices,state,"
        'qg_first=0ifmergedisNoneelsePROJECTION_FIRST_TILE["qg"],)'
    ) in forward
    project = inspect.getsource(qsa.Qwen38TTNNQSA._project_merged)
    assert project.count("ttnn.linear(") == 1 and "self.weights.projections," in project
    assert (
        "program_config=self.proj_program_config" in project
        and "compute_kernel_config=self.projection_compute_config" in project
    )
    assert "memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG" in project
    # the tails: the linears only without the shard, the windows with it
    index_step = _flat(qsa.Qwen38TTNNQSA._index_tail_step)
    assert "ifprojectionsisNone:index_q_ws=ttnn.linear(" in index_step and index_step.count("ttnn.linear(") == 2
    assert (
        "else:index_q_ws=raw_key_ws=projectionsindex_q_first,index_k_first="
        'PROJECTION_FIRST_TILE["index_q"],PROJECTION_FIRST_TILE["index_k"]'
    ) in index_step
    assert "index_q_first=index_q_first,index_k_first=index_k_first,)" in index_step
    assert "ifprojectionsisNone:_deallocate(index_q_ws,raw_key_ws)" in index_step
    main_step = _flat(qsa.Qwen38TTNNQSA._main_tail_step)
    assert "ifprojectionsisNone:qg_ws=ttnn.linear(" in main_step and main_step.count("ttnn.linear(") == 3
    assert (
        'else:qg_ws=k_ws=v_ws=projectionsqg_first,k_first,v_first=(PROJECTION_FIRST_TILE[name]fornamein("qg","k","v"))'
    ) in main_step
    assert "qg_first=qg_first,k_first=k_first,v_first=v_first,)" in main_step
    assert "ifprojectionsisNone:_deallocate(k_ws,v_ws)" in main_step and main_step.endswith("returnsparse_query,qg_ws")
    attention = _flat(qsa.Qwen38TTNNQSA._sparse_value_attention_fused)
    assert (
        "self._post_attention_fused(sparse_output,qg_ws,memory_config=self.out_act_memory_config,qg_first=qg_first)"
    ) in attention
    assert "_deallocate(sparse_output,qg_ws)" in attention  # the last consumer releases the shard
    # the composed chain and the prefill chunk path keep the separate linears
    assert _flat(qsa.Qwen38TTNNQSA._index_projection).count("ttnn.linear(") == 2
    assert _flat(qsa.Qwen38TTNNQSA._main_projection).count("ttnn.linear(") == 3
    rows = _flat(qsa.Qwen38TTNNQSA._index_projection_rows) + _flat(qsa.Qwen38TTNNQSA._main_projection_rows)
    assert rows.count("self._linear_rows(") == 5 and "projections" not in rows
    init = _flat(qsa.Qwen38TTNNQSA.__init__)
    assert (
        "self.merged_projections=(self._index_tail_fusedisnotNoneandself._main_tail_fusedisnotNone"
        "andweights.projectionsisnotNone)"
    ) in init
    assert (
        "_,self.proj_program_config=dram_sharded_matmul_configs(mesh_device,HIDDEN_SIZE,PROJECTIONS_WIDTH,num_cores=8)"
        in init
    )
    assert (
        "ifweights.projectionsisnotNone:validate_dram_sharded_weight(weights.projections,mesh_device,HIDDEN_SIZE,"
        "PROJECTIONS_WIDTH,num_workers_per_dram_bank=1)"
    ) in init


def test_the_tail_programs_take_the_windows_first_tiles_and_the_readers_offset_their_pages() -> None:
    for function, names in (
        (qsa_block.index_tail, ("index_q_first", "index_k_first")),
        (qsa_block.main_tail, ("qg_first", "k_first", "v_first")),
        (qsa_block.post_attention, ("qg_first",)),
    ):
        parameters = inspect.signature(function).parameters
        for name in names:
            parameter = parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default == 0, (
                function.__name__,
                name,
            )
    index_tail = _flat(qsa_block.index_tail)
    assert '_window(index_q_ws,index_q_first,INDEX_HEAD_DIM,"indexquery")' in index_tail
    assert '_window(raw_key_ws,index_k_first,INDEX_HEAD_DIM,"rawkey")' in index_tail
    # the window firsts, then the lanes' lane_first / lane_count / do_query (one lane, the query on core 0)
    assert "row_hit.buffer_address(),rows,index_q_first,index_k_first,u,1,int(u==0),]" in index_tail
    assert "io=_io(index_q_ws,raw_key_ws," in index_tail
    main_tail = _flat(qsa_block.main_tail)
    assert "(c,[qg_ws.buffer_address(),q_norm.buffer_address(),qg_first+2*HEAD_TILES*h])" in main_tail
    assert "[(k_norm_core,[k_ws.buffer_address(),k_norm.buffer_address(),k_first])]" in main_tail
    # v_first, then the lanes' lane_first / lane_count
    assert "row_hit.buffer_address(),rows,v_first,u,1,]" in main_tail
    assert "io=_io(qg_ws,k_ws,v_ws," in main_tail
    post = _flat(qsa_block.post_attention)
    assert '_window(qg_ws,qg_first,QG_WIDTH,"qg")' in post
    assert "[(c,[attention.buffer_address(),qg_ws.buffer_address(),rows,h,qg_first])" in post
    # the readers add the first tile to the page index; everything after the read is untouched
    index_reader = (KERNELS / "index_tail_reader_norm.cpp").read_text()
    assert "const uint32_t q_first = get_arg_val<uint32_t>(8);" in index_reader
    assert "const uint32_t raw_first = get_arg_val<uint32_t>(9);" in index_reader
    assert "noc_async_read_page(q_first + c, q, x_l1 + c * TILE_BYTES);" in index_reader
    assert "noc_async_read_page(raw_first + c, raw, raw_l1 + c * TILE_BYTES);" in index_reader
    norm_reader = (KERNELS / "main_tail_reader_norm.cpp").read_text()
    assert "const uint32_t first = get_arg_val<uint32_t>(2);" in norm_reader
    assert "noc_async_read_page(first + c, x, get_write_ptr(CB_X) + c * TILE_BYTES);" in norm_reader
    staging_reader = (KERNELS / "main_tail_reader_staging.cpp").read_text()
    assert "const uint32_t v_first = get_arg_val<uint32_t>(5);" in staging_reader
    assert "noc_async_read_page(v_first + c, v, pack_l1 + c * TILE_BYTES);" in staging_reader
    post_reader = (KERNELS / "post_attention_reader.cpp").read_text()
    assert "const uint32_t qg_first = get_arg_val<uint32_t>(4);" in post_reader
    assert (
        "noc_async_read_page(qg_first + 2 * HEAD_TILES * head + GATE_FIRST + t, qg, gate_l1 + t * TILE_BYTES);"
        in post_reader
    )
    # the merged windows land on the pages the readers address: qg head h's gate at 8 + 16 h + 8, k at 104, v at 112
    first = qsa.PROJECTION_FIRST_TILE
    assert [
        first["qg"] + 2 * qsa_block.HEAD_TILES * h + qsa_block.HEAD_TILES for h in range(qsa_block.LOCAL_HEADS)
    ] == [16, 32, 48, 64, 80, 96]
    assert first["qg"] + 2 * qsa_block.HEAD_TILES * qsa_block.LOCAL_HEADS == first["k"] == 104
    assert first["k"] + qsa_block.HEAD_TILES == first["v"] == 112 and first["v"] + qsa_block.HEAD_TILES == 120
    assert 120 * TILE == qsa.PROJECTIONS_WIDTH
