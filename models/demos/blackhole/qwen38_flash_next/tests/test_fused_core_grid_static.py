# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every fused program's cores lie inside the device's compute grid, for every supported context and both p150 grids.

A generic_op kernel placed outside ``compute_with_storage_grid_size()`` lands on a dispatch core and the program is
refused at launch ("Illegal kernel placement ... Kernels cannot be placed on dispatch cores").  The QSA score merge
sized its grid from the context (one core per 1024 score columns along row 0): 8 cores at 32768 tokens, 16 at 65536,
past both p150 compute grids.  This test walks the supported contexts and the grids the model opens: the harvested
p150 (120 Tensix as tt-smi reports it, 11 x 10 compute cores after the dispatch column) and the unharvested p150 (130
Tensix, 12 x 10), and asserts, without a device, that the core plan of every fused program whose plan is a pure function
of the grid and the context stays inside it; the programs with a fixed layout are checked against the smaller grid.
"""

from __future__ import annotations

from types import SimpleNamespace

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_fold, gr_read, gr_write, placement
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block, router_tail

SUPPORTED_CONTEXTS = (32768, 65536, 131072, 262144)  # the launchers' admitted --allocated-context values
GRIDS = {"p150 harvested (120 Tensix)": (11, 10), "p150 unharvested (130 Tensix)": (12, 10)}


def _grid(x: int, y: int):
    return SimpleNamespace(x=x, y=y)


def _fake_mesh(x: int, y: int):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: _grid(x, y))


def _inside(cores, grid) -> bool:
    return all(0 <= c.x < grid.x and 0 <= c.y < grid.y for c in cores)


def _range_cores(core_range_set):
    cores = []
    for r in core_range_set.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores


def test_score_merge_stays_inside_the_compute_grid_at_every_supported_context():
    for context in SUPPORTED_CONTEXTS:
        width = context // qsa.COMPRESS_RATIO  # allocated_compressed_blocks
        chunks = width // qsa_block.SCORE_CHUNK
        assert chunks == context // 4096
        for label, (x, y) in GRIDS.items():
            work = qsa_block.score_merge_work(width, _grid(x, y))
            assert _inside([w.core for w in work], _grid(x, y)), (context, label)
            assert sum(w.count for w in work) == chunks and all(w.count >= 1 for w in work)
            assert [w.start for w in work] == [
                sum(v.count for v in work[:i]) for i in range(len(work))
            ]  # contiguous chunk ranges
            assert len(work) == min(chunks, x * y) and (chunks > x * y or all(w.count == 1 for w in work))
            assert _inside(_range_cores(fp.core_rectangle(work, _fake_mesh(x, y))), _grid(x, y))
    # the first form's failure, as a statement: 16 chunks along row 0 leave both grids
    assert 65536 // 4096 == 16 and all(16 > x for x, _ in GRIDS.values())
    # a grid smaller than the chunk count folds several chunks onto one core, still inside
    tiny = _grid(4, 2)
    work = qsa_block.score_merge_work(262144 // qsa.COMPRESS_RATIO, tiny)
    assert (
        _inside([w.core for w in work], tiny) and sum(w.count for w in work) == 64 and max(w.count for w in work) == 8
    )


def test_score_merge_kernels_take_a_chunk_range():
    kernels = fp.REPO_ROOT / fp.KERNEL_ROOT / "qsa_block" / "kernels"
    reader = (kernels / "score_merge_reader.cpp").read_text()
    compute = (kernels / "score_merge_compute.cpp").read_text()
    writer = (kernels / "score_merge_writer.cpp").read_text()
    assert (
        "const uint32_t first_chunk = get_arg_val<uint32_t>(3);" in reader
        and "const uint32_t chunks = get_arg_val<uint32_t>(4);" in reader
    )
    assert "const uint32_t total_chunks = get_arg_val<uint32_t>(5);" in reader
    assert (
        "gathered.get_noc_addr((d * rows + r) * total_chunks + chunk, 0)" in reader
    )  # the 2 KB page of device d, row r, chunk c
    assert "for (uint32_t chunk = first_chunk; chunk < first_chunk + chunks; ++chunk) {" in reader
    assert (
        "const uint32_t chunks = get_arg_val<uint32_t>(1);" in compute
        and "for(uint32_tr=0;r<rows*chunks;++r){" in "".join(compute.split())
    )
    assert (
        "const uint32_t first_chunk = get_arg_val<uint32_t>(2);" in writer
        and "const uint32_t chunks = get_arg_val<uint32_t>(3);" in writer
    )
    assert "for (uint32_t chunk = first_chunk; chunk < first_chunk + chunks; ++chunk) {" in writer
    # the per-element arithmetic is untouched: the same four device tiles added in device order, then the mask
    assert "for (uint32_t d = 0; d < DEVICES; ++d) {\n            add_tiles(CB_IN, CB_ZERO, d, 0, 0);" in compute
    assert "add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);" in compute
    import inspect
    import re

    source = re.sub(r"\s+", "", inspect.getsource(qsa_block.score_merge))
    assert (
        "work=score_merge_work(width,mesh.compute_with_storage_grid_size())" in source
        and "cores=fp.core_rectangle(work,mesh)" in source
    )
    assert (
        "[(w.core,[gathered.buffer_address(),mask.buffer_address(),rows,w.start,w.count,chunks])forwinwork]" in source
    )
    assert (
        "[(w.core,[rows,w.count])forwinwork]" in source
        and "[(w.core,[out.buffer_address(),rows,w.start,w.count])forwinwork]" in source
    )


def test_fixed_layout_fused_programs_fit_the_smaller_grid():
    """The programs whose cores are constants of the module (heads, slices, transports): the extents they declare."""

    small = _grid(*min(GRIDS.values()))
    fixed = {
        "qsa index_tail": [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)],
        "qsa main_tail": _range_cores(qsa_block._rect(0, 0, qsa_block.LOCAL_HEADS + 1, 1)),
        "qsa post_attention": _range_cores(qsa_block._rect(0, 0, qsa_block.LOCAL_HEADS - 1, 0)),
        "qsa widen_partial": _range_cores(qsa_block._rect(0, 0, qsa_block.WIDEN_CORES // 2 - 1, 1)),
        "qsa selection_row": _range_cores(qsa_block._rect(0, 0, qsa_block.SELECTION_SLICES - 1, 0)),
        "gr_read normalize_down (6x2 workers + 4 producers on row 2)": [
            ttnn.CoreCoord(x, y) for x in range(6) for y in range(2)
        ]
        + [ttnn.CoreCoord(x, 2) for x in range(gr_read.BRANCHES)],
        "gr_read low_rank_gate (5x4 workers + 4 producers on row 4)": [
            ttnn.CoreCoord(x, y) for x in range(5) for y in range(4)
        ]
        + [ttnn.CoreCoord(x, 4) for x in range(gr_read.LOW_RANK_CORES)],
        "gr_fold transports": list(gr_fold.TRANSPORT["stats"]) + list(gr_fold.TRANSPORT["partials"]),
        # the top-k asks placement.free_rectangle for its 4 x 5 rectangle (its lane cores = the first row); on the
        # smaller grid with the storage cores reserved it lands in the far corner
        "router_tail lanes": router_tail.lane_cores(
            placement.core_range(
                placement.free_rectangle_in(*min(GRIDS.values()), placement.STORAGE_CORES, *router_tail.RECTANGLE)
            ),
            router_tail.PASSES,
        ),
    }
    for name, cores in fixed.items():
        assert _inside(cores, small), name


def test_grid_split_programs_follow_the_compute_grid():
    """Programs that split their work with fp.split_work take the grid from the mesh: at most grid.x * grid.y cores."""

    for x, y in GRIDS.values():
        mesh = _fake_mesh(x, y)
        for units in (
            gr_write.UNITS,
            gr_read.BRANCHES,
            gr_read.PARTIAL_TILES,
            gr_read.HIDDEN_TILES,
            gr_read.LOW_RANK_CORES,
        ):
            work = fp.split_work(units, mesh)
            assert _inside([w.core for w in work], _grid(x, y)) and sum(w.count for w in work) == units
        assert gr_write.cores_of(mesh, {}) == min(gr_write.UNITS, x * y)


def test_score_row_gather_pages_stay_under_the_all_gather_page_limit():
    """ttnn.all_gather's kernel headers (multicast_common.hpp, unicast_common.hpp) fill a uint16 array with the page
    size: a page of 65,536 bytes or more does not compile.  The block-score row is context / 4 bf16 columns (65,536
    bytes at 131072 tokens), so the fused path gathers it as 1024-column (2 KB) pages at every context and the merge
    reads those pages; the composed path's all_reduce (all_reduce_async) never used that header."""

    import inspect
    import re

    assert qsa.SCORE_GATHER_PAGE == qsa_block.SCORE_CHUNK == 1024 and 2 * qsa.SCORE_GATHER_PAGE < 65536
    for context in SUPPORTED_CONTEXTS:
        width = context // qsa.COMPRESS_RATIO
        assert width % qsa.SCORE_GATHER_PAGE == 0
        assert (
            2 * width >= 65536 or context < 131072
        )  # the whole-row page hits the limit from 128k on: the fault this guards
        pages = width // qsa.SCORE_GATHER_PAGE
        for x, y in GRIDS.values():
            work = qsa_block.score_merge_work(width, _grid(x, y))
            assert sum(w.count for w in work) == pages  # one chunk per page
    fused = re.sub(r"\s+", "", inspect.getsource(qsa.Qwen38TTNNQSA._score_blocks_fused))
    assert "pages=self.allocated_compressed_blocks//SCORE_GATHER_PAGE" in fused
    assert "paged=ttnn.reshape(score_row,(1,1,pages,SCORE_GATHER_PAGE))" in fused
    assert "gathered=ttnn.all_gather(paged,dim=2,cluster_axis=TP_AXIS," in fused
    assert "_require_shape(gathered,(1,1,TP_SIZE*pages,SCORE_GATHER_PAGE)" in fused
    merge = re.sub(r"\s+", "", inspect.getsource(qsa_block.score_merge))
    # the check, whichever way the condition is wrapped: 1024-column pages, four devices' rows of `chunks` pages each,
    # the row count read off the page count (the mask may carry more rows: the lanes pass their 32-row mask)
    assert "shape[3]!=SCORE_CHUNK" in merge and "shape[2]%(DEVICES*chunks)" in merge
    assert "rows=shape[2]//(DEVICES*chunks)" in merge
    composed = re.sub(r"\s+", "", inspect.getsource(qsa_block.score_merge_composed))
    assert "stacked=ttnn.reshape(gathered,(DEVICES,1,1,width))" in composed
