# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The model's cross-chip collectives behind one switch, CCL_IMPL (tt/settings.py; P.5):

    stock   ttnn.all_gather; the exact TP partial sums as all_gather + SFPU ttnn.add in mesh-column order (K-30), the
            hidden-split ones (ATTN_O_SUM / EXPERTS_SUM gather_add) then ttnn.mesh_partition to the chip's columns.
    fabric  the bring-up fabric forks: ttnn.bringup.fabric_all_gather (a drop-in for high_bw_all_gather with a
            link-rate program; preallocated output, caller-owned L1_SMALL semaphores) and
            ttnn.bringup.fabric_reduce_scatter(exact_fp32_add=True) (each add one IEEE fp32 rounding on the SFPU):
            sum_cols = ONE reduce-scatter over the TP axis by column blocks (dim 3: the chip's [S/SP, H/TP] slice
            of the sum, no full sum and no mesh_partition); tp_sum = a reduce-scatter by column blocks + a fabric
            all_gather of the columns (the sum replicated, bit-identical on every chip of the row; TP_SUM_BLOCKS=rows:
            row blocks, whose add order depends on the row's place in the chunk on TP > 2, DS41-128).

    fabric_bf16  (P.6, owner comparison: Marko Staletovic's GLM-5.3 choice, glm53_flash_d_p/tt/common.py:scatter_rows,
            GLM_SCATTER_OP=fabric_bf16) fabric, but the TP partial sums that go through the reduce-scatter (sum_cols:
            ATTN_O_SUM / EXPERTS_SUM; tp_sum: HC_TP_SUM's tile-aligned ones: mHC mixes, q_a, compressor, indexer
            gates) are typecast to bf16 and reduced by fabric_reduce_scatter's default FPU add (bf16 in, bf16 out),
            tp_sum's row gather in bf16 too, then typecast back to fp32 for the consumers. Lossy: one bf16 rounding of
            each partial and of the sum. The sums the fork does not take (norm sum of squares [S/SP, 1], engram gates
            [S/SP, 12]) stay the exact fp32 gather_add.

A group of 2 (TP = 2: 2x2, 2x4 rows) adds a + b once: bit-identical to the stock gather_add. With TP = 4 (Galaxy) the
fabric line adds in its own order (own + upstream partials), still one exact fp32 rounding per add.

Not on a fork (they stay stock under fabric): the head <-> seq all_to_all (ttnn.experimental.all_to_all_async_generic;
no bring-up fork), and any gather the fork's contract refuses (`_fits`: a ROW_MAJOR gather of the innermost dim, a
non-DRAM-interleaved input, a TILE gather of a dim that is not tile-aligned). The optional reduce_scatter / all_reduce
values of ATTN_O_SUM / EXPERTS_SUM / HC_TP_SUM (the FPU CCLs) are stock ttnn in both modes."""

from __future__ import annotations

import ttnn

from .settings import settings

TP_AXIS = 1
DRAM = ttnn.DRAM_MEMORY_CONFIG
AG_LINKS = None  # fabric_all_gather: every link reported usable on the axis (2x2 BH: 4; vs 2 links 1.4-2x faster)

_SEMS: dict = {}
_RS_LINKS: dict = {}


_DEALLOC = [ttnn.deallocate]


_PINNED: set = set()  # EXPERTS_XG_PINNED: buffer addresses of persistent tensors dealloc leaves alone


_SITES: dict = {}  # HOST_ALLOC=on (P.15): id(mesh) -> (mesh, {(site, shape, dtype, layout): load-time output})
_PINS: dict = {}  # id(mesh) -> (mesh, [the load-time tensors pinned on it]: held, so their ids stay theirs)
_KEEP: set = set()  # HOST_ALLOC=on: id() of those tensors (one set lookup in dealloc, no buffer query)
SITE_MISSES: dict = {}  # site -> gathers whose output spec had no load-time buffer (allocated as before; diagnostic)


def pin(mesh, t: ttnn.Tensor) -> None:
    """A persistent tensor of ``mesh`` that ``dealloc`` leaves alone (HOST_ALLOC's load-time outputs). Callers hand
    this very object on (never a new wrapper of its buffer: DS41-149)."""
    hit = _PINS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        if hit is not None:  # (a closed mesh's entry: its tensors are no longer kept)
            _KEEP.difference_update(id(x) for x in hit[1])
        hit = _PINS[id(mesh)] = (mesh, [])
    hit[1].append(t)
    _KEEP.add(id(t))


def reserve(mesh, site: str, shape, dtype, layout) -> ttnn.Tensor:
    """Load time (HOST_ALLOC=on, tt/host_alloc.reserve_sites): the persistent DRAM output of ``site``'s gathers whose
    output is ``shape`` / ``dtype`` / ``layout`` (``all_gather(site=)`` writes it; never freed)."""
    hit = _SITES.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        hit = _SITES[id(mesh)] = (mesh, {})
    key = (site, tuple(int(d) for d in shape), dtype, layout)
    t = hit[1].get(key)
    if t is None:
        t = hit[1][key] = ttnn.allocate_tensor_on_device(ttnn.Shape(list(key[1])), dtype, layout, mesh, DRAM)
        pin(mesh, t)
    return t


def _site_out(mesh, site: str, shape, dtype, layout):
    hit = _SITES.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        return None
    t = hit[1].get((site, tuple(shape), dtype, layout))
    if t is None:
        SITE_MISSES[site] = SITE_MISSES.get(site, 0) + 1
    return t


def dealloc(t: ttnn.Tensor) -> None:
    """dealloc(t); GLUE_FOLD=on (P.19, set by ``configure`` at model load): the binding itself, without
    ttnn's FastOperation wrapper (its config reads, graph-capture check and try / except: ~1.3 us of ~6 us a call,
    ~1000 calls a chunk). The bring-up profiler's op sequences then skip these calls on both of its runs alike."""
    if id(t) in _KEEP:  # HOST_ALLOC=on (P.15): a load-time output
        return
    if _PINNED and t.is_allocated() and t.buffer_address() in _PINNED:
        return
    _DEALLOC[0](t)


def configure() -> None:
    """Model load: bind ``dealloc`` to the setting's value."""
    _DEALLOC[0] = ttnn._ttnn.operations.core.deallocate if glue_fold() else ttnn.deallocate


def glue_fold() -> bool:
    """GLUE_FOLD=on (P.19): fabric_reduce_scatter with program_hash (and the other folds of tt/settings.py)."""
    return settings.get("GLUE_FOLD") == "on"


def prepared() -> bool:
    """GENERIC_OP_PREPARED=on (P.1, galaxy): fabric_reduce_scatter's cached descriptors dispatched as prepared programs
    (fork option prepared=True; needs GLUE_FOLD's program_hash, off without it)."""
    return settings.get("GENERIC_OP_PREPARED") == "on" and glue_fold()


def fabric() -> bool:
    """The bring-up fabric forks (CCL_IMPL fabric or fabric_bf16)."""
    return settings.get("CCL_IMPL") in ("fabric", "fabric_bf16")


def bf16_sums() -> bool:
    """CCL_IMPL=fabric_bf16: the reduce-scattered TP partial sums in bf16 (P.6)."""
    return settings.get("CCL_IMPL") == "fabric_bf16"


def partial_dtype(mesh) -> ttnn.DataType:
    """The dtype a producer writes a TP partial in when the sum is sum_cols / tp_sum's gather_add path: bf16 under
    CCL_IMPL=fabric_bf16 on a split TP axis (the producer's packer rounds its fp32 DEST to bf16 once, the rounding the
    typecast would do, without the typecast's DRAM round trip: [1024, 5120] 0.071 ms), else fp32."""
    return ttnn.bfloat16 if bf16_sums() and mesh.shape[TP_AXIS] > 1 else ttnn.float32


def _rs_bf16(mesh, t: ttnn.Tensor, dim: int) -> ttnn.Tensor:
    """fp32 or bf16 partials ``t`` -> bf16 (typecast when fp32) -> fabric_reduce_scatter with its default FPU add
    (bf16 output, as GLM-5.3's scatter_rows). Does not consume ``t``."""
    tb = t if t.dtype == ttnn.bfloat16 else ttnn.typecast(t, ttnn.bfloat16, memory_config=DRAM)
    out = ttnn.bringup.fabric_reduce_scatter(
        tb,
        cluster_axis=TP_AXIS,
        num_links=rs_links(mesh),
        dim=dim,
        cache_descriptors=True,
        program_hash=glue_fold(),
        prepared=prepared(),
    )
    if tb is not t:
        dealloc(tb)
    return out


def _exact_input(t: ttnn.Tensor) -> ttnn.Tensor:
    """A bf16 partial (partial_dtype) on a path that sums exactly: typecast to fp32 (exact). Returns ``t`` itself
    when it is fp32; the caller frees what it got only when it is not ``t``."""
    return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32, memory_config=DRAM)


def _to_fp32(t: ttnn.Tensor) -> ttnn.Tensor:
    """bf16 -> fp32 (exact), consuming ``t``: the consumers of the sums take fp32."""
    out = ttnn.typecast(t, ttnn.float32, memory_config=DRAM)
    dealloc(t)
    return out


def _sems(mesh):
    """fabric_all_gather's caller-owned semaphores (zero, left zero by every call; L1_SMALL, which the spec's
    device_params reserve): one pair per mesh, the whole worker grid."""
    hit = _SEMS.get(id(mesh))
    if hit is None or hit[0] is not mesh:
        g = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        sems = [ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL) for _ in range(2)]
        hit = _SEMS[id(mesh)] = (mesh, sems)
    return hit[1]


def rs_links(mesh, axis: int = TP_AXIS) -> int:
    """fabric_reduce_scatter's link count on ``axis``: the fewest usable fabric links between any two neighbours of
    the axis (read from the mesh, so it fits every box), capped at half the DRAM banks (the op gives each link two
    final cores with a bank set each). 2x2 BH: 4 links (sum_cols [1024, 5120] fp32: 0.150 ms vs 0.235 at 2)."""
    hit = _RS_LINKS.get((id(mesh), axis))
    if hit is not None and hit[0] is mesh:
        return hit[1]
    from ttnn.bringup.fabric_all_gather_ttnn.fabric_all_gather_py import _num_banks

    rows, cols = tuple(mesh.shape)
    node = lambda r, c: mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c))  # noqa: E731
    pairs = [((r, c), (r, c + 1)) for r in range(rows) for c in range(cols - 1)] if axis == 1 else []
    pairs += [((r, c), (r + 1, c)) for r in range(rows - 1) for c in range(cols)] if axis == 0 else []
    found = min(
        min(
            len(ttnn.get_forwarding_link_indices(node(*a), node(*b))),
            len(ttnn.get_forwarding_link_indices(node(*b), node(*a))),
        )
        for a, b in pairs
    )
    links = max(1, min(found, _num_banks(mesh) // 2))
    _RS_LINKS[(id(mesh), axis)] = (mesh, links)
    return links


def setup(mesh) -> None:
    """Load time (agent rule 5): fabric_all_gather's semaphores and fabric_reduce_scatter's plan and counters on the TP
    axis (the partial sums' axis), so that no forward pass creates a semaphore or synchronizes the mesh.
    Idempotent; nothing under stock."""
    if not fabric():
        return
    from ttnn.bringup.fabric_reduce_scatter_ttnn import prepare

    _sems(mesh)
    if mesh.shape[TP_AXIS] > 1:  # the partial sums reduce over the TP axis only
        prepare(mesh, cluster_axis=TP_AXIS, num_links=rs_links(mesh))


def _fits(t: ttnn.Tensor, dim: int) -> bool:
    """fabric_all_gather's contract (high_bw_all_gather's): DRAM interleaved input; a ROW_MAJOR gather never along
    the innermost dim; a TILE gather of the last two dims only when that dim is tile-aligned (the output is
    preallocated at G x the padded size)."""
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return False
    rank = len(t.shape)
    d = dim % rank
    if t.layout == ttnn.ROW_MAJOR_LAYOUT:
        return d != rank - 1
    if t.layout != ttnn.TILE_LAYOUT:
        return False
    return d < rank - 2 or t.shape[d] % 32 == 0


def all_gather(
    t: ttnn.Tensor, dim: int, cluster_axis: int, out: ttnn.Tensor | None = None, site: str | None = None
) -> ttnn.Tensor:
    """ttnn.all_gather(t, dim, cluster_axis) into DRAM (a data move, exact), on fabric_all_gather when CCL_IMPL=fabric
    and its contract fits. Does not consume ``t``. ``site`` (HOST_ALLOC=on, P.15): the site's load-time output when one
    was reserved for this output spec (``reserve``; the result is then that buffer, which ``dealloc`` skips)."""
    mesh = t.device()
    if not fabric() or not _fits(t, dim):
        return ttnn.all_gather(t, dim=dim, cluster_axis=cluster_axis, memory_config=DRAM)
    g = mesh.shape[cluster_axis]
    shape = list(t.shape)
    shape[dim % len(shape)] *= g
    if out is None and site is not None and _SITES:
        out = _site_out(mesh, site, shape, t.dtype, t.layout)
    if out is None:  # (EXPERTS_XG_PINNED: the caller's persistent output)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), t.dtype, t.layout, mesh, DRAM)
    s0, s1 = _sems(mesh)
    res = ttnn.bringup.fabric_all_gather(
        t, dim, out, cluster_axis=cluster_axis, num_links=AG_LINKS, ready_semaphore=s0, data_valid_semaphore=s1
    )
    # (a load-time output: the op returns a new wrapper of its buffer (DS41-149); hand back the kept object itself)
    return out if site is not None and id(out) in _KEEP else res


def _gather_add(mesh, t: ttnn.Tensor) -> ttnn.Tensor:
    """The exact sum over the TP axis, replicated: all_gather on dim 1 -> [1, TP, R, W], then the TP slices added in
    mesh-column order on the SFPU (K-30). Does not consume ``t``."""
    tp = mesh.shape[TP_AXIS]
    r, w = t.shape[-2], t.shape[-1]
    g = all_gather(t, 1, TP_AXIS)
    acc = ttnn.slice(g, [0, 0, 0, 0], [1, 1, r, w])
    for c in range(1, tp):
        part = ttnn.slice(g, [0, c, 0, 0], [1, c + 1, r, w])
        nxt = ttnn.add(acc, part, dtype=ttnn.float32, memory_config=DRAM)
        dealloc(acc)
        dealloc(part)
        acc = nxt
    dealloc(g)
    return acc


def _rs_ok(t: ttnn.Tensor, dim: int, g: int) -> bool:
    s = list(t.shape)
    mc = t.memory_config()
    ok_dtype = t.dtype == ttnn.float32 or (t.dtype == ttnn.bfloat16 and bf16_sums())
    if t.layout != ttnn.TILE_LAYOUT or not ok_dtype or s[0] != 1 or s[1] != 1:
        return False
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return False
    if s[2] % 32 or s[3] % 32:  # the fork checks a TILE input is tile-aligned (logical == padded)
        return False
    if dim == 2:
        return s[2] % (32 * g) == 0
    return s[3] % (32 * g) == 0


def tp_sum(mesh, t: ttnn.Tensor, exact: bool = False, fp32: bool = True) -> ttnn.Tensor:
    """The exact fp32 sum of ``t`` ([1, 1, R, W] fp32 TILE) over the TP axis, replicated on every chip of the row,
    bit-identical across them. Does not consume ``t``. stock: _gather_add. fabric: fabric_reduce_scatter of column
    blocks (exact SFPU add) -> fabric_all_gather of the columns (TP_SUM_BLOCKS=rows: row blocks and rows); when the
    split dim is not a multiple of 32 TP (the norms' [S/SP, 1] sum of squares, the [S/SP, 32] mHC row),
    _gather_add on the fabric gather (mesh-column order). Column blocks or the gather: every row adds in the same
    order wherever the chunk starts (DS41-128). fabric_bf16 (P.6): the
    RS (FPU add) and the row gather in bf16, the result typecast to fp32; ``t`` may be bf16 (partial_dtype).
    ``exact`` (an fp32 ``t``): the exact fp32 sum under fabric_bf16 too (the engram gate partials, MHC_FUSED=on: the
    [S/SP, 32] row the fork writes is tile-aligned, where the old [S/SP, 12] row was not and stayed exact).
    ``fp32`` False (BF16_ACTIVATIONS=on, P.20): the fabric_bf16 path's bf16 result as it is (no typecast back; the
    consumer reads bf16); the other paths return fp32 either way."""
    tp = mesh.shape[TP_AXIS]
    # TP_SUM_BLOCKS=cols (K.1, DS41-128): column blocks, so a row's add order never depends on its row in the chunk
    dim = 3 if settings.get("TP_SUM_BLOCKS") == "cols" else 2
    if not fabric() or not _rs_ok(t, dim, tp):
        tf = _exact_input(t)
        out = _gather_add(mesh, tf)
        if tf is not t:
            dealloc(tf)
        return out
    if bf16_sums() and not exact:  # CCL_IMPL=fabric_bf16: RS and gather in bf16, back to fp32 (P.6)
        part = _rs_bf16(mesh, t, dim)
        out = all_gather(part, dim, TP_AXIS)
        dealloc(part)
        return _to_fp32(out) if fp32 else out
    part = ttnn.bringup.fabric_reduce_scatter(
        t,
        cluster_axis=TP_AXIS,
        num_links=rs_links(mesh),
        dim=dim,
        exact_fp32_add=True,
        cache_descriptors=True,
        program_hash=glue_fold(),
        prepared=prepared(),
    )
    out = all_gather(part, dim, TP_AXIS)
    dealloc(part)
    return out


def sum_rows(mesh, t: ttnn.Tensor) -> ttnn.Tensor | None:
    """INDEXER_GLUE=on (P.16): the sum of ``t`` ([1, 1, R, W] TILE partials) over the TP axis, only the chip's row
    block [1, 1, R/TP, W] (mesh column c: rows [c R/TP, (c+1) R/TP), what tp_sum + mesh_partition(dim 2) keep): ONE
    fabric_reduce_scatter by row blocks, no row gather. fabric_bf16: the bf16 RS (FPU add) of bf16 / fp32 partials,
    bf16 out, the same values tp_sum's bf16 path gathers; fabric: the exact fp32 RS. None when the fork does not fit
    (stock, untiled rows): the caller takes tp_sum + mesh_partition. Does not consume ``t``."""
    tp = mesh.shape[TP_AXIS]
    if tp == 1 or not fabric() or not _rs_ok(t, 2, tp):
        return None
    if bf16_sums():
        return _rs_bf16(mesh, t, 2)
    return ttnn.bringup.fabric_reduce_scatter(
        t,
        cluster_axis=TP_AXIS,
        num_links=rs_links(mesh),
        dim=2,
        exact_fp32_add=True,
        cache_descriptors=True,
        program_hash=glue_fold(),
        prepared=prepared(),
    )


def sum_cols(mesh, t: ttnn.Tensor, fp32: bool = True) -> ttnn.Tensor:
    """The exact fp32 sum of ``t`` ([1, 1, R, H] fp32 TILE) over the TP axis, the chip's hidden columns
    [1, 1, R, H/TP] (mesh column c: [c H/TP, (c+1) H/TP)). Does not consume ``t``. stock: _gather_add +
    mesh_partition. fabric: ONE fabric_reduce_scatter by column blocks (dim 3, exact SFPU add). fabric_bf16 (P.6):
    the same RS in bf16 (FPU add), the result typecast to fp32 (mhc_post takes fp32); ``t`` may be bf16. ``fp32``
    False (ATTN_GLUE=on, P.15): that bf16 result as is (the caller's consumer takes bf16)."""
    tp = mesh.shape[TP_AXIS]
    if fabric() and _rs_ok(t, 3, tp) and bf16_sums():  # CCL_IMPL=fabric_bf16 (P.6)
        out = _rs_bf16(mesh, t, 3)
        return _to_fp32(out) if fp32 else out
    if fabric() and _rs_ok(t, 3, tp):
        return ttnn.bringup.fabric_reduce_scatter(
            t,
            cluster_axis=TP_AXIS,
            num_links=rs_links(mesh),
            dim=3,
            exact_fp32_add=True,
            cache_descriptors=True,
            program_hash=glue_fold(),
            prepared=prepared(),
        )
    tf = _exact_input(t)
    full = _gather_add(mesh, tf)
    if tf is not t:
        dealloc(tf)
    out = ttnn.mesh_partition(full, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
    dealloc(full)
    return out
