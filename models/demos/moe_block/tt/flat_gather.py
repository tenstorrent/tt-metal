# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""flat_gather: lay the flat routed experts' weights (ttnn.experimental.moe_block.flat_routed_expert's bank layout) out
ON THE DEVICE from expert matrices already in DRAM as bfp tiles (TtMoe's routed-expert ttnn cache): no host dequantize /
pack / requantize (~4 min a Kimi-K3 layer of 896 experts that way).

  flat_face_map(H, I, E, lay)  host, once per layout: for every page (tile) of each laid-out tensor (w_gu, w_d - the
      width-sharded DRAM tensors of flat_routed_expert_weights.bank_sharded, page = row-major tile index t * banks + b)
      and each of its 4 faces (16 x 16), the source face it holds: ((s * NT + tile) * 4 + face), or -1 (padding: zeros).
      Sources s = e * 3 + m (local expert e; m 0 Wg [H, I], 1 Wu [H, I], 2 Wd [I, H]), tile = the source's row-major
      tile index, NT its tiles. Derived by running the C++ op's own weight prep (flat_routed_expert_weights.
      flat_weight_regions + bank_host_tensor) on index tensors, every face checked to be one source face, in order.
  gather(mesh, srcs, face_map, dst, pages=None)  device: one data-movement program a chip (the same everywhere) - each
      dest page: the source tiles its faces come from read into L1, the faces copied (a bfp face: its 16 shared exponent
      bytes and its mantissa bytes), the page written. srcs: the NS source tensors in s order, DRAM interleaved, the
      dst's dtype. pages: the dest page of each face_map row (default: row i is page i) - a subset of the pages.
  gather_batches(face_map, NT, E, per)  host: the face map split by local expert, `per` experts a batch: per batch
      (e0, e1, pages, sub map with sources renumbered from 3 e0) - so a layer's sources go to DRAM a batch at a time
      (the whole layer's sources next to its laid-out dest needed one layer's routed weights twice in DRAM).
  emulate_gather(srcs, face_map, banks)  host: gather's result on float matrices (the layout check, test_flat_gather.py).

TODO: this becomes part of the C++ flat op's weight prep (a "from bfp DRAM sources" path of the flat expert's weight
preparation); it is a generic_op program here only to run the real models now.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

import ttnn

_REPO = Path(__file__).resolve().parents[4]
_FW = None


def flat_weights_module():
    """ttnn/cpp/.../flat_routed_expert/flat_routed_expert_weights.py (the C++ op's Python weight prep)."""
    global _FW
    if _FW is None:
        path = (
            _REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py"
        )
        spec = importlib.util.spec_from_file_location("flat_routed_expert_weights", path)
        _FW = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_FW)
    return _FW


TPP = 32  # (dest pages a face-map table page holds: 32 entries x 8 words (4 faces, the dest page, 3 pad) = 1 KB)
TW = 8  # (words a table entry)

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t NS = get_compile_time_arg_val(0);   // source tensors
constexpr uint32_t NT = get_compile_time_arg_val(1);   // tiles a source
constexpr uint32_t TB = get_compile_time_arg_val(2);   // bytes a tile (page)
constexpr uint32_t EB = get_compile_time_arg_val(3);   // shared exponent bytes a face (the tile's first 4 EB)
constexpr uint32_t MB = get_compile_time_arg_val(4);   // mantissa bytes a face (after the exponents, face-major)
constexpr uint32_t TP = get_compile_time_arg_val(5);   // entries a table page (8 words each: 4 faces, dest page)
constexpr auto src_args = TensorAccessorArgs<6>();
constexpr auto tab_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
constexpr auto dst_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();

FORCE_INLINE void copy_bytes(uint32_t dst, uint32_t src, uint32_t n) {  // (L1 to L1, word-aligned, n % 4 == 0)
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < n / 4; ++i) d[i] = s[i];
}
FORCE_INLINE void zero_bytes(uint32_t dst, uint32_t n) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < n / 4; ++i) d[i] = 0;
}

void kernel_main() {
    const uint32_t tab_addr = get_arg_val<uint32_t>(0), dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t q0 = get_arg_val<uint32_t>(2), nq = get_arg_val<uint32_t>(3), np_ = get_arg_val<uint32_t>(4);
    const auto tab = TensorAccessor(tab_args, tab_addr, TP * 32);
    const auto dst = TensorAccessor(dst_args, dst_addr, TB);
    // L1: the table page | 4 source tile slots | 2 dest tiles
    const uint32_t base = get_write_ptr(0);
    const uint32_t tl1 = base, sl1 = base + TP * 32, dl1 = sl1 + 4 * TB;
    uint32_t wbuf = 0;
    for (uint32_t q = q0; q < q0 + nq; ++q) {  // (table page q: entries q TP ..)
        noc_async_read(tab.get_noc_addr(q), tl1, TP * 32);
        noc_async_read_barrier();
        volatile tt_l1_ptr int32_t* e = reinterpret_cast<volatile tt_l1_ptr int32_t*>(tl1);
        for (uint32_t i = 0; i < TP; ++i) {
            if (q * TP + i >= np_) break;
            const uint32_t p = static_cast<uint32_t>(e[i * 8 + 4]);  // (the dest page)
            int32_t tile_of[4];  // (the slot each face's source tile went to: distinct tiles read once)
            int32_t slot_tile[4] = {-1, -1, -1, -1};
            uint32_t ns = 0;
            for (uint32_t f = 0; f < 4; ++f) {
                const int32_t c = e[i * 8 + f];
                tile_of[f] = -1;
                if (c < 0) continue;
                const int32_t g = c >> 2;  // (s NT + tile)
                uint32_t k = 0;
                while (k < ns && slot_tile[k] != g) ++k;
                if (k == ns) {
                    slot_tile[ns++] = g;
                    const uint32_t s = static_cast<uint32_t>(g) / NT, t = static_cast<uint32_t>(g) % NT;
                    const auto src = TensorAccessor(src_args, get_common_arg_val<uint32_t>(s), TB);
                    noc_async_read(src.get_noc_addr(t), sl1 + k * TB, TB);
                }
                tile_of[f] = static_cast<int32_t>(k);
            }
            const uint32_t d = dl1 + wbuf * TB;
            noc_async_write_barrier();  // (this dest buffer's previous write out)
            noc_async_read_barrier();
            for (uint32_t f = 0; f < 4; ++f) {
                const int32_t c = e[i * 8 + f];
                if (c < 0) {
                    zero_bytes(d + f * EB, EB);
                    zero_bytes(d + 4 * EB + f * MB, MB);
                    continue;
                }
                const uint32_t sf = static_cast<uint32_t>(c) & 3, s0 = sl1 + static_cast<uint32_t>(tile_of[f]) * TB;
                copy_bytes(d + f * EB, s0 + sf * EB, EB);
                copy_bytes(d + 4 * EB + f * MB, s0 + 4 * EB + sf * MB, MB);
            }
            noc_async_write(d, dst.get_noc_addr(p), TB);
            wbuf ^= 1;
        }
    }
    noc_async_write_barrier();
}
"""

# (bytes a bfp tile: 4 faces' shared exponents, then each face's mantissas)
_BFP = {ttnn.bfloat4_b: (576, 16, 128), ttnn.bfloat8_b: (1088, 16, 256)}


def _ids(s: int, NT: int, R: int, C: int) -> torch.Tensor:
    """Source s's [R, C] as index values: 1 + ((s NT + tile) 4 + face) 256 + position in the face (int64)."""
    r = torch.arange(R).view(-1, 1)
    c = torch.arange(C).view(1, -1)
    tile = (r // 32) * (C // 32) + (c // 32)
    face = ((r % 32) // 16) * 2 + (c % 32) // 16
    pos = (r % 16) * 16 + (c % 16)
    return 1 + ((s * NT + tile) * 4 + face) * 256 + pos


def _pages(regions: list, banks: int) -> torch.Tensor:
    """bank_host_tensor's host matrix of one device's regions -> [pages, 32, 32] (page t banks + b)."""
    host = flat_weights_module().bank_host_tensor(regions, banks)
    T = host.shape[0] // 32
    return host.view(T, 32, banks, 32).permute(0, 2, 1, 3).reshape(T * banks, 32, 32)


def flat_face_map(H: int, I: int, E: int, lay: dict) -> dict:
    """{"w_gu" / "w_d": int32 [pages, 4]} - each dest page's 4 faces' sources (module docstring)."""
    NT = (H // 32) * (I // 32)
    assert (E * 3 * NT * 4) * 256 + 256 < 2**31, "index values past int32"
    W_l = [
        (
            _ids(3 * e, NT, H, I).to(torch.int32),
            _ids(3 * e + 1, NT, H, I).to(torch.int32),
            _ids(3 * e + 2, NT, I, H).to(torch.int32),
        )
        for e in range(E)
    ]
    regions, dregs = flat_weights_module().flat_weight_regions(W_l, lay)
    del W_l
    out = {}
    pos = torch.arange(256, dtype=torch.int64)
    for name, regs in (("w_gu", regions), ("w_d", dregs)):
        tiles = _pages(regs, lay["banks"])  # [pages, 32, 32]
        P = tiles.shape[0]
        faces = tiles.reshape(P, 2, 16, 2, 16).permute(0, 1, 3, 2, 4).reshape(P, 4, 256).to(torch.int64)
        first = faces[:, :, 0]
        pad = first == 0
        assert bool((faces[pad] == 0).all()), f"{name}: a face partly padding"
        ok = (faces == first.unsqueeze(-1) + pos) | pad.unsqueeze(-1)
        assert bool(ok.all()), f"{name}: a face is not one source face in order (a transposed or split face)"
        out[name] = torch.where(pad, torch.full_like(first, -1), (first - 1) // 256).to(torch.int32)
    return out


def emulate_gather(srcs: list, face_map: torch.Tensor) -> torch.Tensor:
    """gather on the host: float source matrices (s order) -> the dest pages [pages, 32, 32]."""

    def faces_of(s):  # (each source's own [rows, cols]: Wd is [I, H]; every source has NT tiles)
        R, C = s.shape
        return s.reshape(R // 32, 2, 16, C // 32, 2, 16).permute(0, 3, 1, 4, 2, 5).reshape(-1, 256)

    faces = torch.cat([faces_of(s) for s in srcs])
    fm = face_map.to(torch.int64)
    got = torch.where((fm >= 0).unsqueeze(-1), faces[fm.clamp(min=0)], torch.zeros(()))  # [pages, 4, 256]
    P = fm.shape[0]
    return got.reshape(P, 2, 2, 16, 16).permute(0, 1, 3, 2, 4).reshape(P, 32, 32)


def gather_batches(face_map: dict, NT: int, E: int, per: int) -> list:
    """flat_face_map's {name: [pages, 4]} split into batches of `per` local experts: [(e0, e1, {name: (pages int32 [n],
    sub map int32 [n, 4])})], a page in the batch of the expert its faces come from (a page's faces all come from one
    expert's gate / up / down; all-padding pages go with the first batch), its sources renumbered s - 3 e0."""
    per = max(1, min(per, E))
    out = [(e0, min(E, e0 + per), {}) for e0 in range(0, E, per)]
    for name, fm in face_map.items():
        fm = fm.to(torch.int64)
        pad = fm < 0
        expert = torch.where(pad, torch.full_like(fm, -1), (fm >> 2) // NT // 3)  # [pages, 4]
        hi = expert.max(dim=1).values
        lo = torch.where(pad, torch.full_like(fm, E), expert).min(dim=1).values
        mixed = (hi >= 0) & (lo != hi)
        assert not bool(mixed.any()), f"{name}: {int(mixed.sum())} pages take faces from two experts"
        page_batch = hi.clamp(min=0) // per
        for b, (e0, _, sub) in enumerate(out):
            rows = torch.nonzero(page_batch == b).flatten()
            m = fm[rows]
            m = torch.where(m < 0, m, m - 3 * e0 * NT * 4)
            sub[name] = (rows.to(torch.int32), m.to(torch.int32))
    for name, fm in face_map.items():
        assert sum(int(s[name][0].numel()) for _, _, s in out) == fm.shape[0], f"{name}: batches miss pages"
    return out


def gather(mesh, srcs: list, face_map: torch.Tensor, dst, num_cores: int | None = None, pages=None):
    """dst (a laid-out flat expert weight tensor) <- its faces from srcs (module docstring); face_map int32 [n, 4], row i
    the faces of dest page pages[i] (default i)."""
    tb, eb, mb = _BFP[dst.dtype]
    assert all(s.dtype == dst.dtype for s in srcs), "sources and dest: one bfp dtype"
    shape = tuple(srcs[0].shape)
    NT = (shape[-2] // 32) * (shape[-1] // 32)
    n = face_map.shape[0]
    if n == 0:
        return
    if pages is None:
        pages = torch.arange(n, dtype=torch.int32)
    assert pages.shape[0] == n, "a dest page a face-map row"
    nq = -(-n // TPP)
    tab = torch.full((nq * TPP, TW), -1, dtype=torch.int32)
    tab[:n, :4] = face_map
    tab[:n, 4] = pages.to(torch.int32)
    tab_t = ttnn.from_torch(
        tab.view(nq, TPP * TW),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    grid = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][: num_cores or grid.x * grid.y]
    per = -(-nq // len(cores))
    used = [(c, i * per, min(per, nq - i * per)) for i, c in enumerate(cores) if i * per < nq]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c, _, _ in used])
    rt = ttnn.RuntimeArgs()
    for c, q0, n_q in used:
        rt[c.x][c.y] = [int(tab_t.buffer_address()), int(dst.buffer_address()), q0, n_q, n]
    ct = (
        [len(srcs), NT, tb, eb, mb, TPP]
        + list(ttnn.TensorAccessorArgs(srcs[0]).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(tab_t).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
    )
    # (generic_op's program cache does not re-read this descriptor's runtime args on a hit: every address set and entry
    # count gets its own program - a salt the kernel ignores, after its accessor args)
    salt = 0
    for a in [n, *(int(t.buffer_address()) for t in (tab_t, dst, *srcs))]:
        salt = (salt * 1000003 + a) & 0x7FFFFFFF
    ct.append(salt)
    prog = ttnn.ProgramDescriptor()
    prog.cbs = [
        ttnn.CBDescriptor(
            total_size=TPP * TW * 4 + 6 * tb,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, TPP * TW * 4 + 6 * tb)],
        )
    ]
    prog.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            common_runtime_args=[int(s.buffer_address()) for s in srcs],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
    ]
    R, C = tuple(mesh.shape)
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog
    ttnn.generic_op(list(srcs) + [tab_t, dst], md)
    ttnn.deallocate(tab_t)  # (ordered after the gather on the command queue)
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(tab_t)
