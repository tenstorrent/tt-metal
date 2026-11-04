# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a `mhc_pre`: the decode mHC "pre" of one sublayer as ONE multi-core generic_op program.

Replaces (B=1, fused path) to_memory_config + to_stacked + sharded rms_norm (+ I2S / S2I) + reshape + the 512-batch
fn matmul + fast_reduce_nc (1 core) + slice + FusedPrePost (1 core) + Sinkhorn (1 core) + the sublayer's input
RMSNorm (+ I2S / S2I / typecast): ~15 programs, ~210 us per sublayer; at more rows (stream-major path) the ~25-op
slice / concat / matmul / reduce / mhc_split_sinkhorn / permute / norm chain.

Math per token t (rows of the residual x [T, 16384] fp32, 4 streams of 4096):
    m      = (x @ fn^T) / rms(x)                 rms(x) = sqrt(mean(x^2) + eps_x)   (24 mixes)
    z      = m * scale + bias                    (per-mix scale: pre / post / comb scale; bias = hc base)
    pre    = sigmoid(z[0:4]) + eps_hc;  post = 2 sigmoid(z[4:8]);  comb = Sinkhorn(softmax(z[8:24] as 4 x 4)) (iters)
    h      = sum_s pre_s x_s                     (4096)
    hn     = h / rms(h) * w                      rms(h) = sqrt(mean(h^2) + eps_h); w = the sublayer's norm weight
Outputs: hn [1,1,T,4096] (bf16, or fp32), post / comb in the layout of the mix that follows (stacked mix_streams
format, or the stream-major format), optionally the residual in the mix's layout.

Program (kernels in tt/kernels/mhc_pre/, one source per RISC; roles by define):
    64 K cores (8 x 8)   core k = 8x + y owns column tiles {2k, 2k+1} of all 4 streams: reads its 8 x tiles (32 KB
                         fp32), 8 fn^T tiles (16 KB bf16) and 2 norm-weight tiles; M = x @ fn^T (FPU, fp32 DST,
                         HiFi4) and S = sum x^2 (SFPU, exact fp32) -> its column leader.
    8 leaders (y == 0)   sum their column's 8 partials (SFPU) -> root; relay the root's messages to their column.
    1 root               sums the 8 group partials, 1/rms(x), z (column / row broadcasts), re-lays z into the
                         "plane layout" (lane = token) and runs the whole coefficient + 20-iteration Sinkhorn chain
                         for all T tokens in ONE SFPU pass (custom SFPI, mhc_pre_sfpu.hpp); writes post / comb; sends
                         pre (4 x T fp32) to the leaders.
    K cores              h = sum_s x_s * pre_s (SFPU column broadcast, exact fp32), sum h^2 -> leaders -> root ->
                         1/rms(h) (32 fp32) -> leaders -> K cores; hn = h * inv * w -> output tiles.
NoC split: BRISC (NoC0) reads DRAM / L1 tensors and writes the outputs; NCRISC (NoC1) carries the core-to-core
partials (unicast, write barrier before every semaphore increment) and the messages. No multicast (the 8 x 8 logical
grid spans a harvested / DRAM column on the p300 dies).

L1 per core (see doc/ttnn/07a-fused-streaming-layer/l1_ledger.md): every core 64.5 KB (gather + mailbox, uniform
addresses for the remote writes), K cores +108 KB, root +40 KB.

Numerics vs the op chain: the matmul reads x in the FPU's tf32 like today's 512-batch matmul; every other stage is
exact fp32 SFPU arithmetic (today: tf32 FPU rms_norm, tf32 fast_reduce_nc sums, tf32 collapse matmul). PCC-gated
(>= 0.9999 per output vs the op chain), not bitwise. `reference_step` is the program's arithmetic in torch (float64
accumulation, tf32 x for the matmul) and `decode_taps` reads the DEBUG_TAPS tensor.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import torch

import ttnn

KERNEL_DIR = os.path.join("models", "autoports", "zai_org_glm_5_3_flash", "tt", "kernels", "mhc_pre")
FP32, BF16 = ttnn.float32, ttnn.bfloat16
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
TILE = 32
N_STREAMS = 4
COL_TILES = 128
N_MIX = 24
GRID_X = GRID_Y = 8  # K cores
ROOT = (8, 0)

# mirrors mhc_pre_common.hpp
CB = dict(
    GATHER=0, MSG=1, X=2, X32=3, FN=4, W=5, PART=6, PRE=7, H=8, S2=9, INV=10, OUT=11, XSTK=12, GSUM=13,
    RED=14, PLANE=15, COEF=16, SB=17, INVR=18, SCR=19, DBG=20,
)  # fmt: skip
SEMAPHORES = 4
TAPS = ("M_total", "S_total", "z", "plane", "coef", "inv_h", "k0_M", "k0_S", "k0_pre0", "k0_inv")
N_TAPS = len(TAPS)
# fp32 CBs read by copy_tile: unpacked straight to DST (exact fp32 on the SFPU)
FP32_COPY_CBS = (CB["GATHER"], CB["X32"], CB["W"], CB["PRE"], CB["H"], CB["INV"], CB["PLANE"], CB["SB"])

# x input layouts / residual copy layouts / post-comb layouts
X_FLAT, X_STACKED = 0, 1
XOUT_NONE, XOUT_STACKED, XOUT_FLAT = 0, 1, 2
PC_STACKED, PC_STREAM_MAJOR = 0, 1


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def tile_off(r: int, c: int) -> int:
    return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)


def plane_off(p: int, t: int) -> int:
    return (p >> 3) * 256 + (((p & 7) >> 1) * 4 + (t >> 3)) * 16 + 2 * (t & 7) + (p & 1)


def face_major(tile: torch.Tensor) -> torch.Tensor:
    """[32, 32] row-major -> 1024 elements in the tile's face-major memory order."""
    return tile.reshape(2, 16, 2, 16).permute(0, 2, 1, 3).reshape(1024)


def from_face_major(flat: torch.Tensor) -> torch.Tensor:
    return flat.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)


def to_planes(values: torch.Tensor) -> torch.Tensor:
    """values [P <= 32, T <= 32] (plane p, token t) -> the [32, 32] row-major tile whose memory holds them in the
    plane layout."""
    mem = torch.zeros(1024, dtype=values.dtype)
    for p in range(values.shape[0]):
        for t in range(values.shape[1]):
            mem[plane_off(p, t)] = values[p, t]
    return from_face_major(mem)


def from_planes(tile: torch.Tensor, planes: int, tokens: int) -> torch.Tensor:
    mem = face_major(tile)
    return torch.stack([torch.stack([mem[plane_off(p, t)] for t in range(tokens)]) for p in range(planes)])


def tf32(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> tf32 by truncating the low 13 mantissa bits (the FPU source-register format)."""
    i = x.float().contiguous().view(torch.int32)
    return (i & ~0x1FFF).view(torch.float32)


# ======================================================================================================== oracle
def reference_step(x, fn_t, scale, bias, w, *, hc_eps, x_eps, h_eps, iters, n=N_STREAMS, matmul_tf32=True):
    """The program's arithmetic on host tensors. x [T, n*D] fp32; fn_t [n*D, 24]; scale / bias [24] fp32 (per mix);
    w [D] (norm weight). Returns a dict: M (x @ fn_t, tf32 x), S (sum x^2), inv_x, z, pre, post, comb, h, inv_h,
    hn (fp32; the program packs it to the output dtype)."""
    x = x.double()
    xm = tf32(x.float()).double() if matmul_tf32 else x
    M = xm @ fn_t.double()
    S = (x * x).sum(-1, keepdim=True)
    inv_x = torch.rsqrt(S / x.shape[-1] + x_eps)
    z = (M * inv_x) * scale.double() + bias.double()
    pre = torch.sigmoid(z[:, :n]) + hc_eps
    post = 2 * torch.sigmoid(z[:, n : 2 * n])
    comb = torch.softmax(z[:, 2 * n :].reshape(-1, n, n), dim=-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    d = x.shape[-1] // n
    xs = x.reshape(-1, n, d)
    h = (pre.unsqueeze(-1) * xs).sum(1)
    inv_h = torch.rsqrt((h * h).mean(-1, keepdim=True) + h_eps)
    hn = h * inv_h * w.double()
    return dict(M=M, S=S, inv_x=inv_x, z=z, pre=pre, post=post, comb=comb, h=h, inv_h=inv_h, hn=hn)


def decode_taps(dbg: torch.Tensor, t: int) -> dict:
    """The DEBUG_TAPS tensor [N_TAPS * 32, 32] (host) -> named per-token arrays."""
    tiles = dbg.float().reshape(N_TAPS, 32, 32)
    out = {name: tiles[i] for i, name in enumerate(TAPS)}
    out["z_tok"] = tiles[2][:t, :N_MIX]
    out["plane_tok"] = from_planes(tiles[3], N_MIX, t).T  # [t, 24]
    coef = from_planes(tiles[4], N_MIX, t).T  # [t, 24]
    out["pre"], out["post"], out["comb"] = coef[:, :4], coef[:, 4:8], coef[:, 8:].reshape(t, 4, 4)
    out["inv_h"] = tiles[5][:t, 0]
    return out


def merged_ranges(cores) -> ttnn.CoreRangeSet:
    """A few rectangles covering exactly the given logical (x, y) cores: maximal x-runs per row, then identical runs of
    consecutive rows stacked (multicast dispatch of binaries / launch messages per range)."""
    rows = {}
    for x, y in sorted({(int(x), int(y)) for x, y in cores}):
        rows.setdefault(y, []).append(x)
    runs = []  # (x0, x1, y)
    for y, xs in rows.items():
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            runs.append((start, prev, y))
            if x is not None:
                start = prev = x
    rects = []  # [x0, x1, y0, y1]
    for x0, x1, y in sorted(runs, key=lambda r: (r[0], r[1], r[2])):
        if rects and rects[-1][0] == x0 and rects[-1][1] == x1 and rects[-1][3] == y - 1:
            rects[-1][3] = y
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in rects])


# ======================================================================================================== program
@dataclass
class MhcPreOutputs:
    hn: ttnn.Tensor
    post: ttnn.Tensor
    comb: ttnn.Tensor
    x_out: ttnn.Tensor | None
    coef: ttnn.Tensor  # [1,1,32,32] fp32: row t = post_0..3 | comb_0..15 (the fused post-mix's input)


class MhcPre:
    """One sublayer's fused mHC pre (its fn / biases / scales / norm weight); call per decode step."""

    def __init__(
        self,
        mesh_device,
        fn_t: torch.Tensor,  # [n*D, 24] host (bf16 checkpoint values)
        scale: torch.Tensor,  # [24] per-mix scale (pre_scale x4 | post_scale x4 | comb_scale x16)
        bias: torch.Tensor,  # [24] (hc base: pre | post | comb row-major)
        norm_w: torch.Tensor,  # [D] the sublayer's RMSNorm weight
        *,
        hc_eps: float,
        x_eps: float,
        h_eps: float,
        iters: int,
        hidden: int = 4096,
        n: int = N_STREAMS,
        fn_dtype=BF16,
        placement: dict | None = None,
    ):
        assert n == N_STREAMS and hidden == COL_TILES * TILE and fn_t.shape == (n * hidden, N_MIX), fn_t.shape
        assert fn_dtype in (BF16, FP32), fn_dtype
        self.device = mesh_device
        self.fn_dtype = fn_dtype
        # placement (megakernel): {"k": 64 logical CoreCoords (K core k = 8 cx + cy, cy = 0 the column leader),
        # "root": CoreCoord}; None = the logical 8 x 8 grid at the origin + root (8, 0)
        self.placement = placement
        self.iters, self.hc_eps, self.x_eps, self.h_eps = iters, hc_eps, x_eps, h_eps
        self.hidden, self.width = hidden, n * hidden
        put = lambda t, dt: ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=DRAM
        )
        fn_pad = torch.zeros(n * hidden, TILE)
        fn_pad[:, :N_MIX] = fn_t.float()
        # [16384, 32]: page = K tile (s * 128 + c); bf16 holds the checkpoint values exactly (fp32 = today's copy)
        self.fn = put(fn_pad.to(torch.bfloat16) if fn_dtype == BF16 else fn_pad, fn_dtype)
        sb = torch.zeros(2 * TILE, TILE)
        sb[0, :N_MIX] = scale.float()
        sb[TILE, :N_MIX] = bias.float()
        self.sb = put(sb, FP32)  # page 0 row 0 = scale, page 1 row 0 = bias
        wt = torch.zeros(TILE, hidden)
        wt[0] = norm_w.float()
        self.w = put(wt, FP32)  # page c row 0 = w[32c : 32c + 32]
        self.dbg = put(torch.zeros(N_TAPS * TILE, TILE), FP32)
        self.host = dict(fn_t=fn_t.float(), scale=scale.float(), bias=bias.float(), w=norm_w.float())
        self._placement()

    # ---------------------------------------------------------------------------------------------- placement
    def _noc(self, x, y):
        c = self.device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return c.x, c.y

    def _placement(self):
        """Virtual positions (cx, cy) of the 64 K cores + ROOT -> logical cores (self.place); core-range sets and NoC
        coordinates of every role. The kernels only see NoC coordinates (runtime args), so any placement works."""
        cr = lambda x0, y0, x1, y1: ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))
        if self.placement is None:
            grid = self.device.compute_with_storage_grid_size()
            assert (
                grid.x > ROOT[0] and grid.y >= GRID_Y
            ), f"mhc_pre needs a {ROOT[0] + 1} x {GRID_Y} worker grid, got {grid}"
            self.place = {(x, y): (x, y) for x in range(GRID_X) for y in range(GRID_Y)}
            self.place[ROOT] = ROOT
            self.leaders = ttnn.CoreRangeSet([cr(0, 0, GRID_X - 1, 0)])
            self.members = ttnn.CoreRangeSet([cr(0, 1, GRID_X - 1, GRID_Y - 1)])
            self.kcores = ttnn.CoreRangeSet([cr(0, 0, GRID_X - 1, GRID_Y - 1)])
            self.root = ttnn.CoreRangeSet([cr(ROOT[0], ROOT[1], ROOT[0], ROOT[1])])
            self.all = ttnn.CoreRangeSet([cr(0, 0, GRID_X - 1, GRID_Y - 1), cr(ROOT[0], ROOT[1], ROOT[0], ROOT[1])])
        else:
            ks, root = self.placement["k"], self.placement["root"]
            assert len(ks) == GRID_X * GRID_Y, len(ks)
            self.place = {(k // GRID_Y, k % GRID_Y): (int(c.x), int(c.y)) for k, c in enumerate(ks)}
            self.place[ROOT] = (int(root.x), int(root.y))
            sel = lambda pred: merged_ranges([self.place[v] for v in self.place if v != ROOT and pred(v)])
            self.leaders = sel(lambda v: v[1] == 0)
            self.members = sel(lambda v: v[1] != 0)
            self.kcores = sel(lambda v: True)
            self.root = merged_ranges([self.place[ROOT]])
            self.all = merged_ranges(list(self.place.values()))
        self.noc = {v: self._noc(*xy) for v, xy in self.place.items()}

    # ---------------------------------------------------------------------------------------------- descriptors
    @staticmethod
    def _acc(t) -> list:
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    @staticmethod
    def _cb(index, dtype, page, pages, cores, alias=None):
        fmts = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
        if alias is not None:
            fmts.append(ttnn.CBFormatDescriptor(buffer_index=alias, data_format=dtype, page_size=page))
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores, format_descriptors=fmts)

    def _cbs(self, hn_dtype):
        t32, t16 = 4096, 2048
        hn_page = t32 if hn_dtype == FP32 else t16
        return [
            # remote-written: defined on every core (a CB has ONE address across its core set; the senders resolve the
            # receivers' address with their own get_write_ptr)
            self._cb(CB["GATHER"], FP32, t32, 16, self.all),
            self._cb(CB["MSG"], FP32, 512, 1, self.all),
            # K cores
            self._cb(CB["X"], FP32, t32, 8, self.kcores, alias=CB["X32"]),
            self._cb(CB["FN"], self.fn_dtype, t16 if self.fn_dtype == BF16 else t32, 8, self.kcores),
            self._cb(CB["W"], FP32, t32, 2, self.kcores),
            self._cb(CB["PART"], FP32, t32, 2, self.kcores),
            self._cb(CB["PRE"], FP32, t32, 4, self.kcores),
            self._cb(CB["H"], FP32, t32, 2, self.kcores),
            self._cb(CB["S2"], FP32, t32, 1, self.kcores),
            self._cb(CB["INV"], FP32, t32, 1, self.kcores),
            self._cb(CB["OUT"], hn_dtype, hn_page, 2, self.kcores),
            self._cb(CB["XSTK"], FP32, t32, 1, self.kcores),
            self._cb(CB["GSUM"], FP32, t32, 2, self.kcores),
            # root
            self._cb(CB["RED"], FP32, t32, 1, self.root),
            self._cb(CB["PLANE"], FP32, t32, 2 if self._pre_split() else 1, self.root),
            # GLM53_MHC_PRE_EARLY: page 0 = the pre planes (sent before the Sinkhorn), page 1 = every coefficient
            self._cb(CB["COEF"], FP32, t32, 2 if os.environ.get("GLM53_MHC_PRE_EARLY") == "1" else 1, self.root),
            self._cb(CB["SB"], FP32, t32, 2, self.root),
            self._cb(CB["INVR"], FP32, t32, 1, self.root),
            self._cb(CB["SCR"], FP32, t32, 1, self.root),
            self._cb(CB["DBG"], FP32, t32, 2, self.root),
        ]

    @staticmethod
    def _rt(per_core: dict):
        rt = ttnn.RuntimeArgs()
        for (x, y), args in per_core.items():
            rt[x][y] = [int(a) for a in args]
        return rt

    def _kernel(self, name, cores, ct, rt, defines, config):
        if getattr(self, "_collect", None) is not None:  # parts(): the pieces of every kernel role
            self._collect.append(dict(name=name, cores=cores, ct=list(ct), rt={k: list(v) for k, v in rt.items()},
                                      defines=list(defines), config=config))  # fmt: skip
        return ttnn.KernelDescriptor(
            kernel_source=os.path.join(KERNEL_DIR, name),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            defines=defines,
            runtime_args=self._rt(rt),
            config=config,
        )

    @staticmethod
    def _pre_split() -> bool:
        return os.environ.get("GLM53_MHC_PRE_SPLIT") == "1" and os.environ.get("GLM53_MHC_PRE_EARLY") == "1"

    def _compute_config(self):
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in FP32_COPY_CBS:
            # GLM53_MHC_FPU_COLLAPSE=1: CB_PRE holds diag(pre_s), a matmul operand (through SrcB, not copied)
            if cb == CB["PRE"] and os.environ.get("GLM53_MHC_FPU_COLLAPSE") == "1":
                continue
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode(modes)
        return cfg

    def parts(self, *args, **kwargs):
        """descriptor()'s pieces for megakernel composition: ([role dicts: name, cores, ct, rt {core: args}, defines,
        config], cbs, semaphores). Role keys: (name, MHC_K / MHC_LEADER / MHC_ROOT in the defines)."""
        self._collect = []
        try:
            desc = self.descriptor(*args, **kwargs)
            return self._collect, list(desc.cbs), list(desc.semaphores)
        finally:
            self._collect = None

    def descriptor(self, x, hn, post, comb, x_out, coef, *, t, x_mode, xout_mode, pc_mode, debug=False, probe=False,
                   signal=None, mk_probe=None, defer=None, coef_plane=False):  # fmt: skip
        """The ProgramDescriptor of one call (addresses of the given tensors as runtime args). signal = (semaphore id,
        [(x, y) NoC coordinates]) (megakernel): every K core increments that semaphore on each listed core once its hn
        tiles are written."""
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        stream_cfg = dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
        sync_cfg = dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)
        extra = ([("DEBUG_TAPS", "1")] if debug else []) + ([("MHC_PLANE_PROBE", "1")] if probe else [])
        # Stage 14 glue trims (env, read per program): GLM53_MHC_ROWREAD=1 = read only the decode rows of x / w
        # (bitwise equal; measured 22:39Z: not on the critical path, the x stream lands 2.4 us later -> off by default)
        if os.environ.get("GLM53_MHC_ROWREAD") == "1":
            extra = extra + [("MHC_ROWREAD", "1")]
        # GLM53_MHC_SYNC_MARKS=1 (probe runs): K / leader NCRISC marks of round 1 (partial ready, leader gathered,
        # group sum sent, pre message in) ahead of the role's later marks
        if os.environ.get("GLM53_MHC_SYNC_MARKS") == "1":
            extra = extra + [("MHC_SYNC_MARKS", "1")]
        # Stage 14c GLM53_MHC_MODE containing "dz": the sync RISC's L1 zero fills as NoC zero writes (not CPU loops)
        if "dz" in os.environ.get("GLM53_MHC_MODE", "").split(","):
            extra = extra + [("MHC_DMA_ZERO", "1")]
        # Stage 14c "cp" (t > 4 = stream-major post / comb, megakernel with mhc_post v2 "pv2"): the root writes its
        # Sinkhorn coefficient page AS IS (plane layout) to the coefficient tensor — no post / comb tiles (mhc_post
        # reads only the coefficient tile; they stay unwritten), no per-token compact-row stores; the v2 reader
        # (MPOST_COEF_PLANE) re-lays the planes
        modes = os.environ.get("GLM53_MHC_MODE", "").split(",")
        # coef_plane: the caller's mhc_post consumes the plane layout (MkLayer / MkSparseLayer with fused v2) and nothing
        # else reads post / comb; any other caller keeps the row-layout coefficient tile and the post / comb tiles
        if coef_plane and "cp" in modes and "pv2" in modes and pc_mode == PC_STREAM_MAJOR:
            extra = extra + [("MHC_COEF_PLANE", "1")]
        # GLM53_MHC_PRE_EARLY=1: the root sends pre (sigmoid(z[0:4]) + eps) to the K cores before the 20-iteration
        # Sinkhorn of comb (only mhc_post needs comb): the collapse h = sum_s pre_s x_s starts ~8 us earlier
        if os.environ.get("GLM53_MHC_PRE_EARLY") == "1":
            extra = extra + [("MHC_PRE_EARLY", "1")]
        # GLM53_MHC_FPU_COLLAPSE=1 (Stage 7a, B=1 probe 2026-10-20: 'pre in -> hn packed' = 15.3 us of SFPU tile ops):
        # the K cores' collapse h_j = sum_s diag(pre_s) @ x_{s,j} as 4 accumulated FPU matmuls per column (the sync RISC
        # writes pre_s on the diagonal of its zeroed tile); tf32 operands / fp32 sums, below hn's bf16 rounding
        if os.environ.get("GLM53_MHC_FPU_COLLAPSE") == "1":
            extra = extra + [("MHC_FPU_COLLAPSE", "1")]
        # GLM53_MHC_PRE_SPLIT=1 (with GLM53_MHC_PRE_EARLY; 7a B=32 probe: z -> pre planes 14.9 us = the root RISC's 24 x T
        # element relayout): the pre planes are relaid alone first and the pre message goes out before the full relayout
        if self._pre_split():
            extra = extra + [("MHC_PRE_SPLIT", "1")]
        # 7a GLM53_MHC_MODE containing "fr" (B=32 probe 2026-10-21: the root's full 24 x T relayout + Sinkhorn = 22.6 us
        # vs 7.5 at B=1, on the hn / hn2 critical path): the relayout walks z token-major with plain (cacheable,
        # schedulable) loads / stores and fences once before the push, instead of volatile per-element accesses
        if "fr" in modes:
            extra = extra + [("MHC_FAST_RELAYOUT", "1")]
        # GLM53_MHC_DIAG=nofinish,nos2 (probe runs only, WRONG numerics): bisect the K cores' post-pre compute
        for d_ in os.environ.get("GLM53_MHC_DIAG", "").split(","):
            if d_ in ("nofinish", "nos2"):
                extra = extra + [(f"MHC_DIAG_{d_.upper()}", "1")]
        # defer = (invh tensor [1,1,32,32] fp32, semaphore id, [(x, y)] readers) (Stage 14, megakernel only): hn = h * w
        # without 1/rms(h); the root writes 1/rms(h) to invh and signals the readers, which scale their outputs
        if defer is not None:
            extra = extra + [("MHC_DEFER_INV", "1"), ("MK_INVH_SEM", str(defer[1]))]
            # GLM53_MHC_HN_FIRST=1 (Stage 14, with the deferral only): the K cores (leaders included) publish hn
            # before their S2 / round-2 work — the leaders' hn no longer waits for the round-2 group sums
            if os.environ.get("GLM53_MHC_HN_FIRST") == "1":
                extra = extra + [("MHC_HN_FIRST", "1")]
        xo = x_out if x_out is not None else self.dbg
        dbg = self.dbg.buffer_address()

        # stream (BRISC / NoC0)
        k_stream_ct = [x_mode, xout_mode] + self._acc(x) + self._acc(self.fn) + self._acc(self.w) + self._acc(xo)
        k_stream_ct += self._acc(hn)
        k_stream_rt = {
            self.place[(cx, cy)]: [
                x.buffer_address(), self.fn.buffer_address(), self.w.buffer_address(), xo.buffer_address(),
                hn.buffer_address(), 2 * (GRID_Y * cx + cy), t,
            ]
            for cx in range(GRID_X) for cy in range(GRID_Y)
        }  # fmt: skip
        root_stream = self._kernel(
            "stream.cpp", self.root, self._acc(self.sb), {self.place[ROOT]: [self.sb.buffer_address()]},
            [("MHC_ROOT", "1")] + extra, stream_cfg,
        )  # fmt: skip
        k_defines = [("MHC_K", "1")] + extra
        if signal is not None:
            sem_id, targets = signal
            k_defines += [("MK_SIGNAL_SEM_ID", str(sem_id)), ("MK_SIGNAL_N", str(len(targets)))]
            flat = [v for xy in targets for v in xy]
            k_stream_rt = {core: args + flat for core, args in k_stream_rt.items()}
        if mk_probe is not None:  # megakernel phase probe: (probe tensor address, first slot); K core k -> slot + k
            paddr, pslot = mk_probe
            nargs = len(next(iter(k_stream_rt.values())))
            k_defines += [("MK_PROBE_ARG", str(nargs)), ("MK_PROBE_CB", "31")]
            k_stream_rt = {
                self.place[(cx, cy)]: k_stream_rt[self.place[(cx, cy)]] + [paddr, pslot + GRID_Y * cx + cy]
                for cx in range(GRID_X) for cy in range(GRID_Y)
            }  # fmt: skip
        k_stream = self._kernel("stream.cpp", self.kcores, k_stream_ct, k_stream_rt, k_defines, stream_cfg)

        # sync (NCRISC / NoC1)
        rx, ry = self.noc[ROOT]
        member_rt, leader_rt = {}, {}
        for cx in range(GRID_X):
            lx, ly = self.noc[(cx, 0)]
            for cy in range(GRID_Y):
                base = [t, lx, ly, cy, int(cx == 0 and cy == 0), dbg]
                if cy == 0:
                    members = [v for my in range(1, GRID_Y) for v in self.noc[(cx, my)]]
                    leader_rt[self.place[(cx, cy)]] = base + [rx, ry, cx] + members
                else:
                    member_rt[self.place[(cx, cy)]] = base
        k_sync_ct = [pc_mode] + self._acc(self.dbg)
        leader_sync = self._kernel(
            "sync.cpp", self.leaders, k_sync_ct, leader_rt, [("MHC_K", "1"), ("MHC_LEADER", "1")] + extra, sync_cfg
        )
        member_sync = self._kernel("sync.cpp", self.members, k_sync_ct, member_rt, [("MHC_K", "1")] + extra, sync_cfg)
        root_sync_ct = [pc_mode] + self._acc(post) + self._acc(comb) + self._acc(self.dbg) + self._acc(coef)
        leaders_noc = [v for cx in range(GRID_X) for v in self.noc[(cx, 0)]]
        root_rt = [t, post.buffer_address(), comb.buffer_address(), dbg, coef.buffer_address()] + leaders_noc
        if defer is not None:  # sync.cpp root MHC_DEFER_INV: invh accessor (CT), invh address + the readers (RT)
            root_sync_ct = root_sync_ct + self._acc(defer[0])
            root_rt = root_rt + [defer[0].buffer_address(), len(defer[2])] + [v for xy in defer[2] for v in xy]
        root_defines = [("MHC_ROOT", "1")] + extra
        if mk_probe is not None:  # root's NCRISC -> slot + 64
            root_defines += [("MK_PROBE_ARG", str(len(root_rt))), ("MK_PROBE_CB", "31")]
            root_rt = root_rt + [mk_probe[0], mk_probe[1] + GRID_X * GRID_Y]
        root_sync = self._kernel(
            "sync.cpp", self.root, root_sync_ct, {self.place[ROOT]: root_rt}, root_defines, sync_cfg
        )

        # compute
        compute_ct = [
            self.iters, f32_bits(self.hc_eps), f32_bits(1.0 / self.width), f32_bits(self.x_eps),
            f32_bits(1.0 / self.hidden), f32_bits(self.h_eps),
        ]  # fmt: skip
        cc = self._compute_config()
        computes = [
            self._kernel("compute.cpp", self.members, compute_ct, {}, [("MHC_K", "1")] + extra, cc),
            self._kernel(
                "compute.cpp", self.leaders, compute_ct, {}, [("MHC_K", "1"), ("MHC_LEADER", "1")] + extra, cc
            ),
            self._kernel("compute.cpp", self.root, compute_ct, {}, [("MHC_ROOT", "1")] + extra, cc),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all, initial_value=0) for i in range(SEMAPHORES)]
        return ttnn.ProgramDescriptor(
            kernels=[k_stream, root_stream, leader_sync, member_sync, root_sync, *computes],
            semaphores=sems,
            cbs=self._cbs(hn.dtype),
        )

    # ---------------------------------------------------------------------------------------------- call
    def __call__(
        self, x, *, t: int, x_mode: int, xout_mode: int = XOUT_NONE, pc_mode: int, hn_dtype=BF16, debug=False,
        probe=False, out_memory_config=L1,
    ) -> MhcPreOutputs:  # fmt: skip
        """x: flat [1,1,T,16384] / stream-major [1,4,T,4096] (x_mode=X_FLAT) or stacked [T,1,4,4096] (X_STACKED)
        fp32 TILE, DRAM or L1 interleaved. Allocates the outputs (out_memory_config) and runs the program."""
        if not 1 <= t <= TILE:
            raise ValueError(f"mhc_pre takes 1..32 decode rows, got {t}")
        if x.dtype != FP32 or x.layout != ttnn.TILE_LAYOUT or x.memory_config().is_sharded():
            raise ValueError(f"mhc_pre needs an interleaved fp32 TILE residual, got {x.dtype} {x.layout}")
        shape = tuple(x.shape)
        if x_mode == X_FLAT:
            ok = shape in ((1, 1, t, N_STREAMS * self.hidden), (1, N_STREAMS, t, self.hidden))
        elif x_mode == X_STACKED:
            ok = t <= 4 and shape == (t, 1, N_STREAMS, self.hidden)
        else:
            ok = False
        if not ok:
            raise ValueError(f"mhc_pre x_mode {x_mode}, t={t}: unexpected residual shape {shape}")
        dev = self.device
        mk = lambda shape, dt: ttnn.empty(shape, dt, ttnn.TILE_LAYOUT, dev, out_memory_config)
        hn = mk((1, 1, t, self.hidden), hn_dtype)
        if pc_mode == PC_STACKED:
            post, comb = mk((t, 1, N_STREAMS, 1), FP32), mk((t, 1, N_STREAMS, N_STREAMS), FP32)
        else:
            post, comb = mk((1, N_STREAMS, t, 1), FP32), mk((1, 1, t, N_STREAMS * N_STREAMS), FP32)
        x_out = None
        if xout_mode == XOUT_STACKED:
            x_out = mk((t, 1, N_STREAMS, self.hidden), FP32)
        elif xout_mode == XOUT_FLAT:
            x_out = mk((1, N_STREAMS, t, self.hidden), FP32)
        coef = mk((1, 1, TILE, TILE), FP32)
        desc = self.descriptor(
            x, hn, post, comb, x_out, coef, t=t, x_mode=x_mode, xout_mode=xout_mode, pc_mode=pc_mode, debug=debug,
            probe=probe,
        )  # fmt: skip
        io = [x, self.fn, self.w, self.sb, self.dbg, post, comb, coef] + ([x_out] if x_out is not None else []) + [hn]
        ttnn.generic_op(io, desc)
        return MhcPreOutputs(hn=hn, post=post, comb=comb, x_out=x_out, coef=coef)
