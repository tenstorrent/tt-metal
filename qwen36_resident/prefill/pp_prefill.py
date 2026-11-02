# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel prefill of Qwen3.6-27B on QB2 (1 x 4 mesh), feeding the resident decode.

Prefill is compute-bound, and tensor parallelism over the 4 chips makes it communication-bound (every
half-layer all-reduces a [chunk, 5120] activation over ethernet). Here chip p instead holds the full
weights of layers [p * Ls, (p + 1) * Ls) (Ls = layers / chips) and the prompt flows through the chips in
chunks of C tokens: in tick t chip p runs its Ls layers on chunk t - p, then hands its output to chip
p + 1. With Ls a multiple of the attention interval every stage has the same layer schedule, so one op
sequence runs on all chips at once (each on its own weights, state and chunk); a tick is the same
program every time, and everything that differs per chip and per tick (positions, which KV blocks to
write, whether the chunk is real) is data in small control tensors written before the tick.

Chunks outside a chip's range (pipeline fill and drain) run on the same program but leave the state
alone: their GDN gates are masked to g = 0, beta = 0 (the delta rule is then the identity on the state),
the conv carry update is skipped, and their KV writes go to scratch blocks. The same masks cover the
padding rows of the last chunk.

State per chip: GDN recurrent states fp32 [48, 128, 128] and conv carries (last 3 conv inputs), and paged
bf8 KV caches of its attention layers. export_state() converts them into the resident decode's
TP-sharded State.

run_batch() streams several prompts through one pass (chip p starts the next prompt's first chunk in the tick after
the previous prompt's last one) and hands each chip's part of a prompt off between ticks, so the pipeline fills and
drains once per call: the multi-user rows engine's way to prefill a gang of prompts.
"""

import math
import os

import torch

import ttnn
from qwen36_resident import PKG_DIR
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from qwen36_resident.prefill import pp_weights as PW
from qwen36_resident.prefill.smm import SmallMatmul
from qwen36_resident.common import EPS, HIDDEN, INTER, device_zeros, zero_dram
from qwen36_resident.model import (
    CONV_K,
    DK,
    DV,
    HD,
    NK,
    NKV,
    NQ,
    NV,
    ROPE_THETA,
    ROT,
    State,
    TraceL1Guard,
    is_attn,
    l1_layout,
    quantize_rows,
)

KDIR = f"{PKG_DIR}/prefill/kernels/"
# 1: reset() uploads host zeros (~65 MB / chip) instead of zero-filling on device (A/B)
HOST_ZERO = os.environ.get("RESIDENT_PP_HOST_ZERO", "0") == "1"
QD, VD = NK * DK, NV * DV  # 2048, 6144
CONV_CH = 2 * QD + VD  # 10240
Z0, A0 = CONV_CH, CONV_CH + VD  # z | a | b columns
GDN_COLS = 16512  # q | k | v | z | a | b (16480) padded to 12 x 43 tiles
ATTN_COLS = 2 * NQ * HD + 2 * NKV * HD  # q | k | v | gate = 14336
TILE = 32
MM_SHAPES = dict(
    gdn=(HIDDEN, GDN_COLS), attn=(HIDDEN, ATTN_COLS), out=(VD, HIDDEN), gu=(HIDDEN, INTER), down=(INTER, HIDDEN)
)
# (projection, chunk) -> grid x, grid y, in0_block_w, out_subblock h, w: the fastest of a sweep on QB2
# (LoFi, bf8 / bf4 weights; 250-290 TFLOPS at chunk 1024, 195-240 at 512)
MM_BEST = {
    ("gdn", 1024): (13, 10, 8, 1, 4),
    ("attn", 1024): (12, 8, 8, 4, 2),
    ("out", 1024): (10, 8, 16, 1, 4),
    ("gu", 1024): (13, 10, 10, 4, 2),
    ("down", 1024): (10, 10, 16, 1, 4),
    ("gdn", 512): ("1d", 13, 8, 4, 1, 5),
    ("attn", 512): ("1d", 13, 8, 4, 1, 5),
    ("out", 512): (13, 8, 16, 1, 1),
    ("gu", 512): ("1d", 13, 10, 8, 1, 5),
    ("down", 512): (13, 8, 16, 1, 1),
    # chunks <= 256: weight-read bound, in0 multicast to the whole grid ("1d": grid x, y, in0_block_w,
    # out subblock h, w; these fuse gate's SiLU, ttnn's own choice "auto" does not)
    ("gdn", 256): ("1d", 13, 10, 16, 2, 4),
    ("attn", 256): ("1d", 12, 10, 16, 2, 4),
    ("out", 256): ("1d", 13, 5, 16, 2, 3),
    ("gu", 256): ("1d", 8, 8, 4, 2, 3),
    ("down", 256): ("1d", 13, 5, 16, 2, 3),
    ("gdn", 128): ("1d", 13, 8, 4, 1, 5),
    ("attn", 128): ("1d", 12, 10, 16, 2, 4),
    ("out", 128): ("1d", 13, 5, 8, 2, 3),
    ("gu", 128): ("1d", 13, 10, 8, 1, 5),
    ("down", 128): ("1d", 13, 5, 8, 2, 3),
}
# the same on the P300 die (11 x 10 = 110 cores), tests/sweep_pp_matmul.py 2026-10-29: per (projection, chunk) the
# fastest config whose output is bit-identical to the config used before (MM_BEST where it fits, else the qwen36
# heuristic; gate's fused SiLU included), so the prefill's numbers do not change. ms: before -> after on one chip
MM_BEST_11x10 = {
    ("gdn", 1024): (10, 8, 4, 2, 4),  # 0.665 -> 0.645
    ("attn", 1024): (10, 8, 4, 2, 3),  # 0.691 -> 0.569
    ("out", 1024): (11, 8, 16, 2, 3),  # 0.230 -> 0.222
    ("gu", 1024): (11, 8, 4, 4, 2),  # 0.817 -> 0.669
    ("down", 1024): (11, 8, 16, 2, 3),  # 0.548 -> 0.523
    ("gdn", 512): (11, 8, 4, 2, 1),  # 0.465 -> 0.444
    ("attn", 512): ("1d", 10, 9, 4, 8, 1),  # 0.487 -> 0.340
    ("out", 512): (11, 8, 4, 2, 3),  # 0.201 -> 0.192
    ("gu", 512): (11, 8, 4, 1, 5),  # 0.537 -> 0.415
    ("down", 512): (11, 6, 4, 1, 5),  # 0.520 -> 0.489
    ("gdn", 256): (11, 8, 4, 1, 1),  # 0.424 -> 0.407
    ("attn", 256): ("1d", 10, 9, 4, 8, 1),  # 0.447 -> 0.304
    ("out", 256): ("1d", 10, 8, 4, 4, 2),  # 0.187 -> 0.171
    ("gu", 256): ("1d", 8, 8, 4, 2, 3),  # MM_BEST's (fits): 0.260, nothing identical was faster
    ("down", 256): ("1d", 10, 8, 4, 4, 2),  # 0.488 -> 0.437
    ("out", 128): ("1d", 10, 8, 4, 4, 2),  # 0.366 -> 0.120
    ("down", 128): ("1d", 10, 8, 4, 4, 2),  # 0.468 -> 0.301
}
MM_BEST_BY_GRID = {(13, 10): MM_BEST, (11, 10): MM_BEST_11x10}


# a 64-layer tick at chunk C takes about base + slope * (mean context in K tokens) ms (QB2, traced: the
# attention part grows with the context)
# chunks up to this many tokens are weight-read bound: their big projections use SmallMatmul (kernels/smm_*)
SMM_MAX_C = 128
SMM_SHAPES = dict(
    gu=(HIDDEN, INTER, ttnn.bfloat4_b), gdn=(HIDDEN, GDN_COLS, ttnn.bfloat8_b), attn=(HIDDEN, ATTN_COLS, ttnn.bfloat8_b)
)
TICK_MS = {128: (22.4, 1.6), 256: (32.5, 2.9), 512: (45.5, 3.2), 1024: (81.0, 3.0)}
# the attention layers' chunked SDPA: HiFi2 with the approximate exp (the default since 2026-11-01, the user's decision;
# was HiFi4, exact exp). Softmax exp dominates the long-context ticks: SDPA -50% at 32K-130K, TTFT 4.35 -> 3.62 s at 32K,
# 32.4 -> 21.9 s at 131K, 111 -> 78 s at 260K, with HF teacher-forced top-1 0.9375 -> 0.9413 and top-1 agreement / KL
# against the ttnn TP prefill at 32K / 131K / 260K within noise (0.915 / 0.891 / 0.884 -> 0.922 / 0.884 / 0.891, KL
# 0.028 / 0.032 / 0.037 either way; profiles/resident/prefill/acc1). RESIDENT_PP_SDPA_FIDELITY=HiFi4
# RESIDENT_PP_SDPA_EXP_APPROX=0 restores the old numerics (A/B). RESIDENT_PP_SDPA_KCHUNK=256: opt-in, not bit-identical,
# no gain with the approximate exp
SDPA_FIDELITY = os.environ.get("RESIDENT_PP_SDPA_FIDELITY", "HiFi2")
SDPA_KCHUNK = int(os.environ.get("RESIDENT_PP_SDPA_KCHUNK", "128"))
SDPA_EXP_APPROX = os.environ.get("RESIDENT_PP_SDPA_EXP_APPROX", "1") == "1"
# the handoff's copy engines per chip: one core per fabric link (kernels/handoff.cpp on cores (0, 0), (1, 0), ...), each
# with its share of the jobs (split by bytes) and its own semaphore count
HANDOFF_LINKS = int(os.environ.get("RESIDENT_PP_HANDOFF_LINKS", "2"))
# run_batch: "op" gives a stage's GDN states what the sequential path's masked ticks do to them (see run_batch), "off"
# skips it (not bit-identical to run + handoff; A/B only)
BATCH_STATE_ROUND = os.environ.get("RESIDENT_PP_BATCH_STATE_ROUND", "op")
# run_batch: prompts whose conv carries a pass can stage before it scatters their histories (~7.9 MB per chip each; at
# least the chips: a pass has that many prompts in flight); 0: no run_batch
BATCH_STAGES = int(os.environ.get("RESIDENT_PP_BATCH_STAGES", "8"))


def rope_cos_sin(positions):
    """[T, ROT] cos / sin (both halves) of the rotated dims at the positions"""
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, ROT, 2, dtype=torch.float64) / ROT)
    ang = positions.double()[:, None] * inv[None]
    cos, sin = torch.cos(ang).float(), torch.sin(ang).float()
    return torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)


def rotate_half_matrix():
    """R with (a | b) @ R = (-b | a) on the ROT rotated dims"""
    h = ROT // 2
    r = torch.zeros(ROT, ROT)
    r[torch.arange(h) + h, torch.arange(h)] = -1.0
    r[torch.arange(h), torch.arange(h) + h] = 1.0
    return r


class PPPrefill:
    def __init__(
        self, mesh, ck, layers, interval, chunk=1024, max_len=None, block=64, mm_cols=10, taps=None, tap_window=2048
    ):
        """chunk: one chunk size, or several; run() picks the fastest for each prompt (one trace per size,
        sharing the weights and the state).
        taps: layers whose output (the residual after the layer) the prefill keeps for the last tap_window
        positions (read_taps; the DFlash2 drafter's context): every chip writes its x after its stage-layers
        j = layer % Ls into a ring of tap_window + C positions (a paged fill per tick), the tap's chip holds it
        (tap_rows on device, read_taps on the host)"""
        self.mesh, self.n = mesh, mesh.get_num_devices()
        self.taps = list(taps or [])  # (before the conv taps' helper below takes the name)
        n = self.n
        assert layers % n == 0 and (layers // n) % interval == 0, "every stage needs the same layer schedule"
        self.chunks = sorted(chunk) if isinstance(chunk, (list, tuple)) else [chunk]
        self.layers, self.interval, self.Ls, self.block = layers, interval, layers // n, block
        self.mm_cols = mm_cols
        C = max(self.chunks)
        # whole chunks, and whole 8-block rows of the KV page table (SDPA reads 32 B page-table rows)
        step = math.lcm(C, 8 * block)
        self.max_len = -(-(max_len or 4 * C) // step) * step
        assert all(c % block == 0 and self.max_len % c == 0 for c in self.chunks)
        self.kinds = [is_attn(j, interval) for j in range(self.Ls)]
        shard, rep = ttnn.ShardTensorToMesh(mesh, dim=0), ttnn.ReplicateTensorToMesh(mesh)
        DRAM = ttnn.DRAM_MEMORY_CONFIG

        def dev(t, dtype, layout=ttnn.TILE_LAYOUT, mapper=shard):
            out = ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper, memory_config=DRAM)
            ttnn.synchronize_device(mesh)
            return out

        self._dev = dev
        self.mm = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        self.hifi = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, SDPA_FIDELITY),
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # chunks up to SMM_MAX_C stream the big projections through SmallMatmul, from their own
        # bank-major copies of those weights (the layout does not depend on the chunk)
        self.small = [c for c in self.chunks if c <= SMM_MAX_C]
        if self.small:
            m = min(self.small)
            self.smm_layout = {key: SmallMatmul(mesh, m, K, N, dt) for key, (K, N, dt) in SMM_SHAPES.items()}

        # ---- weights: stage-layer j of chip p is layer p * Ls + j
        self.w = []
        for j in range(self.Ls):
            ws = [PW.layer(ck, p * self.Ls + j, interval) for p in range(n)]
            stack = lambda key, f: torch.stack([f(w[key]) for w in ws])[:, None]
            e = dict(
                G=dev(stack("mlp", lambda m: m["G"]), ttnn.bfloat4_b),
                U=dev(stack("mlp", lambda m: m["U"]), ttnn.bfloat4_b),
                D=dev(stack("mlp", lambda m: m["D"]), ttnn.bfloat8_b),
                out=dev(stack("mixer", lambda m: m["out"]), ttnn.bfloat8_b),
            )
            if self.small:
                e["G_s"] = self.smm_layout["gu"].weights(stack("mlp", lambda m: m["G"])[:, 0])
                e["U_s"] = self.smm_layout["gu"].weights(stack("mlp", lambda m: m["U"])[:, 0])
            if self.kinds[j]:
                e["W"] = dev(stack("mixer", lambda m: m["W"]), ttnn.bfloat8_b)
                if self.small:
                    e["W_s"] = self.smm_layout["attn"].weights(stack("mixer", lambda m: m["W"])[:, 0])
                row = lambda v: v.reshape(1, -1)
                e["wq"] = dev(stack("mixer", lambda m: row(m["wq"] / math.sqrt(HD))), ttnn.bfloat16)
                e["wk"] = dev(stack("mixer", lambda m: row(m["wk"])), ttnn.bfloat16)
            else:

                def padded(m):
                    W = torch.zeros(HIDDEN, GDN_COLS, dtype=torch.bfloat16)
                    W[:, : m["W"].shape[1]] = m["W"]
                    return W

                def taps(m):
                    t = torch.zeros(4 * TILE, CONV_CH)
                    for k in range(CONV_K):
                        t[k * TILE] = m["taps"][:, k]
                    return t

                e["W"] = dev(stack("mixer", padded), ttnn.bfloat8_b)
                if self.small:
                    e["W_s"] = self.smm_layout["gdn"].weights(stack("mixer", padded)[:, 0])
                e["taps"] = dev(stack("mixer", taps), ttnn.bfloat16)

                def row0(v):  # values in row 0 of 32 x 64 (two tiles), broadcast down the rows by the gates
                    t = torch.zeros(TILE, 2 * TILE)
                    t[0, : v.shape[0]] = v
                    return t

                e["dt"] = dev(stack("mixer", lambda m: row0(m["dt"])), ttnn.float32)
                e["neg_a"] = dev(stack("mixer", lambda m: row0(m["neg_a"])), ttnn.float32)
            self.w.append(e)
            del ws

        # ---- state
        self.blocks = self.max_len // block
        self.scratch0 = self.blocks  # C // block scratch blocks after the real ones (largest chunk)
        kv_blocks = self.blocks + C // block
        self.S, self.carry, self.K, self.V = {}, {}, {}, {}
        self.tap_js = sorted({L % self.Ls for L in self.taps})
        self.tap_ring = -(-(tap_window + C) // block)  # blocks of the ring; C // block scratch blocks after it
        self.tap_cache = {
            j: device_zeros(mesh, [self.tap_ring + C // block, 1, block, HIDDEN], ttnn.bfloat16, DRAM)
            for j in self.tap_js
        }
        for j, attn in enumerate(self.kinds):
            if attn:
                # zero-filled on device: at max_len 256K a host image would be 4.3 GB of fp32 per cache
                kv = lambda: device_zeros(mesh, [kv_blocks, NKV, block, HD], ttnn.bfloat8_b, DRAM)
                self.K[j], self.V[j] = kv(), kv()
            else:
                self.S[j] = dev(torch.zeros(n, NV, DK, DV), ttnn.float32)
                self.carry[j] = dev(torch.zeros(n, 1, TILE, CONV_CH), ttnn.bfloat16)

        # run_batch's conv-carry staging buffers: per buffer a carry-shaped tensor per GDN stage-layer. Built here, before
        # any trace capture, because they hold data across tick replays: a DRAM buffer allocated after capture can land
        # on the traced ticks' intermediates, which every replay overwrites
        assert BATCH_STAGES == 0 or BATCH_STAGES >= n, f"RESIDENT_PP_BATCH_STAGES {BATCH_STAGES}: 0 or >= {n}"
        # (and the inputs of run_batch's masked delta-rule step, BATCH_STATE_ROUND "op": one 32-row chunk with g = beta
        # = 0 and finite q / k / v, read at boundaries between tick replays)
        self._masked = dict(
            q=dev(torch.ones(n, TILE, QD), ttnn.bfloat16),
            k=dev(torch.ones(n, TILE, QD), ttnn.bfloat16),
            v=dev(torch.ones(n, TILE, VD), ttnn.bfloat16),
            g=dev(torch.zeros(n, TILE, NV), ttnn.float32),
            beta=dev(torch.zeros(n, TILE, NV), ttnn.float32),
        )
        shape = [1, 1, TILE, CONV_CH]
        self._stages = [
            {j: device_zeros(mesh, shape, ttnn.bfloat16, DRAM) for j in self.carry} for _ in range(BATCH_STAGES)
        ]

        # ---- constants and buffers
        c = TILE  # the delta-rule op's chunk
        consts = [torch.eye(c), torch.tril(torch.ones(c, c)), torch.ones(c, c)]
        ii, jj = torch.arange(32)[:, None], torch.arange(32)[None]
        lo_i, lo_j = ii < 16, jj < 16
        consts.append(torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1))
        self.gdn_consts = [dev(t.reshape(1, 1, *t.shape), ttnn.float32, mapper=rep) for t in consts]
        # the embedding table on every chip (chip 0 looks up the next chunk's rows from its token ids)
        self.embed = dev(
            ck.get("model.language_model.embed_tokens.weight").bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, rep
        )
        # per tick and chip: [valid rows, update] for the conv and the gates
        self.ctrl = dev(torch.zeros(n, 1, 1, 8, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.read_pt = dev(torch.zeros(n, self.blocks, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=SDPA_KCHUNK,
            exp_approx_mode=SDPA_EXP_APPROX,
        )
        self._geo = {}
        for c in self.chunks:
            self._geo[c] = self._geometry(c)
        self._use(C)

    # ------------------------------------------------------------------ chunk sizes
    # what a tick of one chunk size owns: its buffers, programs, matmul configs and trace
    GEOMETRY = (
        "C", "pc", "x_in", "x_last", "x_out", "h_buf", "ids", "q_t", "k_t", "v_t", "g_t", "beta_t", "gate_y",
        "cos", "sin", "fill_pt", "cstart", "_shift_pd", "_shift_sems", "_conv_cores", "_conv_groups",
        "_conv_spec", "_conv_dims", "trace_id", "smm", "smm_out", "tap_pt",
    )  # fmt: skip

    def _geometry(self, C):
        n, dev = self.n, self._dev
        self.C = C
        self.pc = {key: self._mm_config(key) for key in MM_SHAPES}
        self.x_in = dev(torch.zeros(n, 1, C, HIDDEN), ttnn.bfloat16)
        self.x_last = dev(torch.zeros(n, 1, C, HIDDEN), ttnn.bfloat16)  # a stage's output of the last tick
        self.x_out = None
        self.h_buf = dev(torch.zeros(n, 1, C, HIDDEN), ttnn.bfloat16)  # normed residual stream
        self.ids = dev(torch.zeros(n, 1, C, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._shift_program()
        # the delta-rule op's inputs: token-major q, k, v rows and the gates
        self.q_t = dev(torch.zeros(n, C, QD), ttnn.bfloat16)
        self.k_t = dev(torch.zeros(n, C, QD), ttnn.bfloat16)
        self.v_t = dev(torch.zeros(n, C, VD), ttnn.bfloat16)
        self.g_t = dev(torch.zeros(n, C, NV), ttnn.float32)
        self.beta_t = dev(torch.zeros(n, C, NV), ttnn.float32)
        self.gate_y = dev(torch.zeros(n, 1, C, VD), ttnn.bfloat16)
        # per tick and chip: rope, KV blocks
        self.cos = dev(torch.zeros(n, 1, C, ROT), ttnn.bfloat16)
        self.sin = dev(torch.zeros(n, 1, C, ROT), ttnn.bfloat16)
        self.fill_pt = dev(torch.zeros(n, C // self.block, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.cstart = dev(torch.zeros(n, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        # the tap ring's page table: only with taps (a prefill without a drafter does no tap work at all)
        self.tap_pt = (
            dev(torch.zeros(n, C // self.block, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            if self.tap_cache
            else None
        )
        self._conv_program()
        self.trace_id = None
        self.smm = self.smm_out = None
        if C in self.small:
            self.smm = {key: SmallMatmul(self.mesh, C, K, N, dt) for key, (K, N, dt) in SMM_SHAPES.items()}
            cols = dict(g=INTER, u=INTER, gdn=GDN_COLS, attn=ATTN_COLS)
            self.smm_out = {k: dev(torch.zeros(n, 1, C, v), ttnn.bfloat16) for k, v in cols.items()}
        return {k: getattr(self, k) for k in self.GEOMETRY}

    def _use(self, C):
        if getattr(self, "C", None) in self._geo:
            self._geo[self.C].update({k: getattr(self, k) for k in self.GEOMETRY})
        for k, v in self._geo[C].items():
            setattr(self, k, v)

    def pick_chunk(self, T):
        """the chunk size with the shortest estimated prefill: (chunks + n - 1) ticks of TICK_MS at the
        prompt's mean context (sizes without a measurement scale the 512 one)"""

        def est(C):
            base, slope = TICK_MS.get(C, (TICK_MS[512][0] * C / 512, TICK_MS[512][1]))
            return ((T + C - 1) // C + self.n - 1) * (base + slope * T / 2048)

        return min(self.chunks, key=est)

    # ------------------------------------------------------------------ pipeline shift
    def _shift_program(self):
        mesh, n = self.mesh, self.n
        core = ttnn.CoreCoord(0, 0)
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        self._shift_sems = [ttnn.create_global_semaphore(mesh, grid, 0) for _ in range(2)]
        pages = self.C * HIDDEN // (TILE * TILE)
        banks = mesh.dram_grid_size().x
        assert pages % banks == 0
        bank_bytes = pages // banks * TILE * TILE * 2
        cbs = [
            ttnn.CBDescriptor(
                total_size=4 * 4096,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=4096)],
            )
        ]
        pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            coord = ttnn.MeshCoordinate(0, chip)
            rt = ttnn.RuntimeArgs()
            rt[core.x][core.y] = [
                chip,
                n,
                self.x_last.buffer_address(),
                self.x_in.buffer_address(),
                0,
                bank_bytes,
            ] + [ttnn.get_global_semaphore_address(s) for s in self._shift_sems]
            kernel = ttnn.KernelDescriptor(
                kernel_source=KDIR + "shift.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=grid,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
            program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
            args = program.kernels[0].runtime_args[core.x][core.y]
            for nb in (chip + 1, chip - 1):
                if 0 <= nb < n:
                    args.append(1)
                    args.extend(
                        ttnn.setup_fabric_connection(
                            mesh.get_fabric_node_id(coord),
                            mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)),
                            0,
                            program,
                            core,
                        )
                    )
                else:
                    args.append(0)
            pd[ttnn.MeshCoordinateRange(coord, coord)] = program
        self._shift_pd = pd

    # ------------------------------------------------------------------ conv kernel
    def _conv_program(self):
        C, mesh = self.C, self.mesh
        grid = mesh.compute_with_storage_grid_size()
        Ct, Rt, Xw = CONV_CH // TILE, C // TILE, GDN_COLS // TILE
        cores = min(grid.x * grid.y, Ct)
        cols = [(Ct - c + cores - 1) // cores for c in range(cores)]
        f32, bf = (ttnn.float32, 4096), (ttnn.bfloat16, 2048)
        spec = {
            0: (bf, 3),
            1: (bf, 4),
            2: (bf, 1),
            3: (bf, 10),
            4: (f32, 4),
            5: (f32, 4),
            6: (bf, 2),
            7: (bf, 2),
            8: (f32, 1),
            9: (bf, 1),
        }
        self._conv_cores = []
        groups = {}
        for c in range(cores):
            core = ttnn.CoreCoord(c % grid.x, c // grid.x)
            self._conv_cores.append(core)
            groups.setdefault(cols[c], []).append(core)
        self._conv_groups = groups
        self._conv_spec = spec
        self._conv_dims = (Ct, Rt, Xw, cores)

    def _conv(self, p, j):
        """silu(causal conv1d) of the q | k | v columns of p into q_t, k_t, v_t; updates carry[j] per ctrl"""
        Ct, Rt, Xw, cores = self._conv_dims
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self._conv_cores])
        cbs = [
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )
            for i, ((dt, page), cnt) in self._conv_spec.items()
        ]
        rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(self._conv_cores):
            rr[core.x][core.y] = [
                i,
                cores,
                p.buffer_address(),
                self.carry[j].buffer_address(),
                self.w[j]["taps"].buffer_address(),
            ]
            wr[core.x][core.y] = [i, cores] + [
                t.buffer_address() for t in (self.q_t, self.k_t, self.v_t, self.carry[j], self.ctrl)
            ]
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.HiFi4
        compute.fp32_dest_acc_en = True
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "conv_reader.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=all_cores,
                compile_time_args=[Rt, Xw, Ct],
                runtime_args=rr,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KDIR + "conv_writer.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=all_cores,
                compile_time_args=[Rt, Ct, QD // TILE],
                runtime_args=wr,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            ),
        ]
        for cnt, group in self._conv_groups.items():
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "conv_compute.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in group]),
                    compile_time_args=[Rt, cnt],
                    runtime_args=ttnn.RuntimeArgs(),
                    config=compute,
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        mesh_pd = ttnn.MeshProgramDescriptor()
        mesh_pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([p, self.carry[j], self.w[j]["taps"], self.q_t, self.k_t, self.v_t, self.ctrl], mesh_pd)

    def _gates(self, p, j):
        """g = neg_a * softplus(a + dt), beta = sigmoid(b) from the a | b columns of p, both masked to the
        chunk's valid rows (kernels/gates_*.cpp), into g_t and beta_t"""
        mesh, C = self.mesh, self.C
        grid = mesh.compute_with_storage_grid_size()
        Rt = C // TILE
        cores = min(grid.x * grid.y, Rt)
        per = [Rt // cores + (i < Rt % cores) for i in range(cores)]
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        spec = {0: (bf, 3), 1: (bf, 2), 2: (bf, 2), 3: (f32, 2), 4: (f32, 2), 5: (f32, 1), 6: (f32, 1), 7: (f32, 1)}
        spec.update({16: (f32, 2), 17: (f32, 2)})
        cbs = [
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )
            for i, ((dt, page), cnt) in spec.items()
        ]
        w = self.w[j]
        rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        first, by_count = 0, {}
        for c, cnt in zip(coords, per):
            rr[c.x][c.y] = [first, cnt] + [t.buffer_address() for t in (p, w["dt"], w["neg_a"], self.ctrl)]
            wr[c.x][c.y] = [first, cnt, self.g_t.buffer_address(), self.beta_t.buffer_address()]
            by_count.setdefault(cnt, []).append(c)
            first += cnt
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.HiFi4
        compute.fp32_dest_acc_en = True
        kernel = lambda src, cores_, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores_,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        kernels = [
            kernel(
                "gates_reader.cpp",
                all_cores,
                [GDN_COLS // TILE, A0 // TILE],
                rr,
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
            ),
            kernel(
                "gates_writer.cpp", all_cores, [], wr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
            ),
        ]
        for cnt, cs in by_count.items():
            kernels.append(
                kernel(
                    "gates_compute.cpp",
                    ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs]),
                    [cnt],
                    ttnn.RuntimeArgs(),
                    compute,
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        pd = ttnn.MeshProgramDescriptor()
        pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([p, w["dt"], w["neg_a"], self.ctrl, self.g_t, self.beta_t], pd)

    def _gate(self, o, p):
        """rmsnorm over each head's 128 dims of the head-major delta-rule output o, times silu(z) of the
        projection p, as token-major rows (kernels/gnorm_*.cpp) into gate_y"""
        mesh, C = self.mesh, self.C
        grid = mesh.compute_with_storage_grid_size()
        Rt = C // TILE
        groups = Rt * NV
        cores = min(grid.x * grid.y, groups)
        per = [groups // cores + (i < groups % cores) for i in range(cores)]
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        o_fmt = f32 if o.dtype == ttnn.float32 else bf
        spec = {0: (o_fmt, 8), 1: (bf, 8), 2: (bf, 1), 3: (f32, 4), 4: (f32, 1), 5: (f32, 2), 6: (f32, 2), 7: (bf, 4)}
        cbs = [
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )
            for i, ((dt, page), cnt) in spec.items()
        ]
        rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        first, by_count = 0, {}
        for c, cnt in zip(coords, per):
            rr[c.x][c.y] = [first, cnt, o.buffer_address(), p.buffer_address()]
            wr[c.x][c.y] = [first, cnt, self.gate_y.buffer_address()]
            by_count.setdefault(cnt, []).append(c)
            first += cnt
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.HiFi4
        compute.fp32_dest_acc_en = True
        kernel = lambda src, cores_, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores_,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        kernels = [
            kernel(
                "gnorm_reader.cpp",
                all_cores,
                [Rt, NV, GDN_COLS // TILE, Z0 // TILE],
                rr,
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
            ),
            kernel(
                "gnorm_writer.cpp",
                all_cores,
                [NV],
                wr,
                dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
            ),
        ]
        eps_bits = int(torch.tensor([EPS], dtype=torch.float32).view(torch.int32)[0])
        for cnt, cs in by_count.items():
            kernels.append(
                kernel(
                    "gnorm_compute.cpp",
                    ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs]),
                    [cnt, eps_bits],
                    ttnn.RuntimeArgs(),
                    compute,
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        pd = ttnn.MeshProgramDescriptor()
        pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([o, p, self.gate_y], pd)
        return self.gate_y

    def _qkrope(self, t, w):
        """in place on the head-major q or k heads t [1, heads, C, 256]: rmsnorm over each head's dims times w,
        then RoPE on the first ROT dims (kernels/qkrope_*.cpp)"""
        mesh, C = self.mesh, self.C
        grid = mesh.compute_with_storage_grid_size()
        Rt, groups = C // TILE, t.shape[1] * (C // TILE)
        cores = min(grid.x * grid.y, groups)
        per = [groups // cores + (i < groups % cores) for i in range(cores)]
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        spec = {0: (bf, 16), 1: (bf, 8), 2: (bf, 2), 3: (bf, 2), 4: (bf, 1), 5: (f32, 8), 6: (f32, 1), 7: (bf, 8)}
        spec.update({8: (f32, 2), 16: (bf, 16)})
        cbs = [
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )
            for i, ((dt, page), cnt) in spec.items()
        ]
        rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        first, by_count = 0, {}
        for c, cnt in zip(coords, per):
            rr[c.x][c.y] = [first, cnt] + [a.buffer_address() for a in (t, w, self.cos, self.sin)]
            wr[c.x][c.y] = [first, cnt, t.buffer_address()]
            by_count.setdefault(cnt, []).append(c)
            first += cnt
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.HiFi4
        compute.fp32_dest_acc_en = True
        kernel = lambda src, cores_, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores_,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        eps_bits = int(torch.tensor([EPS], dtype=torch.float32).view(torch.int32)[0])
        kernels = [
            kernel(
                "qkrope_reader.cpp",
                all_cores,
                [Rt],
                rr,
                dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
            ),
            kernel(
                "qkrope_writer.cpp", all_cores, [], wr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
            ),
        ]
        for cnt, cs in by_count.items():
            kernels.append(
                kernel(
                    "qkrope_compute.cpp",
                    ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs]),
                    [cnt, eps_bits],
                    ttnn.RuntimeArgs(),
                    compute,
                )
            )
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        pd = ttnn.MeshProgramDescriptor()
        pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([t, w, self.cos, self.sin], pd)
        return t

    def _addnorm(self, x, b, h):
        """x += b in place and h = rmsnorm(x) (kernels/addnorm_*.cpp): core (r, q) owns row tile r, columns
        [q * W, (q + 1) * W) and swaps its part of the row's sum of squares with the row's other cores"""
        mesh, C = self.mesh, self.C
        grid = mesh.compute_with_storage_grid_size()
        Rt, Ht = C // TILE, HIDDEN // TILE
        Q = max(q for q in range(1, Ht + 1) if Ht % q == 0 and (Ht // q) % 2 == 0 and Rt * q <= grid.x * grid.y)
        W = Ht // Q
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(Rt * Q)]
        phys = [mesh.worker_core_from_logical_core(c) for c in coords]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        bf, f32 = (ttnn.bfloat16, 2048), (ttnn.float32, 4096)
        spec = {0: (bf, W), 1: (bf, W), 2: (bf, 1), 3: (f32, 1), 4: (f32, Q), 5: (f32, 1), 6: (bf, W), 16: (bf, W)}
        spec[17] = (bf, W)
        cbs = [
            ttnn.CBDescriptor(
                total_size=cnt * page,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)],
            )
            for i, ((dt, page), cnt) in spec.items()
        ]
        rr, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(coords):
            r, q = divmod(i, Q)
            peers = [v for p in phys[r * Q : (r + 1) * Q] for v in (p.x, p.y)]
            rr[c.x][c.y] = [r, q * W, x.buffer_address(), b.buffer_address()]
            wr[c.x][c.y] = [r, q * W, q, x.buffer_address(), h.buffer_address()] + peers
        compute = ttnn.ComputeConfigDescriptor()
        compute.math_fidelity = ttnn.MathFidelity.HiFi4
        compute.fp32_dest_acc_en = True
        bits = lambda v: int(torch.tensor([v], dtype=torch.float32).view(torch.int32)[0])
        dm = lambda proc, noc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc)
        kernel = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=KDIR + src,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=ct,
            runtime_args=rt,
            config=cfg,
        )
        kernels = [
            kernel("addnorm_reader.cpp", [Ht, W], rr, dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)),
            kernel(
                "addnorm_writer.cpp", [Ht, W, Q], wr, dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
            ),
            kernel("addnorm_compute.cpp", [W, Q, bits(EPS), bits(1.0 / HIDDEN)], ttnn.RuntimeArgs(), compute),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=all_cores, initial_value=0)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        pd = ttnn.MeshProgramDescriptor()
        pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, self.n - 1))] = program
        ttnn.generic_op([x, b, h], pd)
        return h

    # ------------------------------------------------------------------ layers
    def _mm_config(self, key, fused_activation=None):
        """the measured best matmul config of a projection at this chunk for the device grid (MM_BEST_BY_GRID),
        else the qwen36 prefill heuristic"""
        K, N = MM_SHAPES[key]
        grid = self.mesh.compute_with_storage_grid_size()
        table = MM_BEST_BY_GRID.get((grid.x, grid.y), MM_BEST)
        best = table.get((key, self.C))
        if best not in (None, "auto"):
            # a config swept on a larger grid (MM_BEST: 13 x 10) that does not fit takes the heuristic instead
            gx, gy = best[1:3] if best[0] == "1d" else best[:2]
            if gx > grid.x or gy > grid.y:
                best = None
        if best is None:
            try:
                return tpc.create_prefill_mlp_matmul_program_config(
                    self.C, K, N, max_cols=self.mm_cols, fused_activation=fused_activation
                )
            except Exception:
                return None
        if best == "auto":
            return None
        if best[0] == "1d":
            _, gx, gy, bw, sh, sw = best
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=bw,
                out_subblock_h=sh,
                out_subblock_w=sw,
                per_core_M=self.C // TILE,
                per_core_N=math.ceil(N // TILE / (gx * gy)),
                fuse_batch=True,
                fused_activation=fused_activation,
                mcast_in0=True,
            )
        gx, gy, bw, sh, sw = best
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=bw,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=math.ceil(self.C // TILE / gy),
            per_core_N=math.ceil(N // TILE / gx),
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=False,
        )

    def _proj(self, h, j, name):
        """the big projections: G (with SiLU), U, or the mixer's W (gdn / attn), through SmallMatmul at
        short chunks"""
        w = self.w[j]
        key = {"G": "gu", "U": "gu"}.get(name, "attn" if self.kinds[j] else "gdn")
        if self.smm is not None:
            out = {"G": "g", "U": "u"}.get(name, key)
            ws = {"G": "G_s", "U": "U_s"}.get(name, "W_s")
            return self.smm[key](h, w[ws], self.smm_out[out], silu=name == "G")
        return self._linear(h, w[name], key, activation="silu" if name == "G" else None)

    def _linear(self, x, w, key, activation=None):
        pc = self.pc[key]
        if activation is not None and pc is not None:
            if "silu" not in self.pc:
                self.pc["silu"] = self._mm_config(key, fused_activation=ttnn.UnaryOpType.SILU)
            pc, activation = self.pc["silu"], None
        return ttnn.linear(
            x,
            w,
            compute_kernel_config=self.mm,
            program_config=pc,
            activation=activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def _gdn(self, h, j):
        C, w = self.C, self.w[j]
        p = self._proj(h, j, "W")
        # q, k, v rows and the gates go to the delta-rule op token-major: it L2-normalizes q and k per head
        # itself (folding q's scale) and returns o head-major
        self._conv(p, j)
        self._gates(p, j)
        eye, tril, ones, masks = self.gdn_consts
        o, S = ttnn.transformer.chunk_gated_delta_rule(
            self.q_t,
            self.k_t,
            self.v_t,
            self.g_t,
            self.beta_t,
            scale=DK**-0.5,
            initial_state=self.S[j],
            output_final_state=True,
            chunk_size=TILE,
            output_head_major=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        ttnn.copy(S, self.S[j])
        ttnn.deallocate(S)
        # per-head rmsnorm (its weight is folded into out) * silu(z), token-major
        o = self._gate(o, p)
        out = self._linear(o, w["out"], "out")
        return out

    def _attn(self, h, j):
        C, w = self.C, self.w[j]
        p = self._proj(h, j, "W")
        qkv_d = (NQ + 2 * NKV) * HD
        qkv = ttnn.slice(p, (0, 0, 0, 0), (1, 1, C, qkv_d))
        gate = ttnn.slice(p, (0, 0, 0, qkv_d), (1, 1, C, ATTN_COLS))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NQ, num_kv_heads=NKV, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = self._qkrope(q, w["wq"])
        k = self._qkrope(k, w["wk"])
        ttnn.experimental.paged_fill_cache(self.K[j], ttnn.typecast(k, ttnn.bfloat8_b), self.fill_pt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.V[j], ttnn.typecast(v, ttnn.bfloat8_b), self.fill_pt, batch_idx=0)
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.K[j],
            self.V[j],
            self.read_pt,
            chunk_start_idx_tensor=self.cstart,
            scale=1.0,
            program_config=self.sdpa_pc,
            compute_kernel_config=self.hifi,
        )
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        out = self._linear(o, w["out"], "out")
        return out

    def _mlp(self, h, j):
        C, w = self.C, self.w[j]
        g = self._proj(h, j, "G")
        u = self._proj(h, j, "U")
        a = ttnn.multiply(g, u)
        out = self._linear(a, w["D"], "down")
        return out

    def _body(self):
        """one tick: stage inputs (chip 0 the chunk's embeddings, chip p > 0 chip p - 1's last output),
        the stage's layers, the stage output kept for the next tick"""
        emb = ttnn.embedding(self.ids, self.embed, layout=ttnn.TILE_LAYOUT)
        ttnn.copy(ttnn.reshape(emb, (1, 1, self.C, HIDDEN)), self.x_in)
        ttnn.generic_op([self.x_last, self.x_in], self._shift_pd)
        x = self.x_in  # updated in place
        h = ttnn.rms_norm(x, epsilon=EPS)
        for j, attn in enumerate(self.kinds):
            h = self._addnorm(x, self._attn(h, j) if attn else self._gdn(h, j), self.h_buf)
            h = self._addnorm(x, self._mlp(h, j), self.h_buf)
            if j in self.tap_cache:
                ttnn.experimental.paged_fill_cache(self.tap_cache[j], x, self.tap_pt, batch_idx=0)
        ttnn.copy(x, self.x_last)
        return self.x_last

    # ------------------------------------------------------------------ ticks
    def _control(self, t, n_tokens):
        """host control tensors of tick t: chip p runs chunk t - p"""
        chunks = (n_tokens + self.C - 1) // self.C
        return self._control_chips([(t - p, n_tokens) if 0 <= t - p < chunks else None for p in range(self.n)])

    def _control_chips(self, work):
        """host control tensors of a tick: work[p] = (chunk c, the prompt's tokens) chip p runs, or None (idle)"""
        n, C, blk = self.n, self.C, self.block
        ctrl = torch.zeros(n, 1, 1, 8, dtype=torch.int32)
        cos = torch.zeros(n, 1, C, ROT)
        sin = torch.zeros(n, 1, C, ROT)
        fill = torch.zeros(n, C // blk, dtype=torch.int32)
        read = torch.arange(self.blocks, dtype=torch.int32)[None].repeat(n, 1)
        cstart = torch.zeros(n, dtype=torch.int32)
        taps = torch.arange(self.tap_ring, self.tap_ring + C // blk, dtype=torch.int32)[None].repeat(n, 1)
        for p in range(n):
            if work[p] is not None:
                c, n_tokens = work[p]
                if self.tap_cache:
                    taps[p] = torch.arange(c * C // blk, (c + 1) * C // blk) % self.tap_ring
                valid = min(C, n_tokens - c * C)
                ctrl[p, 0, 0, :2] = torch.tensor([valid, 1])
                cos[p, 0], sin[p, 0] = rope_cos_sin(torch.arange(c * C, (c + 1) * C))
                fill[p] = torch.arange(c * C // blk, (c + 1) * C // blk)
                cstart[p] = c * C
            else:
                ctrl[p, 0, 0, :2] = torch.tensor([C, 0])
                fill[p] = torch.arange(self.scratch0, self.scratch0 + C // blk)
                cos[p, 0] = 1.0
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        host = lambda v, dt, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(v, dtype=dt, layout=layout, mesh_mapper=shard)
        writes = [
            (host(ctrl, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.ctrl),
            (host(cos, ttnn.bfloat16), self.cos),
            (host(sin, ttnn.bfloat16), self.sin),
            (host(fill, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.fill_pt),
            (host(read, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.read_pt),
            (host(cstart, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.cstart),
        ]
        if self.tap_cache:  # a drafter's taps: their ring positions this tick (run and run_span alike)
            writes.append((host(taps, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self.tap_pt))
        return writes

    def capture(self):
        """compile the tick body of every chunk size, then record each into a trace (replayed by run()). Call it after
        building the decode model(s) it serves, and again after building another (earlier traces are released): the
        traces replay L1 addresses placed around the decode models' L1 buffers live at capture (model.l1_layout)"""
        for C in self.chunks:
            self._use(C)
            if self.trace_id is not None:
                ttnn.release_trace(self.mesh, self.trace_id)
                self.trace_id = None
        self._l1_layout = l1_layout()
        for C in self.chunks:
            self._use(C)
            for host, dev in self._control(-1, 1):  # every chip idle: KV writes go to the scratch blocks
                ttnn.copy_host_to_device_tensor(host, dev)
            self.x_out = self._body()
            ttnn.synchronize_device(self.mesh)
            self.reset()
            self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.x_out = self._body()
            ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh)
        self._trace_guard = TraceL1Guard(self.mesh, "prefill")  # the L1 its replays assume (model.TraceL1Guard)

    def reset(self):
        """a new prompt: the GDN states and conv carries zero (a stage's first chunk starts from them) and the chunks in
        flight (x_last, every chunk size's) zero. A stage's idle ticks (update flag 0: the gates mask g and beta to 0,
        the conv keeps its carry) leave its states as they are, and stale x_last only ever feeds idle ticks; zeroing
        it keeps them finite. Zero-filled on device (zero_dram) rather than uploaded from the host (~65 MB per chip:
        S 12 x 48 x 128 x 128 fp32, x_last of 4 chunk sizes, carry)"""
        self._use(self.C)
        if not HOST_ZERO:
            zero_dram(
                self.mesh,
                [geo["x_last"] for geo in self._geo.values()] + list(self.S.values()) + list(self.carry.values()),
            )
            return
        if not hasattr(self, "_zeros"):
            # built once: tilizing the zero tensors on the host costs ~0.4 s per reset
            shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)
            z = lambda shape, dtype: ttnn.from_torch(
                torch.zeros(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard
            )
            self._zeros = dict(
                x={C: z((self.n, 1, C, HIDDEN), ttnn.bfloat16) for C in self._geo},
                S=z((self.n, NV, DK, DV), ttnn.float32),
                carry=z((self.n, 1, TILE, CONV_CH), ttnn.bfloat16),
            )
        for C, geo in self._geo.items():
            ttnn.copy_host_to_device_tensor(self._zeros["x"][C], geo["x_last"])
        for j in self.S:
            ttnn.copy_host_to_device_tensor(self._zeros["S"], self.S[j])
            ttnn.copy_host_to_device_tensor(self._zeros["carry"], self.carry[j])
        ttnn.synchronize_device(self.mesh)

    def _check_trace_layout(self):
        """before replaying the captured ticks (run, run_span): the decode models' L1 layout must still be the one at
        capture(), and nothing allocated since may reach the traced programs' CB region (model.TraceL1Guard)"""
        if getattr(self, "_l1_layout", None) is not None:
            if l1_layout() != self._l1_layout:
                raise RuntimeError(
                    "the decode models' L1 layout changed since the prefill traces were captured: their replays could "
                    "overwrite the new model's L1 buffers; call capture() again after building it (model.l1_layout)"
                )
            self._trace_guard.check(full=True)

    def run(self, token_ids, timings=None, chunk=None):
        """prefill the prompt token_ids [T]: one tick per chunk plus the pipeline drain"""
        import time

        T = token_ids.shape[0]
        assert T <= self.max_len
        self._check_trace_layout()
        self._use(chunk or self.pick_chunk(T))
        n, C = self.n, self.C
        chunks = (T + C - 1) // C
        ids = torch.zeros(chunks * C, dtype=torch.int32)
        ids[:T] = token_ids.to(torch.int32)
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        for t in range(chunks + n - 1):
            t0 = time.perf_counter()
            chunk_ids = torch.zeros(n, 1, C, dtype=torch.int32)
            if t < chunks:
                chunk_ids[0, 0] = ids[t * C : (t + 1) * C]
            writes = self._control(t, T) + [
                (
                    ttnn.from_torch(chunk_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard),
                    self.ids,
                )
            ]
            for host, dev in writes:
                ttnn.copy_host_to_device_tensor(host, dev)
            t1 = time.perf_counter()
            if self.trace_id is not None:
                ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            else:
                self._body()
            if timings is not None:
                ttnn.synchronize_device(self.mesh)
                timings.append((t1 - t0, time.perf_counter() - t1))
        ttnn.synchronize_device(self.mesh)
        return T

    def run_span(self, token_ids, first_chunk, n_tokens, drain=True, chunk=None):
        """resume one prompt's prefill across calls (chunked serving; reset() starts a prompt): feed token_ids,
        the prompt's tokens from position first_chunk * C, as chunks first_chunk, first_chunk + 1, .... n_tokens is
        the prompt length seen so far: until the draining call every fed chunk is full and n_tokens ends on a
        chunk. drain: also run the pipeline drain, after which the state is that of run(prompt[:n_tokens]) with the
        same chunk size. The chunk size must stay the same for the whole prompt. Between calls the pipeline holds
        the chunks in flight (x_last), the states and the KV scratch; nothing else may use the prefill."""
        T = token_ids.shape[0]
        C = chunk or self.C
        if C not in self._geo:
            raise ValueError(f"run_span: chunk {C} is not one of the prefill's chunk sizes {sorted(self._geo)}")
        if n_tokens > self.max_len or first_chunk * C + T != n_tokens:
            raise ValueError(
                f"run_span: {T} tokens from chunk {first_chunk} (chunk {C}) do not end at n_tokens {n_tokens} "
                f"(max_len {self.max_len})"
            )
        if not drain and T % C:
            raise ValueError(f"run_span: only the draining call may end on a partial chunk ({T} tokens, chunk {C})")
        if first_chunk == 0:
            self._span_chunk = C
        elif getattr(self, "_span_chunk", None) != C:
            raise ValueError(
                f"run_span: chunk {C} differs from the prompt's first span ({getattr(self, '_span_chunk', None)})"
            )
        self._check_trace_layout()
        self._use(C)
        n = self.n
        chunks = (n_tokens + C - 1) // C
        new = (T + C - 1) // C
        ids = torch.zeros(new * C, dtype=torch.int32)
        ids[:T] = token_ids.to(torch.int32)
        shard = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        for t in range(first_chunk, chunks + n - 1 if drain else chunks):
            chunk_ids = torch.zeros(n, 1, C, dtype=torch.int32)
            if t < chunks:
                chunk_ids[0, 0] = ids[(t - first_chunk) * C : (t - first_chunk + 1) * C]
            writes = self._control(t, n_tokens) + [
                (
                    ttnn.from_torch(chunk_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard),
                    self.ids,
                )
            ]
            for host, dev in writes:
                ttnn.copy_host_to_device_tensor(host, dev)
            if self.trace_id is not None:
                ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
            else:
                self._body()
        ttnn.synchronize_device(self.mesh)
        return n_tokens

    # ------------------------------------------------------------------ several prompts in one pass
    def run_batch(self, items, model, chunk=None, timings=None):
        """prefill several prompts in one pipelined pass and hand each to its slot of the rows-mode decode `model`.
        items: [(token_ids [T >= 1], block_table, slot)], each a prompt's tokens but its last (as run + handoff).

        The items' chunks flow through the pipeline back to back: chip p starts item i + 1's first chunk in the tick
        after item i's last, so the pipeline fills and drains once per call instead of once per prompt (a prompt of
        k chunks costs k ticks, not k + n - 1). Between ticks, every chip that has just run an item's last chunk
        hands its layers' GDN states and KV to the item's slot (handoff.cpp, that chip's jobs only; the states after one
        masked delta-rule step where the sequential path's drain would have run one), copies its conv carries to a
        staging buffer, and zeroes its GDN states and carries, so the next item starts from zero as after reset(). The conv histories go to the slots after the pass (staged carries -> host -> hist_scatter, as
        handoff). Every item's state is that of reset() + run() + handoff() of the prompt alone at the same chunk
        size: the same programs on the same data (a real chunk only reads its own prompt's states, KV positions and
        stage inputs; idle chips' chunks exist only in the pass's fill and drain).

        All items at one chunk size (default: pick_chunk of the longest); no taps (a drafter's context is one
        prompt's); nothing else may use the prefill meanwhile, and no split prompt may be in flight (run_span).
        timings, if given, gets the seconds of the ticks (with the boundary handoffs) and of the history scatter"""
        import time

        if self.tap_cache:
            raise ValueError("run_batch: a prefill with taps (a drafter's context) runs one prompt at a time")
        items = [(torch.as_tensor(ids).reshape(-1), table, slot) for ids, table, slot in items]
        Ts = [int(ids.shape[0]) for ids, _, _ in items]
        if not items or min(Ts) < 1 or max(Ts) > self.max_len:
            raise ValueError(f"run_batch: prompt lengths {Ts} (each 1 .. max_len {self.max_len})")
        if not getattr(model, "rows", False):
            raise ValueError("run_batch: the items go to slots of a rows-mode model")
        C = chunk or self.pick_chunk(max(Ts))
        if C not in self._geo:
            raise ValueError(f"run_batch: chunk {C} is not one of the prefill's chunk sizes {sorted(self._geo)}")
        t_start = time.perf_counter()
        self._check_trace_layout()
        self.reset()
        self._use(C)
        self._ho_setup(model)
        n, mesh = self.n, self.mesh
        gdn_js = [j for j in range(self.Ls) if not self.kinds[j]]
        nchunks = [-(-T // C) for T in Ts]
        sched = [(i, c) for i, k in enumerate(nchunks) for c in range(k)]  # the pass's chunks in order
        last = {s for s, (i, c) in enumerate(sched) if c == nchunks[i] - 1}
        S = len(sched)
        stages = self._stage_buffers()
        free, stage_of, done_items = list(range(len(stages))), {}, []
        # the job table of a boundary's handoff: chip p's jobs for its layers of the longest item, and its carry copies
        tiles = -(-max(Ts) // TILE)
        rows_max = max(len(self._ho_sel(p, tiles)) for p in range(n))
        jobs_ts, most = self._jobs_buffer(rows_max + len(gdn_js))
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)

        def scatter_histories():
            """the conv histories of the items whose last stage is handed off, from their staged carries"""
            ttnn.synchronize_device(mesh)
            for i in done_items:
                self._ho_hist(model, Ts[i], items[i][2], stages[stage_of[i]])
                free.append(stage_of.pop(i))
            done_items.clear()

        for t in range(S + n - 1):
            work = [(sched[t - p][1], Ts[sched[t - p][0]]) if 0 <= t - p < S else None for p in range(n)]
            chunk_ids = torch.zeros(n, 1, C, dtype=torch.int32)
            if t < S:
                i, c = sched[t]
                seg = items[i][0][c * C : (c + 1) * C]
                chunk_ids[0, 0, : seg.shape[0]] = seg.to(torch.int32)
            writes = self._control_chips(work) + [
                (
                    ttnn.from_torch(chunk_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard),
                    self.ids,
                )
            ]
            for host, dev in writes:
                ttnn.copy_host_to_device_tensor(host, dev)
            if self.trace_id is not None:
                ttnn.execute_trace(mesh, self.trace_id, cq_id=0, blocking=False)
            else:
                self._body()
            finished = [(p, sched[t - p][0]) for p in range(n) if 0 <= t - p < S and t - p in last]
            if not finished:
                continue
            jobs = [torch.zeros(0, 8, dtype=torch.int64) for _ in range(n)]
            pks = [torch.zeros(0, dtype=torch.int64) for _ in range(n)]
            for p, i in finished:
                if i not in stage_of:
                    if not free:  # (with >= n buffers some staged item is complete: n - 1 chips are behind chip 0)
                        scatter_histories()
                    stage_of[i] = free.pop(0)
                rows, pk = self._ho_jobs(model, p, Ts[i], items[i][1], items[i][2])
                # chip p's carries -> the item's staging buffer (local copies: 2 KB tile pages, no fabric packets)
                pages = self.carry[gdn_js[0]].buffer_num_pages()
                copies = torch.tensor(
                    [
                        [p, self.carry[j].buffer_address(), 0, 2048, 0, stages[stage_of[i]][j].buffer_address(), 0]
                        + [pages]
                        for j in gdn_js
                    ],
                    dtype=torch.int64,
                )
                jobs[p] = torch.cat([rows, copies])
                pks[p] = torch.cat([pk, torch.zeros(len(gdn_js), dtype=torch.int64)])
            # the sequential path (run + handoff) runs n - 1 - p masked ticks on chip p after the prompt's last chunk,
            # and a fully masked 32-row chunk of the delta rule (g = beta = 0) is not the identity on S: it leaves S
            # rounded (TF32-exact values; measured idempotent: a prompt whose last chunk ends in masked rows hands off the
            # same S either way). A last chunk whose final 32 rows hold tokens skipped it, so give those chips' states
            # one masked step here (not the last stage's: its last chunk ends the sequential run). By construction:
            # the masked tick's own delta rule and copy on every chip's states, whose results only the finished chips'
            # GDN jobs read (the other chips' states stay untouched)
            rnd = [p for p, i in finished if p < n - 1 and Ts[i] - (nchunks[i] - 1) * C > C - 32]
            masked = {}
            if rnd and BATCH_STATE_ROUND == "op":
                masked = self._masked_states(gdn_js)
                for p in rnd:
                    src = jobs[p][:, 1]
                    for addr, t in masked.items():
                        src[src == addr] = t.buffer_address()
            lanes = self._ho_lanes(jobs, pks)
            for (lane_jobs, _), dev in zip(lanes, jobs_ts):
                ttnn.copy_host_to_device_tensor(self._jobs_host(lane_jobs, most), dev)
            self._ho_copy(model, jobs_ts, lanes, items[finished[0][1]][2], extra_io=list(masked.values()))
            for t in masked.values():  # (any later write there is queued after the handoff that reads it)
                ttnn.deallocate(t)
            self._zero_chips([p for p, _ in finished], [self.S[j] for j in gdn_js] + [self.carry[j] for j in gdn_js])
            done_items += [i for p, i in finished if p == n - 1]
        if timings is not None:
            ttnn.synchronize_device(mesh)
            timings["ticks"] = time.perf_counter() - t_start
        scatter_histories()
        if timings is not None:
            timings["total"] = time.perf_counter() - t_start
            timings["histories"] = timings["total"] - timings["ticks"]
        return Ts

    def _masked_states(self, gdn_js):
        """{S[j] address: S[j] after one fully masked 32-row chunk of the delta rule and the tick's copy}, on every chip:
        what a masked tick (gates g = beta = 0) leaves in a stage's GDN states, whatever the FPU and packer do to them
        """
        eye, tril, ones, masks = self.gdn_consts
        mk, out = self._masked, {}
        for j in gdn_js:
            o, S = ttnn.transformer.chunk_gated_delta_rule(
                mk["q"],
                mk["k"],
                mk["v"],
                mk["g"],
                mk["beta"],
                scale=DK**-0.5,
                initial_state=self.S[j],
                output_final_state=True,
                chunk_size=TILE,
                output_head_major=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                eye=eye,
                tril=tril,
                ones=ones,
                masks=masks,
            )
            ttnn.deallocate(o)
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, NV, DK, DV]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.copy(S, t)
            ttnn.deallocate(S)
            out[self.S[j].buffer_address()] = t
        return out

    def _stage_buffers(self):
        """run_batch's conv-carry staging buffers (built in __init__: see BATCH_STAGES)"""
        if not self._stages:
            raise RuntimeError("run_batch needs its staging buffers: RESIDENT_PP_BATCH_STAGES is 0")
        return self._stages

    def _jobs_buffer(self, rows):
        """per handoff lane a device job table of at least `rows` 64 B rows per chip (kept for the next calls; grows in
        powers of 2), and its rows per chip"""
        have = getattr(self, "_jobs_t", None)
        if have is None or have[1] < rows:
            if have is not None:
                for t in have[0]:
                    ttnn.deallocate(t)
            most = 1 << max(10, (rows - 1).bit_length())
            self._jobs_t = [
                ttnn.from_torch(
                    torch.zeros(self.n * most, 16, dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in self._ho["cores"]
            ], most
        return self._jobs_t

    def _jobs_host(self, jobs, most):
        """host image of a job table: chip c's jobs[c] in the first rows of its `most` (kernels/handoff.cpp)"""
        return ttnn.from_torch(
            self._jobs_table(jobs, most),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def _zero_chips(self, chips, tensors):
        """zero-fill DRAM tensors on the given chips only (kernels/dram_zero.cpp, as common.zero_dram), without
        synchronizing"""
        self._chip_ranges(chips, tensors, f"{PKG_DIR}/kernels/dram_zero.cpp", [1])

    def _chip_ranges(self, chips, tensors, src, extra_ct, cores_per_bank=4, chunk=32768):
        """a DRAM-range kernel (dram_zero.cpp's arguments) over every bank's part of each tensor, on the given chips
        only: every chip runs the program, the others with no ranges (one program shape: the cache keeps one)"""
        mesh, n = self.mesh, self.n
        banks = mesh.dram_grid_size().x
        grid = mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(banks * cores_per_bank)]
        ranges = [[] for _ in cores]
        for t in tensors:
            per_bank = -(-t.buffer_num_pages() // banks) * t.buffer_aligned_page_size()
            step = -(-per_bank // cores_per_bank // 64) * 64
            for b in range(banks):
                for k in range(cores_per_bank):
                    lo, hi = k * step, min(per_bank, (k + 1) * step)
                    if lo < hi:
                        ranges[b * cores_per_bank + k] += [t.buffer_address() + lo, hi - lo]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        width = 2 * len(tensors)
        pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            rt = ttnn.RuntimeArgs()
            for i, c in enumerate(cores):
                own = ranges[i] if chip in chips else []
                rt[c.x][c.y] = [i // cores_per_bank, len(own) // 2] + own + [0] * (width - len(own))
            program = ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source=src,
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=core_set,
                        # (the args' length, then the kernel's own: dram_zero's 1 keeps these per-chip programs apart
                        # from zero_dram's in the program cache)
                        compile_time_args=[chunk, len(tensors)] + list(extra_ct),
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                        ),
                    )
                ],
                semaphores=[],
                cbs=[
                    ttnn.CBDescriptor(
                        total_size=chunk,
                        core_ranges=core_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=chunk)
                        ],
                    )
                ],
            )
            coord = ttnn.MeshCoordinate(0, chip)
            pd[ttnn.MeshCoordinateRange(coord, coord)] = program
        ttnn.generic_op(list(tensors), pd)

    def tap_rows(self, s0, rows):
        """per tap layer L: [1, 1, rows, HIDDEN] bf16 on every chip, its stage-layer j = L % Ls outputs of the last
        prefill at positions s0 .. s0 + rows - 1 (block multiples, rows <= the ring): chip L // Ls holds tap L's
        rows, the other chips their own layer j's. Rows past the prefilled positions are stale."""
        blk = self.block
        runs = []  # (first block, blocks) of the ring's contiguous runs
        for i in range(rows // blk):
            b = (s0 // blk + i) % self.tap_ring
            if runs and runs[-1][0] + runs[-1][1] == b:
                runs[-1][1] += 1
            else:
                runs.append([b, 1])
        out = []
        for L in self.taps:
            cache = self.tap_cache[L % self.Ls]
            parts = [
                ttnn.reshape(ttnn.slice(cache, (b, 0, 0, 0), (b + k, 1, blk, HIDDEN)), (1, 1, k * blk, HIDDEN))
                for b, k in runs
            ]
            out.append(parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2))
            if len(parts) > 1:
                for t in parts:
                    ttnn.deallocate(t)
        return out

    def read_taps(self, n_tokens, window):
        """[taps, rows, HIDDEN] bf16: the tap layers' outputs at positions start .. n_tokens - 1 of the last prefill,
        start = max(0, n_tokens - window) rounded down to a block (window <= tap_window); and start"""
        start = max(0, n_tokens - window) // self.block * self.block
        pos = torch.arange(start, n_tokens)
        rows = (pos // self.block) % self.tap_ring * self.block + pos % self.block
        out = []
        for L in self.taps:
            dev = ttnn.get_device_tensors(self.tap_cache[L % self.Ls])[L // self.Ls]
            out.append(ttnn.to_torch(dev).reshape(-1, HIDDEN)[rows])
        return torch.stack(out), start

    # ------------------------------------------------------------------ handoff
    def handoff(self, model, n_tokens, timings=None, block_table=None, slot=None):
        """move the state after n_tokens prompt tokens into the resident decode `model` (TP-sharded):
        GDN states and KV caches chip to chip on device (kernels/handoff.cpp), conv histories via the host.
        A verify model (ResidentModel verify) takes the states in slot 0 and the histories at ring step 0.
        A paged-KV model takes the KV into the pool pages of block_table (the physical block of each logical
        block of the prompt; default: its user's).
        A rows-mode model (multi-user) takes them into user `slot` (launch slot // batch, row slot % batch; default
        0), leaving every other user's state as it is: its GDN state row, its row of the conv histories
        (canonical order), its KV pages.
        timings, if given, gets the seconds of each part"""
        import time

        mark = [time.perf_counter()]

        def lap(name):
            if timings is not None:
                ttnn.synchronize_device(self.mesh)
                mark.append(time.perf_counter())
                timings[name] = mark[-1] - mark[-2]

        self._ho_setup(model)
        lanes = self._ho_lanes(*zip(*(self._ho_jobs(model, p, n_tokens, block_table, slot) for p in range(self.n))))
        lap("jobs")
        jobs_ts = [self._ho_upload(jobs) for jobs, _ in lanes]
        lap("upload")
        self._ho_copy(model, jobs_ts, lanes, slot)
        lap("copy")
        self._ho_hist(model, n_tokens, slot, self.carry, lap)
        for t in jobs_ts:
            ttnn.deallocate(t)

    def _ho_lanes(self, jobs, pks):
        """per copy engine (lane) its (jobs per chip, packets per chip): every chip's job list cut into HANDOFF_LINKS
        consecutive parts of about equal bytes (each job writes its own destination: any order is the same handoff)"""
        lanes = [([], []) for _ in self._ho["cores"]]
        L = len(lanes)
        for jb, pk in zip(jobs, pks):
            if L == 1 or not len(jb):
                cut = [0] * (L - 1) + [len(jb)]
            else:
                ends = torch.cumsum(jb[:, 3] * jb[:, 7], 0)
                cut = torch.searchsorted(ends, ends[-1] * torch.arange(1, L) // L, right=True).tolist() + [len(jb)]
            lo = 0
            for k, hi in enumerate(cut):
                lanes[k][0].append(jb[lo:hi])
                lanes[k][1].append(pk[lo:hi])
                lo = hi
        return lanes

    def _ho_setup(self, model):
        """every handoff job of a max_len prompt into `model`, once per decode model: per chip [m, 8] rows, each row's
        KV tile (-1: a GDN state, always sent), the packets it sends to another chip, and its (K or V, attention
        layer) ((-1, -1) for GDN)"""
        n, Ls, d = self.n, self.Ls, model.d
        mesh = self.mesh
        if not hasattr(self, "_ho"):
            # (an L1 buffer: made by the first handoff, the build's warm-up before any trace capture)
            cores = [ttnn.CoreCoord(k, 0) for k in range(HANDOFF_LINKS)]
            self._ho = dict(
                cores=cores,
                sem=ttnn.create_global_semaphore(mesh, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]), 0),
            )
        if self._ho.get("model") is model:
            return
        banks = d.banks
        kv_row = (HD // TILE) * 1088
        rows, tile_of, pk = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
        kv_of = [[] for _ in range(n)]
        tiles_max = self.max_len // TILE
        per_block = self.block // TILE
        t = torch.arange(tiles_max, dtype=torch.int64)
        bk, rt = t // per_block, t % per_block
        g_i = a_i = 0
        for i in range(self.layers):
            p, j = divmod(i, Ls)
            if self.kinds[j]:
                for src, dst in ((self.K[j], model.K_t), (self.V[j], model.V_t)):
                    for c in range(n):
                        r = torch.empty(tiles_max, 8, dtype=torch.int64)
                        r[:, 0] = c
                        r[:, 1] = src.buffer_address()
                        r[:, 2] = ((bk * NKV + c) * per_block + rt) * (HD // TILE)
                        r[:, 3] = 1088
                        r[:, 4] = 1
                        r[:, 5] = dst.buffer_address() + (a_i * model.rpb + t // banks) * kv_row
                        r[:, 6] = t % banks
                        r[:, 7] = HD // TILE
                        rows[p].append(r)
                        tile_of[p].append(t)
                        pk[p].append(torch.full((tiles_max,), -(-kv_row // 4096) if c != p else 0))
                        kv_of[p].append(torch.tensor([[int(dst is model.V_t), a_i]]).expand(tiles_max, 2))
                a_i += 1
            else:
                pages = d.nv * 16
                # (verify: a head's 16 pages per state slot, slot 0; rows mode: per user row, row 0 here,
                # moved to the slot's row per call)
                per_head = getattr(model, "verify", False) or getattr(model, "rows", False)
                per = [(hh * 16, 16) for hh in range(d.nv)] if per_head else [(0, pages)]
                for c in range(n):
                    for off, cnt in per:
                        dst_page = g_i * pages + off if len(per) == 1 else (g_i * d.nv + off // 16) * model.batch * 16
                        rows[p].append(
                            torch.tensor(
                                [
                                    [c, self.S[j].buffer_address(), c * pages + off, 4096, 0]
                                    + [model.state_t.buffer_address(), dst_page, cnt]
                                ],
                                dtype=torch.int64,
                            )
                        )
                        tile_of[p].append(torch.tensor([-1]))
                        pk[p].append(torch.tensor([cnt if c != p else 0]))
                        kv_of[p].append(torch.tensor([[-1, -1]]))
                g_i += 1
        # per chip its row segments in order: (first row, rows, KV: a block of tiles_max position tiles, else GDN rows),
        # so a prompt's jobs are a gather of whole GDN segments and each KV segment's first tiles (no scan of all rows)
        segs = [[] for _ in range(n)]
        for p in range(n):
            first = 0
            for tt in tile_of[p]:
                kv = bool(tt[0] >= 0)
                if not kv and segs[p] and not segs[p][-1][2] and sum(segs[p][-1][:2]) == first:
                    segs[p][-1] = (segs[p][-1][0], segs[p][-1][1] + len(tt), False)  # (consecutive GDN rows: one)
                else:
                    segs[p].append((first, len(tt), kv))
                first += len(tt)
        cat = lambda xs: [torch.cat(x) for x in xs]
        self._ho.update(
            model=model, rows=cat(rows), tile_of=cat(tile_of), pk=cat(pk), kv_of=cat(kv_of), segs=segs, sel={}
        )

    def _ho_sel(self, p, tiles):
        """the indices of chip p's job rows of a prompt of `tiles` position tiles (in row order: every GDN row, each KV
        block's first `tiles`), cached per (chip, tiles)"""
        key = (p, tiles)
        if key not in self._ho["sel"]:
            if len(self._ho["sel"]) > 4096:
                self._ho["sel"].clear()
            self._ho["sel"][key] = torch.cat(
                [torch.arange(f, f + (min(tiles, m) if kv else m)) for f, m, kv in self._ho["segs"][p]]
            )
        return self._ho["sel"][key]

    def _ho_target(self, model, slot):
        """(rows mode, slot, row, the slot's launch tensors) of a handoff into `model`"""
        rows_mode = getattr(model, "rows", False)
        slot = (slot or 0) if rows_mode else 0
        launch, row = divmod(slot, model.batch)
        return rows_mode, slot, row, model.subs[launch]

    def _ho_jobs(self, model, p, n_tokens, block_table=None, slot=None):
        """chip p's jobs of a handoff of n_tokens prompt tokens (its layers' GDN states and KV) into `model` (see
        handoff): [m, 8] int64 rows (a copy) and the fabric packets of each"""
        tiles = (n_tokens + TILE - 1) // TILE
        paged = getattr(model, "kv_paged", False)
        rows_mode, slot, row, target = self._ho_target(model, slot)
        assert not rows_mode or paged, "rows mode: the KV goes into pool pages"
        sel = self._ho_sel(p, tiles)
        jobs = self._ho["rows"][p][sel]  # (a copy: the cached rows keep their contiguous-mode targets)
        if paged:
            # position tile t of attention layer a -> page a * copy_pages + table[t / tpb] * tpb + t % tpb
            # (attn_paged.hpp): bank page % banks, at row page / banks of the pool's shard
            table = torch.as_tensor(list(model.block_tables[slot] if block_table is None else block_table))
            assert len(table) * model.kv_tpb >= tiles, "the block table does not cover the prompt"
            pools = torch.tensor([model.K_t.buffer_address(), model.V_t.buffer_address()])
            banks, kv_row = model.d.banks, (HD // TILE) * 1088
            kv, t = self._ho["kv_of"][p][sel], self._ho["tile_of"][p][sel]
            is_kv = kv[:, 0] >= 0
            t, which, a = t[is_kv], kv[is_kv, 0], kv[is_kv, 1] % model._attn_copies
            page = a * model.kv_copy_pages + table[t // model.kv_tpb] * model.kv_tpb + t % model.kv_tpb
            jobs[is_kv, 5] = pools[which] + (page // banks) * kv_row
            jobs[is_kv, 6] = page % banks
            if rows_mode:  # GDN state jobs: the slot's launch and row
                jobs[~is_kv, 5] = target["state"].buffer_address()
                jobs[~is_kv, 6] += row * 16
        return jobs, self._ho["pk"][p][sel]

    def _ho_upload(self, jobs):
        """the job table on device: chip c's jobs[c] as 64 B rows (kernels/handoff.cpp)"""
        n = self.n
        most = max(1, max(len(jb) for jb in jobs))
        return ttnn.from_torch(
            self._jobs_table(jobs, most),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _jobs_table(self, jobs, most):
        """[n * most, 16] int32: chip c's jobs[c] (int64 job words, uint32 on device) in the first rows of its `most`"""
        table = torch.zeros(self.n, most, 16, dtype=torch.int32)
        for p in range(self.n):
            # (the low 32 bits, as the device's uint32 words: torch's int64 -> int32 cast wraps)
            table[p, : len(jobs[p]), :8] = (jobs[p] & 0xFFFFFFFF).to(torch.int32)
        return table.reshape(self.n * most, 16)

    def _ho_copy(self, model, jobs_ts, lanes, slot, extra_io=()):
        """run the handoff program: on every chip, lane k's core walks its jobs of lanes[k] (jobs_ts[k] on device) over
        fabric link k, and waits for the packets the other chips' lane-k jobs send it"""
        n, mesh = self.n, self.mesh
        cores, sem = self._ho["cores"], self._ho["sem"]
        target = self._ho_target(model, slot)[3]
        packets_in = torch.zeros(len(lanes), n, dtype=torch.int64)
        for k, (jobs, pks) in enumerate(lanes):
            for p in range(n):
                if len(jobs[p]):
                    packets_in[k] += torch.bincount(jobs[p][:, 0], weights=pks[p].double(), minlength=n).long()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cbs = [
            ttnn.CBDescriptor(
                total_size=8 * 4096,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=4096)],
            ),
            ttnn.CBDescriptor(
                total_size=64 * 64,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=64)],
            ),
        ]
        mesh_pd = ttnn.MeshProgramDescriptor()
        for chip in range(n):
            coord = ttnn.MeshCoordinate(0, chip)
            rt = ttnn.RuntimeArgs()
            for k, core in enumerate(cores):
                rt[core.x][core.y] = [
                    jobs_ts[k].buffer_address(),
                    len(lanes[k][0][chip]),
                    chip,
                    ttnn.get_global_semaphore_address(sem),
                    int(packets_in[k, chip]),
                ]
            kernel = ttnn.KernelDescriptor(
                kernel_source=KDIR + "handoff.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=grid,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
            program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
            for k, core in enumerate(cores):
                args = program.kernels[0].runtime_args[core.x][core.y]
                for nb in (chip + 1, chip - 1):
                    if 0 <= nb < n:
                        args.append(1)
                        args.extend(
                            ttnn.setup_fabric_connection(
                                mesh.get_fabric_node_id(coord),
                                mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, nb)),
                                k,
                                program,
                                core,
                            )
                        )
                    else:
                        args.append(0)
            mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
        io = (
            list(jobs_ts)
            + [model.state_t, model.K_t, model.V_t, target["state"]]
            + list(extra_io)
            + list(self.S.values())
            + list(self.K.values())
            + list(self.V.values())
        )
        ttnn.generic_op(io, mesh_pd)

    def _ho_hist(self, model, n_tokens, slot, carry, lap=lambda name: None):
        """conv histories: the carries' (carry: GDN stage-layer -> its carry tensor, every chip its own layers') 3
        history rows to the host (a few MB), each chip's columns packed compactly in the decode's ring-slot order,
        scattered into its history tensor on device"""
        n, Ls, d = self.n, self.Ls, model.d
        mesh = self.mesh
        rows_mode, slot, row, target = self._ho_target(model, slot)
        gdn_js = [j for j in range(Ls) if not self.kinds[j]]
        rows3 = [
            ttnn.slice(ttnn.to_layout(carry[j], ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0, 0], [1, 1, CONV_K - 1, CONV_CH])
            for j in gdn_js
        ]
        cat = ttnn.to_torch(ttnn.concat(rows3, dim=2), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(
            n, len(gdn_js), CONV_K - 1, CONV_CH
        )
        lap("hist_read")
        gq, gv = d.gq, d.gv
        # chip-local q | k | v conv columns of every chip
        local = torch.stack(
            [
                torch.cat(
                    [
                        torch.arange(chip * gq, (chip + 1) * gq),
                        QD + torch.arange(chip * gq, (chip + 1) * gq),
                        2 * QD + torch.arange(chip * gv, (chip + 1) * gv),
                    ]
                )
                for chip in range(d.n)
            ]
        )
        hrows, hcols, hsrc = model.hist_index()
        K, copies = len(hrows), model._gdn_copies
        order = [(p, gdn_js.index(j)) for p, j in (divmod(i, Ls) for i in range(self.layers)) if not self.kinds[j]]
        G = len(order)
        hist = cat[[p for p, _ in order], [k for _, k in order]]  # [G, 3, CONV_CH], GDN copy order
        packed = hist[:, :, local[:, hsrc]].permute(2, 0, 1, 3)  # [chips, G, age, K * 32]
        verify = getattr(model, "verify", False)
        R = getattr(model, "R", 0)
        written = CONV_K - 1  # compact slots per GDN copy
        if R > 2:
            # the ring of single-row slots at ring step 0 (streamer_common.hpp): history rows 0..2 are rows 0..2 of the
            # first block's R-row tiles (one slot written per copy)
            ages = packed.reshape(d.n, G, CONV_K - 1, K, 2, 16)
            compact = torch.zeros(d.n, G, 1, K, 2, R, 16, dtype=packed.dtype)
            compact[:, :, 0, :, :, : CONV_K - 1] = ages.permute(0, 1, 3, 4, 2, 5)
            written = 1
        elif verify:
            # pair slots at ring step 0 (see streamer_common.hpp): slot j holds ages (j, j + 1) as the rows of
            # each conv tile (row 1 of slot 2 unused), in the 2 x 32 tile order (per face, row 0 then row 1)
            ages = packed.reshape(d.n, G, CONV_K - 1, K, 2, 16)
            nxt = torch.cat([ages[:, :, 1:], torch.zeros_like(ages[:, :, :1])], dim=2)
            compact = torch.stack([ages, nxt], dim=5)  # [chips, G, slot, K, face, row, 16]
        elif rows_mode:
            # canonical order (slot j = the j-th oldest input, see kernels/active_rows.hpp), the slot's row only
            compact = packed.clone()
        else:
            uses = torch.tensor([(model.n_gdn - g + copies - 1) // copies for g in range(G)])
            compact = torch.zeros(d.n, G, CONV_K - 1, K * TILE, dtype=packed.dtype)
            for age in range(CONV_K - 1):
                compact[:, torch.arange(G), (n_tokens * uses + age) % 3] = packed[:, :, age]
        # (rows mode: the tiles are batch rows high, and only the slot's row is written)
        rows_per_tile, slots = (
            (R, 16 // R) if R > 2 else (2, 4) if verify else (model.batch, 3) if rows_mode else (1, 3)
        )
        chunk = 64 if rows_mode else 64 * rows_per_tile
        compact_t = ttnn.from_torch(
            compact.reshape(d.n * G * written * K, chunk // 2),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lap("hist_layout")
        grid = mesh.compute_with_storage_grid_size()
        cores = min(G, grid.x * grid.y)
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        tiles = (hcols[::TILE] // TILE).tolist()
        pairs = [v for k in range(K) for v in (int(hrows[k]), tiles[k])]
        rt, first = ttnn.RuntimeArgs(), 0
        for i, c in enumerate(coords):
            cnt = G // cores + (i < G % cores)
            rt[c.x][c.y] = (
                [compact_t.buffer_address(), target["hist"].buffer_address(), first, cnt]
                + pairs
                + ([row] if rows_mode else [])
            )
            first += cnt
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KDIR + "hist_scatter.cpp",
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=all_cores,
                    compile_time_args=[K, copies, rows_per_tile, slots, model._blk * TILE * model.batch, written]
                    + [int(rows_mode)],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ],
            semaphores=[],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=K * chunk,
                    core_ranges=all_cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=chunk)
                    ],
                )
            ],
        )
        hist_pd = ttnn.MeshProgramDescriptor()
        hist_pd[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, n - 1))] = program
        ttnn.generic_op([compact_t, target["hist"]], hist_pd)
        ttnn.synchronize_device(mesh)
        lap("hist_write")
        ttnn.deallocate(compact_t)

    def export_state(self, d, n_tokens, max_pos):
        """the resident decode State (TP-sharded over d.n chips) after n_tokens prompt tokens"""
        n, Ls = self.n, self.Ls
        cat = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        n_gdn = sum(not a for a in self.kinds) * n
        n_attn = sum(self.kinds) * n
        st = State(d, n_gdn, n_attn, n_tokens, max_pos, zero=True)
        gq, gv = d.gq, d.gv
        S_all = {j: cat(self.S[j]) for j in self.S}
        carry_all = {j: cat(self.carry[j]) for j in self.carry}
        K_all = {j: cat(self.K[j]) for j in self.K}
        V_all = {j: cat(self.V[j]) for j in self.V}
        kv_blocks = K_all[next(iter(K_all))].shape[0] // n if K_all else 0
        g_i = a_i = 0
        for i in range(self.layers):
            p, j = divmod(i, Ls)
            if self.kinds[j]:
                for which, src, dst in (("K", K_all, st.K), ("V", V_all, st.V)):
                    c = src[j].reshape(n, kv_blocks, NKV, self.block, HD)[p, : self.blocks]
                    seq = c.permute(1, 0, 2, 3).reshape(NKV, self.blocks * self.block, HD)[:, :n_tokens]
                    for chip in range(d.n):
                        rows = torch.zeros(max_pos, HD)
                        rows[:n_tokens] = seq[chip]
                        dst[a_i][chip] = quantize_rows(rows)
                a_i += 1
            else:
                S = S_all[j].reshape(n, NV, DK, DV)[p]
                carry = carry_all[j].reshape(n, TILE, CONV_CH)[p, : CONV_K - 1]
                for chip in range(d.n):
                    st.gdn[g_i][chip] = S[chip * d.nv : (chip + 1) * d.nv].clone()
                    st.hist[g_i][chip] = torch.cat(
                        [
                            carry[:, chip * gq : (chip + 1) * gq],
                            carry[:, QD + chip * gq : QD + (chip + 1) * gq],
                            carry[:, 2 * QD + chip * gv : 2 * QD + (chip + 1) * gv],
                        ],
                        dim=1,
                    )
                g_i += 1
        return st
