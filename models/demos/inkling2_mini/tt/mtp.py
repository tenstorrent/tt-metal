# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini DSpark drafter (`mtp.safetensors`) on one p100a, batch 1.

Port of vLLM `vllm/models/inkling2/nvidia/dspark.py` (cross-checked with ollama `mlxrunner/model/inkling2/draft.go`):
- target features: 4 taps per committed position = input (running AttnRes partial) of applications 5, 23, 41 and
  the final-norm output; `alpha = softmax(fuse_weight)` mixes them, one row per draft layer + one for the markov head.
- `commit(feats)` (per decoded position, inside the target's trace): per draft layer, rms(mix) -> k/v_proj_ctx ->
  k_norm_ctx -> RoPE(pos) -> full-length K/V cache at pos; f_rnn = rms(mix row 3) for the next draft.
- `draft()` for anchor token x_p at p (context 0..p-1 committed): block rows = [embed_norm(embed(x_p)), mask x7] at
  positions p..p+7, 3 dense layers (q/k norm + neox RoPE theta 1e6, bidirectional over the block, context keys
  0..p-1 for layer 1, p-1024..p-1 for the local layers 0 and 2), final norm, shared unembed; then the recurrent
  `rnn_hidden` markov head picks k <= 7 greedy tokens on device.
Every position-dependent input is a device tensor rewritten from host (trace safe).

Speed (S6): the drafter's matmuls run as bank-streamed linears (tt/fused/bank_linear.py, ~420 GB/s instead of
~200 for ttnn.linear on interleaved weights); the block's own K/V rows go into the context caches at p..p+7
(fused/kernels/mtp_kv_rows.cpp, one program per layer) and the attention is one flash-decode call per layer
(causal at p+7; local layers with a 1032-key sliding window = keys p-1024 .. p+7), so no op reads max_ctx-wide
masks; the markov head runs only the k steps that are verified. The block rows written at p..p+7 are context
slots that the commit of those positions overwrites before any later draft reads them (anchor p' <= p+k+1).
Drafter numerics never reach the output: every accepted token is the target's own argmax.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

import ttnn
from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear
from models.demos.inkling2_mini.tt.fused.head import FusedLogitsTail

HIDDEN, NH, NKV, D, RANK = 2304, 24, 8, 128, 256
GROUPS = NH // NKV
BLOCK, WINDOW, THETA, EPS, NEG, SOFTCAP = 8, 1024, 1.0e6, 1e-6, -1.0e9, 30.0
N_LAYERS, LOCAL = 3, (0, 2)
STASH = 4  # positions per stash group (the decode step's conv phases: one traced step per slot)
TAP_APPS = (5, 23, 41)  # + final norm output (visit 48)


def rope_tables(pos0, rows):
    """neox cos/sin [rows, 128] for positions pos0..pos0+rows-1 (halves duplicated)."""
    inv = 1.0 / THETA ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
    ang = torch.arange(pos0, pos0 + rows, dtype=torch.float64)[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.cos()], -1).float(), torch.cat([ang.sin(), ang.sin()], -1).float()


KV_PAD = 128  # cache rows past max_ctx: the block rows of a draft at p <= max_ctx - 1 (and the flash-decode chunk)


def bank(device, w, cfg, out_dtype, cores_per_bank=6):
    """BankLinear for one drafter matmul: w torch [K, N]; one reference chunk, up to 6 readers per bank."""
    nb = device.dram_grid_size().x
    K, N = w.shape
    chunk = 14336 if N > 65536 else math.ceil(N / (nb * 32)) * nb * 32
    kt = K // 32
    c = min(cores_per_bank, chunk // 32 // nb)
    return BankLinear(device, w, ttnn.bfloat8_b, chunk, kb=next(b for b in range(8, 0, -1) if kt % b == 0),
                      cores_per_bank=c, compute_kernel_config=cfg, out_cols=N, out_dtype=out_dtype)  # fmt: skip


def mtp_root():
    """The drafter weight cache (convert/mtp.py). It does not depend on the run mode: $INKLING_MTP_CACHE, else the
    first of <decode cache>/mtp, <prefill cache>/mtp and the shared <cache dir>/mtp that holds a manifest."""
    env = os.environ.get("INKLING_MTP_CACHE")
    if env:
        return Path(env)
    cands = [r / "mtp" for r in (wcache.DEFAULT_ROOT, wcache.DEFAULT_TILE_ROOT, wcache.CACHE_DIR)]
    return next((c for c in cands if (c / "manifest.json").is_file()), cands[-1])


class DSparkDrafter:
    def __init__(self, model, root=None, max_ctx=None, k=BLOCK - 1, fast=True, unembed_dtype=None,ckpt="/work/models/Inkling-2-Mini"):  # fmt: skip
        """model: built Inkling2Model (shares embed, embed_norm, unembed, DecodePositions.pos).
        k: draft tokens per pass (markov steps). fast=False: E9's ttnn.linear matmuls (A/B only).
        unembed_dtype: None = the decode head's unembed; ttnn.bfloat4_b = a drafter-only 4-bit copy (+0.25 GB).
        ckpt: source checkpoint for markov_head.hidden_out (the per-step head input, see draft())."""
        dev = self.device = model.device
        self.model = model
        self.k = k
        self.max_ctx = max_ctx or model.max_ctx
        root = root or mtp_root()
        dram = ttnn.DRAM_MEMORY_CONFIG
        host = wcache.load_global(root, None)
        self.ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.mmc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        big = {f"l{i}.{n}" for i in range(N_LAYERS) for n in ("wqkv", "wo", "w_gate_up", "w_down")}
        self.lin = {}
        if fast:
            out = {"wqkv": ttnn.bfloat16}
            for name in sorted(big):
                t = ttnn.to_torch(host[name]).float()
                t = t.reshape(t.shape[-2], t.shape[-1])
                dt = out.get(name.split(".")[-1], ttnn.float32)
                # a bank linear keeps its whole x in L1 (K x 64 B): K > 4608 is cut into K halves (y = x0 W0 + x1 W1)
                # so its CBs stay below the decode staging buffer (tt/mk/overlap.py, ~890 KB up)
                parts = 2 if t.shape[0] > 4608 else 1
                kp = t.shape[0] // parts
                if name.endswith("w_gate_up"):  # gate and up as two linears: no column slices of the output
                    half = t.shape[1] // 2
                    for j, n in enumerate(("w_gate", "w_up")):
                        self.lin[name.replace("w_gate_up", n)] = [
                            bank(dev, t[:, j * half : (j + 1) * half], self.mmc, dt)
                        ]
                else:
                    self.lin[name] = [bank(dev, t[j * kp : (j + 1) * kp], self.mmc, dt) for j in range(parts)]
                del t
        self.w = {k_: ttnn.to_device(v, dev, memory_config=dram) for k_, v in host.items()
                  if k_ not in self.lin and k_ not in ("fold_w", "fold_b")}  # fmt: skip
        del host
        w = self.w
        # markov logits_i = U (hid_{i+1} + H lat_i + b) (= base + fold_w lat + fold_b of the reference, U linear):
        # one decode-head pass (bank unembed + fused softcap / mask / argmax) per draft token, no vocab-wide tile ops
        from safetensors import safe_open

        with safe_open(f"{ckpt}/mtp.safetensors", "pt") as f:
            ho_w = f.get_tensor("markov_head.hidden_out.weight").float()  # [H, R]
            ho_b = f.get_tensor("markov_head.hidden_out.bias").float()
        w["ho_w"] = ttnn.from_torch(ho_w.T.contiguous()[None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=dev, memory_config=dram)  # fmt: skip
        w["ho_b"] = ttnn.from_torch(ho_b.view(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=dram)  # fmt: skip
        self.head = self.make_head(unembed_dtype)

        def put(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, memory_config=dram)

        me = ttnn.to_torch(w["mask_embed"]).float()[0, 0, 0]
        base = torch.zeros(1, 1, 32, HIDDEN)
        base[0, 0, 1:BLOCK] = me
        self.x_base = put(base)
        rot = torch.zeros(D, D)  # x @ rot = rotate_half(x) = [-x2, x1]
        for j in range(D // 2):
            rot[j + D // 2, j] = -1.0
            rot[j, j + D // 2] = 1.0
        self.rot = put(rot.view(1, 1, D, D))
        # host-rewritten inputs
        # RoPE rows by position: device tables, looked up inside the traces (no host tilize / write per pass)
        rows = self.max_ctx + KV_PAD + 32
        c, sn = rope_tables(0, rows)
        self.rope_tab = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                         memory_config=dram) for t in (c, sn)]  # fmt: skip
        del c, sn
        self.blk_ids = ttnn.from_torch(
            torch.zeros(1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )
        self.commit_ids = ttnn.from_torch(torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32,
                                          layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)  # fmt: skip
        self.blk_pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=dev)  # p
        # stash (TappedModel.stash): the taps of up to STASH positions p0 .. wait here for one commit_rows.
        # stash[r] = the 4 feats rows of position p0 + r (allocated by the first eager stash step)
        self.stash = [[None] * (len(TAP_APPS) + 1) for _ in range(STASH)]
        self.stash_pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=dev)  # p0
        self.stash_ids = {
            n: ttnn.from_torch(
                torch.zeros(1, n, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
            )
            for n in range(1, STASH + 1)
        }  # fmt: skip  positions p0 .. p0 + n - 1 (commit_rows RoPE rows)
        self._f_rnn_rows = {}  # rows -> commit_rows f_rnn buffer
        # p + 7, once per kv group (the flash-decode batch)
        self.cur_pos = ttnn.from_torch(torch.zeros(NKV, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        self.f_rnn = put(torch.zeros(1, 1, 1, HIDDEN), ttnn.bfloat16)
        # row selectors: onehot[i] [1,1,1,32] picks block row i of a [1,1,32,C] tensor by matmul (a row slice of a
        # tile tensor is untilize + slice + a single-core tilize, ~70 us)
        self.onehot = []
        for i in range(BLOCK):
            e = torch.zeros(1, 1, 1, 32)
            e[..., i] = 1.0
            self.onehot.append(put(e))
        self.S = self.max_ctx + KV_PAD
        zeros = torch.zeros(1, NKV, self.S, D)
        self.k_cache = [put(zeros, ttnn.bfloat16) for _ in range(N_LAYERS)]
        self.v_cache = [put(zeros, ttnn.bfloat16) for _ in range(N_LAYERS)]
        del zeros
        self.kv_shard = model.layers[0].attn.kv_shard
        grid = dev.compute_with_storage_grid_size()
        self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=128,
                                               exp_approx_mode=False, max_cores_per_head_batch=16)  # fmt: skip
        # block K/V writer: one core per (k | v, head) on rows 0..1 of column 0..7
        self.kv_cores = [ttnn.CoreCoord(x, y) for y in range(2) for x in range(NKV)]
        self._mix = {}  # commit_rows mix matrices (_mix_rows)
        self._alpha = None  # host alpha (commit_chunk)
        self.trace = None
        self.tokens = None
        self.top_probs = False  # early stop (spec_generator.py): also compute each step's argmax probability
        self.probs = None  # [1, 1, 1, k] fp32 TILE: per-step top prob (top_probs)
        self._prob_mask = None
        # anchor token, written by the target's step (TappedModel); allocated before any trace capture
        self.anchor = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )

    def make_head(self, unembed_dtype=None):
        """Logits tail for the markov steps with its own raw / logits / token buffers: the decode head's bank
        unembed, or (unembed_dtype) a drafter-only copy in that dtype."""
        from models.demos.inkling2_mini.tt import model as tmodel

        un = self.model.unembed
        if unembed_dtype is not None:
            uw = ttnn.to_torch(wcache.load_global(wcache.DEFAULT_ROOT, None)["unembed"])
            uw = uw.reshape(uw.shape[-2], uw.shape[-1])
            kt = uw.shape[0] // (32 * 8)
            un = BankLinear(self.device, uw, unembed_dtype, tmodel.UNEMBED_CHUNK,
                            kb=next(b for b in range(8, 0, -1) if kt % b == 0), cores_per_bank=3,
                            compute_kernel_config=self.mmc, split=tmodel.UNEMBED_SPLIT)  # fmt: skip
            del uw
        return FusedLogitsTail(self.device, un, tmodel.VOCAB_PAD, tmodel.VOCAB_REAL, tmodel.SOFTCAP)

    # ------------------------------------------------------------ helpers
    def _rms(self, x, wname, out=ttnn.bfloat16):
        if x.shape[-1] == HIDDEN:  # [1,1,r,2304]: the model's width-sharded fp32 rms (18 cores, not 1)
            return self.model.rows.rms(x, weight=self.w[wname], eps=EPS, dtype=out)
        y = ttnn.rms_norm(x, weight=self.w[wname], epsilon=EPS, compute_kernel_config=self.ckc)
        return y if y.dtype == out else ttnn.typecast(y, out)

    def _lin(self, x, w, dtype=ttnn.bfloat16):
        if isinstance(w, str) and w in self.lin:
            x = x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)
            parts = self.lin[w]
            if len(parts) == 1:
                y = parts[0](x)
            else:
                kp = x.shape[-1] // len(parts)
                ys = [
                    bl(ttnn.slice(x, [0, 0, 0, j * kp], [1, 1, x.shape[-2], (j + 1) * kp]))
                    for j, bl in enumerate(parts)
                ]
                y = ys[0]
                for t in ys[1:]:
                    y = ttnn.add(y, t, dtype=ttnn.float32)
            return y if y.dtype == dtype else ttnn.typecast(y, dtype)
        if not isinstance(w, ttnn.Tensor):
            w = self.w[w]
        return ttnn.linear(x, w, dtype=dtype, compute_kernel_config=self.mmc)

    def _rope_rows(self, ids, n):
        """(cos, sin) [1, 1, n, 128] TILE bf16 for the positions in ids (uint32 [1, n] ROW_MAJOR)."""
        return [ttnn.reshape(ttnn.embedding(ids, t, layout=ttnn.TILE_LAYOUT), [1, 1, n, D]) for t in self.rope_tab]

    def _rope(self, x, cos, sin):
        x32 = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        r = ttnn.matmul(x32, self.rot, dtype=ttnn.float32, compute_kernel_config=self.ckc)
        return ttnn.typecast(ttnn.add(ttnn.multiply(x32, cos), ttnn.multiply(r, sin)), ttnn.bfloat16)

    def _write_block(self, i, k, v, pos=None, n=BLOCK):
        """Rows 0..n-1 of k / v [1, 8, 32, 128] (bf16 TILE, DRAM) -> rows pos..pos+n-1 of layer i's caches (pos: int32
        [1] device tensor, default the draft block's p)."""
        pos = self.blk_pos if pos is None else pos
        dram = ttnn.DRAM_MEMORY_CONFIG
        k, v = ttnn.to_memory_config(k, dram), ttnn.to_memory_config(v, dram)
        kc, vc = self.k_cache[i], self.v_cache[i]
        rt = []
        for j, _ in enumerate(self.kv_cores):
            src, dst = (k, kc) if j < NKV else (v, vc)
            rt.append([src.buffer_address(), dst.buffer_address(), self.S // 32, pos.buffer_address(), j % NKV, n])
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.kv_cores])
        kern = ttnn.KernelDescriptor(
            kernel_source=f"{F.KDIR}/mtp_kv_rows.cpp",
            core_ranges=crs,
            runtime_args=[(c, a) for c, a in zip(self.kv_cores, rt)],
            config=ttnn.ReaderConfigDescriptor(),
        )
        fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=4096 + 128)
        cb = ttnn.CBDescriptor(total_size=4096 + 128, core_ranges=crs, format_descriptors=[fmt])
        ttnn.generic_op([k, v, pos, kc, vc], ttnn.ProgramDescriptor(kernels=[kern], cbs=[cb], semaphores=[]))

    # ------------------------------------------------------------ host inputs
    def set_commit_pos(self, pos):
        """The committed position's RoPE row index (one 4-byte write; the rows are looked up on device)."""
        ids = torch.tensor([[pos]], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        self.commit_ids)  # fmt: skip

    def set_draft_pos(self, p):
        if not 0 < p < self.max_ctx:
            raise ValueError(f"draft position {p} outside (0, {self.max_ctx})")
        ids = torch.arange(p, p + 32, dtype=torch.int32).view(1, 32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.blk_ids
        )
        for t, v in ((self.blk_pos, [p]), (self.cur_pos, [p + BLOCK - 1] * NKV)):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor(v, dtype=torch.int32), dtype=ttnn.int32), t)

    # ------------------------------------------------------------ device
    def commit(self, feats, pos_idx):
        """feats: [1,1,32,C] rows 0..3 = the 4 taps (bf16/fp32) of the position just decoded; pos_idx: int32 [1]
        device tensor (DecodePositions.pos). RoPE row: commit_ids (set_commit_pos)."""
        f = ttnn.typecast(feats, ttnn.float32) if feats.dtype != ttnn.float32 else feats
        mix = ttnn.matmul(self.w["alpha"], f, dtype=ttnn.float32, compute_kernel_config=self.ckc)  # row r = mix r
        cos_pos, sin_pos = self._rope_rows(self.commit_ids, 1)
        for i in range(N_LAYERS):
            c = self._rms(ttnn.slice(mix, [0, 0, i, 0], [1, 1, i + 1, HIDDEN]), f"l{i}.ctx_norm")
            kv = self._lin(c, f"l{i}.wkv_ctx")
            k = ttnn.reshape(ttnn.slice(kv, [0, 0, 0, 0], [1, 1, 1, NKV * D]), [1, 1, NKV, D])
            v = ttnn.reshape(ttnn.slice(kv, [0, 0, 0, NKV * D], [1, 1, 1, 2 * NKV * D]), [1, 1, NKV, D])
            k = self._rope(self._rms(k, f"l{i}.k_norm_ctx", ttnn.float32), cos_pos, sin_pos)
            for cache, t in ((self.k_cache[i], k), (self.v_cache[i], v)):
                ttnn.experimental.paged_update_cache(
                    cache, ttnn.to_memory_config(t, self.kv_shard), update_idxs_tensor=pos_idx
                )
        fr = self._rms(ttnn.slice(mix, [0, 0, N_LAYERS, 0], [1, 1, N_LAYERS + 1, HIDDEN]), "rnn_norm")
        ttnn.copy(fr, self.f_rnn)

    def _mix_rows(self, i, R, ntap):
        """[1, 1, R, ntap * R] fp32: row r = alpha row i on the stacked-feats rows of position r (ntap * r + j, j
        the tap), so one matmul gives mix row i of the R positions as [1, 1, R, C] (a row slice of a tile tensor is
        untilize + slice + tilize; the R slices + concat cost ~55 us per mix row). A position's taps stay adjacent
        and in order: spread over the tile (tap-major rows) the sum rounds differently."""
        key = (i, R, ntap)
        if key not in self._mix:
            a = ttnn.to_torch(self.w["alpha"]).float()[0, 0]
            assert not a[i, ntap:].any()
            m = torch.zeros(1, 1, R, ntap * R)
            for r in range(R):
                m[0, 0, r, ntap * r : ntap * (r + 1)] = a[i, :ntap]
            self._mix[key] = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        return self._mix[key]

    def commit_rows(self, feats, pos0, ids):
        """commit() for R consecutive positions in one pass (speculative verify): feats [1, 1, ntap * R, C], row
        ntap * r + j = tap j of position p0 + r (the commit() feats rows of the R positions in turn), pos0 int32 [1]
        = p0, ids uint32 [1, R] ROW_MAJOR = p0 .. p0+R-1. The context K / V rows go to p0 .. p0+R-1 in one write per
        layer; f_rnn of row r goes to self.f_rnn_rows row r (the caller copies the accepted row into f_rnn). Same
        math per row as commit(): each mix row is the same products in the same order, plus zeros."""
        R = ids.shape[-1]
        ntap = feats.shape[-2] // R
        f = ttnn.typecast(feats, ttnn.float32) if feats.dtype != ttnn.float32 else feats

        def rows_of(j):  # mix row j of every position -> [1, 1, R, C]
            return ttnn.matmul(self._mix_rows(j, R, ntap), f, dtype=ttnn.float32, compute_kernel_config=self.ckc)

        cos, sin = self._rope_rows(ids, R)
        for i in range(N_LAYERS):
            c = self._rms(rows_of(i), f"l{i}.ctx_norm")
            kv = self._lin(c, f"l{i}.wkv_ctx")  # [1, 1, R, 2048]

            def heads(c0):
                t = ttnn.reshape(ttnn.slice(kv, [0, 0, 0, c0], [1, 1, R, c0 + NKV * D]), [1, R, NKV, D])
                return ttnn.permute(t, (0, 2, 1, 3))  # [1, 8, R, 128]

            k = self._rope(self._rms(heads(0), f"l{i}.k_norm_ctx", ttnn.float32), cos, sin)
            v = ttnn.typecast(heads(NKV * D), ttnn.bfloat16)
            self._write_block(i, k, v, pos=pos0, n=R)
        fr = self._rms(rows_of(N_LAYERS), "rnn_norm")  # [1, 1, R, C] bf16
        if R not in self._f_rnn_rows:  # one buffer per row count, made by the first (eager) call: none is freed
            self._f_rnn_rows[R] = ttnn.clone(fr)  # while a trace captured earlier still reads it
        ttnn.copy(fr, self._f_rnn_rows[R])  # (also on the first call: it compiles the copy a capture will run)
        self.f_rnn_rows = self._f_rnn_rows[R]

    # ------------------------------------------------------------ stash: commit a run of plain steps at once
    def set_stash_pos(self, p0):
        """The first position of the stash group (host writes; the stash steps and stash_commit read them)."""
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([p0], dtype=torch.int32), dtype=ttnn.int32), self.stash_pos
        )
        for n, t in self.stash_ids.items():
            ids = torch.arange(p0, p0 + n, dtype=torch.int32).view(1, n)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), t)

    def stash_put(self, r, j, row):
        """feats row j ([1, 1, 1, C]) of the group's position r. Returns the stash buffer."""
        if self.stash[r][j] is None:  # first use must be eager (and compiles the copy a capture will run)
            self.stash[r][j] = ttnn.clone(row)
        ttnn.copy(row, self.stash[r][j])
        return self.stash[r][j]

    def stash_commit(self, n):
        """commit() of the group's first n positions in one pass (commit_rows), f_rnn = the last one's."""
        feats = ttnn.concat([x for r in range(n) for x in self.stash[r]], dim=2)
        self.commit_rows(feats, self.stash_pos, self.stash_ids[n])
        ttnn.copy(ttnn.slice(self.f_rnn_rows, [0, 0, n - 1, 0], [1, 1, n, HIDDEN]), self.f_rnn)

    def commit_chunk(self, taps, s, n=None):
        """Chunked prefill (Inkling2Prefill.drafter): the drafter context of positions s .. s + t - 1 in one pass.
        taps: the 4 commit() feats rows as tensors [1, 1, t, C] (the inputs of the tapped apps, then the final-norm
        rows), s and t multiples of 32. n: on the last chunk, its real rows (f_rnn = row n - 1; rows n .. t - 1 land
        at positions the first draft after the prompt overwrites). Same math as commit() with the generic ops, so
        the rows agree with it to rounding, not bit for bit: draft quality only, the decoded tokens do not depend
        on it."""
        t = taps[0].shape[-2]
        assert len(taps) == len(TAP_APPS) + 1 and s % 32 == 0 and t % 32 == 0 and s + t <= self.S, (s, t)
        if self._alpha is None:
            self._alpha = ttnn.to_torch(self.w["alpha"]).float()[0, 0]
        f = [x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32) for x in taps]

        def mix(i):
            y = None
            for j, x in enumerate(f):
                term = ttnn.multiply(x, float(self._alpha[i, j]))
                y = term if y is None else ttnn.add(y, term)
            return y

        def norm(x, name):
            return ttnn.rms_norm(x, weight=self.w[name], epsilon=EPS, compute_kernel_config=self.ckc)

        cos, sin = (
            ttnn.from_torch(x.view(1, 1, t, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            for x in rope_tables(s, t)
        )
        for i in range(N_LAYERS):
            c = ttnn.typecast(norm(mix(i), f"l{i}.ctx_norm"), ttnn.bfloat16)
            kv = ttnn.linear(c, self.w[f"l{i}.wkv_ctx"], dtype=ttnn.bfloat16, compute_kernel_config=self.mmc)

            def heads(c0):
                h = ttnn.reshape(ttnn.slice(kv, [0, 0, 0, c0], [1, 1, t, c0 + NKV * D]), [1, t, NKV, D])
                return ttnn.permute(h, (0, 2, 1, 3))  # [1, 8, t, 128]

            k = self._rope(norm(heads(0), f"l{i}.k_norm_ctx"), cos, sin)
            v = heads(NKV * D)
            for cache, x in ((self.k_cache[i], k), (self.v_cache[i], v)):
                x = x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)
                ttnn.experimental.slice_write(x, cache, [0, 0, s, 0], [1, NKV, s + t, D], [1, 1, 1, 1])
        if n is not None:
            fr = ttnn.slice(norm(mix(N_LAYERS), "rnn_norm"), [0, 0, n - 1, 0], [1, 1, n, HIDDEN])
            ttnn.copy(ttnn.typecast(fr, ttnn.bfloat16), self.f_rnn)

    def _layer(self, i, res):
        h = self._rms(res, f"l{i}.in_norm")
        qkv = self._lin(h, f"l{i}.wqkv")  # [1,1,32,5120]

        qh, kh, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=NH, num_kv_heads=NKV,
                                                           transpose_k_heads=False)  # fmt: skip
        rope = lambda x: ttnn.experimental.rotary_embedding(x, *self.rope_blk)  # noqa: E731
        q = rope(self._rms(qh, f"l{i}.q_norm"))  # [1, 24, 32, 128] bf16
        k = rope(self._rms(kh, f"l{i}.k_norm"))
        self._write_block(i, k, v)
        # kv groups as the flash-decode batch: user kv holds q rows g*8 + r (head h = kv*3 + g, block row r), one
        # 24-row Q tile per user, cache viewed as [8 users, 1 head, S, 128]
        qg = ttnn.reshape(ttnn.slice(q, [0, 0, 0, 0], [1, NH, BLOCK, D]), [1, NKV, GROUPS * BLOCK, D])
        kc, vc = (ttnn.reshape(c, [NKV, 1, self.S, D]) for c in (self.k_cache[i], self.v_cache[i]))
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qg,
            kc,
            vc,
            is_causal=True,
            cur_pos_tensor=self.cur_pos,
            scale=D**-0.5,
            sliding_window_size=(WINDOW + BLOCK) if i in LOCAL else None,
            program_config=self.sdpa_cfg,
            compute_kernel_config=self.ckc,
        )  # fmt: skip  [1, 8, 32, 128], rows 0..23 valid
        o = ttnn.reshape(ttnn.slice(o, [0, 0, 0, 0], [1, NKV, GROUPS * BLOCK, D]), [1, NH, BLOCK, D])
        o = ttnn.experimental.nlp_concat_heads(ttnn.pad(o, [(0, 0), (0, 0), (0, 32 - BLOCK), (0, 0)], 0.0))
        res = ttnn.add(res, self._lin(o, f"l{i}.wo", ttnn.float32), dtype=ttnn.float32)
        h = self._rms(res, f"l{i}.post_norm")
        if f"l{i}.w_gate" in self.lin:
            g, u = self._lin(h, f"l{i}.w_gate", ttnn.float32), self._lin(h, f"l{i}.w_up", ttnn.float32)
        else:
            gu = self._lin(h, f"l{i}.w_gate_up", ttnn.float32)
            inter = gu.shape[-1] // 2
            g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 32, inter])
            u = ttnn.slice(gu, [0, 0, 0, inter], [1, 1, 32, 2 * inter])
        a = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16)
        return ttnn.add(res, self._lin(a, f"l{i}.w_down", ttnn.float32), dtype=ttnn.float32)

    def draft(self, anchor):
        """anchor: uint32 device token [1,1] (or anything reshapeable to it). Returns uint32 [1,1,1,k] device
        tensor of greedy draft tokens for positions p+1..p+k (set_draft_pos(p) first)."""
        m = self.model
        tok = ttnn.reshape(anchor, [1, 1])
        self.rope_blk = self._rope_rows(self.blk_ids, 32)  # used by every layer
        e = ttnn.embedding(tok, m.embed, layout=ttnn.TILE_LAYOUT)
        e = m._rms(ttnn.reshape(e, [1, 1, 1, HIDDEN]), m.embed_norm)
        res = ttnn.add(self.x_base, ttnn.pad(ttnn.typecast(e, ttnn.float32), [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0))
        for i in range(N_LAYERS):
            res = self._layer(i, res)
        hid = self._rms(res, "norm")  # rows 1..7 = mask slots
        ctx = ttnn.add(self._lin(hid, "rnn_w_h", ttnn.float32), self._lin(self.f_rnn, "rnn_w_f", ttnn.float32))
        ctx = ttnn.add(ctx, ttnn.slice(self.w["rnn_b"], [0, 0, 0, 0], [1, 1, 1, 3 * RANK]))
        hb = ttnn.add(ttnn.typecast(hid, ttnn.float32), self.w["ho_b"])  # hid rows + hidden_out bias
        state = None
        prev = tok
        out, probs = [], []
        for i in range(self.k):
            pe = ttnn.reshape(ttnn.embedding(prev, self.w["markov_w1"], layout=ttnn.TILE_LAYOUT), [1, 1, 1, RANK])
            st = ttnn.zeros_like(pe) if state is None else ttnn.typecast(state, ttnn.bfloat16)
            raw = self._lin(ttnn.concat([st, pe], dim=-1), "rnn_w_sp", ttnn.float32)
            raw = ttnn.add(
                raw, ttnn.matmul(self.onehot[i + 1], ctx, dtype=ttnn.float32, compute_kernel_config=self.ckc)
            )
            gate = ttnn.sigmoid(ttnn.slice(raw, [0, 0, 0, 0], [1, 1, 1, RANK]))
            cand = ttnn.tanh(ttnn.slice(raw, [0, 0, 0, RANK], [1, 1, 1, 2 * RANK]))
            lat = ttnn.tanh(ttnn.slice(raw, [0, 0, 0, 2 * RANK], [1, 1, 1, 3 * RANK]))
            upd = ttnn.multiply(ttnn.rsub(gate, 1.0), cand)
            state = upd if state is None else ttnn.add(ttnn.multiply(gate, state), upd)
            x = ttnn.add(ttnn.matmul(self.onehot[i + 1], hb, dtype=ttnn.float32, compute_kernel_config=self.ckc),
                         self._lin(ttnn.typecast(lat, ttnn.bfloat16), "ho_w", ttnn.float32))  # fmt: skip
            lg, am = self.head(ttnn.typecast(x, ttnn.bfloat16))  # softcap 30 + vocab mask + argmax, [1,1,1,1]
            am = ttnn.clone(am)  # the head's token buffer is rewritten by the next step
            out.append(am)
            if self.top_probs:
                probs.append(self._top_prob(lg))
            prev = ttnn.reshape(am, [1, 1])
        if self.top_probs:
            self.probs = ttnn.concat(probs, dim=-1) if len(probs) > 1 else probs[0]
        return ttnn.concat(out, dim=-1) if len(out) > 1 else out[0]

    def _top_prob(self, lg):
        """Softmax probability of the argmax of one step's softcapped logits (early stop, tt/spec_generator.py):
        1 / sum(exp(l - max l)) over the real vocab. The [1, V] row is folded to [V / 32, 32] tiles (no 32-row
        padding of the whole row). Returns [1, 1, 1, 1] fp32 TILE."""
        V = lg.shape[-1]
        if self._prob_mask is None:  # first (eager) draft: allocated before any capture
            from models.demos.inkling2_mini.tt import model as tmodel

            mask = torch.zeros(V)
            mask[tmodel.VOCAB_REAL :] = -1.0e4
            self._prob_mask = ttnn.from_torch(mask.reshape(1, 1, V // 32, 32), dtype=ttnn.float32,
                                              layout=ttnn.TILE_LAYOUT, device=self.device)  # fmt: skip
        x = ttnn.to_layout(ttnn.reshape(lg, [1, 1, V // 32, 32]), ttnn.TILE_LAYOUT)
        x = ttnn.add(x, self._prob_mask)
        mx = ttnn.max(ttnn.max(x, dim=-1, keepdim=True), dim=-2, keepdim=True)
        e = ttnn.exp(ttnn.subtract(x, mx))
        return ttnn.reciprocal(ttnn.sum(ttnn.sum(e, dim=-1, keepdim=True), dim=-2, keepdim=True))

    # ------------------------------------------------------------ trace
    def capture(self, compile=True):
        """Trace draft(self.anchor). Every program must be compiled before ANY trace is captured (compile=False
        when the caller already did): programs first compiled after a capture put their program-cache buffers in
        that trace's scratch, its replay corrupts them, and the device hangs (E9-report.md)."""
        if compile:
            self.draft(self.anchor)
        ttnn.synchronize_device(self.device)
        self.trace = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.tokens = self.draft(self.anchor)
        ttnn.end_trace_capture(self.device, self.trace, cq_id=0)
        ttnn.synchronize_device(self.device)
        # rewritten by every draft replay and read right after it, before any target replay: corruptible by design
        acknowledge_corruptible(self.tokens)
        if self.top_probs:
            acknowledge_corruptible(self.probs)

    def run(self, p):
        """Traced draft at anchor position p -> k host ints."""
        self.set_draft_pos(p)
        ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=False)
        return [int(x) for x in ttnn.to_torch(self.tokens).reshape(-1)[: self.k].tolist()]

    def release(self):
        if self.trace is not None:
            ttnn.release_trace(self.device, self.trace)
            self.trace = None
