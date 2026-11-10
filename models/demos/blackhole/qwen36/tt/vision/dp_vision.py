# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vendored as is from the PUP session-5 script (worst/s5/dp_vision.py) for the media encoder. Its Q_DPV_* switches are
pinned to the deployed values by vision/encoder.py until the M2 port (M2_PLAN.md section 3) makes them constants.
Added here: Q_DPV_SELECT=rows (each chip returns its own merged rows; no output gather).

Data-parallel Qwen3.5 vision tower on a (1, N) mesh: replicated bf16 weights, each chip runs WHOLE images (no CCL
inside the tower), then one all-gather + a per-chip column select lays the merged rows out hidden-sharded for the text
model. Same math as HF Qwen3_5VisionModel (pre-LN blocks, qkv + HF rotate_half rope, windowed non-causal SDPA per
image, GELU-tanh MLP, LN + 2x2 merge + GELU MLP merger), plain ttnn ops (no new kernels).

Usage: dv = DPVision(model.vision_model, ttnn); out_rows = dv(pix, grid) -> [N_tokens, 4096/N] hidden-sharded, rows in
image order (same contract as FastVision). dv.prepare_trace(grid) / dv.capture(grid) as FastVision.
"""
import math
import os
import time

import torch


SP_MIN = int(os.environ.get("Q_DPV_SP_MIN", "2048"))  # one image with >= this many patches: sequence-parallel


def _vision_seq(U):
    if U <= 1024:
        return -(-U // 128) * 128
    if U <= 2048:
        return 2048
    return -(-U // 2048) * 2048


class DPVision:
    def __init__(self, vm, ttnn, table_rows=None):
        self.ttnn = ttnn
        self.vm = vm
        self.args = vm.model_args
        self.mesh = self.args.mesh_device
        self.n = int(self.mesh.get_num_devices())
        ref = vm.reference_model
        vc = self.args.hf_config.vision_config
        self.H = int(vc.hidden_size)  # 1152
        self.nh = int(vc.num_heads)  # 16
        self.hd = self.H // self.nh  # 72
        self.hdp = -(-self.hd // 32) * 32  # 96
        self.sm = int(vc.spatial_merge_size)
        self.out_h = int(vc.out_hidden_size)
        self.rep = ttnn.ReplicateTensorToMesh(self.mesh)
        self.shard0 = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        dram = ttnn.DRAM_MEMORY_CONFIG
        self.dram = dram
        # Q_DPV_L1=1: the tower's activations (residual stream, qkv, heads, MLP hidden) in L1 interleaved, not DRAM
        self.act_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("Q_DPV_L1", "0") == "1" else dram
        # Q_DPV_FC1_BF8=1: the MLP hidden (fc1 + gelu output, 4x the row width) written as bf8
        # Q_DPV_FC1_ACT: the block MLP fused activation (HF hidden_act = gelu_pytorch_tanh -> "gelu_tanh"; "gelu" = erf)
        self.fc1_act = os.environ.get("Q_DPV_FC1_ACT", "gelu") or None
        self.smallm = int(os.environ.get("Q_DPV_SMALLM", "0"))
        self.sp_heads = os.environ.get("Q_DPV_SP_HEADS", "0") == "1" and self.nh % self.n == 0
        self.mixed_hp = self.sp_heads and os.environ.get("Q_DPV_MIXED_HP", "0") == "1"
        # Q_DPV_ATT_L1=1: the mixed / HP attention tensors (q, k, v after the head split, rope, bf8 casts) in L1
        self.att_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("Q_DPV_ATT_L1", "0") in ("1", "2") else None
        # Q_DPV_ATT_L1=2: also the SDPA outputs, head concats, the mixed concat and the HP gather / qkv / all-to-all
        self.att2_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("Q_DPV_ATT_L1", "0") == "2" else None
        # Q_DPV_QKV_L1=1: the whole-image qkv linear output in L1; Q_DPV_FC1_L1=1: the mixed block's fc1 output in L1
        self.qkv_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("Q_DPV_QKV_L1", "0") == "1" else None
        self.fc1_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("Q_DPV_FC1_L1", "0") == "1" else None
        # Q_DPV_SPLIT_L1=1: the mixed block's LN1 row split (whole-image / split-image rows) in L1
        self.split_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("Q_DPV_SPLIT_L1", "0") == "1" else None
        self.fc1_kw = {"dtype": ttnn.bfloat8_b} if os.environ.get("Q_DPV_FC1_BF8", "0") == "1" else {}
        # Q_DPV_L1=2: only the [S, 1152] row stream (residual, LN outputs, proj / fc2 outputs) in L1; the wide matmul
        # outputs (qkv, fc1) and the attention tensors stay DRAM (L1 room for the matmul CBs)
        self.row_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("Q_DPV_L1", "0") == "2" else self.act_mc
        arch = self.mesh.arch()
        fid = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}[
            os.environ.get("Q_DPV_FID", "hifi2")
        ]
        self.ck = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, fp32_dest_acc_en=os.environ.get("Q_DPV_FP32ACC", "1") == "1", packer_l1_acc=True
        )
        # Q_DPV_FID_MLP / Q_DPV_FP32ACC_MLP (session 5): the block MLP linears' (fc1, fc2) own fidelity / fp32 dest acc
        _fm = os.environ.get("Q_DPV_FID_MLP", "")
        self.ck_mlp = (
            self.ck
            if not _fm and os.environ.get("Q_DPV_FP32ACC_MLP", "") == ""
            else ttnn.init_device_compute_kernel_config(
                arch,
                math_fidelity={
                    "lofi": ttnn.MathFidelity.LoFi,
                    "hifi2": ttnn.MathFidelity.HiFi2,
                    "hifi4": ttnn.MathFidelity.HiFi4,
                }[_fm or os.environ.get("Q_DPV_FID", "hifi2")],
                fp32_dest_acc_en=os.environ.get("Q_DPV_FP32ACC_MLP", os.environ.get("Q_DPV_FP32ACC", "1")) == "1",
                packer_l1_acc=True,
            )
        )
        wdt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("Q_DPV_WDT", "bf16")]
        self.ck_ln = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # Q_DPV_SDPA_FID (hifi4 default = ck_ln): the tower SDPA's math fidelity
        _sf = os.environ.get("Q_DPV_SDPA_FID", "hifi4")
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity={
                "lofi": ttnn.MathFidelity.LoFi,
                "hifi2": ttnn.MathFidelity.HiFi2,
                "hifi4": ttnn.MathFidelity.HiFi4,
            }[_sf],
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.ck_sel = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )

        def W(t, dtype=wdt):  # [out, in] torch -> [1, 1, in, out] replicated TILE
            t = t.detach().float().t().contiguous()
            return ttnn.from_torch(
                t.reshape(1, 1, *t.shape),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.rep,
                memory_config=dram,
            )

        def B(t):
            t = t.detach().float().reshape(1, 1, 1, -1)
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.rep,
                memory_config=dram,
            )

        def LN(t):
            t = t.detach().float().reshape(1, 1, -1).expand(1, 32, -1).contiguous()
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.rep,
                memory_config=dram,
            )

        pw = ref.patch_embed.proj.weight.detach().float()
        self.pdim = int(pw[0].numel())
        # Q_DPV_PIX_T1=1: an IMAGE's patch is its frame repeated over the temporal patch (HF tiles images to
        # temporal_patch_size frames): x . W = x_frame . (sum_t W_t) -> half the pixel bytes (host copy + H2D) and half the
        # patch-embed K. Weight [H, C, T, P*P] summed over T in fp32. (Images only: video frames differ.)
        self.pix_t1 = os.environ.get("Q_DPV_PIX_T1", "0") == "1"
        if self.pix_t1:
            _c, _t = int(pw.shape[1]), int(pw.shape[2])
            self._pix_ct = (_c, _t, self.pdim // (_c * _t))
            pw = pw.reshape(self.H, _c, _t, -1).sum(2)
            self.pdim = int(pw[0].numel())
        self.w_patch = W(pw.reshape(self.H, -1))
        self.b_patch = B(ref.patch_embed.proj.bias)
        hd, hdp, nh = self.hd, self.hdp, self.nh

        half = hd // 2
        # rope: "llama" = ttnn.experimental.rotary_embedding_llama (the tower's own op) on META-interleaved q/k dims
        # (dim 2i = HF dim i, 2i+1 = HF dim i+half; pad 72 -> 96 at the end); "matmul" = x*cos + (x @ R)*sin with the
        # pad in the middle of each half so R pairs i <-> i + hdp/2.
        self.rope_mode = os.environ.get("Q_DPV_ROPE", "llama")
        if self.rope_mode == "llama":
            self.head_idx = torch.arange(hd)
            self.qk_perm = torch.stack([torch.arange(half), torch.arange(half, hd)], dim=1).reshape(-1)
        else:
            self.head_idx = torch.cat([torch.arange(half), torch.arange(hdp // 2, hdp // 2 + half)])
            self.qk_perm = torch.arange(hd)

        def pad_heads_out(w, b, perm=None):  # rows grouped [nh, hd] -> [nh, hdp] (zero rows at the pad slots)
            w = w.detach().float().reshape(nh, hd, -1)
            b = b.detach().float().reshape(nh, hd)
            if perm is not None:
                w, b = w[:, perm], b[:, perm]
            wp = torch.zeros(nh, hdp, w.shape[-1])
            wp[:, self.head_idx] = w
            bp = torch.zeros(nh, hdp)
            bp[:, self.head_idx] = b
            return wp.reshape(nh * hdp, -1), bp.reshape(-1)

        self.blocks = []
        for blk in ref.blocks:
            qkv_w = blk.attn.qkv.weight.detach().float()
            qkv_b = blk.attn.qkv.bias.detach().float()
            parts_w, parts_b = [], []
            for j in range(3):
                w_, b_ = pad_heads_out(
                    qkv_w[j * self.H : (j + 1) * self.H],
                    qkv_b[j * self.H : (j + 1) * self.H],
                    perm=self.qk_perm if j < 2 else None,
                )
                parts_w.append(w_)
                parts_b.append(b_)
            # nlp_create_qkv_heads expects [q (nh*hdp) | k (nh*hdp) | v (nh*hdp)]
            pw_ = blk.attn.proj.weight.detach().float().reshape(self.H, nh, hd)
            proj_w = torch.zeros(self.H, nh, hdp)
            proj_w[:, :, self.head_idx] = pw_
            proj_w = proj_w.reshape(self.H, nh * hdp)
            f1w = blk.mlp.linear_fc1.weight.detach().float()
            f1b = blk.mlp.linear_fc1.bias.detach().float()
            inter = int(f1w.shape[0])
            interp = -(-inter // 32) * 32
            f1w = torch.nn.functional.pad(f1w, (0, 0, 0, interp - inter))
            f1b = torch.nn.functional.pad(f1b, (0, interp - inter))
            f2w = torch.nn.functional.pad(blk.mlp.linear_fc2.weight.detach().float(), (0, interp - inter))
            hp_kw = {}
            if self.sp_heads:
                hpc = nh // self.n
                wq = torch.cat(parts_w)  # [3 * nh * hdp, H] rows: q heads | k heads | v heads
                bq = torch.cat(parts_b)
                ws, bs = [], []
                for d in range(self.n):
                    rows = torch.cat(
                        [
                            torch.arange(j * nh * hdp + d * hpc * hdp, j * nh * hdp + (d + 1) * hpc * hdp)
                            for j in range(3)
                        ]
                    )
                    ws.append(wq[rows].t().contiguous())
                    bs.append(bq[rows])
                hp_kw = dict(
                    qkv_hp_w=ttnn.from_torch(
                        torch.stack(ws).reshape(self.n, 1, self.H, -1),
                        dtype=wdt,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        mesh_mapper=self.shard0,
                        memory_config=dram,
                    ),
                    qkv_hp_b=ttnn.from_torch(
                        torch.stack(bs).reshape(self.n, 1, 1, -1),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        mesh_mapper=self.shard0,
                        memory_config=dram,
                    ),
                )
            self.blocks.append(
                dict(
                    **hp_kw,
                    ln1w=LN(blk.norm1.weight),
                    ln1b=LN(blk.norm1.bias),
                    ln2w=LN(blk.norm2.weight),
                    ln2b=LN(blk.norm2.bias),
                    qkv_w=W(torch.cat(parts_w)),
                    qkv_b=B(torch.cat(parts_b)),
                    proj_w=W(proj_w),
                    proj_b=B(blk.attn.proj.bias),
                    f1w=W(f1w),
                    f1b=B(f1b),
                    f2w=W(f2w),
                    f2b=B(blk.mlp.linear_fc2.bias),
                )
            )
        m = ref.merger
        self.m_lnw, self.m_lnb = LN(m.norm.weight), LN(m.norm.bias)
        self.m_f1w, self.m_f1b = W(m.linear_fc1.weight), B(m.linear_fc1.bias)
        self.m_f2w, self.m_f2b = W(m.linear_fc2.weight), B(m.linear_fc2.bias)
        # HF rotate_half as a matrix on the padded head dim: (x @ R)[:half] = -x[half:hd], [half:hd] = x[:half]
        h2 = hdp // 2
        R = torch.zeros(hdp, hdp)
        for i in range(h2):
            R[h2 + i, i] = -1.0
            R[i, h2 + i] = 1.0
        self.R = ttnn.from_torch(
            R.reshape(1, 1, hdp, hdp),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=self.rep,
        )
        # per-chip column select for the hidden-sharded output: chip d keeps columns [d*w, (d+1)*w)
        w = self.out_h // self.n
        eye = torch.eye(self.out_h).reshape(1, 1, self.out_h, self.out_h)
        self.sel_cols = ttnn.from_torch(
            eye,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(None, -1), mesh_shape=self.args.cluster_shape),
            memory_config=dram,
        )
        self.scale = 1.0 / math.sqrt(hd)
        if self.rope_mode == "llama":
            from models.tt_transformers.tt.common import get_rot_transformation_mat

            self.trans = ttnn.from_torch(
                get_rot_transformation_mat(hd),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.rep,
            )
        sq = os.environ.get("Q_DPV_SDPA", "64,1024")
        g = self.mesh.compute_with_storage_grid_size()
        self.lin_kw = {"core_grid": ttnn.CoreGrid(y=g.y, x=g.x)} if os.environ.get("Q_DPV_GRID", "0") == "1" else {}
        self.sdpa_chunks = tuple(int(v) for v in sq.split(",")) if sq else None
        # Q_DPV_SDPA_SP=q,k: the single-image sequence-parallel SDPA's chunks (q rows = this chip's slice, k = all rows)
        _ssp = os.environ.get("Q_DPV_SDPA_SP", "")
        self.sdpa_sp_chunks = tuple(int(v) for v in _ssp.split(",")) if _ssp else None
        from models.tt_transformers.tt.common import Mode

        self._sdpa_cfg = lambda S: self.vm.tt_model.blocks[0].attention.configuration.get_attn_sdpa_program_config(
            Mode.PREFILL, S, None, None
        )
        self.cache = {}
        self.table = None
        if table_rows:
            self.set_table(table_rows)

    def _lin(self, x, w, b, mc, act=None, extra=None, ck=None):
        """Tower linear. Q_DPV_SMALLM=<rows>: a single [1, 1, M, K] input with M <= rows (the single-image
        sequence-parallel slices) runs an explicit 2D multicast config on 12 x min(M/32, 8) cores (ttnn's
        auto config picks ~72 cores there); otherwise ttnn's config on the full grid (lin_kw)."""
        ttnn = self.ttnn
        extra = extra or {}
        ck = ck or self.ck
        M, K, N = int(x.shape[-2]), int(x.shape[-1]), int(w.shape[-1])
        if 0 < self.smallm >= M and int(x.shape[0]) * int(x.shape[1]) == 1 and M % 32 == 0:
            mt, kt, nt = M // 32, K // 32, N // 32
            gy = min(mt, 8)
            pm, pn = -(-mt // gy), -(-nt // 12)
            bw = 4
            while kt % bw:
                bw -= 1
            fa = None
            if act == "gelu_tanh":
                fa = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
            elif act == "gelu":
                fa = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)
            elif act is not None:
                raise ValueError(act)
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(12, gy),
                in0_block_w=bw,
                out_subblock_h=1,
                out_subblock_w=max(d for d in (4, 3, 2, 1) if pn % d == 0),
                out_block_h=pm,
                out_block_w=pn,
                per_core_M=pm,
                per_core_N=pn,
                transpose_mcast=False,
                fused_activation=fa,
                fuse_batch=False,
            )
            return ttnn.linear(x, w, bias=b, compute_kernel_config=ck, memory_config=mc, program_config=pc, **extra)
        return ttnn.linear(
            x, w, bias=b, activation=act, compute_kernel_config=ck, memory_config=mc, **self.lin_kw, **extra
        )

    def _sdpa_sp_program(self, S, Sk):
        q, k = self.sdpa_sp_chunks
        _big = os.environ.get("Q_DPV_SDPA_SP_BIG", "")  # q,k for slices of > 256 rows (1 MP: 1024 rows per chip)
        if _big and S > 256:
            q, k = (int(v) for v in _big.split(","))
        return self.ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=min(q, S),
            k_chunk_size=min(k, Sk),
            exp_approx_mode=os.environ.get("Q_DPV_EXP_APPROX", "0") == "1",
        )

    def _sdpa_program(self, S, Sk=None):
        if self.sdpa_chunks is None:
            return self._sdpa_cfg(S)
        q, k = self.sdpa_chunks
        Sk = S if Sk is None else Sk
        return self.ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=min(q, S),
            k_chunk_size=min(k, Sk),
            exp_approx_mode=os.environ.get("Q_DPV_EXP_APPROX", "0") == "1",
        )

    def _pad_cs(self, c, s_):
        """HF cos/sin [U, hd] -> [U, hdp] in the padded head layout (pad slots: cos 1, sin 0)."""
        U = c.shape[0]
        cp = torch.ones(U, self.hdp)
        sp = torch.zeros(U, self.hdp)
        c, s_ = c.float(), s_.float()
        if self.rope_mode == "llama":
            c, s_ = c[:, self.qk_perm], s_[:, self.qk_perm]  # meta order (cos[i] == cos[i + half] in HF)
        cp[:, self.head_idx] = c
        sp[:, self.head_idx] = s_
        return cp, sp

    def set_table(self, rows):
        ttnn = self.ttnn
        self.table_rows = rows
        self.table = ttnn.from_torch(
            torch.zeros(rows, self.out_h, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, dims=(None, -1), mesh_shape=self.args.cluster_shape),
            memory_config=self.dram,
        )

    # ---------------------------------------------------------------- per grid-set host constants (cached)
    def _prep(self, grid):
        from models.demos.blackhole.qwen36.tt.vision import model as VM

        ttnn = self.ttnn
        grid = [g.reshape(1, 3) for g in grid]
        sizes = [int(g.prod()) for g in grid]
        if (
            len(grid) == 1
            and sizes[0] >= SP_MIN
            and sizes[0] % (self.n * 128) == 0
            and os.environ.get("Q_DPV_SP", "1") == "1"
        ):
            return self._prep_sp(grid, sizes)
        if (
            os.environ.get("Q_DPV_MIXED", "0") == "1"
            and len(grid) > self.n
            and len(grid) % self.n != 0
            and len({tuple(g.reshape(-1).tolist()) for g in grid}) == 1
            and sizes[0] % (self.n * 32) == 0
            and (sizes[0] // self.n) % (self.sm * self.sm) == 0
        ):
            return self._prep_mixed(grid, sizes)
        if (
            len(grid) > self.n
            and len({tuple(g.reshape(-1).tolist()) for g in grid}) == 1
            and _vision_seq(sizes[0]) == sizes[0]
            and os.environ.get("Q_DPV_BATCH", "1") == "1"
        ):
            return self._prep_batch(grid, sizes)
        # assign images to chips: largest first onto the least loaded chip (keeps per-chip patch counts close)
        load = [0] * self.n
        owner = [0] * len(grid)
        for i in sorted(range(len(grid)), key=lambda i: -sizes[i]):
            d = min(range(self.n), key=lambda d: load[d])
            owner[i] = d
            load[d] += sizes[i]
        S = _vision_seq(max(max(load), 32))
        per_dev = [[i for i in range(len(grid)) if owner[i] == d] for d in range(self.n)]
        pos_all, cos_all, sin_all, cu_all = [], [], [], []
        ncu = max(len(p) for p in per_dev) + 2
        with torch.no_grad():
            pos_i, cos_i, sin_i = {}, {}, {}
            for i, g in enumerate(grid):
                _cu, (c, s_) = VM.qwen3_5_vision_transformer_preprocess(
                    seq_len=sizes[i], grid_thw=g, head_dim=self.args.head_dim, spatial_merge_size=self.sm
                )
                pos_i[i] = self.vm.reference_model.fast_pos_embed_interpolate(g).float()
                cos_i[i], sin_i[i] = self._pad_cs(c, s_)
        row_of = {}  # image i -> (chip, first merged row on that chip)
        for d in range(self.n):
            ps, cs, ss, cu = [], [], [], [0]
            for i in per_dev[d]:
                row_of[i] = (d, cu[-1] // (self.sm * self.sm))
                ps.append(pos_i[i])
                cs.append(cos_i[i])
                ss.append(sin_i[i])
                cu.append(cu[-1] + sizes[i])
            U = cu[-1]
            pos = torch.cat(ps) if ps else torch.zeros(0, self.H)
            cos = torch.cat(cs) if cs else torch.zeros(0, self.hdp)
            sin = torch.cat(ss) if ss else torch.zeros(0, self.hdp)
            pos_all.append(torch.nn.functional.pad(pos, (0, 0, 0, S - U)))
            cos_all.append(torch.nn.functional.pad(cos, (0, 0, 0, S - U), value=1.0))
            sin_all.append(torch.nn.functional.pad(sin, (0, 0, 0, S - U), value=0.0))
            if S > U:
                cu.append(S)
            cu = cu + [S] * (ncu - len(cu))
            cu_all.append(torch.tensor(cu, dtype=torch.int32))
        rows_per_chip = S // (self.sm * self.sm)
        # output row order (image order) inside the all-gathered [n * rows_per_chip] rows
        order = []
        for i in range(len(grid)):
            d, r0 = row_of[i]
            base = d * rows_per_chip + r0
            order.extend(range(base, base + sizes[i] // (self.sm * self.sm)))
        sh = self.shard0
        ent = dict(
            S=S,
            sizes=sizes,
            order=torch.tensor(order, dtype=torch.long),
            per_dev=per_dev,
            rows_per_chip=rows_per_chip,
            pos=ttnn.from_torch(
                torch.stack(pos_all).unsqueeze(1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=sh,
                memory_config=self.dram,
            ),
            cos=ttnn.from_torch(
                torch.stack(cos_all).unsqueeze(1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=sh,
            ),
            sin=ttnn.from_torch(
                torch.stack(sin_all).unsqueeze(1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=sh,
            ),
            cu=ttnn.from_torch(
                torch.cat(cu_all), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=sh
            ),
            windows=max(len(p) + (1 if sum(sizes[i] for i in p) < S else 0) for p in per_dev),
        )
        self.cache[tuple(tuple(int(v) for v in g.reshape(-1)) for g in grid)] = ent
        return ent

    def _prep_batch(self, grid, sizes):
        """Same-size images, more than one per chip: chip d runs B = ceil(n_img / n) images as a BATCH (no
        concatenation, no SDPA windows: per-image attention is the batched SDPA). Image i -> chip i % n, slot i // n;
        empty slots are zero images whose rows are never read."""
        from models.demos.blackhole.qwen36.tt.vision import model as VM

        ttnn = self.ttnn
        g = grid[0]
        S = sizes[0]
        B = -(-len(grid) // self.n)
        with torch.no_grad():
            _cu, (c, s_) = VM.qwen3_5_vision_transformer_preprocess(
                seq_len=S, grid_thw=g, head_dim=self.args.head_dim, spatial_merge_size=self.sm
            )
            pos = self.vm.reference_model.fast_pos_embed_interpolate(g).float()
        cos, sin = self._pad_cs(c, s_)
        rpi = S // (self.sm * self.sm)  # merged rows per image
        order = []
        for i in range(len(grid)):
            d, b = i % self.n, i // self.n
            base = (d * B + b) * rpi
            order.extend(range(base, base + rpi))
        rep = self.rep
        ent = dict(
            S=S,
            B=B,
            sizes=sizes,
            batch=True,
            order=torch.tensor(order, dtype=torch.long),
            per_dev=None,
            rows_per_chip=B * rpi,
            pos=ttnn.from_torch(
                pos.reshape(1, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
                memory_config=self.dram,
            ),
            cos=ttnn.from_torch(
                cos.reshape(1, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
            ),
            sin=ttnn.from_torch(
                sin.reshape(1, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
            ),
            cu=None,
            windows=1,
        )
        self.cache[tuple(tuple(int(v) for v in g.reshape(-1)) for g in grid)] = ent
        return ent

    def _prep_mixed(self, grid, sizes):
        """Same-size images, n_img = q*n + rem (0 < rem < n): every chip runs q WHOLE images (image i -> chip i % n,
        slot i // n) plus one 1/n slice of each of the rem remaining images (sequence-parallel: chip d owns patches
        [d*S/n, (d+1)*S/n) of each, its attention all-gathers their K/V). Per chip ONE row stream of q*S + rem*S/n
        patches for every row-wise op; only attention splits into the whole-image batch and the SP slices. Balances
        10 images over 4 chips as 2.5 per chip instead of 3/3/2/2 (= 3 batch slots on every chip)."""
        from models.demos.blackhole.qwen36.tt.vision import model as VM

        ttnn = self.ttnn
        g = grid[0]
        S = sizes[0]
        n = self.n
        q = len(grid) // n
        rem = len(grid) - q * n
        Sq = S // n
        T = q * S + rem * Sq
        with torch.no_grad():
            _cu, (c_, s_) = VM.qwen3_5_vision_transformer_preprocess(
                seq_len=S, grid_thw=g, head_dim=self.args.head_dim, spatial_merge_size=self.sm
            )
            pos = self.vm.reference_model.fast_pos_embed_interpolate(g).float()
        cos, sin = self._pad_cs(c_, s_)
        mm = self.sm * self.sm
        rpi = S // mm  # merged rows per whole image
        rq = Sq // mm  # merged rows per SP slice
        rows_per_chip = T // mm
        order = []
        for i in range(len(grid)):
            if i < q * n:
                d, b = i % n, i // n
                base = d * rows_per_chip + b * rpi
                order.extend(range(base, base + rpi))
            else:
                j = i - q * n
                for d in range(n):
                    base = d * rows_per_chip + q * rpi + j * rq
                    order.extend(range(base, base + rq))
        pos_all = torch.stack([torch.cat([pos] * q + [pos[d * Sq : (d + 1) * Sq]] * rem) for d in range(n)])
        cos_sp = torch.stack([cos[d * Sq : (d + 1) * Sq] for d in range(n)])
        sin_sp = torch.stack([sin[d * Sq : (d + 1) * Sq] for d in range(n)])
        rep, sh = self.rep, self.shard0
        ent = dict(
            S=T,
            S_img=S,
            q=q,
            rem=rem,
            Sq=Sq,
            sizes=sizes,
            mixed=True,
            order=torch.tensor(order, dtype=torch.long),
            per_dev=None,
            rows_per_chip=rows_per_chip,
            pos=ttnn.from_torch(
                pos_all.unsqueeze(1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=sh,
                memory_config=self.dram,
            ),
            cos=ttnn.from_torch(
                cos.reshape(1, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
            ),
            sin=ttnn.from_torch(
                sin.reshape(1, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
            ),
            cos_sp=ttnn.from_torch(
                cos_sp.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=sh
            ),
            sin_sp=ttnn.from_torch(
                sin_sp.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=sh
            ),
            cu=None,
            windows=1,
        )
        self.cache[tuple(tuple(int(v) for v in g_.reshape(-1)) for g_ in grid)] = ent
        return ent

    def _prep_sp(self, grid, sizes):
        """ONE large image split by sequence over the chips: chip d owns patches [d*S, (d+1)*S); every per-row op is
        local, attention gathers K and V (the whole image) per block. Output rows are chip-major = image order."""
        from models.demos.blackhole.qwen36.tt.vision import model as VM

        ttnn = self.ttnn
        g = grid[0]
        U = sizes[0]
        S = U // self.n
        with torch.no_grad():
            _cu, (c, s_) = VM.qwen3_5_vision_transformer_preprocess(
                seq_len=U, grid_thw=g, head_dim=self.args.head_dim, spatial_merge_size=self.sm
            )
            pos = self.vm.reference_model.fast_pos_embed_interpolate(g).float()
        cos, sin = self._pad_cs(c, s_)
        sh = self.shard0
        ent = dict(
            S=S,
            sizes=sizes,
            sp=True,
            order=torch.arange(U // (self.sm * self.sm)),
            per_dev=None,
            rows_per_chip=S // (self.sm * self.sm),
            pos=ttnn.from_torch(
                pos.reshape(self.n, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=sh,
                memory_config=self.dram,
            ),
            cos=ttnn.from_torch(
                cos.reshape(self.n, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=sh,
            ),
            sin=ttnn.from_torch(
                sin.reshape(self.n, 1, S, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=sh,
            ),
            cu=None,
            windows=1,
        )
        if self.sp_heads:
            ent["cos_all"] = ttnn.from_torch(
                cos.reshape(1, 1, U, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.rep,
            )
            ent["sin_all"] = ttnn.from_torch(
                sin.reshape(1, 1, U, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.rep,
            )
        self.cache[tuple(tuple(int(v) for v in g.reshape(-1)) for g in grid)] = ent
        return ent

    def _entry(self, grid):
        key = tuple(tuple(int(v) for v in g.reshape(-1)) for g in grid)
        return self.cache.get(key) or self._prep(list(grid))

    def _host_pixels(self, pix, c):
        """[n, 1, S, pdim] host tensor: chip d's images' patches back to back (zero padded), sharded over dim 0."""
        S = c["S"]
        t1v = None
        if self.pix_t1:
            _c, _t, _pp = self._pix_ct
            t1v = pix.reshape(-1, _c, _t, _pp)[:, :, 0, :]  # [N, C, P*P] view: the first frame of every patch
            if not c.get("mixed"):
                pix = t1v.reshape(-1, self.pdim)  # (copy) the non-mixed paths take a plain [N, pdim]
        if c.get("sp"):
            x = pix.to(torch.bfloat16).reshape(self.n, 1, S, self.pdim)
            return self.ttnn.from_torch(
                x, dtype=self.ttnn.bfloat16, layout=self.ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard0
            )
        if c.get("mixed"):
            x = c.get("_hostbuf")
            if x is None:
                x = c["_hostbuf"] = torch.zeros(self.n, 1, S, self.pdim, dtype=torch.bfloat16)
            Si, q, Sq = c["S_img"], c["q"], c["Sq"]
            if t1v is not None:
                pv = t1v.reshape(-1, Si, *t1v.shape[1:])  # [n_img, Si, C, P*P] view
                xv = x.view(self.n, 1, S, *t1v.shape[1:])
            else:
                pv = pix.reshape(-1, Si, self.pdim)
                xv = x
            for i in range(pv.shape[0]):
                if i < q * self.n:
                    d, b = i % self.n, i // self.n
                    xv[d, 0, b * Si : (b + 1) * Si].copy_(pv[i])
                else:
                    j = i - q * self.n
                    for d in range(self.n):
                        o = q * Si + j * Sq
                        xv[d, 0, o : o + Sq].copy_(pv[i, d * Sq : (d + 1) * Sq])
            return self.ttnn.from_torch(
                x, dtype=self.ttnn.bfloat16, layout=self.ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard0
            )
        if c.get("batch"):
            B = c["B"]
            if os.environ.get("Q_DPV_FASTHOST", "0") == "1":
                # persistent host buffer (no zeros per request; slots of absent images keep stale rows that are
                # never read), one fused fp32->bf16 copy per image, then one from_torch
                x = c.get("_hostbuf")
                if x is None:
                    x = c["_hostbuf"] = torch.zeros(self.n, B, S, self.pdim, dtype=torch.bfloat16)
                pv = pix.reshape(-1, S, self.pdim)
                for i in range(pv.shape[0]):
                    x[i % self.n, i // self.n].copy_(pv[i])
            else:
                x = torch.zeros(self.n, B, S, self.pdim, dtype=torch.bfloat16)
                pv = pix.to(torch.bfloat16).reshape(-1, S, self.pdim)
                for i in range(pv.shape[0]):
                    x[i % self.n, i // self.n] = pv[i]
            return self.ttnn.from_torch(
                x.reshape(self.n * B, 1, S, self.pdim),
                dtype=self.ttnn.bfloat16,
                layout=self.ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.shard0,
            )
        offs = [0]
        for s in c["sizes"]:
            offs.append(offs[-1] + s)
        x = torch.zeros(self.n, 1, S, self.pdim, dtype=torch.bfloat16)
        for d, imgs in enumerate(c["per_dev"]):
            r = 0
            for i in imgs:
                k = c["sizes"][i]
                x[d, 0, r : r + k] = pix[offs[i] : offs[i] + k].to(torch.bfloat16)
                r += k
        return self.ttnn.from_torch(
            x, dtype=self.ttnn.bfloat16, layout=self.ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard0
        )

    # ---------------------------------------------------------------- device body
    def _block(self, x, p, c):
        ttnn = self.ttnn
        S = c["S"]
        dram = self.act_mc
        rmc = self.row_mc
        h = ttnn.layer_norm(
            x, weight=p["ln1w"], bias=p["ln1b"], epsilon=1e-6, compute_kernel_config=self.ck_ln, memory_config=rmc
        )
        if c.get("mixed") and os.environ.get("Q_DPV_MIXED_SPLITH", "1") == "1":
            # slice the narrow LN output (not the 4x wider qkv) into the whole-image and SP row blocks
            Hh = int(h.shape[-1])
            r0 = c["q"] * c["S_img"]
            ha = ttnn.slice(h, (0, 0, 0, 0), (1, 1, r0, Hh), memory_config=self.split_mc or dram)
            hb = ttnn.slice(h, (0, 0, r0, 0), (1, 1, c["S"], Hh), memory_config=self.split_mc or dram)
            ttnn.deallocate(h)
            qa = ttnn.linear(
                ha,
                p["qkv_w"],
                bias=p["qkv_b"],
                compute_kernel_config=self.ck,
                memory_config=self.qkv_mc or dram,
                **self.lin_kw,
            )
            ttnn.deallocate(ha)
            if self.mixed_hp:
                # Q_DPV_MIXED_HP=1: the split images' attention runs HEAD-parallel (see _mixed_hp_attn); hand over
                # their LN rows instead of their qkv
                return self._block_mixed_tail(x, (qa, ("hp", hb)), p, c)
            qb = ttnn.linear(
                hb, p["qkv_w"], bias=p["qkv_b"], compute_kernel_config=self.ck, memory_config=dram, **self.lin_kw
            )
            ttnn.deallocate(hb)
            return self._block_mixed_tail(x, (qa, qb), p, c)
        if c.get("sp") and self.sp_heads:
            o2 = self._attn_sp_heads(h, p, c)
            ttnn.deallocate(h)
            return self._block_tail(x, o2, p, c)
        qkv = self._lin(h, p["qkv_w"], p["qkv_b"], dram)
        ttnn.deallocate(h)
        if c.get("mixed"):
            return self._block_mixed_tail(x, qkv, p, c)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.nh, num_kv_heads=self.nh, transpose_k_heads=False, memory_config=dram
        )
        ttnn.deallocate(qkv)

        def rope(t):
            if self.rope_mode == "llama":
                Bt = int(t.shape[0])
                tt_ = ttnn.reshape(t, (1, Bt * int(t.shape[1]), int(t.shape[2]), int(t.shape[3]))) if Bt > 1 else t
                o = ttnn.experimental.rotary_embedding_llama(tt_, c["cos"], c["sin"], self.trans, is_decode_mode=False)
                ttnn.deallocate(t)
                if Bt > 1:
                    o = ttnn.reshape(o, (Bt, self.nh, int(o.shape[2]), int(o.shape[3])))
                return o
            tr = ttnn.matmul(t, self.R, compute_kernel_config=self.ck_sel, memory_config=dram)
            a = ttnn.multiply(t, c["cos"], memory_config=dram)
            b = ttnn.multiply(tr, c["sin"], memory_config=dram)
            ttnn.deallocate(tr)
            o = ttnn.add(a, b, memory_config=dram)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            ttnn.deallocate(t)
            return o

        q = rope(q)
        k = rope(k)
        if os.environ.get("Q_DPV_KV_BF8", "0") == "1":  # bf8 K/V (and Q with =2) into the SDPA: halves its K/V reads
            k8 = ttnn.typecast(k, ttnn.bfloat8_b)
            v8 = ttnn.typecast(v, ttnn.bfloat8_b)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k, v = k8, v8
        if c.get("sp") and os.environ.get("Q_DPV_SP_ONEGATHER", "0") == "1":
            # one all-gather for K and V (heads stacked), then split (as the mixed path)
            nkv = int(k.shape[1])
            kv = ttnn.concat([k, v], dim=1, memory_config=dram)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            kvg = ttnn.all_gather(kv, dim=2, memory_config=dram)
            ttnn.deallocate(kv)
            B_, _, L_, D_ = (int(d_) for d_ in kvg.shape)
            k = ttnn.slice(kvg, (0, 0, 0, 0), (B_, nkv, L_, D_), memory_config=dram)
            v = ttnn.slice(kvg, (0, nkv, 0, 0), (B_, 2 * nkv, L_, D_), memory_config=dram)
            ttnn.deallocate(kvg)
        elif c.get("sp"):
            kg = ttnn.all_gather(k, dim=2, memory_config=dram)
            vg = ttnn.all_gather(v, dim=2, memory_config=dram)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k, v = kg, vg
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            cu_window_seqlens=c["cu"] if c["windows"] > 1 else None,  # (sp: one image, no window)
            program_config=(
                self._sdpa_sp_program(S, int(k.shape[2]))
                if c.get("sp") and self.sdpa_sp_chunks
                else self._sdpa_program(S, int(k.shape[2]) if os.environ.get("Q_DPV_SP_KCHUNK", "0") == "1" else None)
            ),
            compute_kernel_config=self.ck_sdpa,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o2 = ttnn.experimental.nlp_concat_heads(o, memory_config=dram)
        ttnn.deallocate(o)
        return self._block_tail(x, o2, p, c)

    def _block_tail(self, x, o2, p, c):
        """Block after attention: o2 [.., S, nh*hdp] (this chip's rows) -> proj + residual + LN2 + MLP + residual."""
        ttnn = self.ttnn
        S = c["S"]
        dram = self.act_mc
        rmc = self.row_mc
        a = self._lin(o2, p["proj_w"], p["proj_b"], rmc)
        ttnn.deallocate(o2)
        x2 = ttnn.add(x, a, memory_config=rmc)
        ttnn.deallocate(a)
        ttnn.deallocate(x)
        h = ttnn.layer_norm(
            x2, weight=p["ln2w"], bias=p["ln2b"], epsilon=1e-6, compute_kernel_config=self.ck_ln, memory_config=rmc
        )
        Bx = int(x2.shape[0])
        if S >= 1024:
            h = ttnn.reshape(h, [1, Bx * S // 1024, 1024, -1])
        if os.environ.get("Q_DPV_GELU", "fused") == "separate":  # fc1 without the fused activation + one gelu op
            f = ttnn.linear(
                h, p["f1w"], bias=p["f1b"], compute_kernel_config=self.ck, memory_config=dram, **self.lin_kw
            )
            f2 = ttnn.gelu(
                f, fast_and_approximate_mode=os.environ.get("Q_DPV_GELU_FAST", "0") == "1", memory_config=dram
            )
            ttnn.deallocate(f)
            f = f2
        else:
            f = self._lin(h, p["f1w"], p["f1b"], dram, act=self.fc1_act, extra=self.fc1_kw, ck=self.ck_mlp)
        ttnn.deallocate(h)
        g = self._lin(f, p["f2w"], p["f2b"], rmc, ck=self.ck_mlp)
        ttnn.deallocate(f)
        if S >= 1024:
            g = ttnn.reshape(g, [Bx, 1, S, -1])
        out = ttnn.add(x2, g, memory_config=rmc)
        ttnn.deallocate(x2)
        ttnn.deallocate(g)
        return out

    def _attn_sp_heads(self, h, p, c):
        """Q_DPV_SP_HEADS=1, single-image split: attention HEAD-parallel instead of gathering K and V. All-gather the
        LN1 rows (narrow: 1152 wide, optionally bf8), every chip projects q/k/v of its nh/n heads for ALL rows, runs the
        full non-causal SDPA for those heads, and one all-to-all hands each chip its own rows with every head
        ([U, heads_c * hdp] -> [S, nh * hdp], source chips in head order = the proj weight's row order)."""
        ttnn = self.ttnn
        dram = self.act_mc
        hh = h
        if os.environ.get("Q_DPV_SP_HEADS_BF8", "0") == "1":
            hh = ttnn.typecast(h, ttnn.bfloat8_b, memory_config=dram)
        hg = ttnn.all_gather(hh, dim=2, memory_config=dram)
        if hh is not h:
            ttnn.deallocate(hh)
        qkv = ttnn.linear(
            hg,
            p["qkv_hp_w"],
            bias=p["qkv_hp_b"],
            compute_kernel_config=self.ck,
            memory_config=dram,
            dtype=ttnn.bfloat16,
            **self.lin_kw,
        )
        ttnn.deallocate(hg)
        hpc = self.nh // self.n
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=hpc, num_kv_heads=hpc, transpose_k_heads=False, memory_config=dram
        )
        ttnn.deallocate(qkv)
        q = self._rope_llama(q, c["cos_all"], c["sin_all"])
        k = self._rope_llama(k, c["cos_all"], c["sin_all"])
        if os.environ.get("Q_DPV_KV_BF8", "0") == "1":
            k8 = ttnn.typecast(k, ttnn.bfloat8_b)
            v8 = ttnn.typecast(v, ttnn.bfloat8_b)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k, v = k8, v8
        U = int(q.shape[2])
        _cs = os.environ.get("Q_DPV_SP_HEADS_SDPA", "128,512")
        if U <= 1024 and os.environ.get("Q_DPV_SP_HEADS_SDPA_SMALL"):  # chunks for the small images (512^2: U=1024)
            _cs = os.environ["Q_DPV_SP_HEADS_SDPA_SMALL"]
        qc, kc = (int(v_) for v_ in _cs.split(","))
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            q_chunk_size=min(qc, U),
            k_chunk_size=min(kc, U),
            exp_approx_mode=os.environ.get("Q_DPV_EXP_APPROX", "0") == "1",
        )
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scale, program_config=pc, compute_kernel_config=self.ck_sdpa
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o2 = ttnn.experimental.nlp_concat_heads(o, memory_config=dram)  # [1, 1, U, hpc * hdp]
        ttnn.deallocate(o)
        # (the layer.tokpar_a2a convention: out_dim = the dim that is split across chips, in_dim = the one that grows)
        oa = ttnn.experimental.all_to_all_async_generic(
            o2, in_dim=3, out_dim=2, num_links=2, topology=ttnn.Topology.Ring, memory_config=dram
        )
        ttnn.deallocate(o2)
        return oa  # [1, 1, S, nh * hdp]

    def _mixed_hp_attn(self, hb, p, c):
        """Q_DPV_MIXED_HP=1: attention of the rem split images (chip d holds patch slice d of each, rows
        [j*Sq, (j+1)*Sq) of hb) HEAD-parallel instead of all-gathering their K and V: all-gather the (bf8) LN rows per
        image ([rem, 1, Sq, H] -> [rem, 1, S, H], chip blocks = patch order), q/k/v of this chip's nh/n heads for every
        row (qkv_hp_w), full-image rope, SDPA, then one all-to-all back to this chip's rows with all heads
        ([rem, 1, S, heads_c * hdp] -> [rem, 1, Sq, nh * hdp]). Returns [1, 1, rem * Sq, nh * hdp]."""
        ttnn = self.ttnn
        dram = self.act_mc
        rem, Sq, S = c["rem"], c["Sq"], c["S_img"]
        H = int(hb.shape[-1])
        hh = ttnn.reshape(hb, (rem, 1, Sq, H))
        h8 = (
            ttnn.typecast(hh, ttnn.bfloat8_b, memory_config=dram)
            if os.environ.get("Q_DPV_SP_HEADS_BF8", "0") == "1"
            else hh
        )
        hg = ttnn.all_gather(h8, dim=2, memory_config=self.att2_mc or dram)
        if h8 is not hh:
            ttnn.deallocate(h8)
        ttnn.deallocate(hb)
        qkv = ttnn.linear(
            hg,
            p["qkv_hp_w"],
            bias=p["qkv_hp_b"],
            compute_kernel_config=self.ck,
            memory_config=self.att2_mc or dram,
            dtype=ttnn.bfloat16,
            **self.lin_kw,
        )
        ttnn.deallocate(hg)
        hpc = self.nh // self.n
        amc = self.att_mc or dram
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=hpc, num_kv_heads=hpc, transpose_k_heads=False, memory_config=amc
        )
        ttnn.deallocate(qkv)

        def rope(t):
            tt_ = ttnn.reshape(t, (1, rem * hpc, S, int(t.shape[3]))) if rem > 1 else t
            o = ttnn.experimental.rotary_embedding_llama(
                tt_,
                c["cos"],
                c["sin"],
                self.trans,
                is_decode_mode=False,
                **({"memory_config": amc} if self.att_mc else {}),
            )
            ttnn.deallocate(t)
            return ttnn.reshape(o, (rem, hpc, S, int(o.shape[3]))) if rem > 1 else o

        q = rope(q)
        k = rope(k)
        if os.environ.get("Q_DPV_KV_BF8", "0") == "1":
            k8 = ttnn.typecast(k, ttnn.bfloat8_b, memory_config=amc)
            v8 = ttnn.typecast(v, ttnn.bfloat8_b, memory_config=amc)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k, v = k8, v8
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=self._sdpa_program(S),
            compute_kernel_config=self.ck_sdpa,
            **({"memory_config": self.att2_mc} if self.att2_mc else {}),
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o2 = ttnn.experimental.nlp_concat_heads(o, memory_config=self.att2_mc or dram)  # [rem, 1, S, hpc * hdp]
        ttnn.deallocate(o)
        oa = ttnn.experimental.all_to_all_async_generic(
            o2, in_dim=3, out_dim=2, num_links=2, topology=ttnn.Topology.Ring, memory_config=dram
        )
        ttnn.deallocate(o2)
        return ttnn.reshape(oa, (1, 1, rem * Sq, int(oa.shape[-1])))

    def _rope_llama(self, t, cos, sin, mc=None):
        ttnn = self.ttnn
        Bt = int(t.shape[0])
        tt_ = ttnn.reshape(t, (1, Bt * int(t.shape[1]), int(t.shape[2]), int(t.shape[3]))) if Bt > 1 else t
        o = ttnn.experimental.rotary_embedding_llama(
            tt_, cos, sin, self.trans, is_decode_mode=False, **({"memory_config": mc} if mc is not None else {})
        )
        ttnn.deallocate(t)
        if Bt > 1:
            o = ttnn.reshape(o, (Bt, self.nh, int(o.shape[2]), int(o.shape[3])))
        return o

    def _block_mixed_tail(self, x, qkv, p, c):
        """Mixed-mode block after the qkv linear: whole images (batch q) + SP slices (batch rem) attention, then the
        shared row-wise tail (proj, residual, LN, MLP, residual) on the [1, 1, T, H] row stream."""
        ttnn = self.ttnn
        dram = self.act_mc
        Si, q_, rem, Sq, T = c["S_img"], c["q"], c["rem"], c["Sq"], c["S"]
        if isinstance(qkv, tuple):
            qa, qb = qkv
            W = int(qa.shape[-1])
        else:
            W = int(qkv.shape[-1])
            qa = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, q_ * Si, W), memory_config=dram)
            qb = ttnn.slice(qkv, (0, 0, q_ * Si, 0), (1, 1, T, W), memory_config=dram)
            ttnn.deallocate(qkv)
        hp_rows = None
        if isinstance(qb, tuple) and qb[0] == "hp":
            hp_rows = qb[1]
        qa = ttnn.reshape(qa, (q_, 1, Si, W))
        parts = [(qa, Si, c["cos"], c["sin"], False)]
        if hp_rows is None:
            qb = ttnn.reshape(qb, (rem, 1, Sq, W))
            parts.append((qb, Sq, c["cos_sp"], c["sin_sp"], True))
        outs = []
        for part, L, cs, sn, sp in parts:
            amc = self.att_mc or dram
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                part, num_heads=self.nh, num_kv_heads=self.nh, transpose_k_heads=False, memory_config=amc
            )
            ttnn.deallocate(part)
            q = self._rope_llama(q, cs, sn, mc=self.att_mc)
            k = self._rope_llama(k, cs, sn, mc=self.att_mc)
            if os.environ.get("Q_DPV_KV_BF8", "0") == "1":
                k8 = ttnn.typecast(k, ttnn.bfloat8_b, memory_config=amc)
                v8 = ttnn.typecast(v, ttnn.bfloat8_b, memory_config=amc)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                k, v = k8, v8
            if sp and os.environ.get("Q_DPV_KV_ONEGATHER", "1") == "1":
                # one all-gather for K and V (heads stacked), then split
                nkv = int(k.shape[1])
                kv = ttnn.concat([k, v], dim=1, memory_config=dram)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                if os.environ.get("Q_DPV_AG_TP", "0") == "1" and getattr(self, "tt_ccl", None) is not None:
                    # the text model's TP all-gather (all_gather_async, every TP link, its semaphores)
                    from models.demos.blackhole.qwen36.tt import tp_common as _tpc

                    kvg = _tpc.tp_all_gather(kv, self.tt_ccl, dim=2, memory_config=dram)
                else:
                    kvg = ttnn.all_gather(kv, dim=2, memory_config=dram)
                ttnn.deallocate(kv)
                B_, _, L_, D_ = (int(d_) for d_ in kvg.shape)
                k = ttnn.slice(kvg, (0, 0, 0, 0), (B_, nkv, L_, D_), memory_config=dram)
                v = ttnn.slice(kvg, (0, nkv, 0, 0), (B_, 2 * nkv, L_, D_), memory_config=dram)
                ttnn.deallocate(kvg)
            elif sp:
                kg = ttnn.all_gather(k, dim=2, memory_config=dram)
                vg = ttnn.all_gather(v, dim=2, memory_config=dram)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                k, v = kg, vg
            o = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.scale,
                program_config=self._sdpa_program(L),
                compute_kernel_config=self.ck_sdpa,
                **({"memory_config": self.att2_mc} if self.att2_mc else {}),
            )
            for t in (q, k, v):
                ttnn.deallocate(t)
            o2 = ttnn.experimental.nlp_concat_heads(o, memory_config=self.att2_mc or dram)  # [B, 1, L, H]
            ttnn.deallocate(o)
            outs.append(ttnn.reshape(o2, (1, 1, int(o2.shape[0]) * L, int(o2.shape[-1]))))
        if hp_rows is not None:
            outs.append(self._mixed_hp_attn(hp_rows, p, c))
        o2 = ttnn.concat(outs, dim=2, memory_config=self.att2_mc or dram)
        for t in outs:
            ttnn.deallocate(t)
        rmc = self.row_mc
        a = ttnn.linear(
            o2, p["proj_w"], bias=p["proj_b"], compute_kernel_config=self.ck, memory_config=rmc, **self.lin_kw
        )
        ttnn.deallocate(o2)
        x2 = ttnn.add(x, a, memory_config=rmc)
        ttnn.deallocate(a)
        ttnn.deallocate(x)
        h = ttnn.layer_norm(
            x2, weight=p["ln2w"], bias=p["ln2b"], epsilon=1e-6, compute_kernel_config=self.ck_ln, memory_config=rmc
        )
        f = ttnn.linear(
            h,
            p["f1w"],
            bias=p["f1b"],
            activation=self.fc1_act,
            compute_kernel_config=self.ck_mlp,
            memory_config=self.fc1_mc or dram,
            **self.lin_kw,
            **self.fc1_kw,
        )
        ttnn.deallocate(h)
        g = ttnn.linear(f, p["f2w"], bias=p["f2b"], compute_kernel_config=self.ck_mlp, memory_config=rmc, **self.lin_kw)
        ttnn.deallocate(f)
        out = ttnn.add(x2, g, memory_config=rmc)
        ttnn.deallocate(x2)
        ttnn.deallocate(g)
        return out

    def _body(self, x_rm, c):
        ttnn = self.ttnn
        dram = self.dram
        S = c["S"]
        x = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT, memory_config=dram)
        h = ttnn.linear(
            x, self.w_patch, bias=self.b_patch, compute_kernel_config=self.ck, memory_config=dram, **self.lin_kw
        )
        ttnn.deallocate(x)
        x = ttnn.add(h, c["pos"], memory_config=self.row_mc)
        ttnn.deallocate(h)
        for p in self.blocks:
            x = self._block(x, p, c)
        h = ttnn.layer_norm(x, weight=self.m_lnw, bias=self.m_lnb, epsilon=1e-6, compute_kernel_config=self.ck_ln)
        ttnn.deallocate(x)
        Bm = int(h.shape[0])
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = ttnn.reshape(h, (1, 1, Bm * S // (self.sm * self.sm), self.H * self.sm * self.sm))
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        f = ttnn.linear(
            h,
            self.m_f1w,
            bias=self.m_f1b,
            activation="gelu",
            compute_kernel_config=self.ck,
            memory_config=dram,
            **self.lin_kw,
        )
        ttnn.deallocate(h)
        y = ttnn.linear(
            f, self.m_f2w, bias=self.m_f2b, compute_kernel_config=self.ck, memory_config=dram, **self.lin_kw
        )
        ttnn.deallocate(f)
        # Q_DPV_SELECT=rows (media encoder): no gather -- every chip returns its own merged rows [rows, 4096] row-major
        # (concatenated over the chips = the chip-major row order row_order() indexes); the host reads each row once
        if os.environ.get("Q_DPV_SELECT", "matmul") == "rows":
            out = ttnn.to_layout(
                ttnn.reshape(y, (int(y.shape[-2]), int(y.shape[-1]))), ttnn.ROW_MAJOR_LAYOUT, memory_config=dram
            )
            ttnn.deallocate(y)
            return out
        # every chip: all chips' merged rows [n*rows, 4096], then its own column shard via the identity select
        if self.n == 1:
            ys = y
        else:
            if os.environ.get("Q_DPV_SELECT", "matmul") == "a2a":
                # rows (this chip's images) -> column shard of all rows in one all-to-all (1/4 of the gather's bytes)
                ys = ttnn.experimental.all_to_all_async_generic(
                    y, in_dim=2, out_dim=3, num_links=2, topology=ttnn.Topology.Ring, memory_config=dram
                )
                ttnn.deallocate(y)
                yg = None
            else:
                yg = ttnn.all_gather(y, dim=2, memory_config=dram)
                ttnn.deallocate(y)
            if yg is None:
                pass
            elif os.environ.get("Q_DPV_SELECT", "matmul") == "partition":
                # each chip keeps its column shard of the gathered rows (a per-device slice, no fabric) -- replaces
                # the identity-select matmul (bit-exact: no arithmetic)
                ys = ttnn.mesh_partition(yg, dim=-1, cluster_axis=1, memory_config=dram)
            else:
                ys = ttnn.matmul(yg, self.sel_cols, compute_kernel_config=self.ck_sel, memory_config=dram)
            if yg is not None:
                ttnn.deallocate(yg)
        out = ttnn.reshape(ys, (int(ys.shape[-2]), int(ys.shape[-1])))  # [n*rows, 4096/n], chip-major rows
        if self.table is not None:
            n_rows = int(out.shape[0])
            assert n_rows < self.table_rows, (n_rows, self.table_rows)
            padded = ttnn.pad(out, [(0, self.table_rows - n_rows), (0, 0)], value=0.0)
            rm = ttnn.to_layout(padded, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(padded)
            ttnn.copy(rm, self.table)
            ttnn.deallocate(rm)
        return out

    # ---------------------------------------------------------------- API (FastVision-compatible)
    def prepare_trace(self, grid):
        ttnn = self.ttnn
        c = self._entry(grid)
        if "xbuf" not in c:
            c["xbuf"] = ttnn.from_torch(
                torch.zeros(self.n * c.get("B", 1), 1, c["S"], self.pdim, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=self.shard0,
                memory_config=self.dram,
            )
            out = self._body(c["xbuf"], c)
            ttnn.synchronize_device(self.mesh)
            ttnn.deallocate(out)
        return c

    def capture(self, grid):
        ttnn = self.ttnn
        c = self.prepare_trace(grid)
        if c.get("tid") is None:
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            c["out"] = self._body(c["xbuf"], c)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            c["tid"] = tid
        return c

    def row_order(self, grid):
        """Index (into the chip-major output / table rows) of each merged token in image order."""
        return self._entry(grid)["order"]

    def __call__(self, pix, grid, split=None, traced=None):
        ttnn = self.ttnn
        t0 = time.perf_counter()
        c = self._entry(grid)
        host = self._host_pixels(pix, c)
        t1 = time.perf_counter()
        if traced is None:
            traced = c.get("tid") is not None
        if traced:
            ttnn.copy_host_to_device_tensor(host, c["xbuf"])
            ttnn.execute_trace(self.mesh, c["tid"], cq_id=0, blocking=False)
            out = c["out"]
        else:
            x = ttnn.to_device(host, self.mesh, memory_config=self.dram)
            out = self._body(x, c)
            ttnn.deallocate(x)
        if split is not None:
            split.append((t1 - t0, time.perf_counter() - t1))
        self._last_host = host
        return out
