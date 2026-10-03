# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSpark drafter (checkpoint ``mtp.0..2``) on the 4x8 mesh. See docs/superpowers/specs/2026-10-02-dsv41-spec-decode-design.md section 0.1.

One draft pass per user = 3 stacked blocks (mHC + window attention + 128-expert top-3 MoE + shared expert) over a block of 5 positions
``[t, noise x4]``, then the backbone's tied LM head (stage-2 norm) with a sequential markov-bigram bias that yields the 5 draft tokens.

Row layouts (per mesh row, U users):
  * draft rows ``T_d = 5 * U``: block-index-major, row ``i * U + u`` (block index i of user u): the markov step i is a contiguous slice.
  * verify rows (``write_main``): user-major ``u * n + j`` like the backbone verify step (n = 1 + k).
Stage KV cache (paged, one page per user, L_d = 192 slots): ring slots [0, 160) with slot = pos % 160 (128 + margin >= k), block slots [160, 165) for the 5
draft rows' own K/V (non-causal inside the block). The ring holds ``main_kv`` of every position the backbone verified (rejected positions beyond the
frontier are masked by the frontier-dependent mask and overwritten later).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, NH, WINDOW, DSV41Attention
from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
from models.demos.blackhole.deepseek_v41_flash.tt.loader import _fp8, rope_freqs
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import (
    HIDDEN,
    INTER,
    WEIGHT_CACHE,
    _fp4_expert,
    _fp8_linear,
    _Shards,
)
from models.demos.blackhole.deepseek_v41_flash.tt.router import ROUTED_GAIN
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import DSV41SharedExpertV2 as DSV41SharedExpert
from models.demos.blackhole.deepseek_v41_flash.tt.spec_attention import _PagedMixin

BLOCK = 5  # dspark_block_size
NOISE_ID = 128799
N_ROUTED, TOPK = 128, 3
RING = 160  # 128 + margin (k <= 5 needs 133)
L_D = 192  # ring 160 + 32 block slots (k_chunk 64)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "deepseek_v41_flash.yaml"


def _l1v(md, tag):
    """DSV41_L1_DIAG=1: L1 allocator reading (debug the static-CB clash)."""
    if os.environ.get("DSV41_L1_DIAG") == "1":
        ttnn.synchronize_device(md)
        mv = ttnn.get_memory_view(md, ttnn.BufferType.L1)
        print(
            f"L1V {tag:36s} allocated/bank {mv.total_bytes_allocated_per_bank:8d} largest_free {mv.largest_contiguous_bytes_free_per_bank:8d}",
            flush=True,
        )


# ---------------------------------------------------------------------------------------------------- weights
def mtp_cache_dir(stage):
    return os.path.join(WEIGHT_CACHE, f"mtp_{stage}") if WEIGHT_CACHE and WEIGHT_CACHE != "0" else None


def _cache_warm(stage):
    d = mtp_cache_dir(stage)
    tag = "bfp8" if os.environ.get("MOE_COMPUTE_BFP8_WEIGHTS", "0") != "0" else "bfp4"
    return bool(d) and all(
        os.path.exists(os.path.join(d, f))
        for f in (f"moe_w0_w1_{tag}.tensorbin", f"moe_w2_{tag}.tensorbin", f"moe_meta_{tag}.json")
    )


def load_mtp_moe(stage, sh, dtype=torch.bfloat16):
    p = f"mtp.{stage}.ffn."
    warm = _cache_warm(stage)
    w0 = w1 = w2 = None
    if not warm:
        w0, w1 = (torch.zeros(1, N_ROUTED, HIDDEN, INTER, dtype=dtype) for _ in range(2))
        w2 = torch.zeros(1, N_ROUTED, INTER, HIDDEN, dtype=dtype)

        def one(e):
            ep = f"{p}experts.{e}."
            w0[0, e] = _fp4_expert(sh, ep + "w1", dtype)
            w1[0, e] = _fp4_expert(sh, ep + "w3", dtype)
            w2[0, e] = _fp4_expert(sh, ep + "w2", dtype)

        with ThreadPoolExecutor(max_workers=int(os.environ.get("DSV41_LOAD_THREADS", "24"))) as ex:
            list(ex.map(one, range(N_ROUTED)))
    sp = p + "shared_experts."
    return {
        "gate_weight": sh.get(p + "gate.weight").T.contiguous().to(torch.float32),  # [hidden, 128]
        "gate_bias": sh.get(p + "gate.bias").to(torch.float32),
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "cache_dir": mtp_cache_dir(stage),
        "shared": (
            _fp8_linear(sh, sp + "w1", dtype)[None, None],
            _fp8_linear(sh, sp + "w3", dtype)[None, None],
            _fp8_linear(sh, sp + "w2", dtype)[None, None],
        ),
    }


def load_mtp_stage(stage, sh=None, with_moe=True):
    sh = sh or _Shards()
    p = f"mtp.{stage}."
    attn = {
        "wq_a": _fp8(sh, p + "attn.wq_a"),
        "q_norm": sh.get(p + "attn.q_norm.weight").float(),
        "wq_b": _fp8(sh, p + "attn.wq_b"),
        "wkv": _fp8(sh, p + "attn.wkv"),
        "kv_norm": sh.get(p + "attn.kv_norm.weight").float(),
        "wo_a": _fp8(sh, p + "attn.wo_a"),
        "wo_b": _fp8(sh, p + "attn.wo_b"),
        "attn_sink": sh.get(p + "attn.attn_sink").float(),
    }
    out = {
        "attn": attn,
        "norms": {
            "attn_norm": sh.get(p + "attn_norm.weight").float(),
            "ffn_norm": sh.get(p + "ffn_norm.weight").float(),
        },
        "mhc": {
            n: (
                sh.get(p + f"hc_{n}_fn").float(),
                sh.get(p + f"hc_{n}_base").float(),
                sh.get(p + f"hc_{n}_scale").float(),
            )
            for n in ("attn", "ffn")
        },
    }
    if with_moe:
        out["moe"] = load_mtp_moe(stage, sh)
    if stage == 0:
        out["main_proj"] = _fp8(sh, p + "main_proj")  # [5120, 15360]
        out["main_norm"] = sh.get(p + "main_norm.weight").float()
    if stage == 2:
        out["norm"] = sh.get(p + "norm.weight").float()
        out["markov_embed"] = sh.get(p + "markov_head.embed.weight")  # [129280, 256] bf16
        out["markov_head"] = sh.get(p + "markov_head.head.weight")  # [129280, 256] bf16
        out["conf"] = sh.get(p + "confidence_head.proj.weight").float()  # [1, 5376]
    return out


# ---------------------------------------------------------------------------------------------------- router / MoE (128 experts, top-3)
class DraftGate:
    """Exact fp32 router (``DSV41Gate._forward_exact_fast2`` maths) for 128 experts, top-3: no bf16 kernel path, no bias shift needed."""

    def __init__(self, md, gate_weight, gate_bias, k=TOPK, scale=1.5):
        E = gate_bias.numel()
        self.k, self.E, self.scale = k, E, scale
        rep = ttnn.ReplicateTensorToMesh(md)
        c = lambda t, dt=ttnn.float32: ttnn.from_torch(
            t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
        )
        self._w = c(gate_weight.to(torch.bfloat16).reshape(1, 1, -1, E), ttnn.bfloat16)
        self._bias = c(gate_bias.reshape(1, 1, 1, E))
        self._zero = c(torch.zeros(1, 1, 1, E))
        self._arange = c(torch.arange(E, dtype=torch.float32).reshape(1, 1, 1, E))
        self._arange_col = c(torch.arange(E, dtype=torch.float32).reshape(1, 1, E, 1))
        self._ones_k = c(torch.ones(1, 1, k, k))
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def forward(self, tt_x):
        T, E, k = tt_x.shape[2], self.E, self.k
        mc = ttnn.L1_MEMORY_CONFIG
        U, UT = ttnn.UnaryWithParam, ttnn.UnaryOpType
        acts = [U(UT.SOFTPLUS, 1.0, 20.0), U(UT.SQRT)]
        logits = ttnn.matmul(
            tt_x,
            self._w,
            compute_kernel_config=self.ckc,
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=1, x=4),
            memory_config=mc,
        )
        score = ttnn.add(logits, self._zero, input_tensor_a_activations=acts, memory_config=mc)
        ttnn.deallocate(logits)
        rank = ttnn.add(score, self._bias, memory_config=mc)
        _, idx = ttnn.topk(rank, k=k, dim=-1, largest=True, sorted=True, memory_config=mc)
        ttnn.deallocate(rank)
        idxf = ttnn.typecast(idx, ttnn.float32, memory_config=mc)
        div_kw = dict(
            input_tensor_b_activations=[U(UT.ADD_UNARY_SFPU, 1e-20)],
            activations=[U(UT.MUL_UNARY_SFPU, self.scale / ROUTED_GAIN)],
            dtype=ttnn.bfloat16,
            memory_config=mc,
        )
        onehot = ttnn.eq(
            ttnn.permute(idxf, (0, 3, 2, 1), memory_config=mc), self._arange, memory_config=mc
        )  # [1,k,T,E]
        sel = ttnn.sum(
            ttnn.multiply(onehot, score, memory_config=mc), dim=-1, keepdim=True, memory_config=mc
        )  # [1,k,T,1]
        den = ttnn.sum(sel, dim=1, keepdim=True, memory_config=mc)
        w = ttnn.permute(ttnn.div(sel, den, **div_kw), (0, 3, 2, 1), memory_config=mc)  # [1,1,T,k]
        weights = ttnn.view(
            ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG), (T, 1, 1, k)
        )
        indices = ttnn.view(
            ttnn.to_layout(
                ttnn.typecast(idx, ttnn.uint16, memory_config=mc),
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            (T, 1, 1, k),
        )
        return weights, indices


class DraftMoEBlock:
    """``DSV41MoEBlock`` for 128 routed experts / top-3 (4 experts per chip), shared expert outside ``moe_compute``."""

    def __init__(self, md, weights, batch_per_device, topology=ttnn.Topology.Linear, buffers=None):
        from models.common.modules.moe.tt_moe_decode import TTMoEDecode
        from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig

        text = CONFIG_PATH.read_text()
        text = text.replace("batch_per_device: 4 ", f"batch_per_device: {batch_per_device} ", 1)
        text = text.replace("num_routed_experts: 384", f"num_routed_experts: {N_ROUTED}").replace(
            "select_experts_k: 6", f"select_experts_k: {TOPK}"
        )
        text = text.replace("num_shared_experts: 1", "num_shared_experts: 0").replace(
            "  shared_expert_ids_to_devices: fully_replicated\n", ""
        )
        cfg = TTMoEDecodeConfig.from_yaml(text, topology=topology)
        mesh_shape = tuple(md.shape)
        if cfg.mesh_shape != mesh_shape:
            cfg = cfg.with_mesh_shape(mesh_shape)
        if cfg.batch_per_device != batch_per_device:
            cfg = cfg.model_copy(update={"batch_per_device": batch_per_device})
        if cfg.num_fast_reduce_outputs == 1:
            cfg = cfg.model_copy(
                update={"reduce": cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})}
            )
        self.md, self.decode_config = md, cfg
        self.gate = DraftGate(md, weights["gate_weight"], weights["gate_bias"])
        self.decode = TTMoEDecode(
            mesh_device=md,
            config=cfg,
            torch_w0=weights["w0"],
            torch_w1=weights["w1"],
            torch_w2=weights["w2"],
            weight_cache_dir=weights.get("cache_dir"),
            buffers=buffers,
        )

    def warmup(self):
        """Same L1 steering as ``DSV41MoEBlock.warmup`` (hole under the persistent allocations before the program's first compile)."""
        md = self.md
        nb = ttnn.get_memory_view(md, ttnn.BufferType.L1).num_banks
        tile_row = lambda n: ttnn.empty(
            [1, 1, 32, 32 * nb * n],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _l1v(md, "    warmup: start")
        hole, fence = tile_row(int(os.environ.get("DSV41_HOLE_TILES", "16"))), tile_row(
            2
        )  # hole >> the small L1 outputs of moe_compute (they eat a 2 KB hole before the semaphore is created)
        _l1v(md, "    warmup: hole+fence alloc")
        ttnn.deallocate(hole)
        _l1v(md, "    warmup: hole freed")
        T, H = self.decode_config.batch_per_device, self.decode_config.hidden_size
        zeros = lambda shape, lay: ttnn.from_torch(
            torch.zeros(shape),
            device=md,
            dtype=ttnn.bfloat16,
            layout=lay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        xg, xt = zeros([1, 1, T, H], ttnn.TILE_LAYOUT), zeros([T, 1, 1, H], ttnn.ROW_MAJOR_LAYOUT)
        _l1v(md, "    warmup: zeros uploaded")
        gw, gi = self.gate.forward(xg)
        _l1v(md, "    warmup: after gate")
        out = self.decode.forward(
            tt_x=xt,
            tt_scores=gw if gw.dtype == ttnn.bfloat16 else ttnn.typecast(gw, ttnn.bfloat16),
            tt_indices=gi,
            layer_id=0,
        )
        ttnn.synchronize_device(md)
        _l1v(md, "    warmup: after decode (out alive)")
        ttnn.deallocate(out)
        ttnn.deallocate(fence)
        _l1v(md, "    warmup: end")

    def forward(self, tt_x_gate, tt_x_tokens, forced_routing=None):
        w, idx = self.gate.forward(tt_x_gate)
        if idx.dtype != ttnn.uint16:
            idx = ttnn.typecast(idx, ttnn.uint16)
        if idx.layout != ttnn.ROW_MAJOR_LAYOUT:
            idx = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
        if w.dtype != ttnn.bfloat16:
            w = ttnn.typecast(w, ttnn.bfloat16)
        if w.layout != ttnn.ROW_MAJOR_LAYOUT:
            w = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT)
        return self.decode.forward(tt_x=tt_x_tokens, tt_scores=w, tt_indices=idx, layer_id=0)


class DraftLayer(DSV41Layer):
    """One DSpark stage: ``DSV41Layer.forward`` (mHC, attention, MoE, shared expert) with the draft attention and the 128-expert MoE."""

    def __init__(
        self, md, mesh_config, ccl, attention, norms, mhc_params, moe_weights, users_rows, eps=1e-20, moe_buffers=None
    ):
        self.mesh_device, self.mesh_config, self.ccl = md, mesh_config, ccl
        self.T, self.eps, self.debug = users_rows, eps, None
        self.attention = attention
        _l1v(md, "  layer: start")
        self.mhc_attn = DSV41MHC(md, *mhc_params["attn"])
        self.mhc_ffn = DSV41MHC(md, *mhc_params["ffn"])
        _l1v(md, "  layer: after mhc x2")
        self.moe = DraftMoEBlock(md, moe_weights, users_rows, buffers=moe_buffers)
        _l1v(md, "  layer: after moe block ctor")
        self.moe.warmup()
        _l1v(md, "  layer: after moe warmup")
        w0, w1, w2 = moe_weights["shared"]
        self.shared = DSV41SharedExpert(md, w0, w1, w2)
        _l1v(md, "  layer: after shared expert")
        up = lambda t: ttnn.from_torch(
            t.reshape(1, 1, 1, -1).float(),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        self.attn_norm_w, self.ffn_norm_w = up(norms["attn_norm"]), up(norms["ffn_norm"])


# ---------------------------------------------------------------------------------------------------- attention
class DraftAttention(_PagedMixin, DSV41Attention):
    """DSparkAttention: queries/K/V of the 5 draft rows (T_d rows) + the ring of ``main_kv``; one non-causal paged SDPA over [ring | block slots]."""

    def __init__(self, md, mesh_config, ccl, w, freqs_cis, users_per_row, n):
        T_d = BLOCK * users_per_row
        super().__init__(md, mesh_config, ccl, w, freqs_cis, users_per_row=T_d, max_seq=32)
        self.U, self.n = users_per_row, n
        self.page, self.ppu = L_D, 1
        U = users_per_row
        mk = lambda per_dev: ttnn.from_torch(
            per_dev.repeat(self.rows, 1).to(torch.int32),
            device=md,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )
        self.pt = mk(torch.arange(U).repeat(BLOCK).reshape(T_d, 1))  # draft rows i*U+u -> page u
        self.pt_v = mk(torch.arange(U).repeat_interleave(n).reshape(U * n, 1))  # verify rows u*n+j -> page u
        self.cache = self._up(torch.zeros(U, 1, L_D, HEAD_DIM))
        # row-masked block-slot indices: call i writes the rows of block index i (rows i*U .. i*U+U-1) at slot RING + i
        blk = []
        for i in range(BLOCK):
            idx = torch.full((T_d,), -1, dtype=torch.int32)
            idx[i * U : (i + 1) * U] = RING + i
            blk.append(
                ttnn.from_torch(
                    idx.repeat(self.rows),
                    device=md,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
                )
            )
        self.blk_idx = blk
        self._k_chunk = 64

    def seed_ring(self, ring, S):
        """ring [rows*U, 128, 512] reference window cache (slot = pos % 128) after S prefilled positions -> paged cache slots pos % RING."""
        full = torch.zeros(ring.shape[0], L_D, HEAD_DIM)
        for y in range(max(0, S - WINDOW), S):
            full[:, y % RING] = ring[:, y % WINDOW]
        self._seed_cache(full)

    def write_main_rows(self, kv_rows, idx_j):
        """kv_rows [1,T_v,32,512] L1 sharded (row 0 = RoPE'd main_kv of verify row t); idx_j: per block index j an int32 [T_v] slot tensor (-1 = skip)."""
        for idx in idx_j:
            ttnn.experimental.paged_update_cache(self.cache, kv_rows, update_idxs_tensor=idx, page_table=self.pt_v)

    def forward(self, x, st):
        q, kv, k = self._qkv(x, st)
        for idx in self.blk_idx:
            ttnn.experimental.paged_update_cache(self.cache, kv, update_idxs_tensor=idx, page_table=self.pt)
        ttnn.deallocate(kv)
        ttnn.deallocate(k)
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.cache,
            self.cache,
            page_table_tensor=self.pt,
            is_causal=False,
            attn_mask=st["mask"],
            attention_sink=self.sinks,
            scale=self.scale,
            program_config=self._sdpa_cfg(self._k_chunk),
            compute_kernel_config=self.ckc_sdpa,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._finish(o, st)


class DraftState:
    """Position tables of the drafter: RoPE rows (plain RoPE, theta 10000), ring slot of a verify position, frontier mask of the draft block."""

    def __init__(self, md, attn, U, n, max_pos=256):
        self.md, self.U, self.n = md, U, n
        self.T_d, self.T_v = BLOCK * U, U * n
        self.attn = attn
        P = max_pos
        pos = torch.arange(P)
        rep = ttnn.ReplicateTensorToMesh(md)
        up = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        c, s = attn._rope_inputs(pos)
        self.t = {"C": up(c), "S": up(s), "nS": up(-s), "ring": up((pos % RING).reshape(P, 1) * torch.ones(1, 32))}
        mask = torch.full((P, L_D), -1e9)
        for f in range(P):
            for sl in range(RING):
                d = (f - sl) % RING
                if d <= WINDOW - 1 and f - d >= 0:
                    mask[f, sl] = 0.0
            mask[f, RING : RING + BLOCK] = 0.0
        self.t["mask"] = up(mask)
        fl = lambda v, T: ttnn.from_torch(
            v.float().reshape(1, 1, 1, T),
            device=md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.off_d = fl(torch.arange(self.T_d) // U + 1, self.T_d)  # query position = f + 1 + i
        self.jmask = [fl((torch.arange(self.T_v) % n == j), self.T_v) for j in range(n)]

    def _rows(self, name, idx, shape):
        return ttnn.to_layout(
            ttnn.reshape(ttnn.embedding(idx, self.t[name], layout=ttnn.ROW_MAJOR_LAYOUT), shape), ttnn.TILE_LAYOUT
        )

    def _gather(self, pos, T):
        idx = ttnn.typecast(ttnn.reshape(pos, [T, 1]), ttnn.uint32)
        return {
            "Ch": self._rows("C", idx, [1, T, 1, HEAD_DIM]),
            "Sh": self._rows("S", idx, [1, T, 1, HEAD_DIM]),
            "nSh": self._rows("nS", idx, [1, T, 1, HEAD_DIM]),
        }, idx

    def build_draft(self, f_rows):
        """f_rows: int32 [T_d] row-major, the frontier position (position of the last verified input token) of every draft row's user."""
        T = self.T_d
        ftile = ttnn.typecast(ttnn.to_layout(ttnn.reshape(f_rows, [1, 1, 1, T]), ttnn.TILE_LAYOUT), ttnn.float32)
        q = ttnn.reshape(
            ttnn.to_layout(ttnn.typecast(ttnn.add(ftile, self.off_d), ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [T]
        )
        st, _ = self._gather(q, T)
        fidx = ttnn.typecast(ttnn.reshape(f_rows, [T, 1]), ttnn.uint32)
        m = self._rows("mask", fidx, [T, 1, 1, L_D])
        st["mask"] = ttnn.repeat(m, [1, 1, NH, 1])
        return st

    def _i32_rows(self, e, T):
        v = ttnn.to_layout(ttnn.slice(e, [0, 0, 0], [T, 1, 1]), ttnn.TILE_LAYOUT)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(v, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [T])

    def build_verify(self, pos_v):
        """pos_v: int32 [T_v] row-major (position of every verify row) -> RoPE rows + the per-j ring slot indices (``paged_update_cache`` RMWs whole
        tiles: rows of one user must write in separate calls, the others skip with -1)."""
        T = self.T_v
        st, idx = self._gather(pos_v, T)
        slot = self._i32_rows(ttnn.embedding(idx, self.t["ring"], layout=ttnn.ROW_MAJOR_LAYOUT), T)
        f1 = ttnn.add(
            ttnn.typecast(ttnn.to_layout(ttnn.reshape(slot, [1, 1, 1, T]), ttnn.TILE_LAYOUT), ttnn.float32), 1.0
        )
        st["ring_j"] = [
            ttnn.reshape(
                ttnn.to_layout(
                    ttnn.typecast(ttnn.subtract(ttnn.multiply(f1, self.jmask[j]), 1.0), ttnn.int32),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                [T],
            )
            for j in range(self.n)
        ]
        return st


# ---------------------------------------------------------------------------------------------------- the drafter
class DSparkDrafter:
    def __init__(self, md, mesh_config, ccl, stage_w, embed_weight, head, users_per_row=4, n=2, bfp8_main=True):
        """stage_w: [load_mtp_stage(0), (1), (2)]; embed_weight: the backbone's device embedding table [vocab, 5120] (bf16 row-major, replicated);
        head: the backbone's ``DSV41DeviceHead`` (tied LM head weight + sampling); n = verify rows per user (1 + k)."""
        self.md, self.mesh_config, self.ccl, self.head = md, mesh_config, ccl, head
        self.rows, self.cols = tuple(md.shape)
        self.U, self.n = users_per_row, n
        self.T_d, self.T_v = BLOCK * users_per_row, users_per_row * n
        U, T_d, T_v = self.U, self.T_d, self.T_v
        rep = ttnn.ReplicateTensorToMesh(md)
        up = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT, mapper=None: ttnn.from_torch(
            t.contiguous(),
            device=md,
            dtype=dt,
            layout=lay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper or rep,
        )
        self._up = up
        freqs = rope_freqs(0)
        _l1v(md, "drafter init start")
        self.attn = [DraftAttention(md, mesh_config, ccl, w["attn"], freqs, U, n) for w in stage_w]
        _l1v(md, "after attn x3")
        self.state = DraftState(md, self.attn[0], U, n)
        _l1v(md, "after state")
        self.layers, bufs = [], None
        for a, w in zip(self.attn, stage_w):
            layer = DraftLayer(md, mesh_config, ccl, a, w["norms"], w["mhc"], w["moe"], T_d, moe_buffers=bufs)
            if bufs is None:
                bufs = layer.moe.decode.buffers
            self.layers.append(layer)
            _l1v(md, f"after layer {len(self.layers) - 1} init")
        w0 = stage_w[0]
        wdt = ttnn.bfloat8_b if bfp8_main else ttnn.bfloat16
        self.main_proj = up(w0["main_proj"].T.reshape(1, 1, 3 * 5120, 5120), wdt)
        self.main_norm = up(w0["main_norm"].reshape(1, 1, 1, 5120))
        self.wkv_cat = up(
            torch.cat([w["attn"]["wkv"].T for w in stage_w], dim=1).reshape(1, 1, 5120, 3 * HEAD_DIM), wdt
        )
        self.ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        w2 = stage_w[2]
        self.norm_w = up(w2["norm"].reshape(1, 1, 1, 5120))
        self.embed_w = embed_weight
        self.embed_m = up(w2["markov_embed"], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)  # [129280, 256]
        self.head_m = up(
            w2["markov_head"].T.reshape(1, 1, 256, -1),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(self.rows, self.cols)),
        )  # vocab-sharded over the columns
        cw = w2["conf"].reshape(-1)  # [5376] = [x (5120) | markov_embed (256)]
        pad = lambda v: torch.nn.functional.pad(v.reshape(-1, 1), (0, 31))
        self.conf_x, self.conf_m = up(pad(cw[:5120]).reshape(1, 1, 5120, 32), ttnn.float32), up(
            pad(cw[5120:]).reshape(1, 1, 256, 32), ttnn.float32
        )
        self.pre0 = up(torch.tensor([1.0, 0, 0, 0]).repeat(T_d, 1, 1, 1), ttnn.float32)
        self.ucfg_v = ttnn.create_sharded_memory_config(
            shape=(32, HEAD_DIM),
            core_grid=ttnn.num_cores_to_corerangeset(T_v, ttnn.CoreCoord(8, 8), row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # -- verify side: main_kv of every verified position into the 3 rings --------------------------------------------------
    def write_main(self, hidden, st_v):
        """hidden [1,1,T_v,15360] bf16 tile (concat of the means of the 4 streams at the INPUT of backbone layers 37/38/39 for the verify rows);
        st_v = state.build_verify(pos_v)."""
        T = self.T_v
        _l1v(self.md, "write_main start")
        mp = ttnn.matmul(
            hidden,
            self.main_proj,
            compute_kernel_config=self.ckc,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
        )
        main_x = ttnn.rms_norm(mp, weight=self.main_norm, epsilon=1e-20)
        kvc = ttnn.matmul(
            main_x, self.wkv_cat, compute_kernel_config=self.ckc, core_grid=ttnn.CoreGrid(y=2, x=8), dtype=ttnn.bfloat16
        )
        for s, a in enumerate(self.attn):
            kv = ttnn.rms_norm(
                ttnn.slice(kvc, [0, 0, 0, s * HEAD_DIM], [1, 1, T, (s + 1) * HEAD_DIM]), weight=a.kv_norm, epsilon=1e-20
            )
            rm = ttnn.reshape(ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT), [1, T, 1, HEAD_DIM])
            rows = ttnn.to_layout(
                ttnn.pad(rm, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0), ttnn.TILE_LAYOUT
            )  # [1,T,32,512], row 0 valid
            rows = a._rope_heads(rows, st_v["Ch"], st_v["Sh"])
            a.write_main_rows(ttnn.to_memory_config(rows, self.ucfg_v), st_v["ring_j"])
            _l1v(self.md, f"write_main stage {s} done")

    # -- draft side --------------------------------------------------------------------------------------------------------
    def draft(self, tok_rows, f_rows):
        """tok_rows [T_d,1] uint32 row-major: draft block input ids (row i*U+u: t_u for i = 0, the noise id otherwise); f_rows [T_d] int32: frontier
        position per row. -> dict(tokens=[5 x [U,1] uint32 draft tokens d_1..d_5], logits=[5 x [1,1,U,V/cols] fp32 markov-biased], conf [1,1,T_d,32] fp32).
        """
        U, T = self.U, self.T_d
        st = self.state.build_draft(f_rows)
        emb = ttnn.embedding(tok_rows, self.embed_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # [T,1,5120]
        x = ttnn.repeat(ttnn.typecast(ttnn.reshape(emb, [T, 1, 1, 5120]), ttnn.float32), [1, 1, 4, 1])
        pre = self.pre0
        diag = os.environ.get("DSV41_L1_DIAG") == "1"
        for li, layer in enumerate(self.layers):
            if diag:
                prof = {"_l1_trace": []}
                try:
                    x, pre = layer.forward(x, pre, st, profile=prof)
                finally:
                    for name, alloc, free in prof["_l1_trace"]:
                        print(
                            f"L1DIAG stage {li} after {name:24s} allocated/bank {alloc:8d} largest_free {free:8d}",
                            flush=True,
                        )
            else:
                x, pre = layer.forward(x, pre, st)
        y = ttnn.matmul(pre, x, compute_kernel_config=self.ckc)  # hc_pre -> [T,1,1,D]
        xb = ttnn.typecast(ttnn.reshape(y, [1, 1, T, 5120]), ttnn.bfloat16)  # pre-norm (the confidence head reads this)
        xn = ttnn.rms_norm(xb, weight=self.norm_w, epsilon=1e-20)
        logits = ttnn.matmul(
            xn, self.head.head_w, compute_kernel_config=self.head.ckc, dtype=ttnn.float32
        )  # [1,1,T,V/cols]
        toks, lgs, mes = [], [], []
        tok = ttnn.slice(tok_rows, [0, 0], [U, 1])  # t_u
        for i in range(BLOCK):
            me = ttnn.reshape(
                ttnn.embedding(tok, self.embed_m, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), [1, 1, U, 256]
            )
            mes.append(me)
            bias = ttnn.matmul(me, self.head_m, compute_kernel_config=self.ckc, dtype=ttnn.float32)
            lg = ttnn.add(ttnn.slice(logits, [0, 0, i * U, 0], [1, 1, (i + 1) * U, logits.shape[3]]), bias)
            lgs.append(lg)
            tok = self.head.sample_global(
                lg, self.mesh_config, self.ccl
            )  # [U,1] uint32 row-major, identical on every device
            toks.append(tok)
        me_all = ttnn.typecast(ttnn.concat(mes, dim=2), ttnn.float32)  # [1,1,T,256] block-index-major like the rows
        conf = ttnn.add(
            ttnn.matmul(ttnn.typecast(xb, ttnn.float32), self.conf_x, compute_kernel_config=self.ckc),
            ttnn.matmul(me_all, self.conf_m, compute_kernel_config=self.ckc),
        )
        return {"tokens": toks, "logits": lgs, "conf": conf}
