# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-P150 MTP speculative decode for Qwen3.5-2B / 4B (batch 1, greedy).

Per iteration: the MTP head drafts K tokens, then ONE base forward verifies the T = K+1 rows
[pending, d1..dK] at positions P..P+K, and the host accepts the longest prefix the base agrees with.

Verify rows run as [1, 1, T, dim] through the existing single-P150 decode norms / MLP / projection
configs (all of them take up to one 32-row tile). Only the two sequence-mixing cores differ:
  * GDN: the packed KDA conv over a [K-1 carry ; T new] window + the fused multi-token recurrence in
    ring mode. The commit is a deferred select (as in the TP spec engine): the host writes the last
    accepted row index into two shared selectors (conv window rows, ring block), so nothing is copied.
  * Full attention: the T rows are T pseudo-users of the paged decode SDPA (one per position, same
    page table), which is exactly causal. KV writes go out one row per call, since rows sharing a
    tile would race inside one batched paged_update_cache.
Rejected rows' KV is simply overwritten by the next verify.
"""
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import kda_conv_prefill, kda_pack_gather, spec_conv_sel, spec_state_blk_idx
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.mtp import argmax_last
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _get_paged_sdpa_decode_program_config,
    apply_rotary_pos_emb_ttnn,
)
from models.tt_transformers.tt.common import Mode

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _ceil_tile(n):
    return -(-n // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


class _GDNVerify:
    """Spec buffers + verify body for one single-device GDN layer."""

    def __init__(self, gdn, T, onehot_cfg):
        cfg = gdn.cfg
        assert cfg.split_conv_state and cfg.tile_aligned_mega, "spec_p150 needs the single-P150 GDN decode layout"
        assert gdn.weights.fused_conv_bias_dev is None, "the KDA conv op has no bias input"
        self.gdn, self.T, self.cfg_onehot = gdn, T, onehot_cfg
        self.Kc = gdn.conv_kernel_size
        self.C = cfg.q_dim + cfg.k_dim + cfg.v_dim
        self.L = self.Kc - 1 + T
        self.Lp = _ceil_tile(self.L)
        dev = gdn.device
        Nv, Dk, Dv = gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
        self.ring = ttnn.zeros(
            [T * Nv, Dk, Dv], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_DRAM
        )
        self.win = ttnn.zeros(
            [1, self.L, self.C], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_DRAM
        )
        self.zero_hist = ttnn.zeros(
            [1, self.Kc - 1, self.C], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=_DRAM
        )
        self.actual_start = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
        )
        self.gather = ttnn.from_torch(
            kda_pack_gather(1, T, self.Kc, self.Lp),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=_DRAM,
        )

    def seed(self):
        """Live decode state -> ring block 0 and conv window rows 1..K-1 (what a mi = 0 selector reads)."""
        g = self.gdn
        Nv, Dk, Dv = g.num_v_heads, g.head_k_dim, g.head_v_dim
        ring = torch.zeros(self.T * Nv, Dk, Dv)
        ring[:Nv] = ttnn.to_torch(g.recurrent_state).float().reshape(Nv, Dk, Dv)
        win = torch.zeros(1, self.L, self.C)
        for k, row in enumerate(g.conv_state_rows):
            win[0, 1 + k] = ttnn.to_torch(row).float().reshape(-1)[: self.C]
        dev = g.device
        ttnn.copy(
            ttnn.from_torch(ring, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_DRAM),
            self.ring,
        )
        ttnn.copy(
            ttnn.from_torch(win, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=_DRAM),
            self.win,
        )

    def forward(self, x, conv_sel, state_idx):
        """x [1,1,T,dim] (normed) -> GDN output [1,1,T,dim]; advances ring + window in place."""
        g, T, C = self.gdn, self.T, self.C
        w, cfg = g.weights, g.cfg
        ckc = g.compute_kernel_config_decode
        mega = ttnn.linear(
            x,
            w.mega_fused_weight,
            compute_kernel_config=ckc,
            program_config=g.decode_program_configs.get("mega"),
            memory_config=_L1,
        )
        qd, ad, bd = w.mega_qkv_dim, w.mega_a_dim, w.mega_b_dim
        b_start = qd + _ceil_tile(ad)
        g_start = b_start + _ceil_tile(bd)
        N = mega.shape[-1]
        qkv = ttnn.slice(mega, [0, 0, 0, 0], [1, 1, T, qd], memory_config=_L1)
        a = ttnn.slice(mega, [0, 0, 0, qd], [1, 1, T, qd + ad], memory_config=_L1)
        b = ttnn.slice(mega, [0, 0, 0, b_start], [1, 1, T, b_start + bd], memory_config=_L1)
        gate = ttnn.slice(mega, [0, 0, 0, g_start], [1, 1, T, N], memory_config=_L1)
        ttnn.deallocate(mega)

        # Conv window: [carry kept by the last commit ; T new rows], then conv + SiLU + q/k/v split.
        qkv3 = ttnn.reshape(qkv, [1, T, C])
        cat = ttnn.concat([self.win, qkv3], dim=1, memory_config=_DRAM)
        ttnn.deallocate(qkv3)
        E = ttnn.matmul(conv_sel, cat, compute_kernel_config=self.cfg_onehot, memory_config=_DRAM)
        ttnn.deallocate(cat)
        ttnn.copy(E, self.win)
        xin = ttnn.to_layout(E, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
        ttnn.deallocate(E)
        if self.Lp != self.L:
            xp = ttnn.pad(xin, [(0, 0), (0, self.Lp - self.L), (0, 0)], 0.0)
            ttnn.deallocate(xin)
            xin = xp
        q, k, v, _ = kda_conv_prefill(
            xin,
            self.Lp,
            self.zero_hist,
            w.fused_conv_weight_taps,
            (cfg.q_dim, cfg.k_dim, cfg.v_dim),
            self.actual_start,
            emit_state=False,
        )
        rows = []
        for t in (q, k, v):
            rows.append(ttnn.matmul(self.gather, t, compute_kernel_config=self.cfg_onehot, memory_config=_L1))
            ttnn.deallocate(t)
        q, k, v = rows

        Hk, Hv, Dk, Dv = g.num_heads, g.num_v_heads, g.head_k_dim, g.head_v_dim
        q = ttnn.reshape(q, [1, T, Hk, Dk])
        k = ttnn.reshape(k, [1, T, Hk, Dk])
        v = ttnn.reshape(v, [1, T, Hv, Dv])
        if Hv > Hk:
            q = ttnn.repeat_interleave(q, Hv // Hk, dim=2)
            k = ttnn.repeat_interleave(k, Hv // Hk, dim=2)
        beta = ttnn.sigmoid(ttnn.reshape(b, [1, T, Hv]), memory_config=_L1)
        softplus = ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)
        sp = ttnn.add(ttnn.reshape(a, [1, T, Hv]), w.dt_bias, activations=[softplus], memory_config=_L1)
        gl = ttnn.multiply(w.A_neg, sp, memory_config=_L1)
        o, _ = fused_recurrent_gated_delta_rule_ttnn(
            q,
            k,
            v,
            beta,
            gl,
            scale=Dk**-0.5,
            initial_state=self.ring,
            device=g.device,
            output_per_token_state=True,
            initial_state_block_idx=state_idx,
        )
        for t in (q, k, v, beta, sp, gl):
            ttnn.deallocate(t)
        if o.dtype != ttnn.bfloat16:
            o = ttnn.typecast(o, ttnn.bfloat16)
        o = ttnn.rms_norm(o, weight=w.o_norm_weight, epsilon=g.norm_eps, memory_config=_L1)
        gate = ttnn.clip(ttnn.reshape(gate, [1, T, Hv, Dv]), min=-1e4, max=1e4)
        o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], memory_config=_L1)
        o = ttnn.clip(o, min=-1e4, max=1e4, memory_config=_L1)
        o = ttnn.reshape(o, [1, 1, T, Hv * Dv])
        return ttnn.linear(
            o,
            w.o_proj_weight,
            compute_kernel_config=ckc,
            program_config=g.decode_program_configs.get("o"),
            memory_config=_L1,
        )


def _attention_verify(attn, x, cos, sin, ctx):
    """x [1,1,T,dim] (normed) -> attention output [1,1,T,dim]. Row j is a pseudo-user at position P+j."""
    cfg, w = attn.config, attn.weights
    T = x.shape[-2]
    H, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    ckc, pcs = attn.compute_kernel_config_decode, attn.decode_program_configs

    qg = ttnn.linear(x, w.q_proj, compute_kernel_config=ckc, program_config=pcs.get("q"), memory_config=_L1)
    qg = ttnn.reshape(qg, [T, 1, H, 2 * D])
    query, gate = ttnn.chunk(qg, 2, dim=-1)
    ttnn.deallocate(qg)
    gate = ttnn.reshape(gate, [1, 1, T, H * D])
    query = ttnn.transpose(ttnn.rms_norm(query, weight=w.q_norm, epsilon=cfg.norm_eps), 1, 2)  # [T,H,1,D]
    key = ttnn.linear(x, w.k_proj, compute_kernel_config=ckc, program_config=pcs.get("k"), memory_config=_L1)
    key = ttnn.rms_norm(ttnn.reshape(key, [T, 1, Hkv, D]), weight=w.k_norm, epsilon=cfg.norm_eps)
    key = ttnn.transpose(key, 1, 2)
    value = ttnn.linear(x, w.v_proj, compute_kernel_config=ckc, program_config=pcs.get("v"), memory_config=_L1)
    value = ttnn.transpose(ttnn.reshape(value, [T, 1, Hkv, D]), 1, 2)
    rd = cos.shape[-1]
    query, key = apply_rotary_pos_emb_ttnn(
        query, key, ttnn.reshape(cos, [T, 1, 1, rd]), ttnn.reshape(sin, [T, 1, 1, rd])
    )

    k_p = ttnn.reshape(key, [1, T, Hkv, D])
    v_p = ttnn.reshape(value, [1, T, Hkv, D])
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    one_core = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, D], ttnn.ShardOrientation.ROW_MAJOR),
    )
    for j in range(T):
        for src, cache in ((k_p, attn.paged_kv_cache_key), (v_p, attn.paged_kv_cache_value)):
            row = ttnn.to_memory_config(ttnn.slice(src, [0, j, 0, 0], [1, j + 1, Hkv, D]), one_core)
            ttnn.experimental.paged_update_cache(cache, row, update_idxs_tensor=ctx.pos_rows[j], page_table=ctx.pt_one)
            ttnn.deallocate(row)
    ttnn.deallocate(k_p)
    ttnn.deallocate(v_p)

    q_dec = ttnn.to_memory_config(ttnn.reshape(query, [1, T, H, D]), _DRAM)
    ttnn.deallocate(query)
    k_cache = attn.paged_kv_cache_key
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_dec,
        k_cache,
        attn.paged_kv_cache_value,
        cur_pos_tensor=ctx.pos_vec,
        page_table_tensor=ctx.pt_rows,
        is_causal=True,
        scale=D**-0.5,
        program_config=_get_paged_sdpa_decode_program_config(attn.device, k_cache.shape[0] * k_cache.shape[2]),
    )
    ttnn.deallocate(q_dec)
    out = ttnn.reshape(out, [1, 1, T, H * D])
    out = ttnn.multiply(out, ttnn.sigmoid(gate), memory_config=_L1)
    ttnn.deallocate(gate)
    return ttnn.linear(out, w.o_proj, compute_kernel_config=ckc, program_config=pcs.get("o"), memory_config=_L1)


class SpecDecodeP150:
    """Batch-1 greedy MTP spec decode on one P150. Needs the model built with enable_mtp and paged KV allocated.

    One iteration = one body over persistent input buffers: anchor select from the last verify's post-norm
    rows, K chained MTP draft steps (ids stay on device), then the T-row base verify. The host stages the
    position / rope / commit inputs, replays (or runs eagerly), and reads K + T ids once.
    """

    def __init__(self, model, page_table, draft_len=2):
        assert model.num_devices == 1 and model.mtp is not None, "single device with an MTP head"
        assert getattr(model.args, "p150_decode_opt", False), "spec_p150 rides the single-P150 decode configs"
        self.model, self.args, self.device = model, model.args, model.device
        self.K, self.T = draft_len, draft_len + 1
        K, T, dim = self.K, self.T, self.args.dim
        self.onehot_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.gdn = {
            i: _GDNVerify(layer.attention, T, self.onehot_cfg)
            for i, layer in enumerate(model.layers)
            if not layer.is_full_attention
        }
        self.nv = next(iter(self.gdn.values())).gdn.num_v_heads
        self.kc = self.args.linear_conv_kernel_dim
        rd = model.rope.cos_cpu.shape[-1]
        i32, u32, bf, rm, tile = ttnn.int32, ttnn.uint32, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        z = torch.zeros
        self.pt_one = self._buf(page_table, i32, rm)
        self.pt_rows = self._buf(page_table.repeat(T, 1), i32, rm)
        self.hn_buf = self._buf(z(1, 1, T, dim), bf, tile)
        self.anchor_sel = self._buf(z(1, 1, 1, T), bf, tile)
        self.tok0 = self._buf(z(1, 1, dtype=torch.int32), u32, rm)
        self.dr_pos = [self._buf(z(1, dtype=torch.int32), i32, rm) for _ in range(K)]
        self.dr_cos = [self._buf(z(1, 1, rd), bf, tile) for _ in range(K)]
        self.dr_sin = [self._buf(z(1, 1, rd), bf, tile) for _ in range(K)]
        self.pos_vec = self._buf(z(T, dtype=torch.int32), i32, rm)
        self.pos_rows = [self._buf(z(1, dtype=torch.int32), i32, rm) for _ in range(T)]
        self.v_cos = self._buf(z(T, 1, rd), bf, tile)
        self.v_sin = self._buf(z(T, 1, rd), bf, tile)
        self.conv_sel = self._buf(spec_conv_sel([0], 1, T, self.kc), bf, tile)
        self.state_idx = self._buf(spec_state_blk_idx([0], 1, self.nv), u32, rm)
        self.vtok = self._buf(z(1, T, dtype=torch.int32), u32, rm)
        self.forced = False  # verify staged tokens (vtok) instead of drafting: teacher-forced scoring
        self.traces = {}  # forced -> (trace id, output tensors)
        self.stats = {"verifies": 0, "accepted": 0, "iter_s": 0.0}

    # ----------------------------------------------------------------------------------------- #
    def _buf(self, t, dtype, layout):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.device, memory_config=_DRAM)

    @staticmethod
    def _stage(t, buf, dtype, layout):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dtype, layout=layout), buf)

    def prefill(self, token_ids):
        """Prompt [1, S] -> first generated token. Warms base KV / GDN state and the MTP KV over the prompt."""
        model = self.model
        S = token_ids.shape[1]
        assert S <= 1024, "spec_p150 prefill covers the concat-prefill range"
        model._build_request_rope(token_ids, None)
        model.reset_state()
        ids = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = model.embd(ids)
        ttnn.deallocate(ids)
        cos, sin = model.rope.get_prefill_rot_mats(0, S)
        for layer in model.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")
        h = model.norm(x, mode=Mode.PREFILL)  # [1, S, dim] post-norm: the drafter feed
        ttnn.deallocate(x)
        logits = ttnn.to_torch(model._lm_head(h[:, -1:, :])).float().reshape(-1)
        model._fill_paged_cache_from_prefill(self.pt_one)

        # MTP KV over slots 0..S-2: (h_s, t_{s+1}).
        if S > 1:
            mtp = model.mtp
            tok = ttnn.from_torch(token_ids[:, 1:].to(torch.int32), dtype=ttnn.uint32, device=self.device)
            emb = mtp.embd(tok)
            ttnn.deallocate(tok)
            fused = mtp._fuse(emb, h[:, : S - 1, :], Mode.PREFILL)
            ttnn.deallocate(emb)
            mcos, msin = model.rope.get_prefill_rot_mats(0, S - 1)
            out = mtp.decoder.forward(
                fused,
                cos=mcos,
                sin=msin,
                mode="prefill",
                page_table=self.pt_one,
                chunk_page_table=self.pt_one,
                chunk_start_idx=0,
            )
            ttnn.deallocate(fused)
            ttnn.deallocate(out)

        rows = torch.zeros(1, 1, self.T, self.args.dim)
        rows[0, 0, 0] = ttnn.to_torch(h[:, S - 1 : S, :]).float().reshape(-1)  # h_{P-1} as "row m = 0"
        ttnn.deallocate(h)
        self._stage(rows, self.hn_buf, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        for gv in self.gdn.values():
            gv.seed()
        self.mi, self.P = 0, S
        self.pending = int(logits.argmax())
        return self.pending

    def _mtp_step(self, hidden, tok_dev, pos, cos, sin):
        """Qwen36MTP.forward_decode for one device: -> (fp32 logits [1,1,1,vocab], chain hidden [1,1,1,dim] DRAM)."""
        model, mtp = self.model, self.model.mtp
        emb = ttnn.reshape(mtp.embd(tok_dev), [1, 1, 1, self.args.dim])
        fused = mtp._fuse(emb, hidden, Mode.DECODE)
        ttnn.deallocate(emb)
        block = mtp.decoder.forward(fused, cos=cos, sin=sin, mode="decode", position_tensor=pos, page_table=self.pt_one)
        ttnn.deallocate(fused)
        normed = mtp.head_norm(block, mode=Mode.DECODE)
        ttnn.deallocate(block)
        nxt = ttnn.to_memory_config(normed, _DRAM)
        if normed.is_sharded():
            ttnn.deallocate(normed)
        logits = model._lm_head(nxt, out_dtype=ttnn.float32)
        return logits, nxt

    def _body(self):
        """Device-only iteration over the staged buffers -> (K draft id tensors, verify ids [1,1,T,1])."""
        model, T, dim = self.model, self.T, self.args.dim
        drafts = []
        if self.forced:
            x = ttnn.reshape(model.embd(self.vtok), [1, 1, T, dim])
        else:
            h = ttnn.matmul(self.anchor_sel, self.hn_buf, compute_kernel_config=self.onehot_cfg, memory_config=_DRAM)
            tok, toks = self.tok0, [self.tok0]
            for k in range(self.K):
                logits, nxt = self._mtp_step(h, tok, self.dr_pos[k], self.dr_cos[k], self.dr_sin[k])
                idx = argmax_last(logits)
                ttnn.deallocate(logits)
                ttnn.deallocate(h)
                h, tok = nxt, ttnn.reshape(idx, (1, 1))
                drafts.append(idx)
                toks.append(tok)
            ttnn.deallocate(h)
            embs = [ttnn.reshape(model.embd(t), [1, 1, 1, dim]) for t in toks]
            x = ttnn.concat(embs, dim=2, memory_config=_L1)
            for e in embs:
                ttnn.deallocate(e)
        p150_norm = self.args.p150_decode_norm_config
        for i, layer in enumerate(model.layers):
            a_in = Qwen36DecoderLayer._p150_decode_norm(layer.attention_norm, x, p150_norm)
            if layer.is_full_attention:
                a_out = _attention_verify(layer.attention, a_in, self.v_cos, self.v_sin, self)
            else:
                a_out = self.gdn[i].forward(a_in, self.conv_sel, self.state_idx)
            ttnn.deallocate(a_in)
            hres = ttnn.add(x, a_out, memory_config=p150_norm[0])
            ttnn.deallocate(a_out)
            ttnn.deallocate(x)
            f_in = Qwen36DecoderLayer._p150_decode_norm(layer.ffn_norm, hres, p150_norm)
            f_out = layer.feed_forward.forward(f_in, mode="decode")
            ttnn.deallocate(f_in)
            x = ttnn.add(hres, f_out, memory_config=p150_norm[0])
            ttnn.deallocate(hres)
            ttnn.deallocate(f_out)
        hn = model._final_norm_decode(x)
        logits = model._lm_head(hn)
        ttnn.copy(hn, self.hn_buf)
        ttnn.deallocate(hn)
        vidx = argmax_last(ttnn.reshape(logits, [1, 1, T, logits.shape[-1]]))
        ttnn.deallocate(logits)
        return drafts, vidx

    def _stage_iteration(self):
        """Host -> device inputs for the iteration at P with last commit index mi (= anchor row)."""
        K, T, P, mi = self.K, self.T, self.P, self.mi
        bf, i32, u32, rm, tile = ttnn.bfloat16, ttnn.int32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        rope = self.model.rope
        sel = torch.zeros(1, 1, 1, T)
        sel[..., mi] = 1.0
        self._stage(sel, self.anchor_sel, bf, tile)
        self._stage(torch.tensor([[self.pending]], dtype=torch.int32), self.tok0, u32, rm)
        for k in range(K):
            slot = P - 1 + k
            r = slot + rope.rope_delta
            self._stage(torch.tensor([slot], dtype=torch.int32), self.dr_pos[k], i32, rm)
            self._stage(rope.cos_cpu[r : r + 1].unsqueeze(0), self.dr_cos[k], bf, tile)
            self._stage(rope.sin_cpu[r : r + 1].unsqueeze(0), self.dr_sin[k], bf, tile)
        pos = torch.arange(P, P + T, dtype=torch.int32)
        self._stage(pos, self.pos_vec, i32, rm)
        for j in range(T):
            self._stage(pos[j : j + 1], self.pos_rows[j], i32, rm)
        r = (pos + rope.rope_delta).long()
        self._stage(rope.cos_cpu[r].reshape(T, 1, -1), self.v_cos, bf, tile)
        self._stage(rope.sin_cpu[r].reshape(T, 1, -1), self.v_sin, bf, tile)
        self._stage(spec_conv_sel([mi], 1, T, self.kc), self.conv_sel, bf, tile)
        self._stage(spec_state_blk_idx([mi], 1, self.nv), self.state_idx, u32, rm)

    def capture(self):
        """Record the current mode's iteration body (after one eager step compiled every program)."""
        tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            outs = self._body()
        finally:
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
        self.traces[self.forced] = (tid, outs)

    def release(self):
        for tid, _ in self.traces.values():
            ttnn.release_trace(self.device, tid)
        self.traces = {}

    def _run(self, use_trace):
        """Stage + run one iteration -> (K draft ids, T verify ids); compiles + captures on first traced use."""
        self._stage_iteration()
        rec = self.traces.get(self.forced) if use_trace else None
        if rec is not None:
            ttnn.execute_trace(self.device, rec[0], cq_id=0, blocking=False)
            drafts_t, vidx = rec[1]
        else:
            drafts_t, vidx = self._body()
        drafts = [int(ttnn.to_torch(t).reshape(-1)[0]) for t in drafts_t]
        ids = [int(t) for t in ttnn.to_torch(vidx).reshape(-1)[: self.T].tolist()]
        if rec is None:
            for t in (*drafts_t, vidx):
                ttnn.deallocate(t)
            if use_trace:
                self.capture()
        return drafts, ids

    def step(self, use_trace=True):
        """One draft + verify iteration -> newly committed tokens (accepted drafts + bonus)."""
        t0 = time.perf_counter()
        timed = use_trace and self.forced in self.traces
        drafts, ids = self._run(use_trace)
        m = 0
        while m < self.K and drafts[m] == ids[m]:
            m += 1
        new = drafts[:m] + [ids[m]]
        self.mi, self.P, self.pending = m, self.P + m + 1, new[-1]
        if timed or not use_trace:
            s = self.stats
            s["verifies"] += 1
            s["accepted"] += m
            s["tokens"] += len(new)
            s["iter_s"] += time.perf_counter() - t0
        return new

    def _reset_stats(self):
        self.stats = {"verifies": 0, "accepted": 0, "tokens": 0, "iter_s": 0.0}

    def generate(self, token_ids, max_new_tokens, trace=True):
        """Greedy generate. With trace, the request's first iteration runs eagerly, then the body is captured
        and replayed. The trace lives for one request: prefill allocates, which must not happen while a trace
        is live (replays reuse the freed intermediates' addresses)."""
        self.forced = False
        out = [self.prefill(token_ids)]
        self._reset_stats()
        try:
            while len(out) < max_new_tokens:
                out.extend(self.step(trace))
        finally:
            self.release()
        return out[:max_new_tokens]

    def teacher_forced(self, token_ids, tokens, n, trace=True):
        """Teacher-forced scoring of the verify path: prefill token_ids, then verify ``tokens`` T at a time
        (every row 'accepted'). Returns n predictions; prediction i follows token_ids + tokens[:i]."""
        self.forced = True
        preds = [self.prefill(token_ids)]
        ref = [int(t) for t in tokens]
        j = 0
        try:
            while len(preds) < n:
                rows = (ref[j : j + self.T] + [0] * self.T)[: self.T]
                self._stage(torch.tensor([rows], dtype=torch.int32), self.vtok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                self.pending = rows[0]
                _, ids = self._run(trace)
                preds.extend(ids)
                self.mi, self.P = self.K, self.P + self.T
                j += self.T
        finally:
            self.forced = False
            self.release()
        return preds[:n]
