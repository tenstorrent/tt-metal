# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TT DFlash2 drafter: context pass (target taps -> K/V rows in a sliding ring cache), block pass and draft head."""

import math
import os
import time

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.muse_glimmer_30b.reference.dflash2 import DFlash2Config
from models.demos.muse_glimmer_30b.tt.attention import MuseLinear, heads_memcfg, kv_page_table
from models.demos.muse_glimmer_30b.tt.common import as_tt, decode_memcfg, replicate_mapper, sdpa_compute_config
from models.demos.muse_glimmer_30b.tt.dflash.conv import TtDynamicCausalConv
from models.demos.muse_glimmer_30b.tt.dflash.kv_rows import write_kv_rows
from models.demos.muse_glimmer_30b.tt.dflash.layout import RowLayout
from models.demos.muse_glimmer_30b.tt.dflash.topk import device_topk, make_vocab_bias, read_topk
from models.demos.muse_glimmer_30b.tt.mlp import MuseGlimmerMLP
from models.demos.muse_glimmer_30b.tt.model import DECODE_B_PAD
from models.demos.muse_glimmer_30b.tt.model_config import MuseGlimmerModelArgs, dflash_cache_dir, dflash_checkpoint_dir
from models.demos.muse_glimmer_30b.tt.rms_norm import RMSNorm
from models.demos.muse_glimmer_30b.tt.weight_packing import pack_column_parallel, pack_o, pack_qkv

TILE = 32  # tile width (head-norm gamma layout)


class TtDFlash2Drafter:
    def __init__(
        self,
        mesh_device,
        target_args,
        ccl,
        rope,
        *,
        ckpt_dir=None,
        weight_dtype=ttnn.bfloat8_b,
        num_layers=None,
        max_users=1,
    ):
        t0 = time.time()
        self.mesh_device = mesh_device
        self.ccl = ccl
        self.rope = rope
        ckpt_dir = ckpt_dir or dflash_checkpoint_dir()
        self.cfg = cfg = DFlash2Config.from_json(os.path.join(ckpt_dir, "config.json"))
        assert cfg.hidden_size == target_args.hidden_size and cfg.head_dim == target_args.head_dim
        assert cfg.rope_theta == target_args.rope_theta, (cfg.rope_theta, target_args.rope_theta)
        self.tp = tp = int(mesh_device.shape[1])
        assert cfg.num_attention_heads % tp == 0 and cfg.num_key_value_heads % tp == 0
        self.q_local = cfg.num_attention_heads // tp
        self.kv_local = cfg.num_key_value_heads // tp
        self.hd = hd = cfg.head_dim
        self.hidden = H = cfg.hidden_size
        self.num_layers = num_layers or cfg.num_hidden_layers
        self.max_users = max_users
        self.weight_dtype = dt = weight_dtype
        self.dargs = dargs = MuseGlimmerModelArgs.from_checkpoint(
            target_args.ckpt_dir,
            mesh_device=mesh_device,
            cache_dir=dflash_cache_dir(),
            max_seq_len=target_args.max_seq_len,
            max_batch_size=target_args.max_batch_size,
            attn_weight_dtype=dt,
            mlp_weight_dtype=dt,
        )
        self.block_size = dargs.block_size
        self.n_ring = dargs.sliding_kv_blocks
        eps = cfg.rms_norm_eps

        self._st = safe_open(os.path.join(ckpt_dir, "model.safetensors"), framework="pt")
        get = lambda k: self._st.get_tensor(k)  # noqa: E731
        cache = lambda name, d=dt, tag="v1": dargs.cache_file(name, d, tag)  # noqa: E731

        def norm(key):
            return RMSNorm(
                mesh_device, get(key + ".weight"), eps, add_unit_offset=False, cache_file_name=cache(key, ttnn.bfloat16)
            )

        # The drafter's hidden states carry huge outlier channels (|x| up to ~5e3); LoFi matmuls (the target's
        # LOW_PRECISION default) drop the 5-layer PCC to ~0.95, so every drafter matmul runs HiFi2 with fp32 accumulation.
        self.precise_cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.fc = MuseLinear(
            pack_column_parallel(get("fc.weight"), tp),
            mesh_device,
            shard_dim=-1,
            k=len(cfg.target_layer_ids) * H,
            n=H // tp,
            dtype=dt,
            cache_file_name=cache("fc", tag="v1"),
            sharded=False,  # K = 5*H: the DRAM-sharded decode matmul overflows L1
        )
        self._set_precise(self.fc)
        self.hidden_norm = norm("hidden_norm")
        self.norm = norm("norm")

        self.layers = []
        for i in range(self.num_layers):
            p = f"layers.{i}"
            conv = {}
            for cname in ("attention_conv", "mlp_conv"):
                conv[cname] = TtDynamicCausalConv(
                    mesh_device,
                    get(f"{p}.{cname}.base_kernel").float(),
                    get(f"{p}.{cname}.kernel_projection.weight").float(),
                    group_size=cfg.conv_group_size,
                    dtype=dt,
                    cache_file_name=cache(f"{p}.{cname}.kernel_projection"),
                )
            a = f"{p}.self_attn"
            wqkv = MuseLinear(
                pack_qkv(get(f"{a}.q_proj.weight"), get(f"{a}.k_proj.weight"), get(f"{a}.v_proj.weight"), tp, hd),
                mesh_device,
                shard_dim=-1,
                k=H,
                n=(self.q_local + 2 * self.kv_local) * hd,
                dtype=dt,
                cache_file_name=cache(f"{a}.wqkv"),
                readers=2,
                block=2,
            )
            wo = MuseLinear(
                pack_o(get(f"{a}.o_proj.weight"), tp),
                mesh_device,
                shard_dim=-2,
                k=self.q_local * hd,
                n=H,
                dtype=dt,
                cache_file_name=cache(f"{a}.wo"),
                readers=3,
                block=2,
            )
            q_norm = self._head_gamma(get(f"{a}.q_norm.weight"), cache(f"{a}.q_norm", ttnn.bfloat16))
            k_norm = self._head_gamma(get(f"{a}.k_norm.weight"), cache(f"{a}.k_norm", ttnn.bfloat16))
            mlp = MuseGlimmerMLP(
                dargs,
                mesh_device,
                ccl,
                i,
                {f"mlp.{n}_proj.weight": get(f"{p}.mlp.{n}_proj.weight") for n in ("gate", "up", "down")},
            )
            for lin in (mlp.w_gate_up, mlp.w_down, wqkv, wo):
                self._set_precise(lin)
            for c in conv.values():
                c.proj_cfg = self.precise_cfg
            self.layers.append(
                dict(
                    attention_conv=conv["attention_conv"],
                    mlp_conv=conv["mlp_conv"],
                    wqkv=wqkv,
                    wo=wo,
                    q_norm=q_norm,
                    k_norm=k_norm,
                    mlp=mlp,
                    input_layernorm=norm(f"{p}.input_layernorm"),
                    post_attention_layernorm=norm(f"{p}.post_attention_layernorm"),
                )
            )
            logger.info(f"DFlash2 layer {i + 1}/{self.num_layers} built ({time.time() - t0:.0f}s)")

        # KV ring per layer: [max_users * n_ring, kv_local, block_size, hd] bf16 zeros (per-device contents after
        # fill); slot u owns blocks [u * n_ring, (u + 1) * n_ring).
        self.kv_caches = []
        for _ in range(self.num_layers):
            self.kv_caches.append(
                tuple(
                    ttnn.as_tensor(
                        torch.zeros(max_users * self.n_ring, self.kv_local, self.block_size, hd),
                        device=mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                )
            )
        nb = dargs.max_num_blocks_per_user
        self.ring_rows = kv_page_table(dargs, max_users, sliding=True)  # host [max_users, nb]: slot u's ring row
        zeros_pt = torch.zeros(DECODE_B_PAD, nb, dtype=torch.int32)
        self.page_table_tt = self._table(zeros_pt)  # SDPA: row r -> the ring of row r's slot
        self.write_page_table_tt = self._table(zeros_pt)  # KV write: row g -> the ring of group g's slot
        # slot -> [32, nb] table, every row = the slot's ring (batch-1 context fill); all slots before any trace capture
        self._fill_tables = {
            u: self._table(self.ring_rows[u].repeat(DECODE_B_PAD, 1).contiguous()) for u in range(max_users)
        }
        self.layout = None
        self.set_layout(RowLayout(16, (0,)))
        # Draft head: vocab padding mask (sharded like the lm_head) and the selector's hidden projection.
        padded_vocab = dargs.padded_vocab_size
        self.vocab_local = dargs.vocab_local
        self.vocab_bias = make_vocab_bias(mesh_device, padded_vocab, cfg.vocab_size)
        w_hp = get("candidate_selector.hidden_projection.weight").to(torch.bfloat16).T.contiguous()
        self.w_hp = as_tt(
            w_hp.view(1, 1, *w_hp.shape),
            mesh_device,
            dtype=ttnn.bfloat16,
            cache_file_name=cache("candidate_selector.hidden_projection", ttnn.bfloat16),
        )
        self.init_seconds = time.time() - t0
        logger.info(f"TtDFlash2Drafter init {self.init_seconds:.1f}s")

    def set_block_rows(self, block_rows):
        """Set the drafter block size per user (R = K + 1) in every conv layer."""
        for layer in self.layers:
            layer["attention_conv"].set_block_rows(block_rows)
            layer["mlp_conv"].set_block_rows(block_rows)

    def _table(self, t):
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def set_layout(self, layout):
        """Rows follow `layout` (host copies only): the 32-row SDPA table, the compact write table (batch = groups)
        and the conv block size. Rows and groups without a slot keep slot 0's ring (SDPA skips them: position -1)."""
        assert all(u is None or 0 <= u < self.max_users for u in layout.slots), (layout, self.max_users)
        sdpa = self.ring_rows[0].repeat(DECODE_B_PAD, 1)
        write = sdpa.clone()
        for r in range(layout.n_rows):
            u = layout.slot_of_row(r)
            if u is not None:
                sdpa[r] = self.ring_rows[u]
        for g, u in enumerate(layout.slots):
            if u is not None:
                write[g] = self.ring_rows[u]
        mapper = replicate_mapper(self.mesh_device)
        for t, dev in ((sdpa, self.page_table_tt), (write, self.write_page_table_tt)):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper), dev
            )
        self.set_block_rows(layout.R)
        self.layout = layout

    def fill_table(self, slot):
        """[32, nb] device page table, every row = slot's ring: batch-1 writes into one slot (its context fill)."""
        return self._fill_tables[slot]

    def _set_precise(self, lin):
        lin.hifi_cfg = lin.decode_cfg = self.precise_cfg

    def _head_gamma(self, w, cache_name):
        return ttnn.as_tensor(
            w.to(torch.bfloat16).reshape(1, 1, self.hd // TILE, TILE),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name,
        )

    def make_write_idxs(self, start, count, valid=None):
        """`count` device tensors [1] int32 ROW_MAJOR replicated: start+i, or -1 for i >= valid."""
        valid = count if valid is None else valid
        out = []
        for i in range(count):
            v = start + i if i < valid else -1
            out.append(
                ttnn.from_torch(
                    torch.tensor([v], dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=replicate_mapper(self.mesh_device),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
        return out

    def make_rope_idx(self, positions):
        """uint32 [1, 32] ROW_MAJOR replicated; positions padded with 0."""
        pos = list(positions)
        assert len(pos) <= DECODE_B_PAD
        t = torch.zeros(1, DECODE_B_PAD, dtype=torch.int32)
        t[0, : len(pos)] = torch.tensor(pos, dtype=torch.int32)
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def context_features(self, taps):
        """taps: list of 5 [1,1,32,H] bf16 replicated -> hidden_norm(fc(concat)) [1,1,32,H] DRAM."""
        cat = ttnn.concat(taps, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = self.fc.decode(cat)
        cat.deallocate(True)
        y = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
        g = self.ccl.all_gather(y, dim=3)
        if g is not y:
            y.deallocate(True)
        ctx = self.hidden_norm(g)
        g.deallocate(True)
        return ctx

    def context_pass(self, taps, rope_idx_tt, write_idxs, page_table=None, batch=None):
        """Write K/V rows for up to 32 context rows into every layer's ring cache.
        taps: 5 tensors [1,1,32,H]; rows 0..P-1 valid. write_idxs: list of device idx tensors (see make_write_idxs).
        page_table / batch: the write table and batch of write_kv_rows (default: the layout's compact table and group count; the batch-1 fill of one slot passes fill_table(slot), 1).
        """
        assert len(taps) == len(self.cfg.target_layer_ids)
        ctx = self.context_features(taps)
        cos, sin = self.rope.decode(rope_idx_tt, memory_config=heads_memcfg(DECODE_B_PAD, self.hd))
        for li, layer in enumerate(self.layers):
            qkv = layer["wqkv"].decode(ctx)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=self.q_local,
                num_kv_heads=self.kv_local,
                memory_config=heads_memcfg(DECODE_B_PAD, self.hd),
            )
            qkv.deallocate(True)
            q.deallocate(True)
            k_i, k = k, ttnn.rms_norm(
                ttnn.to_memory_config(k, ttnn.L1_MEMORY_CONFIG),
                epsilon=self.cfg.rms_norm_eps,
                weight=layer["k_norm"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            k_i.deallocate(True)
            k_i, k = k, ttnn.experimental.rotary_embedding_hf(
                ttnn.to_memory_config(k, heads_memcfg(DECODE_B_PAD, self.hd)),
                cos,
                sin,
                is_decode_mode=True,
            )
            k_i.deallocate(True)
            self._write_kv(li, k, v, write_idxs, page_table, batch)
            k.deallocate(True)
            v.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)
        ctx.deallocate(True)

    def _write_kv(self, li, k, v, write_idxs, page_table=None, batch=None):
        """k, v: [1, 32, kv_local, hd] heads layout -> rows written into layer li's ring cache."""
        k_cache, v_cache = self.kv_caches[li]
        write_kv_rows(
            k_cache,
            v_cache,
            k,
            v,
            write_idxs,
            self.write_page_table_tt if page_table is None else page_table,
            num_kv_heads=self.kv_local,
            block_size=self.block_size,
            batch=self.layout.groups if batch is None else batch,
        )

    def make_sdpa_pos(self, pos, rows=None):
        """int32 [32] row-major replicated: pos for i < rows (default: the block size), -1 for the rest."""
        rows = self.cfg.block_size if rows is None else rows
        t = torch.full((DECODE_B_PAD,), -1, dtype=torch.int32)
        t[:rows] = pos
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate_mapper(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def embed_block(self, tokens_tt, embedding_weight):
        """tokens_tt uint32 [1,32] RM replicated; embedding_weight: target table (sharded on hidden, ROW_MAJOR).
        Raw embedding (no embed_norm) -> [1,1,32,H] bf16 TILE."""
        m = tokens_tt.shape[-1]
        e = ttnn.embedding(tokens_tt, embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.reshape(e, (1, 1, m, self.hidden // self.tp))
        g = self.ccl.all_gather(e, dim=3)
        if g is not e:
            ttnn.deallocate(e)
        return g

    def _rope_qk(self, t, cos, sin):
        t_i, t = t, ttnn.experimental.rotary_embedding_hf(
            ttnn.to_memory_config(t, heads_memcfg(DECODE_B_PAD, self.hd)), cos, sin, is_decode_mode=True
        )
        t_i.deallocate(True)
        return t

    def _head_norm(self, t, gamma):
        t_i = t
        t = ttnn.rms_norm(
            ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG),
            epsilon=self.cfg.rms_norm_eps,
            weight=gamma,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        t_i.deallocate(True)
        return t

    def _add_residual(self, x, y):
        """x + y -> new DRAM interleaved tensor; frees y, never x."""
        out = ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y.deallocate(True)
        return out

    def block_forward(self, x, rope_idx_tt, write_idxs, sdpa_pos_tt):
        """x [1,1,32,H] bf16 replicated (the row groups of self.layout = block inputs; not freed). Returns norm(hidden) [1,1,32,H] DRAM."""
        H = self.hidden
        dmem = decode_memcfg(H, self.mesh_device)
        cos, sin = self.rope.decode(rope_idx_tt, memory_config=heads_memcfg(DECODE_B_PAD, self.hd))
        grid = self.mesh_device.compute_with_storage_grid_size()
        span = self.dargs.max_num_blocks_per_user * self.block_size
        sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=math.gcd(span, 256),
            exp_approx_mode=False,
        )
        sdpa_cfg = sdpa_compute_config(self.mesh_device)
        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        cur = x
        for li, layer in enumerate(self.layers):
            h = layer["input_layernorm"](cur, memory_config=dmem)
            h2, dyn = layer["attention_conv"].prepare(h)
            h.deallocate(True)
            qkv = layer["wqkv"].decode(h2)
            h2.deallocate(True)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=self.q_local,
                num_kv_heads=self.kv_local,
                memory_config=heads_memcfg(DECODE_B_PAD, self.hd),
            )
            qkv.deallocate(True)
            q = self._rope_qk(self._head_norm(q, layer["q_norm"]), cos, sin)
            k = self._rope_qk(self._head_norm(k, layer["k_norm"]), cos, sin)
            self._write_kv(li, k, v, write_idxs)
            k.deallocate(True)
            v.deallocate(True)
            q_d = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            q.deallocate(True)
            k_cache, v_cache = self.kv_caches[li]
            sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_d,
                k_cache,
                v_cache,
                page_table_tensor=self.page_table_tt,
                cur_pos_tensor=sdpa_pos_tt,
                scale=self.hd**-0.5,
                sliding_window_size=self.cfg.sliding_window,
                program_config=sdpa_pc,
                compute_kernel_config=sdpa_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_d.deallocate(True)
            attn = ttnn.experimental.nlp_concat_heads_decode(
                ttnn.to_memory_config(sdpa, heads_memcfg(DECODE_B_PAD, self.hd)),
                num_heads=self.q_local,
                sub_core_grids=all_cores,
            )
            sdpa.deallocate(True)
            o = layer["wo"].decode(attn)
            attn.deallocate(True)
            o_r = self.ccl.all_reduce(o, memory_config=dmem)
            if o_r is not o:
                o.deallocate(True)
            a = layer["attention_conv"].finish(o_r, dyn)
            o_r.deallocate(True)
            dyn.deallocate(True)
            x1 = self._add_residual(cur, a)
            if cur is not x:
                cur.deallocate(True)

            h = layer["post_attention_layernorm"](x1, memory_config=dmem)
            h2, dyn2 = layer["mlp_conv"].prepare(h)
            h.deallocate(True)
            m = layer["mlp"](h2)
            h2.deallocate(True)
            m2 = layer["mlp_conv"].finish(m, dyn2)
            m.deallocate(True)
            dyn2.deallocate(True)
            cur = self._add_residual(x1, m2)
            x1.deallocate(True)
        cos.deallocate(True)
        sin.deallocate(True)
        out = self.norm(cur)
        if cur is not x:
            cur.deallocate(True)
        return out

    def draft_head(self, h, lm_head):
        """h [1,1,32,H] bf16 (the drafter's normed output; not freed), lm_head: the target's vocab-sharded MuseLinear.
        Returns (vals_tt [1,1,32,32], ids_tt [1,1,32,32] local ids, both per device; hp_tt [1,1,32,256] replicated)."""
        cfg = self.cfg
        h_in = ttnn.to_memory_config(h, decode_memcfg(self.hidden, self.mesh_device))
        lg = lm_head.decode(h_in)
        h_in.deallocate(True)
        lg_d = ttnn.sharded_to_interleaved(lg, ttnn.DRAM_MEMORY_CONFIG)
        lg.deallocate(True)
        x = ttnn.mul(lg_d, cfg.output_multiplier / cfg.final_logit_softcapping)
        lg_d.deallocate(True)
        t = ttnn.tanh(x, fast_and_approximate_mode=False)
        x.deallocate(True)
        logits = ttnn.mul(t, cfg.final_logit_softcapping)
        t.deallocate(True)
        vals, ids = device_topk(logits, self.vocab_bias)
        logits.deallocate(True)
        hp = ttnn.linear(h, self.w_hp, compute_kernel_config=self.precise_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return vals, ids, hp

    def read_draft(self, vals_tt, ids_tt, hp_tt, rows=None):
        """Host: (scores [len(rows),16] f32, cands [len(rows),16] i64, hp [len(rows),256] f32).
        rows defaults to the draft rows 1..block_size-1 (row 0 is the anchor)."""
        rows = list(range(1, self.cfg.block_size)) if rows is None else list(rows)
        scores, cands = read_topk(
            vals_tt,
            ids_tt,
            rows,
            vocab_local=self.vocab_local,
            vocab_size=self.cfg.vocab_size,
            k=self.cfg.selector_top_k,
        )
        hp = ttnn.to_torch(ttnn.get_device_tensors(hp_tt)[0]).float()[0, 0, rows]
        return scores, cands, hp

    def read_draft_groups(self, vals_tt, ids_tt, hp_tt):
        """Host draft candidates per used group of self.layout: {slot: (scores [R-1,16], cands [R-1,16], hp
        [R-1,256])} for the draft rows g*R+1 .. g*R+R-1 (row g*R is the slot's anchor). One device read."""
        lay = self.layout
        groups = [(g, u) for g, u in enumerate(lay.slots) if u is not None]
        rows = [r for g, _ in groups for r in list(lay.rows(g))[1:]]
        scores, cands, hp = self.read_draft(vals_tt, ids_tt, hp_tt, rows)
        n = lay.R - 1
        return {
            u: (scores[i * n : (i + 1) * n], cands[i * n : (i + 1) * n], hp[i * n : (i + 1) * n])
            for i, (_, u) in enumerate(groups)
        }
