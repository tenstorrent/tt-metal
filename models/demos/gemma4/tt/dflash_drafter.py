# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""dFlash block-diffusion drafter for gemma4 (z-lab arxiv 2602.06036).

Drafts ``block_size-1`` tokens in ONE bidirectional forward, conditioned on the
target's residual stream tapped at ``target_layer_ids`` and projected by a
single ``fc`` [len(taps)*H -> H] (+ RMS ``hidden_norm``). Per layer, queries
come from the noise block ``[anchor_embed, mask_embed x15]`` (RAW target
embedding rows — no sqrt(H) scale) and keys/values from
``cat([projected_context, noise_block])``; attention is NON-causal, with a
sliding window on the layers marked ``sliding_attention``. Draft logits =
target lm_head (tied embeddings) over the LAST block_size-1 rows, with
tanh-softcapping. Reference: z-lab/dflash model.py (validated on CPU against
gemma-4-31B-it: mean greedy acceptance ~3-4.3 of a 16 block).

v1 scope (bring-up; perf items ledgered in gemma4_galaxy_effort.md):
  * batch 1, untraced, interleaved DRAM tensors.
  * context K/V are RECOMPUTED each draft from a growing projected-context
    buffer instead of true cache-append: 5 layers x 2 tiny matmuls over
    [ctx, H] — negligible next to a 60-layer target verify, and it removes all
    cache-append/rope-on-append bookkeeping.
  * noise-row embeddings host-gathered from the memory-mapped target
    safetensors (anchor row changes per iteration; mask row is constant).
  * requires UNBOUNDED sliding KV on the target side (same constraint as the
    assistant drafter; the drafter itself keeps no target KV).
"""

import json
import math
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allgather, ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name


def _rope_tables(head_dim, theta, max_pos):
    """HF rotate_half-convention cos/sin tables [max_pos, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


class DFlashDrafter:
    """Device-side dFlash drafter (see module docstring)."""

    def __init__(
        self,
        mesh_device,
        drafter_path,
        target_embed_weight_loader,
        mesh_config,
        ccl_manager,
        tensor_cache_path=None,
        dtype=ttnn.bfloat16,
        max_ctx=8192,
    ):
        """
        Args:
            mesh_device: the (1, N) mesh shared with the target.
            drafter_path: local snapshot dir of the DFlashDraftModel checkpoint
                (config.json + model.safetensors).
            target_embed_weight_loader: zero-arg callable returning the target's
                RAW embed_tokens weight [vocab, H] as a torch tensor (mmap ok).
                Used for the tied lm_head (device, vocab-sharded) and for host
                gathers of the per-iteration noise rows.
            mesh_config: gemma4 MeshConfig (tp over mesh axis 1).
            ccl_manager: the target's CCL manager (all_reduce/all_gather).
        """
        from safetensors import safe_open

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.dtype = dtype
        self.max_ctx = max_ctx

        cfg = json.load(open(Path(drafter_path) / "config.json"))
        self.cfg = cfg
        self.hidden = cfg["hidden_size"]
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.block_size = cfg["block_size"]
        self.mask_token_id = cfg["dflash_config"]["mask_token_id"]
        self.target_layer_ids = list(cfg["dflash_config"]["target_layer_ids"])
        self.sliding_window = cfg.get("sliding_window")
        self.layer_types = list(cfg["layer_types"])
        if len(self.layer_types) < self.n_layers:  # config lists per-layer types
            self.layer_types = self.layer_types + [self.layer_types[-1]] * (self.n_layers - len(self.layer_types))
        self.softcap = float(cfg.get("final_logit_softcapping") or 0.0)
        self.rms_eps = cfg["rms_norm_eps"]

        tp = mesh_config.tp if mesh_config else 1
        self.tp = tp
        assert self.n_kv_heads % tp == 0, f"dflash drafter: kv heads {self.n_kv_heads} % tp {tp} != 0"
        assert self.n_heads % tp == 0, f"dflash drafter: q heads {self.n_heads} % tp {tp} != 0"
        self.local_heads = self.n_heads // tp
        self.local_kv = self.n_kv_heads // tp

        is_mesh = hasattr(mesh_device, "shape")
        self._replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        col = mesh_config.column_parallel(mesh_device) if tp > 1 else None
        row = mesh_config.row_parallel(mesh_device) if tp > 1 else None

        sd = {}
        with safe_open(str(Path(drafter_path) / "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)

        def _dev(name, w, mapper, transpose=True):
            wt = w.transpose(-2, -1).contiguous() if transpose else w
            wt = wt.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
            return ttnn.as_tensor(
                wt,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mapper if mapper is not None else self._replicate,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"dflash_{name}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # fc + norms: replicated (fc is [6H, H] transposed for x@W — small).
        self.fc = _dev("fc", sd["fc.weight"], None)
        self.hidden_norm_w = _dev("hidden_norm", sd["hidden_norm.weight"].reshape(1, 1, 1, -1), None, transpose=False)
        self.final_norm_w = _dev("final_norm", sd["norm.weight"].reshape(1, 1, 1, -1), None, transpose=False)

        self.layers = []
        for i in range(self.n_layers):
            p = f"layers.{i}."
            lyr = {
                # col-parallel: shard output features (heads) across tp
                "q_proj": _dev(f"l{i}_q", sd[p + "self_attn.q_proj.weight"], col),
                "k_proj": _dev(f"l{i}_k", sd[p + "self_attn.k_proj.weight"], col),
                "v_proj": _dev(f"l{i}_v", sd[p + "self_attn.v_proj.weight"], col),
                # row-parallel: shard input features; all_reduce after
                "o_proj": _dev(f"l{i}_o", sd[p + "self_attn.o_proj.weight"], row),
                "q_norm": _dev(
                    f"l{i}_qn", sd[p + "self_attn.q_norm.weight"].reshape(1, 1, 1, -1), None, transpose=False
                ),
                "k_norm": _dev(
                    f"l{i}_kn", sd[p + "self_attn.k_norm.weight"].reshape(1, 1, 1, -1), None, transpose=False
                ),
                "gate": _dev(f"l{i}_gate", sd[p + "mlp.gate_proj.weight"], col),
                "up": _dev(f"l{i}_up", sd[p + "mlp.up_proj.weight"], col),
                "down": _dev(f"l{i}_down", sd[p + "mlp.down_proj.weight"], row),
                "in_norm": _dev(
                    f"l{i}_in", sd[p + "input_layernorm.weight"].reshape(1, 1, 1, -1), None, transpose=False
                ),
                "post_norm": _dev(
                    f"l{i}_post", sd[p + "post_attention_layernorm.weight"].reshape(1, 1, 1, -1), None, transpose=False
                ),
                "sliding": self.layer_types[i] == "sliding_attention",
            }
            self.layers.append(lyr)

        # Tied lm_head from the TARGET's embedding table, vocab-sharded.
        embed_w = target_embed_weight_loader()  # [vocab, H] torch (raw, unscaled)
        self._embed_w_host = embed_w  # host reference for noise-row gathers
        self.vocab = embed_w.shape[0]
        self.lm_head = ttnn.as_tensor(
            embed_w.transpose(0, 1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col if col is not None else self._replicate,
            cache_file_name=get_cache_file_name(tensor_cache_path, "dflash_lm_head"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Drafter-specific RoPE tables (theta differs from the target's),
        # persistent on device in ROW_MAJOR for per-iteration position gathers
        # (house pattern: rope_caches_2d) — no per-draft host uploads.
        cos, sin = _rope_tables(self.head_dim, float(cfg.get("rope_theta", 1e6)), max_ctx + 64)
        mk2 = dict(device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._replicate)
        self._cos_2d = ttnn.from_torch(cos, **mk2)
        self._sin_2d = ttnn.from_torch(sin, **mk2)
        # Persistent mask-token noise rows [1,1,block-1,H] (constant per model).
        self._mask_rows = None
        # Greedy on-device argmax (softcap is monotonic -> argmax-invariant);
        # GEMMA4_DFLASH_HOST_ARGMAX=1 reverts to the full-vocab host readback.
        import os as _os

        self._host_argmax = _os.environ.get("GEMMA4_DFLASH_HOST_ARGMAX", "0") == "1"

        # HiFi4 + fp32 accumulation everywhere (house parity with the target's
        # SDPA path); default-fidelity matmuls measurably drift draft argmaxes
        # vs the CPU reference (12/15 -> see effort ledger).
        self._ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch() if hasattr(mesh_device, "arch") else ttnn.get_arch_name(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Growing projected-context buffer [1, 1, ctx, H] (fc output, pre-norm).
        self._ctx_acc = None
        self._ctx_len = 0

    # ------------------------------------------------------------------ ctx

    def reset(self):
        if self._ctx_acc is not None:
            self._ctx_acc.deallocate(True)
        self._ctx_acc = None
        self._ctx_len = 0

    def append_taps_torch(self, taps_cat):
        """Append target taps for newly committed rows (host path, v1/tests).

        Args:
            taps_cat: torch [1, rows, len(taps)*H] concatenated tap hiddens
                (hidden_states[layer_id+1] rows, in target_layer_ids order).
        """
        t = ttnn.from_torch(
            taps_cat.unsqueeze(0).to(torch.bfloat16),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._replicate,
        )
        self.append_taps_tt(t)
        t.deallocate(True)

    def append_taps_tt_list(self, taps):
        """Append one forward's captured taps (list of [1,1,rows,H], tap order)."""
        assert len(taps) == len(self.target_layer_ids), f"expected {len(self.target_layer_ids)} taps, got {len(taps)}"
        cat = ttnn.concat(taps, dim=3)
        for t in taps:
            t.deallocate(True)
        self.append_taps_tt(cat)
        cat.deallocate(True)

    def append_taps_tt(self, taps_cat_tt):
        """Append device taps [1, 1, rows, len(taps)*H] -> fc -> ctx buffer."""
        proj = ttnn.linear(taps_cat_tt, self.fc, compute_kernel_config=self._ckc)  # replicated [1,1,rows,H]
        if self._ctx_acc is None:
            self._ctx_acc = proj
        else:
            new = ttnn.concat([self._ctx_acc, proj], dim=2)
            self._ctx_acc.deallocate(True)
            proj.deallocate(True)
            self._ctx_acc = new
        self._ctx_len = self._ctx_acc.shape[2]

    # ---------------------------------------------------------------- draft

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.rms_eps, weight=w)

    def _rope4d(self, positions):
        # On-device row gather from the persistent tables; only the position ids
        # (a few dozen uint32) cross the host boundary.
        idx = ttnn.from_torch(
            positions.to(torch.int64).unsqueeze(0),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate,
        )
        cos = ttnn.unsqueeze_to_4D(ttnn.embedding(idx, self._cos_2d, layout=ttnn.TILE_LAYOUT))
        sin = ttnn.unsqueeze_to_4D(ttnn.embedding(idx, self._sin_2d, layout=ttnn.TILE_LAYOUT))
        idx.deallocate(True)
        return cos, sin

    def _apply_rope(self, x_bhsd, cos, sin):
        # x [1, heads, S, head_dim]; rotate_half convention.
        d = self.head_dim
        x1 = x_bhsd[:, :, :, : d // 2]
        x2 = x_bhsd[:, :, :, d // 2 :]
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=3)
        out = ttnn.add(ttnn.multiply(x_bhsd, cos), ttnn.multiply(rot, sin))
        x1.deallocate(True)
        x2.deallocate(True)
        rot.deallocate(True)
        return out

    def block_forward(self, x, h_ctx, cos_ctx, sin_ctx, cos_blk, sin_blk, mask_full_tt, mask_slide_tt, ctx_rows):
        """The dFlash block-draft graph (trace-capturable; no host round trips).

        Args:
            x: [1,1,K+1,H] noise block (anchor row + K mask rows, raw embeds).
            h_ctx: [1,1,ctx_rows,H] hidden_norm'd projected context (pad rows
                beyond the live context are masked via the additive masks).
            cos/sin_{ctx,blk}: rope tensors for the context / block rows.
            mask_{full,slide}_tt: additive [1,1,K+1,ctx_rows+K+1] masks.

        Returns draft token ids [.., K] uint32 (greedy argmax on device; the
        tanh softcap is monotonic and argmax-invariant, so it is skipped).
        """
        K1 = x.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        for lyr in self.layers:
            resid = x
            xn = self._rms(x, lyr["in_norm"])
            q = ttnn.linear(xn, lyr["q_proj"], compute_kernel_config=self._ckc)
            q = ttnn.transpose(ttnn.reshape(q, (1, K1, self.local_heads, self.head_dim)), 1, 2)
            q = self._rms(q, lyr["q_norm"])
            q = self._apply_rope(q, cos_blk, sin_blk)
            k_ctx = ttnn.linear(h_ctx, lyr["k_proj"], compute_kernel_config=self._ckc)
            v_ctx = ttnn.linear(h_ctx, lyr["v_proj"], compute_kernel_config=self._ckc)
            k_blk = ttnn.linear(xn, lyr["k_proj"], compute_kernel_config=self._ckc)
            v_blk = ttnn.linear(xn, lyr["v_proj"], compute_kernel_config=self._ckc)
            xn.deallocate(True)

            def _heads(t, n_rows):
                return ttnn.transpose(ttnn.reshape(t, (1, n_rows, self.local_kv, self.head_dim)), 1, 2)

            k_ctx = self._rms(_heads(k_ctx, ctx_rows), lyr["k_norm"])
            k_blk = self._rms(_heads(k_blk, K1), lyr["k_norm"])
            k_ctx = self._apply_rope(k_ctx, cos_ctx, sin_ctx)
            k_blk = self._apply_rope(k_blk, cos_blk, sin_blk)
            k = ttnn.concat([k_ctx, k_blk], dim=2)
            v = ttnn.concat([_heads(v_ctx, ctx_rows), _heads(v_blk, K1)], dim=2)
            for t in (k_ctx, k_blk, v_ctx, v_blk):
                t.deallocate(True)
            if self.local_kv != self.local_heads:
                k = ttnn.repeat_interleave(k, self.local_heads // self.local_kv, dim=1)
                v = ttnn.repeat_interleave(v, self.local_heads // self.local_kv, dim=1)
            scores = ttnn.matmul(q, ttnn.transpose(k, 2, 3), compute_kernel_config=self._ckc)
            scores = ttnn.multiply(scores, scale)
            scores = ttnn.add(scores, mask_slide_tt if lyr["sliding"] else mask_full_tt)
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self._ckc, numeric_stable=True)
            scores.deallocate(True)
            attn = ttnn.matmul(probs, v, compute_kernel_config=self._ckc)
            probs.deallocate(True)
            k.deallocate(True)
            v.deallocate(True)
            q.deallocate(True)
            attn = ttnn.reshape(ttnn.transpose(attn, 1, 2), (1, 1, K1, self.local_heads * self.head_dim))
            o = ttnn.linear(attn, lyr["o_proj"], compute_kernel_config=self._ckc)
            attn.deallocate(True)
            if self.tp > 1:
                o = ccl_allreduce(o, self.mesh_config, self.ccl_manager)
            x = ttnn.add(resid, o)
            o.deallocate(True)
            resid.deallocate(True)

            resid = x
            xn = self._rms(x, lyr["post_norm"])
            gate = ttnn.linear(xn, lyr["gate"], compute_kernel_config=self._ckc)
            up = ttnn.linear(xn, lyr["up"], compute_kernel_config=self._ckc)
            xn.deallocate(True)
            act = ttnn.multiply(ttnn.silu(gate), up)
            gate.deallocate(True)
            up.deallocate(True)
            mlp = ttnn.linear(act, lyr["down"], compute_kernel_config=self._ckc)
            act.deallocate(True)
            if self.tp > 1:
                mlp = ccl_allreduce(mlp, self.mesh_config, self.ccl_manager)
            x = ttnn.add(resid, mlp)
            mlp.deallocate(True)
            resid.deallocate(True)

        h = self._rms(x, self.final_norm_w)
        x.deallocate(True)
        h_drafts = h[:, :, 1:, :]
        h.deallocate(True)
        logits = ttnn.linear(h_drafts, self.lm_head, compute_kernel_config=self._ckc)
        h_drafts.deallocate(True)
        # Greedy ids via the model's PROVEN on-device sampling module when
        # available (force-argmax over vocab-SHARDED logits; it applies the
        # invalid/padded-vocab mask that a naive per-shard argmax gets wrong,
        # and it is the same path serving uses at B=32). enable_trace=False so
        # its ops inline into OUR single fused trace instead of nesting one.
        sampler = getattr(self, "_sampler", None)
        if sampler is not None:
            tt_tokens, _lp = sampler.sample(logits, enable_trace=False)
            logits.deallocate(True)
            return tt_tokens
        if self.tp > 1:
            logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)
        ids = ttnn.argmax(logits[:, :, :, : self.vocab], dim=-1)
        logits.deallocate(True)
        return ids

    def draft(self, anchor_id, start_pos, num_drafts=None):
        """One untraced block draft (bring-up / A-B path; the fused traced loop
        lives in DFlashFusedDecoder). Returns list of K greedy draft ids."""
        B = self.block_size
        K = num_drafts if num_drafts is not None else B - 1
        assert self._ctx_len > 0, "append taps before drafting"
        ctx = self._ctx_len

        if self._mask_rows is None or self._mask_rows.shape[2] != K:
            mrows = self._embed_w_host[[self.mask_token_id] * K].to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
            self._mask_rows = ttnn.from_torch(
                mrows, device=self.mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._replicate
            )
        arow = self._embed_w_host[[anchor_id]].to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        anchor_tt = ttnn.from_torch(
            arow, device=self.mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._replicate
        )
        x = ttnn.concat([anchor_tt, self._mask_rows], dim=2)
        anchor_tt.deallocate(True)

        h_ctx = self._rms(self._ctx_acc, self.hidden_norm_w)
        ctx_first = start_pos - ctx
        cos_ctx, sin_ctx = self._rope4d(torch.arange(ctx_first, ctx_first + ctx))
        cos_blk, sin_blk = self._rope4d(torch.arange(start_pos, start_pos + K + 1))

        S = ctx + K + 1
        qpos = torch.arange(start_pos, start_pos + K + 1)[:, None]
        kpos = torch.cat([torch.arange(ctx_first, ctx_first + ctx), torch.arange(start_pos, start_pos + K + 1)])[
            None, :
        ]
        vis_full = torch.ones(K + 1, S, dtype=torch.bool)
        mask_full = torch.where(vis_full, 0.0, float("-inf"))
        mk = dict(device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._replicate)
        mask_full_tt = ttnn.from_torch(mask_full.to(torch.bfloat16).unsqueeze(0).unsqueeze(0), **mk)
        if self.sliding_window:
            vis_slide = (qpos - kpos < self.sliding_window) & (kpos - qpos < self.sliding_window)
            mask_slide = torch.where(vis_slide, 0.0, float("-inf"))
            mask_slide_tt = ttnn.from_torch(mask_slide.to(torch.bfloat16).unsqueeze(0).unsqueeze(0), **mk)
        else:
            mask_slide_tt = mask_full_tt

        ids_dev = self.block_forward(x, h_ctx, cos_ctx, sin_ctx, cos_blk, sin_blk, mask_full_tt, mask_slide_tt, ctx)
        ids_t = ttnn.to_torch(ttnn.get_device_tensors(ids_dev)[0] if self.tp > 1 else ids_dev)
        ids_dev.deallocate(True)
        for t in (cos_ctx, sin_ctx, cos_blk, sin_blk, mask_full_tt):
            t.deallocate(True)
        if self.sliding_window:
            mask_slide_tt.deallocate(True)
        h_ctx.deallocate(True)
        return ids_t.reshape(-1).to(torch.int64).tolist()[:K]


class DFlashFusedDecoder:
    """Single-fused-trace dFlash loop: drafter block forward + target verify +
    both argmaxes + tap fc in ONE metal trace, replayed once per iteration.

    One trace total (no draft/verify trace alternation — the documented CCL
    trace-interleave deadlock). Between replays only allocation-free host I/O:
    read 15 draft ids + 16 posterior ids + the accepted rows of the fc output;
    upload the anchor row/token, block+verify positions, refreshed masks and
    the (host-mirrored) padded context buffer.

    v1: B=1, greedy, single ctx bucket (ctx_len + block must stay <= ctx_cap).
    """

    def __init__(self, target_model, drafter, kv_layers, page_table_torch, ctx_cap=2048):
        self.target = target_model
        self.drafter = drafter
        self.kv_layers = kv_layers
        self.page_table_torch = page_table_torch
        self.cap = ctx_cap
        self.mesh_device = drafter.mesh_device
        self._mapper = drafter._replicate
        self._tp = drafter.tp
        K = drafter.block_size - 1
        self.K = K
        H = drafter.hidden

        mkT = dict(device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper)
        z = torch.zeros
        self.ctx_dev = ttnn.from_torch(z(1, 1, ctx_cap, H, dtype=torch.bfloat16), **mkT)
        self.mirror = torch.zeros(ctx_cap, H, dtype=torch.bfloat16)
        self.ctx_len = 0
        # static ctx rope (row r == absolute position r at B=1)
        cos, sin = _rope_tables(drafter.head_dim, float(drafter.cfg.get("rope_theta", 1e6)), ctx_cap)
        self.cos_ctx = ttnn.from_torch(cos.unsqueeze(0).unsqueeze(0), **mkT)
        self.sin_ctx = ttnn.from_torch(sin.unsqueeze(0).unsqueeze(0), **mkT)
        self.anchor_row = ttnn.from_torch(z(1, 1, 1, H, dtype=torch.bfloat16), **mkT)
        S = ctx_cap + K + 1
        self.mask_full = ttnn.from_torch(z(1, 1, K + 1, S, dtype=torch.bfloat16), **mkT)
        self.mask_slide = ttnn.from_torch(z(1, 1, K + 1, S, dtype=torch.bfloat16), **mkT)
        mkU = dict(device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper)
        self.blk_pos = ttnn.from_torch(z(1, K + 1, dtype=torch.int64), **mkU)
        self.v_pu = ttnn.from_torch(z(1, 32, dtype=torch.int64), **mkU)
        self.v_pi = ttnn.from_torch(
            z(K + 1, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mapper,
        )
        self.anchor_tok = ttnn.from_torch(z(1, 1, dtype=torch.int64), **mkU)
        pt = page_table_torch[0:1].repeat(K + 1, 1).to(torch.int32)
        self.v_pt = ttnn.from_torch(
            pt, device=self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper
        )
        # Reuse the model's on-device sampling module for BOTH the draft ids
        # and the verify posterior (proven force-argmax over sharded logits;
        # see tt/model.py sampling init and the batched-prefill call site in
        # tt_transformers generator.py). Falls back to gathered ttnn.argmax.
        self._sampler = getattr(target_model, "sampling", None)
        drafter._sampler = self._sampler
        # persistent tap buffers for the verify's copy-mode capture —
        # lazily allocated by the hook on the compile pass (shape-proof).
        self.tap_bufs = [None for _ in drafter.target_layer_ids]
        self.trace = None
        self.start = None
        self.anchor = None

    # ---------------------------------------------------------------- host I/O

    def _upload_iter_inputs(self, anchor_id, start):
        d = self.drafter
        K = self.K
        arow = d._embed_w_host[[anchor_id]].to(torch.bfloat16).reshape(1, 1, 1, -1)
        h = ttnn.from_torch(arow, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=self._mapper)
        ttnn.copy_host_to_device_tensor(h, self.anchor_row)
        h.deallocate(True)
        tok = ttnn.from_torch(
            torch.tensor([[anchor_id]], dtype=torch.int64),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=self._mapper,
        )
        ttnn.copy_host_to_device_tensor(tok, self.anchor_tok)
        tok.deallocate(True)
        bp = torch.zeros(1, K + 1, dtype=torch.int64)
        bp[0, :] = torch.arange(start, start + K + 1)
        h = ttnn.from_torch(bp, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)
        ttnn.copy_host_to_device_tensor(h, self.blk_pos)
        h.deallocate(True)
        vp = torch.zeros(1, 32, dtype=torch.int64)
        vp[0, : K + 1] = torch.arange(start, start + K + 1)
        h = ttnn.from_torch(vp, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)
        ttnn.copy_host_to_device_tensor(h, self.v_pu)
        h.deallocate(True)
        h = ttnn.from_torch(
            torch.arange(start, start + K + 1, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        ttnn.copy_host_to_device_tensor(h, self.v_pi)
        h.deallocate(True)
        # masks: block queries at positions [start .. start+K], keys = padded ctx
        # rows (positions == row index; rows >= ctx_len masked) then the block
        ctx_len = self.ctx_len
        qpos = torch.arange(start, start + K + 1)[:, None]
        kpos = torch.cat([torch.arange(self.cap), torch.arange(start, start + K + 1)])[None, :]
        kvalid = torch.cat([torch.arange(self.cap) < ctx_len, torch.ones(K + 1, dtype=torch.bool)])[None, :]
        vis_full = kvalid.expand(K + 1, -1)
        mf = torch.where(vis_full, 0.0, float("-inf")).to(torch.bfloat16)
        w = self.drafter.sliding_window
        if w:
            vis_s = vis_full & (qpos - kpos < w) & (kpos - qpos < w)
            ms = torch.where(vis_s, 0.0, float("-inf")).to(torch.bfloat16)
        else:
            ms = mf
        for host, dev in ((mf, self.mask_full), (ms, self.mask_slide)):
            h = ttnn.from_torch(
                host.unsqueeze(0).unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=self._mapper
            )
            ttnn.copy_host_to_device_tensor(h, dev)
            h.deallocate(True)

    def _upload_ctx(self):
        h = ttnn.from_torch(
            self.mirror.reshape(1, 1, self.cap, -1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=self._mapper,
        )
        ttnn.copy_host_to_device_tensor(h, self.ctx_dev)
        h.deallocate(True)

    # ---------------------------------------------------------------- graph

    def _body(self):
        d = self.drafter
        K = self.K
        cos_blk = ttnn.unsqueeze_to_4D(ttnn.embedding(self.blk_pos, d._cos_2d, layout=ttnn.TILE_LAYOUT))
        sin_blk = ttnn.unsqueeze_to_4D(ttnn.embedding(self.blk_pos, d._sin_2d, layout=ttnn.TILE_LAYOUT))
        noise = ttnn.concat([self.anchor_row, d._mask_rows], dim=2)
        h_ctx = d._rms(self.ctx_dev, d.hidden_norm_w)
        draft_ids = d.block_forward(
            noise, h_ctx, self.cos_ctx, self.sin_ctx, cos_blk, sin_blk, self.mask_full, self.mask_slide, self.cap
        )  # [1,1,1,K] uint32
        flat = ttnn.reshape(draft_ids, (1, K))
        vx = ttnn.concat([self.anchor_tok, flat], dim=-1)  # [1, K+1] uint32
        # sharded verify logits when the sampling module can consume them
        self.target._dflash_sharded_logits = self._sampler is not None
        logits, hidden = self.target.ttnn_verify_forward(
            x=vx, current_pos=self.v_pu, current_pos_cache=self.v_pi, page_table=self.v_pt, kv_cache=self.kv_layers
        )
        if self._sampler is not None:
            vidx, _lp = self._sampler.sample(logits, enable_trace=False)
        else:
            vidx = ttnn.argmax(logits[:, :, :, : d.vocab], dim=-1)
        # tap fc (buffers were filled by the verify's copy-mode hook)
        cat = ttnn.concat(self.tap_bufs, dim=3)
        fc_out = ttnn.linear(cat, d.fc, compute_kernel_config=d._ckc)
        return draft_ids, vidx, fc_out

    # ---------------------------------------------------------------- lifecycle

    def prefill_ingest(self, taps, n):
        """Fill the mirror from prefill taps (list of 6 [1,1,rows,H] device
        tensors, clone-mode) via the device fc; rows [:n] enter the mirror."""
        d = self.drafter
        cat = ttnn.concat([t[:, :, :n, :] for t in taps], dim=3)
        for t in taps:
            t.deallocate(True)
        proj = ttnn.linear(cat, d.fc, compute_kernel_config=d._ckc)
        cat.deallocate(True)
        host = ttnn.to_torch(ttnn.get_device_tensors(proj)[0] if self._tp > 1 else proj)
        proj.deallocate(True)
        self.mirror[:n] = host.reshape(-1, host.shape[-1])[:n].to(torch.bfloat16)
        self.ctx_len = n
        self._upload_ctx()

    def capture(self, anchor_id, start):
        """Compile-run then capture the fused body at the first-iteration inputs."""
        self.anchor, self.start = anchor_id, start
        self.target.dflash_capture_taps(self.drafter.target_layer_ids, buffers=self.tap_bufs)
        self._upload_iter_inputs(anchor_id, start)
        a, b, c = self._body()  # compile pass (idempotent KV writes)
        ttnn.synchronize_device(self.mesh_device)
        for t in (a, b, c):
            t.deallocate(True)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._out = self._body()
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        self.trace = tid

    def _read_ids(self, t):
        src = ttnn.get_device_tensors(t)[0] if self._tp > 1 else t
        return ttnn.to_torch(src).reshape(-1).to(torch.int64).tolist()

    def step(self, first=False):
        """One iteration: (replay unless first) -> acceptance -> commit -> host updates.

        Returns (accepted_tokens_list, bonus, produced)."""
        if not first:
            self._upload_iter_inputs(self.anchor, self.start)
            ttnn.execute_trace(self.mesh_device, self.trace, cq_id=0, blocking=False)
        draft_ids_t, vidx_t, fc_out = self._out
        drafts = self._read_ids(draft_ids_t)[: self.K]
        posterior = self._read_ids(vidx_t)[: self.K + 1]
        acc = 0
        for dtok, ptok in zip(drafts, posterior[:-1]):
            if dtok == ptok:
                acc += 1
            else:
                break
        bonus = posterior[acc]
        produced = acc + 1
        # commit taps: fc_out rows [0..produced) are positions [start..start+produced)
        src = ttnn.get_device_tensors(fc_out)[0] if self._tp > 1 else fc_out
        rows = ttnn.to_torch(src).reshape(-1, self.mirror.shape[-1])[:produced].to(torch.bfloat16)
        self.mirror[self.start : self.start + produced] = rows
        self.ctx_len = self.start + produced
        self._upload_ctx()
        accepted = [self.anchor] if False else drafts[:acc]
        self.anchor = bonus
        self.start = self.start + produced
        return accepted, bonus, produced
