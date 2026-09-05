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
import os as _os
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allgather, ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name

_SHARD_ARGMAX_K = 32


def _argmax_last(logits, rows):
    """argmax over the last dim -- [1,1,rows] uint32. ttnn.argmax is only
    correct multicore on ROW_MAJOR input with the row dim EXACTLY one tile, so
    pad each <=32-row chunk to 32, untilize multicore, argmax, slice back
    (ported from ign/gemma4_31B_MTP_Dflash)."""
    R32 = 32
    if rows > R32:
        vocab = logits.shape[-1]
        chunks = []
        off = 0
        while off < rows:
            n = min(R32, rows - off)
            part = ttnn.slice(logits, [0, 0, off, 0], [1, 1, off + n, vocab])
            chunks.append(_argmax_last(part, n))
            part.deallocate(True)
            off += n
        out = ttnn.concat(chunks, dim=2)
        for c in chunks:
            c.deallocate(True)
        return out
    src = logits
    padded = None
    if rows < R32:
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, R32 - rows), (0, 0)], value=0.0)
        src = padded
    u = ttnn.untilize(src, use_multicore=True)
    if padded is not None:
        padded.deallocate(True)
    idx = ttnn.argmax(u, dim=-1, keepdim=False)
    u.deallocate(True)
    if rows < R32:
        sliced = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])
        idx.deallocate(True)
        idx = sliced
    return idx


def _shard_offset_tables(cache, mesh_device, mapper, shard_w, tp, k):
    """Cached replicated TILE tables (built OUTSIDE trace capture -- the compile
    pass runs eagerly first, so the lazy build lands before begin_trace)."""
    key = (shard_w, tp, k)
    if cache.get("key") == key:
        return cache["off"], cache["cols"]
    n = k * tp
    off = torch.zeros(1, 1, 1, n, dtype=torch.int32)
    cols = torch.arange(n, dtype=torch.int32).reshape(1, 1, 1, n)
    for d in range(tp):
        off[0, 0, 0, d * k : (d + 1) * k] = d * shard_w
    cache["off"] = ttnn.from_torch(
        off, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32, mesh_mapper=mapper
    )
    cache["cols"] = ttnn.from_torch(
        cols, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32, mesh_mapper=mapper
    )
    cache["key"] = key
    return cache["off"], cache["cols"]


def _shard_argmax(logits, rows, mesh_device, mapper, mesh_config, ccl, cache):
    """Global greedy ids from TP-sharded vocab logits WITHOUT the 262k-wide
    all-gather: per-shard topk(32), all-gather the 32*tp scalars, offset to
    global vocab ids, argmax the gathered scores, index-select. Matches full
    argmax up to exact bf16 ties (which follow gather/device order). Returns
    [1,1,rows] uint32 RM (ported from ign/gemma4_31B_MTP_Dflash)."""
    k = _SHARD_ARGMAX_K
    tp = mesh_config.tp
    shard_w = int(logits.shape[-1])
    vals, idxs = ttnn.topk(logits, k=k, dim=-1)
    if int(vals.shape[2]) != rows:
        vals_s = ttnn.slice(vals, [0, 0, 0, 0], [1, 1, rows, k])
        idxs_s = ttnn.slice(idxs, [0, 0, 0, 0], [1, 1, rows, k])
        vals.deallocate(True)
        idxs.deallocate(True)
        vals, idxs = vals_s, idxs_s
    gvals = ccl_allgather(vals, mesh_config, ccl)
    gidxs = ccl_allgather(idxs, mesh_config, ccl)
    off, cols = _shard_offset_tables(cache, mesh_device, mapper, shard_w, tp, k)
    gidxs_i = ttnn.typecast(gidxs, ttnn.int32)
    gidxs.deallocate(True)
    global_i = ttnn.add(gidxs_i, off)
    gidxs_i.deallocate(True)
    win = _argmax_last(gvals, rows)
    gvals.deallocate(True)
    win4 = ttnn.to_layout(ttnn.reshape(win, (1, 1, rows, 1)), ttnn.TILE_LAYOUT)
    win.deallocate(True)
    win_i = ttnn.typecast(win4, ttnn.int32)
    win4.deallocate(True)
    mask = ttnn.eq(cols, win_i)
    win_i.deallocate(True)
    zeros = ttnn.subtract(global_i, global_i)
    picked = ttnn.where(mask, global_i, zeros)
    mask.deallocate(True)
    global_i.deallocate(True)
    zeros.deallocate(True)
    sel = ttnn.sum(picked, dim=3, keepdim=True)
    picked.deallocate(True)
    sel_u = ttnn.typecast(sel, ttnn.uint32)
    sel.deallocate(True)
    out = ttnn.reshape(sel_u, (1, 1, rows))
    sel_u.deallocate(True)
    if out.layout != ttnn.ROW_MAJOR_LAYOUT:
        rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out.deallocate(True)
        out = rm
    return out


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
        max_ctx=262144 + 2048,
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
        # GEMMA4_DFLASH_REPLICATED=1: run the (tiny, 5-layer) drafter fully
        # REPLICATED -- every device computes the whole drafter redundantly.
        # Kills the ~10 per-iteration allreduces TP costs here (the drafter is
        # far too small for sharding to pay), and leaves the fabric to the
        # target -- the precondition for any draft/verify overlap later. Costs
        # ~3 GB/device of replicated drafter weights. The tied lm_head STAYS
        # vocab-sharded (replicating 262k x H is 2.7 GB and the on-device
        # sampler consumes sharded logits anyway).
        self.replicated = _os.environ.get("GEMMA4_DFLASH_REPLICATED", "0") == "1"
        assert self.n_kv_heads % tp == 0, f"dflash drafter: kv heads {self.n_kv_heads} % tp {tp} != 0"
        assert self.n_heads % tp == 0, f"dflash drafter: q heads {self.n_heads} % tp {tp} != 0"
        self.local_heads = self.n_heads if self.replicated else self.n_heads // tp
        self.local_kv = self.n_kv_heads if self.replicated else self.n_kv_heads // tp

        is_mesh = hasattr(mesh_device, "shape")
        self._replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        col = mesh_config.column_parallel(mesh_device) if (tp > 1 and not self.replicated) else None
        row = mesh_config.row_parallel(mesh_device) if (tp > 1 and not self.replicated) else None

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
                # Namespace the tensor cache by mode: replicated and sharded
                # loads of the same weight must never share a cache file.
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"dflash_{'rep_' if self.replicated else ''}{name}"
                ),
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
        # per-shard topk argmax (no 262k all-gather); SAFE here because gemma4's
        # padded vocab == real vocab (no invalid tail to mask). A/B knob.
        self._use_shard_argmax = _os.environ.get("GEMMA4_DFLASH_SHARD_ARGMAX", "0") == "1"
        self._sa_cache = {}
        # flash-decode drafter attention (replaces the explicit repeat_interleave/
        # matmul/softmax chain; keys zero-padded to a 64-multiple, masked out)
        self._use_sdpa = _os.environ.get("GEMMA4_DFLASH_SDPA", "1") == "1"
        self._sdpa_zpad = None
        # lm_head stays VOCAB-SHARDED even when the drafter runs replicated:
        # a replicated 262k x H head is 2.7 GB/device, and the on-device
        # sampler consumes sharded logits (its own tiny top-1 gather).
        col_lm = mesh_config.column_parallel(mesh_device) if tp > 1 else None
        self.lm_head = ttnn.as_tensor(
            embed_w.transpose(0, 1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_lm if col_lm is not None else self._replicate,
            cache_file_name=get_cache_file_name(tensor_cache_path, "dflash_lm_head"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Drafter-specific RoPE tables (theta differs from the target's),
        # persistent on device in ROW_MAJOR for per-iteration position gathers
        # (house pattern: rope_caches_2d) — no per-draft host uploads.
        # max_ctx sizes the tables by ABSOLUTE position and must cover the
        # model's full context: ttnn.embedding gathers past the table return
        # GARBAGE rope rows SILENTLY, which collapses draft acceptance to ~1.0
        # at any ISL above the table (measured: 1.85 -> 1.05 at 131k with an
        # 8k table). Cheap: [max_pos, head_dim] bf16 = ~67 MB at 256k.
        cos, sin = _rope_tables(self.head_dim, float(cfg.get("rope_theta", 1e6)), max_ctx + 64)
        mk2 = dict(device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._replicate)
        self._cos_2d = ttnn.from_torch(cos, **mk2)
        self._sin_2d = ttnn.from_torch(sin, **mk2)
        # Persistent mask-token noise rows [1,1,block-1,H] (constant per model).
        self._mask_rows = None
        # Greedy on-device argmax (softcap is monotonic -> argmax-invariant);
        # GEMMA4_DFLASH_HOST_ARGMAX=1 reverts to the full-vocab host readback.

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

    def project_ctx_kv(self, raw_rows, cos_rows, sin_rows):
        """Project RAW fc/context rows into per-layer roped K and V.

        The ctx K/V cache-append path: each committed row is projected ONCE at
        commit time (hidden_norm -> k/v proj -> k_norm -> rope at its absolute
        position) instead of the whole context being re-projected every
        iteration. Returns [(k, v)] per layer, k/v: [1, local_kv, R, hd].
        """
        R = raw_rows.shape[2]
        hn = self._rms(raw_rows, self.hidden_norm_w)
        out = []
        for lyr in self.layers:
            k = ttnn.linear(hn, lyr["k_proj"], compute_kernel_config=self._ckc)
            v = ttnn.linear(hn, lyr["v_proj"], compute_kernel_config=self._ckc)
            k = ttnn.transpose(ttnn.reshape(k, (1, R, self.local_kv, self.head_dim)), 1, 2)
            v = ttnn.transpose(ttnn.reshape(v, (1, R, self.local_kv, self.head_dim)), 1, 2)
            k = self._rms(k, lyr["k_norm"])
            k = self._apply_rope(k, cos_rows, sin_rows)
            out.append((k, v))
        hn.deallocate(True)
        return out

    def block_forward_cached(self, x, ctx_k, ctx_v, cos_blk, sin_blk, mask_full_tt, mask_slide_tt, ctx_rows):
        """block_forward with PRE-CACHED per-layer context K/V (roped).

        Removes the per-iteration ctx work entirely: no hidden_norm/rms over
        cap rows, no k/v projections over cap, no ctx rope gathers -- the
        measured ~12 ms/iter ctx share at cap 2048 plus the shared prologue.
        ctx_k/ctx_v: lists of [1, local_kv, cap, hd] persistent caches.
        """
        K1 = x.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)
        use_sdpa = self._use_sdpa
        if use_sdpa:
            # keys padded to a k_chunk multiple; pad columns arrive NEG from the
            # mask builder. Masks are H-repeated once per iteration (h-major
            # rows, the packed-verify contract).
            S_k = ctx_rows + K1
            padn = (-S_k) % 64
            if padn and self._sdpa_zpad is None:  # compile-pass one-time alloc
                self._sdpa_zpad = ttnn.from_torch(
                    torch.zeros(1, self.local_kv, padn, self.head_dim, dtype=torch.bfloat16),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=self._replicate,
                )
            mf_r = ttnn.repeat(mask_full_tt, ttnn.Shape([1, 1, self.local_heads, 1]))
            ms_r = ttnn.repeat(mask_slide_tt, ttnn.Shape([1, 1, self.local_heads, 1]))
            sdpa_pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                q_chunk_size=32,
                k_chunk_size=64,
                exp_approx_mode=False,
                max_cores_per_head_batch=16,
            )
        for li, lyr in enumerate(self.layers):
            resid = x
            xn = self._rms(x, lyr["in_norm"])
            q = ttnn.linear(xn, lyr["q_proj"], compute_kernel_config=self._ckc)
            q = ttnn.transpose(ttnn.reshape(q, (1, K1, self.local_heads, self.head_dim)), 1, 2)
            q = self._rms(q, lyr["q_norm"])
            q = self._apply_rope(q, cos_blk, sin_blk)
            k_blk = ttnn.linear(xn, lyr["k_proj"], compute_kernel_config=self._ckc)
            v_blk = ttnn.linear(xn, lyr["v_proj"], compute_kernel_config=self._ckc)
            xn.deallocate(True)

            def _heads(t, n_rows):
                return ttnn.transpose(ttnn.reshape(t, (1, n_rows, self.local_kv, self.head_dim)), 1, 2)

            k_blk = self._rms(_heads(k_blk, K1), lyr["k_norm"])
            k_blk = self._apply_rope(k_blk, cos_blk, sin_blk)
            if use_sdpa:
                tail = [self._sdpa_zpad] if padn else []
                k = ttnn.concat([ctx_k[li], k_blk] + tail, dim=2)
                v = ttnn.concat([ctx_v[li], _heads(v_blk, K1)] + tail, dim=2)
                k_blk.deallocate(True)
                v_blk.deallocate(True)
                q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
                qp = ttnn.to_layout(ttnn.reshape(q_rm, (1, 1, self.local_heads * K1, self.head_dim)), ttnn.TILE_LAYOUT)
                a = ttnn.transformer.scaled_dot_product_attention_decode(
                    qp,
                    k,
                    v,
                    is_causal=False,
                    attn_mask=ms_r if lyr["sliding"] else mf_r,
                    scale=scale,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=sdpa_pc,
                )
                k.deallocate(True)
                v.deallocate(True)
                q.deallocate(True)
                a_rm = ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT)
                a.deallocate(True)
                a4 = ttnn.permute(ttnn.reshape(a_rm, (1, self.local_heads, K1, self.head_dim)), (0, 2, 1, 3))
                attn = ttnn.to_layout(ttnn.reshape(a4, (1, 1, K1, self.local_heads * self.head_dim)), ttnn.TILE_LAYOUT)
            else:
                k = ttnn.concat([ctx_k[li], k_blk], dim=2)
                v = ttnn.concat([ctx_v[li], _heads(v_blk, K1)], dim=2)
                k_blk.deallocate(True)
                v_blk.deallocate(True)
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
            if self.tp > 1 and not self.replicated:
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
            if self.tp > 1 and not self.replicated:
                mlp = ccl_allreduce(mlp, self.mesh_config, self.ccl_manager)
            x = ttnn.add(resid, mlp)
            mlp.deallocate(True)
            resid.deallocate(True)

        if use_sdpa:
            mf_r.deallocate(True)
            ms_r.deallocate(True)
        # identical id-producing tail to block_forward
        h = self._rms(x, self.final_norm_w)
        x.deallocate(True)
        h_drafts = h[:, :, 1:, :]
        h.deallocate(True)
        n_draft_rows = int(h_drafts.shape[2])
        logits = ttnn.linear(h_drafts, self.lm_head, compute_kernel_config=self._ckc)
        h_drafts.deallocate(True)
        sampler = getattr(self, "_sampler", None)
        if self._use_shard_argmax and self.tp > 1:
            ids = _shard_argmax(
                logits,
                n_draft_rows,
                self.mesh_device,
                self._replicate,
                self.mesh_config,
                self.ccl_manager,
                self._sa_cache,
            )
            logits.deallocate(True)
            return ids
        if sampler is not None:
            tt_tokens, _lp = sampler.sample(logits, enable_trace=False)
            logits.deallocate(True)
            return tt_tokens
        if self.tp > 1:
            logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)
        ids = ttnn.argmax(logits[:, :, :, : self.vocab], dim=-1)
        logits.deallocate(True)
        return ids

    def block_forward_cached_batched(self, x, ctx_k, ctx_v, cos_blk, sin_blk, mask_full_tt, mask_slide_tt, B, cap):
        """B-user row-folded block draft over STACKED cached ctx K/V.

        One trace-friendly forward for all B users: q rows are user-major
        [blk_0 | .. | blk_{B-1}] (B*K1 rows), keys/values are
        [ctx_0 | .. | ctx_{B-1} | blk_0 | .. | blk_{B-1}] (B*cap + B*K1 rows),
        and the additive masks are BLOCK-DIAGONAL per user (built by the
        batched decoder; stationary in steady state exactly like B=1).
        Attention compute scales B^2 on the folded rows -- fine for the B<=4-8
        this targets (the B*(V+1)<=32 verify row cliff caps B anyway).

        ctx_k/ctx_v: per-layer stacked caches [1, local_kv, B*cap, hd].
        cos/sin_blk: [1, 1, B*K1, hd] per-row rope for the folded block rows.
        """
        BK = x.shape[2]
        K1 = BK // B
        scale = 1.0 / math.sqrt(self.head_dim)
        for li, lyr in enumerate(self.layers):
            resid = x
            xn = self._rms(x, lyr["in_norm"])
            q = ttnn.linear(xn, lyr["q_proj"], compute_kernel_config=self._ckc)
            q = ttnn.transpose(ttnn.reshape(q, (1, BK, self.local_heads, self.head_dim)), 1, 2)
            q = self._rms(q, lyr["q_norm"])
            q = self._apply_rope(q, cos_blk, sin_blk)
            k_blk = ttnn.linear(xn, lyr["k_proj"], compute_kernel_config=self._ckc)
            v_blk = ttnn.linear(xn, lyr["v_proj"], compute_kernel_config=self._ckc)
            xn.deallocate(True)

            def _heads(t, n_rows):
                return ttnn.transpose(ttnn.reshape(t, (1, n_rows, self.local_kv, self.head_dim)), 1, 2)

            k_blk = self._rms(_heads(k_blk, BK), lyr["k_norm"])
            k_blk = self._apply_rope(k_blk, cos_blk, sin_blk)
            k = ttnn.concat([ctx_k[li], k_blk], dim=2)
            v = ttnn.concat([ctx_v[li], _heads(v_blk, BK)], dim=2)
            k_blk.deallocate(True)
            v_blk.deallocate(True)
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
            attn = ttnn.reshape(ttnn.transpose(attn, 1, 2), (1, 1, BK, self.local_heads * self.head_dim))
            o = ttnn.linear(attn, lyr["o_proj"], compute_kernel_config=self._ckc)
            attn.deallocate(True)
            if self.tp > 1 and not self.replicated:
                o = ccl_allreduce(o, self.mesh_config, self.ccl_manager)
            x = ttnn.add(resid, o)
            o.deallocate(True)
            # At li==0 resid IS the caller's x -- a PERSISTENT buffer in the
            # batched decoder (noise_rows), refilled every replay. Freeing it
            # here let the compile pass complete, then the trace pass read a
            # dead buffer (segfault). This function does not consume x.
            if li > 0:
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
            if self.tp > 1 and not self.replicated:
                mlp = ccl_allreduce(mlp, self.mesh_config, self.ccl_manager)
            x = ttnn.add(resid, mlp)
            mlp.deallocate(True)
            resid.deallocate(True)

        # id tail over ALL users' draft rows: per user drop the anchor row.
        h = self._rms(x, self.final_norm_w)
        x.deallocate(True)
        # rows are user-major [b*K1 + i]; drafts are i in [1, K1) per user.
        parts = [h[:, :, b * K1 + 1 : (b + 1) * K1, :] for b in range(B)]
        h_drafts = ttnn.concat(parts, dim=2) if B > 1 else parts[0]
        n_draft_rows = B * (K1 - 1)
        if B > 1:  # at B==1 h_drafts IS parts[0]
            for t in parts:
                try:
                    t.deallocate(True)
                except Exception:
                    pass
        h.deallocate(True)
        logits = ttnn.linear(h_drafts, self.lm_head, compute_kernel_config=self._ckc)
        h_drafts.deallocate(True)
        sampler = getattr(self, "_sampler", None)
        if self._use_shard_argmax and self.tp > 1:
            ids = _shard_argmax(
                logits,
                n_draft_rows,
                self.mesh_device,
                self._replicate,
                self.mesh_config,
                self.ccl_manager,
                self._sa_cache,
            )
            logits.deallocate(True)
            return ids
        if sampler is not None:
            tt_tokens, _lp = sampler.sample(logits, enable_trace=False)
            logits.deallocate(True)
            return tt_tokens  # [.., B*(K1-1)] uint32, user-major
        if self.tp > 1:
            logits = ccl_allgather(logits, self.mesh_config, self.ccl_manager)
        ids = ttnn.argmax(logits[:, :, :, : self.vocab], dim=-1)
        logits.deallocate(True)
        return ids

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
            if self.tp > 1 and not self.replicated:
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
            if self.tp > 1 and not self.replicated:
                mlp = ccl_allreduce(mlp, self.mesh_config, self.ccl_manager)
            x = ttnn.add(resid, mlp)
            mlp.deallocate(True)
            resid.deallocate(True)

        h = self._rms(x, self.final_norm_w)
        x.deallocate(True)
        h_drafts = h[:, :, 1:, :]
        h.deallocate(True)
        n_draft_rows = int(h_drafts.shape[2])
        logits = ttnn.linear(h_drafts, self.lm_head, compute_kernel_config=self._ckc)
        h_drafts.deallocate(True)
        # Greedy ids via the model's PROVEN on-device sampling module when
        # available (force-argmax over vocab-SHARDED logits; it applies the
        # invalid/padded-vocab mask that a naive per-shard argmax gets wrong,
        # and it is the same path serving uses at B=32). enable_trace=False so
        # its ops inline into OUR single fused trace instead of nesting one.
        sampler = getattr(self, "_sampler", None)
        if self._use_shard_argmax and self.tp > 1:
            ids = _shard_argmax(
                logits,
                n_draft_rows,
                self.mesh_device,
                self._replicate,
                self.mesh_config,
                self.ccl_manager,
                self._sa_cache,
            )
            logits.deallocate(True)
            return ids
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
        # Host mirror: SEED ONLY (prefill_ingest fills it once and uploads once).
        # Steady-state context commits happen ON DEVICE: the fused body starts by
        # embed-gather-merging [ctx_dev | fc_prev] through ``merge_idx`` (a host-
        # refreshed [1, cap] row map, ~8 KB/iter), replacing the old full-buffer
        # re-upload (cap x H bf16 = ~22 MB/iter at cap 2048) and the fc-row
        # readback. Same loop-free pattern as the packed verify's staging merge.
        self.mirror = torch.zeros(ctx_cap, H, dtype=torch.bfloat16)
        self.ctx_len = 0
        # Packed-verify + truncation config resolves EARLY: the tap/fc row count
        # equals the VERIFY row count (taps are captured during the verify
        # forward), so fc_prev and the ctx-merge width are sized to P_v.
        # Truncation (V < K) requires the packed path -- the batch-dim verify
        # always runs K+1 rows.
        self.use_packed = _os.environ.get("GEMMA4_DFLASH_PACKED", "1") == "1"
        self.V = min(int(_os.environ.get("GEMMA4_DFLASH_VERIFY", str(K))), K) if self.use_packed else K
        self.P_v = self.V + 1
        self.pv_pos = None  # allocated in capture() (needs the generation horizon)
        # Persistent OUTPUT slots (lazy first-use clone on the compile pass,
        # then ttnn.copy per body run). The body's fresh draft_ids/vidx are
        # created MID-GRAPH; ops after them (the whole packed verify) can
        # reclaim their trace-region buffers -- observed as float-bit garbage
        # ids at P_v=16 (allocator-layout dependent: V=7 and the short-prompt
        # e2e happened to survive). Same hygiene as fc_prev / the tap buffers.
        self.out_ids = None
        self.fc_prev = ttnn.from_torch(z(1, 1, self.P_v, H, dtype=torch.bfloat16), **mkT)
        # ── ctx K/V CACHE-APPEND (GEMMA4_DFLASH_CTX_CACHE, default on) ──
        # Per drafter layer, persistent roped ctx K/V [1, local_kv, cap, hd].
        # Committed rows are projected ONCE at commit (project_ctx_kv on
        # fc_prev at their absolute positions) and merged with the SAME
        # merge_idx row map; block_forward_cached then skips ALL per-iteration
        # ctx work (measured ~12 ms/iter at cap 2048, plus the shared
        # hidden_norm/rope prologue).
        self.ctx_cache = _os.environ.get("GEMMA4_DFLASH_CTX_CACHE", "1") == "1"
        self.ctx_k = None
        self.ctx_v = None
        if self.ctx_cache:
            kvshape = (1, drafter.local_kv, ctx_cap, drafter.head_dim)
            self.ctx_k = [ttnn.from_torch(z(*kvshape, dtype=torch.bfloat16), **mkT) for _ in drafter.layers]
            self.ctx_v = [ttnn.from_torch(z(*kvshape, dtype=torch.bfloat16), **mkT) for _ in drafter.layers]
        # [1, P_v] absolute positions of fc_prev's rows (host-refreshed per
        # iteration in step(); zeros at capture where the identity merge makes
        # the commit a no-op).
        self.commit_pos = ttnn.from_torch(
            torch.zeros(1, self.P_v, dtype=torch.int64),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mapper,
        )
        self.merge_idx = ttnn.from_torch(
            torch.arange(ctx_cap, dtype=torch.int64).reshape(1, ctx_cap),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mapper,
        )
        # Sliding window: mirror row r holds ABSOLUTE position win_first + r
        # (4/5 drafter layers are sliding-2048 by architecture, so a window is
        # the natural context). Row->position is a per-iteration device input;
        # ctx rope is gathered in-graph from the drafter's rope tables, so no
        # static row==position assumption and no per-iter rope upload.
        self.win_first = 0
        mkU0 = dict(device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper)
        self.ctx_pos = ttnn.from_torch(torch.zeros(1, ctx_cap, dtype=torch.int64), **mkU0)
        self.anchor_row = ttnn.from_torch(z(1, 1, 1, H, dtype=torch.bfloat16), **mkT)
        S = ctx_cap + K + 1
        if drafter._use_sdpa:
            S += (-S) % 64  # keys are zero-padded to a k_chunk multiple (masked NEG)
        self._dmask_w = S
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
        # ── PACKED target verify (GEMMA4_DFLASH_PACKED=0 reverts to batch-dim) ──
        # The batch-dim verify re-reads the whole target KV once PER CANDIDATE
        # ROW: measured 194 ms/iter at 131k = 16 rows x 2.7 GB bounded KV. The
        # packed verify (candidates in the query-heads dim) reads KV ONCE and
        # writes candidates via the per-position fallback loop (proven exact in
        # the MTP-side bisect) -- no staging port needed.
        # GEMMA4_DFLASH_VERIFY truncates HOW MANY of the K drafts are verified:
        # on prose the block accepts ~0.6 of 15, so verifying 15 wastes verify
        # rows; V<K keeps the drafter unchanged and shrinks the verify.
        # Known trade-off inherited from MTP: the packed non-causal softmax
        # drifts off exact greedy at very large S_k (mesh op defaults). The
        # committed tokens remain the verify posterior (self-consistent /
        # coherent); validate per ISL.
        pt = page_table_torch[0:1].repeat(K + 1, 1).to(torch.int32)
        self.v_pt = ttnn.from_torch(
            pt, device=self.mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper
        )
        # BOUNDED sliding target: hybrid per-layer tables, each layer's user row
        # replicated across the K+1 verify candidates (the batch-alias trick).
        # Without these the verify falls through to the model's batch-1 set and
        # decode_forward slices row b>=1 of a 1-row table -- the same failure the
        # MTP fused body had (see spec_decode._capture_fused_trace). None when
        # the target has no per-layer tables installed (unbounded: unchanged).
        self.v_ptl = None
        installed = getattr(target_model, "_active_page_tables_per_layer", None)
        if installed:
            self.v_ptl = []
            for lpt in installed:
                if lpt is None or not hasattr(lpt, "dim"):
                    self.v_ptl.append(lpt)
                    continue
                row = lpt[0:1] if lpt.dim() > 1 else lpt.unsqueeze(0)
                self.v_ptl.append(row.repeat(K + 1, 1).to(torch.int32))
        # Reuse the model's on-device sampling module for BOTH the draft ids
        # and the verify posterior (proven force-argmax over sharded logits;
        # see tt/model.py sampling init and the batched-prefill call site in
        # tt_transformers generator.py). Falls back to gathered ttnn.argmax.
        self._sampler = getattr(target_model, "sampling", None)
        drafter._sampler = self._sampler
        # The drafter's mask-token noise rows are created lazily by the
        # UNTRACED draft() path; the fused decoder may be the only consumer
        # (e.g. the ISL sweep), so materialize them here.
        if drafter._mask_rows is None or drafter._mask_rows.shape[2] != K:
            _mrows = drafter._embed_w_host[[drafter.mask_token_id] * K].to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
            drafter._mask_rows = ttnn.from_torch(_mrows, **mkT)
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
        if self.use_packed and self.pv_pos is not None:
            self._pv_upload(start)
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
        if not self.use_packed:
            # v_pu / v_pi feed the batch-dim verify only -- dead in packed mode.
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
        win_len = self.ctx_len - self.win_first
        ctx_positions = torch.arange(self.win_first, self.win_first + self.cap)
        if not self.ctx_cache:
            # ctx_pos feeds the NON-cached body's rope gathers only; the cached
            # path bakes rope at commit time and never reads it.
            h = ttnn.from_torch(
                ctx_positions.clamp(min=0).reshape(1, -1).to(torch.int64),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=self._mapper,
            )
            ttnn.copy_host_to_device_tensor(h, self.ctx_pos)
            h.deallocate(True)
        # The drafter masks depend only on (win_len, start - win_first): with a
        # FULL window both advance in lockstep, so in steady state (any prompt
        # longer than cap -- e.g. every long-context run) the masks are
        # STATIONARY and the torch.where builds + two uploads are skipped.
        _mask_key = (win_len, start - self.win_first)
        if getattr(self, "_mask_key", None) == _mask_key:
            return
        qpos = torch.arange(start, start + K + 1)[:, None]
        kpos = torch.cat([ctx_positions, torch.arange(start, start + K + 1)])[None, :]
        kvalid = torch.cat([torch.arange(self.cap) < win_len, torch.ones(K + 1, dtype=torch.bool)])[None, :]
        vis_full = kvalid.expand(K + 1, -1)
        # -1e9, not -inf: equivalent under the explicit softmax path and
        # REQUIRED by the SDPA drafter branch (flash kernels NaN on -inf)
        mf = torch.where(vis_full, 0.0, -1e9).to(torch.bfloat16)
        w = self.drafter.sliding_window
        if w:
            vis_s = vis_full & (qpos - kpos < w) & (kpos - qpos < w)
            ms = torch.where(vis_s, 0.0, -1e9).to(torch.bfloat16)
        else:
            ms = mf
        padn = self._dmask_w - mf.shape[1]
        if padn:
            neg = torch.full((K + 1, padn), -1e9, dtype=torch.bfloat16)
            mf = torch.cat([mf, neg], dim=1)
            ms = torch.cat([ms, neg], dim=1) if ms is not mf else mf
        for host, dev in ((mf, self.mask_full), (ms, self.mask_slide)):
            h = ttnn.from_torch(
                host.unsqueeze(0).unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=self._mapper
            )
            ttnn.copy_host_to_device_tensor(h, dev)
            h.deallocate(True)
        self._mask_key = _mask_key

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
        # ── on-device context commit (start-of-replay) ─────────────────────
        # fc_prev holds the PREVIOUS replay's projected tap rows; merge_idx maps
        # each ctx row to [ctx_dev | fc_prev] (identity / window-shift / commit
        # rows), refreshed per iteration by step(). At capture merge_idx is
        # identity, so compile + capture are no-ops here (idempotent).
        if self.ctx_cache:
            # commit: project the PREVIOUS replay's fc rows at their absolute
            # positions and gather-merge into every layer's roped K/V cache.
            cos_c = ttnn.unsqueeze_to_4D(ttnn.embedding(self.commit_pos, d._cos_2d, layout=ttnn.TILE_LAYOUT))
            sin_c = ttnn.unsqueeze_to_4D(ttnn.embedding(self.commit_pos, d._sin_2d, layout=ttnn.TILE_LAYOUT))
            kv_new = d.project_ctx_kv(self.fc_prev, cos_c, sin_c)
            cos_c.deallocate(True)
            sin_c.deallocate(True)
            hd = d.head_dim
            for li, (k_new, v_new) in enumerate(kv_new):
                for cache, new in ((self.ctx_k[li], k_new), (self.ctx_v[li], v_new)):
                    for h_i in range(d.local_kv):
                        src = ttnn.concat([cache[:, h_i : h_i + 1, :, :], new[:, h_i : h_i + 1, :, :]], dim=2)
                        src2d = ttnn.reshape(src, (self.cap + self.P_v, hd))
                        m = ttnn.embedding(self.merge_idx, src2d, layout=ttnn.TILE_LAYOUT)
                        m4 = ttnn.reshape(m, (1, 1, self.cap, hd))
                        dst = cache if d.local_kv == 1 else None
                        if dst is None:
                            raise NotImplementedError("ctx cache merge assumes local_kv == 1 per device")
                        ttnn.assign(m4, dst)
                        for t in (m4, m, src2d, src):
                            try:
                                t.deallocate(True)
                            except Exception:
                                pass
                k_new.deallocate(True)
                v_new.deallocate(True)
        else:
            src = ttnn.concat([self.ctx_dev, self.fc_prev], dim=2)  # [1,1,cap+P_v,H]
            src2d = ttnn.reshape(src, (self.cap + self.P_v, d.hidden))
            merged = ttnn.embedding(self.merge_idx, src2d, layout=ttnn.TILE_LAYOUT)  # [1,cap,H]
            merged4 = ttnn.reshape(merged, (1, 1, self.cap, d.hidden))
            ttnn.assign(merged4, self.ctx_dev)
            for t in (merged4, merged, src2d, src):
                try:
                    t.deallocate(True)
                except Exception:
                    pass
        cos_blk = ttnn.unsqueeze_to_4D(ttnn.embedding(self.blk_pos, d._cos_2d, layout=ttnn.TILE_LAYOUT))
        sin_blk = ttnn.unsqueeze_to_4D(ttnn.embedding(self.blk_pos, d._sin_2d, layout=ttnn.TILE_LAYOUT))
        noise = ttnn.concat([self.anchor_row, d._mask_rows], dim=2)
        if self.ctx_cache:
            draft_ids = d.block_forward_cached(
                noise, self.ctx_k, self.ctx_v, cos_blk, sin_blk, self.mask_full, self.mask_slide, self.cap
            )  # [1,1,1,K] uint32
        else:
            h_ctx = d._rms(self.ctx_dev, d.hidden_norm_w)
            hd = d.head_dim
            cos_ctx = ttnn.reshape(
                ttnn.embedding(self.ctx_pos, d._cos_2d, layout=ttnn.TILE_LAYOUT), (1, 1, self.cap, hd)
            )
            sin_ctx = ttnn.reshape(
                ttnn.embedding(self.ctx_pos, d._sin_2d, layout=ttnn.TILE_LAYOUT), (1, 1, self.cap, hd)
            )
            draft_ids = d.block_forward(
                noise, h_ctx, cos_ctx, sin_ctx, cos_blk, sin_blk, self.mask_full, self.mask_slide, self.cap
            )  # [1,1,1,K] uint32
        flat = ttnn.reshape(draft_ids, (1, K))
        vx = ttnn.concat([self.anchor_tok, flat], dim=-1)  # [1, K+1] uint32
        # sharded verify logits when the sampling module (or shard-argmax)
        # consumes them pre-gather
        self.target._dflash_sharded_logits = (self._sampler is not None) or (
            self.drafter._use_shard_argmax and self.drafter.tp > 1
        )
        if self.use_packed:
            # ONE KV read for all P_v rows (vs P_v reads batch-dim). Fallback
            # per-position writes (kv_write_idxs) -- no staging. Masks arrive
            # P-row and are repeated H x in-trace (h*P+p packed row order).
            vxp = ttnn.slice(vx, [0, 0], [1, self.P_v]) if self.P_v < K + 1 else vx
            H_l = self.target.layers[0].self_attn.config.num_attention_heads // self.drafter.tp
            # full-range ttnn.slice can alias its input -- never wrap-and-free
            # the persistent blk_pos at P_v == K+1
            pv_pos = ttnn.slice(self.blk_pos, [0, 0], [1, self.P_v]) if self.P_v < K + 1 else self.blk_pos
            widx = [ttnn.slice(self.pv_widx_all, [p], [p + 1]) for p in range(self.P_v)]
            # device-built verify masks: full = -1e9 where iota > pos (row-wise
            # outer broadcast, fp32 for exact integer positions); sliding
            # (unbounded) additionally blocks iota <= pos - W. Bounded slide
            # comes from the host ring upload.
            posf = ttnn.typecast(
                ttnn.to_layout(ttnn.reshape(pv_pos, (1, 1, self.P_v, 1)), ttnn.TILE_LAYOUT), ttnn.float32
            )
            diff = ttnn.sub(self.pv_iota, posf)
            gt = ttnn.gtz(diff)
            mf_p = ttnn.typecast(ttnn.multiply(gt, -1e9), ttnn.bfloat16)
            m_full = ttnn.repeat(mf_p, ttnn.Shape([1, 1, H_l, 1]))
            if self.pv_slide_ring:
                m_slide = ttnn.repeat(self.pv_mask_slide, ttnn.Shape([1, 1, H_l, 1]))
            else:
                W_t = float(self.target.hf_config.sliding_window)
                far = ttnn.gez(ttnn.multiply(ttnn.add(diff, W_t), -1.0))  # iota <= pos - W
                ind = ttnn.add(gt, far)
                ms_p = ttnn.typecast(ttnn.multiply(ind, -1e9), ttnn.bfloat16)
                m_slide = ttnn.repeat(ms_p, ttnn.Shape([1, 1, H_l, 1]))
            logits, hidden = self.target.ttnn_packed_verify_forward(
                x=vxp,
                position_idx=pv_pos,
                attn_mask_full=m_full,
                attn_mask_sliding=m_slide,
                packed_p=self.P_v,
                page_table=self.v_pt,
                kv_cache=self.kv_layers,
                kv_write_idxs=widx,
                page_tables_per_layer=self.pv_tables,
            )
            m_full.deallocate(True)
            m_slide.deallocate(True)
            # NOTE: pv_pos and the widx slices are NOT deallocated. 1-element
            # ttnn.slice views can ALIAS their source buffer; freeing them
            # double-frees pv_widx_all/blk_pos and corrupts the allocator (the
            # verify sampler then throws "Tensor is not allocated" on a healthy
            # logits tensor). Fixed-footprint per-replay temporaries are fine.
            if vxp is not vx:
                vxp.deallocate(True)
        else:
            logits, hidden = self.target.ttnn_verify_forward(
                x=vx,
                current_pos=self.v_pu,
                current_pos_cache=self.v_pi,
                page_table=self.v_pt,
                kv_cache=self.kv_layers,
                page_tables_per_layer=self.v_ptl,
            )
        if self.drafter._use_shard_argmax and self.drafter.tp > 1:
            vidx = _shard_argmax(
                logits,
                self.P_v if self.use_packed else K + 1,
                self.mesh_device,
                self._mapper,
                self.target.mesh_config,
                self.target.ccl_manager,
                self.drafter._sa_cache,
            )
        elif self._sampler is not None:
            vidx, _lp = self._sampler.sample(logits, enable_trace=False)
        else:
            vidx = ttnn.argmax(logits[:, :, :, : d.vocab], dim=-1)
        # tap fc (buffers were filled by the verify's copy-mode hook). The rows
        # also land in the persistent fc_prev so the NEXT replay's start-of-body
        # merge can commit the accepted prefix on device.
        cat = ttnn.concat(self.tap_bufs, dim=3)
        fc_out = ttnn.linear(cat, d.fc, compute_kernel_config=d._ckc)
        ttnn.assign(fc_out, self.fc_prev)
        fc_out.deallocate(True)
        # ONE fused id output ([1, K+P_v]: drafts then posterior) -> ONE
        # blocking readback in step() instead of two.
        dflat = ttnn.reshape(draft_ids, (1, K))
        vflat = ttnn.reshape(vidx, (1, self.P_v))
        ids_cat = ttnn.concat([dflat, vflat], dim=-1)
        if self.out_ids is None:  # compile pass: allocate at the real shape
            self.out_ids = ttnn.clone(ids_cat)
        ttnn.copy(ids_cat, self.out_ids)
        ids_cat.deallocate(True)
        draft_ids.deallocate(True)
        vidx.deallocate(True)
        return self.out_ids, self.fc_prev

    # ---------------------------------------------------------------- lifecycle

    def prefill_ingest(self, taps, n):
        """Fill the mirror window from prefill taps.

        ``taps`` is one group of len(target_layer_ids) tensors per prefill
        FORWARD -- a chunked long prefill fires the hook once per chunk, so
        several groups arrive (the hook's keep_last cap bounds how many are
        retained; dropped early chunks are outside the window by construction).
        Only the last ``cap`` rows are kept.
        """
        d = self.drafter
        n_taps = len(d.target_layer_ids)
        assert len(taps) % n_taps == 0, f"taps {len(taps)} not a multiple of {n_taps}"
        groups = [taps[i : i + n_taps] for i in range(0, len(taps), n_taps)]
        seen = 0
        chunks = []
        for gi, g in enumerate(groups):
            avail = int(g[0].shape[2])
            remaining = n - seen
            if remaining <= 0:
                for t in g:
                    t.deallocate(True)
                continue
            # never slice past what this forward actually produced; the final
            # chunk is tile-padded so its valid rows are what remains of n
            rv = min(avail, remaining)
            cat = ttnn.concat([t[:, :, :rv, :] for t in g], dim=3)
            for t in g:
                t.deallocate(True)
            proj = ttnn.linear(cat, d.fc, compute_kernel_config=d._ckc)
            cat.deallocate(True)
            host = ttnn.to_torch(ttnn.get_device_tensors(proj)[0] if self._tp > 1 else proj)
            proj.deallocate(True)
            chunks.append(host.reshape(-1, host.shape[-1])[:rv])
            seen += rv
        rows = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        keep = min(self.cap, rows.shape[0], n)
        self.mirror[:keep] = rows[-keep:].to(torch.bfloat16)
        self.win_first = n - keep
        self.ctx_len = n
        self._upload_ctx()
        if self.ctx_cache:
            # One-time seed of the per-layer roped ctx K/V caches from the
            # freshly uploaded raw ctx (row r holds absolute position
            # win_first + r; dead rows beyond the live window are masked by the
            # additive masks, their rope position is clamped and harmless).
            d = self.drafter
            cp = torch.arange(self.win_first, self.win_first + self.cap, dtype=torch.int64).clamp(min=0)
            h = ttnn.from_torch(
                cp.reshape(1, self.cap), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper
            )
            ttnn.copy_host_to_device_tensor(h, self.ctx_pos)
            h.deallocate(True)
            cos = ttnn.reshape(
                ttnn.embedding(self.ctx_pos, d._cos_2d, layout=ttnn.TILE_LAYOUT), (1, 1, self.cap, d.head_dim)
            )
            sin = ttnn.reshape(
                ttnn.embedding(self.ctx_pos, d._sin_2d, layout=ttnn.TILE_LAYOUT), (1, 1, self.cap, d.head_dim)
            )
            kv = d.project_ctx_kv(self.ctx_dev, cos, sin)
            cos.deallocate(True)
            sin.deallocate(True)
            for li, (k_new, v_new) in enumerate(kv):
                ttnn.assign(k_new, self.ctx_k[li])
                ttnn.assign(v_new, self.ctx_v[li])
                k_new.deallocate(True)
                v_new.deallocate(True)
        self._mask_key = None  # fresh generation: ramp state differs, rebuild masks
        # Fresh seed: the next replay's start-of-body merge must be a no-op
        # (identity), not the previous generation's stale commit map.
        h = ttnn.from_torch(
            torch.arange(self.cap, dtype=torch.int64).reshape(1, self.cap),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=self._mapper,
        )
        ttnn.copy_host_to_device_tensor(h, self.merge_idx)
        h.deallocate(True)

    def _pv_setup(self, start, max_new):
        """Allocate the packed-verify persistent inputs (see use_packed).

        S_k is capped at capture to cover the whole generation (columns past the
        live top are NEG -> exact; the captured program never changes shape).
        Ring metadata per layer type mirrors SpeculativeDecoder._pv_setup; tables
        are WIDTH-MATCHED to their masks (the packed SDPA attends the table
        width -- root cause #4 on the MTP side).
        """
        t = self.target
        P_v = self.P_v
        self.pv_ring = {}
        rep = {}
        for i, layer in enumerate(t.layers):
            lt = t.hf_config.layer_types[i]
            mod = getattr(layer.self_attn.config, "cache_position_modulo", None)
            self.pv_ring.setdefault(lt, int(mod) if mod is not None else None)
            rep.setdefault(lt, i)
            if lt in self.pv_ring and self.pv_ring[lt] != (int(mod) if mod is not None else None):
                raise NotImplementedError("packed dflash verify needs uniform pools per layer type")
        horizon = start + max_new + P_v + 64
        self.pv_sk = ((horizon + 1023) // 1024) * 1024
        rows_t = {}
        installed = getattr(t, "_active_page_tables_per_layer", None)
        flat = (self.page_table_torch[0] if self.page_table_torch.dim() > 1 else self.page_table_torch).to(torch.int64)
        for lt in self.pv_ring:
            rows_t[lt] = flat
        if installed:
            for lt, i in rep.items():
                lpt = installed[i]
                if lpt is not None and hasattr(lpt, "dim"):
                    rows_t[lt] = (lpt[0] if lpt.dim() > 1 else lpt).to(torch.int64)
        bs = 64
        self.pv_tables = []
        cache_by_type = {}
        for i in range(len(t.layers)):
            lt = t.hf_config.layer_types[i]
            if lt not in cache_by_type:
                ring = self.pv_ring.get(lt)
                width = (ring // bs) if ring else max(1, self.pv_sk // bs)
                row = rows_t[lt]
                width = min(width, int(row.shape[0]))
                cache_by_type[lt] = ttnn.from_torch(
                    row[:width].to(torch.int32).reshape(1, width),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self._mapper,
                )
            self.pv_tables.append(cache_by_type[lt])
        z = torch.zeros
        mkT = dict(device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper)
        mkU = dict(device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._mapper)
        ring = self.pv_ring.get("sliding_attention")
        self.pv_ssl = ring if ring else self.pv_sk
        self.pv_slide_ring = ring
        self.pv_pos = ttnn.from_torch(z(1, P_v, dtype=torch.int64), **mkU)
        # Verify masks are built ON DEVICE from pv_iota + the block positions
        # (sub/gtz/mul -- probe-verified boundary-exact in fp32; bf16 would
        # round positions > 256). Only the bounded RING slide mask stays a host
        # upload (ring-width, tiny; the wrap modulo is host math).
        self.pv_iota = ttnn.from_torch(
            torch.arange(self.pv_sk, dtype=torch.float32).reshape(1, 1, 1, self.pv_sk),
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._mapper,
        )
        self.pv_mask_slide = ttnn.from_torch(z(1, 1, P_v, self.pv_ssl, dtype=torch.bfloat16), **mkT) if ring else None
        # ONE [P_v] int32 upload; the body slices per-position [1] views for the
        # fallback KV writes (was P_v singleton uploads -- pure dispatch waste).
        self.pv_widx_all = ttnn.from_torch(
            z(P_v, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mapper,
        )

    def _pv_host_masks(self, start):
        """P-row target-side masks (H-repeated in-trace). Full: causal to
        start+p over the capped S_k. Sliding: ring-slot formula when bounded
        (include iff (P_v-1-p) <= (top-j) mod ring < (P_v-1-p)+W and <= top),
        else the absolute window."""
        P_v = self.P_v
        W = self.target.hf_config.sliding_window
        NEG = -1e9
        j = torch.arange(self.pv_sk)
        mf = torch.empty(P_v, self.pv_sk)
        for pi in range(P_v):
            mf[pi] = torch.where(j <= start + pi, 0.0, NEG)
        ring = self.pv_ring.get("sliding_attention")
        if ring:
            top = start + P_v - 1
            jr = torch.arange(ring)
            d = torch.remainder(top - jr, ring)
            ms = torch.empty(P_v, ring)
            for pi in range(P_v):
                lo = P_v - 1 - pi
                ok = (d >= lo) & (d < lo + W) & (d <= top)
                ms[pi] = torch.where(ok, 0.0, NEG)
        else:
            ms = torch.empty(P_v, self.pv_sk)
            for pi in range(P_v):
                up = start + pi
                ms[pi] = torch.where((j <= up) & (j > up - W), 0.0, NEG)
        return (
            mf.reshape(1, 1, P_v, -1).to(torch.bfloat16),
            ms.reshape(1, 1, P_v, -1).to(torch.bfloat16),
        )

    def _pv_upload(self, start):
        P_v = self.P_v
        # pv positions are a PREFIX of blk_pos ([start..start+K]); the body
        # slices them in-trace and builds both masks ON DEVICE (unbounded).
        # Only the bounded ring slide mask (wrap modulo) crosses the host
        # boundary, plus the [P_v] write-idx vector.
        if self.pv_slide_ring:
            ring, W = self.pv_slide_ring, self.target.hf_config.sliding_window
            NEG = -1e9  # NOT -inf: the flash SDPA NaNs on -inf additive masks
            top = start + P_v - 1
            jr = torch.arange(ring)
            d = torch.remainder(top - jr, ring)
            ms = torch.empty(P_v, ring)
            for pi in range(P_v):
                lo = P_v - 1 - pi
                ok = (d >= lo) & (d < lo + W) & (d <= top)
                ms[pi] = torch.where(ok, 0.0, NEG)
            h = ttnn.from_torch(
                ms.reshape(1, 1, P_v, ring).to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=self._mapper,
            )
            ttnn.copy_host_to_device_tensor(h, self.pv_mask_slide)
            h.deallocate(True)
        h = ttnn.from_torch(
            torch.arange(start, start + P_v, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        ttnn.copy_host_to_device_tensor(h, self.pv_widx_all)
        h.deallocate(True)

    def capture(self, anchor_id, start, max_new=256):
        """Compile-run then capture the fused body at the first-iteration inputs."""
        self.anchor, self.start = anchor_id, start
        if self.use_packed:
            self._pv_setup(start, max_new)
            self._pv_upload(start)
        self.target.dflash_capture_taps(self.drafter.target_layer_ids, buffers=self.tap_bufs)
        self._upload_iter_inputs(anchor_id, start)
        self._body()  # compile pass (idempotent KV writes)
        ttnn.synchronize_device(self.mesh_device)
        # outputs are PERSISTENT slots (out_ids/fc_prev) -- do not deallocate
        # them between compile and capture.
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._out = self._body()
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        self.trace = tid

    def restore_model_logits_mode(self):
        """Leave the target in its default gathered-logits mode.

        The fused body sets ``_dflash_sharded_logits`` so the verify can hand
        back pre-gather logits for the sampling module; anything else reading
        full-vocab logits from this model afterwards (host argmax, a control
        arm, the plain demo path) must not inherit that.
        """
        self.target._dflash_sharded_logits = False

    def _read_ids(self, t):
        src = ttnn.get_device_tensors(t)[0] if self._tp > 1 else t
        return ttnn.to_torch(src).reshape(-1).to(torch.int64).tolist()

    def step(self, first=False):
        """One iteration: (replay unless first) -> acceptance -> commit -> host updates.

        Returns (accepted_tokens_list, bonus, produced)."""
        _prof = _os.environ.get("GEMMA4_DFLASH_PROF") == "1"
        if _prof:
            import time as _t

            _t0 = _t.perf_counter()
        if not first:
            self._upload_iter_inputs(self.anchor, self.start)
            if _prof:
                _t1 = _t.perf_counter()
            ttnn.execute_trace(self.mesh_device, self.trace, cq_id=0, blocking=False)
        elif _prof:
            _t1 = _t0
        out_ids_t, fc_out = self._out
        # Verify truncation: only the first V drafts were verified (P_v rows).
        ids = self._read_ids(out_ids_t)
        drafts = ids[: self.V]
        posterior = ids[self.K : self.K + self.P_v]
        if _prof:
            _t2 = _t.perf_counter()
        if _os.environ.get("GEMMA4_DFLASH_DEBUG_STEP") == "1":
            print(f"[dbg] start={self.start} drafts={drafts} posterior={posterior}", flush=True)
        acc = 0
        for dtok, ptok in zip(drafts, posterior[:-1]):
            if dtok == ptok:
                acc += 1
            else:
                break
        bonus = posterior[acc]
        produced = acc + 1
        # Commit taps ON DEVICE: fc rows [0..produced) are positions
        # [start..start+produced). Build the row map the NEXT replay's start-of-
        # body merge applies to [ctx_dev | fc_prev]: identity (optionally window-
        # shifted) for kept rows, cap+j for committed row j. ~8 KB upload
        # replaces the old fc readback + full ctx re-upload (~22 MB/iter).
        win_len = self.ctx_len - self.win_first
        idx = torch.arange(self.cap, dtype=torch.int64)
        if win_len + produced <= self.cap:
            m = idx.clone()
            m[win_len : win_len + produced] = self.cap + torch.arange(produced)
        else:
            shift = win_len + produced - self.cap
            keep_n = self.cap - produced
            m = torch.empty(self.cap, dtype=torch.int64)
            m[:keep_n] = idx[:keep_n] + shift
            m[keep_n:] = self.cap + torch.arange(produced)
            self.win_first += shift
        h = ttnn.from_torch(
            m.reshape(1, self.cap),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=self._mapper,
        )
        ttnn.copy_host_to_device_tensor(h, self.merge_idx)
        h.deallocate(True)
        if self.ctx_cache:
            # positions of fc_prev's rows for the NEXT replay's roped commit
            # (pre-advance start: row j holds position start + j).
            cp = torch.arange(self.start, self.start + self.P_v, dtype=torch.int64).reshape(1, self.P_v)
            h = ttnn.from_torch(cp, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)
            ttnn.copy_host_to_device_tensor(h, self.commit_pos)
            h.deallocate(True)
        self.ctx_len = self.start + produced
        accepted = [self.anchor] if False else drafts[:acc]
        self.anchor = bonus
        self.start = self.start + produced
        if _prof:
            _t3 = _t.perf_counter()
            print(
                f"[prof] upload={1000*(_t1-_t0):.1f} trace+read={1000*(_t2-_t1):.1f} "
                f"commit={1000*(_t3-_t2):.1f} total={1000*(_t3-_t0):.1f}",
                flush=True,
            )
        return accepted, bonus, produced
