"""DFlash2 block-diffusion drafter for tt-metal.

Two classes:

* ``DFlash2Draft`` — the component-validated reference port (tests/dflash2_ttnn_bringup.py, PCC
  A1-A6). Recomputes the whole context every call (O(C) per block, variable shapes), so it is a
  correctness oracle, NOT the loop drafter.
* ``DFlash2Drafter`` — the production form. Every per-iteration forward has FIXED shapes (8 block
  rows; 8 context-extend rows), so all programs compile once, before the verify trace is captured,
  and nothing recompiles while a trace is parked (see SpeculativeDecoder for why that matters). The
  5 draft layers keep a paged K/V cache of the CONTEXT — k/v_proj of hctx = hidden_norm(fc(taps)),
  k_norm'd + RoPE'd at absolute positions, exactly the reference's ``past_key_values_draft`` — so a
  block draft is O(block) compute plus one paged decode-SDPA read over the cache.

Indexing (aligned to SpeculativeDecoder, anchor p = last position the base has consumed):
  context = positions 0..p (taps written by fill_context / extend_context), block = [pending @ p+1,
  MASK x7 @ p+2..p+8]. ``draft(pending, C=p+1)`` returns drafts[j] = candidate for p+2+j.

All matmuls/norms/SDPA use HiFi4 + fp32 dest-acc (the bf16-variance rms_norm was the real-input
precision killer; see the bring-up). RoPE is computed here (theta 1e7, head_dim 128, full).
"""
import glob
import os

import torch
from safetensors.torch import load_file

import ttnn
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill

H, NH, NKV, HD = 5120, 32, 8, 128
BLOCK, MASK_ID = 8, 248070
K_DRAFT = BLOCK - 1
TAP_LAYERS = (5, 19, 33, 47, 61)
SLIDING_WINDOW = 2048
# Drafter checkpoint: a local directory or a Hugging Face repo id (resolved through the HF cache,
# honouring HF_HUB_OFFLINE). incoai/Qwen3.8-27B-DFlash2 targets Qwen3.8-27B; the served Qwen3.6-27B
# needs z-lab/Qwen3.6-27B-DFlash (model_config sets that default via DFLASH_WEIGHTS).
DEFAULT_WEIGHTS = "incoai/Qwen3.8-27B-DFlash2"


def resolve_weights_dir(weights):
    """Local dir -> itself; otherwise an HF repo id ('org/name' or 'org/name@revision') snapshotted
    into the HF cache (local_files_only when HF_HUB_OFFLINE=1)."""
    if os.path.isdir(weights):
        return weights
    from huggingface_hub import snapshot_download

    repo, _, rev = weights.partition("@")
    return snapshot_download(repo, revision=rev or None, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")


def load_config(weights_dir):
    """Drafter config.json -> the fields the port needs. Covers DFlash2 (incoai/z-lab *-DFlash2: conv +
    selector, all layers bidirectional sliding) and DFlash v1 (z-lab *-DFlash: no conv/selector, causal
    sliding layers + a bidirectional full-attention last layer, block 16, taps [1,16,31,46,61])."""
    import json

    with open(f"{resolve_weights_dir(weights_dir)}/config.json") as f:
        c = json.load(f)
    d = c.get("dflash_config") or {}
    n = int(c.get("num_hidden_layers", 5))
    types = c.get("layer_types") or ["sliding_attention"] * n
    is_causal = c.get("is_causal", None)
    causal = [(t == "sliding_attention") if is_causal is None else bool(is_causal) for t in types]
    window = int(c.get("sliding_window") or 0) if c.get("use_sliding_window", True) else 0
    windows = [window if t == "sliding_attention" else 0 for t in types]
    rp = c.get("rope_parameters") or {}
    return {
        "num_layers": n,
        "block": int(d.get("block_size", c.get("block_size", 8))),
        "taps": tuple(int(x) for x in d.get("target_layer_ids", TAP_LAYERS)),
        "mask_id": int(d.get("mask_token_id", MASK_ID)),
        "rope_theta": float(rp.get("rope_theta") or c.get("rope_theta") or 1e7),
        "causal": causal,  # per layer: causal within the block (v1 sliding layers) or bidirectional
        "windows": windows,  # per layer: sliding window (0 = none)
        "has_conv": "conv_kernel_size" in d,
        "has_selector": "selector_rank" in d,
        "arch": (c.get("architectures") or ["?"])[0],
    }


CKC = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def _expansion(groups=320, gsz=16):
    E = torch.zeros(groups, H)
    for g in range(groups):
        E[g, g * gsz : (g + 1) * gsz] = 1.0
    return E


def _shift(L):
    S = torch.zeros(1, L, L)
    for t in range(1, L):
        S[0, t, t - 1] = 1.0
    return S


def rope_tables(theta, positions):
    """cos/sin (rows, HD) fp32 for absolute ``positions`` (theta 1e7, full 128-dim rotate-half)."""
    inv = 1.0 / (theta ** (torch.arange(0, HD, 2).float() / HD))
    f = torch.outer(positions.float(), inv)
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def select_path(unary, cand, hp, anchor, pred_cb, succ_cb):
    """CandidateSelector greedy chain (host). unary (K,16) float, cand (K,16) long, hp (K,256) float
    = hidden_projection(draft_hidden); returns the K chosen ids (reference CandidateSelector.select,
    temperature 0)."""
    pred = int(anchor)
    path = []
    for pos in range(unary.shape[0]):
        edge = (pred_cb[pred] * hp[pos]) @ succ_cb[cand[pos]].T  # (16,)
        idx = int((unary[pos] + edge).argmax())
        pred = int(cand[pos, idx])
        path.append(pred)
    return path


def taps_to_host(mesh, taps_dev, rows=None):
    """5 dim-fractured device taps [1,1,n,dim/tp] -> (1, rows or n, 25600) host float (concat of the
    5 layers' residuals, each gathered across the TP mesh)."""
    comp = ttnn.ConcatMeshToTensor(mesh, dim=-1)
    outs = []
    for t in taps_dev:
        x = ttnn.to_torch(t, mesh_composer=comp).float().reshape(1, -1, H)
        outs.append(x if rows is None else x[:, :rows])
    return torch.cat(outs, dim=-1)


class DFlash2Draft:
    """Validated reference port: recomputes the context each call (oracle for DFlash2Drafter)."""

    def __init__(self, mesh, weights_dir, embed_host, lmhead_host, rope_theta=None):
        self.mesh = mesh
        self.embed = embed_host  # (V,H) host
        self.lmhead = lmhead_host  # (V,H) host float (or None: propose_hidden only)
        self.cfg = load_config(weights_dir)
        self.theta = float(rope_theta if rope_theta is not None else self.cfg["rope_theta"])
        self.block = self.cfg["block"]
        self.taps = self.cfg["taps"]
        self.mask_id = self.cfg["mask_id"]
        weights_dir = resolve_weights_dir(weights_dir)
        s = {}
        for f in sorted(glob.glob(f"{weights_dir}/*.safetensors")):
            s.update(load_file(f))
        self.fc = self._dev(s["fc.weight"].T.contiguous())
        self.hnorm = self._dev(s["hidden_norm.weight"])
        self.fnorm = self._dev(s["norm.weight"])
        self.E = self._dev(_expansion())
        self.layers = [self._load_layer(s, i) for i in range(self.cfg["num_layers"])]
        self.has_selector = self.cfg["has_selector"]
        if self.has_selector:
            self.pred_cb = s["candidate_selector.predecessor_codebook"].float()  # (V,256)
            self.succ_cb = s["candidate_selector.successor_codebook"].float()
            self.hproj = s["candidate_selector.hidden_projection.weight"].float()  # (256,H)
        self.topk = 16

    # ---- device helpers ----
    def _dev(self, t, dtype=ttnn.bfloat16):
        if t.ndim == 1:
            t = t.reshape(1, -1)
        if t.is_floating_point() and dtype == ttnn.bfloat16 and t.dtype != torch.bfloat16:
            # Host-side cast so every upload site (fp32 prompt/seed taps, bf16 verify taps, fp32
            # cos/sin) constructs the device tensor from the SAME source dtype -> one program, compiled
            # before the verify trace is captured; a new source dtype in the loop would compile a new
            # program while the trace is parked.
            t = t.to(torch.bfloat16)
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )

    def free_weights(self):
        """Release every device weight (a reload with another checkpoint/block must not leak ~3.6 GB)."""
        seen = set()

        def _free(t):
            if isinstance(t, ttnn.Tensor) and id(t) not in seen:
                seen.add(id(t))
                ttnn.deallocate(t)

        for name in ("fc", "hnorm", "fnorm", "E", "S8", "hproj_dev"):
            _free(getattr(self, name, None))
            setattr(self, name, None)
        for lw in getattr(self, "layers", []):
            for t in lw.values():
                _free(t)
        self.layers = []
        self.dead = True

    def _host(self, x):
        return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))[:1]

    def _lin(self, x, w, **k):
        return ttnn.linear(x, w, compute_kernel_config=CKC, **k)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=1e-6, compute_kernel_config=CKC)

    def _load_layer(self, s, i):
        p = f"layers.{i}"
        lw = {
            "in_ln": self._dev(s[f"{p}.input_layernorm.weight"]),
            "post_ln": self._dev(s[f"{p}.post_attention_layernorm.weight"]),
        }
        for k in ("q", "k", "v", "o"):
            lw[k] = self._dev(s[f"{p}.self_attn.{k}_proj.weight"].T.contiguous())
        lw["qn"] = self._dev(s[f"{p}.self_attn.q_norm.weight"])
        lw["kn"] = self._dev(s[f"{p}.self_attn.k_norm.weight"])
        for tag, cp in (("ac", f"{p}.attention_conv"), ("mc", f"{p}.mlp_conv")):
            if f"{cp}.kernel_projection.weight" not in s:
                continue  # DFlash v1: no dynamic convs
            lw[f"{tag}_kp"] = self._dev(s[f"{cp}.kernel_projection.weight"].T.contiguous())
            bk = s[f"{cp}.base_kernel"].float()
            lw[f"{tag}_b00"] = self._dev(bk[0, 0].reshape(1, 1, H))
            lw[f"{tag}_b01"] = self._dev(bk[0, 1].reshape(1, 1, H))
            lw[f"{tag}_b10"] = self._dev(bk[1, 0].reshape(1, 1, H))
            lw[f"{tag}_b11"] = self._dev(bk[1, 1].reshape(1, 1, H))
        for k in ("gate", "up", "down"):
            lw[k] = self._dev(s[f"{p}.mlp.{k}_proj.weight"].T.contiguous())
        return lw

    # ---- draft ops (validated) ----
    def _rope(self, T):
        cos, sin = rope_tables(self.theta, torch.arange(T))
        return cos.reshape(1, T, HD), sin.reshape(1, T, HD)

    def _rot_half(self, x):
        sh = list(x.shape)
        d = sh[-1] // 2
        return ttnn.concat(
            [ttnn.neg(ttnn.slice(x, [0] * (len(sh) - 1) + [d], sh)), ttnn.slice(x, [0] * len(sh), sh[:-1] + [d])],
            dim=-1,
        )

    def _rope_apply(self, x, cos, sin):
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(self._rot_half(x), sin))

    def _gdc(self, values, kp, b0, b1, S, sel, L):
        off = sel * 640
        d0 = ttnn.slice(kp, [0, 0, off], [1, L, off + 320])
        d1 = ttnn.slice(kp, [0, 0, off + 320], [1, L, off + 640])
        c0 = ttnn.add(self._lin(d0, self.E), b0)
        c1 = ttnn.add(self._lin(d1, self.E), b1)
        return ttnn.add(ttnn.mul(c0, values), ttnn.mul(c1, ttnn.matmul(S, values, compute_kernel_config=CKC)))

    def _attn(self, hidden, ctx, cos, sin, lw, L, C, causal=False, window=0):
        q = self._rms(ttnn.reshape(self._lin(hidden, lw["q"]), (1, L, NH, HD)), lw["qn"])
        q = ttnn.transpose(q, 1, 2)
        k = ttnn.concat([self._lin(ctx, lw["k"]), self._lin(hidden, lw["k"])], dim=1)
        k = ttnn.transpose(self._rms(ttnn.reshape(k, (1, C + L, NKV, HD)), lw["kn"]), 1, 2)
        v = ttnn.transpose(
            ttnn.reshape(
                ttnn.concat([self._lin(ctx, lw["v"]), self._lin(hidden, lw["v"])], dim=1), (1, C + L, NKV, HD)
            ),
            1,
            2,
        )
        cos4 = ttnn.reshape(cos, (1, 1, C + L, HD))
        sin4 = ttnn.reshape(sin, (1, 1, C + L, HD))
        cq = ttnn.slice(cos4, [0, 0, C, 0], [1, 1, C + L, HD])
        sq = ttnn.slice(sin4, [0, 0, C, 0], [1, 1, C + L, HD])
        q = self._rope_apply(q, cq, sq)
        k = self._rope_apply(k, cos4, sin4)
        if causal or (window and C + L > window):
            # Reference _attention_mask: keys k visible to query q iff (not causal or k <= q) and
            # (no window or q - k < window and (causal or k - q < window)). q rows sit at C..C+L-1,
            # keys at 0..C+L-1.
            qp = torch.arange(C, C + L).view(L, 1)
            kp = torch.arange(C + L).view(1, C + L)
            vis = torch.ones(L, C + L, dtype=torch.bool)
            if causal:
                vis &= kp <= qp
            if window:
                vis &= (qp - kp) < window
                if not causal:
                    vis &= (kp - qp) < window
            m = torch.where(vis, 0.0, float("-inf")).view(1, 1, L, C + L)
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=self._dev(m), is_causal=False, scale=HD**-0.5, compute_kernel_config=CKC
            )
        else:
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=HD**-0.5, compute_kernel_config=CKC
            )
        return self._lin(ttnn.reshape(ttnn.transpose(o, 1, 2), (1, L, NH * HD)), lw["o"])

    def _mlp(self, x, lw):
        return self._lin(ttnn.mul(self._lin(x, lw["gate"], activation="silu"), self._lin(x, lw["up"])), lw["down"])

    def _layer(self, hidden, ctx, cos, sin, lw, S, L, C, li):
        h = self._rms(hidden, lw["in_ln"])
        causal, window = self.cfg["causal"][li], self.cfg["windows"][li]
        if "ac_kp" in lw:
            kp = self._lin(h, lw["ac_kp"])
            a = self._gdc(
                self._attn(
                    self._gdc(h, kp, lw["ac_b00"], lw["ac_b01"], S, 0, L), ctx, cos, sin, lw, L, C, causal, window
                ),
                kp,
                lw["ac_b10"],
                lw["ac_b11"],
                S,
                1,
                L,
            )
        else:
            a = self._attn(h, ctx, cos, sin, lw, L, C, causal, window)
        h = ttnn.add(hidden, a)
        h2 = self._rms(h, lw["post_ln"])
        if "mc_kp" in lw:
            kpm = self._lin(h2, lw["mc_kp"])
            m = self._gdc(
                self._mlp(self._gdc(h2, kpm, lw["mc_b00"], lw["mc_b01"], S, 0, L), lw),
                kpm,
                lw["mc_b10"],
                lw["mc_b11"],
                S,
                1,
                L,
            )
        else:
            m = self._mlp(h2, lw)
        return ttnn.add(h, m)

    def _select(self, hidden, logits, anchor):
        if not self.has_selector:
            return [int(t) for t in logits[0].argmax(-1)]  # DFlash v1: plain argmax per position
        unary, cand = logits.topk(self.topk, -1)  # (1,K,16)
        hp = hidden @ self.hproj.T  # (1,K,256)
        return select_path(unary[0], cand[0], hp[0], anchor, self.pred_cb, self.succ_cb)

    # ---- public ----
    def block_embedding(self, anchor, L=None):
        L = L or self.block
        blk = torch.tensor([[int(anchor)] + [self.mask_id] * (L - 1)])
        return torch.nn.functional.embedding(blk, self.embed)  # (1,L,H)

    def propose_hidden(self, taps_cat_host, anchor, C, noise_host=None, L=None):
        """(1,C,25600) host taps + anchor -> draft_hidden (1,L-1,5120) host float (rows 1..L-1 of
        the final-normed block)."""
        L = L or self.block
        hctx = self._rms(self._lin(self._dev(taps_cat_host), self.fc), self.hnorm)
        if noise_host is None:
            noise_host = self.block_embedding(anchor, L)
        cos_h, sin_h = self._rope(C + L)
        cos = self._dev(cos_h)
        sin = self._dev(sin_h)
        S = self._dev(_shift(L))
        h = self._dev(noise_host)
        for li, lw in enumerate(self.layers):
            h = self._layer(h, hctx, cos, sin, lw, S, L, C, li)
        h = self._rms(h, self.fnorm)
        return self._host(h)[:, 1 - L :, :].float()  # (1,L-1,5120)

    def propose(self, taps_cat_host, anchor, C, noise_host=None, L=None):
        dh = self.propose_hidden(taps_cat_host, anchor, C, noise_host, L)
        logits = dh @ self.lmhead.T
        return self._select(dh, logits, anchor)


class DFlash2Drafter(DFlash2Draft):
    """Fixed-shape, KV-cached DFlash2 drafter for the spec-decode loop (see module docstring).

      fill_context(taps, chunk_start)   prompt context: S (bucket) rows -> paged_fill_cache
      extend_context(taps8, slot0, n)   8 verify-tap rows, first n real -> slots slot0..slot0+n-1,
                                        the rest -> the scratch block (fixed shape whatever n is)
      draft(anchor, C)                  block [anchor, MASK*7] at C..C+7 -> 7 draft ids

    Weights REPLICATED on the mesh (correctness-first; TP-sharding is the next step). ``lm_head_fn``
    maps the final-normed block [1,1,8,H] to replicated logits [1,1,8,V] (the target's _lm_head);
    top-16 runs on device, the 7-step selector chain on host from ~10 KB of readback.
    """

    def __init__(
        self,
        mesh,
        weights_dir=DEFAULT_WEIGHTS,
        embed_host=None,
        lm_head_fn=None,
        rope_theta=None,
        kv_dtype=ttnn.bfloat16,
        sliding_window=None,
        block=None,
    ):
        super().__init__(mesh, weights_dir, embed_host, None, rope_theta)
        assert embed_host is not None and lm_head_fn is not None
        self.lm_head = lm_head_fn
        self.kv_dtype = kv_dtype
        # Draft block (rows per draft/extend, = the verify T). Defaults to the checkpoint's block_size;
        # a smaller one is allowed (the reference drafts short blocks at the end of a sequence).
        if block is not None:
            assert 2 <= block <= self.cfg["block"], f"block {block} not in [2, {self.cfg['block']}]"
            self.block = int(block)
        self.K = self.block - 1
        _w = os.environ.get("QWEN36_DFLASH_WINDOW")
        if sliding_window is not None:
            self.windows = [sliding_window if w else 0 for w in self.cfg["windows"]]
        elif _w is not None:
            self.windows = [int(_w) if w else 0 for w in self.cfg["windows"]]
        else:
            self.windows = list(self.cfg["windows"])
        self.causal = list(self.cfg["causal"])
        self.S8 = self._dev(_shift(self.block))
        self.hproj_dev = self._dev(self.hproj.T.contiguous()) if self.has_selector else None  # (H,256)
        # Single-row HEIGHT shard for the per-row paged_update_cache (one 32-row tile on one core).
        self._sc1 = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD),
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        grid = mesh.compute_with_storage_grid_size()
        self._sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        self.kv = None
        self.page_table = None
        self.block_size = None
        self.scratch_block = None
        self._ptB = None
        self.last_hidden = None  # (1,K,5120) host float after draft() (tests / debug)
        self.last_topk = None

    # ---- draft KV cache ----
    def alloc_kv(self, page_table_torch, block_size):
        """Paged context K/V for the 5 draft layers: num_blocks + 1 blocks (the extra one is the
        extend's scratch for padding rows — same trick as the MTP reseed). Call before any trace is
        captured (the buffers must be live at capture time)."""
        assert self.layers and not getattr(self, "dead", False), "drafter weights were released; rebuild the decoder"
        assert self.kv is None, "draft KV already allocated"
        nb = int(page_table_torch.shape[-1])
        shape = [nb + 1, NKV, block_size, HD]
        rep = ttnn.ReplicateTensorToMesh(self.mesh)

        def _mk():
            return ttnn.as_tensor(
                torch.zeros(shape, dtype=torch.bfloat16),
                device=self.mesh,
                dtype=self.kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        self.kv = [(_mk(), _mk()) for _ in range(len(self.layers))]
        self.page_table = page_table_torch.to(torch.int32)
        self.block_size = int(block_size)
        self.scratch_block = nb
        self._ptB = self._rm(self.page_table.repeat(self.block, 1).contiguous())

    def free_kv(self):
        if self.kv is None:
            return
        for kc, vc in self.kv:
            ttnn.deallocate(kc)
            ttnn.deallocate(vc)
        ttnn.deallocate(self._ptB)
        self.kv = self._ptB = None

    def _rm(self, t):
        """int32 ROW_MAJOR replicated device tensor."""
        return ttnn.from_torch(
            t.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _rope_dec(self, positions):
        """Decode-layout tables [1,B,1,HD] (one rotation per row) for the given absolute positions."""
        cos, sin = rope_tables(self.theta, positions)
        B = positions.shape[0]
        return self._dev(cos.reshape(1, B, 1, HD)), self._dev(sin.reshape(1, B, 1, HD))

    def _write_rows(self, li, k, v, pos_t, pt_t):
        """Write B=8 rows of K/V ([1,8,NKV,HD] each) into layer li's cache at absolute ``pos_t[i]`` via
        page table row ``pt_t[i]``. One single-row paged_update_cache per row: the rows share physical
        blocks, and a batched call puts several cores on one 32-row tile (last writer wins) — the
        same idiom TPAttention.forward_decode uses under alias_kv_write."""
        kc, vc = self.kv[li]
        B = k.shape[1]
        nb = pt_t.shape[-1]
        k_p = ttnn.pad(k, [1, B, ttnn.TILE_SIZE, HD], [0, 0, 0, 0], 0.0)
        v_p = ttnn.pad(v, [1, B, ttnn.TILE_SIZE, HD], [0, 0, 0, 0], 0.0)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        for i in range(B):
            pos_i = ttnn.slice(pos_t, (i,), (i + 1,))
            pt_i = ttnn.slice(pt_t, (i, 0), (i + 1, nb))
            for cache, src in ((kc, k_p), (vc, v_p)):
                row = ttnn.slice(src, (0, i, 0, 0), (1, i + 1, ttnn.TILE_SIZE, HD))
                row_sh = ttnn.to_memory_config(row, self._sc1)
                ttnn.deallocate(row)
                ttnn.experimental.paged_update_cache(cache, row_sh, update_idxs_tensor=pos_i, page_table=pt_i)
                ttnn.deallocate(row_sh)
            ttnn.deallocate(pos_i)
            ttnn.deallocate(pt_i)
        ttnn.deallocate(k_p)
        ttnn.deallocate(v_p)

    def _hctx(self, taps_host):
        """(1,n,25600) host taps -> hidden_norm(fc(taps)) [1,n,H] device."""
        return self._rms(self._lin(self._dev(taps_host), self.fc), self.hnorm)

    def _ctx_kv(self, hctx, lw, n):
        """Context K (k_norm'd, pre-RoPE) / V for n rows: [1,n,NKV,HD] each."""
        k = self._rms(ttnn.reshape(self._lin(hctx, lw["k"]), (1, n, NKV, HD)), lw["kn"])
        v = ttnn.reshape(self._lin(hctx, lw["v"]), (1, n, NKV, HD))
        return k, v

    # ---- context maintenance ----
    def fill_context(self, taps_host, chunk_start):
        """Prompt context for one prefill chunk: taps (1,S,25600) host, S a multiple of block_size,
        chunk_start block-aligned. Rows past the real prompt (bucket padding) write junk that is
        overwritten (seed / block writes) before any read reaches it."""
        assert self.kv is not None, "alloc_kv first"
        S = taps_host.shape[1]
        bs = self.block_size
        assert (
            S % bs == 0 and chunk_start % bs == 0
        ), f"fill needs block-aligned rows/start (S={S}, start={chunk_start})"
        blk0 = chunk_start // bs
        blkN = min(blk0 + S // bs, int(self.page_table.shape[-1]))
        rows = (blkN - blk0) * bs
        assert rows > 0, f"chunk_start {chunk_start} is past the page table"
        hctx = self._hctx(taps_host)
        cos, sin = rope_tables(self.theta, torch.arange(chunk_start, chunk_start + S))
        cos = self._dev(cos.reshape(1, 1, S, HD))
        sin = self._dev(sin.reshape(1, 1, S, HD))
        chunk_pt = self._rm(self.page_table[:, blk0:blkN].contiguous())
        for li, lw in enumerate(self.layers):
            kc, vc = self.kv[li]
            k, v = self._ctx_kv(hctx, lw, S)
            k = apply_partial_rope_prefill(ttnn.transpose(k, 1, 2), cos, sin, NKV, HD)  # [1,NKV,S,HD]
            v = ttnn.transpose(v, 1, 2)
            if rows < S:
                k = ttnn.slice(k, (0, 0, 0, 0), (1, NKV, rows, HD))
                v = ttnn.slice(v, (0, 0, 0, 0), (1, NKV, rows, HD))
            if k.dtype != self.kv_dtype:
                k = ttnn.typecast(k, self.kv_dtype)
                v = ttnn.typecast(v, self.kv_dtype)
            ttnn.experimental.paged_fill_cache(kc, k, chunk_pt, batch_idx=0)
            ttnn.experimental.paged_fill_cache(vc, v, chunk_pt, batch_idx=0)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        ttnn.deallocate(hctx)
        ttnn.deallocate(chunk_pt)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

    def extend_context(self, taps8_host, slot0, n_valid):
        """Append the committed positions' context: taps8 (1,block,25600) host (verify-window taps),
        rows 0..n_valid-1 -> slots slot0..slot0+n_valid-1; the rest -> the scratch block (their
        position is 0, their page-table row names only the scratch block). Fixed shape for every n."""
        assert self.kv is not None, "alloc_kv first"
        B = self.block
        assert taps8_host.shape[1] == B and 0 <= n_valid <= B
        n = int(n_valid)
        pos = torch.zeros(B, dtype=torch.int32)
        pos[:n] = torch.arange(slot0, slot0 + n, dtype=torch.int32)
        pt = self.page_table.repeat(B, 1).contiguous()
        pt[n:, :] = self.scratch_block
        hctx = self._hctx(taps8_host)
        cos, sin = self._rope_dec(pos)
        pos_t = self._rm(pos)
        pt_t = self._rm(pt)
        for li, lw in enumerate(self.layers):
            k, v = self._ctx_kv(hctx, lw, B)
            k = apply_partial_rope_decode(k, cos, sin, NKV, B, HD)
            self._write_rows(li, k, v, pos_t, pt_t)
        for t in (hctx, cos, sin, pos_t, pt_t):
            ttnn.deallocate(t)

    # ---- block draft ----
    def _attn_cached(self, hb, lw, li, cos, sin, pos_t, cur_t):
        """Block-row attention over [cached context ++ block]. hb [1,B,H] (conv-prepared) normed rows.
        Writes the block's own K/V at pos_t (C..C+B-1), then one paged decode-SDPA. cur_t is per layer
        type: bidirectional layers use C+B-1 for every row (row i attends [0, C+B-1] = whole context +
        whole block, the reference's is_causal=False, minus the window's per-row start); causal layers
        (DFlash v1 sliding layers) use C+i for row i (the substrate's own verify pattern), which also
        makes the sliding window exact per row."""
        B = self.block
        q = self._rms(ttnn.reshape(self._lin(hb, lw["q"]), (1, B, NH, HD)), lw["qn"])
        k = self._rms(ttnn.reshape(self._lin(hb, lw["k"]), (1, B, NKV, HD)), lw["kn"])
        v = ttnn.reshape(self._lin(hb, lw["v"]), (1, B, NKV, HD))
        q = apply_partial_rope_decode(q, cos, sin, NH, B, HD)
        k = apply_partial_rope_decode(k, cos, sin, NKV, B, HD)
        self._write_rows(li, k, v, pos_t, self._ptB)
        kc, vc = self.kv[li]
        kw = {"sliding_window_size": self.windows[li]} if self.windows[li] else {}
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kc,
            vc,
            page_table_tensor=self._ptB,
            cur_pos_tensor=cur_t,
            scale=HD**-0.5,
            program_config=self._sdpa_cfg,
            compute_kernel_config=CKC,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **kw,
        )
        ttnn.deallocate(q)
        o = ttnn.reshape(o, (1, B, NH * HD))
        return self._lin(o, lw["o"])

    def _layer_cached(self, hidden, lw, li, cos, sin, pos_t, cur_t):
        L, S = self.block, self.S8
        h = self._rms(hidden, lw["in_ln"])
        if "ac_kp" in lw:
            kp = self._lin(h, lw["ac_kp"])
            a = self._gdc(
                self._attn_cached(
                    self._gdc(h, kp, lw["ac_b00"], lw["ac_b01"], S, 0, L), lw, li, cos, sin, pos_t, cur_t
                ),
                kp,
                lw["ac_b10"],
                lw["ac_b11"],
                S,
                1,
                L,
            )
        else:
            a = self._attn_cached(h, lw, li, cos, sin, pos_t, cur_t)
        h = ttnn.add(hidden, a)
        h2 = self._rms(h, lw["post_ln"])
        if "mc_kp" in lw:
            kpm = self._lin(h2, lw["mc_kp"])
            m = self._gdc(
                self._mlp(self._gdc(h2, kpm, lw["mc_b00"], lw["mc_b01"], S, 0, L), lw),
                kpm,
                lw["mc_b10"],
                lw["mc_b11"],
                S,
                1,
                L,
            )
        else:
            m = self._mlp(h2, lw)
        return ttnn.add(h, m)

    def draft(self, anchor, C):
        """Block [anchor, MASK*(B-1)] at positions C..C+B-1 over context 0..C-1 -> B-1 = K draft ids
        (drafts[j] is the candidate for position C+1+j). ``anchor`` is the token AT position C (the
        substrate's ``pending``), ``C`` = len(context) = p+1."""
        assert self.kv is not None, "alloc_kv first"
        B = self.block
        pos = torch.arange(C, C + B, dtype=torch.int32)
        cos, sin = self._rope_dec(pos)
        pos_t = self._rm(pos)
        cur_full = self._rm(torch.full((B,), C + B - 1, dtype=torch.int32))
        cur_causal = self._rm(pos) if any(self.causal) else None
        h = self._dev(self.block_embedding(anchor))  # [1,B,H]
        for li, lw in enumerate(self.layers):
            h = self._layer_cached(h, lw, li, cos, sin, pos_t, cur_causal if self.causal[li] else cur_full)
        h = self._rms(h, self.fnorm)
        logits = self.lm_head(ttnn.reshape(h, (1, 1, B, H)))  # [1,1,B,V] replicated
        d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        if self.has_selector:
            vals, idx = ttnn.topk(logits, self.topk, dim=-1, largest=True, sorted=True)
            ttnn.deallocate(logits)
            hp = self._lin(h, self.hproj_dev)  # [1,B,256]
            unary = d0(vals).float().reshape(B, self.topk)[1:]
            cand = d0(idx).long().reshape(B, self.topk)[1:]
            hph = d0(hp).float().reshape(B, -1)[1:]
            self.last_topk = (unary, cand)
            out = select_path(unary, cand, hph, anchor, self.pred_cb, self.succ_cb)
            for t in (vals, idx, hp):
                ttnn.deallocate(t)
        else:
            # DFlash v1: greedy argmax per block row (same untilize+argmax idiom as the verify trace).
            u = ttnn.untilize(logits, use_multicore=True)
            ids = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,B] uint32 RM
            out = [int(x) for x in d0(ids).reshape(-1)[:B][1:]]
            for t in (u, ids, logits):
                ttnn.deallocate(t)
        self.last_hidden = self._host(h)[:, 1:, :].float()
        for t in (h, cos, sin, pos_t, cur_full):
            ttnn.deallocate(t)
        if cur_causal is not None:
            ttnn.deallocate(cur_causal)
        return out
