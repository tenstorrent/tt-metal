# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash2 drafter (RedHatAI/gemma-4-31B-it-speculator.dflash2) for the QB2 DFlash speculator.

Architecture per vllm-project/speculators @ 442ce1bf (models/dflash2), adapted there from z-lab/dflash
@ 07ebd93 (MIT): five Qwen3 layers (32 Q / 16 KV heads x 256, MLP 21504, all sliding, window 2048
anchor-relative, non-causal in the block, RoPE theta 1e4), each wrapping attention and MLP in a 2-tap
grouped dynamic causal convolution inside the block:
    conv(h)[t] = h[t] * (base[0] + d[t, 0]) + [t >= 1] h[t-1] * (base[1] + d[t, 1])
with d one coefficient per 16-channel group, projected from the sublayer's pre-conv input. Draft ids
come from a candidate selector over the unary top-16 of the target lm_head (raw, no softcap): a greedy
path from the anchor scoring unary + (pred_codebook[prev] * hidden_projection(h)) . succ_codebook[c].
The selector walk follows select_path in changh95/qwen3.8-27b-p300x2 dflash2.py (Apache-2.0).

On device, TP4 as in DFlashDrafter (8 Q / 4 KV heads and 5376 MLP columns per chip). Conv coefficients:
one replicated projection [H, 4 x 352] (four (io, tap) sections of 336 groups, each padded to 11 tiles),
the sections stacked as 128 rows and expanded to channels by one 0/1 matmul; the tap-1 shift is a
32x32 0/1 matmul over the block rows. The unary top-16 runs per vocabulary shard on device; the
host merges the 4 x 16 candidates and walks the 7-step selector path (codebooks on host, fp32).
"""

import os

import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.dflash import CTX_ROWS, ROWS, DFlashDrafter

DFLASH2_ID = "RedHatAI/gemma-4-31B-it-speculator.dflash2"
DFLASH2_REVISION = "b0cd01b4918aae6b6388272950c4f9851734674b"
SECTION = 352  # 336 conv groups padded to whole tiles


def select_path(unary, cand, hp, anchor, pred_cb, succ_cb):
    """Greedy selector path: position 0's predecessor is the anchor, later positions condition on
    the previous selection. unary/cand [K, top_k], hp [K, rank]; first maximum wins ties.

    As in changh95's select_path, every edge a step can take is scored at once (the predecessor of
    step k is the anchor or one of step k-1's candidates); the walk is then table lookups."""
    k, n = cand.shape
    preds = torch.cat([torch.full((1, n), int(anchor), dtype=cand.dtype), cand[:-1]])
    edges = torch.matmul(pred_cb[preds] * hp[:, None, :], succ_cb[cand].transpose(1, 2))
    score, ids = (unary[:, None, :] + edges).tolist(), cand.tolist()
    i, path = 0, []
    for pos in range(k):
        row = score[pos][i]
        i = max(range(n), key=row.__getitem__)
        path.append(ids[pos][i])
    return path


class DFlash2Drafter(DFlashDrafter):
    REPO, REVISION = DFLASH2_ID, DFLASH2_REVISION
    # The 2048 anchor-relative window plus the block must be live at once; 72 tiles keep any
    # 32 consecutive written positions in distinct tiles and give 256-key SDPA chunks.
    RING = 2304
    SEED_ROWS = 2080  # covers the 2048-position anchor-relative window for any tile-padding remainder
    DEFAULT_WEIGHTS = "bf8"  # same acceptance as BF16 within 2.5%, draft 1.7 ms cheaper (DFLASH2_RESULTS.md)
    # Opt-in: flash-decode attention over the ring (-0.42 ms per draft) lowers draft quality (code 4.41 ->
    # 3.61 tok/iter, prose 2.45 -> 2.20; bench/dflash2/e2e_attr2_*.json) and, together with the sharded
    # norms, intermittently HANGS the device (HANG_REPRO.md). Default off.
    decode_attention = os.environ.get("DFLASH_DECODE_ATTN", "0") == "1"
    # Same per-chip shapes as the target's sliding layers, so the target's tuned decode geometry.
    DRAM_GEOMETRY = {"qkv": (8, 2, 21), "gate_up": (56, 2, 3), "o": (8, 2, 8), "down": (8, 2, 21)}
    # Per-shard unary top-k in vocabulary chunks: two 32768-wide top-k save ~0.06 ms over one 65536-wide
    # (large-indices route), but together with sharded norms and decode attention the draft trace HANGS
    # the device (bench/dflash2 stress_norms_dattn_topk2.log; every pair passes). Default: one chunk.
    TOPK_SPLITS = int(os.environ.get("DFLASH_TOPK_SPLITS", "1"))
    # Selector edges on device (one 7x17x16 fp32 readback); the host keeps only the 7-step lookup walk.
    DEVICE_SELECTOR = os.environ.get("DFLASH_DEVICE_SELECTOR", "1") == "1"

    def __init__(self, model, checkpoint=None, **kwargs):
        super().__init__(model, checkpoint, **kwargs)
        sd = self.state_dict
        expand = torch.zeros(SECTION, self.hidden)
        for g in range(self.groups):
            expand[g, g * self.group : (g + 1) * self.group] = 1
        self.expand = self._tensor(expand, dtype=ttnn.bfloat16)
        shift = torch.zeros(1, 1, ROWS, ROWS)
        for r in range(1, self.block):
            shift[0, 0, CTX_ROWS + r, CTX_ROWS + r - 1] = 1
        self.shift = self._tensor(shift, dtype=ttnn.bfloat16)
        self.hproj = self._tensor(sd["candidate_selector.hidden_projection.weight"].T)
        self.rank = sd["candidate_selector.hidden_projection.weight"].shape[0]
        self.pred_cb = sd["candidate_selector.predecessor_codebook"].float()
        self.succ_cb = sd["candidate_selector.successor_codebook"].float()
        self.outputs = None
        if self.DEVICE_SELECTOR:
            rm = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.pred_cb_dev = self._tensor(sd["candidate_selector.predecessor_codebook"], **rm)
            self.succ_cb_dev = self._tensor(sd["candidate_selector.successor_codebook"], **rm)
            self.anchor_ids = self._upload(torch.zeros(1, self.top_k, dtype=torch.int32), ttnn.uint32)
            # Global id offset of every per-shard top-k column: shard d, vocabulary chunk c.
            shard = self.model.config.vocab_size // 4
            cols = torch.arange(4 * self.TOPK_SPLITS * self.top_k) // self.top_k
            offsets = (cols // self.TOPK_SPLITS) * shard + (cols % self.TOPK_SPLITS) * (shard // self.TOPK_SPLITS)
            self.id_offsets = self._tensor(
                offsets.float().reshape(1, 1, 1, -1).expand(1, 1, ROWS, -1), dim=3, dtype=ttnn.float32
            )

    def _configure(self, cfg, block_size):
        tl = cfg["transformer_layer_config"]
        self.hidden = tl["hidden_size"]
        self.heads = tl["num_attention_heads"] // 4
        self.kv_heads = tl["num_key_value_heads"] // 4
        self.head_dim = tl["head_dim"]
        self.eps = tl["rms_norm_eps"]
        self.block = block_size or cfg["block_size"]
        self.mask_token = cfg["mask_token_id"]
        # speculators layer id L is HF hidden_states[L], the residual entering layer L (vLLM
        # extract_hidden_states), i.e. the output of layer L-1; z-lab target_layer_ids are outputs.
        self.tap_layers = [layer - 1 for layer in cfg["aux_hidden_state_layer_ids"]]
        assert min(self.tap_layers) >= 0, "a tap of the embedding output is not supported"
        self.theta = float(tl["rope_parameters"]["rope_theta"])
        self.num_layers = tl["num_hidden_layers"]
        self.local_mlp = tl["intermediate_size"] // 4
        self.window = tl["sliding_window"]
        self.CONTEXT = self.window  # the first draft sees the window's seeded positions
        self.group = cfg["conv_group_size"]
        self.groups = self.hidden // self.group
        self.top_k = cfg["selector_top_k"]
        assert all(t == "sliding_attention" for t in tl["layer_types"]) and cfg["sliding_window_non_causal"]
        assert not cfg["sample_from_anchor"] and cfg["draft_vocab_size"] == tl["vocab_size"]
        assert cfg["conv_kernel_size"] == 2 and self.groups <= SECTION and self.hidden % self.group == 0
        assert self.RING >= self.window + 2 * CTX_ROWS and self.RING // 32 >= ROWS
        assert (self.hidden, self.heads, self.kv_heads, self.head_dim) == (5376, 8, 4, 256)

    def _load_layer(self, sd, i):
        layer = super()._load_layer(sd, i)
        for name in ("attention_conv", "mlp_conv"):
            # kernel_projection rows are (io, tap, group); base_kernel is (io, tap, H).
            w = sd[f"layers.{i}.{name}.kernel_projection.weight"].reshape(4, self.groups, self.hidden)
            wp = torch.zeros(4, SECTION, self.hidden, dtype=w.dtype)
            wp[:, : self.groups] = w
            layer[name + "_kp"] = self._tensor(wp.reshape(4 * SECTION, self.hidden).T)
            base = sd[f"layers.{i}.{name}.base_kernel"].reshape(4, self.hidden)
            layer[name + "_base"] = self._tensor(
                base.repeat_interleave(ROWS, dim=0).reshape(1, 1, 4 * ROWS, self.hidden), dtype=ttnn.bfloat16
            )
        return layer

    def _conv_coefs(self, h, kp, base):
        """Per-channel coefficients (in tap0, in tap1, out tap0, out tap1), each [1,1,32,H]."""
        k = self._linear(h, kp)
        k = ttnn.concat([k[..., s * SECTION : (s + 1) * SECTION] for s in range(4)], dim=2)
        c = ttnn.add(
            ttnn.linear(k, self.expand, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            base,
        )
        return [c[:, :, s * ROWS : (s + 1) * ROWS, :] for s in range(4)]

    def _conv(self, h, c0, c1):
        prev = ttnn.matmul(self.shift, h, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(ttnn.multiply(h, c0), ttnn.multiply(prev, c1))

    def _layer(self, x, ctx, layer, cos, sin):
        h = self._hrms(x, layer["input_norm"])
        ci0, ci1, co0, co1 = self._conv_coefs(h, layer["attention_conv_kp"], layer["attention_conv_base"])
        xin = ttnn.add(ttnn.multiply(self._conv(h, ci0, ci1), self.block_rows), ctx)
        x = ttnn.add(x, self._conv(self._attention(xin, layer, cos, sin), co0, co1))
        h = self._hrms(x, layer["post_norm"])
        ci0, ci1, co0, co1 = self._conv_coefs(h, layer["mlp_conv_kp"], layer["mlp_conv_base"])
        return ttnn.add(x, self._conv(self._mlp(self._conv(h, ci0, ci1), layer), co0, co1))

    def prepare(self, anchor, position, committed_start=None, committed=0):
        super().prepare(anchor, position, committed_start, committed)
        if self.DEVICE_SELECTOR:
            self._refresh(torch.full((1, self.top_k), anchor, dtype=torch.int32), self.anchor_ids)

    def device_step(self, sample=None):
        """Draft trace body: per-shard unary top-k and the selector's hidden projection."""
        h = self.block_hidden()
        logits = self.logits(h)
        width = logits.shape[-1] // self.TOPK_SPLITS
        tops = [
            ttnn.topk(
                logits[..., i * width : (i + 1) * width] if self.TOPK_SPLITS > 1 else logits,
                k=self.top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            for i in range(self.TOPK_SPLITS)
        ]
        vals = ttnn.concat([v for v, _ in tops], dim=-1) if self.TOPK_SPLITS > 1 else tops[0][0]
        idx = ttnn.concat([i for _, i in tops], dim=-1) if self.TOPK_SPLITS > 1 else tops[0][1]
        hp = self._linear(h, self.hproj)
        if not self.DEVICE_SELECTOR:
            self.outputs = (vals, idx, hp)
            return
        self.outputs = self._selector_edges(vals, idx, hp)

    def _selector_edges(self, vals, idx, hp):
        """Global top-k candidates and every selector edge score of the draft positions on device.
        Returns [1, B-1, k+1, k] fp32: rows 0..k-1 = score[pos, predecessor i, candidate j]
        (unary + transition), row k = candidate ids (exact in fp32)."""
        k, rows = self.top_k, slice(CTX_ROWS + 1, CTX_ROWS + self.block)
        gather = self.model.gather_hidden
        ids = ttnn.add(ttnn.typecast(idx, ttnn.float32), self.id_offsets)
        top, pos = ttnn.topk(gather(vals), k=k, dim=-1, largest=True, sorted=True)
        ids = ttnn.gather(gather(ids), dim=-1, index=pos)
        flat = ttnn.reshape(ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, ROWS * k))
        succ = ttnn.reshape(ttnn.embedding(flat, self.succ_cb_dev, layout=ttnn.TILE_LAYOUT), (1, ROWS, k, self.rank))
        pred = ttnn.reshape(ttnn.embedding(flat, self.pred_cb_dev, layout=ttnn.TILE_LAYOUT), (1, ROWS, k, self.rank))
        anchor = ttnn.reshape(ttnn.embedding(self.anchor_ids, self.pred_cb_dev, layout=ttnn.TILE_LAYOUT), (1, 1, k, self.rank))
        # Predecessors: the anchor for the first draft position, else the previous row's candidates.
        pred = ttnn.concat([anchor, pred[:, rows.start : rows.stop - 1]], dim=1)
        hp = ttnn.reshape(hp, (1, ROWS, 1, self.rank))[:, rows]
        context = ttnn.multiply(ttnn.typecast(pred, ttnn.float32), ttnn.typecast(hp, ttnn.float32))
        edges = ttnn.matmul(
            context,
            ttnn.typecast(succ[:, rows], ttnn.float32),
            transpose_b=True,
            compute_kernel_config=self.compute,
            dtype=ttnn.float32,
        )
        unary = ttnn.typecast(ttnn.reshape(top, (1, ROWS, 1, k))[:, rows], ttnn.float32)
        return ttnn.concat([ttnn.add(edges, unary), ttnn.reshape(ids, (1, ROWS, 1, k))[:, rows]], dim=2)

    def read_candidates(self):
        """Host merge of the per-shard top-k -> (unary [B-1, k], candidate ids [B-1, k], hp [B-1, rank])."""
        vals_t, idx_t, hp_t = self.outputs
        rows = slice(CTX_ROWS + 1, CTX_ROWS + self.block)
        composer = ttnn.ConcatMeshToTensor(self.mesh, dim=3)
        vals = ttnn.to_torch(vals_t, mesh_composer=composer).float().reshape(ROWS, -1)[rows]
        ids = ttnn.to_torch(idx_t, mesh_composer=composer)
        ids = (ids.to(torch.int64) & 0xFFFF) if ids.dtype == torch.int16 else ids.to(torch.int64)
        # Columns: shard d, then vocabulary chunk s of that shard, k candidates each, local to the chunk.
        shard = self.model.config.vocab_size // 4
        chunk = torch.arange(4 * self.TOPK_SPLITS).repeat_interleave(self.top_k)
        offsets = (chunk // self.TOPK_SPLITS) * shard + (chunk % self.TOPK_SPLITS) * (shard // self.TOPK_SPLITS)
        ids = ids.reshape(ROWS, -1)[rows] + offsets
        order = vals.sort(dim=-1, descending=True, stable=True).indices[:, : self.top_k]
        hp = ttnn.to_torch(ttnn.get_device_tensors(hp_t)[0]).float().reshape(ROWS, -1)[rows]
        return vals.gather(-1, order), ids.gather(-1, order), hp

    def read_drafts(self, anchor):
        if not self.DEVICE_SELECTOR:
            unary, cand, hp = self.read_candidates()
            return select_path(unary, cand, hp, anchor, self.pred_cb, self.succ_cb)
        t = ttnn.to_torch(ttnn.get_device_tensors(self.outputs)[0]).float()[0]
        score, ids = t[:, : self.top_k].tolist(), t[:, self.top_k].long().tolist()
        # Greedy walk (first maximum wins): the first position's predecessor rows are all the anchor.
        i, path = 0, []
        for pos in range(self.block - 1):
            row = score[pos][i]
            i = max(range(self.top_k), key=row.__getitem__)
            path.append(ids[pos][i])
        return path
