# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The audio patch encoder (RVQ codes -> LLM embeddings) on one 1x8 row, tensor-parallel (TP=8).

    patch = TtAudioPatchEncoder.from_checkpoint(row_mesh, model_path, cache_dir)
    patch.warm()                                  # every bucket's programs, before the LLM captures its traces
    embeds = patch.embed_audio_codes(codes)       # codes [n, 20] (the front end's audio_codes) -> host bf16 [G, 4096]

The reference is ``reference/audio.AudioPatchEncoder`` (HF ``MiMoAudioEncoder`` after the tokenizer): the 20 code
embeddings of each 25 Hz frame summed, groups of 4 frames through a 6-layer Qwen2 transformer (d 1024, 16 heads,
attention within each group, RoPE theta 640000 on positions 0..3, SwiGLU 4096, RMSNorm), each group's 4 x 1024
outputs concatenated and projected 4096 -> 16384 -> GELU -> 4096: one LLM token per group.

* The code embeddings are summed on the host in bf16, channel after channel, exactly as the bf16 host patch encoder
  does (~2 ms for 30 s of audio): the device pass starts from the host path's input.
* Per chip: 2 of the 16 heads, a 512-wide slice of the MLP and a 2048-wide slice of the projection's hidden layer;
  the attention output projection, the MLP down projection and the projection's second linear end in a bf16
  all-reduce over the row. 59 MB of bf16 weights per chip (472 MB unsharded).
* Residual stream fp32, matmuls HiFi4 with fp32 accumulation, bf16 matmul inputs (as the host path's bf16 weights
  and activations).
* Attention: the groups are independent 4-row sequences, so every 32 rows (8 groups) form one batch entry: [32, 32]
  scores per head from one batched matmul, a block-diagonal mask (0 inside a group, -1e9 across groups), softmax
  and a batched matmul with V. The 1/sqrt(64) scale is folded into the Q weights and bias (exact in bf16).
* Passes: groups are independent, so an item runs in passes of at most ``max_groups`` groups, each padded to one of
  ``GROUP_BUCKETS`` (zero embeddings; their rows are dropped), so the programs a call runs are a fixed set that
  ``warm()`` compiles up front. Every call runs eagerly and leaves no device tensor behind (it can run between the
  LLM's traced decode steps).
"""

import json
import os
import time

import torch
from safetensors import safe_open

import ttnn

from ...reference import audio as ref
from ..common import MeshPlan, as_weight, cache_fill, cache_name, compute_config
from ..decoder import TtRMSNorm
from .host_backend import as_codes

GROUP_BUCKETS = (32, 64, 128, 256, 512, 1024)  # groups per pass (4 rows each): 5.1 s .. 164 s of audio
ROWS_PER_BATCH = 32  # attention batch entry: one tile row = 8 groups of 4
MASK_NEG = -1.0e9


def default_cache_dir(model_path):
    """MIMO_AUDIO_CACHE/<checkpoint name>, or None (convert from the checkpoint at every load, ~2 s)."""
    root = os.environ.get("MIMO_AUDIO_CACHE")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path))) if root else None


def padded_groups(g, buckets=GROUP_BUCKETS):
    """Groups a pass of ``g`` groups runs with: the smallest bucket that holds it (else a multiple of 32)."""
    for b in buckets:
        if b >= g:
            return b
    return -(-g // 32) * 32


def load_code_embeddings(model_path, cfg):
    """The 20 ``speech_embeddings.i.weight`` tables (bf16, as stored) without the rest of the checkpoint."""
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    names = [f"speech_embeddings.{i}.weight" for i in range(cfg.audio_channels)]
    tables = {}
    for fname in sorted({weight_map[n] for n in names}):
        with safe_open(os.path.join(model_path, fname), framework="pt") as f:
            tables.update({n: f.get_tensor(n) for n in names if weight_map[n] == fname})
    return [tables[n].to(torch.bfloat16) for n in names]


class TtAudioPatchLayer:
    def __init__(self, plan, cfg, i, sd, cache_dir, mm_config):
        self.plan, self.cfg, self.mm_config = plan, cfg, mm_config
        tp = plan.tp
        hd = cfg.local_dim // cfg.local_heads
        self.local_heads = cfg.local_heads // tp
        assert self.local_heads * tp == cfg.local_heads and (cfg.local_ffn // tp) % 32 == 0
        name = lambda n: cache_name(cache_dir, f"audio_patch_tp{tp}_l{i}", n)
        p = f"audio_encoder.input_local_transformer.layers.{i}."
        w = {}
        if sd is not None:
            scale = hd**-0.5  # 1/8: exact
            q_w, q_b = sd[p + "self_attn.q_proj.weight"] * scale, sd[p + "self_attn.q_proj.bias"] * scale
            k_w, k_b = sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.k_proj.bias"]
            v_w, v_b = sd[p + "self_attn.v_proj.weight"], sd[p + "self_attn.v_proj.bias"]
            per = self.local_heads * hd
            chips_w, chips_b = [], []
            for c in range(tp):
                s = slice(c * per, (c + 1) * per)
                chips_w += [q_w[s], k_w[s], v_w[s]]
                chips_b += [q_b[s], k_b[s], v_b[s]]
            w["wqkv"] = torch.cat(chips_w).T.contiguous()  # [d, tp * 3 * per]: chip blocks side by side
            w["bqkv"] = torch.cat(chips_b).reshape(1, -1)
            w["wo"] = sd[p + "self_attn.o_proj.weight"].T.contiguous()  # [heads * hd, d]: chip c's rows = its heads
            w["wg"] = sd[p + "mlp.gate_proj.weight"].T.contiguous()
            w["wu"] = sd[p + "mlp.up_proj.weight"].T.contiguous()
            w["wd"] = sd[p + "mlp.down_proj.weight"].T.contiguous()

        def load(key, mapper):
            return as_weight(w.get(key), plan, mapper, ttnn.bfloat16, name(key))

        self.wqkv = load("wqkv", plan.shard(-1))
        self.bqkv = load("bqkv", plan.shard(-1))
        self.wo = load("wo", plan.shard(-2))
        self.wg = load("wg", plan.shard(-1))
        self.wu = load("wu", plan.shard(-1))
        self.wd = load("wd", plan.shard(-2))
        eps = cfg.local_rms_eps
        self.norm1 = TtRMSNorm(plan, sd and sd[p + "input_layernorm.weight"], eps, name("in_norm"))
        self.norm2 = TtRMSNorm(plan, sd and sd[p + "post_attention_layernorm.weight"], eps, name("post_norm"))

    def linear(self, x, w, b=None, dtype=ttnn.bfloat16):
        return ttnn.linear(
            x, w, bias=b, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm_config
        )

    def __call__(self, x, inp):
        h = _norm_bf16(self.norm1, x)
        attn = self.attention(h, inp)
        h.deallocate(True)
        x2 = ttnn.add(x, attn, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        h = _norm_bf16(self.norm2, x2)
        gate = self.linear(h, self.wg)
        up = self.linear(h, self.wu)
        h.deallocate(True)
        act = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate.deallocate(True)
        up.deallocate(True)
        down = self.linear(act, self.wd)
        act.deallocate(True)
        down = self.plan.all_reduce(down)
        out = ttnn.add(x2, down, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x2.deallocate(True)
        down.deallocate(True)
        return out

    def attention(self, h, inp):
        rows = h.shape[-2]
        heads, hd = self.local_heads, self.cfg.local_dim // self.cfg.local_heads
        qkv = self.linear(h, self.wqkv, self.bqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=heads, num_kv_heads=heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        qkv.deallocate(True)
        q_rot = ttnn.experimental.rotary_embedding_hf(q, inp.cos, inp.sin, compute_kernel_config=inp.attn_config)
        k_rot = ttnn.experimental.rotary_embedding_hf(k, inp.cos, inp.sin, compute_kernel_config=inp.attn_config)
        q.deallocate(True)
        k.deallocate(True)
        nb = rows // ROWS_PER_BATCH
        qb = ttnn.reshape(q_rot, (heads, nb, ROWS_PER_BATCH, hd))  # the same tiles: 32-row blocks as a batch dim
        kb = ttnn.reshape(k_rot, (heads, nb, ROWS_PER_BATCH, hd))
        vb = ttnn.reshape(v, (heads, nb, ROWS_PER_BATCH, hd))
        scores = ttnn.matmul(
            qb,
            kb,
            transpose_b=True,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=inp.attn_config,
        )
        for t in (q_rot, k_rot):
            t.deallocate(True)
        masked = ttnn.add(scores, inp.mask, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores.deallocate(True)
        probs = ttnn.softmax(
            masked,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=inp.attn_config,
        )
        masked.deallocate(True)
        probs16 = ttnn.typecast(probs, ttnn.bfloat16)
        probs.deallocate(True)
        ob = ttnn.matmul(probs16, vb, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=inp.attn_config)
        probs16.deallocate(True)
        v.deallocate(True)
        o = ttnn.reshape(ob, (1, heads, rows, hd))
        cat = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ob.deallocate(True)
        out = self.linear(cat, self.wo)
        cat.deallocate(True)
        return self.plan.all_reduce(out)


def _norm_bf16(norm, x):
    h = norm(x, prefill=True)
    if h.dtype == ttnn.bfloat16:
        return h
    out = ttnn.typecast(h, ttnn.bfloat16)
    h.deallocate(True)
    return out


class PassInputs:
    """Per-bucket device constants (replicated): RoPE tables for positions 0..3 repeated, the block-diagonal mask."""

    def __init__(self, plan, cfg, rows, attn_config):
        mesh, rep = plan.mesh_device, plan.replicate()
        hd = cfg.local_dim // cfg.local_heads
        inv_freq = 1.0 / (cfg.local_rope_theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
        pos = torch.arange(rows) % cfg.group_size
        freqs = pos[:, None].float() * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # [rows, hd]

        def dev(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

        # The host bf16 path rounds cos / sin to bf16 too (reference: emb.cos().to(x.dtype)).
        self.cos = dev(emb.cos().reshape(1, 1, rows, hd))
        self.sin = dev(emb.sin().reshape(1, 1, rows, hd))
        g = torch.arange(ROWS_PER_BATCH) // cfg.group_size
        mask = torch.zeros(ROWS_PER_BATCH, ROWS_PER_BATCH).masked_fill_(g[:, None] != g[None, :], MASK_NEG)
        self.mask = dev(mask.reshape(1, 1, ROWS_PER_BATCH, ROWS_PER_BATCH))
        self.attn_config = attn_config

    def deallocate(self):
        for t in (self.cos, self.sin, self.mask):
            t.deallocate(True)


class TtAudioPatchEncoder:
    """``speech_embeddings`` + ``audio_encoder`` on one 1x8 mesh (or submesh). ``state_dict``: the patch half of
    ``reference.audio.load_audio_weights`` (or None to load from a warm ``cache_dir``)."""

    def __init__(self, mesh_device, cfg, state_dict=None, cache_dir=None, max_groups=None, buckets=GROUP_BUCKETS):
        assert tuple(mesh_device.shape)[0] == 1, f"one row (1 x TP), got {tuple(mesh_device.shape)}"
        assert ROWS_PER_BATCH % cfg.group_size == 0 and all(b % 32 == 0 for b in buckets), (cfg.group_size, buckets)
        self.plan = MeshPlan(mesh_device)
        self.cfg = cfg
        self.buckets = tuple(sorted(buckets))
        self.max_groups = max_groups or self.buckets[-1]
        tp = self.plan.tp
        sd = state_dict
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=True)
        self.attn_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)
        name = lambda n: cache_name(cache_dir, f"audio_patch_tp{tp}", n)
        w = {}
        if sd is not None:
            w["proj0"] = sd["audio_encoder.projection.mlp.0.weight"].T.contiguous()  # [4 d, 4 * out]
            w["proj2"] = sd["audio_encoder.projection.mlp.2.weight"].T.contiguous()  # [4 * out, out]
        load = lambda key, mapper: as_weight(w.get(key), self.plan, mapper, ttnn.bfloat16, name(key))
        self.proj0 = load("proj0", self.plan.shard(-1))
        self.proj2 = load("proj2", self.plan.shard(-2))
        p = "audio_encoder.input_local_transformer."
        self.norm = None
        if cfg.add_post_norm:
            self.norm = TtRMSNorm(self.plan, sd and sd[p + "norm.weight"], cfg.local_rms_eps, name("norm"))
        self.layers = [
            TtAudioPatchLayer(self.plan, cfg, i, sd, cache_dir, self.mm_config) for i in range(cfg.local_layers)
        ]
        # The code embedding tables stay on the host (bf16), summed there as the host path sums them.
        self.embeddings = None
        if sd is not None:
            self.embeddings = [
                sd[f"speech_embeddings.{i}.weight"].to(torch.bfloat16) for i in range(cfg.audio_channels)
            ]
        self._inputs = {}

    @classmethod
    def from_checkpoint(cls, mesh_device, model_path, cache_dir=None, **kwargs):
        """Converts the device weights once per ``cache_dir`` (``cache_fill`` lock, as the LLM's weights); the code
        embedding tables (4.6M parameters, host) are read from the checkpoint."""
        cfg = ref.MiMoAudioConfig.from_model_path(model_path)
        tp = tuple(mesh_device.shape)[1]
        with cache_fill(cache_dir, f"audio_patch_tp{tp}") as need:
            sd = ref.load_audio_weights(model_path, tokenizer=False) if need else None
            enc = cls(mesh_device, cfg, sd, cache_dir, **kwargs)
        if enc.embeddings is None:
            enc.embeddings = load_code_embeddings(model_path, cfg)
        return enc

    def _pass_inputs(self, rows):
        if rows not in self._inputs:
            self._inputs[rows] = PassInputs(self.plan, self.cfg, rows, self.attn_config)
        return self._inputs[rows]

    def embed_codes(self, grouped):
        """grouped codes [G, 4, 20] -> bf16 [G * 4, d]: ``reference.audio.AudioPatchEncoder.embed_codes`` in bf16."""
        out = torch.zeros((grouped.shape[0] * self.cfg.group_size, self.cfg.local_dim), dtype=torch.bfloat16)
        flat = grouped.reshape(-1, self.cfg.audio_channels).long()
        for i, table in enumerate(self.embeddings):
            out.add_(torch.nn.functional.embedding(flat[:, i], table))
        return out

    def forward_pass(self, x_host, groups):
        """bf16 [rows = 4 * groups_padded, d] summed code embeddings (padding rows zero) -> host bf16 [groups, out]."""
        cfg = self.cfg
        rows = x_host.shape[0]
        inp = self._pass_inputs(rows)
        x = ttnn.from_torch(
            x_host.reshape(1, 1, rows, -1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            mesh_mapper=self.plan.replicate(),
        )
        for layer in self.layers:
            x_new = layer(x, inp)
            x.deallocate(True)
            x = x_new
        h = _norm_bf16(self.norm, x) if self.norm is not None else ttnn.typecast(x, ttnn.bfloat16)
        x.deallocate(True)
        # [rows, d] -> [groups, group_size * d]: each group's 4 rows side by side (a row-major view).
        rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h.deallocate(True)
        grouped_rm = ttnn.reshape(rm, (1, 1, rows // cfg.group_size, cfg.group_size * cfg.local_dim))
        grouped = ttnn.to_layout(grouped_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rm.deallocate(True)
        kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm_config)
        h0 = ttnn.linear(grouped, self.proj0, **kw)
        grouped.deallocate(True)
        g = ttnn.gelu(h0, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h0.deallocate(True)
        out = ttnn.linear(g, self.proj2, **kw)
        g.deallocate(True)
        out = self.plan.all_reduce(out)
        result = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(rows // cfg.group_size, -1)
        out.deallocate(True)
        return result[:groups].to(torch.bfloat16)

    @torch.no_grad()
    def __call__(self, grouped):
        """grouped codes [G, 4, 20] (``reference.audio.pad_and_group_codes``) -> host bf16 [G, out_hidden_size]."""
        outs = []
        for start in range(0, grouped.shape[0], self.max_groups):
            part = grouped[start : start + self.max_groups]
            g = part.shape[0]
            x = self.embed_codes(part)
            pad = padded_groups(g, self.buckets) * self.cfg.group_size - x.shape[0]
            if pad:
                x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype)])
            outs.append(self.forward_pass(x, g))
        return torch.cat(outs, dim=0)

    def embed_audio_codes(self, codes):
        """[n_codes, 20] RVQ codes (the TT processor's ``audio_codes``) -> host bf16 [ceil(n_codes / 4), 4096]."""
        return self(ref.pad_and_group_codes(as_codes(codes, self.cfg), self.cfg))

    def warm(self, buckets=None):
        """Compile every bucket's programs (default: all), one zero-code pass each. Returns {groups: seconds}. Run it
        before the LLM captures its traces: compiling later allocates program binaries in DRAM."""
        times = {}
        for g in buckets if buckets is not None else self.buckets:
            t0 = time.time()
            self.forward_pass(torch.zeros(g * self.cfg.group_size, self.cfg.local_dim, dtype=torch.bfloat16), g)
            times[g] = time.time() - t0
        return times
