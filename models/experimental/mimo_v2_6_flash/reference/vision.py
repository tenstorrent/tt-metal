# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fp32 torch reference for the MiMo ViT (``visual.*``), without transformers remote code or vLLM layers.

Mirrors ``MiMoVisionTransformer.forward(pixel_values, grid_thw)`` of the checkpoint's ``modeling_mimo_v2.py`` op for
op (same torch functions in the same order, so fp32 outputs match the HF module bit for bit):

* patch embed: Conv3d 2x16x16 with stride = kernel (a linear map of each 1536-value patch), no bias;
* 2D RoPE on (h, w) patch positions, rotate-half over the 64-dim head (h angles, then w angles);
* 28 blocks ``x += attn(rms_norm1(x)); x += mlp(rms_norm2(x))``, attention with 32 Q / 8 KV heads (head dim 64) and
  a SwiGLU MLP 1280 -> 4608 -> 1280 with biases;
* every temporal grid step of every item (a frame pair of a video, or an image) is its own attention sequence
  (``cu_seqlens``): attention never crosses frames or items;
* blocks 0/9/18/27 (``fullatt_block_indexes``) attend to the whole sequence; the others attend to positions
  ``|i - j| <= visual_token_window_size`` (64) of the current token order, and add a learned per-head sink to the
  logit of the sequence's first key (only where that key is inside the window);
* ``vit_window_attn_types == 1`` blocks (5-8, 14-17, 23-26) run on the column-major token order: the 2x2 merge
  units of each frame are transposed (h, w) -> (w, h) on entering the run and restored on leaving it;
* merger: norm over each patch, 2x2 merge units concatenated to 5120, Linear 5120 -> 5120, GELU, Linear -> 4096.

HF vs vLLM (vLLM 0.31.0 ``mimo_v2_omni.py``, what serves the model on GPU): the merger norm differs. HF builds
``nn.LayerNorm(1280)`` (mean-centred; its bias, absent from the checkpoint, initialises to 0); vLLM builds
``RMSNorm(1280)``. The checkpoint holds only ``visual.merger.ln_q.weight``, as an RMSNorm would. We follow vLLM
(``merger_norm="rms"``, the default); ``merger_norm="layernorm"`` reproduces HF. The merger linears have no bias in
the checkpoint: vLLM builds them without bias, HF with a bias that from_pretrained zero-initialises (same result).
Window and sink semantics agree: vLLM's Triton kernel masks ``|i - j| > 64`` and adds the sink to key 0's logit only
inside the window (``sinks_bias_key0``), exactly HF's dense mask.

Long sequences: HF materialises a [1, heads, L, L] fp32 bias per sequence (32 GB at L = 16384). With
``query_block`` set (default 4096), sequences longer than it run their queries in blocks against all keys, with the
same mask rows; that changes fp32 rounding (not bit-identical to HF), never the math.
"""

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

VISION_PREFIX = "visual."
QUERY_BLOCK = 4096


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    in_channels: int
    spatial_merge_size: int
    out_hidden_size: int
    fullatt_block_indexes: tuple
    window_attn_types: tuple
    window_size: int
    use_sink: bool
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_hf_config(cls, hf_config):
        """``hf_config``: the checkpoint's config.json dict (or its ``vision_config``), parsed like HF's modeling."""
        vc = hf_config.get("vision_config", hf_config)
        depth = vc["depth"]
        return cls(
            depth=depth,
            hidden_size=vc["hidden_size"],
            num_heads=vc["num_heads"],
            num_kv_heads=vc.get("num_key_value_heads", vc["num_heads"]),
            head_dim=vc.get("qk_channels", 64),
            intermediate_size=vc["intermediate_size"],
            patch_size=vc["patch_size"],
            temporal_patch_size=vc["temporal_patch_size"],
            in_channels=vc.get("in_channels") or vc.get("in_chans", 3),
            spatial_merge_size=vc.get("spatial_merge_size", 2),
            out_hidden_size=vc["out_hidden_size"],
            fullatt_block_indexes=tuple(vc.get("fullatt_block_indexes", [])),
            window_attn_types=tuple(vc.get("vit_window_attn_types") or [-1] * depth),
            window_size=vc.get("visual_token_window_size", -1),
            use_sink=bool(vc.get("use_sink", False)),
            rms_norm_eps=vc.get("rms_norm_eps", 1e-6),
        )

    @classmethod
    def from_model_path(cls, model_path):
        with open(os.path.join(model_path, "config.json")) as f:
            return cls.from_hf_config(json.load(f))

    @property
    def patch_dim(self):
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def merge_unit(self):
        return self.spatial_merge_size**2

    @property
    def merged_dim(self):
        return self.hidden_size * self.merge_unit

    def is_full(self, i):
        return i in self.fullatt_block_indexes

    def has_sink(self, i):
        return self.use_sink and not self.is_full(i)

    def is_col(self, i):
        return self.window_attn_types[i] == 1

    def order_switches(self, i):
        """(enter_col, leave_col) before block ``i``, as HF's forward decides them."""
        types = self.window_attn_types
        enter = types[i] == 1 and (i == 0 or types[i - 1] != 1)
        leave = i > 0 and types[i] != 1 and types[i - 1] == 1
        return enter, leave


def load_vision_state_dict(model_path, dtype=torch.float32):
    """``visual.*`` tensors of the checkpoint, keys without the ``visual.`` prefix, cast to ``dtype``."""
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    names = [n for n in weight_map if n.startswith(VISION_PREFIX)]
    by_file = {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    out = {}
    for filename, file_names in by_file.items():
        with safe_open(os.path.join(model_path, filename), framework="pt") as f:
            for n in file_names:
                out[n[len(VISION_PREFIX) :]] = f.get_tensor(n).to(dtype)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Geometry (host side; the TTNN implementation uses the same helpers)
# ---------------------------------------------------------------------------------------------------------------


def as_grid_list(grid_thw):
    """[n, 3] tensor / nested list -> list of (t, h, w) python ints."""
    if isinstance(grid_thw, torch.Tensor):
        grid_thw = grid_thw.reshape(-1, 3).tolist()
    return [tuple(int(v) for v in g) for g in grid_thw]


def cu_seqlens(grid_thw):
    """[0, h*w, 2*h*w, ...]: one sequence per temporal grid step of every item (int32), as HF / vLLM."""
    grid = torch.tensor(as_grid_list(grid_thw), dtype=torch.int64)
    seqlens = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0])
    return F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def rotary_freqs(grid_thw, head_dim, merge, theta=10000.0):
    """Row-order RoPE angles [N, head_dim / 2]: for each patch its h angles then its w angles (HF ``rot_pos_emb``)."""
    grid = as_grid_list(grid_thw)
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    pos_ids = []
    for t, h, w in grid:
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
        pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid = max(max(h, w) for _, h, w in grid)
    seq = torch.arange(max_grid, dtype=inv_freq.dtype)
    full = torch.outer(seq, inv_freq)
    return full[pos_ids].flatten(1)


def col_window_index(grid_thw, merge):
    """Merge-unit permutation into column-major order per frame (HF ``get_window_index_1d(col=True)``)."""
    out, base = [], 0
    for t, h, w in as_grid_list(grid_thw):
        gh, gw = h // merge, w // merge
        index = torch.arange(t * gh * gw).reshape(t, gh, gw)
        out.append(index.transpose(1, 2).reshape(-1) + base)
        base += t * gh * gw
    return torch.cat(out, dim=0)


def apply_index(x, index, merge_unit):
    """Reorder ``x`` [N, ...] by merge units (HF ``apply_index``)."""
    return x.unflatten(0, (-1, merge_unit))[index].flatten(0, 1)


def patch_row_index(index, merge_unit):
    """Merge-unit permutation -> the equivalent patch-row permutation (``x[rows] == apply_index(x, index)``)."""
    return (index[:, None] * merge_unit + torch.arange(merge_unit)[None, :]).reshape(-1)


def num_merged_tokens(grid_thw, merge):
    return sum(t * h * w for t, h, w in as_grid_list(grid_thw)) // (merge * merge)


# ---------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """HF ``_apply_rotary_pos_emb_vision``: q [L, Hq, D], k [L, Hk, D], cos / sin [L, D]."""
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q), k_embed.to(orig_k)


def window_bias(rows, seq_len, window, dtype):
    """Additive mask rows ``rows`` x all keys: -inf where |i - j| > window (HF ``_build_window_mask``)."""
    row_idx = rows.unsqueeze(1)
    col_idx = torch.arange(seq_len).unsqueeze(0)
    mask = torch.zeros(rows.numel(), seq_len, dtype=dtype)
    return mask.masked_fill((row_idx - col_idx).abs() > window, float("-inf"))


class MiMoVisionReference:
    """fp32 functional MiMo ViT on host tensors. ``state_dict``: ``load_vision_state_dict`` output."""

    def __init__(self, cfg: VisionConfig, state_dict, merger_norm="rms", query_block=QUERY_BLOCK):
        assert merger_norm in ("rms", "layernorm"), merger_norm
        self.cfg = cfg
        self.sd = state_dict
        self.merger_norm = merger_norm
        self.query_block = query_block
        self.scaling = cfg.head_dim**-0.5

    @classmethod
    def from_model_path(cls, model_path, **kwargs):
        return cls(VisionConfig.from_model_path(model_path), load_vision_state_dict(model_path), **kwargs)

    def w(self, name):
        return self.sd[name]

    # -- pieces ------------------------------------------------------------------------------------------------

    def patch_embed(self, pixel_values):
        cfg = self.cfg
        weight = self.w("patch_embed.proj.weight")
        x = pixel_values.to(weight.dtype).view(
            -1, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
        )
        stride = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        return F.conv3d(x, weight, None, stride).view(-1, cfg.hidden_size)

    def attention(self, i, x, seqlens, cos, sin):
        cfg = self.cfg
        p = f"blocks.{i}.attn."
        seq_len = x.shape[0]
        qkv = F.linear(x, self.w(p + "qkv.weight"), self.w(p + "qkv.bias"))
        q_dim = cfg.num_heads * cfg.head_dim
        kv_dim = cfg.num_kv_heads * cfg.head_dim
        q = qkv[:, :q_dim].view(seq_len, cfg.num_heads, cfg.head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].view(seq_len, cfg.num_kv_heads, cfg.head_dim)
        v = qkv[:, q_dim + kv_dim :].view(seq_len, cfg.num_kv_heads, cfg.head_dim)
        q, k = apply_rope(q, k, cos, sin)

        groups = cfg.num_heads // cfg.num_kv_heads
        sinks = self.w(p + "sinks") if cfg.has_sink(i) else None
        full = cfg.is_full(i)
        outputs = []
        for q_c, k_c, v_c in zip(q.split(seqlens), k.split(seqlens), v.split(seqlens)):
            q_c = q_c.unsqueeze(0).transpose(1, 2)
            k_c = k_c.unsqueeze(0).transpose(1, 2)
            v_c = v_c.unsqueeze(0).transpose(1, 2)
            if groups > 1:
                k_c = k_c.repeat_interleave(groups, dim=1)
                v_c = v_c.repeat_interleave(groups, dim=1)
            n = q_c.shape[2]
            block = n if (self.query_block is None or n <= self.query_block) else self.query_block
            parts = []
            for s in range(0, n, block):
                rows = torch.arange(s, min(s + block, n))
                attn_mask = None
                if not full and cfg.window_size > 0:
                    attn_mask = window_bias(rows, n, cfg.window_size, q_c.dtype)
                if sinks is not None:
                    sink_bias = torch.zeros(1, cfg.num_heads, rows.numel(), n, dtype=q_c.dtype)
                    sink_bias[..., 0] = sinks.view(1, cfg.num_heads, 1)
                    attn_mask = sink_bias if attn_mask is None else attn_mask + sink_bias
                q_rows = q_c if block == n else q_c[:, :, s : s + block]
                out = F.scaled_dot_product_attention(q_rows, k_c, v_c, attn_mask=attn_mask, scale=self.scaling)
                parts.append(out)
            attn_out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
            outputs.append(attn_out.squeeze(0).transpose(0, 1))
        attn_output = torch.cat(outputs, dim=0).reshape(seq_len, -1)
        return F.linear(attn_output, self.w(p + "proj.weight"), self.w(p + "proj.bias"))

    def mlp(self, i, x):
        p = f"blocks.{i}.mlp."
        gate = F.linear(x, self.w(p + "gate_proj.weight"), self.w(p + "gate_proj.bias"))
        up = F.linear(x, self.w(p + "up_proj.weight"), self.w(p + "up_proj.bias"))
        return F.linear(F.silu(gate) * up, self.w(p + "down_proj.weight"), self.w(p + "down_proj.bias"))

    def block(self, i, x, seqlens, cos, sin):
        eps = self.cfg.rms_norm_eps
        shape = (self.cfg.hidden_size,)
        h = F.rms_norm(x, shape, self.w(f"blocks.{i}.norm1.weight"), eps)
        x = x + self.attention(i, h, seqlens, cos, sin)
        h = F.rms_norm(x, shape, self.w(f"blocks.{i}.norm2.weight"), eps)
        return x + self.mlp(i, h)

    def merger(self, x):
        cfg = self.cfg
        weight = self.w("merger.ln_q.weight")
        if self.merger_norm == "rms":
            x = F.rms_norm(x, (cfg.hidden_size,), weight, 1e-6)
        else:
            x = F.layer_norm(x, (cfg.hidden_size,), weight, torch.zeros_like(weight), 1e-6)
        x = F.linear(x.view(-1, cfg.merged_dim), self.w("merger.mlp.0.weight"), self.sd.get("merger.mlp.0.bias"))
        x = F.gelu(x)
        return F.linear(x, self.w("merger.mlp.2.weight"), self.sd.get("merger.mlp.2.bias"))

    # -- forward ---------------------------------------------------------------------------------------------------

    def forward(self, pixel_values, grid_thw, return_hidden=False):
        """pixel_values [N, 1536] (processor layout), grid_thw [n, 3] -> [N / 4, out_hidden_size] (fp32).

        ``return_hidden``: also return each block's output [N, hidden], in row (processor) order."""
        cfg = self.cfg
        merge_unit = cfg.merge_unit
        x = self.patch_embed(pixel_values)

        emb = rotary_freqs(grid_thw, cfg.head_dim, cfg.spatial_merge_size)
        emb = torch.cat((emb, emb), dim=-1)
        index = col_window_index(grid_thw, cfg.spatial_merge_size)
        reverse = torch.argsort(index)
        row_cos_sin = (emb.cos(), emb.sin())
        col_emb = apply_index(emb, index, merge_unit)
        col_cos_sin = (col_emb.cos(), col_emb.sin())
        cu = cu_seqlens(grid_thw)
        seqlens = (cu[1:] - cu[:-1]).tolist()

        hidden = []
        col = False
        for i in range(cfg.depth):
            enter, leave = cfg.order_switches(i)
            if enter:
                x = apply_index(x, index, merge_unit)
                col = True
            if leave:
                x = apply_index(x, reverse, merge_unit)
                col = False
            cos, sin = col_cos_sin if cfg.is_col(i) else row_cos_sin
            x = self.block(i, x, seqlens, cos, sin)
            if return_hidden:
                hidden.append(apply_index(x, reverse, merge_unit) if col else x)
        if col:  # vLLM restores row order after a trailing column run; HF's config never ends on one
            x = apply_index(x, reverse, merge_unit)
        out = self.merger(x)
        return (out, hidden) if return_hidden else out

    __call__ = forward
