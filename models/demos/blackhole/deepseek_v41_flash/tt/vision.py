# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash image encoder on the 4x8 mesh: ViT (32 blocks, 2D RoPE, full attention inside one image) + 3x3 aligner.

Reference: ``inference/vision.py`` (``ViT``, ``Aligner``) and ``inference/model.py`` (``encode_image`` / ``merge_image_embeddings``) of the checkpoint.

Layout: tensor parallel over the 8 mesh columns (2 of the 16 heads, 1/8 of the MLP and of the aligner per column), replicated over the 4 mesh rows
(every row computes the same image). Activations are bf16 [1,1,n,dim] replicated after each all-reduce. The weights cost ~140 MiB per chip
(~17 MiB per DRAM bank).

``encode(patches, n_h, n_w)`` -> [n_tokens, 5120] bf16 host rows (one per LLM token of the image's IMAGE slots, reading order);
``image_rows(...)`` adds the learned start / newline / end vectors and returns the whole span [h*(w+1)+2, 5120] of ``image_processor.image_token_types``.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

WEIGHT_CACHE = os.environ.get("DSV41_WEIGHT_CACHE", "/mnt/tt-data/ssinghal/dsv4-weight-cache")
TILE = 32
SDPA_CHUNK = 128  # q / k chunk of the flash attention; image lengths are padded to a multiple of it


class VisionArgs:
    """Vision part of the checkpoint ``config.json`` (names of ``inference/model.py`` ``ModelArgs``)."""

    def __init__(self, cfg=None):
        if cfg is None:
            import json

            from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT

            cfg = json.load(open(os.path.join(CKPT, "config.json")))
        v = cfg["vision_config"]
        self.dim = cfg.get("hidden_size", 5120)
        self.n_layers = v["num_hidden_layers"]
        self.vdim = v["hidden_size"]
        self.n_heads = v["num_attention_heads"]
        self.inter = v["intermediate_size"]
        self.patch = v["patch_size"]
        self.theta = float(v.get("rope_theta", 10000.0))
        self.ratio = v["downsample_ratio"]
        self.max_tokens = v["max_image_tokens"]
        self.min_pixels = v["min_pixels"]
        self.max_wh_ratio = v.get("max_wh_ratio")
        self.image_token_id = cfg["image_token_id"]
        self.eps = 1e-6
        self.head_dim = self.vdim // self.n_heads


def _cache(name):
    if not WEIGHT_CACHE or WEIGHT_CACHE == "0":
        return None
    Path(WEIGHT_CACHE).mkdir(parents=True, exist_ok=True)
    return str(Path(WEIGHT_CACHE) / name)


def rope_tables(n_h, n_w, head_dim, theta):
    """2D RoPE of ``vision.get_vision_cos_sin`` -> cos, sin [n, head_dim] in the INTERLEAVED pair order of ``rotary_embedding_llama``
    (pair j of the checkpoint's split-half rotation sits at channels 2j, 2j+1)."""
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq).flatten(1)  # [n, head_dim/2]
    return freqs.cos().repeat_interleave(2, dim=-1), freqs.sin().repeat_interleave(2, dim=-1)


def _interleave_perm(head_dim):
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    return perm


class DSV41Vision:
    def __init__(self, mesh_device, mesh_config, ccl, args=None, shards=None, log=print):
        self.md, self.mc, self.ccl = mesh_device, mesh_config, ccl
        self.a = args or VisionArgs()
        self.rows, self.cols = tuple(mesh_device.shape)
        a = self.a
        assert a.n_heads % self.cols == 0 and a.inter % (self.cols * TILE) == 0 and a.dim % (self.cols * TILE) == 0
        self.hl = a.n_heads // self.cols  # heads per column
        self.il = a.inter // self.cols
        self.log = log
        self.sh = shards or _Shards()
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        self.trans_mat = self._rep(get_rot_transformation_mat(), ttnn.bfloat16)
        self._load()

    # ---- weights ------------------------------------------------------------------------------------------------------------
    def _rep(self, t, dtype, name=None, layout=ttnn.TILE_LAYOUT):
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=layout,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
            cache_file_name=_cache(name) if name else None,
        )

    def _col(self, t, name, dtype=ttnn.bfloat16):
        """t [1,1,K,cols*N] -> per column [1,1,K,N] (replicated over the rows)."""
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(None, 3), mesh_shape=(self.rows, self.cols)),
            cache_file_name=_cache(name),
        )

    def _rowp(self, t, name, dtype=ttnn.bfloat16):
        """t [1,1,cols*K,N] -> per column [1,1,K,N]."""
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(None, 2), mesh_shape=(self.rows, self.cols)),
            cache_file_name=_cache(name),
        )

    def _load(self):
        a, sh, C, hd = self.a, self.sh, self.cols, self.a.head_dim
        tag = f"vision_tp{C}"
        g = lambda k: sh.get(k)
        w = g("vision.patch_embed.proj.weight").float()  # [1024, 588]
        kp = -(-w.shape[1] // TILE) * TILE
        wp = torch.zeros(kp, w.shape[0])
        wp[: w.shape[1]] = w.t()
        self.k_patch = kp
        self.patch_w = self._rep(wp.reshape(1, 1, kp, -1), ttnn.bfloat16, f"{tag}_patch_w")
        self.patch_b = self._rep(
            g("vision.patch_embed.proj.bias").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{tag}_patch_b"
        )
        perm = _interleave_perm(hd)
        self.blocks = []
        for L in range(a.n_layers):
            p = f"vision.blocks.{L}."
            n = f"{tag}_L{L}"
            wqkv = g(p + "attn.wqkv.weight").float()  # [3*vdim, vdim], rows q | k | v, head-major
            bqkv = g(p + "attn.wqkv.bias").float()

            def pack(
                t,
            ):  # [3*vdim, ...] -> [..., C * 3 * hl * hd] with per column (q heads | k heads | v heads); q, k channels interleaved
                parts = list(t.chunk(3, dim=0))
                for i in (0, 1):
                    parts[i] = (
                        parts[i]
                        .reshape(a.n_heads, hd, *t.shape[1:])
                        .index_select(1, perm)
                        .reshape(a.vdim, *t.shape[1:])
                    )
                cols = [
                    torch.cat([pt[c * self.hl * hd : (c + 1) * self.hl * hd] for pt in parts], dim=0) for c in range(C)
                ]
                return torch.cat(cols, dim=0)

            qkv_w = pack(wqkv).t().contiguous()  # [vdim, C*3*hl*hd]
            qkv_b = pack(bqkv)
            w1 = g(p + "mlp.w1.weight").float()  # [2*inter, vdim]: gate | up
            gate_w, up_w = w1[: a.inter], w1[a.inter :]
            blk = {
                "norm1": self._rep(g(p + "norm1.weight").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{n}_norm1"),
                "norm2": self._rep(g(p + "norm2.weight").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{n}_norm2"),
                "wqkv": self._col(qkv_w.reshape(1, 1, *qkv_w.shape), f"{n}_wqkv"),
                "bqkv": self._col(qkv_b.reshape(1, 1, 1, -1), f"{n}_bqkv"),
                "wo": self._rowp(
                    g(p + "attn.wo.weight").float().t().contiguous().reshape(1, 1, a.vdim, a.vdim), f"{n}_wo"
                ),
                "bo": self._rep(g(p + "attn.wo.bias").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{n}_bo"),
                "w_gate": self._col(gate_w.t().contiguous().reshape(1, 1, a.vdim, a.inter), f"{n}_wgate"),
                "w_up": self._col(up_w.t().contiguous().reshape(1, 1, a.vdim, a.inter), f"{n}_wup"),
                "w2": self._rowp(
                    g(p + "mlp.w2.weight").float().t().contiguous().reshape(1, 1, a.inter, a.vdim), f"{n}_w2"
                ),
            }
            self.blocks.append(blk)
        self.norm = self._rep(g("vision.norm.weight").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{tag}_norm")
        r2 = a.vdim * a.ratio**2
        self.al_w1 = self._col(g("aligner.w1.weight").float().t().contiguous().reshape(1, 1, r2, a.dim), f"{tag}_al_w1")
        self.al_b1 = self._col(g("aligner.w1.bias").float().reshape(1, 1, 1, -1), f"{tag}_al_b1")
        self.al_w2 = self._rowp(
            g("aligner.w2.weight").float().t().contiguous().reshape(1, 1, a.dim, a.dim), f"{tag}_al_w2"
        )
        self.al_b2 = self._rep(g("aligner.w2.bias").float().reshape(1, 1, 1, -1), ttnn.bfloat16, f"{tag}_al_b2")
        self.delim = {k: g(k).to(torch.bfloat16) for k in ("image_start", "image_end", "image_newline")}

    # ---- forward ------------------------------------------------------------------------------------------------------------
    def _allreduce(self, x):
        return self.mc.allreduce(x, self.ccl, axis=1)

    def _block(self, x, blk, cos, sin, cu_window, n_pad):
        a = self.a
        h = ttnn.rms_norm(x, weight=blk["norm1"], epsilon=a.eps, compute_kernel_config=self.ckc)
        qkv = ttnn.linear(h, blk["wqkv"], bias=blk["bqkv"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.hl, num_kv_heads=self.hl, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        q = ttnn.experimental.rotary_embedding_llama(
            q, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.sdpa_ckc
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=self.sdpa_ckc
        )
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.md.compute_with_storage_grid_size(),
            q_chunk_size=SDPA_CHUNK,
            k_chunk_size=SDPA_CHUNK,
            exp_approx_mode=False,
        )
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=a.head_dim**-0.5,
            program_config=pc,
            compute_kernel_config=self.sdpa_ckc,
            cu_window_seqlens=cu_window,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,n,hl*hd]
        part = ttnn.linear(o, blk["wo"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(o)
        red = self._allreduce(part)
        att = ttnn.add(red, blk["bo"])
        ttnn.deallocate(red)
        x = ttnn.add(x, att)
        ttnn.deallocate(att)
        h = ttnn.rms_norm(x, weight=blk["norm2"], epsilon=a.eps, compute_kernel_config=self.ckc)
        gate = ttnn.linear(h, blk["w_gate"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        up = ttnn.linear(h, blk["w_up"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        act = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        part = ttnn.linear(act, blk["w2"], compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(act)
        red = self._allreduce(part)
        x = ttnn.add(x, red)
        ttnn.deallocate(red)
        return x

    def _up(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )

    def _host(self, t):
        """Replicated device tensor -> host torch (device 0)."""
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()

    def encode(self, patches, n_h, n_w):
        """patches [n_h*n_w, 3, p, p] (bf16, ``image_processor.load_image``) -> aligner rows [n_llm_h*n_llm_w, dim] bf16 (host)."""
        a = self.a
        n = n_h * n_w
        assert patches.shape[0] == n
        n_pad = -(-n // SDPA_CHUNK) * SDPA_CHUNK
        xin = torch.zeros(1, 1, n_pad, self.k_patch)
        xin[0, 0, :n, : patches[0].numel()] = patches.reshape(n, -1).float()
        cos, sin = rope_tables(n_h, n_w, a.head_dim, a.theta)
        cos_p = torch.ones(n_pad, a.head_dim)
        sin_p = torch.zeros(n_pad, a.head_dim)
        cos_p[:n], sin_p[:n] = cos, sin
        cos_d, sin_d = self._up(cos_p.reshape(1, 1, n_pad, -1)), self._up(sin_p.reshape(1, 1, n_pad, -1))
        windows = [0, n] + (
            [n_pad] if n_pad > n else []
        )  # the zero-padded tail is its own window: it never mixes into the image
        cu = ttnn.from_torch(
            torch.tensor(windows, dtype=torch.int32),
            device=self.md,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )
        x = ttnn.linear(
            self._up(xin), self.patch_w, bias=self.patch_b, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16
        )
        for blk in self.blocks:
            x = self._block(x, blk, cos_d, sin_d, cu, n_pad)
        x = ttnn.rms_norm(x, weight=self.norm, epsilon=a.eps, compute_kernel_config=self.ckc)
        feats = self._host(x)[0, 0, :n].to(torch.bfloat16)  # [n, vdim]
        for t in (x, cos_d, sin_d, cu):
            ttnn.deallocate(t)
        # 3x3 unfold (host, exactly the checkpoint's ``Aligner``), then the aligner MLP on the device
        r = a.ratio
        xx = feats.view(n_h, n_w, -1).permute(2, 0, 1)
        xx = F.pad(xx, (0, -n_w % r, 0, -n_h % r))
        un = F.unfold(xx.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)  # [n_tok, vdim*r*r]
        nt = un.shape[0]
        nt_pad = -(-nt // TILE) * TILE
        u = torch.zeros(1, 1, nt_pad, un.shape[1], dtype=torch.bfloat16)
        u[0, 0, :nt] = un
        ud = self._up(u)
        h1 = ttnn.linear(ud, self.al_w1, bias=self.al_b1, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(ud)
        h1 = ttnn.gelu(h1, fast_and_approximate_mode=False)
        part = ttnn.linear(h1, self.al_w2, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16)
        ttnn.deallocate(h1)
        red = self._allreduce(part)
        out = ttnn.add(red, self.al_b2)
        ttnn.deallocate(red)
        rows = self._host(out)[0, 0, :nt].to(torch.bfloat16)
        ttnn.deallocate(out)
        return rows

    def image_rows(self, patches, n_h, n_w, types):
        """The whole image span [len(types), dim] bf16: aligner rows at the IMAGE slots, the learned vectors at start / newline / end
        (``types`` = ``image_processor.image_token_types``)."""
        from models.demos.blackhole.deepseek_v41_flash.tt.vision_prompt import (
            IMAGE,
            IMAGE_END,
            IMAGE_NEW_LINE,
            IMAGE_START,
        )

        out = torch.empty(types.numel(), self.a.dim, dtype=torch.bfloat16)
        out[types == IMAGE_START] = self.delim["image_start"]
        out[types == IMAGE_END] = self.delim["image_end"]
        out[types == IMAGE_NEW_LINE] = self.delim["image_newline"]
        out[types == IMAGE] = self.encode(patches, n_h, n_w)
        return out
