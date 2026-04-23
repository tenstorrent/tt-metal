# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Vision Attention for Dots OCR.

Runs QKV, RoPE, attention, and out-projection fully in TTNN.
"""

from __future__ import annotations

import math

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs


def _rotate_half(x):
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required")
    last = x.shape[-1]
    half = last // 2
    x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], half))
    x2 = ttnn.slice(x, (0, 0, 0, half), (x.shape[0], x.shape[1], x.shape[2], last))
    neg_x2 = ttnn.mul(x2, -1, use_legacy=False)
    return ttnn.concat([neg_x2, x1], dim=-1)


def _apply_rotary_tt(q, k, cos, sin, *, out_dtype):
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required")
    f32 = getattr(ttnn, "float32", None)
    # HF `apply_rotary_pos_emb_vision` upcasts to float32, applies, then casts back to activations
    # (see remote `modeling_dots_vision.apply_rotary_pos_emb_vision`). Do the same in TTNN so Q/K
    # fed into attention are not computed in bfloat16 RoPE math.
    if f32 is not None:
        qf = ttnn.typecast(q, dtype=f32)
        kf = ttnn.typecast(k, dtype=f32)
        cos_f = ttnn.typecast(cos, dtype=f32)
        sin_f = ttnn.typecast(sin, dtype=f32)
    else:
        qf, kf, cos_f, sin_f = q, k, cos, sin

    # q,k: [B, H, S, D], cos/sin broadcastable to [B, H, S, D] (e.g. [1,1,S,D]).
    # Match the HF rotate-half formulation used by the reference vision tower.
    q_embed = ttnn.add(
        ttnn.mul(qf, cos_f, use_legacy=False),
        ttnn.mul(_rotate_half(qf), sin_f, use_legacy=False),
    )
    k_embed = ttnn.add(
        ttnn.mul(kf, cos_f, use_legacy=False),
        ttnn.mul(_rotate_half(kf), sin_f, use_legacy=False),
    )
    if f32 is not None and out_dtype is not None:
        q_embed = ttnn.typecast(q_embed, dtype=out_dtype)
        k_embed = ttnn.typecast(k_embed, dtype=out_dtype)
    return q_embed, k_embed


def _pk(state_dict: dict, *cands: str) -> str | None:
    for c in cands:
        if c in state_dict:
            return c
    return None


def _find_key_contains(sd: dict, *, contains_all: tuple[str, ...], endswith: str | None = None) -> str | None:
    for k in sd.keys():
        if endswith is not None and not k.endswith(endswith):
            continue
        ok = True
        for s in contains_all:
            if s not in k:
                ok = False
                break
        if ok:
            return k
    return None


def _tt_to_torch_single_replica(mesh_device, x_ttnn):
    """
    Convert a potentially mesh-distributed tensor to torch by gathering, then
    selecting the first replica shard (for replicated tensors).
    """
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required")
    composer = None
    if mesh_device is not None and hasattr(ttnn, "ConcatMeshToTensor"):
        composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    out = ttnn.to_torch(x_ttnn, mesh_composer=composer) if composer is not None else ttnn.to_torch(x_ttnn)
    try:
        num = mesh_device.get_num_devices() if mesh_device is not None else 1
        if num > 1 and out.shape[0] % num == 0:
            per = out.shape[0] // num
            return out[:per]
    except Exception:
        pass
    return out


class VisionAttentionTT(LightweightModule):
    def __init__(
        self,
        mesh_device,
        model_args: DotsVisionModelArgs,
        state_dict: dict,
        layer_num: int,
        weight_cache_path=None,
        dtype=None,
    ):
        super().__init__()
        ttnn = get_ttnn()
        if dtype is None:
            dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16
        self.mesh_device = mesh_device
        self.model_args = model_args
        self.layer_num = layer_num
        self.dtype = dtype
        self.hidden_size = model_args.vision_dim
        self.num_heads = model_args.vision_n_heads
        self.head_dim = model_args.vision_head_dim
        self.num_kv_heads = model_args.vision_n_kv_heads
        self._load_weights(state_dict, weight_cache_path, dtype)

    def _as_tt(self, w, name, weight_cache_path, dtype, ttnn, mc, mesh, layer: int):
        if ttnn is None or mesh is None:
            return w.clone() if hasattr(w, "clone") else w
        # TTNN `linear` expects weights shaped [in_features, out_features] so K matches input width.
        # HF state_dict weights are typically [out_features, in_features].
        if hasattr(w, "dim") and callable(w.dim) and w.dim() == 2:
            w = torch.transpose(w, -2, -1).contiguous()
        return ttnn.as_tensor(
            w,
            device=mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mc,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if hasattr(ttnn, "ReplicateTensorToMesh") else None,
            cache_file_name=(weight_cache_path / f"layer_{layer}_{name}" if weight_cache_path else None),
        )

    def _load_weights(self, state_dict: dict, weight_cache_path, dtype):
        base_prefix = self.model_args.get_state_dict_prefix("VisionAttention", self.layer_num)
        # HF checkpoints have varied naming for the attention module: attention / attn / self_attn.
        prefixes = []
        if base_prefix:
            prefixes.append(base_prefix)
            prefixes.append(base_prefix.replace("attention", "attn"))
            prefixes.append(base_prefix.replace("attention", "self_attn"))
        else:
            prefixes.append("")
        # normalize to trailing dot for key construction
        prefixes = [p + "." if p and not p.endswith(".") else p for p in prefixes]
        ttnn = get_ttnn()
        mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None) if ttnn is not None else None
        mesh = self.mesh_device

        # Find a prefix that exists in the state_dict.
        # Some checkpoints use slightly different layouts; as a backstop, fall back to substring search
        # for this block/layer.
        chosen = None
        for prefix in prefixes:
            k_qkv = _pk(state_dict, f"{prefix}qkv.weight", f"{prefix}qkv_proj.weight")
            if k_qkv is not None:
                chosen = prefix
                break
            # unfused path: require at least q_proj.weight to exist
            if _pk(state_dict, f"{prefix}wq.weight", f"{prefix}q_proj.weight") is not None:
                chosen = prefix
                break
        if chosen is None:
            # Backstop: search for keys containing the layer id + qkv/o_proj.
            layer_tags = (
                f"blocks.{self.layer_num}.",
                f"layers.{self.layer_num}.",
                f"layer.{self.layer_num}.",
            )
            k_qkv = None
            for layer_tag in layer_tags:
                k_qkv = _find_key_contains(state_dict, contains_all=(layer_tag, "qkv"), endswith="weight")
                if k_qkv is not None:
                    break
            if k_qkv is not None:
                # Use whatever prefix precedes "qkv" in the found key.
                chosen = k_qkv.split("qkv")[0]
            else:
                k_qproj = None
                for layer_tag in layer_tags:
                    k_qproj = _find_key_contains(state_dict, contains_all=(layer_tag, "q_proj"), endswith="weight")
                    if k_qproj is not None:
                        break
                if k_qproj is None:
                    # Some checkpoints use "query_key_value" naming.
                    for layer_tag in layer_tags:
                        k_qproj = _find_key_contains(state_dict, contains_all=(layer_tag, "query"), endswith="weight")
                        if k_qproj is not None:
                            break
                if k_qproj is not None:
                    if "q_proj" in k_qproj:
                        chosen = k_qproj.split("q_proj")[0]
                    elif "query" in k_qproj:
                        chosen = k_qproj.split("query")[0]

        if chosen is None:
            self.qkv_weight = None
            self.qkv_bias = None
            self.o_proj_weight = None
            self.o_proj_bias = None
            return
        prefix = chosen

        k_qkv = _pk(
            state_dict,
            f"{prefix}qkv.weight",
            f"{prefix}qkv_proj.weight",
            f"{prefix}qkv_proj._linear.weight",
            f"{prefix}query_key_value.weight",
            f"{prefix}query_key_value._linear.weight",
        )
        k_b = _pk(state_dict, f"{prefix}qkv.bias", f"{prefix}qkv_proj.bias", f"{prefix}qkv_proj._linear.bias")
        wq, wk, wv = (
            _pk(state_dict, f"{prefix}wq.weight", f"{prefix}q_proj.weight", f"{prefix}query.weight"),
            _pk(state_dict, f"{prefix}wk.weight", f"{prefix}k_proj.weight", f"{prefix}key.weight"),
            _pk(state_dict, f"{prefix}wv.weight", f"{prefix}v_proj.weight", f"{prefix}value.weight"),
        )
        bq, bk, bv = (
            _pk(state_dict, f"{prefix}wq.bias", f"{prefix}q_proj.bias"),
            _pk(state_dict, f"{prefix}wk.bias", f"{prefix}k_proj.bias"),
            _pk(state_dict, f"{prefix}wv.bias", f"{prefix}v_proj.bias"),
        )
        if k_qkv is not None:
            w_fused = state_dict[k_qkv]
            b_fused = state_dict[k_b] if k_b else None
        elif wq and wk and wv:
            w_fused = torch.cat((state_dict[wq], state_dict[wk], state_dict[wv]), dim=0)
            if bq and bk and bv:
                b_fused = torch.cat((state_dict[bq], state_dict[bk], state_dict[bv]), dim=0)
            else:
                b_fused = None
        else:
            w_fused = None
            b_fused = None
        if w_fused is not None:
            self.qkv_weight = self._as_tt(w_fused, "qkv_w", weight_cache_path, dtype, ttnn, mc, mesh, self.layer_num)
        else:
            self.qkv_weight = None
        if b_fused is not None:
            self.qkv_bias = self._as_tt(b_fused, "qkv_b", weight_cache_path, dtype, ttnn, mc, mesh, self.layer_num)
        else:
            self.qkv_bias = None

        o_w = _pk(
            state_dict,
            f"{prefix}wo.weight",
            f"{prefix}o_proj.weight",
            f"{prefix}out_proj.weight",
            f"{prefix}o_proj._linear.weight",
            f"{prefix}proj.weight",
        )
        o_b = _pk(
            state_dict,
            f"{prefix}wo.bias",
            f"{prefix}o_proj.bias",
            f"{prefix}out_proj.bias",
            f"{prefix}o_proj._linear.bias",
        )
        self.o_proj_weight = (
            self._as_tt(state_dict[o_w], "o_w", weight_cache_path, dtype, ttnn, mc, mesh, self.layer_num)
            if o_w
            else None
        )
        self.o_proj_bias = (
            self._as_tt(state_dict[o_b], "o_b", weight_cache_path, dtype, ttnn, mc, mesh, self.layer_num)
            if o_b
            else None
        )

    def forward(
        self,
        x,
        rot_mats: tuple[object, object] | None = None,
        **kwargs,
    ):
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("VisionAttentionTT requires ttnn")
        if self.qkv_weight is None or self.o_proj_weight is None:
            raise ValueError("VisionAttentionTT weights not loaded")

        # Expect x as ttnn.Tensor with shape [1, 1, S, D] (concatenated tokens across images/chunks).
        if not isinstance(x, ttnn.Tensor):
            raise TypeError(f"Expected ttnn.Tensor, got {type(x)}")
        if len(x.shape) != 4:
            raise ValueError(f"Expected x rank-4 [B,1,S,D], got shape={x.shape}")
        b, one, s, d = x.shape
        if int(b) != 1 or int(one) != 1:
            raise ValueError(f"Only B=1 supported, got {x.shape}")

        qkv = ttnn.linear(
            x, self.qkv_weight, bias=self.qkv_bias, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        )
        h = int(self.num_heads)
        hd = int(self.head_dim)
        qkv = ttnn.reshape(qkv, (1, 1, int(s), 3, h, hd))  # [1,1,S,3,H,hd]
        q = ttnn.slice(qkv, (0, 0, 0, 0, 0, 0), (1, 1, int(s), 1, h, hd))
        k = ttnn.slice(qkv, (0, 0, 0, 1, 0, 0), (1, 1, int(s), 2, h, hd))
        v = ttnn.slice(qkv, (0, 0, 0, 2, 0, 0), (1, 1, int(s), 3, h, hd))
        q = ttnn.reshape(q, (1, h, int(s), hd))
        k = ttnn.reshape(k, (1, h, int(s), hd))
        v = ttnn.reshape(v, (1, h, int(s), hd))

        if rot_mats is not None and len(rot_mats) == 2:
            cos, sin = rot_mats
            if not isinstance(cos, ttnn.Tensor) or not isinstance(sin, ttnn.Tensor):
                raise TypeError("rot_mats must be ttnn tensors")
            # Match HF: RoPE in fp32, activations (bf16) after rotation unless caller promotes later.
            q, k = _apply_rotary_tt(q, k, cos, sin, out_dtype=self.dtype)

        # Block-diagonal attention using cu_seqlens to prevent cross-image attention.
        cu = kwargs.get("cu_seqlens")
        if cu is None:
            # Single segment fallback.
            ctx = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
            ctx = ttnn.reshape(ctx, (1, 1, int(s), h * hd))
            return ttnn.linear(
                ctx, self.o_proj_weight, bias=self.o_proj_bias, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            )

        # For correctness/PCC, use the explicit per-segment SDPA path below.
        # The windowed/varlen SDPA kernels can have different numerics and have historically
        # produced low PCC vs HF for Dots vision.
        #
        # Additionally, even the fused `ttnn.transformer.scaled_dot_product_attention` can diverge
        # from HF eager attention. Match HF ``VisionAttention`` (eager) exactly:
        #   - q, k, v stay in activation dtype (bf16) for both matmuls
        #   - ``attn_weights = (q @ k^T) / sqrt(d)`` in bf16 (same as PyTorch on bf16 inputs)
        #   - ``softmax(..., dtype=float32)`` on those logits, then ``.to(q.dtype)`` for probs
        #   - ``ctx = attn_probs @ v`` in bf16
        # Promoting q/k/v to float32 for the whole pipeline (as we did earlier) does *not* match
        # eager and tanks PCC vs the HF reference.

        if isinstance(cu, ttnn.Tensor):
            cu_host = _tt_to_torch_single_replica(self.mesh_device, cu).flatten().to(torch.int64).tolist()
        elif isinstance(cu, torch.Tensor):
            cu_host = cu.flatten().to(torch.int64).tolist()
        else:
            cu_host = list(cu)
        if len(cu_host) < 2 or cu_host[0] != 0 or cu_host[-1] != int(s):
            raise ValueError(f"Invalid cu_seqlens={cu_host} for S={int(s)}")

        ctx_segments = []
        scale = 1.0 / math.sqrt(float(hd))
        f32 = getattr(ttnn, "float32", None)

        # Use default ttnn matmul configuration for bf16@bf16 attention. Tuning
        # ``WormholeComputeKernelConfig`` (e.g. fp32_dest accum) is aimed mainly at f32
        # matmuls; forcing it for bf16 Q@K^T and attn@V can diverge from CPU eager PyTorch
        # and depress ``Block0 attn`` PCC.
        def _matmul(a, b):
            return ttnn.matmul(
                a,
                b,
                memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
            )

        for a, b0 in zip(cu_host[:-1], cu_host[1:]):
            a = int(a)
            b0 = int(b0)
            seg = b0 - a
            if seg <= 0:
                continue
            q_seg = ttnn.slice(q, (0, 0, a, 0), (1, h, b0, hd))
            k_seg = ttnn.slice(k, (0, 0, a, 0), (1, h, b0, hd))
            v_seg = ttnn.slice(v, (0, 0, a, 0), (1, h, b0, hd))
            # scores: [1, H, seg, seg] in activation dtype (match PyTorch bf16 @ bf16 -> bf16)
            kt = ttnn.transpose(k_seg, -2, -1)
            scores = _matmul(q_seg, kt)
            scores = ttnn.mul(scores, scale, use_legacy=False)
            # Eager HF: F.softmax(logits, dim=-1, dtype=float32) then .to(q.dtype) before @ v
            if f32 is not None:
                scores_f = ttnn.typecast(scores, dtype=f32)
                attn_f = ttnn.softmax(scores_f, dim=-1, numeric_stable=True)
                attn = ttnn.typecast(attn_f, dtype=self.dtype)
            else:
                attn = ttnn.softmax(scores, dim=-1, numeric_stable=True)
            ctx_seg = _matmul(attn, v_seg)
            ctx_segments.append(ctx_seg)

        ctx = ttnn.concat(ctx_segments, dim=2) if len(ctx_segments) > 1 else ctx_segments[0]
        ctx = ttnn.reshape(ctx, (1, 1, int(s), h * hd))
        return ttnn.linear(
            ctx, self.o_proj_weight, bias=self.o_proj_bias, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        )


def create_vision_attention(mesh_device, model_args, state_dict, layer_num, weight_cache_path=None, dtype=None):
    return VisionAttentionTT(
        mesh_device=mesh_device,
        model_args=model_args,
        state_dict=state_dict,
        layer_num=layer_num,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
    )
