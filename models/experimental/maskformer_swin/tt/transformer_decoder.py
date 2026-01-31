# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Transformer decoder for MaskFormer queries.

Implements the DETR-style decoder stack using TTNN ops:
  - Learned query embeddings (100 queries, dim=256)
  - 2D sine/cos positional embeddings for encoder tokens
  - Per-layer: self-attention, cross-attention, FFN, LayerNorms

Weights are loaded from the HuggingFace-style state dict under:
  - model.transformer_module.input_projection.*
  - model.transformer_module.queries_embedder.weight
  - model.transformer_module.decoder.*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import os

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from models.common.utility_functions import is_blackhole
except ModuleNotFoundError:  # pragma: no cover

    def is_blackhole() -> bool:
        return False


from .backbone_swin import DEFAULT_TT_DTYPE
from .tt_configs import build_decoder_program_configs
from .ttnn_compat import ttnn, require_ttnn
from .weights import extract_transformer_state


@dataclass
class TransformerDecoderConfig:
    """Configuration for the DETR-style decoder."""

    num_layers: int = 6
    num_attention_heads: int = 8
    hidden_dim: int = 256
    dim_feedforward: int = 2048
    dropout: float = 0.0
    activation: str = "relu"
    in_features: int = 1024
    # Kept for backward compatibility with older runner paths (unused)
    maskformer_config: Optional[Dict[str, object]] = None


class MaskFormerTransformerDecoder:
    """Runs query refinement via TTNN attention and MLP blocks."""

    def __init__(
        self,
        config: TransformerDecoderConfig,
        device: Optional[object],
        *,
        dtype: Optional[object] = DEFAULT_TT_DTYPE,
    ) -> None:
        self.config = config
        if device is not None and ttnn is None:
            require_ttnn("allocate the MaskFormer transformer decoder on a TT device")
        self.device = device
        self.dtype = dtype

        # Input projection (Conv1x1): in_features -> hidden_dim
        self._input_proj_w = None
        self._input_proj_b = None

        # Queries embedder (torch tensor, [Q, C])
        self._queries_embed: Optional["torch.Tensor"] = None

        # Final LayerNorm after all decoder layers
        self._final_ln_w = None
        self._final_ln_b = None

        # TTNN-prepared parameters per layer
        self._tt_params: Dict[str, Any] = {}

    def forward_tt(
        self,
        image_features: Any,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple["torch.Tensor", list, list]:
        """Run a DETR-style decoder stack using TTNN ops."""

        _ = output_attentions  # attention tensors are not collected in this bring-up path
        if self.device is None or ttnn is None:
            require_ttnn("run MaskFormer transformer decoder on device")
        if torch is None:
            raise RuntimeError("torch is required for MaskFormer TT decoder (positional embedding generation).")
        if not self._tt_params:
            raise RuntimeError("Transformer decoder weights are not loaded.")

        dtype = self.dtype or DEFAULT_TT_DTYPE

        # ------------------------------------------------------------------
        # 1) Input projection to hidden_dim (Conv1x1) and flatten to sequence
        # ------------------------------------------------------------------
        if torch is not None and isinstance(image_features, torch.Tensor):
            # NCHW -> NHWC
            feat_nhwc = image_features.detach().contiguous().permute(0, 2, 3, 1)
            mem_cfg = ttnn.DRAM_MEMORY_CONFIG
            tt_in = ttnn.from_torch(
                feat_nhwc,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=mem_cfg,
            )
        else:
            tt_in = image_features
            if getattr(tt_in, "get_layout", None) is not None and tt_in.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                tt_in = ttnn.to_layout(tt_in, ttnn.ROW_MAJOR_LAYOUT)

        B = int(tt_in.shape[0])
        H = int(tt_in.shape[1])
        W = int(tt_in.shape[2])

        tt_proj = ttnn.conv2d(
            input_tensor=tt_in,
            weight_tensor=self._input_proj_w,
            bias_tensor=self._input_proj_b,
            in_channels=int(tt_in.shape[-1]),
            out_channels=int(self.config.hidden_dim),
            batch_size=B,
            input_height=H,
            input_width=W,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        mem_seq = ttnn.reshape(tt_proj, (B, H * W, int(self.config.hidden_dim)))
        mem_seq = ttnn.to_layout(mem_seq, ttnn.TILE_LAYOUT)

        # 2D sine positional embeddings for encoder tokens (torch -> TT)
        pos_torch = self._build_sine_pos_embed(B, H, W, int(self.config.hidden_dim), dtype=torch.float32)
        tt_mem_pos = ttnn.from_torch(
            pos_torch,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ------------------------------------------------------------------
        # 2) Queries (learned) + initial hidden state (zeros)
        # ------------------------------------------------------------------
        if self._queries_embed is None:
            raise RuntimeError("queries_embedder.weight missing (load_weights not called).")
        q_embed = self._queries_embed.detach().contiguous()
        q_embed = q_embed.unsqueeze(0).repeat(B, 1, 1).contiguous()  # [B, Q, C]
        hidden = torch.zeros_like(q_embed)

        # Program/memory configs for attention + MLP
        num_heads = int(self.config.num_attention_heads)
        head_dim = int(self.config.hidden_dim // self.config.num_attention_heads)
        prog_cfg = build_decoder_program_configs(
            seq_q=int(hidden.shape[1]),
            seq_k=int(mem_seq.shape[1]),
            hidden_dim=self.config.hidden_dim,
            num_heads=num_heads,
            batch_size=int(hidden.shape[0]),
        )
        seq_memory_cfg = prog_cfg.sequence_memory or ttnn.DRAM_MEMORY_CONFIG

        def to_tt_sequence(x: "torch.Tensor"):
            return ttnn.from_torch(
                x,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=seq_memory_cfg,
            )

        tt_hidden = to_tt_sequence(hidden)
        tt_qpos = to_tt_sequence(q_embed)
        tt_mem = ttnn.to_memory_config(mem_seq, seq_memory_cfg)
        tt_mem_pos = ttnn.to_memory_config(tt_mem_pos, seq_memory_cfg)

        # ------------------------------------------------------------------
        # 3) Decoder stack
        # ------------------------------------------------------------------
        compute_cfg = self._make_compute_kernel_config()
        matmul_qkv_kwargs = {}
        matmul_ctx_kwargs = {}
        if prog_cfg.core_grid is not None:
            matmul_qkv_kwargs["core_grid"] = prog_cfg.core_grid
            matmul_ctx_kwargs["core_grid"] = prog_cfg.core_grid
        if prog_cfg.matmul_qkv is not None:
            matmul_qkv_kwargs["program_config"] = prog_cfg.matmul_qkv
        if prog_cfg.matmul_out is not None:
            matmul_ctx_kwargs["program_config"] = prog_cfg.matmul_out

        hidden_list: list = []
        attn_list: list = []

        fuse_linear_act = int(os.environ.get("MASKFORMER_TT_FUSE_LINEAR_ACT", "1")) == 1 and hasattr(ttnn, "linear")
        prefer_linear = int(os.environ.get("MASKFORMER_TT_USE_LINEAR", "1")) == 1 and hasattr(ttnn, "linear")

        def _project(tt_x, w, b, *, activation: Optional[str] = None, program: Optional[Any] = None):
            linear_kwargs = {}
            matmul_kwargs = {}
            if prog_cfg.core_grid is not None:
                linear_kwargs["core_grid"] = prog_cfg.core_grid
                matmul_kwargs["core_grid"] = prog_cfg.core_grid
            if program is None:
                program = prog_cfg.matmul_mlp
            if program is not None:
                linear_kwargs["program_config"] = program
                matmul_kwargs["program_config"] = program
            if (activation is not None and fuse_linear_act) or (activation is None and prefer_linear):
                try:
                    return ttnn.linear(
                        tt_x,
                        w,
                        b,
                        activation=activation,
                        compute_kernel_config=compute_cfg,
                        dtype=dtype,
                        **linear_kwargs,
                    )
                except Exception:
                    pass
            y = ttnn.matmul(tt_x, w, transpose_b=True, compute_kernel_config=compute_cfg, **matmul_kwargs)
            if b is not None:
                y = ttnn.add(y, b)
            if activation is not None:
                act = activation.lower()
                if act == "relu":
                    y = ttnn.relu(y)
                elif act == "gelu" and hasattr(ttnn, "gelu"):
                    try:
                        y = ttnn.gelu(y)
                    except Exception:
                        y = ttnn.relu(y)
                else:
                    y = ttnn.relu(y)
            return y

        def _project_qkv(tt_x, w, b):
            linear_kwargs = {}
            if prog_cfg.core_grid is not None:
                linear_kwargs["core_grid"] = prog_cfg.core_grid
            if prog_cfg.matmul_qkv is not None:
                linear_kwargs["program_config"] = prog_cfg.matmul_qkv
            try:
                qkv = ttnn.linear(
                    tt_x,
                    w,
                    b,
                    compute_kernel_config=compute_cfg,
                    dtype=dtype,
                    **linear_kwargs,
                )
                if hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "split_query_key_value_and_split_heads"):
                    return ttnn.transformer.split_query_key_value_and_split_heads(
                        qkv, num_heads=num_heads, transpose_key=False
                    )
            except Exception:
                pass
            return None

        def _split_heads(tt_x):
            # [B, L, C] -> [B, H, L, Hd]
            B0 = int(tt_x.shape[0])
            L0 = int(tt_x.shape[1])
            x4 = ttnn.reshape(tt_x, (B0, L0, num_heads, head_dim))
            return ttnn.permute(x4, (0, 2, 1, 3))

        def _merge_heads(tt_x):
            # [B, H, L, Hd] -> [B, L, C]
            B0 = int(tt_x.shape[0])
            L0 = int(tt_x.shape[2])
            x4 = ttnn.permute(tt_x, (0, 2, 1, 3))
            return ttnn.reshape(x4, (B0, L0, num_heads * head_dim))

        def _concat_heads(tt_x):
            if hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "concatenate_heads"):
                try:
                    return ttnn.transformer.concatenate_heads(tt_x)
                except Exception:
                    pass
            return _merge_heads(tt_x)

        def _manual_attention(attn_layer, q_in, k_in, v_in):
            q = _project(q_in, attn_layer["self_q_w"], attn_layer["self_q_b"], program=prog_cfg.matmul_qkv)
            k = _project(k_in, attn_layer["self_k_w"], attn_layer["self_k_b"], program=prog_cfg.matmul_qkv)
            v = _project(v_in, attn_layer["self_v_w"], attn_layer["self_v_b"], program=prog_cfg.matmul_qkv)
            q = _split_heads(q)
            k = _split_heads(k)
            v = _split_heads(v)
            k_t = ttnn.permute(k, (0, 1, 3, 2))
            scores = ttnn.matmul(q, k_t, compute_kernel_config=compute_cfg, **matmul_qkv_kwargs)
            attn = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=compute_cfg)
            ctx = ttnn.matmul(attn, v, compute_kernel_config=compute_cfg, **matmul_ctx_kwargs)
            return _merge_heads(ctx)

        for layer_idx in range(self.config.num_layers):
            layer = self._tt_params["layers"][layer_idx]

            # Self-attention: Q,K from (hidden + qpos), V from hidden
            qkv_in = ttnn.add(tt_hidden, tt_qpos)
            use_sdpa = hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "scaled_dot_product_attention")
            ctx = None
            if use_sdpa and layer.get("self_qkv_w") is not None:
                try:
                    q, k, v = _project_qkv(qkv_in, layer["self_qkv_w"], layer.get("self_qkv_b"))
                    if q is not None and k is not None and v is not None:
                        attn_ctx = ttnn.transformer.scaled_dot_product_attention(
                            q, k, v, is_causal=False, program_config=prog_cfg.sdpa, compute_kernel_config=compute_cfg
                        )
                        ctx = _concat_heads(attn_ctx)
                except Exception:
                    ctx = None
            if ctx is None:
                ctx = _manual_attention(layer, qkv_in, qkv_in, tt_hidden)
            sa_out = _project(ctx, layer["self_out_w"], layer.get("self_out_b"), program=prog_cfg.matmul_out)
            sa_res = ttnn.add(tt_hidden, sa_out)
            tt_hidden = ttnn.layer_norm(sa_res, weight=layer["ln1_w"], bias=layer["ln1_b"], epsilon=1e-5)

            # Cross-attention: Q from hidden+qpos, K from mem+pos, V from mem
            q_in = ttnn.add(tt_hidden, tt_qpos)
            k_in = ttnn.add(tt_mem, tt_mem_pos)
            v_in = tt_mem

            ctx = None
            if use_sdpa:
                try:
                    q = _project(q_in, layer["cross_q_w"], layer["cross_q_b"], program=prog_cfg.matmul_qkv)
                    k = _project(k_in, layer["cross_k_w"], layer["cross_k_b"], program=prog_cfg.matmul_qkv)
                    v = _project(v_in, layer["cross_v_w"], layer["cross_v_b"], program=prog_cfg.matmul_qkv)
                    qh = _split_heads(q)
                    kh = _split_heads(k)
                    vh = _split_heads(v)
                    attn_ctx = ttnn.transformer.scaled_dot_product_attention(
                        qh, kh, vh, is_causal=False, program_config=prog_cfg.sdpa, compute_kernel_config=compute_cfg
                    )
                    ctx = _concat_heads(attn_ctx)
                except Exception:
                    ctx = None
            if ctx is None:
                ctx = _manual_attention(
                    {
                        "self_q_w": layer["cross_q_w"],
                        "self_q_b": layer["cross_q_b"],
                        "self_k_w": layer["cross_k_w"],
                        "self_k_b": layer["cross_k_b"],
                        "self_v_w": layer["cross_v_w"],
                        "self_v_b": layer["cross_v_b"],
                    },
                    q_in,
                    k_in,
                    v_in,
                )
            ca_out = _project(ctx, layer["cross_out_w"], layer.get("cross_out_b"), program=prog_cfg.matmul_out)
            ca_res = ttnn.add(tt_hidden, ca_out)
            tt_hidden = ttnn.layer_norm(ca_res, weight=layer["ln2_w"], bias=layer["ln2_b"], epsilon=1e-5)

            # FFN
            act_name = (self.config.activation or "relu").lower()
            mlp_hidden = _project(tt_hidden, layer["mlp_w1"], layer.get("mlp_b1"), activation=act_name)
            mlp_2 = _project(mlp_hidden, layer["mlp_w2"], layer.get("mlp_b2"))
            mlp_res = ttnn.add(tt_hidden, mlp_2)
            tt_hidden = ttnn.layer_norm(mlp_res, weight=layer["ln3_w"], bias=layer["ln3_b"], epsilon=1e-5)

            if output_hidden_states:
                hidden_list.append(self._tt_to_torch(tt_hidden))

        # Final LayerNorm
        tt_hidden = ttnn.layer_norm(tt_hidden, weight=self._final_ln_w, bias=self._final_ln_b, epsilon=1e-5)
        last_hidden = self._tt_to_torch(tt_hidden)
        return last_hidden, hidden_list, attn_list

    def load_weights(self, weights: Dict[str, object]) -> None:
        if self.device is None or ttnn is None:
            require_ttnn("load MaskFormer transformer decoder weights on device")
        if torch is None:
            raise RuntimeError("torch is required to load MaskFormer transformer decoder weights.")

        state = extract_transformer_state(weights)
        dtype = self.dtype or DEFAULT_TT_DTYPE
        mem_cfg = ttnn.L1_MEMORY_CONFIG

        # Input projection conv1x1
        w = state["input_projection.weight"]
        b = state["input_projection.bias"]
        if not isinstance(w, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("Expected torch tensors for input_projection.*")
        self._input_proj_w = ttnn.from_torch(
            w.detach().contiguous(),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=mem_cfg,
        )
        self._input_proj_b = ttnn.from_torch(
            b.detach().contiguous().view(1, 1, 1, -1),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=mem_cfg,
        )

        # Queries embedder
        q = state["queries_embedder.weight"]
        if not isinstance(q, torch.Tensor):
            raise TypeError("Expected torch tensor for queries_embedder.weight")
        self._queries_embed = q

        # Final decoder layernorm
        self._final_ln_w, self._final_ln_b = self._to_tt_norm(
            state["decoder.layernorm.weight"], state["decoder.layernorm.bias"]
        )

        # Per-layer weights
        num_layers = int(self.config.num_layers)
        num_heads = int(self.config.num_attention_heads)
        head_dim = int(self.config.hidden_dim // self.config.num_attention_heads)
        scale = 1.0 / math.sqrt(head_dim)

        def _maybe_tensor(value):
            return value if isinstance(value, torch.Tensor) else None

        params: Dict[str, Any] = {"layers": []}
        for li in range(num_layers):
            prefix = f"decoder.layers.{li}"

            # Self-attention
            sa_q_w = state[f"{prefix}.self_attn.q_proj.weight"].detach().contiguous() * scale
            sa_q_b = _maybe_tensor(state.get(f"{prefix}.self_attn.q_proj.bias"))
            sa_k_w = state[f"{prefix}.self_attn.k_proj.weight"]
            sa_k_b = _maybe_tensor(state.get(f"{prefix}.self_attn.k_proj.bias"))
            sa_v_w = state[f"{prefix}.self_attn.v_proj.weight"]
            sa_v_b = _maybe_tensor(state.get(f"{prefix}.self_attn.v_proj.bias"))
            sa_o_w = state[f"{prefix}.self_attn.out_proj.weight"]
            sa_o_b = _maybe_tensor(state.get(f"{prefix}.self_attn.out_proj.bias"))

            tt_sa_q_w, tt_sa_q_b = self._to_tt_linear(sa_q_w, sa_q_b)
            tt_sa_k_w, tt_sa_k_b = self._to_tt_linear(sa_k_w, sa_k_b)
            tt_sa_v_w, tt_sa_v_b = self._to_tt_linear(sa_v_w, sa_v_b)
            tt_sa_o_w, tt_sa_o_b = self._to_tt_linear(sa_o_w, sa_o_b)

            # Optional fused QKV for SDPA
            try:
                qkv_w = torch.cat([sa_q_w, sa_k_w.detach().contiguous(), sa_v_w.detach().contiguous()], dim=0)
                qkv_b = None
                if sa_q_b is not None and sa_k_b is not None and sa_v_b is not None:
                    qkv_b = torch.cat(
                        [sa_q_b.detach().contiguous(), sa_k_b.detach().contiguous(), sa_v_b.detach().contiguous()],
                        dim=0,
                    )
                tt_qkv_w, tt_qkv_b = self._to_tt_linear(qkv_w, qkv_b)
            except Exception:
                tt_qkv_w, tt_qkv_b = (None, None)

            # Cross-attention
            ca_q_w = state[f"{prefix}.encoder_attn.q_proj.weight"].detach().contiguous() * scale
            ca_q_b = _maybe_tensor(state.get(f"{prefix}.encoder_attn.q_proj.bias"))
            ca_k_w = state[f"{prefix}.encoder_attn.k_proj.weight"]
            ca_k_b = _maybe_tensor(state.get(f"{prefix}.encoder_attn.k_proj.bias"))
            ca_v_w = state[f"{prefix}.encoder_attn.v_proj.weight"]
            ca_v_b = _maybe_tensor(state.get(f"{prefix}.encoder_attn.v_proj.bias"))
            ca_o_w = state[f"{prefix}.encoder_attn.out_proj.weight"]
            ca_o_b = _maybe_tensor(state.get(f"{prefix}.encoder_attn.out_proj.bias"))

            tt_ca_q_w, tt_ca_q_b = self._to_tt_linear(ca_q_w, ca_q_b)
            tt_ca_k_w, tt_ca_k_b = self._to_tt_linear(ca_k_w, ca_k_b)
            tt_ca_v_w, tt_ca_v_b = self._to_tt_linear(ca_v_w, ca_v_b)
            tt_ca_o_w, tt_ca_o_b = self._to_tt_linear(ca_o_w, ca_o_b)

            # FFN
            mlp_w1, mlp_b1 = self._to_tt_linear(
                state[f"{prefix}.fc1.weight"], _maybe_tensor(state.get(f"{prefix}.fc1.bias"))
            )
            mlp_w2, mlp_b2 = self._to_tt_linear(
                state[f"{prefix}.fc2.weight"], _maybe_tensor(state.get(f"{prefix}.fc2.bias"))
            )

            # LayerNorms
            ln1_w, ln1_b = self._to_tt_norm(
                state[f"{prefix}.self_attn_layer_norm.weight"], state[f"{prefix}.self_attn_layer_norm.bias"]
            )
            ln2_w, ln2_b = self._to_tt_norm(
                state[f"{prefix}.encoder_attn_layer_norm.weight"], state[f"{prefix}.encoder_attn_layer_norm.bias"]
            )
            ln3_w, ln3_b = self._to_tt_norm(
                state[f"{prefix}.final_layer_norm.weight"], state[f"{prefix}.final_layer_norm.bias"]
            )

            params["layers"].append(
                {
                    "self_q_w": tt_sa_q_w,
                    "self_q_b": tt_sa_q_b,
                    "self_k_w": tt_sa_k_w,
                    "self_k_b": tt_sa_k_b,
                    "self_v_w": tt_sa_v_w,
                    "self_v_b": tt_sa_v_b,
                    "self_out_w": tt_sa_o_w,
                    "self_out_b": tt_sa_o_b,
                    "self_qkv_w": tt_qkv_w,
                    "self_qkv_b": tt_qkv_b,
                    "cross_q_w": tt_ca_q_w,
                    "cross_q_b": tt_ca_q_b,
                    "cross_k_w": tt_ca_k_w,
                    "cross_k_b": tt_ca_k_b,
                    "cross_v_w": tt_ca_v_w,
                    "cross_v_b": tt_ca_v_b,
                    "cross_out_w": tt_ca_o_w,
                    "cross_out_b": tt_ca_o_b,
                    "mlp_w1": mlp_w1,
                    "mlp_b1": mlp_b1,
                    "mlp_w2": mlp_w2,
                    "mlp_b2": mlp_b2,
                    "ln1_w": ln1_w,
                    "ln1_b": ln1_b,
                    "ln2_w": ln2_w,
                    "ln2_b": ln2_b,
                    "ln3_w": ln3_w,
                    "ln3_b": ln3_b,
                }
            )

        self._tt_params = params

    @classmethod
    def from_huggingface(
        cls,
        weights: Dict[str, object],
        *,
        config: TransformerDecoderConfig,
        device: Optional[object] = None,
    ) -> "MaskFormerTransformerDecoder":
        decoder = cls(config=config, device=device)
        decoder.load_weights(weights)
        return decoder

    def _to_tt_linear(self, w: "torch.Tensor", b: Optional["torch.Tensor"]):
        dtype = self.dtype or DEFAULT_TT_DTYPE
        wt = ttnn.from_torch(
            w.detach().contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        bt = None
        if b is not None:
            bt = ttnn.from_torch(
                b.detach().contiguous().view(1, 1, -1),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            bt = ttnn.to_layout(bt, ttnn.TILE_LAYOUT)
        return wt, bt

    def _to_tt_norm(self, w: "torch.Tensor", b: "torch.Tensor"):
        dtype = self.dtype or DEFAULT_TT_DTYPE
        wt = ttnn.from_torch(
            w.detach().contiguous().view(1, 1, -1),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        bt = ttnn.from_torch(
            b.detach().contiguous().view(1, 1, -1),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        wt = ttnn.to_layout(wt, ttnn.TILE_LAYOUT)
        bt = ttnn.to_layout(bt, ttnn.TILE_LAYOUT)
        return wt, bt

    def _tt_to_torch(self, tensor: Any) -> "torch.Tensor":
        if hasattr(ttnn, "to_torch"):
            return ttnn.to_torch(tensor)
        if hasattr(tensor, "to_torch"):
            return tensor.to_torch()
        raise TypeError("Unsupported TTNN tensor conversion to torch.")

    def _make_compute_kernel_config(self):
        if ttnn is None:
            raise RuntimeError("TTNN runtime is required to construct compute kernel configs.")
        ComputeConfigClass = ttnn.WormholeComputeKernelConfig
        try:
            if is_blackhole() and hasattr(ttnn, "types") and hasattr(ttnn.types, "BlackholeComputeKernelConfig"):
                ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig  # type: ignore[assignment]
        except Exception:
            pass
        return ComputeConfigClass(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @staticmethod
    def _build_sine_pos_embed(batch_size: int, height: int, width: int, hidden_dim: int, *, dtype: "torch.dtype"):
        """Standard DETR sine positional embedding (no mask/padding). Returns [B, HW, C]."""

        if torch is None:
            raise RuntimeError("torch is required to build sine positional embeddings.")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for sine positional embedding, got {hidden_dim}")

        num_pos_feats = hidden_dim // 2
        temperature = 10000
        scale = 2 * math.pi
        eps = 1e-6

        y_embed = torch.arange(1, height + 1, device="cpu", dtype=dtype).view(1, height, 1).repeat(batch_size, 1, width)
        x_embed = torch.arange(1, width + 1, device="cpu", dtype=dtype).view(1, 1, width).repeat(batch_size, height, 1)

        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, device="cpu", dtype=dtype)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # [B,H,W,C]
        pos = pos.view(batch_size, height * width, hidden_dim).contiguous()
        return pos
