# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Transformer decoder for MaskFormer queries.

The decoder follows the DETR pattern: multi-head self-attention over queries,
cross-attention against pixel embeddings, and feed-forward sublayers.  This
module will glue together TT-NN building blocks to implement both attention
paths and layer normalisation / MLP stages while honouring layout constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import os
import torch
import warnings

try:
    from transformers import MaskFormerConfig
    from transformers.models.maskformer.modeling_maskformer import MaskFormerTransformerModule, DetrDecoderOutput
except ModuleNotFoundError:  # pragma: no cover - fallback optional
    MaskFormerConfig = None
    MaskFormerTransformerModule = None
    DetrDecoderOutput = None

try:
    from models.common.utility_functions import tt_to_torch_tensor, is_blackhole
except ModuleNotFoundError:  # pragma: no cover - optional when running outside repo context
    tt_to_torch_tensor = None

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
    maskformer_config: Optional[Dict[str, object]] = None


class MaskFormerTransformerDecoder:
    """Runs query refinement via TT-NN attention and MLP blocks."""

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
        self._hf_decoder: Optional[MaskFormerTransformerModule] = None
        if MaskFormerTransformerModule and MaskFormerConfig and config.maskformer_config:
            hf_config = MaskFormerConfig(**config.maskformer_config)
            self._hf_decoder = MaskFormerTransformerModule(config.in_features, hf_config)
        self._torch_state: Dict[str, Any] = {}
        # TTNN-prepared parameters per layer
        self._tt_params: Dict[str, Any] = {}

    def forward(
        self,
        image_features: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, list, list]:
        """Execute the decoder stack using the HuggingFace fallback."""

        if self._hf_decoder is None:
            raise NotImplementedError("TT-NN transformer decoder pending; install transformers for fallback execution.")

        inputs = self._ensure_torch_tensor(image_features)
        with torch.no_grad():
            result: DetrDecoderOutput = self._hf_decoder(
                inputs,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )
        hidden_states = list(result.hidden_states) if result.hidden_states is not None else []
        attentions = list(result.attentions) if result.attentions is not None else []
        return result.last_hidden_state, hidden_states, attentions

    def forward_tt(
        self,
        image_features: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, list, list]:
        """Run a DETR-style decoder stack using TTNN attention and MLP blocks.

        This implementation mirrors HuggingFace's DetrDecoderLayer with:
        - self-attention over queries (adds query position embeddings)
        - cross-attention against encoder (image) sequence (adds spatial pos enc)
        - two-layer MLP with activation
        - post-attention/post-MLP LayerNorms (eps=1e-5)

        Notes:
        - Uses HiFi2 + fp32_dest_acc for matmuls/softmax.
        - Input-projection (1x1 conv when in_features != hidden_dim) is performed via HF module for simplicity.
        - Returns torch tensors for downstream heads.
        """

        if self.device is None or ttnn is None:
            # Fall back to HF if device or TTNN is not available
            return self.forward(
                image_features, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
        if self._hf_decoder is None:
            # Need HF module for position/query embedders and optional input projection
            raise NotImplementedError(
                "HF transformer module unavailable for TT decoder scaffolding (position/query embeddings)."
            )

        # Prepare TT weights on first use
        if not self._tt_params:
            self._prepare_tt_params()

        # 1) Optional input projection to hidden_dim using HF conv1x1 (cheap on CPU)
        with torch.no_grad():
            feats = image_features
            if getattr(self._hf_decoder, "input_projection", None) is not None:
                feats = self._hf_decoder.input_projection(self._ensure_torch_tensor(feats))
            else:
                feats = self._ensure_torch_tensor(feats)

        # 2) Build position embeddings for encoder tokens and query embeddings
        with torch.no_grad():
            bsz, ch, h, w = feats.shape
            try:
                pos = self._hf_decoder.position_embedder(feats)  # HF >=4.34 signature: (x, mask=None)
            except TypeError:
                # Backward-compat: older HF versions expect shape/device/dtype triplet
                pos = self._hf_decoder.position_embedder(feats.shape, feats.device, feats.dtype)
            # Flatten to sequences [B, HW, C]
            mem = feats.view(bsz, ch, h * w).permute(0, 2, 1).contiguous()
            pos_mem = pos.view(bsz, ch, h * w).permute(0, 2, 1).contiguous()
            # Queries: learned embeddings repeated per batch
            q_embed = self._hf_decoder.queries_embedder.weight.detach()  # [Q, C]
            q_embed = q_embed.unsqueeze(0).repeat(bsz, 1, 1).contiguous()  # [B, Q, C]
            # Initial hidden states are zeros (inputs_embeds in HF path)
            hidden = torch.zeros_like(q_embed, dtype=feats.dtype, device=feats.device)

        # Program/memory configs for attention + MLP
        num_heads = int(self.config.num_attention_heads)
        head_dim = int(self.config.hidden_dim // self.config.num_attention_heads)
        prog_cfg = build_decoder_program_configs(
            seq_q=int(hidden.shape[1]),
            seq_k=int(mem.shape[1]),
            hidden_dim=self.config.hidden_dim,
            num_heads=num_heads,
            batch_size=int(hidden.shape[0]),
        )
        seq_memory_cfg = prog_cfg.sequence_memory or ttnn.DRAM_MEMORY_CONFIG

        # Convert sequences to TTNN tiles (prefer L1 when configured)
        def to_tt_sequence(x):
            try:
                return ttnn.from_torch(
                    x,
                    dtype=self.dtype or DEFAULT_TT_DTYPE,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=seq_memory_cfg,
                )
            except Exception:
                return ttnn.from_torch(
                    x,
                    dtype=self.dtype or DEFAULT_TT_DTYPE,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

        tt_hidden = to_tt_sequence(hidden)
        tt_qpos = to_tt_sequence(q_embed)
        tt_mem = to_tt_sequence(mem)
        tt_mem_pos = to_tt_sequence(pos_mem)

        if ttnn is None:
            raise RuntimeError("TT-NN runtime is required for TT transformer decoder execution.")
        ComputeConfigClass = ttnn.WormholeComputeKernelConfig
        try:
            if is_blackhole() and hasattr(ttnn, "types") and hasattr(ttnn.types, "BlackholeComputeKernelConfig"):
                ComputeConfigClass = ttnn.types.BlackholeComputeKernelConfig  # type: ignore[assignment]
        except Exception:
            # Fall back to Wormhole config on detection errors.
            pass
        compute_cfg = ComputeConfigClass(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        matmul_qkv_kwargs = {}
        matmul_ctx_kwargs = {}
        if prog_cfg.core_grid is not None:
            matmul_qkv_kwargs["core_grid"] = prog_cfg.core_grid
            matmul_ctx_kwargs["core_grid"] = prog_cfg.core_grid
        if prog_cfg.matmul_qkv is not None:
            matmul_qkv_kwargs["program_config"] = prog_cfg.matmul_qkv
        if prog_cfg.matmul_out is not None:
            matmul_ctx_kwargs["program_config"] = prog_cfg.matmul_out

        # Decoder stack
        hidden_list: list = []
        attn_list: list = []  # not populated in this minimal TT path

        # Optional fused Linear+Bias(+Activation) support when available
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
                        dtype=self.dtype or DEFAULT_TT_DTYPE,
                        **linear_kwargs,
                    )
                except Exception:
                    # Fallback to matmul path if fused is unsupported at runtime
                    pass
            y = ttnn.matmul(tt_x, w, transpose_b=True, compute_kernel_config=compute_cfg, **matmul_kwargs)
            if b is not None:
                y = ttnn.add(y, b)
            if activation is not None:
                # Apply activation separately when fusion unavailable
                if activation == "relu":
                    y = ttnn.relu(y)
                elif activation == "gelu" and hasattr(ttnn, "gelu"):
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
                    dtype=self.dtype or DEFAULT_TT_DTYPE,
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
            # [B, L, C] -> [B, H, L, Hd] as a 4D view
            B = int(tt_x.shape[0])
            L = int(tt_x.shape[1])
            C = int(tt_x.shape[2])
            assert C == num_heads * head_dim
            return ttnn.reshape(tt_x, (B, L, num_heads, head_dim))

        def _merge_heads(tt_x):
            # [B, L, H, Hd] -> [B, L, C]
            B = int(tt_x.shape[0])
            L = int(tt_x.shape[1])
            return ttnn.reshape(tt_x, (B, L, num_heads * head_dim))

        def _split_heads_sdpa(tt_x):
            # [B, L, C] -> [B, H, L, Hd] but head dim first for SDPA
            B = int(tt_x.shape[0])
            L = int(tt_x.shape[1])
            return ttnn.reshape(tt_x, (B, num_heads, L, head_dim))

        def _concat_heads(tt_x):
            if hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "concatenate_heads"):
                try:
                    return ttnn.transformer.concatenate_heads(tt_x)
                except Exception:
                    pass
            B = int(tt_x.shape[0])
            L = int(tt_x.shape[2])
            return ttnn.reshape(tt_x, (B, L, num_heads * head_dim))

        def _manual_attention(attn_layer, q_in, k_in, v_in):
            q = _project(q_in, attn_layer["self_q_w"], attn_layer["self_q_b"], program=prog_cfg.matmul_qkv)
            k = _project(k_in, attn_layer["self_k_w"], attn_layer["self_k_b"], program=prog_cfg.matmul_qkv)
            v = _project(v_in, attn_layer["self_v_w"], attn_layer["self_v_b"], program=prog_cfg.matmul_qkv)
            q = _split_heads(q)
            k = _split_heads(k)
            v = _split_heads(v)
            B = int(q.shape[0])
            Lq = int(q.shape[1])
            Lk = int(k.shape[1])
            q_bh = ttnn.reshape(q, (B * num_heads, Lq, head_dim))
            k_bh = ttnn.reshape(k, (B * num_heads, Lk, head_dim))
            scores = ttnn.matmul(q_bh, k_bh, transpose_b=True, compute_kernel_config=compute_cfg, **matmul_qkv_kwargs)
            attn = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=compute_cfg)
            v_bh = ttnn.reshape(v, (B * num_heads, Lk, head_dim))
            ctx = ttnn.matmul(attn, v_bh, compute_kernel_config=compute_cfg, **matmul_ctx_kwargs)
            ctx = ttnn.reshape(ctx, (B, Lq, num_heads, head_dim))
            return _merge_heads(ctx)

        for layer_idx in range(self.config.num_layers):
            layer = self._tt_params["layers"][layer_idx]
            if int(os.environ.get("MASKFORMER_DEBUG_DECODER", "0")) and layer_idx == 0:
                try:
                    B = int(tt_hidden.shape[0])
                    Q = int(tt_hidden.shape[1])
                    C = int(tt_hidden.shape[2])
                    MB = int(tt_mem.shape[0])
                    ML = int(tt_mem.shape[1])
                    MC = int(tt_mem.shape[2])
                    print(f"[tt-decoder] hidden {B}x{Q}x{C} mem {MB}x{ML}x{MC} heads={num_heads} head_dim={head_dim}")
                except Exception:
                    pass
            # Self-attention: add query position embeddings to queries/keys inputs
            qkv_in = ttnn.add(tt_hidden, tt_qpos)
            use_sdpa = hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "scaled_dot_product_attention")
            ctx = None
            if use_sdpa and layer.get("self_qkv_w") is not None:
                try:
                    q, k, v = _project_qkv(qkv_in, layer["self_qkv_w"], layer.get("self_qkv_b"))
                    if q is not None and k is not None and v is not None:
                        attn_ctx = ttnn.transformer.scaled_dot_product_attention(
                            q,
                            k,
                            v,
                            is_causal=False,
                            program_config=prog_cfg.sdpa,
                            compute_kernel_config=compute_cfg,
                        )
                        ctx = _concat_heads(attn_ctx)
                except Exception:
                    ctx = None
            if ctx is None:
                ctx = _manual_attention(layer, qkv_in, qkv_in, tt_hidden)
            sa_out = _project(ctx, layer["self_out_w"], layer.get("self_out_b"), program=prog_cfg.matmul_out)
            # Residual + LayerNorm (post-norm)
            sa_res = ttnn.add(tt_hidden, sa_out)
            tt_hidden = ttnn.layer_norm(
                sa_res,
                weight=layer["ln1_w"],
                bias=layer["ln1_b"],
                epsilon=1e-5,
            )

            # Cross-attention: Q from tt_hidden+qpos, K from mem+pos, V from mem
            q_in = ttnn.add(tt_hidden, tt_qpos)
            k_in = ttnn.add(tt_mem, tt_mem_pos)
            v_in = tt_mem
            use_sdpa = hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "scaled_dot_product_attention")
            ctx = None
            if use_sdpa:
                try:
                    q = _project(q_in, layer["cross_q_w"], layer["cross_q_b"], program=prog_cfg.matmul_qkv)
                    k = _project(k_in, layer["cross_k_w"], layer["cross_k_b"], program=prog_cfg.matmul_qkv)
                    v = _project(v_in, layer["cross_v_w"], layer["cross_v_b"], program=prog_cfg.matmul_qkv)
                    qh = _split_heads_sdpa(q)
                    kh = _split_heads_sdpa(k)
                    vh = _split_heads_sdpa(v)
                    attn_ctx = ttnn.transformer.scaled_dot_product_attention(
                        qh,
                        kh,
                        vh,
                        is_causal=False,
                        program_config=prog_cfg.sdpa,
                        compute_kernel_config=compute_cfg,
                    )
                    ctx = _concat_heads(attn_ctx)
                except Exception:
                    ctx = None
            if ctx is None:
                q = _project(q_in, layer["cross_q_w"], layer["cross_q_b"], program=prog_cfg.matmul_qkv)
                k = _project(k_in, layer["cross_k_w"], layer["cross_k_b"], program=prog_cfg.matmul_qkv)
                v = _project(v_in, layer["cross_v_w"], layer["cross_v_b"], program=prog_cfg.matmul_qkv)
                q = _split_heads(q)
                k = _split_heads(k)
                v = _split_heads(v)
                B = int(q.shape[0])
                Lq = int(q.shape[1])
                Lv = int(v.shape[1])
                q_bh = ttnn.reshape(q, (B * num_heads, Lq, head_dim))
                k_bh = ttnn.reshape(k, (B * num_heads, Lv, head_dim))
                scores = ttnn.matmul(
                    q_bh, k_bh, transpose_b=True, compute_kernel_config=compute_cfg, **matmul_qkv_kwargs
                )
                attn = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=compute_cfg)
                v_bh = ttnn.reshape(v, (B * num_heads, Lv, head_dim))
                ctx = ttnn.matmul(attn, v_bh, compute_kernel_config=compute_cfg, **matmul_ctx_kwargs)
                ctx = ttnn.reshape(ctx, (B, Lq, num_heads, head_dim))
                ctx = _merge_heads(ctx)
            ca_out = _project(ctx, layer["cross_out_w"], layer.get("cross_out_b"), program=prog_cfg.matmul_out)
            ca_res = ttnn.add(tt_hidden, ca_out)
            tt_hidden = ttnn.layer_norm(
                ca_res,
                weight=layer["ln2_w"],
                bias=layer["ln2_b"],
                epsilon=1e-5,
            )

            # MLP block: Linear1 -> activation -> Linear2
            # Prefer fused Linear+Bias+Act on first MLP layer when available
            act_name = (self.config.activation or "relu").lower()
            mlp_hidden = _project(tt_hidden, layer["mlp_w1"], layer.get("mlp_b1"), activation=act_name)
            mlp_2 = _project(mlp_hidden, layer["mlp_w2"], layer.get("mlp_b2"))
            mlp_res = ttnn.add(tt_hidden, mlp_2)
            tt_hidden = ttnn.layer_norm(
                mlp_res,
                weight=layer["ln3_w"],
                bias=layer["ln3_b"],
                epsilon=1e-5,
            )

            if output_hidden_states:
                # Convert to torch lazily (only when requested)
                hidden_list.append(self._tt_to_torch(tt_hidden))

        # Convert final hidden state to torch tensor for heads
        last_hidden = self._tt_to_torch(tt_hidden)
        return last_hidden, hidden_list, attn_list

    def load_weights(self, weights: Dict[str, object]) -> None:
        """Load transformer decoder weights from HuggingFace-style state dict."""

        if self._hf_decoder is None:
            return

        state = extract_transformer_state(weights)
        torch_state = {name: self._ensure_torch_tensor(tensor) for name, tensor in state.items()}
        missing, unexpected = self._hf_decoder.load_state_dict(torch_state, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Transformer decoder weight load mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}",
                RuntimeWarning,
            )
        self._torch_state = torch_state
        # Prepare TTNN parameters if a device is present
        if self.device is not None and ttnn is not None:
            try:
                self._prepare_tt_params()
            except Exception as e:
                warnings.warn(f"Failed to prepare TT decoder params: {e}")

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

    def _ensure_torch_tensor(self, tensor: Any) -> torch.Tensor:
        if isinstance(tensor, torch.Tensor):
            return tensor
        if tt_to_torch_tensor is not None:
            try:
                return tt_to_torch_tensor(tensor)
            except Exception:
                pass
        if hasattr(tensor, "to_torch"):
            return tensor.to_torch()
        if hasattr(tensor, "cpu"):
            return torch.tensor(tensor.cpu().numpy())
        if isinstance(tensor, (list, tuple)):
            return torch.tensor(tensor)
        raise TypeError(f"Unsupported tensor type for conversion to torch: {type(tensor)!r}")

    def _tt_to_torch(self, tensor: Any) -> torch.Tensor:
        if hasattr(ttnn, "to_torch"):
            try:
                return ttnn.to_torch(tensor)
            except Exception:
                pass
        return self._ensure_torch_tensor(tensor)

    # ------------------------------------------------------------------
    # TT weight preparation and small utilities
    # ------------------------------------------------------------------
    def _to_tt_linear(self, w: torch.Tensor, b: Optional[torch.Tensor]):
        wt = ttnn.from_torch(
            w.detach().contiguous(),
            dtype=self.dtype or DEFAULT_TT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        bt = None
        if b is not None:
            b_reshaped = b.detach().contiguous().view(1, 1, -1)
            bt = ttnn.from_torch(
                b_reshaped,
                dtype=self.dtype or DEFAULT_TT_DTYPE,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            bt = ttnn.to_layout(bt, ttnn.TILE_LAYOUT)
        return wt, bt

    def _to_tt_norm(self, w: torch.Tensor, b: torch.Tensor):
        wt = ttnn.from_torch(
            w.detach().contiguous().view(1, 1, -1),
            dtype=self.dtype or DEFAULT_TT_DTYPE,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        bt = ttnn.from_torch(
            b.detach().contiguous().view(1, 1, -1),
            dtype=self.dtype or DEFAULT_TT_DTYPE,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        wt = ttnn.to_layout(wt, ttnn.TILE_LAYOUT)
        bt = ttnn.to_layout(bt, ttnn.TILE_LAYOUT)
        return wt, bt

    def _prepare_tt_params(self) -> None:
        if self.device is None or ttnn is None or self._hf_decoder is None:
            return
        params: Dict[str, Any] = {"layers": []}
        # Iterate over HF decoder layers and extract weights
        try:
            hf_decoder = self._hf_decoder.decoder
        except Exception:
            # Older versions may expose layers directly
            hf_decoder = getattr(self._hf_decoder, "layers", None)
        layers = list(getattr(hf_decoder, "layers", [])) if hasattr(hf_decoder, "layers") else list(hf_decoder)
        if not layers:
            raise RuntimeError("Unable to locate HF DETR decoder layers for TT weight prep.")

        for li, layer in enumerate(layers):
            # Self-attention projections
            sa = layer.self_attn
            q_w, q_b = sa.q_proj.weight, sa.q_proj.bias
            k_w, k_b = sa.k_proj.weight, sa.k_proj.bias
            v_w, v_b = sa.v_proj.weight, sa.v_proj.bias
            o_w, o_b = sa.out_proj.weight, sa.out_proj.bias
            # Pre-scale the query projection by 1/sqrt(head_dim) to absorb attention scaling
            try:
                head_dim = int(getattr(sa, "head_dim", self.config.hidden_dim // self.config.num_attention_heads))
            except Exception:
                head_dim = self.config.hidden_dim // self.config.num_attention_heads
            scale = 1.0 / (head_dim**0.5)
            q_w_s = q_w.detach().contiguous() * scale
            tt_q_w, tt_q_b = self._to_tt_linear(q_w_s, q_b)
            tt_k_w, tt_k_b = self._to_tt_linear(k_w, k_b)
            tt_v_w, tt_v_b = self._to_tt_linear(v_w, v_b)
            tt_o_w, tt_o_b = self._to_tt_linear(o_w, o_b)
            # Fused QKV for SDPA
            try:
                qkv_w = torch.cat([q_w_s, k_w.detach().contiguous(), v_w.detach().contiguous()], dim=0)
                qkv_b = None
                if q_b is not None and k_b is not None and v_b is not None:
                    qkv_b = torch.cat(
                        [q_b.detach().contiguous(), k_b.detach().contiguous(), v_b.detach().contiguous()], dim=0
                    )
                tt_qkv_w, tt_qkv_b = self._to_tt_linear(qkv_w, qkv_b)
            except Exception:
                tt_qkv_w, tt_qkv_b = (None, None)

            # Cross-attention projections
            ca = layer.encoder_attn
            cq_w, cq_b = ca.q_proj.weight, ca.q_proj.bias
            ck_w, ck_b = ca.k_proj.weight, ca.k_proj.bias
            cv_w, cv_b = ca.v_proj.weight, ca.v_proj.bias
            co_w, co_b = ca.out_proj.weight, ca.out_proj.bias
            # Apply the same scaling to cross-attention query projection
            cq_w_s = cq_w.detach().contiguous() * scale
            tt_cq_w, tt_cq_b = self._to_tt_linear(cq_w_s, cq_b)
            tt_ck_w, tt_ck_b = self._to_tt_linear(ck_w, ck_b)
            tt_cv_w, tt_cv_b = self._to_tt_linear(cv_w, cv_b)
            tt_co_w, tt_co_b = self._to_tt_linear(co_w, co_b)

            # LayerNorms
            ln1_w, ln1_b = layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias
            ln2_w, ln2_b = layer.encoder_attn_layer_norm.weight, layer.encoder_attn_layer_norm.bias
            ln3_w, ln3_b = layer.final_layer_norm.weight, layer.final_layer_norm.bias
            tt_ln1_w, tt_ln1_b = self._to_tt_norm(ln1_w, ln1_b)
            tt_ln2_w, tt_ln2_b = self._to_tt_norm(ln2_w, ln2_b)
            tt_ln3_w, tt_ln3_b = self._to_tt_norm(ln3_w, ln3_b)

            # MLP
            w1, b1 = layer.fc1.weight, layer.fc1.bias
            w2, b2 = layer.fc2.weight, layer.fc2.bias
            tt_w1, tt_b1 = self._to_tt_linear(w1, b1)
            tt_w2, tt_b2 = self._to_tt_linear(w2, b2)

            params["layers"].append(
                {
                    "self_q_w": tt_q_w,
                    "self_q_b": tt_q_b,
                    "self_k_w": tt_k_w,
                    "self_k_b": tt_k_b,
                    "self_v_w": tt_v_w,
                    "self_v_b": tt_v_b,
                    "self_out_w": tt_o_w,
                    "self_out_b": tt_o_b,
                    "self_qkv_w": tt_qkv_w,
                    "self_qkv_b": tt_qkv_b,
                    "cross_q_w": tt_cq_w,
                    "cross_q_b": tt_cq_b,
                    "cross_k_w": tt_ck_w,
                    "cross_k_b": tt_ck_b,
                    "cross_v_w": tt_cv_w,
                    "cross_v_b": tt_cv_b,
                    "cross_out_w": tt_co_w,
                    "cross_out_b": tt_co_b,
                    "ln1_w": tt_ln1_w,
                    "ln1_b": tt_ln1_b,
                    "ln2_w": tt_ln2_w,
                    "ln2_b": tt_ln2_b,
                    "ln3_w": tt_ln3_w,
                    "ln3_b": tt_ln3_b,
                    "mlp_w1": tt_w1,
                    "mlp_b1": tt_b1,
                    "mlp_w2": tt_w2,
                    "mlp_b2": tt_b2,
                }
            )

        self._tt_params = params

    def _apply_activation(self, tt_x):
        act = (self.config.activation or "relu").lower()
        if act in {"relu", "gelu"}:
            # Default to RELU; GELU falls back to RELU for initial bring-up
            try:
                if act == "gelu" and hasattr(ttnn, "gelu"):
                    return ttnn.gelu(tt_x)
            except Exception:
                pass
            return ttnn.relu(tt_x)
        return ttnn.relu(tt_x)
