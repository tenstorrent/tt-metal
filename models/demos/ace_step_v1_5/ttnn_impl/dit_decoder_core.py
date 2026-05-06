from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._ttnn import get_ttnn


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl")
    return ttnn


@dataclass(frozen=True)
class AceStepDecoderConfigTTNN:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    sliding_window: Optional[int]


def _maybe_get(state_dict: dict, key: str) -> np.ndarray:
    if key not in state_dict:
        raise KeyError(f"Missing key in state_dict: {key}")
    return state_dict[key]


class TtTimestepEmbedding:
    """
    TTNN version of `TimestepEmbedding`, implemented as a precomputed lookup.

    We precompute (on host) the exact same sinusoidal `t_freq` for a fixed set
    of timesteps and keep the result device-resident. This avoids requiring
    sin/cos/exp kernels for bring-up.
    """

    def __init__(
        self,
        *,
        cfg: AceStepDecoderConfigTTNN,
        state_dict: dict,
        base_address: str,  # e.g. "time_embed" or "time_embed_r"
        mesh_device,
        timesteps_host: np.ndarray,  # shape [N]
        scale: float = 1000.0,
        dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.hidden_size = int(cfg.hidden_size)
        self.scale = float(scale)

        self.in_channels = 256
        self.time_embed_dim = int(cfg.hidden_size)
        if timesteps_host.ndim != 1:
            raise ValueError("timesteps_host must be rank-1 [N]")
        self.num_steps = int(timesteps_host.shape[0])

        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        # Build sinusoidal embeddings on host (float32).
        t = timesteps_host.astype(np.float32) * self.scale  # [N]
        half = self.in_channels // 2
        freqs = np.exp((-math.log(10000.0)) * (np.arange(0, half, dtype=np.float32) / float(half)))  # [half]
        args = t[:, None] * freqs[None, :]  # [N, half]
        emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)  # [N, in_channels]

        # Load weights (host arrays) and transfer once.
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(key: str):
            return ttnn.as_tensor(
                _maybe_get(state_dict, f"{base_address}.{key}.weight"),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        def as_b(key: str):
            b = _maybe_get(state_dict, f"{base_address}.{key}.bias")
            return ttnn.as_tensor(
                b.reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.w1, self.b1 = as_w("linear_1"), as_b("linear_1")
        self.w2, self.b2 = as_w("linear_2"), as_b("linear_2")
        self.wt, self.bt = as_w("time_proj"), as_b("time_proj")

        self.t_freq_table = ttnn.as_tensor(
            emb.reshape(self.num_steps, 1, 1, self.in_channels),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

    def __call__(self, timestep_index: int):
        ttnn = self.ttnn
        if not (0 <= int(timestep_index) < self.num_steps):
            raise ValueError(f"timestep_index out of range: {timestep_index} not in [0,{self.num_steps})")

        # Slice out [1,1,1,in_channels]
        t_freq = ttnn.slice(self.t_freq_table, (timestep_index, 0, 0, 0), (timestep_index + 1, 1, 1, self.in_channels))
        temb = ttnn.linear(t_freq, self.w1, bias=self.b1, transpose_b=True)
        temb = ttnn.silu(temb) if hasattr(ttnn, "silu") else ttnn.gelu(temb)
        temb = ttnn.linear(temb, self.w2, bias=self.b2, transpose_b=True)

        # time_proj(act2(temb)) -> [1,1,1,6*D] -> reshape to [1,6,1,D] then [1,6,D]
        h = ttnn.silu(temb) if hasattr(ttnn, "silu") else ttnn.gelu(temb)
        tp = ttnn.linear(h, self.wt, bias=self.bt, transpose_b=True)
        d = self.time_embed_dim
        tp = ttnn.reshape(tp, (1, 6, 1, d))
        tp = ttnn.reshape(tp, (1, 6, d))
        temb = ttnn.reshape(temb, (1, d))
        return temb, tp


class TtAceStepAttentionSDPA:
    """
    SDPA-based attention for AceStep DiT.

    Shapes:
      - hidden_states: [B, 1, S, D]
      - encoder_hidden_states (cross): [B, 1, S_enc, D]
    """

    def __init__(self, *, cfg: AceStepDecoderConfigTTNN, state_dict: dict, base_address: str, mesh_device, dtype=None):
        ttnn = _require_ttnn()
        transformer = getattr(ttnn, "transformer", None)
        sdpa = getattr(transformer, "scaled_dot_product_attention", None) if transformer is not None else None
        if sdpa is None:
            raise RuntimeError("TTNN build missing ttnn.transformer.scaled_dot_product_attention")

        self.ttnn = ttnn
        self._sdpa = sdpa
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        self.d_model = int(cfg.hidden_size)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.d_head = int(cfg.head_dim)
        self.scale = 1.0 / math.sqrt(float(self.d_head))

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(suffix: str):
            return ttnn.as_tensor(
                _maybe_get(state_dict, f"{base_address}.{suffix}.weight"),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        def as_b(suffix: str):
            # Biases are optional (AceStepConfig.attention_bias defaults to False).
            key = f"{base_address}.{suffix}.bias"
            b = state_dict.get(key, None)
            if b is None:
                return None
            return ttnn.as_tensor(
                b.reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.wq, self.bq = as_w("q_proj"), as_b("q_proj")
        self.wk, self.bk = as_w("k_proj"), as_b("k_proj")
        self.wv, self.bv = as_w("v_proj"), as_b("v_proj")
        self.wo, self.bo = as_w("o_proj"), as_b("o_proj")

        # Per-head RMSNorm weights (shape [Dh]).
        qn = _maybe_get(state_dict, f"{base_address}.q_norm.weight")
        kn = _maybe_get(state_dict, f"{base_address}.k_norm.weight")
        self.q_norm_w = ttnn.as_tensor(
            qn,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.k_norm_w = ttnn.as_tensor(
            kn,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.eps = float(cfg.rms_norm_eps)

    def __call__(self, hidden_states, *, encoder_hidden_states=None, is_causal: bool = False):
        ttnn = self.ttnn
        x = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)  # [B,1,S,D]
        q = ttnn.linear(x, self.wq, bias=self.bq, transpose_b=True)
        if encoder_hidden_states is None:
            k = ttnn.linear(x, self.wk, bias=self.bk, transpose_b=True)
            v = ttnn.linear(x, self.wv, bias=self.bv, transpose_b=True)
        else:
            enc = ttnn.to_layout(encoder_hidden_states, ttnn.TILE_LAYOUT)
            k = ttnn.linear(enc, self.wk, bias=self.bk, transpose_b=True)
            v = ttnn.linear(enc, self.wv, bias=self.bv, transpose_b=True)

        B = int(q.shape[0])
        S = int(q.shape[2])
        H = self.n_heads
        Dh = self.d_head

        # Q: [B,1,S,H*Dh] -> [B,H,S,Dh]
        q = ttnn.reshape(q, (B, 1, S, H, Dh))
        q = ttnn.permute(q, (0, 3, 2, 4, 1))
        q = ttnn.reshape(q, (B, H, S, Dh))

        # K/V use num_key_value_heads; we expand to num_attention_heads by repeat if needed.
        S_k = int(k.shape[2])
        kv_h = self.n_kv
        k = ttnn.reshape(k, (B, 1, S_k, kv_h, Dh))
        v = ttnn.reshape(v, (B, 1, S_k, kv_h, Dh))
        k = ttnn.permute(k, (0, 3, 2, 4, 1))
        v = ttnn.permute(v, (0, 3, 2, 4, 1))
        k = ttnn.reshape(k, (B, kv_h, S_k, Dh))
        v = ttnn.reshape(v, (B, kv_h, S_k, Dh))

        if kv_h != H:
            # Grouped-query attention: repeat kv heads to match q heads.
            if H % kv_h != 0:
                raise ValueError(f"num_attention_heads {H} not divisible by num_key_value_heads {kv_h}")
            rep = H // kv_h
            k = ttnn.repeat(k, (1, rep, 1, 1)) if hasattr(ttnn, "repeat") else ttnn.repeat_interleave(k, rep, dim=1)
            v = ttnn.repeat(v, (1, rep, 1, 1)) if hasattr(ttnn, "repeat") else ttnn.repeat_interleave(v, rep, dim=1)

        # Head-dim RMSNorm on q and k.
        q = ttnn.rms_norm(
            q, weight=self.q_norm_w, epsilon=self.eps, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        )
        k = ttnn.rms_norm(
            k, weight=self.k_norm_w, epsilon=self.eps, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        )

        ctx = self._sdpa(q, k, v, attn_mask=None, is_causal=is_causal, scale=self.scale)
        # [B,H,S,Dh] -> [B,1,S,H*Dh]
        ctx = ttnn.permute(ctx, (0, 2, 1, 3))
        ctx = ttnn.reshape(ctx, (B, 1, S, H * Dh))
        out = ttnn.linear(ctx, self.wo, bias=self.bo, transpose_b=True)
        return out


class TtQwen3MLP:
    """
    Qwen3-style gated MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self, *, state_dict: dict, base_address: str, mesh_device, hidden_size: int, intermediate_size: int, dtype=None
    ):
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        def as_w(name: str):
            return ttnn.as_tensor(
                _maybe_get(state_dict, f"{base_address}.{name}.weight"),
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self.w_gate = as_w("gate_proj")
        self.w_up = as_w("up_proj")
        self.w_down = as_w("down_proj")

        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)

    def __call__(self, x):
        ttnn = self.ttnn
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        gate = ttnn.linear(x, self.w_gate, bias=None, transpose_b=True)
        up = ttnn.linear(x, self.w_up, bias=None, transpose_b=True)
        gate = ttnn.silu(gate) if hasattr(ttnn, "silu") else ttnn.gelu(gate)
        h = ttnn.multiply(gate, up)
        out = ttnn.linear(h, self.w_down, bias=None, transpose_b=True)
        return out


class TtAceStepDiTLayer:
    """
    TTNN port of `AceStepDiTLayer` (self-attn + cross-attn + gated MLP) with modulation.
    """

    def __init__(
        self,
        *,
        cfg: AceStepDecoderConfigTTNN,
        state_dict: dict,
        layer_idx: int,
        mesh_device,
        dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        d = int(cfg.hidden_size)
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        # Norm weights
        self.self_norm_w = ttnn.as_tensor(
            _maybe_get(state_dict, f"layers.{layer_idx}.self_attn_norm.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.cross_norm_w = ttnn.as_tensor(
            _maybe_get(state_dict, f"layers.{layer_idx}.cross_attn_norm.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.mlp_norm_w = ttnn.as_tensor(
            _maybe_get(state_dict, f"layers.{layer_idx}.mlp_norm.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.eps = float(cfg.rms_norm_eps)

        # Attention modules
        self.self_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.self_attn",
            mesh_device=mesh_device,
            dtype=self.dtype,
        )
        self.cross_attn = TtAceStepAttentionSDPA(
            cfg=cfg,
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.cross_attn",
            mesh_device=mesh_device,
            dtype=self.dtype,
        )

        # MLP sizes (from config.json; store in state dict as well, but we pass explicit)
        # Keys exist under layers.{i}.mlp.(gate_proj/up_proj/down_proj).weight
        gate_w = _maybe_get(state_dict, f"layers.{layer_idx}.mlp.gate_proj.weight")
        intermediate = int(gate_w.shape[0])
        self.mlp = TtQwen3MLP(
            state_dict=state_dict,
            base_address=f"layers.{layer_idx}.mlp",
            mesh_device=mesh_device,
            hidden_size=d,
            intermediate_size=intermediate,
            dtype=self.dtype,
        )

        # Scale-shift table: [1,6,D]
        sst = _maybe_get(state_dict, f"layers.{layer_idx}.scale_shift_table")
        self.scale_shift_table = ttnn.as_tensor(
            sst,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

    def __call__(self, hidden_states, timestep_proj_b6d, encoder_hidden_states):
        """
        Args:
            hidden_states: [B, 1, S, D] TILE/ROW_MAJOR ok
            timestep_proj_b6d: [B, 6, D] row-major
            encoder_hidden_states: [B, 1, S_enc, D]
        """
        ttnn = self.ttnn
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        b = int(hidden_states.shape[0])

        temb = timestep_proj_b6d
        if tuple(temb.shape) != (b, 6, int(hidden_states.shape[-1])):
            raise ValueError(f"Expected timestep_proj [B,6,D], got {tuple(temb.shape)}")

        # (scale_shift_table + temb) -> chunk 6 along dim=1 => each [B,1,D]
        sst = ttnn.add(self.scale_shift_table, temb)
        # reshape to [B,6,1,D] then slice
        d = int(hidden_states.shape[-1])
        sst4 = ttnn.reshape(sst, (b, 6, 1, d))

        def chunk(i: int):
            c = ttnn.slice(sst4, (0, i, 0, 0), (b, i + 1, 1, d))
            return ttnn.reshape(c, (b, 1, 1, d))

        shift_msa = chunk(0)
        scale_msa = chunk(1)
        gate_msa = chunk(2)
        c_shift = chunk(3)
        c_scale = chunk(4)
        c_gate = chunk(5)

        # Self-attn AdaLN: norm(x) * (1+scale) + shift
        x_norm = ttnn.rms_norm(
            hidden_states,
            weight=self.self_norm_w,
            epsilon=self.eps,
            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        )
        ones = ttnn.ones_like(scale_msa)
        one_plus = ttnn.add(scale_msa, ones)
        h = ttnn.add(ttnn.multiply(x_norm, one_plus), shift_msa)

        attn_out = self.self_attn(h, encoder_hidden_states=None, is_causal=False)
        gated = ttnn.multiply(attn_out, gate_msa)
        hidden_states = ttnn.add(hidden_states, gated)

        # Cross-attn
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        x2 = ttnn.rms_norm(
            hidden_states,
            weight=self.cross_norm_w,
            epsilon=self.eps,
            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        )
        ca = self.cross_attn(x2, encoder_hidden_states=encoder_hidden_states, is_causal=False)
        hidden_states = ttnn.add(hidden_states, ca)

        # MLP AdaLN + gated residual
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        x3 = ttnn.rms_norm(
            hidden_states,
            weight=self.mlp_norm_w,
            epsilon=self.eps,
            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
        )
        ones2 = ttnn.ones_like(c_scale)
        one_plus2 = ttnn.add(c_scale, ones2)
        h3 = ttnn.add(ttnn.multiply(x3, one_plus2), c_shift)
        ff = self.mlp(h3)
        ff = ttnn.multiply(ff, c_gate)
        hidden_states = ttnn.add(hidden_states, ff)
        return hidden_states


class TtAceStepDiTCore:
    """
    Decoder core for `AceStepDiTModel` (proj_in already handled elsewhere).
    """

    def __init__(
        self,
        *,
        cfg: AceStepDecoderConfigTTNN,
        state_dict: dict,
        mesh_device,
        dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.mesh_device = mesh_device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None

        # condition_embedder: Linear(condition_dim -> hidden_size)
        self.cond_w = ttnn.as_tensor(
            _maybe_get(state_dict, "condition_embedder.weight"),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        self.cond_b = ttnn.as_tensor(
            _maybe_get(state_dict, "condition_embedder.bias").reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        self.layers: List[TtAceStepDiTLayer] = [
            TtAceStepDiTLayer(cfg=cfg, state_dict=state_dict, layer_idx=i, mesh_device=mesh_device, dtype=self.dtype)
            for i in range(int(cfg.num_hidden_layers))
        ]

    def __call__(self, hidden_states_patches, timestep_proj_b6d, encoder_hidden_states):
        ttnn = self.ttnn
        # encoder_hidden_states: [B, S_enc, cond_dim] row-major -> [B,1,S_enc,D]
        enc = ttnn.to_layout(encoder_hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        enc = ttnn.unsqueeze(enc, 1)
        enc = ttnn.to_layout(enc, ttnn.TILE_LAYOUT)
        enc = ttnn.linear(enc, self.cond_w, bias=self.cond_b, transpose_b=True)

        x = ttnn.to_layout(hidden_states_patches, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.unsqueeze(x, 1)  # [B,1,S,D]
        for layer in self.layers:
            x = layer(x, timestep_proj_b6d, enc)
        x = ttnn.squeeze(x, 1)  # [B,S,D]
        return x
