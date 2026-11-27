# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Gemma-4 text encoder for LTX-2.5.

Forward-only (no KV cache): runs all tokens through the decoder stack and returns
the hidden states, mirroring the Gemma-3 encoder next door.

Gemma-4 keeps Gemma-3's backbone (3840 hidden, 48 layers, 16 heads, 15360 FFN)
but diverges on its ``full_attention`` layers, which use a wider head_dim, a
single KV head, and K=V tying. Model-wide it also drops Gemma-3's ``(1 + weight)``
norm convention and its ``1/sqrt(head_dim)`` attention scale.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import LoadingError, Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate
from ...utils.tracing import StateTensor

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


class Gemma4Config:
    """Configuration for the Gemma-4 text encoder."""

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 3840,
        intermediate_size: int = 15360,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        global_head_dim: int = 512,
        num_global_key_value_heads: int = 1,
        attention_k_eq_v: bool = True,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        global_rope_theta: float = 1000000.0,
        partial_rotary_factor: float = 0.25,
        sliding_window: int = 1024,
        sliding_window_pattern: int = 6,
        layer_types: tuple[str, ...] | None = None,
        max_position_embeddings: int = 262144,
        hidden_layer_index: int = -1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        # full_attention layers project wider heads and collapse to (typically) a
        # single KV head; sliding layers keep the Gemma-3 shape.
        self.global_head_dim = global_head_dim
        self.num_global_key_value_heads = num_global_key_value_heads
        self.attention_k_eq_v = attention_k_eq_v
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.global_rope_theta = global_rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.hidden_layer_index = hidden_layer_index

        if layer_types is None:
            pattern = [SLIDING_ATTENTION] * (sliding_window_pattern - 1) + [FULL_ATTENTION]
            repeated = pattern * (num_hidden_layers // sliding_window_pattern + 1)
            layer_types = tuple(repeated[:num_hidden_layers])
        self.layer_types = tuple(layer_types)

    @classmethod
    def from_hf_text_config(cls, text_config: dict) -> "Gemma4Config":
        """Build from the ``gemma_config.text_config`` block of a packed checkpoint."""
        rope_params = text_config.get("rope_parameters") or {}
        sliding_rope = rope_params.get(SLIDING_ATTENTION) or {}
        full_rope = rope_params.get(FULL_ATTENTION) or {}

        num_global_kv = text_config.get("num_global_key_value_heads")
        if num_global_kv is None:
            num_global_kv = text_config.get("num_key_value_heads", 8)

        return cls(
            vocab_size=text_config["vocab_size"],
            hidden_size=text_config["hidden_size"],
            intermediate_size=text_config["intermediate_size"],
            num_hidden_layers=text_config["num_hidden_layers"],
            num_attention_heads=text_config["num_attention_heads"],
            num_key_value_heads=text_config.get("num_key_value_heads", 8),
            head_dim=text_config.get("head_dim", 256),
            global_head_dim=text_config.get("global_head_dim", 512),
            num_global_key_value_heads=num_global_kv,
            attention_k_eq_v=text_config.get("attention_k_eq_v", False),
            rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
            rope_theta=sliding_rope.get("rope_theta", 10000.0),
            global_rope_theta=full_rope.get("rope_theta", 1000000.0),
            partial_rotary_factor=full_rope.get("partial_rotary_factor", 1.0),
            sliding_window=text_config.get("sliding_window", 1024),
            layer_types=tuple(text_config["layer_types"]) if text_config.get("layer_types") else None,
            max_position_embeddings=text_config.get("max_position_embeddings", 262144),
        )

    def is_global(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == FULL_ATTENTION

    def attn_head_dim(self, is_global: bool) -> int:
        return self.global_head_dim if is_global else self.head_dim

    def attn_kv_heads(self, is_global: bool) -> int:
        return self.num_global_key_value_heads if is_global else self.num_key_value_heads


def gemma4_rms_norm(config: Gemma4Config, mesh_device, dim: int | None = None, *, affine: bool = True) -> RMSNorm:
    """RMSNorm applying the checkpoint weight as-is.

    Gemma-3 stores its norm weights centered at 0 and scales by ``(1 + weight)``;
    Gemma-4 stores them already offset, so folding the ``+1`` here would double it.
    ``affine=False`` gives the scale-free norm the V projection uses — it has no
    weight in the checkpoint at all.
    """
    return RMSNorm(
        embedding_dim=dim if dim is not None else config.hidden_size,
        norm_eps=config.rms_norm_eps,
        norm_elementwise_affine=affine,
        bias=False,
        mesh_device=mesh_device,
    )


class Gemma4RotaryEmbedding(Module):
    """Precompute RoPE cos/sin tables on host, store on device.

    ``partial_rotary_factor < 1`` selects the "proportional" schedule the global
    layers use: only the leading ``factor * head_dim / 2`` frequencies are real
    and the remainder are zero, so those dimensions carry ``cos=1, sin=0`` and
    pass through unrotated while the table stays full ``head_dim`` wide.
    """

    def __init__(
        self,
        mesh_device,
        head_dim: int,
        base: float,
        max_seq_len: int,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.mesh_device = mesh_device

        rope_angles = int(partial_rotary_factor * head_dim // 2)
        # float32 end-to-end, matching the reference tables exactly. Computing the
        # frequencies in float64 moves a handful of entries across a bfloat16 rounding
        # boundary, so the device tables would not be bit-identical to the reference.
        exponents = torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (base**exponents)
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles, dtype=torch.float32)])

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq, D/2)
        self._sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0)
        # seq_len is fixed per pipeline, so the device cos/sin are constant: bind once
        # and keep resident so the whole forward stays traceable.
        self._tt_cos = StateTensor()
        self._tt_sin = StateTensor()

    def forward(self, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if self._tt_cos.value is None:
            # HF-format cos/sin for ttnn.experimental.rotary_embedding_hf: the half-dim
            # freqs are duplicated to full head_dim by concatenation, not interleaving.
            cos = torch.cat([self._cos_cached, self._cos_cached], dim=-1)[:, :, :seq_len, :].bfloat16()
            sin = torch.cat([self._sin_cached, self._sin_cached], dim=-1)[:, :, :seq_len, :].bfloat16()
            self._tt_cos.update(
                ttnn.from_torch(cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), False
            )
            self._tt_sin.update(
                ttnn.from_torch(sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), False
            )
        return self._tt_cos.value, self._tt_sin.value


class Gemma4Attention(Module):
    """Gemma-4 GQA self-attention with RoPE. No KV cache.

    Shapes depend on the layer type: ``full_attention`` layers use
    ``global_head_dim`` with ``num_global_key_value_heads`` KV heads and tie V to
    K; sliding layers keep the ordinary GQA shape with their own V projection.
    """

    def __init__(
        self,
        config: Gemma4Config,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        is_global: bool,
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_global = is_global

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.attn_kv_heads(is_global)
        self.head_dim = config.attn_head_dim(is_global)
        self.hidden_size = config.hidden_size
        # K=V tying applies only where the checkpoint omits v_proj.
        self.k_eq_v = config.attention_k_eq_v and is_global

        tp = parallel_config.tensor_parallel.factor
        self.num_local_heads = self.num_heads // tp
        # Fewer KV heads than devices cannot be fractured, so each device keeps a
        # whole KV head — the one its Q heads map to under GQA.
        self.kv_replicated = self.num_kv_heads < tp
        self.num_local_kv_heads = 1 if self.kv_replicated else self.num_kv_heads // tp

        # FSDP: shard weights on the sequence-parallel axis (gathered per-op).
        sp = parallel_config.sequence_parallel
        fsdp_mesh_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None

        col_kwargs = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
        }
        if fsdp_mesh_axis is not None:
            col_kwargs["fsdp_mesh_axis"] = fsdp_mesh_axis
            col_kwargs["ccl_manager"] = ccl_manager

        # Fused QKV laid out per-TP-shard [q|k|v] so the output splits cleanly with
        # nlp_create_qkv_heads. Replicated KV widens the total: every device carries
        # its own copy of the KV heads.
        qkv_out = tp * (self.num_local_heads + 2 * self.num_local_kv_heads) * self.head_dim
        self.wqkv = ColParallelLinear(self.hidden_size, qkv_out, **col_kwargs)
        # o_proj carries the ccl_manager so its head-input all_gather can fuse into the
        # matmul on Ring; wqkv's input is replicated so it never needs it.
        o_proj_kwargs = dict(col_kwargs)
        o_proj_kwargs.setdefault("ccl_manager", ccl_manager)
        self.o_proj = ColParallelLinear(self.num_heads * self.head_dim, self.hidden_size, **o_proj_kwargs)
        # The fused CCL-matmul ops are a ring-write scheme (4x8 FABRIC_1D_RING); on Linear
        # (2x4) they trip a worker-partitioning assert, so gate them on Ring.
        self._tp_ring = ccl_manager.topology == ttnn.Topology.Ring

        self.input_layernorm = gemma4_rms_norm(config, mesh_device)
        self.post_attention_layernorm = gemma4_rms_norm(config, mesh_device)

        self.q_norm = gemma4_rms_norm(config, mesh_device, dim=self.head_dim)
        self.k_norm = gemma4_rms_norm(config, mesh_device, dim=self.head_dim)
        # V is normalized too, but with no learnable scale.
        self.v_norm = gemma4_rms_norm(config, mesh_device, dim=self.head_dim, affine=False)

        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Fuse q/k/v into one wqkv weight. HF weights are [out, in]; transpose to
        # [in, out], group heads per TP shard, cat q|k|v on the head axis, flatten back
        # so ColParallelLinear column-fracturing hands each shard its local [q|k|v].
        q = pop_substate(state, "q_proj")
        k = pop_substate(state, "k_proj")
        v = pop_substate(state, "v_proj") if not self.k_eq_v else k
        n_dev = self.parallel_config.tensor_parallel.factor
        d = self.head_dim

        def _q_heads(w: torch.Tensor) -> torch.Tensor:
            return w.T.reshape(self.hidden_size, n_dev, self.num_local_heads, d)

        def _kv_heads(w: torch.Tensor) -> torch.Tensor:
            if not self.kv_replicated:
                return w.T.reshape(self.hidden_size, n_dev, self.num_local_kv_heads, d)
            heads = w.T.reshape(self.hidden_size, self.num_kv_heads, d)
            q_per_dev = self.num_heads // n_dev
            assign = [(i * q_per_dev) * self.num_kv_heads // self.num_heads for i in range(n_dev)]
            return heads[:, assign, :].unsqueeze(2)  # [in, n_dev, 1, d]

        qkv = torch.cat([_q_heads(q["weight"]), _kv_heads(k["weight"]), _kv_heads(v["weight"])], dim=2)
        state["wqkv.weight"] = qkv.reshape(self.hidden_size, -1).T.contiguous()

    def forward(self, hidden_states, cos, sin, attn_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        qkv = self.wqkv(hidden_states, compute_kernel_config=self.compute_config)
        qkv = ttnn.reshape(qkv, (qkv.shape[0], 1, qkv.shape[1], qkv.shape[2]))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_local_heads,
            num_kv_heads=self.num_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Per-head RMSNorm before RoPE. V is normalized but never rotated.
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)

        # GQA: expand KV heads to match Q heads. Must use repeat_interleave (each kv head
        # duplicated contiguously) to match HF repeat_kv — ttnn.repeat does block-tile,
        # which mispairs q/kv heads.
        if self.num_local_kv_heads < self.num_local_heads:
            repeats = self.num_local_heads // self.num_local_kv_heads
            k = ttnn.repeat_interleave(k, repeats, dim=1)
            v = ttnn.repeat_interleave(v, repeats, dim=1)

        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        # Gemma-4 folds the query scaling into its trained weights, so attention runs
        # unscaled. TTNN SDPA can't take is_causal and attn_mask together.
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(attn_mask is None),
            attn_mask=attn_mask,
            scale=1.0,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute_config,
        )

        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        attn_output = ttnn.unsqueeze(attn_output, 0)
        tp_gt1 = self.parallel_config.tensor_parallel.factor > 1
        if tp_gt1 and self._tp_ring:
            output = self.o_proj(
                attn_output, compute_kernel_config=self.compute_config, parallel_config=self.parallel_config
            )
        else:
            if tp_gt1:
                attn_output = self.ccl_manager.all_gather(
                    attn_output,
                    dim=3,
                    mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                    use_hyperparams=True,
                )
            output = self.o_proj(attn_output, compute_kernel_config=self.compute_config)
        if tp_gt1:
            output = self.ccl_manager.all_gather(
                output,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        output = ttnn.squeeze(output, 0)

        output = self.post_attention_layernorm(output)
        return output + residual


class Gemma4FF(Module):
    """Gated MLP: down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))."""

    def __init__(
        self,
        config: Gemma4Config,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        sp = parallel_config.sequence_parallel
        fsdp_mesh_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None

        col_kwargs = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
        }
        if fsdp_mesh_axis is not None:
            col_kwargs["fsdp_mesh_axis"] = fsdp_mesh_axis
            col_kwargs["ccl_manager"] = ccl_manager

        self.gate_proj = ColParallelLinear(
            config.hidden_size, config.intermediate_size, activation_fn="gelu_tanh", **col_kwargs
        )
        self.up_proj = ColParallelLinear(config.hidden_size, config.intermediate_size, **col_kwargs)
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.pre_feedforward_layernorm = gemma4_rms_norm(config, mesh_device)
        self.post_feedforward_layernorm = gemma4_rms_norm(config, mesh_device)

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "mlp.gate_proj", "gate_proj")
        rename_substate(state, "mlp.up_proj", "up_proj")
        rename_substate(state, "mlp.down_proj", "down_proj")

    def forward(self, x):
        residual = x
        x = self.pre_feedforward_layernorm(x)

        gate = self.gate_proj(x, compute_kernel_config=self.compute_config)
        up = self.up_proj(x, compute_kernel_config=self.compute_config)
        x = gate * up

        x = self.down_proj(x, compute_kernel_config=self.compute_config)
        x = ttnn.unsqueeze(x, 0)
        if self.parallel_config.tensor_parallel.factor > 1:
            x = self.ccl_manager.all_gather(
                x,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        x = ttnn.squeeze(x, 0)

        x = self.post_feedforward_layernorm(x)
        return x + residual


class Gemma4EncoderLayer(Module):
    """Single Gemma-4 decoder layer used as encoder (no KV cache)."""

    def __init__(self, config, mesh_device, ccl_manager, parallel_config, is_global: bool):
        super().__init__()
        self.self_attn = Gemma4Attention(config, mesh_device, ccl_manager, parallel_config, is_global)
        self.ff = Gemma4FF(config, mesh_device, ccl_manager, parallel_config)
        self.layer_scalar = 1.0

    #: Sidecar for :attr:`layer_scalar`, which is NOT a Parameter and so is invisible to
    #: ``Module.save``/``Module.load``. See :meth:`save`.
    _SCALAR_FILE = "layer_scalar.json"

    def _prepare_torch_state(self, state):
        # A single learned scalar; kept on the host and folded into the output multiply
        # rather than round-tripped through a device Parameter.
        scalar = state.pop("layer_scalar", None)
        if scalar is not None:
            self.layer_scalar = float(scalar.reshape(-1)[0].item())
        rename_substate(state, "input_layernorm", "self_attn.input_layernorm")
        rename_substate(state, "post_attention_layernorm", "self_attn.post_attention_layernorm")
        rename_substate(state, "pre_feedforward_layernorm", "ff.pre_feedforward_layernorm")
        rename_substate(state, "post_feedforward_layernorm", "ff.post_feedforward_layernorm")
        rename_substate(state, "mlp.gate_proj", "ff.gate_proj")
        rename_substate(state, "mlp.up_proj", "ff.up_proj")
        rename_substate(state, "mlp.down_proj", "ff.down_proj")

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        """Weights, plus ``layer_scalar`` beside them.

        ``layer_scalar`` is a host float, not a Parameter, so the base class does not persist it
        and ``load`` does not run ``_prepare_torch_state``. Without this the cached path silently
        kept the constructor's 1.0 for all 48 layers where the checkpoint carries 0.0045-0.93 --
        every layer's residual contribution inflated, the text embedding destroyed, and video that
        ignores its prompt while looking otherwise well-formed. The gate for it is
        tests/unit/test_gemma4_cache_roundtrip.py.
        """
        super().save(directory, prefix=prefix)
        (Path(directory) / f"{prefix}{self._SCALAR_FILE}").write_text(json.dumps(self.layer_scalar))

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)
        path = Path(directory) / f"{prefix}{self._SCALAR_FILE}"
        if not path.exists():
            # Refuse rather than fall back to 1.0: a cache written before this existed holds the
            # right tensors and no scalar, and silently using the default is the whole bug. Deleting
            # the cache directory rebuilds it in about a minute.
            msg = (
                f"cache at '{directory}' predates layer_scalar persistence and would silently drop "
                f"the per-layer scalars. Delete it and let it rebuild"
            )
            raise LoadingError(msg)
        self.layer_scalar = json.loads(path.read_text())

    def forward(self, hidden_states, cos, sin, attn_mask=None):
        hidden_states = self.self_attn(hidden_states, cos, sin, attn_mask=attn_mask)
        hidden_states = self.ff(hidden_states)
        if self.layer_scalar != 1.0:
            hidden_states = ttnn.multiply(hidden_states, self.layer_scalar)
        return hidden_states


class Gemma4Encoder(Module):
    """
    TTNN Gemma-4 text encoder.

    Runs the full decoder stack and returns the list of all hidden states
    (input embedding, each decoder layer, then the final norm). Layer selection
    is the caller's responsibility.
    """

    def __init__(
        self,
        config: Gemma4Config,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, device=mesh_device)
        # One rope table per layer type: the global layers use the proportional
        # schedule over the wider head_dim, the sliding layers a plain one.
        self.rotary_emb_global = Gemma4RotaryEmbedding(
            mesh_device,
            head_dim=config.global_head_dim,
            base=config.global_rope_theta,
            max_seq_len=max_seq_len,
            partial_rotary_factor=config.partial_rotary_factor,
        )
        self.rotary_emb_local = Gemma4RotaryEmbedding(
            mesh_device,
            head_dim=config.head_dim,
            base=config.rope_theta,
            max_seq_len=max_seq_len,
        )

        self.layers = ModuleList(
            Gemma4EncoderLayer(config, mesh_device, ccl_manager, parallel_config, config.is_global(idx))
            for idx in range(config.num_hidden_layers)
        )

        self.norm = gemma4_rms_norm(config, mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # The packed LTX checkpoint stores the text stack flat under model.*, alongside
        # the multimodal towers and the LTX-side projection, which the encoder ignores.
        prefix = "model."
        stripped = {}
        for k, v in list(state.items()):
            if k.startswith(prefix):
                stripped[k[len(prefix) :]] = v
                del state[k]
        state.update(stripped)

        pop_substate(state, "lm_head")
        for unused in [
            "vision_model",
            "vision_tower",
            "audio_projector",
            "multi_modal_projector",
            "text_embedding_projection",
        ]:
            pop_substate(state, unused)
        for sidecar in [k for k in state if k.startswith("hf_asset__") or k == "tokenizer_json"]:
            del state[sidecar]

    def build_attn_mask(self, attention_mask, seq_len: int) -> ttnn.Tensor | None:
        """Host-build the combined causal+padding SDPA mask (B,1,seq,seq) as a device tensor.
        Kept out of the traced forward: per-prompt host work whose output the trace copies in.
        TTNN SDPA can't take is_causal + attn_mask together, so causal and padding are merged."""
        if attention_mask is None:
            return None
        mask_host = (
            attention_mask
            if isinstance(attention_mask, torch.Tensor)
            else ttnn.to_torch(ttnn.get_device_tensors(attention_mask)[0])
        )
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)[None, None, :, :]
        pad_mask = torch.where(mask_host[:, None, None, :].bool(), 0.0, float("-inf"))
        combined = causal + pad_mask  # broadcasts to (B, 1, seq, seq)
        return ttnn.from_torch(
            combined.bfloat16(),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, token_ids: ttnn.Tensor, *, tt_attn_mask: ttnn.Tensor | None = None) -> list[ttnn.Tensor]:
        """Embed → decoder stack → final norm. Returns [embed, L0..L47, final_norm] on device.
        Pure device graph (mask pre-built, RoPE cos/sin resident) so the forward replays
        as one ttnn trace."""
        seq_len = token_ids.shape[-1]
        # The sliding layers are run as plain causal attention, which only matches the
        # reference while every query can see the whole prefix.
        if seq_len > self.config.sliding_window:
            raise NotImplementedError(
                f"prompt length {seq_len} exceeds the {self.config.sliding_window}-token sliding window; "
                "the sliding layers would need a windowed mask"
            )

        hidden_states = self.embed_tokens(token_ids)
        hidden_states = ttnn.multiply(hidden_states, math.sqrt(self.config.hidden_size))

        cos_g, sin_g = self.rotary_emb_global(seq_len)
        cos_l, sin_l = self.rotary_emb_local(seq_len)

        all_hidden_states = [hidden_states]
        for idx, layer in enumerate(self.layers):
            cos, sin = (cos_g, sin_g) if self.config.is_global(idx) else (cos_l, sin_l)
            hidden_states = layer(hidden_states, cos, sin, attn_mask=tt_attn_mask)
            all_hidden_states.append(hidden_states)

        output = self.norm(hidden_states)
        all_hidden_states.append(output)
        return all_hidden_states
