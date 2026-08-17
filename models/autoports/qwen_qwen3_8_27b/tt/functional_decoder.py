# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for ``Qwen/Qwen3.8-27B``.

The checkpoint uses the Qwen3.5 hybrid decoder architecture despite its public
Qwen3.8 name.  Layers whose ``layer_types[layer_idx]`` value is
``linear_attention`` use a fixed-size Gated DeltaNet recurrent/causal-conv
state.  ``full_attention`` layers use a caller-owned paged KV cache.

Runtime contract
----------------
``prefill_forward`` accepts an on-device BF16 tile tensor ``[B, S, 5120]``.
Full-attention callers also supply on-device RoPE ``cos``/``sin``, an int32
logical ``page_table`` and the chunk's int32 ``chunk_page_table``. DeltaNet
prefill is processed in 128-token chunks; ``logical_seq_len`` masks a padded
last chunk and the returned tensor is sliced back to the logical length.

``decode_forward`` accepts an on-device BF16 tile tensor ``[B, 1, 5120]``.
For full attention, ``current_position`` is an on-device int32 tensor ``[B]``
and ``page_table`` is the on-device logical-to-physical block map.  These
device tensors are stable trace inputs: callers update their contents before
``ttnn.execute_trace`` rather than replacing their buffers. DeltaNet decode
updates its fixed-address recurrent and convolution state on device and does
not use a page table.

Weight conversion and all host work are confined to ``from_state_dict`` and
test/input preparation. Neither forward method imports torch nor invokes
``ttnn.from_torch``/``ttnn.to_torch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_attention import (
    _get_paged_sdpa_decode_program_config,
    apply_rotary_pos_emb_ttnn,
)

_GDN_CHUNK = 128


@dataclass(frozen=True)
class DecoderContract:
    layer_kind: str
    hidden_size: int
    max_context: int
    page_block_size: int
    gdn_chunk_size: int = _GDN_CHUNK


def _text_config(hf_config):
    return hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config


def _internal_state_dict(state_dict, *, config, layer_idx):
    """Normalize canonical HF layer keys to the existing TTNN kernel contract."""
    prefix_options = (
        f"model.language_model.layers.{layer_idx}.",
        f"model.layers.{layer_idx}.",
        f"layers.{layer_idx}.",
    )
    local = {}
    for key, value in state_dict.items():
        short = key
        for prefix in prefix_options:
            if key.startswith(prefix):
                short = key[len(prefix) :]
                break
        local[short] = value

    layer_prefix = f"layers.{layer_idx}."
    out = {layer_prefix + key: value for key, value in local.items() if not key.startswith("layers.")}
    # Also accept an already-normalized state dict.
    out.update({key: value for key, value in state_dict.items() if key.startswith(layer_prefix)})

    qkv_key = layer_prefix + "linear_attn.in_proj_qkv.weight"
    conv_key = layer_prefix + "linear_attn.conv1d.weight"
    if qkv_key in out:
        out[layer_prefix + "linear_attn.qkv_proj.weight"] = out.pop(qkv_key)
    if conv_key in out:
        conv = out.pop(conv_key)
        q_dim = int(config.linear_num_key_heads * config.linear_key_head_dim)
        k_dim = q_dim
        out[layer_prefix + "linear_attn.q_conv.weight"] = conv[:q_dim]
        out[layer_prefix + "linear_attn.k_conv.weight"] = conv[q_dim : q_dim + k_dim]
        out[layer_prefix + "linear_attn.v_conv.weight"] = conv[q_dim + k_dim :]
    return out


def _kernel_args(config, *, layer_idx, mesh_device, max_context):
    rope = config.rope_parameters or {}
    return SimpleNamespace(
        dim=int(config.hidden_size),
        hidden_dim=int(config.intermediate_size),
        n_heads=int(config.num_attention_heads),
        n_kv_heads=int(config.num_key_value_heads),
        head_dim=int(config.head_dim),
        norm_eps=float(config.rms_norm_eps),
        max_seq_len=int(max_context),
        max_batch_size=32,
        num_devices=1,
        linear_num_key_heads=int(config.linear_num_key_heads),
        linear_num_value_heads=int(config.linear_num_value_heads),
        linear_key_head_dim=int(config.linear_key_head_dim),
        linear_value_head_dim=int(config.linear_value_head_dim),
        linear_conv_kernel_dim=int(config.linear_conv_kernel_dim),
        linear_q_dim=int(config.linear_num_key_heads * config.linear_key_head_dim),
        linear_k_dim=int(config.linear_num_key_heads * config.linear_key_head_dim),
        linear_v_dim=int(config.linear_num_value_heads * config.linear_value_head_dim),
        rope_head_dim=int(config.head_dim * rope.get("partial_rotary_factor", 0.25)),
        rope_theta=float(rope.get("rope_theta", 10_000_000.0)),
        attention_type_list=list(config.layer_types),
        is_full_attention_layer=lambda idx: config.layer_types[idx] == "full_attention",
        is_deltanet_layer=lambda idx: config.layer_types[idx] == "linear_attention",
        mlp_1d_decode=False,
        prefill_progcfg=None,
    )


class FunctionalDecoder(LightweightModule):
    """One target-shape Qwen3.8 decoder layer, parameterized by layer kind."""

    def __init__(self, layer, *, contract, layer_idx, mesh_device):
        self.layer = layer
        self.contract = contract
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self._gdn_forward = None
        if contract.layer_kind == "linear_attention":
            self._gdn_forward = layer.attention.forward
            layer.attention.forward = self._gdn_forward_with_dram_state
        else:
            # The shared experimental paged-decode path uses transpose(1, 2),
            # which aliases token and batch only for B=1.  Keep the correction
            # autoport-local: this instance override implements the same paged
            # decode boundary with explicit [B,H,1,D] <-> [1,B,H,D] permutes.
            self._full_attention_forward = layer.attention.forward
            layer.attention.forward = self._full_attention_forward_with_batched_paged_decode

    def _full_attention_forward_with_batched_paged_decode(
        self,
        x,
        cos,
        sin,
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
    ):
        attention = self.layer.attention
        batch = int(x.shape[0])
        if attention.use_paged_attention and int(x.shape[1]) > 1 and batch > 1:
            # The shared paged-prefill helper hard-codes ``batch_idx=0`` in
            # paged_fill_cache.  Preserve a batched public contract by issuing
            # one device-only row fill/SDPA per user, each with its own logical
            # and chunk page-table row, then concatenate the device outputs.
            outputs = []
            for user in range(batch):
                user_x = ttnn.slice(x, (user, 0, 0), (user + 1, x.shape[1], x.shape[2]))
                user_page_table = ttnn.slice(page_table, (user, 0), (user + 1, page_table.shape[1]))
                user_chunk_page_table = ttnn.slice(chunk_page_table, (user, 0), (user + 1, chunk_page_table.shape[1]))
                user_cos = cos
                user_sin = sin
                if int(cos.shape[0]) == batch:
                    user_cos = ttnn.slice(cos, (user, 0, 0), (user + 1, cos.shape[1], cos.shape[2]))
                    user_sin = ttnn.slice(sin, (user, 0, 0), (user + 1, sin.shape[1], sin.shape[2]))
                outputs.append(
                    self._full_attention_forward(
                        user_x,
                        user_cos,
                        user_sin,
                        position_tensor=position_tensor,
                        page_table=user_page_table,
                        chunk_page_table=user_chunk_page_table,
                        chunk_start_idx=chunk_start_idx,
                        chunk_start_idx_tensor=chunk_start_idx_tensor,
                    )
                )
            return ttnn.concat(outputs, dim=0)

        if not (attention.use_paged_attention and int(x.shape[1]) == 1):
            return self._full_attention_forward(
                x,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )

        config, weights = attention.config, attention.weights
        qg = ttnn.linear(
            x,
            weights.q_proj,
            compute_kernel_config=attention.compute_kernel_config_decode,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        qg = ttnn.reshape(qg, [batch, 1, config.num_heads, config.head_dim * 2])
        query, gate = ttnn.chunk(qg, 2, dim=-1)
        ttnn.deallocate(qg)
        gate = ttnn.reshape(gate, [batch, 1, config.num_heads * config.head_dim])

        # The loader stores the already offset (+1) Q/K norm weights, matching
        # the shared attention path's ``norm_weights_pre_offset=True`` branch.
        query = ttnn.rms_norm(query, weight=weights.q_norm, epsilon=config.norm_eps)
        query = ttnn.transpose(query, 1, 2)
        key = ttnn.linear(
            x,
            weights.k_proj,
            compute_kernel_config=attention.compute_kernel_config_decode,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        key = ttnn.reshape(key, [batch, 1, config.num_kv_heads, config.head_dim])
        key = ttnn.rms_norm(key, weight=weights.k_norm, epsilon=config.norm_eps)
        key = ttnn.transpose(key, 1, 2)
        value = ttnn.linear(
            x,
            weights.v_proj,
            compute_kernel_config=attention.compute_kernel_config_decode,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        value = ttnn.reshape(value, [batch, 1, config.num_kv_heads, config.head_dim])
        value = ttnn.transpose(value, 1, 2)

        cos_4d = ttnn.reshape(cos, [cos.shape[0], 1, 1, cos.shape[-1]])
        sin_4d = ttnn.reshape(sin, [sin.shape[0], 1, 1, sin.shape[-1]])
        query, key = apply_rotary_pos_emb_ttnn(query, key, cos_4d, sin_4d)

        key_update = ttnn.reshape(key, [1, batch, config.num_kv_heads, config.head_dim])
        value_update = ttnn.reshape(value, [1, batch, config.num_kv_heads, config.head_dim])
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(batch - 1, 0))})
        shard_spec = ttnn.ShardSpec(shard_grid, [32, config.head_dim], ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        key_update = ttnn.to_memory_config(key_update, sharded_mc)
        value_update = ttnn.to_memory_config(value_update, sharded_mc)
        ttnn.experimental.paged_update_cache(
            attention.paged_kv_cache_key,
            key_update,
            update_idxs_tensor=position_tensor,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            attention.paged_kv_cache_value,
            value_update,
            update_idxs_tensor=position_tensor,
            page_table=page_table,
        )

        # Native paged SDPA decode is sequence-major: [1, B, H, D].
        query_decode = ttnn.permute(query, (2, 0, 1, 3))
        query_decode = ttnn.to_memory_config(query_decode, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query_decode,
            attention.paged_kv_cache_key,
            attention.paged_kv_cache_value,
            cur_pos_tensor=position_tensor,
            page_table_tensor=page_table,
            is_causal=True,
            scale=config.head_dim**-0.5,
            program_config=_get_paged_sdpa_decode_program_config(
                self.mesh_device,
                attention.paged_kv_cache_key.shape[0] * attention.paged_kv_cache_key.shape[2],
            ),
        )
        output = ttnn.permute(output, (1, 2, 0, 3))
        output = ttnn.transformer.concatenate_heads(output)
        output = ttnn.multiply(output, ttnn.sigmoid(gate))
        return ttnn.linear(
            output,
            weights.o_proj,
            compute_kernel_config=attention.compute_kernel_config_decode,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _gdn_forward_with_dram_state(self, *args, **kwargs):
        """Keep persistent decode state out of the small interleaved L1 arena."""
        output = self._gdn_forward(*args, **kwargs)
        attention = self.layer.attention
        for name in ("recurrent_state", "fused_conv_state"):
            state = getattr(attention, name)
            if state is not None and state.memory_config().buffer_type == ttnn.BufferType.L1:
                dram_state = ttnn.to_memory_config(state, ttnn.DRAM_MEMORY_CONFIG)
                setattr(attention, name, dram_state)
                ttnn.deallocate(state)
        return output

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        tensor_cache_path: str | Path | None = None,
        max_context: int | None = None,
        page_block_size: int = 64,
        **_kwargs,
    ):
        config = _text_config(hf_config)
        advertised_context = int(config.max_position_embeddings)
        supported_context = advertised_context if max_context is None else int(max_context)
        if not 0 < supported_context <= advertised_context:
            raise ValueError(f"max_context must be in [1, {advertised_context}], got {supported_context}")
        kind = config.layer_types[layer_idx]
        if kind not in {"linear_attention", "full_attention"}:
            raise ValueError(f"Unsupported Qwen3.8 layer kind: {kind}")
        normalized = _internal_state_dict(state_dict, config=config, layer_idx=layer_idx)
        args = _kernel_args(config, layer_idx=layer_idx, mesh_device=mesh_device, max_context=supported_context)
        cache_path = Path(tensor_cache_path) if tensor_cache_path is not None else None
        layer = Qwen36DecoderLayer(mesh_device, args, normalized, layer_idx, tensor_cache_path=cache_path)
        contract = DecoderContract(kind, int(config.hidden_size), supported_context, int(page_block_size))
        return cls(layer, contract=contract, layer_idx=layer_idx, mesh_device=mesh_device)

    @property
    def layer_kind(self):
        return self.contract.layer_kind

    def allocate_runtime_state(self, *, batch_size=1, num_physical_blocks=None, cache_dtype=ttnn.bfloat16):
        """Allocate persistent cache/state before warmup or trace capture."""
        if self.layer_kind == "linear_attention":
            self.layer.attention.reset_state(batch_size)
            return None
        if num_physical_blocks is None:
            num_physical_blocks = (
                self.contract.max_context + self.contract.page_block_size - 1
            ) // self.contract.page_block_size
        shape = (
            int(num_physical_blocks),
            self.layer.args.n_kv_heads,
            self.contract.page_block_size,
            self.layer.args.head_dim,
        )
        key = ttnn.zeros(shape, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        value = ttnn.zeros(shape, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        self.layer.attention.set_paged_kv_cache(key, value)
        return key, value

    def prefill_forward(
        self,
        hidden_states,
        *,
        cos=None,
        sin=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=0,
        chunk_start_idx_tensor=None,
        logical_seq_len=None,
    ):
        seq_len = int(hidden_states.shape[1] if logical_seq_len is None else logical_seq_len)
        if not 0 < seq_len <= self.contract.max_context:
            raise ValueError(f"logical_seq_len must be in [1, {self.contract.max_context}], got {seq_len}")
        if self.layer_kind == "full_attention":
            if cos is None or sin is None or page_table is None or chunk_page_table is None:
                raise ValueError("full_attention prefill requires cos, sin, page_table, and chunk_page_table")
            return self.layer.forward(
                hidden_states,
                cos=cos,
                sin=sin,
                mode="prefill",
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )

        padded_len = ((seq_len + _GDN_CHUNK - 1) // _GDN_CHUNK) * _GDN_CHUNK
        if int(hidden_states.shape[1]) < padded_len:
            hidden_states = ttnn.pad(
                hidden_states,
                [(0, 0), (0, padded_len - int(hidden_states.shape[1])), (0, 0)],
                value=0.0,
            )
        outputs = []
        for start in range(0, padded_len, _GDN_CHUNK):
            chunk = ttnn.slice(
                hidden_states, (0, start, 0), (hidden_states.shape[0], start + _GDN_CHUNK, hidden_states.shape[2])
            )
            valid_len = min(_GDN_CHUNK, seq_len - start)
            outputs.append(
                self.layer.forward(chunk, mode="prefill", chunk_size=_GDN_CHUNK, valid_len=max(valid_len, 0))
            )
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
        return (
            output
            if seq_len == padded_len
            else ttnn.slice(output, (0, 0, 0), (output.shape[0], seq_len, output.shape[2]))
        )

    def decode_forward(self, hidden_states, *, cos=None, sin=None, current_position=None, page_table=None):
        if int(hidden_states.shape[1]) != 1:
            raise ValueError(f"decode requires one token, got shape {tuple(hidden_states.shape)}")
        if self.layer_kind == "full_attention":
            if cos is None or sin is None or current_position is None or page_table is None:
                raise ValueError("full_attention decode requires cos, sin, current_position, and page_table")
            return self.layer.forward(
                hidden_states,
                cos=cos,
                sin=sin,
                mode="decode",
                position_tensor=current_position,
                page_table=page_table,
            )
        return self.layer.forward(hidden_states, mode="decode")

    def enable_trace_safe_state_updates(self):
        """Keep DeltaNet state buffer addresses stable across decode trace replay."""
        if self.layer_kind == "linear_attention":
            attention = self.layer.attention
            if attention.recurrent_state is None or (
                attention.split_conv_state is None and attention.fused_conv_state is None
            ):
                raise RuntimeError("run prefill and one eager decode compile pass before enabling traced state updates")
            attention.use_inplace_state = True

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
