# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One Mistral-Medium-3.5 decoder layer. Ported from ``gpt_oss_d_p/tt/layer.py``.

    h = x + Attention(input_layernorm(x))
    y = h + MLP(post_attention_layernorm(h))

Pre-norm, two residual adds, dense MLP on every one of the 88 layers. No MoE branch, no
layer-type table (``config.json`` has no ``layer_types`` and ``sliding_window`` is null, so there
is no alternating local/global pattern to dispatch on), and no sharded-residual mode — the
residual stream stays ``[1, 1, tokens_local, hidden_size]``, sequence SP-sharded and hidden
TP-replicated, from input to output. Both sub-blocks return in that same layout, which is what
makes the residual adds plain element-wise ops.
"""

import ttnn
from models.demos.mistral_medium_3_5_128b.tt.attention import Attention, AttentionConfig, ProgramConfig
from models.demos.mistral_medium_3_5_128b.tt.mlp import MLP
from models.demos.mistral_medium_3_5_128b.tt.rms_norm import RMSNorm
from models.demos.mistral_medium_3_5_128b.utils.substate import substate


class DecoderLayer:
    """A single decoder layer: norm -> attention -> residual -> norm -> MLP -> residual."""

    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        layer_idx: int,
        ccl_manager,
        mesh_config,
        *,
        max_seq_len: int,
        transformation_mat=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        sequence_parallel: bool = False,
        program_config=None,
    ):
        """
        Args:
            mesh_device: the open mesh.
            config: a :class:`~...reference.model_config.MistralMediumConfig`. The layer derives its
                own :class:`~.attention.config.AttentionConfig` from it.
            state_dict: this layer's weights with the ``model.layers.N.`` prefix stripped, i.e.
                ``self_attn.*``, ``mlp.*``, ``input_layernorm.weight``,
                ``post_attention_layernorm.weight``. Empty dict => cache-only load.
            layer_idx: index into the 88-layer stack; selects the KV cache slot.
            ccl_manager: :class:`~...tt.ccl.CCLManager`.
            mesh_config: :class:`~...tt.config.MeshConfig`.
            max_seq_len: sequence capacity this layer's attention is configured for.
            transformation_mat: shared RoPE transformation matrix; built once by the caller.
            weight_dtype: on-device weight dtype (bfloat8_b per the spec).
            tensor_cache_path: directory for the tilized-weight cache, or None.
            sequence_parallel: True when the sequence is SP-sharded across the mesh rows (the
                production prefill layout). False runs the whole sequence on every row, which the
                single-chunk op-level tests use.
            program_config: :class:`~.attention.config.ProgramConfig`; a default is built if None.
        """
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.mesh_config = mesh_config

        def _sub_cache(name):
            return f"{tensor_cache_path}/{name}" if tensor_cache_path else None

        self.input_layernorm = RMSNorm(
            mesh_device,
            config,
            substate(state_dict, "input_layernorm"),
            mesh_config=mesh_config,
            tensor_cache_path=_sub_cache("input_layernorm"),
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            config,
            substate(state_dict, "post_attention_layernorm"),
            mesh_config=mesh_config,
            tensor_cache_path=_sub_cache("post_attention_layernorm"),
        )
        self.self_attn = Attention(
            mesh_device,
            AttentionConfig(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_seq_len=max_seq_len,
                rms_norm_eps=config.rms_norm_eps,
                sequence_parallel=sequence_parallel,
            ),
            substate(state_dict, "self_attn"),
            ccl_manager,
            mesh_config,
            ProgramConfig() if program_config is None else program_config,
            layer_idx=layer_idx,
            transformation_mat=transformation_mat,
            weight_dtype=weight_dtype,
            tensor_cache_path=_sub_cache("self_attn"),
        )
        self.mlp = MLP(
            mesh_device,
            config,
            substate(state_dict, "mlp"),
            mesh_config,
            ccl_manager,
            weight_dtype=weight_dtype,
            tensor_cache_path=_sub_cache("mlp"),
        )

    def __call__(self, hidden_states, rope_mats, kv_cache=None, user_id: int = 0, cached_len: int = 0):
        """One chunk through the layer.

        Args:
            hidden_states: ``[1, 1, tokens_local, hidden_size]``.
            rope_mats: ``[cos, sin]`` for absolute positions
                ``[cached_len, cached_len + tokens_global)``.
            kv_cache: a :class:`~.attention.kv_cache.MistralKVCache`, or None.
            user_id: cache user slot.
            cached_len: tokens already written for this user/layer.

        Returns:
            ``[1, 1, tokens_local, hidden_size]``, same layout as the input.
        """
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, rope_mats, kv_cache=kv_cache, user_id=user_id, cached_len=cached_len)
        normed.deallocate(True)
        # Plain element-wise adds: both sub-blocks return the hidden-replicated residual layout.
        h = ttnn.add(hidden_states, attn_out)
        attn_out.deallocate(True)

        normed = self.post_attention_layernorm(h)
        mlp_out = self.mlp(normed)
        normed.deallocate(True)
        out = ttnn.add(h, mlp_out)
        h.deallocate(True)
        mlp_out.deallocate(True)
        return out
