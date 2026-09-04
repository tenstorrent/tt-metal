# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B TTNN prefill model. Copied from ``gpt_oss_d_p/tt/model.py``.

    embedding -> [DecoderLayer] * 32 -> final RMSNorm -> lm_head

Like the gpt-oss donor, RoPE is the on-device INDEXED rope: the whole-cache, block-cyclic, SP cos/sin
are built ONCE by the runtime (``tt/rope.build_indexed_rope``) and passed into ``prefill_forward`` as
``rot_mats_global``; the model owns only the replicated transformation matrix. There is no
``RotarySetup``.

Differences from the donor, all following from Llama being dense:
  * every layer is the same block — no MoE, no ``layer_types`` dispatch, no EP arguments;
  * ``skip_lm_head=True`` is the prefill default, as in the donor: disaggregated prefill produces KV,
    not logits, so the lm_head runs only in tests and in a future decode bring-up;
  * on-device sampling is not wired at all (the donor keeps hooks for a decode bring-up; this package
    is prefill-only and dead hooks are worse than none).
"""

import torch

import ttnn
from models.demos.llama3_1_8b_d_p.tt.config import MeshConfig
from models.demos.llama3_1_8b_d_p.tt.rope import build_transformation_mat
from models.demos.llama3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama3_1_8b_d_p.utils.substate import substate

from .layer import DecoderLayer
from .parallel_embedding import TtParallelEmbedding, cache_name_for, embed_shard_2d
from .rms_norm import RMSNorm


def compute_per_device_vocab(vocab_size, num_tp):
    """Per-device lm_head vocab width: tile-aligned, then rounded up to the next power of 2.

    The power-of-2 rounding is what lets ``ttnn.topk`` take its multi-core (bitonic) path, and it must
    match the lm_head weight padding — hence one helper rather than the arithmetic inline.
    Llama: 128256 / 4 = 32064 -> 32064 (tile-aligned) -> 32768.
    """
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()


def padded_vocab_size(vocab_size, num_tp):
    """Total padded vocab across TP = per-device power-of-2 width x TP. 128256 -> 131072 at TP=4."""
    return compute_per_device_vocab(vocab_size, num_tp) * num_tp


def load_lm_head_weight(mesh_device, weight, *, vocab_size, mesh_config, dtype=ttnn.bfloat8_b, tensor_cache_path=None):
    """Pad the lm_head vocab, then shard it column-parallel across TP.

    The pad happens BEFORE the shard so each chip owns a contiguous power-of-2 vocab slice and the
    padding lands entirely in the tail of the LAST chip. Padding after sharding would interleave dead
    columns into every chip's slice, which no test of a KV-only prefill run would ever notice.

    ``weight`` is HF's ``[vocab, hidden]``; it is transposed to ``[hidden, vocab]`` for ttnn.linear.
    ``None`` means cache-only load.
    """
    total = padded_vocab_size(vocab_size, mesh_config.tp)
    if weight is not None:
        weight = weight.transpose(0, 1)  # [hidden, vocab]
        if weight.shape[1] < total:
            weight = torch.nn.functional.pad(weight, (0, total - weight.shape[1]), "constant", 0)
    return ttnn.as_tensor(
        weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head_padded_pow2.weight"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
    )


class Model:
    """Llama 3.1 8B TTNN prefill model (GQA + dense SwiGLU, SP x TP on the target mesh)."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        tensor_cache_path=None,
        mesh_config=None,
        max_seq_len=128 * 1024,
        attn_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat4_b,
        embed_dtype=ttnn.bfloat16,
        lm_head_dtype=ttnn.bfloat8_b,
        sequence_parallel=False,
        num_layers=None,
    ):
        """
        Args:
            state_dict: full model state dict with ``model.*`` / ``lm_head.*`` keys. Empty dict =>
                cache-only load (every tilized tensor comes from ``tensor_cache_path``).
            num_layers: override the layer count — for fast tests that want 1-2 layers at real dims.
        """
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config or MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1])
        self.hf_config = hf_config
        self.vocab_size = hf_config.vocab_size
        self.num_layers = hf_config.num_hidden_layers if num_layers is None else num_layers
        self.max_seq_len = max_seq_len

        # Only the transformation matrix lives here; the cos/sin are the runtime's (built once).
        self.transformation_mats = {"prefill": build_transformation_mat(mesh_device)}

        shard_vocab = embed_shard_2d()
        self.embed_tokens = TtParallelEmbedding(
            mesh_device=mesh_device,
            vocab_size=self.vocab_size,
            emb_dim=hf_config.hidden_size,
            mesh_config=self.mesh_config,
            ccl_manager=ccl_manager,
            torch_weight=substate(state_dict, "model.embed_tokens").get("weight") if state_dict else None,
            cache_file_name=get_cache_file_name(tensor_cache_path, cache_name_for(shard_vocab)),
            dtype=embed_dtype,
            shard_vocab_on_sp=shard_vocab,
        )

        self.layers = [
            DecoderLayer(
                mesh_device,
                hf_config,
                substate(state_dict, f"model.layers.{layer_idx}"),
                layer_idx=layer_idx,
                ccl_manager=ccl_manager,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"model.layers.{layer_idx}"),
                mesh_config=self.mesh_config,
                transformation_mats=self.transformation_mats,
                max_seq_len=max_seq_len,
                attn_weight_dtype=attn_weight_dtype,
                mlp_weight_dtype=mlp_weight_dtype,
                sequence_parallel=sequence_parallel,
            )
            for layer_idx in range(self.num_layers)
        ]

        self.norm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "model.norm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "norm"),
            mesh_config=self.mesh_config,
        )

        self.padded_vocab_size = padded_vocab_size(self.vocab_size, self.mesh_config.tp)
        self.lm_head_weight = load_lm_head_weight(
            mesh_device,
            substate(state_dict, "lm_head").get("weight") if state_dict else None,
            vocab_size=self.vocab_size,
            mesh_config=self.mesh_config,
            dtype=lm_head_dtype,
            tensor_cache_path=tensor_cache_path,
        )

    def embed(self, tokens):
        """uint32 token ids ``[1, 1, s_local]`` -> hidden ``[1, 1, s_local, hidden]``."""
        return self.embed_tokens(tokens)

    def forward_layers(
        self,
        hidden_states,
        rope_mats,
        *,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
        on_layer_complete=None,
    ):
        """Run the decoder stack.

        ``on_layer_complete``: optional ``fn(layer_idx)`` invoked after each layer — the SEAM for
        per-layer KV migration / validation in the disaggregated prefill pipeline. Default None.
        """
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=rope_mats,
                kv_cache=kv_cache,
                user_id=user_id,
                batch_size=batch_size,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            if on_layer_complete is not None:
                on_layer_complete(i)
        return hidden_states

    def lm_head(self, hidden_states):
        """Final norm + lm_head projection -> logits ``[1, 1, S, padded_vocab/tp]`` per chip."""
        normed = self.norm(hidden_states)
        logits = ttnn.matmul(normed, self.lm_head_weight, dtype=ttnn.bfloat8_b)
        normed.deallocate(True)
        return logits

    def prefill_forward(
        self,
        tokens,
        rot_mats_global,
        *,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=True,
        skip_lm_head=True,
        on_layer_complete=None,
    ):
        """One prefill chunk: embed -> layers -> (optionally) final norm + lm_head.

        ``skip_lm_head=True`` is the default and the disaggregated-prefill path: the product of a
        prefill run is the KV cache, not logits, and the lm_head matmul against a 128k vocab is pure
        cost there. Tests and a future decode bring-up pass False.
        """
        hidden_states = self.embed(tokens)
        hidden_states = self.forward_layers(
            hidden_states,
            rot_mats_global,
            kv_cache=kv_cache,
            user_id=user_id,
            batch_size=batch_size,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
            on_layer_complete=on_layer_complete,
        )
        if skip_lm_head:
            return hidden_states
        return self.lm_head(hidden_states)
