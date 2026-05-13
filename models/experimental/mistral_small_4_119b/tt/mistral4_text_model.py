# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B text model — prefill and decode modes.

``TtMistral4TextModel`` wraps the decoder-layer stack with per-layer KV caches,
exposing two entry points:

    logits = model.prefill(input_ids, position_embeddings)
        - Runs the full prefill forward and fills all KV caches.
        - Returns [1, seq_len, vocab_size].

    logits = model.decode(input_id, position_embeddings, current_pos)
        - Decodes one token, updating each layer's KV cache at current_pos.
        - Returns [1, 1, vocab_size].

Call prefill once for the prompt, then decode in a loop for generation:

    logits = model.prefill(input_ids, pos_emb_prefill)
    next_tok = logits[0, -1].argmax()
    for pos in range(prefill_len, prefill_len + n_tokens):
        logits = model.decode(next_tok.unsqueeze(0).unsqueeze(0), pos_emb_decode, pos)
        next_tok = logits[0, 0].argmax()
"""

from __future__ import annotations

from typing import Tuple

import torch

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_NUM_LAYERS,
    HIDDEN_SIZE,
    QK_ROPE_HEAD_DIM,
    TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _load_norm_weight
from models.experimental.mistral_small_4_119b.tt.mistral4_text_prefill import (
    TtMistral4DecoderLayer,
    _rms_norm,
)


class TtMistral4TextModel:
    """
    Mistral-Small-4 text model with prefill and decode support.

    Args:
        mesh_device:        TTNN MeshDevice (e.g. 4×P150 → [1, 4] mesh)
        state_dict:         HF checkpoint dict (filtered to required prefixes)
        text_config:        HF ``text_config`` object
        num_decoder_layers: layers to instantiate (1..36; default 36)
        max_seq_len:        maximum total tokens (prefill + decode); sets cache size
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        text_config,
        num_decoder_layers: int = EXPECTED_NUM_LAYERS,
        max_seq_len: int = 4096,
    ):
        self.mesh_device = mesh_device
        self.num_decoder_layers = num_decoder_layers
        self.max_seq_len = max_seq_len

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # ── Embedding ──────────────────────────────────────────────────
        embed_w = state_dict[TEXT_MODEL_EMBED_TOKENS_WEIGHT_KEY].to(torch.bfloat16)
        self.embed_weight = ttnn.as_tensor(
            embed_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # ── Decoder layers + per-layer KV caches ───────────────────────
        self.decoder_layers: list[TtMistral4DecoderLayer] = []
        self.kv_caches: list[tuple] = []

        for i in range(num_decoder_layers):
            layer = TtMistral4DecoderLayer(
                mesh_device=mesh_device,
                state_dict=state_dict,
                layer_idx=i,
                compute_kernel_config=self.compute_kernel_config,
            )
            self.decoder_layers.append(layer)
            self.kv_caches.append(layer.attn.allocate_kv_cache(max_seq_len))

        # ── Final norm + LM head ────────────────────────────────────────
        self.final_norm_w = _load_norm_weight(state_dict, "language_model.model.norm.weight", HIDDEN_SIZE, mesh_device)

        lm_head_w = state_dict["language_model.lm_head.weight"].to(torch.bfloat16).T.contiguous()
        # bfloat8_b: 64 MB per device (vs 128 MB bfloat16) after sharding across 8 devices.
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _prepare_rope(
        self,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        cos_t, sin_t = position_embeddings

        def _prep(t: torch.Tensor) -> ttnn.Tensor:
            t = t.to(torch.bfloat16)
            while t.dim() < 4:
                t = t.unsqueeze(0)
            if t.shape[-1] > QK_ROPE_HEAD_DIM:
                t = t[..., :QK_ROPE_HEAD_DIM].contiguous()
            t = t[:, :, :seq_len, :].contiguous()
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return _prep(cos_t), _prep(sin_t)

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        ids_tt = ttnn.as_tensor(
            input_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x = ttnn.embedding(ids_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ids_tt)
        return x

    def embed_tokens(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """
        Public token-embedding lookup for multimodal scatter.

        Returns a ttnn tensor of shape ``[1, 1, seq_len, HIDDEN_SIZE]`` in
        TILE layout, replicated on the mesh. Callers may slice/concat this
        with vision embeddings before calling ``prefill_from_embeds``.
        """
        seq_len = input_ids.shape[1]
        x = self._embed(input_ids)
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def _to_logits(self, x: ttnn.Tensor) -> torch.Tensor:
        """Final norm → lm_head → gather to host."""
        x = _rms_norm(x, self.final_norm_w, self.compute_kernel_config)
        logits_tt = ttnn.linear(
            x,
            self.lm_head_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        # lm_head_weight is column-sharded across devices (dim=1 of [hidden, vocab]).
        # Each device produces partial logits [1, 1, seq_len, vocab/n_devices].
        # Concatenate along the vocab dim to reconstruct full logits.
        logits_host = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3),
        )
        ttnn.deallocate(logits_tt)
        return logits_host[0].to(torch.bfloat16)

    def _next_token_on_device(self, x_last: ttnn.Tensor) -> int:
        """
        On-device greedy next-token from one hidden state.

        x_last: [1, 1, 1, HIDDEN_SIZE] (last position's hidden, replicated on mesh).
        Returns the token id as a Python int.

        Pipeline: rms_norm → lm_head (partial) → all_gather over vocab → argmax → readback.
        Only a single uint32 per device crosses the PCIe boundary.
        """
        x_normed = _rms_norm(x_last, self.final_norm_w, self.compute_kernel_config)
        partial = ttnn.linear(
            x_normed,
            self.lm_head_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # per device: [1, 1, 1, vocab/num_devices]
        ttnn.deallocate(x_normed)

        num_devices = self.mesh_device.get_num_devices()
        if num_devices > 1:
            full = ttnn.all_gather(
                partial,
                dim=3,
                num_links=1,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # per device: [1, 1, 1, vocab]
            ttnn.deallocate(partial)
        else:
            full = partial

        idx_tt = ttnn.argmax(full, dim=-1)  # per device: [1, 1, 1] uint32 ROW_MAJOR
        ttnn.deallocate(full)

        # Every device holds the same full logits, so every argmax result is identical.
        # Pull the first device's value.
        idx_host = ttnn.to_torch(
            idx_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )
        ttnn.deallocate(idx_tt)
        return int(idx_host.flatten()[0].item())

    # ── Public API ─────────────────────────────────────────────────────────

    def prefill(
        self,
        input_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Run prefill and populate all KV caches.

        Args:
            input_ids:           [1, seq_len] long tensor on CPU
            position_embeddings: (cos, sin) from HF Mistral4RotaryEmbedding,
                                 each [1, seq_len, D] or [1, 1, seq_len, D]
        Returns:
            logits: [1, seq_len, vocab_size] bfloat16 CPU tensor
        """
        seq_len = input_ids.shape[1]
        cos_tt, sin_tt = self._prepare_rope(position_embeddings, seq_len)

        x = self._embed(input_ids)
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        return self._to_logits(x)

    def decode(
        self,
        input_id: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        current_pos: int,
    ) -> torch.Tensor:
        """
        Decode one token.

        Args:
            input_id:            [1, 1] long tensor on CPU (single next token)
            position_embeddings: (cos, sin) for position current_pos,
                                 each [1, 1, D] or [1, 1, 1, D]
            current_pos:         cache slot to write the new K/V into.
                                 Typically prefill_len + decode_step.
        Returns:
            logits: [1, 1, vocab_size] bfloat16 CPU tensor
        """
        cos_tt, sin_tt = self._prepare_rope(position_embeddings, 1)

        x = self._embed(input_id)
        x = ttnn.reshape(x, [1, 1, 1, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_decode(x, cos_tt, sin_tt, kv_cache, current_pos)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        return self._to_logits(x)

    # ── Greedy generation entry points (on-device argmax) ──────────────────

    def prefill_next_token(
        self,
        input_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> int:
        """
        Run prefill (filling all KV caches) and return the greedy next token id.

        Same as ``prefill`` but the argmax over the last position's logits runs
        on device — only a single uint32 crosses the PCIe boundary.
        """
        seq_len = input_ids.shape[1]
        cos_tt, sin_tt = self._prepare_rope(position_embeddings, seq_len)

        x = self._embed(input_ids)
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        # Extract the last position's hidden state: [1, 1, 1, HIDDEN_SIZE].
        x_last = ttnn.slice(x, [0, 0, seq_len - 1, 0], [1, 1, seq_len, HIDDEN_SIZE])
        ttnn.deallocate(x)
        token_id = self._next_token_on_device(x_last)
        ttnn.deallocate(x_last)
        return token_id

    def decode_next_token(
        self,
        input_id: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        current_pos: int,
    ) -> int:
        """Decode one token and return the greedy next token id (on-device argmax)."""
        cos_tt, sin_tt = self._prepare_rope(position_embeddings, 1)

        x = self._embed(input_id)
        x = ttnn.reshape(x, [1, 1, 1, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_decode(x, cos_tt, sin_tt, kv_cache, current_pos)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        token_id = self._next_token_on_device(x)
        ttnn.deallocate(x)
        return token_id

    # ── Embedding-input entry points (multimodal) ──────────────────────────

    def prefill_from_embeds(
        self,
        inputs_embeds: ttnn.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Prefill from a caller-built embedding sequence (skip ``embed_tokens``).

        Args:
            inputs_embeds:       ttnn [1, 1, seq_len, HIDDEN_SIZE], replicated on mesh.
                                 Typically built by ``embed_tokens`` for text positions
                                 and the multi-modal projector for image positions,
                                 spliced together via slice/concat on device.
            position_embeddings: (cos, sin) covering positions [0, seq_len).
        Returns:
            logits: [1, seq_len, vocab_size] bf16 CPU tensor.
        """
        seq_len = inputs_embeds.shape[-2]
        cos_tt, sin_tt = self._prepare_rope(position_embeddings, seq_len)

        x = inputs_embeds
        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
        return self._to_logits(x)

    def prefill_from_embeds_next_token(
        self,
        inputs_embeds: ttnn.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> int:
        """
        Same as ``prefill_from_embeds`` but returns the greedy next-token id with
        on-device argmax — only a single uint32 crosses the PCIe boundary.
        """
        seq_len = inputs_embeds.shape[-2]
        cos_tt, sin_tt = self._prepare_rope(position_embeddings, seq_len)

        x = inputs_embeds
        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        x_last = ttnn.slice(x, [0, 0, seq_len - 1, 0], [1, 1, seq_len, HIDDEN_SIZE])
        ttnn.deallocate(x)
        token_id = self._next_token_on_device(x_last)
        ttnn.deallocate(x_last)
        return token_id
