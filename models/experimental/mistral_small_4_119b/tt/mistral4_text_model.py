# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-Small-4-119B text model — prefill and decode modes.

``TtMistral4TextModel`` wraps the decoder-layer stack with per-layer KV caches.
RoPE ``(cos, sin)`` tables are uploaded once via ``cache_rope_tables`` and live
in device DRAM; per-step lookups are on-device ``ttnn.slice`` calls keyed on
``current_pos``, so no host→device cos/sin upload happens during the decode loop.

Usage::

    model = TtMistral4TextModel(...)
    cos_full, sin_full = rotary(...)              # once, on host
    model.cache_rope_tables(cos_full, sin_full)   # one-shot upload

    logits = model.prefill(input_ids)             # positions [0, seq_len)
    next_tok = logits[0, -1].argmax()
    for pos in range(prefill_len, prefill_len + n_tokens):
        logits = model.decode(next_tok.unsqueeze(0).unsqueeze(0), pos)
        next_tok = logits[0, 0].argmax()
"""

from __future__ import annotations

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
        # bfloat4_b: ~32 MB per device after sharding across 4 devices (vs 64 MB at bfloat8_b).
        # Paired with LoFi math (self.lm_head_compute_kernel_config) for ~2x speedup on the
        # LM head matmul. EXPECTED to regress generation quality — gate behind a PCC test.
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_w,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        self.lm_head_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # RoPE tables — uploaded once via ``cache_rope_tables``. Per-step lookup
        # is a ``ttnn.slice`` keyed on ``current_pos``; no host crossing.
        self.cos_table_tt: ttnn.Tensor | None = None
        self.sin_table_tt: ttnn.Tensor | None = None
        self._rope_table_positions: int = 0

        # ── Pre-allocated per-step device tensors for decode ──────────────
        # The decode loop calls ttnn.as_tensor(...) for input_id and
        # current_pos every step. as_tensor on a 4-device mesh allocates
        # fresh device buffers and runs ReplicateTensorToMesh. Pre-allocating
        # once and updating in-place via ttnn.copy_host_to_device_tensor
        # skips the per-step device allocation and the mesh-mapper dispatch.
        self._decode_input_id_device = ttnn.as_tensor(
            torch.zeros((1, 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self._decode_cur_pos_device = ttnn.as_tensor(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # ── RoPE table caching ─────────────────────────────────────────────────

    def cache_rope_tables(self, cos_full: torch.Tensor, sin_full: torch.Tensor) -> None:
        """
        Upload the full RoPE ``(cos, sin)`` table to device DRAM once.

        Accepts HF ``Mistral4RotaryEmbedding`` output (shape ``[1, S, D]`` or
        ``[1, 1, S, D]``); the last dim is trimmed to ``QK_ROPE_HEAD_DIM`` if
        wider. Replicated on every mesh device; TILE layout so subsequent
        ``ttnn.slice`` calls produce TILE tensors ready for the RoPE math.
        """

        def _upload(t: torch.Tensor) -> ttnn.Tensor:
            t = t.to(torch.bfloat16)
            while t.dim() < 4:
                t = t.unsqueeze(0)
            if t.shape[-1] > QK_ROPE_HEAD_DIM:
                t = t[..., :QK_ROPE_HEAD_DIM]
            t = t.contiguous()
            return ttnn.as_tensor(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        if self.cos_table_tt is not None:
            ttnn.deallocate(self.cos_table_tt)
        if self.sin_table_tt is not None:
            ttnn.deallocate(self.sin_table_tt)

        self.cos_table_tt = _upload(cos_full)
        self.sin_table_tt = _upload(sin_full)
        self._rope_table_positions = self.cos_table_tt.shape[-2]

    # ── Internals ──────────────────────────────────────────────────────────

    def _rope_slice(self, start: int, end: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """On-device slice of the cached cos/sin tables for positions ``[start, end)``."""
        assert (
            self.cos_table_tt is not None and self.sin_table_tt is not None
        ), "cache_rope_tables() must be called before prefill/decode"
        assert (
            0 <= start < end <= self._rope_table_positions
        ), f"position range [{start}, {end}) outside cached RoPE table of size {self._rope_table_positions}"
        cos = ttnn.slice(self.cos_table_tt, [0, 0, start, 0], [1, 1, end, QK_ROPE_HEAD_DIM])
        sin = ttnn.slice(self.sin_table_tt, [0, 0, start, 0], [1, 1, end, QK_ROPE_HEAD_DIM])
        return cos, sin

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
        x = _rms_norm(x, self.final_norm_w, self.compute_kernel_config, ttnn.L1_MEMORY_CONFIG)
        logits_tt = ttnn.linear(
            x,
            self.lm_head_weight,
            compute_kernel_config=self.lm_head_compute_kernel_config,
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
        x_normed = _rms_norm(x_last, self.final_norm_w, self.compute_kernel_config, ttnn.L1_MEMORY_CONFIG)
        partial = ttnn.linear(
            x_normed,
            self.lm_head_weight,
            compute_kernel_config=self.lm_head_compute_kernel_config,
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

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run prefill and populate all KV caches for positions ``[0, seq_len)``.

        Args:
            input_ids: [1, seq_len] long tensor on CPU
        Returns:
            logits: [1, seq_len, vocab_size] bfloat16 CPU tensor
        """
        seq_len = input_ids.shape[1]
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

        x = self._embed(input_ids)
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        return self._to_logits(x)

    def prefill_device(self, input_ids_tt: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """
        Device-only prefill for trace capture/execution.

        Identical to ``prefill`` but: takes a pre-uploaded device tensor for
        input_ids, returns a device tensor for logits (no host download), and
        does NOT deallocate the RoPE slices (so the trace can be re-executed
        without re-caching the RoPE tables).

        Args:
            input_ids_tt: uint32 device tensor of shape ``[1, seq_len]``,
                          replicated across the mesh.
            seq_len:      sequence length (passed explicitly so it's a Python
                          int known at trace-capture time).

        Returns:
            logits_tt: device tensor ``[1, 1, seq_len, vocab/n_devices]``
                       column-sharded across the mesh.
        """
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

        x = ttnn.embedding(
            input_ids_tt,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        # NOTE: do not deallocate cos_tt/sin_tt — they may alias the cached
        # RoPE buffer; deallocation would invalidate the buffer for trace replay.

        x = _rms_norm(x, self.final_norm_w, self.compute_kernel_config)
        logits_tt = ttnn.linear(
            x,
            self.lm_head_weight,
            compute_kernel_config=self.lm_head_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        return logits_tt

    def _decode_upload_step_state(self, input_id: torch.Tensor, current_pos: int) -> None:
        """Update pre-allocated input_id and cur_pos device tensors in-place.

        Skips the per-step device allocation and ReplicateTensorToMesh dispatch
        that ttnn.as_tensor(device=...) does for small per-step tensors.
        """
        input_id_host = ttnn.from_torch(
            input_id.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(input_id_host, self._decode_input_id_device)

        cur_pos_host = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(cur_pos_host, self._decode_cur_pos_device)

    def decode(self, input_id: torch.Tensor, current_pos: int) -> torch.Tensor:
        """
        Decode one token at position ``current_pos``.

        Args:
            input_id:    [1, 1] long tensor on CPU (single next token)
            current_pos: cache slot to write the new K/V into. Typically
                         prefill_len + decode_step.
        Returns:
            logits: [1, 1, vocab_size] bfloat16 CPU tensor
        """
        self._decode_upload_step_state(input_id, current_pos)

        cos_tt, sin_tt = self._rope_slice(current_pos, current_pos + 1)

        x = ttnn.embedding(
            self._decode_input_id_device,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [1, 1, 1, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_decode(x, cos_tt, sin_tt, kv_cache, current_pos, self._decode_cur_pos_device)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        return self._to_logits(x)

    # ── Greedy generation entry points (on-device argmax) ──────────────────

    def prefill_next_token(self, input_ids: torch.Tensor) -> int:
        """
        Run prefill (filling all KV caches) and return the greedy next token id.

        Same as ``prefill`` but the argmax over the last position's logits runs
        on device — only a single uint32 crosses the PCIe boundary.
        """
        seq_len = input_ids.shape[1]
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

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

    def decode_next_token(self, input_id: torch.Tensor, current_pos: int) -> int:
        """Decode one token and return the greedy next token id (on-device argmax)."""
        self._decode_upload_step_state(input_id, current_pos)

        cos_tt, sin_tt = self._rope_slice(current_pos, current_pos + 1)

        x = ttnn.embedding(
            self._decode_input_id_device,
            self.embed_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.reshape(x, [1, 1, 1, HIDDEN_SIZE])
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_decode(x, cos_tt, sin_tt, kv_cache, current_pos, self._decode_cur_pos_device)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        token_id = self._next_token_on_device(x)
        ttnn.deallocate(x)
        return token_id

    # ── Embedding-input entry points (multimodal) ──────────────────────────

    def prefill_from_embeds(self, inputs_embeds: ttnn.Tensor) -> torch.Tensor:
        """
        Prefill from a caller-built embedding sequence (skip ``embed_tokens``).

        Args:
            inputs_embeds: ttnn [1, 1, seq_len, HIDDEN_SIZE], replicated on mesh.
                           Typically built by ``embed_tokens`` for text positions
                           and the multi-modal projector for image positions,
                           spliced together via slice/concat on device.
        Returns:
            logits: [1, seq_len, vocab_size] bf16 CPU tensor.
        """
        seq_len = inputs_embeds.shape[-2]
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

        x = inputs_embeds
        for layer, kv_cache in zip(self.decoder_layers, self.kv_caches):
            x = layer.forward_with_cache(x, cos_tt, sin_tt, kv_cache)

        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)
        return self._to_logits(x)

    def prefill_from_embeds_next_token(self, inputs_embeds: ttnn.Tensor) -> int:
        """
        Same as ``prefill_from_embeds`` but returns the greedy next-token id with
        on-device argmax — only a single uint32 crosses the PCIe boundary.
        """
        seq_len = inputs_embeds.shape[-2]
        cos_tt, sin_tt = self._rope_slice(0, seq_len)

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
