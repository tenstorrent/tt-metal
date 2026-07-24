# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS prefill pipeline — wraps the (validated) model with the per-layer
KV-migration callback for disaggregated prefill/decode serving.

See PREFILL_PROPOSAL.md §5/§7 for architecture and flow.

Status:
  * model forward (all 36 layers, prefill + decode paths) — VALIDATED
  * on_layer_complete seam — wired into model._forward_layers_and_head
  * _prepare_input_tensor / prefill / compile — IMPLEMENTED (Tier 1)
  * migration transport — NoOp until the migration team's endpoint lands

Key differences from MiniMax-M2 pipeline:
  * 36 layers (not 62); migration fires 36 times per request
  * Separate k_cache + v_cache per layer — endpoint must address both tensors
  * No chunking (PREFILL_PROPOSAL.md §8.4); single full-sequence forward

Reference: models/demos/minimax_m2/tt/tt_minimax_prefill_pipeline.py
"""

import math
import time

import torch
from loguru import logger

import ttnn

from .runners.migration_setup import MigrationEndpoint, NoOpMigrationEndpoint

# Migration chunk granularity: 32 tokens (DRAM-bank aligned, matches migration team spec).
_MIGRATION_BLOCK = 32

# Pad token id used to fill sequences shorter than max_seq_len.
_PAD_TOKEN_ID = 1


class GptOssPrefillPipeline:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        hf_config,
        model,
        kv_cache: list,
        mesh_config,
        sp_factor: int = 4,
        max_seq_len: int = 131072,
    ):
        """
        Args:
            mesh_device: The 4×8 BH Galaxy mesh (SP rows × TP cols).
            hf_config: HuggingFace config for the GPT-OSS model.
            model: The validated GptOss Model (tt/model.py).
            kv_cache: Per-layer (k_cache, v_cache) list from create_tt_model.
            mesh_config: MeshConfig(mesh_shape, decode=...) instance.
            sp_factor: Number of SP rows (sequence parallel). Must equal mesh rows.
            max_seq_len: Maximum input sequence length (must be divisible by sp_factor).
        """
        assert max_seq_len % sp_factor == 0, f"max_seq_len={max_seq_len} must be divisible by SP={sp_factor}"
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.model = model
        self.kv_cache = kv_cache
        self.mesh_config = mesh_config
        self.sp_factor = sp_factor
        self.max_seq_len = max_seq_len
        self.compiled = False
        self.endpoint: MigrationEndpoint = NoOpMigrationEndpoint()

    def setup_migration(self, endpoint: MigrationEndpoint) -> None:
        """Bind a real migration endpoint (else stays NoOp)."""
        self.endpoint = endpoint

    def compile(self) -> None:
        """Warm-up the model with a max-length dummy sequence to trigger JIT compilation."""
        logger.info(f"GptOssPrefillPipeline.compile(): warming up with {self.max_seq_len} tokens")
        t0 = time.perf_counter()
        dummy_ids = [0] * self.max_seq_len
        tt_input = self._prepare_input_tensor(dummy_ids)
        isl_per_row = self.max_seq_len // self.sp_factor
        # last real token is the last token of the sequence — on the last SP row
        get_last_token = ((isl_per_row - 1) // 32) * 32
        self.model.ttnn_prefill_forward(
            x=tt_input,
            kv_cache=self.kv_cache,
            get_last_token=get_last_token,
        )
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[prefill timing] task_id=WARMUP num_tokens={self.max_seq_len} compile() = {warmup_ms:.2f} ms")
        self.compiled = True

    def _prepare_input_tensor(self, padded_token_ids: list) -> ttnn.Tensor:
        """Upload and embed token IDs, SP-sharded across mesh rows.

        Args:
            padded_token_ids: List of ints of length max_seq_len (caller must pad).

        Returns:
            SP-sharded embedded tensor ready for model.ttnn_prefill_forward(x=...).
            Shape per SP row: [1, 1, max_seq_len // sp_factor, hidden_size].
        """
        sp = self.sp_factor
        seq_len = len(padded_token_ids)
        assert seq_len == self.max_seq_len, f"Expected {self.max_seq_len} tokens, got {seq_len}"
        isl_per_row = seq_len // sp

        # Reshape to [SP, 1, 1, isl_per_row] for ShardTensor2dMesh(dims=(0, None)):
        # rows of the mesh each get one [1, 1, 1, isl_per_row] slice; all TP cols replicate.
        t = torch.tensor(padded_token_ids, dtype=torch.int64).reshape(sp, 1, 1, isl_per_row)
        tokens = ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                mesh_shape=self.mesh_device.shape,
                dims=(0, None),
            ),
        )
        tt_embeds = ttnn.embedding(tokens, self.model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        tokens.deallocate(True)
        if len(tt_embeds.shape) == 3:
            tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)
        return tt_embeds

    def _build_migration_callback(self, slot_id: int, actual_isl: int, dst_slot: int):
        """Return on_layer_complete(layer_idx) that migrates that layer's KV.

        pos_end is rounded up to the nearest _MIGRATION_BLOCK boundary so every
        migration covers a whole number of 32-token chunks (DRAM-bank aligned).
        NoOp endpoint makes this a cheap no-op until the migration API lands.

        GPT-OSS note: the real endpoint must migrate BOTH k_cache AND v_cache for
        each layer (two separate DRAM addresses). See PREFILL_PROPOSAL.md §5.
        """
        pos_end = math.ceil(actual_isl / _MIGRATION_BLOCK) * _MIGRATION_BLOCK

        def on_layer_complete(layer_idx: int) -> None:
            uuid = self.endpoint.migrate_layer(layer_idx, 0, pos_end, slot_id, dst_slot)
            self.endpoint.wait(uuid)

        return on_layer_complete

    def _extract_first_token(self, tt_logits: ttnn.Tensor, actual_isl: int, get_last_token: int) -> int:
        """Pull the argmax first token from the SP row that holds the last real token.

        With SP=4, the sequence is split across rows. Only the row whose shard
        contains position (actual_isl - 1) has the meaningful last-token logit.

        Args:
            tt_logits: Multi-device tensor after ttnn_prefill_forward with get_last_token set.
            actual_isl: Number of real (non-pad) tokens in the input.
            get_last_token: The 32-tile start offset passed to ttnn_prefill_forward
                            (computed as (local_last_pos // 32) * 32).

        Returns:
            Integer token id of the predicted first token.
        """
        isl_per_row = self.max_seq_len // self.sp_factor
        sp_row = (actual_isl - 1) // isl_per_row
        local_last_pos = (actual_isl - 1) % isl_per_row
        pos_within_tile = local_last_pos % 32

        tp = self.mesh_device.shape[1]
        device_tensors = ttnn.get_device_tensors(tt_logits)
        # Gather the TP column tensors from the target SP row
        row_tensors = [ttnn.to_torch(device_tensors[sp_row * tp + col]) for col in range(tp)]
        # Concatenate along vocab (last) dim: [1, 1, 32, vocab_size_padded]
        torch_logits = torch.cat(row_tensors, dim=-1)
        # Extract the one real-token position and truncate to true vocab_size
        first_token_logits = torch_logits[0, 0, pos_within_tile, : self.hf_config.vocab_size]
        return int(torch.argmax(first_token_logits))

    def prefill(self, token_ids: list, slot_id: int, actual_isl: int, dst_slot: int) -> int:
        """Full-sequence prefill for one request, migrating KV per layer after each.

        Args:
            token_ids: Raw (unpadded) token IDs for the prompt (len == actual_isl).
            slot_id: KV cache slot for this request on the prefill side.
            actual_isl: Number of real tokens (== len(token_ids)).
            dst_slot: KV cache slot on the decode side to migrate into.

        Returns:
            First generated token id (argmax of last-position logits).
        """
        assert self.compiled, "Call compile() before prefill()"
        logger.info(f"GptOssPrefillPipeline.prefill: isl={actual_isl} slot={slot_id} dst={dst_slot}")

        if actual_isl > self.max_seq_len:
            raise ValueError(f"actual_isl={actual_isl} exceeds max_seq_len={self.max_seq_len}")

        # Pad to max_seq_len so every run has a fixed shape for the trace.
        padded = list(token_ids) + [_PAD_TOKEN_ID] * (self.max_seq_len - actual_isl)

        # SP-aware get_last_token: the last real token lives on a specific SP row.
        # We need the 32-tile start within THAT row's local sequence shard.
        isl_per_row = self.max_seq_len // self.sp_factor
        local_last_pos = (actual_isl - 1) % isl_per_row
        get_last_token = (local_last_pos // 32) * 32

        callback = self._build_migration_callback(slot_id, actual_isl, dst_slot)
        tt_input = self._prepare_input_tensor(padded)

        t0 = time.perf_counter()
        tt_logits = self.model.ttnn_prefill_forward(
            x=tt_input,
            kv_cache=self.kv_cache,
            get_last_token=get_last_token,
            on_layer_complete=callback,
        )
        first_token = self._extract_first_token(tt_logits, actual_isl, get_last_token)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[prefill timing] isl={actual_isl} slot={slot_id} prefill={dt_ms:.2f} ms first_token={first_token}"
        )
        return first_token
