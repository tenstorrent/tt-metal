# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional decoder under active validation; see stage README for passed gates.

Single-request contract:
  hidden_states: device BF16 TILE [1,1,logical_tokens,10240], one request.
  state: explicit caller-owned recurrent/PLE state or paged QSA state.
  token_ids: PLE only, device UINT32 RM [1,logical_tokens,16]; word 0 is
    the raw token ID, word 1 must be 1 (a valid logical token).
  prefill_forward returns every logical row in bounded chunks of at most32.
    Padded tokens must not be supplied as logical rows.
  decode_forward consumes one row and mutates stable state buffers on device.
  QSA state.attention.position is the next absolute token position, advanced
    on device after each step. Its page table may be permuted at setup or
    refreshed in place with correspondingly remapped cache contents.
Batch contract: pass a tuple of inputs and a matching tuple of states (and
PLE token tensors). Each row has independent length, position and state;
the baseline loops over requests and returns a tuple of logical outputs.
Fresh requests require fresh/reset state; continuation reuses the previous
state, including all convolution/recurrent/hash/indexer state and paged KV.
Positions must remain within the allocated capacity; restoring only position
does not restore a prefix. Setup/hash buckets must precede trace capture.
No model intermediates cross the host boundary inside a pass. Arbitrary mask
holes and multimodal positions are outside this causal-text layer contract.
This file does not claim stage acceptance until the documented gates pass.
"""

from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule

from .chunk_input import prepare_chunk_io, read_token_chunk, read_token_words
from .chunk_output import write_token_chunk
from .decoder_components import GatedDeltaNet, GatedResidual, SparseMoe, WeightLoader
from .ngram_hash import EOS, DeviceNGramHash
from .ple import PLE
from .qsa import QSA
from .qsa_chunked import ChunkedQSA


@dataclass
class LinearState:
    conv: ttnn.Tensor | None = None
    recurrent: ttnn.Tensor | None = None
    ple_conv: ttnn.Tensor | None = None
    token_history: ttnn.Tensor | None = None
    hash_operations: dict | None = None
    attention: object | None = None


class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(
        cls, state_dict, *, hf_config, layer_idx, mesh_device, packed_embedding=None, expert_dtype=ttnn.bfloat16
    ):
        config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if config.model_type != "qwen4_exp_text" or config.hidden_size != 2560 or config.hc_count != 4:
            raise ValueError("This autoport requires the actual Qwen3.8-Flash-Next text config")
        kind = config.layer_types[layer_idx]
        if kind not in ("linear_attention", "indexed_attention"):
            raise ValueError(f"Unsupported target layer kind: {kind}")
        has_ple = layer_idx + 1 in config.ple_layer_ids
        if has_ple and (packed_embedding is None or packed_embedding.rows != 320001536):
            raise ValueError("PLE requires the complete resident checkpoint table")
        prefix = f"model.language_model.layers.{layer_idx}."
        weights = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
        if not weights:
            weights = state_dict
        loader = WeightLoader(weights, mesh_device)
        result = cls()
        result.mesh_device = mesh_device
        result.config = config
        result.layer_idx = layer_idx
        result.kind = kind
        result.ple = PLE(loader, config) if has_ple else None
        result.embedding = packed_embedding
        result.attn_mix = GatedResidual(loader, "attn_hyper_connection.", config)
        result.attn = GatedDeltaNet(loader, config) if kind == "linear_attention" else QSA(loader, config)
        result.chunked_attn = ChunkedQSA(result.attn) if kind == "indexed_attention" else None
        result.mlp_mix = GatedResidual(loader, "mlp_hyper_connection.", config)
        result.mlp = SparseMoe(loader, config, expert_dtype=expert_dtype, reuse_private_buffers=has_ple)
        prepare_chunk_io(mesh_device)
        return result

    def create_state(self, *, hash_lengths=tuple(range(1, 33)), capacity=None, page_table=None):
        """Setup boundary; allocate before trace capture."""
        if self.kind == "indexed_attention":
            return LinearState(attention=self.attn.make_state(capacity=capacity, page_table=page_table))
        state = LinearState(
            ttnn.zeros((1, 1, 3, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device),
            ttnn.zeros((1, 48, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh_device),
        )
        if self.ple is not None:
            import torch

            def integer(value):
                return ttnn.from_torch(value, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device)

            state.ple_conv = ttnn.zeros(
                (1, 1, 9, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
            )
            history = torch.zeros(1, 16, dtype=torch.int32)
            history[:, :2] = EOS
            state.token_history = integer(history)
            state.hash_operations = {}
            for length in sorted(set(hash_lengths)):
                if not 1 <= length <= 32:
                    raise ValueError("Hash physical lengths must be in [1,32]")
                control = torch.zeros(1, 16, dtype=torch.int32)
                control[:, 0] = length
                operation = DeviceNGramHash(
                    tokens=integer(torch.zeros(1, length, 16, dtype=torch.int32)),
                    control=integer(control),
                    history=state.token_history,
                    output=integer(torch.zeros(1, length, 16, dtype=torch.int32)),
                )
                operation.prepare()
                state.hash_operations[length] = operation
        return state

    def _chunk(self, hidden_states, state, token_ids=None):
        if self.ple is not None:
            length = hidden_states.shape[-2]
            if token_ids is None or tuple(token_ids.shape) != (1, length, 16):
                raise ValueError("PLE requires device token words [1,tokens,16]")
            operation = state.hash_operations[length]
            ttnn.copy(token_ids, operation.tensors[0])
            operation.forward()
            ids = ttnn.reshape(operation.tensors[-1], (1, 1, length, 16))
            embeddings = self.embedding(ids)
            hidden_states = ttnn.add(hidden_states, self.ple(hidden_states, embeddings, conv_state=state.ple_conv))
        mixed, injection = self.attn_mix.read(hidden_states)
        if self.kind == "linear_attention":
            block = self.attn(mixed, conv_state=state.conv, recurrent_state=state.recurrent)
        else:
            block = self.chunked_attn.forward_chunk(mixed, state=state.attention)
            self.chunked_attn.advance(state.attention, hidden_states.shape[-2])
        hidden_states = self.attn_mix.write(hidden_states, block, injection)
        mixed, injection = self.mlp_mix.read(hidden_states)
        return self.mlp_mix.write(hidden_states, self.mlp(mixed), injection)

    def prefill_forward(self, hidden_states, *, state, token_ids=None):
        """Process arbitrary positive logical token count, continuing supplied state."""
        if isinstance(hidden_states, (tuple, list)):
            self._validate_batch(hidden_states, state, token_ids)
            return tuple(
                self.prefill_forward(x, state=state[i], token_ids=None if token_ids is None else token_ids[i])
                for i, x in enumerate(hidden_states)
            )
        if tuple(hidden_states.shape)[:2] != (1, 1) or hidden_states.shape[-1] != 10240:
            raise ValueError("Expected [1,1,tokens,10240] hidden states")
        length = hidden_states.shape[-2]
        if not 0 < length <= self.config.max_position_embeddings:
            raise ValueError("Invalid logical token count")
        chunk_size = 32
        if length <= chunk_size:
            return self._chunk(hidden_states, state, token_ids)
        output = ttnn.empty(
            tuple(hidden_states.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for start in range(0, length, chunk_size):
            count = min(chunk_size, length - start)
            chunk = self._chunk(
                read_token_chunk(hidden_states, start, count),
                state,
                read_token_words(token_ids, start, count) if token_ids is not None else None,
            )
            write_token_chunk(chunk, output, start)
            # The queued writer is the last consumer; do not retain this output
            # while computing the next chunk's decoder intermediates.
            del chunk
        return output

    def decode_forward(self, hidden_states, *, state, token_ids=None):
        """One token using stable device state addresses; suitable for capture."""
        if isinstance(hidden_states, (tuple, list)):
            self._validate_batch(hidden_states, state, token_ids)
            return tuple(
                self.decode_forward(x, state=state[i], token_ids=None if token_ids is None else token_ids[i])
                for i, x in enumerate(hidden_states)
            )
        if tuple(hidden_states.shape) != (1, 1, 1, 10240):
            raise ValueError("Expected [1,1,1,10240] decode hidden states")
        return self._chunk(hidden_states, state, token_ids)

    @staticmethod
    def _validate_batch(hidden_states, state, token_ids):
        if not 1 <= len(hidden_states) <= 32 or len(state) != len(hidden_states):
            raise ValueError("Expected 1..32 inputs and one independent state per request")
        if token_ids is not None and len(token_ids) != len(hidden_states):
            raise ValueError("PLE token rows must match the request count")
