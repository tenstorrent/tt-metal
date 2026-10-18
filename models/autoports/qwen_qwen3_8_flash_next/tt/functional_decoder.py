"""Single-device Qwen3.8-Flash-Next decoder, including all three layer kinds.

Inputs/outputs are BF16 TILE [1,B,S,10240] (four residual streams). Decode
requires S=1. Positions and request slots are UINT32 ROW_MAJOR [B,32], column
zero carrying the absolute first-token position and state-bank slot. Position
zero starts a fresh request. All lanes have the same logical S; padding is an
internal TT tile property, never a token. PLE additionally takes UINT32 RM token
IDs [B,S]. Indexed attention takes an INT32 RM page table [B,num_pages].

``prepare(sequence_lengths)`` allocates PLE workspaces before warmup/capture.
Prefill splits arbitrary logical S into bounded chunks and preserves every
output. A caller may stream chunks with absolute positions to bound input and
output storage at the full advertised context. Runtime never reads host data.
"""
from collections.abc import Mapping

import ttnn
from models.common.lightweightmodule import LightweightModule

from .gdn import GatedDeltaNet
from .hyperconnection import HyperConnection
from .moe import MoE
from .ple import PLE
from .qsa import QSA


class _PrefixView(Mapping):
    def __init__(self, state, prefix):
        self.state, self.prefix = state, prefix

    def __getitem__(self, key):
        return self.state[self.prefix + key]

    def __iter__(self):
        return (k[len(self.prefix) :] for k in self.state if k.startswith(self.prefix))

    def __len__(self):
        return sum(1 for _ in self)


class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch_size=1,
        max_requests=None,
        chunk_size=128,
        cache_dir=None,
    ):
        import torch  # Setup only.

        obj = cls()
        cfg = getattr(hf_config, "text_config", hf_config)
        if (cfg.hidden_size, cfg.hc_count) != (2560, 4):
            raise ValueError("Expected the target Qwen3.8-Flash-Next configuration")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        prefix = f"model.language_model.layers.{layer_idx}."
        state = _PrefixView(state_dict, prefix) if prefix + "mlp.gate.weight" in state_dict else state_dict
        obj.batch_size, obj.chunk_size = batch_size, chunk_size
        obj.context = cfg.max_position_embeddings
        obj.mesh = mesh_device
        obj.kind = cfg.layer_types[layer_idx]
        obj.attn_hyper = HyperConnection.from_state_dict(
            state, prefix="attn_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp_hyper = HyperConnection.from_state_dict(
            state, prefix="mlp_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp = MoE.from_state_dict(state, hf_config=cfg, mesh_device=mesh_device)
        requests = max_requests or batch_size
        if obj.kind == "linear_attention":
            obj.attention = GatedDeltaNet.from_state_dict(
                _PrefixView(state, "linear_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
                chunk_size=chunk_size,
            )
        else:
            obj.attention = QSA.from_state_dict(
                _PrefixView(state, "self_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
            )
        obj.ple = None
        if layer_idx + 1 in cfg.ple_layer_ids:
            obj.ple = PLE.from_state_dict(
                state, hf_config=cfg, layer_idx=layer_idx, prefix="ple.", mesh_device=mesh_device, cache_dir=cache_dir
            )
            obj.ple_cache = obj.ple.make_cache(max_requests=requests)
        obj.lengths = {}
        obj.prepare([1, chunk_size])
        return obj

    def prepare(self, sequence_lengths):
        """Prepare every chunk/tail shape before a live trace is captured."""
        import torch  # Explicit setup boundary.

        for length in sequence_lengths:
            if not 1 <= length <= self.context:
                raise ValueError("Sequence length is outside the advertised context")
            for start in range(0, length, self.chunk_size):
                size = min(self.chunk_size, length - start)
                if size not in self.lengths:
                    host = torch.zeros(self.batch_size, 32, dtype=torch.int32)
                    host[:, 0] = size
                    self.lengths[size] = ttnn.from_torch(
                        host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh
                    )
                    if self.ple is not None:
                        self.ple.prepare(batch=self.batch_size, physical_seq_len=size)

    def _chunk(self, x, *, positions, slots, tokens, page_table, decode):
        size = x.shape[2]
        if self.ple is not None:
            x = ttnn.add(
                x,
                self.ple.forward(
                    x,
                    tokens=tokens,
                    positions=positions,
                    slots=slots,
                    valid_lengths=self.lengths[size],
                    cache=self.ple_cache,
                ),
            )
        mixed, inject = self.attn_hyper.read(x)
        forward = self.attention.decode_forward if decode else self.attention.prefill_forward
        if self.kind == "linear_attention":
            update = forward(mixed, current_positions=positions, slots=slots)
        else:
            update = forward(mixed, current_pos=positions, page_table=page_table, slots=slots)
        x = self.attn_hyper.write(x, update, inject)
        mixed, inject = self.mlp_hyper.read(x)
        return self.mlp_hyper.write(x, self.mlp(mixed), inject)

    def prefill_forward(self, hidden_states, *, current_positions, slots=None, tokens=None, page_table=None):
        if tuple(hidden_states.shape)[:2] != (1, self.batch_size) or hidden_states.shape[-1] != 10240:
            raise ValueError("Expected [1,B,S,10240] residual streams")
        length = hidden_states.shape[2]
        if not 1 <= length <= self.context:
            raise ValueError("Sequence length is outside the advertised context")
        if self.ple is not None and (tokens is None or slots is None):
            raise ValueError("PLE requires token IDs and request slots")
        # Reject missing setup before the first chunk can mutate request state.
        if self.ple is not None:
            sizes = {min(self.chunk_size, length - start) for start in range(0, length, self.chunk_size)}
            if not sizes.issubset(self.lengths):
                raise ValueError("Call prepare([sequence_length]) before executing this PLE prefill shape")
        outputs = []
        for start in range(0, length, self.chunk_size):
            stop = min(start + self.chunk_size, length)
            position = current_positions if start == 0 else ttnn.add(current_positions, start)
            x = hidden_states[:, :, start:stop, :]
            ids = None if tokens is None else tokens[:, start:stop]
            outputs.append(
                self._chunk(x, positions=position, slots=slots, tokens=ids, page_table=page_table, decode=False)
            )
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)

    def decode_forward(self, hidden_states, *, current_positions, slots=None, tokens=None, page_table=None):
        if tuple(hidden_states.shape) != (1, self.batch_size, 1, 10240):
            raise ValueError("Decode requires [1,B,1,10240]")
        return self._chunk(
            hidden_states, positions=current_positions, slots=slots, tokens=tokens, page_table=page_table, decode=True
        )
