# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `token_embed` for FLUX.2-klein-9B's text encoder.

Bound to `model.embed_tokens`: the `nn.Embedding(vocab=151936, hidden=4096)` that
turns token ids into the hidden states the encoder stack consumes.

`ttnn.embedding` is exactly this op, so the forward is one call. The setup work
is the table's layout: `ttnn.embedding` wants the weight ROW_MAJOR, shaped
`[1, 1, vocab, hidden]`, and returns TILE-layout activations — the same way
`models/tt_transformers/tt/embedding.py::Embedding` stages and calls it.

An embedding is a LOOKUP table, not a matmul weight, so under the TP principles
it stays REPLICATED — every chip holds the whole table and looks up the same
rows. That is why this component is single-phase: there is no axis to split that
would leave the math unchanged.
"""
from __future__ import annotations

import torch

import ttnn


class TtTokenEmbed:
    def __init__(self, device, weights, hidden_size) -> None:
        self.device = device
        self.weights = weights
        self.hidden_size = hidden_size

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("token_embed stub needs the torch module to source its table")

        weight = torch_module.weight.detach().to(torch.bfloat16)
        vocab_size, hidden_size = int(weight.shape[0]), int(weight.shape[1])

        return cls(
            device=device,
            weights=ttnn.from_torch(
                weight.reshape(1, 1, vocab_size, hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(device),
            ),
            hidden_size=hidden_size,
        )

    # -------------------------------------------------------------- forward

    def __call__(self, input_ids, *args, **kwargs):
        out = ttnn.embedding(
            input_ids,
            self.weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(out, tuple(input_ids.shape) + (self.hidden_size,))


# ------------------------------------------------------------------ helpers


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _replicate_mapper(device):
    if _num_devices(device) <= 1:
        return None
    return ttnn.ReplicateTensorToMesh(device)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtTokenEmbed.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def token_embed(device, torch_module=None):
    return TtTokenEmbed.build(device, torch_module)
