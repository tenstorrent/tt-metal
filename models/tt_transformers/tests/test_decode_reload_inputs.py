# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify that traced decode follows the explicit input reload command.

The caller owns the reload decision. With ``reload_inputs=True``, decode copies
the supplied host token and position into the trace inputs. With
``reload_inputs=False``, decode keeps the device-resident values from the prior
step. Model capabilities must not change this command.
"""

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.generator import Generator

BATCH = 4
EMBED_DIM = 32
VOCAB = 256
POSITION = 10

RESIDENT_TOKENS = [101, 102, 103, 104]
HOST_TOKENS = [11, 12, 13, 14]


class _ModelArgsStub:
    """Provide the model arguments used by the shared decode path."""

    def __init__(self, mesh_device, max_batch_size):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size


class _RecordingDecodeModel:
    """Record the tokens consumed by traced decode."""

    def __init__(self, mesh_device, batch):
        self.mesh_device = mesh_device
        self.batch = batch
        self.mode = None
        self.sampling = None
        table = torch.arange(VOCAB, dtype=torch.float32).unsqueeze(-1).repeat(1, EMBED_DIM)
        self.embedding_weights = ttnn.from_torch(
            table.reshape(1, 1, VOCAB, EMBED_DIM),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.token_witness = None

    def switch_mode(self, mode):
        self.mode = mode

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        padded = torch.nn.functional.pad(tokens.reshape(-1), (0, 32 - tokens.shape[0]), "constant", 0)
        tt_tokens = ttnn.unsqueeze_to_4D(
            ttnn.from_torch(
                padded,
                device=None,
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        )
        tt_pos = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tt_tokens, tt_pos, None, None

    def prepare_inputs_decode(self, *inputs):
        return copy_host_to_device(self.prepare_decode_inputs_host(*inputs), mesh_device=self.mesh_device)

    def ttnn_decode_forward(self, tokens, current_pos, rot_mat_idxs=None, page_table=None, **kwargs):
        embedded = ttnn.embedding(tokens, self.embedding_weights, layout=ttnn.ROW_MAJOR_LAYOUT)
        if self.token_witness is None:
            # The eager precompile pass allocates this buffer before trace capture.
            self.token_witness = ttnn.from_torch(
                torch.zeros(tuple(embedded.shape), dtype=torch.float32),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        ttnn.copy(embedded, self.token_witness)
        return embedded

    def decoded_tokens(self):
        witness = ttnn.to_torch(ttnn.get_device_tensors(self.token_witness)[0]).reshape(-1, EMBED_DIM)
        return [int(round(float(row[0]))) for row in witness[: self.batch]]


@pytest.fixture
def generator(mesh_device):
    model = _RecordingDecodeModel(mesh_device, BATCH)
    return Generator([model], [_ModelArgsStub(mesh_device, BATCH)], mesh_device), model


def _decode(gen, tokens, *, reload_inputs):
    return gen.decode_forward(
        torch.tensor(tokens, dtype=torch.int32).reshape(len(tokens), 1),
        torch.tensor([POSITION] * len(tokens), dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        # Select the device-sampling trace without requiring a sampling module.
        sampling_params=None,
        defer_device_sampling=True,
        reload_inputs=reload_inputs,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000}], indirect=True)
def test_reload_inputs_controls_traced_decode(mesh_device, reset_seeds, ensure_gc, generator):
    gen, model = generator

    _decode(gen, RESIDENT_TOKENS, reload_inputs=True)
    assert model.decoded_tokens() == RESIDENT_TOKENS

    _decode(gen, HOST_TOKENS, reload_inputs=False)
    assert model.decoded_tokens() == RESIDENT_TOKENS

    _decode(gen, HOST_TOKENS, reload_inputs=True)
    assert model.decoded_tokens() == HOST_TOKENS
    assert gen.mode is Mode.DECODE
