# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The token the async-ahead keep settles on is the token the model decodes.

``test_async_ahead_token_keep_unit.py`` stubs the decode to check the keep's
eligibility rules. Those rules only matter if the outcome reaches the model, and
two things run in between: ``_decode_forward_trace_text`` re-stages the decode
inputs from host, and the trace then executes. Either could override the keep.

So ``decode_forward`` runs here unmodified and the model reports what it received.
The keep never touches the model, so a stand-in serves: it implements the decode
contract the generator calls, embeds its tokens through a table whose row ``i`` is
the constant ``i``, and copies the result into a device buffer that survives the
call. Reading that buffer back gives the token ids the decode ran with. A real
checkpoint would only expose a sampled token, which on-device sampling writes back
over the trace inputs and which can collide between the two candidate inputs.
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

# Kept far apart so an assertion failure says which request's tokens were decoded.
PRIMING_TOKENS = [101, 102, 103, 104]
THIS_TOKENS = [11, 12, 13, 14]


class _ModelArgsStub:
    """The ``model_args`` attributes the generator reads on the decode path."""

    def __init__(self, mesh_device, max_batch_size):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size


class _RecordingDecodeModel:
    """The model side of the decode contract, recording the tokens it is given.

    The embedding and the copy are ordinary ttnn ops, so they capture into the
    decode trace and replay on every execution as a real model's layers would.
    """

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
        """Decode inputs laid out as ``Transformer.prepare_decode_inputs_host`` lays them.

        The rope-index and page-table slots stay empty; the generator passes
        ``None`` entries through staging untouched.
        """
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
            # Allocated on the eager pre-compile pass the generator runs before
            # capturing the trace; allocation inside a capture is not allowed.
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


def _decode(gen, tokens, positions):
    return gen.decode_forward(
        torch.tensor(tokens, dtype=torch.int32).reshape(len(tokens), 1),
        torch.tensor(positions, dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        # The keep is gated on on-device sampling; deferring it selects that path
        # without needing a sampling module.
        sampling_params=None,
        defer_device_sampling=True,
        # The keep applies only where the batch composition changed.
        reset_batch=True,
    )


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000}], indirect=True)
@pytest.mark.parametrize("supports_async_decode", [False, True], ids=["non_async", "async"])
def test_decode_receives_the_tokens_the_capability_selects(
    mesh_device, reset_seeds, ensure_gc, generator, supports_async_decode
):
    gen, model = generator
    gen.model_capabilities = {
        "supports_async_decode": supports_async_decode,
        "supports_sample_on_device": True,
    }

    # An earlier request leaves its tokens and positions staged on the device; the
    # next request reuses the position, which is what makes the keep eligible.
    _decode(gen, PRIMING_TOKENS, [POSITION] * BATCH)
    assert model.decoded_tokens() == PRIMING_TOKENS, "the priming decode did not run as issued"

    _decode(gen, THIS_TOKENS, [POSITION] * BATCH)
    expected = PRIMING_TOKENS if supports_async_decode else THIS_TOKENS
    assert model.decoded_tokens() == expected, (
        f"decode ran with {model.decoded_tokens()}, expected {expected} "
        f"(supports_async_decode={supports_async_decode}, leftovers were {PRIMING_TOKENS})"
    )
    assert gen.mode is Mode.DECODE
