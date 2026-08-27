# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Second half of the async-ahead token keep regression tests: no mocking.

Test A observes the keep's decision by replacing the decode. That proves the
eligibility rules are right, but not that the decision survives the rest of
``decode_forward`` -- the host re-staging in ``_decode_forward_trace_text`` and the
traced execution both run after it, and either could overwrite what the keep chose.

This half therefore runs ``decode_forward`` from entry to return with its real body:
nothing on ``Generator`` is patched, subclassed or wrapped. What is substituted is the
*model*, which the keep never touches -- it runs before any transformer layer and
only ever reads torch tensors and the two device trace-input buffers. So instead of a
checkpoint, this uses a stand-in that implements exactly the decode contract the
generator calls, and whose whole forward pass is "record what you were handed":
a token embedding whose row ``i`` is the constant ``i``, copied into a persistent
device buffer. Reading that buffer back after ``decode_forward`` returns therefore
yields the token ids the decode actually ran with, straight out of the device.

Using a real checkpoint here would be actively worse: the observable would be a
sampled token, which is what the device writes back over the trace inputs, and which
was measured to collide between the correct and the stale input anyway.
"""

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import Mode, copy_host_to_device
from models.tt_transformers.tt.generator import Generator

BATCH = 4
EMBED_DIM = 32
VOCAB = 256
POSITION = 10  # the position same-shaped serving requests keep reusing

# Deliberately far apart so a wrongly adopted leftover is unmistakable.
PRIMING_TOKENS = [101, 102, 103, 104]  # left on the device by an unrelated request
THIS_TOKENS = [11, 12, 13, 14]  # what the request under test is handed


class _ModelArgsStub:
    """The two attributes the generator reads off ``model_args`` on the decode path."""

    def __init__(self, mesh_device, max_batch_size):
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size


class _RecordingDecodeModel:
    """A stand-in implementing the model side of the decode contract, nothing more.

    ``ttnn_decode_forward`` embeds the tokens it is given through a table whose row
    ``i`` is the constant vector ``i``, then copies the result into a persistent
    device buffer. Both are ordinary ttnn ops, so they capture into the decode trace
    and re-run on every trace execution exactly like a real model's layers would.
    """

    def __init__(self, mesh_device, batch):
        self.mesh_device = mesh_device
        self.batch = batch
        self.mode = None
        self.sampling = None  # no on-device sampler: nothing writes back over the inputs
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
        """Host-side decode inputs, laid out as ``Transformer`` lays them out.

        Tokens are padded to a full tile and replicated; positions stay per-slot. The
        rope-index and page-table slots of the tuple are unused here and left empty --
        the generator propagates ``None`` entries through staging untouched.
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
            # Allocated on the eager pre-compile pass, which the generator always runs
            # before capturing the trace; allocation inside a capture is not allowed.
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
        """The token ids the last decode ran with, read back from the device."""
        witness = ttnn.to_torch(ttnn.get_device_tensors(self.token_witness)[0]).reshape(-1, EMBED_DIM)
        return [int(round(float(row[0]))) for row in witness[: self.batch]]


@pytest.fixture
def generator(mesh_device):
    model = _RecordingDecodeModel(mesh_device, BATCH)
    gen = Generator([model], [_ModelArgsStub(mesh_device, BATCH)], mesh_device)
    return gen, model


def _decode(gen, tokens, positions):
    """One ``decode_forward`` as the vLLM adapter issues it at a reset step.

    ``defer_device_sampling`` selects the on-device-sampling path -- the path the keep
    is gated on -- while returning before any sampler runs. ``reset_batch`` marks a
    step at which the batch composition changed, which is when the keep applies.
    """
    return gen.decode_forward(
        torch.tensor(tokens, dtype=torch.int32).reshape(len(tokens), 1),
        torch.tensor(positions, dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        defer_device_sampling=True,
        reset_batch=True,
    )


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 30000000}], indirect=True)
@pytest.mark.parametrize("supports_async_decode", [False, True], ids=["non_async", "async"])
def test_decode_forward_runs_with_the_expected_tokens(
    mesh_device, reset_seeds, ensure_gc, generator, supports_async_decode
):
    """End of ``decode_forward``: the tokens that reached the model are the right ones.

    The first call is an unrelated earlier request; it leaves its tokens and positions
    staged in the device trace inputs, which is the state a same-shaped follow-up
    request meets. The second call is issued at that same position, making the keep
    eligible. A model declaring ``supports_async_decode = False`` must nonetheless
    decode the tokens it was handed; before the fix it decoded the leftovers.
    """
    gen, model = generator
    gen.model_capabilities = {
        "supports_async_decode": supports_async_decode,
        "supports_sample_on_device": True,
    }

    _decode(gen, PRIMING_TOKENS, [POSITION] * BATCH)
    assert model.decoded_tokens() == PRIMING_TOKENS, "setup: the priming decode did not run as issued"

    _decode(gen, THIS_TOKENS, [POSITION] * BATCH)
    expected = PRIMING_TOKENS if supports_async_decode else THIS_TOKENS
    assert model.decoded_tokens() == expected, (
        f"decode ran with {model.decoded_tokens()}, expected {expected} "
        f"(supports_async_decode={supports_async_decode}, leftovers were {PRIMING_TOKENS})"
    )
    assert gen.mode is Mode.DECODE
