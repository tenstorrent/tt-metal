# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The async-ahead token keep applies only to models that declare async decode.

At a reset step ``Generator.decode_forward`` may discard the host's token and
position for a slot and adopt whatever the previous decode left in the device
trace-input buffers, whenever the two positions are continuous
(``dev_pos == host_pos`` or ``host_pos + 1``). That recovers the authoritative
token under vLLM async scheduling, where the host trails the device by one step.
Without that lag there is nothing to recover, so for a model declaring
``model_capabilities["supports_async_decode"] = False`` the same rule can only
substitute a stale token from an earlier request that happened to reuse the
position.

The keep runs before any transformer layer and reads only torch tensors and the
device trace-input buffers, so these tests stub the decode out and observe the
decision directly; a device is needed only to hold those buffers.
``test_async_ahead_token_keep_decode.py`` covers the same rule through an
unstubbed ``decode_forward``.
"""

import pytest
import torch

import ttnn
from models.tt_transformers.tt.generator import Generator

# Host and device token ids are kept far apart so an assertion failure says which
# side the decode took.
HOST_TOKENS = [1, 2, 3, 4]
DEVICE_TOKENS = [101, 102, 103, 104]
DEVICE_POSITIONS = [10, 11, 12, 13]
# One slot per branch of the eligibility rule: continuous, continuous one step
# behind, continuous but re-prefilled, and unrelated.
HOST_POSITIONS = [10, 10, 12, 50]
PREFILLED_SLOTS = {2}
BATCH = len(HOST_TOKENS)

# Slots 0 and 1 are the only ones the keep may take, so the last two entries are
# the host's either way.
KEPT_TOKENS = [101, 102, 3, 4]
KEPT_POSITIONS = [10, 11, 12, 50]


class _SwitchOnlyModel:
    """The whole model surface ``decode_forward`` uses before reaching the keep."""

    def switch_mode(self, mode):
        pass


class _KeepProbe(Generator):
    """A ``Generator`` carrying just the state the keep reads.

    ``Generator.__init__`` is bypassed because it builds trace bookkeeping the
    keep never consults and cannot run without a real model.
    """

    def __init__(self, mesh_device, capabilities):
        self.mesh_device = mesh_device
        self.model = [_SwitchOnlyModel()]
        self.model_args = [None]
        self.data_parallel = 1
        self.mode = None
        self.decoded_with = None
        if capabilities is not None:
            self.model_capabilities = capabilities
        self.trace_inputs_decode = {
            True: [
                [
                    _replicated(mesh_device, DEVICE_TOKENS, ttnn.uint32),
                    _replicated(mesh_device, DEVICE_POSITIONS, ttnn.int32),
                    None,
                    None,
                ]
            ]
        }

    def _decode_forward_trace_text(self, tokens, current_pos, **kwargs):
        self.decoded_with = (
            [int(t) for t in tokens[0].reshape(-1)],
            [int(p) for p in current_pos[0].reshape(-1)],
        )
        return []


def _replicated(mesh_device, values, dtype):
    return ttnn.from_torch(
        torch.tensor([values], dtype=torch.int32),
        device=mesh_device,
        dtype=dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _decoded_tokens_and_positions(mesh_device, capabilities, slot_remap=None):
    """Run ``decode_forward`` and return what the keep handed to the decode."""
    probe = _KeepProbe(mesh_device, capabilities)
    probe._slots_prefilled_since_decode = set(PREFILLED_SLOTS)
    probe.decode_forward(
        torch.tensor(HOST_TOKENS, dtype=torch.int32).reshape(BATCH, 1),
        torch.tensor(HOST_POSITIONS, dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        # The keep is gated on on-device sampling; deferring it selects that path
        # without needing a sampling module.
        sampling_params=None,
        defer_device_sampling=True,
        reset_batch=True,
        slot_remap=slot_remap,
    )
    return probe.decoded_with


@torch.no_grad()
@pytest.mark.parametrize(
    ("capabilities", "expected_tokens", "expected_positions"),
    [
        pytest.param(
            {"supports_async_decode": False, "supports_sample_on_device": True},
            HOST_TOKENS,
            HOST_POSITIONS,
            id="non_async_model_decodes_the_tokens_it_was_given",
        ),
        pytest.param(
            {"supports_async_decode": True, "supports_sample_on_device": True},
            KEPT_TOKENS,
            KEPT_POSITIONS,
            id="async_model_keeps_the_device_token",
        ),
        pytest.param(
            None,
            KEPT_TOKENS,
            KEPT_POSITIONS,
            id="undeclared_capability_keeps_the_device_token",
        ),
    ],
)
def test_keep_follows_the_declared_capability(
    mesh_device, reset_seeds, ensure_gc, capabilities, expected_tokens, expected_positions
):
    tokens, positions = _decoded_tokens_and_positions(mesh_device, capabilities)
    assert tokens == expected_tokens, (
        f"decoded tokens {tokens}, expected {expected_tokens} "
        f"(device had {DEVICE_TOKENS} staged at {DEVICE_POSITIONS})"
    )
    assert positions == expected_positions, f"decoded positions {positions}, expected {expected_positions}"


@torch.no_grad()
def test_keep_reads_device_state_through_slot_remap(mesh_device, reset_seeds, ensure_gc):
    """A condense move must redirect which device slot each host slot compares against."""
    # Slot 0 now sees device position 11 (one ahead) and slot 1 sees 10 (equal), so
    # both remain eligible and return each other's token.
    tokens, positions = _decoded_tokens_and_positions(
        mesh_device,
        {"supports_async_decode": True, "supports_sample_on_device": True},
        slot_remap=[1, 0, 2, 3],
    )
    assert tokens == [102, 101, 3, 4], f"decoded tokens {tokens} do not reflect the remap"
    assert positions == [11, 10, 12, 50], f"decoded positions {positions} do not reflect the remap"
