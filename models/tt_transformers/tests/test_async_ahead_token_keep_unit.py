# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the async-ahead token keep in ``Generator.decode_forward``.

``decode_forward`` adopts the token/position a previous decode left in the device
trace-input buffers whenever the device position is continuous with the host view
(``dev_pos == host_pos`` or ``host_pos + 1``). That is correct under vLLM async
scheduling, where the host token state legitimately trails the device by one step.

The keep applied that assumption unconditionally, but it is not universal: it is a
property a model declares through ``model_capabilities["supports_async_decode"]``.
A model declaring ``False`` never has the one-step lag, so the only thing the keep
can do there is adopt a stale token from an unrelated earlier request whose position
happens to coincide -- silently changing what the model conditions on.

This is a bug in the shared ``Generator``, not in any one model: the affected code is
model-independent and every model declaring ``supports_async_decode = False``
alongside on-device sampling inherits it. It was first observed on Qwen3.6-27B
p150x4, where answers were truncated at the second token with an early ``<|im_end|>``
once enough same-shaped requests had reused a position.

Because the keep only reads torch tensors and the two device trace-input buffers --
it runs before any transformer layer -- neither test below needs a checkpoint, model
config or weights. A device is needed only to hold the trace-input tensors.

Test A (this half) mocks the decode away and observes the keep's decision directly:
it is the exhaustive truth table for the eligibility rules. It deliberately says
nothing about whether that decision reaches the model; test B covers that.
"""

import pytest
import torch

import ttnn
from models.tt_transformers.tt.generator import Generator

# Deliberately far-apart ids so a wrongly adopted leftover is unmistakable in the
# assertion message rather than a plausible neighbour of the expected token.
HOST_TOKENS = [1, 2, 3, 4]
DEVICE_TOKENS = [101, 102, 103, 104]
DEVICE_POSITIONS = [10, 11, 12, 13]
# Per slot, this exercises every branch of the eligibility rule in one shot:
#   slot 0: dev_pos == host_pos                  -> continuous, keep eligible
#   slot 1: dev_pos == host_pos + 1              -> async-ahead by one, keep eligible
#   slot 2: continuous but freshly prefilled     -> host token is authoritative
#   slot 3: dev_pos unrelated to host_pos        -> not continuous, host token
HOST_POSITIONS = [10, 10, 12, 50]
PREFILLED_SLOTS = {2}
BATCH = len(HOST_TOKENS)

# What the keep produces when it is allowed to fire, i.e. for a model that really is
# async-ahead. Slots 2 and 3 fall back to the host values even then.
KEEP_TOKENS = [101, 102, 3, 4]
KEEP_POSITIONS = [10, 11, 12, 50]
# What a model that is not async-ahead must decode: exactly what it was handed.
HOST_ONLY_POSITIONS = [10, 10, 12, 50]


class _SwitchOnlyModel:
    """The entire model surface ``decode_forward`` touches before the keep."""

    def switch_mode(self, mode):
        pass


class _KeepProbe(Generator):
    """A ``Generator`` reduced to the state the keep reads, with the decode mocked out.

    ``Generator.__init__`` is bypassed on purpose: it builds prefill/decode trace
    bookkeeping that the keep never looks at, and going through it would require a
    real model. Only the attributes ``decode_forward`` touches on its way to the keep
    are set here, so the test cannot accidentally depend on anything else.
    """

    def __init__(self, mesh_device, device_tokens, device_positions, capabilities):
        self.mesh_device = mesh_device
        self.model = [_SwitchOnlyModel()]
        self.model_args = [None]
        self.data_parallel = 1
        self.mode = None
        self.decoded_with = None
        if capabilities is not None:
            # Capabilities live on the vLLM adapter subclass in production (e.g.
            # Qwen36ForCausalLM); leaving them unset exercises the inherited default.
            self.model_capabilities = capabilities
        # The buffers a previous decode would have left staged on the device.
        self.trace_inputs_decode = {
            True: [
                [
                    _replicated(mesh_device, device_tokens, ttnn.uint32),
                    _replicated(mesh_device, device_positions, ttnn.int32),
                    None,
                    None,
                ]
            ]
        }

    def _decode_forward_trace_text(self, tokens, current_pos, **kwargs):
        """Mocked: record what the keep decided to decode with, and run nothing."""
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


def _run_keep(mesh_device, capabilities, slot_remap=None):
    """Drive ``decode_forward`` up to the keep and report the decision it made.

    ``defer_device_sampling`` selects the on-device-sampling path (which is what makes
    the keep eligible) while returning before the sampler runs, so no sampling module
    is needed. ``reset_batch`` marks this as a reset step, as vLLM does when a request
    joins or leaves the batch -- the only steps at which the keep applies.
    """
    probe = _KeepProbe(mesh_device, DEVICE_TOKENS, DEVICE_POSITIONS, capabilities)
    probe._slots_prefilled_since_decode = set(PREFILLED_SLOTS)
    probe.decode_forward(
        torch.tensor(HOST_TOKENS, dtype=torch.int32).reshape(BATCH, 1),
        torch.tensor(HOST_POSITIONS, dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        defer_device_sampling=True,
        reset_batch=True,
        slot_remap=slot_remap,
    )
    return probe.decoded_with


@torch.no_grad()
@pytest.mark.parametrize(
    "capabilities, expected_tokens, expected_positions",
    [
        pytest.param(
            {"supports_async_decode": False, "supports_sample_on_device": True},
            HOST_TOKENS,
            HOST_ONLY_POSITIONS,
            id="declared_non_async_decodes_its_own_token",
        ),
        pytest.param(
            {"supports_async_decode": True, "supports_sample_on_device": True},
            KEEP_TOKENS,
            KEEP_POSITIONS,
            id="declared_async_keeps_the_ahead_token",
        ),
        pytest.param(
            None,
            KEEP_TOKENS,
            KEEP_POSITIONS,
            id="undeclared_defaults_to_async",
        ),
    ],
)
def test_keep_follows_the_declared_capability(
    mesh_device, reset_seeds, ensure_gc, capabilities, expected_tokens, expected_positions
):
    """The keep must follow ``supports_async_decode``, not assume async scheduling.

    The non-async case is the regression: without the capability check the keep fires
    on slots 0 and 1 and the model conditions on tokens from an unrelated request.
    The async and undeclared cases pin the behaviour that must not change.
    """
    tokens, positions = _run_keep(mesh_device, capabilities)
    assert tokens == expected_tokens, (
        f"decoded tokens {tokens}, expected {expected_tokens} "
        f"(device had {DEVICE_TOKENS} staged at {DEVICE_POSITIONS})"
    )
    assert positions == expected_positions, f"decoded positions {positions}, expected {expected_positions}"


@torch.no_grad()
def test_force_env_var_restores_the_unconditional_keep(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """``TT_FORCE_ASYNC_AHEAD_KEEP=1`` is the escape hatch back to the old behaviour."""
    monkeypatch.setenv("TT_FORCE_ASYNC_AHEAD_KEEP", "1")
    tokens, positions = _run_keep(mesh_device, {"supports_async_decode": False, "supports_sample_on_device": True})
    assert (tokens, positions) == (KEEP_TOKENS, KEEP_POSITIONS)


@torch.no_grad()
def test_slot_remap_is_honoured_by_the_keep(mesh_device, reset_seeds, ensure_gc):
    """Condense moves must still be honoured for a model that does keep the token.

    ``slot_remap`` carries GLOBAL slot indices; with a single rank that is a
    permutation of ``[0, BATCH)``. Swapping the first two slots swaps which device
    entry each host slot is compared against, so the kept tokens come back swapped
    too -- evidence that the keep reads the remapped device state and not the raw
    buffer order.
    """
    swap_first_two = [1, 0, 2, 3]
    tokens, positions = _run_keep(
        mesh_device,
        {"supports_async_decode": True, "supports_sample_on_device": True},
        slot_remap=swap_first_two,
    )
    # Without the remap this is ``KEEP_TOKENS`` == [101, 102, 3, 4]: slot 0 matches
    # device position 10 exactly and slot 1 is one ahead at 11. After the swap slot 0
    # sees position 11 (host + 1) and slot 1 sees 10 (equal), so both still keep -- but
    # the tokens they keep are exchanged.
    assert tokens == [102, 101, 3, 4], f"decoded tokens {tokens} do not reflect the remap"
    assert positions == [11, 10, 12, 50], f"decoded positions {positions} do not reflect the remap"
