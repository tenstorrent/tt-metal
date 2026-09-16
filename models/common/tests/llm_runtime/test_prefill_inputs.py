# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import models.common.llm_runtime.prefill.inputs as inputs_module
from models.common.llm_runtime.prefill.inputs import (
    PrefillDeviceInputs,
    PrefillHostInputs,
    PrefillInputStager,
    allocate_device_tensors,
    copy_into_device_tensors,
)


class _Model:
    def __init__(self, *, rotary_capacity=8, rotary_outputs=("cos", "sin")):
        self.config = SimpleNamespace(dim=64)
        self.rope_setup = SimpleNamespace(
            cos_matrix=torch.zeros(1, 1, rotary_capacity),
            load_device_weights=lambda: None,
        )
        self.rotary_outputs = rotary_outputs

    def prepare_prefill_rot_mats(self, position_indices):
        del position_indices
        return self.rotary_outputs


def _stager(*, model=None, released=None):
    released = [] if released is None else released
    return PrefillInputStager(
        model=_Model() if model is None else model,
        mesh_device="mesh",
        release_transient=lambda values: released.append(values) or [],
    )


def _patch_host_conversion(monkeypatch):
    converted = []
    monkeypatch.setattr(inputs_module.ttnn, "ReplicateTensorToMesh", lambda mesh: ("mapper", mesh))
    monkeypatch.setattr(
        inputs_module.ttnn,
        "from_torch",
        lambda value, **kwargs: converted.append((value.clone(), kwargs)) or value.clone(),
    )
    return converted


@pytest.mark.parametrize("shape", [(8,), (1, 2, 8)])
def test_prepare_host_inputs_rejects_non_matrix_tokens_before_conversion(monkeypatch, expect_error, shape):
    monkeypatch.setattr(inputs_module.ttnn, "from_torch", lambda *args, **kwargs: pytest.fail("converted"))

    with expect_error(ValueError, "rank 2"):
        _stager().prepare_host_inputs(torch.zeros(shape), torch.zeros(1, 1, dtype=torch.int32))


def test_prepare_host_inputs_rejects_negative_start_and_last_token_beyond_rotary_capacity(
    monkeypatch,
    expect_error,
):
    _patch_host_conversion(monkeypatch)
    stager = _stager(model=_Model(rotary_capacity=8))
    tokens = torch.zeros(1, 4, dtype=torch.long)
    page_table = torch.zeros(1, 1, dtype=torch.int32)

    with expect_error(ValueError, "start position must be nonnegative"):
        stager.prepare_host_inputs(tokens, page_table, start_pos=-1)
    with expect_error(ValueError, "exceeds rotary capacity 8"):
        stager.prepare_host_inputs(tokens, page_table, last_token_idx=8)


def test_prepare_host_inputs_clamps_padded_positions_to_last_rotary_entry(monkeypatch):
    converted = _patch_host_conversion(monkeypatch)

    _stager(model=_Model(rotary_capacity=8)).prepare_host_inputs(
        torch.zeros(1, 4, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.int32),
        start_pos=6,
    )

    position_indices = converted[1][0]
    assert position_indices.tolist() == [[6, 7, 7, 7]]


@pytest.mark.parametrize("relative_last,sequence_length", [(-1, 32), (32, 32), (0, 0)])
def test_prepare_position_inputs_rejects_positions_outside_padded_sequence(
    monkeypatch,
    expect_error,
    relative_last,
    sequence_length,
):
    monkeypatch.setattr(inputs_module.ttnn, "from_torch", lambda *args, **kwargs: pytest.fail("converted"))

    with expect_error(ValueError, "last-token position"):
        _stager().prepare_position_inputs_host(relative_last, sequence_length)


def test_allocate_device_tensors_releases_partial_allocation_on_failure(monkeypatch, expect_error):
    first_device = object()
    calls = []

    def to_device(host_tensor, *, device):
        calls.append((host_tensor, device))
        if len(calls) == 2:
            raise RuntimeError("allocation failed")
        return first_device

    released = []
    monkeypatch.setattr(inputs_module.ttnn, "to_device", to_device)
    monkeypatch.setattr(
        inputs_module,
        "best_effort_deallocate_owned_tensors",
        lambda values: released.append(tuple(values)) or [],
    )

    with expect_error(RuntimeError, "allocation failed"):
        allocate_device_tensors(("host-0", "host-1"), mesh_device="mesh")

    assert released == [(first_device,)]


def test_stage_device_inputs_releases_raw_and_malformed_rotary_outputs(monkeypatch, expect_error):
    raw = ["tokens", "positions", "page", None, None]
    released = []
    model = _Model(rotary_outputs=("cos-only",))
    monkeypatch.setattr(inputs_module, "allocate_device_tensors", lambda values, *, mesh_device: raw)

    host = PrefillHostInputs("host-tokens", "host-positions", "host-page", None, None)
    with expect_error(ValueError, "cosine and sine"):
        _stager(model=model, released=released).stage_device_inputs(host)

    assert released == [(model.rotary_outputs, raw)]


def test_copy_rotary_inputs_rejects_malformed_output_count_and_releases_it(monkeypatch, expect_error):
    released = []
    model = _Model(rotary_outputs=("cos-only",))
    device = PrefillDeviceInputs("tokens", "cos", "sin", "page", None, "positions", None)
    monkeypatch.setattr(inputs_module.ttnn, "copy", lambda **kwargs: pytest.fail("copied"))

    with expect_error(ValueError, "cosine and sine"):
        _stager(model=model, released=released).copy_rotary_inputs(device)

    assert released == [model.rotary_outputs]


@pytest.mark.parametrize(
    ("host", "device"),
    [
        ((None,), ("device",)),
        (("host",), (None,)),
        (("host", None), ("device", "unexpected-device")),
        (("host", None), ("device",)),
    ],
)
def test_copy_into_device_tensors_rejects_structure_changes_before_copy(
    monkeypatch,
    expect_error,
    host,
    device,
):
    monkeypatch.setattr(
        inputs_module.ttnn,
        "copy_host_to_device_tensor",
        lambda *args: pytest.fail("copied"),
    )

    with expect_error(ValueError, "host/device"):
        copy_into_device_tensors(host, device)
