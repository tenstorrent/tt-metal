# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from models.common.llm_runtime import tensor_resources
from models.common.llm_runtime.decode import DecodeDeviceInputs, DecodePersistentInputs
from models.common.llm_runtime.prefill import PrefillDeviceInputs, PrefillPersistentInputs, PrefillPositionInputs


def test_owned_runtime_containers_release_aliased_tensors_once(monkeypatch):
    class FakeTensor:
        pass

    shared = FakeTensor()
    other = FakeTensor()
    released = []
    monkeypatch.setattr(tensor_resources.ttnn, "Tensor", FakeTensor)
    monkeypatch.setattr(tensor_resources.ttnn, "deallocate", released.append)

    decode_inputs = DecodeDeviceInputs(shared, other, shared, None)
    prefill_inputs = PrefillDeviceInputs(shared, other, None, shared, None, other, None)
    positions = PrefillPositionInputs(other, shared, other)
    values = (
        decode_inputs,
        DecodePersistentInputs(decode_inputs, (shared, other, shared)),
        prefill_inputs,
        positions,
        PrefillPersistentInputs(prefill_inputs, positions, (shared, other, shared), shared),
    )

    assert tensor_resources.best_effort_deallocate_owned_tensors(values) == []
    assert released == [shared, other]


def test_arbitrary_dataclass_is_not_treated_as_an_ownership_projection(monkeypatch):
    class FakeTensor:
        pass

    @dataclass
    class BorrowedValue:
        tensor: FakeTensor

    released = []
    monkeypatch.setattr(tensor_resources.ttnn, "Tensor", FakeTensor)
    monkeypatch.setattr(tensor_resources.ttnn, "deallocate", released.append)

    assert tensor_resources.best_effort_deallocate_owned_tensors(BorrowedValue(FakeTensor())) == []
    assert released == []


def test_raise_cleanup_failures_preserves_primary_and_attaches_rest(expect_error):
    primary = RuntimeError("primary")
    secondary = RuntimeError("secondary")

    with expect_error(RuntimeError, "primary") as raised:
        tensor_resources.raise_cleanup_failures((primary, secondary))

    assert raised.value is primary
    assert primary.cleanup_failures == (secondary,)
