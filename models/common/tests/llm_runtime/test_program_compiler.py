# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import pytest
import torch

import models.common.llm_runtime.program_compiler as program_compiler_module
import ttnn
from models.common.llm_runtime.program_compiler import OutputSpec, ProgramCompiler, ProgramKey


@dataclass(frozen=True)
class _Signature:
    mode: str
    batch: int
    optional: int | None = None
    runtime_token: int = field(default=0, compare=False)

    @property
    def key_material(self):
        return (("mode", self.mode), ("batch", self.batch), ("optional", self.optional))


def _compiler(context=object()):
    return ProgramCompiler("mesh", lambda: context)


def _patch_sync(monkeypatch, events=None):
    events = [] if events is None else events
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: events.append(("sync", mesh)))
    return events


def test_program_key_has_stable_golden_digest_and_separate_trace_domain():
    from models.common.llm_runtime.trace_compiler import TraceKey

    signature = _Signature("decode", 32)

    assert ProgramKey.from_signature(signature).digest == (
        "c807e5574b34f3f83d386fc441580d1ae499aa534c8a426541dbeafdefa2ffdb"
    )
    assert TraceKey.from_signature(signature).digest == (
        "27bea60de6b500f448e680d3423ad62efbd01294c254524e37c114ff8b5dff35"
    )
    assert ProgramKey.from_signature(signature).digest != TraceKey.from_signature(signature).digest


def test_program_key_changes_for_each_material_field_but_not_runtime_values():
    base = _Signature("decode", 32)

    assert ProgramKey.from_signature(base) != ProgramKey.from_signature(_Signature("prefill", 32))
    assert ProgramKey.from_signature(base) != ProgramKey.from_signature(_Signature("decode", 16))
    assert ProgramKey.from_signature(base) != ProgramKey.from_signature(_Signature("decode", 32, 1))
    assert ProgramKey.from_signature(base) == ProgramKey.from_signature(_Signature("decode", 32, runtime_token=99))


def test_compiler_memoizes_program_key_by_canonical_material(monkeypatch):
    calls = []
    digest = program_compiler_module.signature_digest
    monkeypatch.setattr(
        program_compiler_module,
        "signature_digest",
        lambda *args: calls.append(args) or digest(*args),
    )
    compiler = _compiler()

    first = compiler.key_for(_Signature("decode", 32))
    second = compiler.key_for(_Signature("decode", 32, runtime_token=99))

    assert second is first
    assert len(calls) == 1


def test_compiler_program_key_memo_tracks_mutated_material(monkeypatch):
    class MutableSignature:
        def __init__(self):
            self.value = 1

        def key_material(self):
            return (("value", self.value),)

    calls = []
    digest = program_compiler_module.signature_digest
    monkeypatch.setattr(
        program_compiler_module,
        "signature_digest",
        lambda *args: calls.append(args) or digest(*args),
    )
    compiler = _compiler()
    signature = MutableSignature()

    first = compiler.key_for(signature)
    signature.value = 2
    second = compiler.key_for(signature)

    assert second != first
    assert compiler.key_for(signature) is second
    assert len(calls) == 2


def test_compiler_program_key_memo_preserves_tagged_type_identity():
    compiler = _compiler()

    keys = {
        compiler.key_for(type("Signature", (), {"key_material": (("value", value),)})())
        for value in (True, 1, 1.0, "1")
    }

    assert len(keys) == 4


@pytest.mark.parametrize("material", [{"unordered": "mapping"}, ["list"], torch.zeros(1)])
def test_program_key_rejects_noncanonical_material(material, expect_error):
    class InvalidSignature:
        key_material = (("invalid", material),)

    with expect_error(TypeError, "key material"):
        ProgramKey.from_signature(InvalidSignature())


def test_same_digest_with_different_retained_signature_is_rejected(monkeypatch, expect_error):
    _patch_sync(monkeypatch)
    compiler = _compiler()
    forced_key = ProgramKey("0" * 64)
    monkeypatch.setattr(compiler, "key_for", lambda signature: forced_key)
    compiler.compile(_Signature("decode", 32), lambda context: torch.zeros(1))

    with expect_error(RuntimeError, "collision"):
        compiler.compile(_Signature("prefill", 1), lambda context: torch.zeros(1))


def test_compile_receives_exact_context_deduplicates_and_checks_output_contract(monkeypatch, expect_error):
    events = _patch_sync(monkeypatch)
    context = object()
    compiler = _compiler(context)
    calls = []
    expected = OutputSpec((2, 3), torch.bfloat16)

    program = compiler.compile(
        _Signature("decode", 32),
        lambda supplied: calls.append(supplied) or torch.zeros(2, 3, dtype=torch.bfloat16),
        expected_output_spec=expected,
    )
    duplicate = compiler.compile(
        _Signature("decode", 32),
        lambda supplied: (_ for _ in ()).throw(AssertionError("duplicate invoked")),
        expected_output_spec=expected,
    )

    assert program is duplicate
    assert calls == [context]
    assert program.signature == _Signature("decode", 32)
    assert program.output_spec == expected
    assert events == [("sync", "mesh"), ("sync", "mesh")]
    with expect_error(ValueError, "different output contract"):
        compiler.compile(
            _Signature("decode", 32),
            lambda supplied: torch.zeros(1),
            expected_output_spec=OutputSpec((1,), torch.float32),
        )


def test_compile_uses_explicit_result_value_and_owned_release_selector(monkeypatch):
    _patch_sync(monkeypatch)

    class OwnedTensor:
        pass

    @dataclass
    class InvocationResult:
        value: torch.Tensor
        owned: object

    released = []
    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", released.append)
    owned = OwnedTensor()
    compiler = _compiler()

    program = compiler.compile(
        _Signature("prefill", 1),
        lambda context: InvocationResult(torch.zeros(1, 2), owned),
        output_spec=lambda result: OutputSpec.from_value(result.value),
        release_output=lambda result: result.owned,
    )

    assert program.output_spec.shape == (1, 2)
    assert released == [owned]


def test_compile_release_failure_is_retryable_and_blocks_further_compile(monkeypatch, expect_error):
    _patch_sync(monkeypatch)

    class OwnedTensor:
        pass

    first = OwnedTensor()
    retry = OwnedTensor()
    attempts = []
    failure = RuntimeError("release once")

    def deallocate(value):
        attempts.append(value)
        if value is retry and attempts.count(retry) == 1:
            raise failure

    monkeypatch.setattr(ttnn, "Tensor", OwnedTensor)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    compiler = _compiler()

    with expect_error(RuntimeError, "Failed to deallocate") as caught:
        compiler.compile(
            _Signature("decode", 32),
            lambda context: (first, retry),
            output_spec=lambda output: OutputSpec((1,), "dtype"),
        )

    assert caught.value.cleanup_failures == (failure,)
    assert compiler.compile_orphan_count == 1
    with expect_error(RuntimeError, "unreleased compile outputs"):
        compiler.compile(_Signature("decode", 16), lambda context: torch.zeros(1))

    compiler.cleanup()
    assert attempts.count(first) == 1
    assert attempts.count(retry) == 2


def test_compile_gate_distinguishes_capture_from_activation(monkeypatch, expect_error):
    _patch_sync(monkeypatch)
    compiler = _compiler()
    compiler.set_trace_capture_in_progress(True)
    with expect_error(RuntimeError, "capture is in progress"):
        compiler.compile(_Signature("decode", 32), lambda context: torch.zeros(1))

    compiler.set_trace_capture_in_progress(False)
    compiler.set_trace_active(True)
    with expect_error(RuntimeError, "after trace activation"):
        compiler.compile(_Signature("decode", 32), lambda context: torch.zeros(1))


def test_cleanup_terminalizes_only_program_metadata(monkeypatch, expect_error):
    _patch_sync(monkeypatch)
    compiler = _compiler()
    program = compiler.compile(_Signature("decode", 32), lambda context: torch.zeros(1))

    compiler.cleanup()
    compiler.cleanup()

    assert not program.ready
    with expect_error(RuntimeError, "released"):
        compiler.require_compiled(program.key)
