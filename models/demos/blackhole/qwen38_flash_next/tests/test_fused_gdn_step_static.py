# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the fused GDN step: the CB table shared by the three kernels and the Python side, the
kernel argument layouts, the registry entry, and ``reference_step`` against tt/gdn.py's Qwen38GDN-style arithmetic."""

import re
from pathlib import Path

import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as module

KERNELS = Path(module.__file__).parent / "kernels"


def _constants(source: str) -> dict[str, int]:
    text = (KERNELS / source).read_text()
    return {name: int(value) for name, value in re.findall(r"\b(CB_[A-Z0-9]+)\s*=\s*(\d+)", text)}


def test_cb_indices_agree_between_kernels_and_python():
    compute, reader, writer = _constants("compute.cpp"), _constants("reader.cpp"), _constants("writer.cpp")
    for name, index in reader.items():
        assert compute[name] == index, name
    for name, index in writer.items():
        assert compute[name] == index, name
    declared = {index for index, _, _ in module.CBS} | {module.CB_SNEW, module.CB_OUTS, module.CB_DEBUG}
    used = {v for k, v in compute.items()}
    assert used <= declared, sorted(used - declared)
    assert max(declared) <= 31 and len(declared) == len({i for i in declared})
    assert (
        compute["CB_SNEW"] == module.CB_SNEW
        and compute["CB_OUTS"] == module.CB_OUTS
        and compute["CB_DBG"] == module.CB_DEBUG
    )
    # exact fp32 copies: only CBs the compute consumes with copy_tile; matmul/reduce/bcast operands must stay Default
    assert set(module.FP32_COPY_CBS) == {
        compute[n] for n in ("CB_STATE", "CB_DTNA", "CB_BETA", "CB_DECAY", "CB_SDECC", "CB_VREAD")
    }
    assert not set(module.FP32_COPY_CBS) & {
        compute[n] for n in ("CB_QROW", "CB_KROW", "CB_KCOL", "CB_SDEC", "CB_DELTAB", "CB_SNEW", "CB_SCALER", "CB_RS")
    }


def test_kernel_argument_layouts():
    reader = (KERNELS / "reader.cpp").read_text()
    assert reader.count("TensorAccessorArgs<") == 12  # projected, 3 slots, 4 taps, dtna, norm, state, newest
    assert reader.count("get_arg_val<uint32_t>(arg++)") == 13 + 2  # 12 addresses, items, then (lane, head)
    writer = (KERNELS / "writer.cpp").read_text()
    assert writer.count("TensorAccessorArgs<") == 3  # state, out, debug
    compute = (KERNELS / "compute.cpp").read_text()
    assert "get_arg_val<uint32_t>(0)" in compute and "DEBUG_TAPS" in compute
    assert (
        module.DEBUG_TILES == len(module.DEBUG_TAPS) == 28
    )  # conv sum, 12 conv, 4 q-norm statistics, beta, decay, 5 state-update taps, 4 o


def test_registry_entry():
    entry = fused.kernel("gdn_step")
    assert entry.tolerance == fused.COMPONENT
    assert (
        entry.component_proof
    )  # a COMPONENT-class kernel serves by default only with its recorded component-gate proof
    assert entry.fused is module.gdn_step and entry.composed is module.gdn_step_composed
    assert fused.resolve("gdn_step", {}) is module.gdn_step
    assert fused.resolve("gdn_step", {"QWEN38_FUSED_OFF": "gdn_step"}) is module.gdn_step_composed
    assert fused.resolve("gdn_step", {"QWEN38_FUSED": "gdn_step"}) is module.gdn_step
    assert entry.admits is module.admits


class _Fake:
    def __init__(self, shape, padded_shape=None, dtype=None):
        self.shape = tuple(shape)
        self.padded_shape = tuple(padded_shape) if padded_shape is not None else self.shape
        self.dtype = dtype


class _State:
    def __init__(self, recurrent):
        self.recurrent = recurrent


def _served_call(rows=1):
    projected = _Fake((1, 1, rows, module.PROJECTION_WIDTH), padded_shape=(1, 1, 32, module.PROJECTION_WIDTH))
    window = (object(), object(), object(), object())
    state = _State(_Fake((rows, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.FP32))
    return projected, window, state


def test_admission_is_the_fused_steps_input_contract():
    projected, window, state = _served_call()
    assert module.admits(None, projected, window, state) is True
    # a host fake reports its logical shape as its padded shape: outside the tile contract, the chain runs
    unpadded = _Fake((1, 1, 1, module.PROJECTION_WIDTH))
    assert module.admits(None, unpadded, window, state) is False
    assert module.admits(None, _Fake((1, 1, 1, 4096), padded_shape=(1, 1, 32, 4096)), window, state) is False
    assert (
        module.admits(
            None,
            _Fake((1, 1, 33, module.PROJECTION_WIDTH), padded_shape=(1, 1, 64, module.PROJECTION_WIDTH)),
            window,
            state,
        )
        is False
    )
    assert module.admits(None, projected, window[:3], state) is False
    assert (
        module.admits(
            None,
            projected,
            window,
            _State(_Fake((2, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.FP32)),
        )
        is False
    )
    assert (
        module.admits(
            None,
            projected,
            window,
            _State(_Fake((1, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.BF16)),
        )
        is False
    )
    assert module.admits(None, projected, window, object()) is False


def test_the_served_default_dispatches_per_call_and_the_switch_restores_the_chain(expect_error):
    step = fused.resolve_admitted("gdn_step", {})
    assert isinstance(step, fused.AdmittedStep)
    assert step.fused is module.gdn_step and step.composed is module.gdn_step_composed and step.admits is module.admits
    assert fused.resolve_admitted("gdn_step", {"QWEN38_FUSED_OFF": "gdn_step"}) is module.gdn_step_composed
    assert fused.resolve_admitted("gdn_step", {"QWEN38_FUSED": "gdn_step"}).fused is module.gdn_step
    # both branches of the dispatcher, on a probe kernel with the same admission rule
    calls = []
    probe = fused.FusedKernel(
        "probe_step",
        "a probe",
        fused.BITWISE,
        fused=lambda *args: calls.append("fused"),
        composed=lambda *args: calls.append("composed"),
        admits=module.admits,
    )
    dispatch = fused.AdmittedStep(probe)
    projected, window, state = _served_call()
    dispatch(None, _Fake((1, 1, 1, module.PROJECTION_WIDTH)), window, state)
    dispatch(None, projected, window, state)
    assert calls == ["composed", "fused"]

    with expect_error(ValueError):
        fused.AdmittedStep(
            fused.FusedKernel("plain_step", "a probe", fused.BITWISE, fused=lambda: None, composed=lambda: None)
        )


def test_reference_step_matches_the_oracle_functions():
    from models.demos.blackhole.qwen38_flash_next.reference import causal_depthwise_conv1d

    g = torch.Generator().manual_seed(5)
    rows = 3
    projected = (torch.randn(rows, module.PROJECTION_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    older = [(torch.randn(rows, module.QKV_WIDTH, generator=g) * 0.6).to(torch.bfloat16) for _ in range(3)]
    taps = [(torch.randn(module.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16) for _ in range(4)]
    dt, na = torch.randn(12, generator=g), -torch.exp(torch.rand(12, generator=g) * 3)
    norm = (1 + torch.randn(128, generator=g) * 0.1).to(torch.bfloat16)
    state = torch.randn(rows, 12, 128, 128, generator=g) * 0.4
    new_state, gated, conv, o, beta, decay = module.reference_step(projected, older, taps, dt, na, norm, state)
    assert new_state.shape == (rows, 12, 128, 128) and gated.shape == (rows, 1536) and gated.dtype == torch.bfloat16
    assert (
        conv.shape == (rows, 2560)
        and o.shape == (rows, 12, 128)
        and beta.shape == (rows, 12)
        and decay.shape == (rows, 12)
    )
    # the conv is tt/gdn.py's causal_depthwise_conv1d over the same window, lane by lane
    for lane in range(rows):
        window = torch.stack([*(s[lane] for s in older), projected[lane, : module.QKV_WIDTH]], dim=-1)  # [2560, 4]
        expected, _ = causal_depthwise_conv1d(
            window[:, 3:].unsqueeze(0),
            torch.stack(taps, dim=-1),
            initial_state=window[:, :3].unsqueeze(0),
            activation="silu",
        )
        assert torch.equal(expected[0, :, 0], conv[lane])
    assert torch.all(decay > 0) and torch.all(decay <= 1) and torch.all((beta > 0) & (beta < 1))
