# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runtime fallback audit of one audio forward (inputs uploaded and conv weights prepared beforehand).

During `AudioModel.forward`:
  * every torch function call is recorded with a TorchFunctionMode (must be none),
  * host<->device transfer entry points of ttnn raise if called,
  * ttnn's `throw_exception_on_fallback` is on, so an op that silently falls back to torch raises.
The Tracy op list of the same forward (test_audio_perf.py::test_tracy_audio_forward) is the
device-side evidence: every op in the signposted window is a device op.
"""

import pytest
import torch
from loguru import logger
from torch.overrides import TorchFunctionMode

import ttnn
from models.demos.embedding_gemma2.tests.pcc.audio.audio_common import device_inputs

_HOST_ENTRY_POINTS = (
    "to_torch",
    "from_torch",
    "as_tensor",
    "to_device",
    "from_device",
    "copy_host_to_device_tensor",
    "copy_device_to_host_tensor",
)


class _RecordTorch(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append(getattr(func, "__qualname__", repr(func)))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("case", ["b2_s10_s2"])
def test_no_host_fallback_in_audio_forward(case, eg2_device, tt_audio_encoder, audio_case, monkeypatch):
    feats, mask = audio_case(case)
    enc = tt_audio_encoder
    _, x = device_inputs(enc, feats, mask)
    enc.model(x)  # compile pass
    ttnn.synchronize_device(eg2_device)

    def _forbidden(name):
        def raiser(*_a, **_k):
            raise AssertionError(f"host transfer ttnn.{name} called inside the forward")

        return raiser

    originals = {n: getattr(ttnn, n) for n in _HOST_ENTRY_POINTS}
    for n in _HOST_ENTRY_POINTS:
        monkeypatch.setattr(ttnn, n, _forbidden(n))
    rec = _RecordTorch()
    with ttnn.manage_config("throw_exception_on_fallback", True), rec:
        soft, hidden = enc.model(x)
    for n, f in originals.items():
        monkeypatch.setattr(ttnn, n, f)
    ttnn.synchronize_device(eg2_device)

    logger.info(f"torch calls inside audio forward: {len(rec.calls)} {sorted(set(rec.calls))[:20]}")
    assert rec.calls == [], f"torch functions called inside the forward: {sorted(set(rec.calls))}"
    assert torch.isfinite(ttnn.to_torch(soft).float()).all()
