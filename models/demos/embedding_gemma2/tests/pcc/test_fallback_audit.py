# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runtime fallback audit of one text forward (inputs already on device).

During `TextModel.forward`:
  * every torch function call is recorded with a TorchFunctionMode (must be none),
  * host<->device transfer entry points of ttnn raise if called,
  * ttnn's `throw_exception_on_fallback` is on, so an op that silently falls back to torch raises.
The same forward is also captured in a trace by test_trace.py, which rejects host reads/writes.
"""

import pytest
import torch
from loguru import logger
from torch.overrides import TorchFunctionMode

import ttnn
from models.demos.embedding_gemma2.tests.common import make_batch, upload_inputs

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


@pytest.mark.parametrize("lengths", [(1025, 513, 31)], ids=["b3_pad"])
def test_no_host_fallback_in_forward(lengths, eg2_device, tt_text_model, eg2_snapshot, monkeypatch):
    ids, mask = make_batch(eg2_snapshot, list(lengths))
    ids_d, keep_d, S = upload_inputs(eg2_device, ids, mask, tt_text_model.config.max_seq_len)
    tt_text_model.prepare(S)
    tt_text_model(ids_d, keep_d)  # compile pass
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
        last, emb = tt_text_model(ids_d, keep_d)
    for n, f in originals.items():
        monkeypatch.setattr(ttnn, n, f)
    ttnn.synchronize_device(eg2_device)

    logger.info(f"torch calls inside forward: {len(rec.calls)} {sorted(set(rec.calls))[:20]}")
    assert rec.calls == [], f"torch functions called inside the forward: {sorted(set(rec.calls))}"
    out = ttnn.to_torch(emb).float()
    assert torch.isfinite(out).all()
