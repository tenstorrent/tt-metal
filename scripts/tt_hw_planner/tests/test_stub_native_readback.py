import importlib
from pathlib import Path

bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")

_MLP_ADAPTER = '''\
import torch
import ttnn


class _MLPTT:
    def __init__(self, device, torch_module):
        self.device = device

    def _mk(self, t):
        return ttnn.from_torch(t.float(), dtype=ttnn.float32, device=self.device)

    def _to_dev_fp32(self, x):
        if isinstance(x, torch.Tensor):
            return self._mk(x)
        try:
            return ttnn.typecast(x, ttnn.float32)
        except Exception:
            return self._mk(ttnn.to_torch(x).float())

    def __call__(self, x):
        xd = self._to_dev_fp32(x)
        return ttnn.linear(xd, self.W)
'''

_BARE_READBACK_HOST_MATH = '''\
import torch
import ttnn


class _T:
    def __init__(self, device, torch_module):
        self.device = device

    def __call__(self, x):
        y = x.to_torch()
        return torch.matmul(y, self.w)
'''

_TORCH_FORWARD = '''\
import torch
import ttnn


class _T:
    def __init__(self, device, torch_module):
        self.device = device

    def __call__(self, x):
        return torch.softmax(x, dim=-1)
'''


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src)
    return p


def test_ttnn_to_torch_input_adapter_is_native(tmp_path):
    stub = _write(tmp_path, "mlp.py", _MLP_ADAPTER)
    assert bl._stub_body_is_native(stub) is True


def test_bare_readback_with_host_math_is_rejected(tmp_path):
    stub = _write(tmp_path, "bad.py", _BARE_READBACK_HOST_MATH)
    assert bl._stub_body_is_native(stub) is False


def test_torch_call_in_forward_is_rejected(tmp_path):
    stub = _write(tmp_path, "bad2.py", _TORCH_FORWARD)
    assert bl._stub_body_is_native(stub) is False


def test_readback_regex_exempts_ttnn_prefix_only():
    assert bl._DEVICE_READBACK_RE.search("ttnn.to_torch(x)") is None
    assert bl._DEVICE_READBACK_RE.search("y.to_torch()") is not None
    assert bl._DEVICE_READBACK_RE.search("self._t.to_torch(z)") is not None
