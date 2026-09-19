# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A graduated module is not un-graduated by a regex that never ran it.

RUN_REPORT.md claimed for nine days that two modules ran on CPU. Both had PASSED the on-device
graduation gate, and their runtime probes record what they actually dispatch:

    llama_attention          ttnn_dispatch=12  torch_ops=0
    llama_rotary_embedding   ttnn_dispatch=2   torch_ops=0

`_stub_body_is_native` prefers that probe and falls back to reading the source when the sidecar is
missing or older than the stub. The fallback rejected both, for reasons that had nothing to do with
whether they run on the device:

  * `torch.full((batch,), int(pos))` in llama_attention -- an index built to hand to ttnn -- matched
    a regex that treats ANY `torch.<fn>(` in the compute path as host compute.
  * `with torch.no_grad():` in llama_rotary_embedding's __init__ -- one-time rope-table prep -- was
    judged as a per-forward torch fallback, though the function's own docstring says weight prep in
    __init__ is allowed.

Neither is decidable from text; both are decidable by running, which is what the probe does. What IS
decidable statically is kept: delegation to the torch reference (its PCC pass would be meaningless,
the output being the reference itself), a bare `.to_torch(` readback, and a compute path that calls
torch while dispatching NO ttnn at all -- a host reimplementation.
"""

import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from scripts.tt_hw_planner.bringup_loop import _stub_body_is_native  # noqa: E402


def _stub(tmp, name, body):
    p = Path(tmp) / name
    p.write_text(body)
    return p


_INDEX_PLUMBING = """\
import torch
import ttnn


class _T:
    def __init__(self, device, torch_module):
        self.device = device

    def __call__(self, x, pos=0, batch=1):
        idx = torch.full((batch,), int(pos), dtype=torch.int32)
        idx_tt = ttnn.from_torch(idx, device=self.device)
        return ttnn.embedding(idx_tt, x)
"""

_INIT_WEIGHT_PREP = """\
import torch
import ttnn


class _T:
    def __init__(self, device, torch_module):
        with torch.no_grad():
            cos, sin = torch_module(torch.zeros(1, 8, 1), torch.arange(0, 8).unsqueeze(0))
        self.cos = ttnn.from_torch(cos.float(), device=device)

    def __call__(self, x):
        return ttnn.mul(x, self.cos)
"""

_HOST_REIMPLEMENTATION = """\
import torch


class _T:
    def __init__(self, device, torch_module):
        self.device = device

    def __call__(self, x):
        return torch.softmax(x, dim=-1)
"""

_DELEGATES = """\
import ttnn


class _T:
    def __init__(self, device, torch_module):
        self._torch_module = torch_module

    def __call__(self, x):
        return self._torch_module.forward(x)
"""


def test_an_index_built_for_ttnn_is_not_host_compute(tmp_path):
    """THE BUG, on llama_attention's shape: a torch tensor constructed only to be handed to ttnn."""
    assert _stub_body_is_native(_stub(tmp_path, "a.py", _INDEX_PLUMBING)) is True


def test_weight_prep_in_init_is_allowed_as_documented(tmp_path):
    """llama_rotary_embedding's shape: the rope table built once under no_grad at construction.
    The docstring always said this was allowed; the walk checked every method anyway."""
    assert _stub_body_is_native(_stub(tmp_path, "b.py", _INIT_WEIGHT_PREP)) is True


def test_a_forward_that_dispatches_nothing_is_still_rejected(tmp_path):
    """The protection this check exists for. `return torch.softmax(x)` computes on the host, so its
    PCC pass proves nothing -- and no ttnn is dispatched anywhere in the path."""
    assert _stub_body_is_native(_stub(tmp_path, "c.py", _HOST_REIMPLEMENTATION)) is False


def test_delegating_to_the_reference_is_still_rejected(tmp_path):
    """A stub whose forward returns the torch reference's own output is ~PCC 1.0 by construction."""
    assert _stub_body_is_native(_stub(tmp_path, "d.py", _DELEGATES)) is False


def test_the_real_voxtral_stubs_read_native_without_any_probe():
    """The end-to-end condition that produced the wrong report: no usable probe sidecar anywhere.

    Before: 15 ON_DEVICE / 2 CPU_REUSE. After: 17 / 0 -- matching what the hardware measured.
    """
    demo = _ROOT / "models" / "tt_transformers" / "demo" / "voxtral_mini_3b_2507"
    if not (demo / "_stubs").is_dir():
        return  # the model is not checked out in this tree
    tmp = Path(tempfile.mkdtemp())
    for f in sorted((demo / "_stubs").glob("*.py")):
        shutil.copy(f, tmp / f.name)  # deliberately WITHOUT the probe sidecars
        assert _stub_body_is_native(tmp / f.name) is True, f.name
