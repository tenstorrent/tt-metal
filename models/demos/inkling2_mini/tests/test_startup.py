# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Demo startup progress (demo/startup.py): no device, no ttnn.

pytest models/demos/inkling2_mini/tests/test_startup.py
"""

import json
import os
import time

from models.demos.inkling2_mini.demo.startup import KernelCount, Startup


def test_kernel_count(tmp_path):
    old = tmp_path / "key" / "kernels" / "old_kernel" / "123"
    old.mkdir(parents=True)
    past = time.time() - 3600
    os.utime(old, (past, past))
    os.utime(old.parent, (past, past))
    k = KernelCount(tmp_path)
    assert k() == 0
    (tmp_path / "key" / "kernels" / "old_kernel" / "456").mkdir()
    (tmp_path / "key" / "kernels" / "new_kernel" / "789").mkdir(parents=True)
    assert k() == 2


def test_phases_and_record(tmp_path):
    lines = []
    rec = tmp_path / "startup.json"
    st = Startup(lines.append, key="k", every=3600, record=rec, kernels=lambda: 0)
    assert st.line() is None
    st.phase("weights", tensors=10, gb=2.0)
    st.tensors, st.bytes = 5, 10**9  # as the load_tensor wrapper counts
    assert "tensors 5/10, 1.0/2.0 GB" in st.line() and "eta" in st.line()
    st.phase("traces")
    st.traces = 3
    r = st.done()
    assert set(r["phases"]) == {"weights", "traces"} and r["traces"] == 3 and r["tensors"] == 5
    assert lines[0] == "startup: weights ..." and "model ready in" in lines[-1]
    # the next start knows the counts and phase times
    st2 = Startup(lines.append, key="k", every=3600, record=rec, kernels=lambda: 0)
    st2.phase("traces")
    assert "last start" in lines[-1]
    st2.traces = 1
    assert "traces 1/3" in st2.line()
    st2.fail()
    assert json.loads(rec.read_text())["k"]["traces"] == 3
