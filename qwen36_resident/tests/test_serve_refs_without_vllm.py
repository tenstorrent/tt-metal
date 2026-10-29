# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The direct references (serve/greedy_ref.py, mtp_ref.py, dflash_ref.py) run in the tt-metal python, which has no
vLLM: they must import without it (dflash_ref once took its decode constants from generator_vllm, which imports vLLM,
and the device run failed at import). CPU only: a subprocess with vllm and vllm_tt_plugin blocked.

usage: PYTHONPATH=<tt-metal>:<tt-metal>/ttnn python -m pytest --noconftest -c /dev/null \\
    qwen36_resident/tests/test_serve_refs_without_vllm.py
"""
import os
import subprocess
import sys

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CODE = """
import importlib.abc, sys
class NoVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("vllm", "vllm_tt_plugin"):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked)")
sys.meta_path.insert(0, NoVllm())
import qwen36_resident.serve.greedy_ref, qwen36_resident.serve.mtp_ref, qwen36_resident.serve.dflash_ref
from qwen36_resident.spec_serving import DFLASH_BLOCK, DFLASH_POS_MARGIN
print("IMPORTED", DFLASH_BLOCK, DFLASH_POS_MARGIN)
"""


def test_the_direct_references_import_without_vllm():
    env = dict(os.environ, PYTHONPATH=f"{WT}:{WT}/ttnn", VLLM_PLUGINS="")
    p = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=600, cwd=WT)
    assert "IMPORTED 8 80" in p.stdout, p.stdout[-2000:] + p.stderr[-4000:]


def test_the_served_classes_use_the_references_constants():
    """generator_vllm takes its decode constants from spec_serving, so served and direct decode cannot drift"""
    src = open(f"{WT}/qwen36_resident/generator_vllm.py").read()
    assert "from qwen36_resident.spec_serving import" in src
    for name in ("MTP_BLOCK =", "MTP_DEPTH =", "DFLASH_BLOCK =", "DFLASH_POS_MARGIN ="):
        assert name not in src, f"{name} redefined in generator_vllm"
