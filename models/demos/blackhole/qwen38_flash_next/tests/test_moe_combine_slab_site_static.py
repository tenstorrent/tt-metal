# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The model's switch site of the one-pass MoE combine program (``ttnn/moe.py``, read as text: the module needs the
runtime): the flag is the one-call slab with ``moe_combine`` enabled (the default; ``QWEN38_FUSED_OFF=moe_combine``), the owner row is built for it, the owned
buffers list names it, and ``_weighted_reduce_slab_blocks`` takes the program only when its admission holds, marks the
partial as the chain does, and keeps the 512-row blocks otherwise."""

from __future__ import annotations

import re

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_combine as mc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

MOE = (fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/moe.py").read_text()


def _body(name: str) -> str:
    match = re.search(rf"\n    def {name}\(.*?(?=\n    def |\n    @property|\Z)", MOE, re.S)
    assert match, name
    return match.group(0)


def test_flag_is_the_one_call_slab_under_the_switch():
    assert (
        "self.slab_combine_fused = (\n"
        '            self.slab_one_call and fused.resolve("moe_combine") is fused.kernel("moe_combine").fused\n'
        "        )\n"
    ) in MOE
    assert "    slab_combine_fused = False\n" in MOE  # the class default: instances built without __init__ in tests
    assert fused.kernel(mc.NAME).default_on  # the site serves the program unless QWEN38_FUSED_OFF names it


def test_owner_row_and_owned_buffers_cover_the_program():
    assert "if self.moe_post_fused or self.slab_combine_fused:" in MOE
    assert '("expert_owner",) if (self.moe_post_fused or self.slab_combine_fused) else ()' in MOE
    assert 'for name in ("expert_mapping", "local_combine_output", "expert_owner"):' in MOE  # released with the rest


def test_slab_reduce_takes_the_program_under_admission_and_keeps_the_blocks():
    body = _body("_weighted_reduce_slab_blocks")
    fused_call = body.index("fused.moe_combine.moe_combine(")
    assert body.index('if getattr(self, "slab_combine_fused", False) and fused.moe_combine.admits(') < fused_call
    assert "combine, routing.scores, routing.indices, self.expert_owner, memory_config=dram" in body
    marked = body.index(
        "self.mesh_contract.mark_local_partial(\n                partial, replicated_reference=full_hidden", fused_call
    )
    assert fused_call < marked < body.index('phase_observer("after-selective-reduce")\n            return partial')
    # the chain stays as the fallback, after the program's early return
    assert body.index("block = SLAB_REDUCE_BLOCK_ROWS") > marked
    assert "ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(" in body and "ttnn.concat(partials, dim=2" in body
    assert body.index('phase_observer("before-selective-reduce")') < body.index('if getattr(self, "slab_combine_fused"')


def test_fused_branch_marks_the_program_partial_and_slices_nothing(monkeypatch):
    """The site with the flag on: the program's output is marked as the chain's partial and returned; no slice, tilize,
    reduce or concat runs (the blocks path stays untouched behind the early return).  Fakes stand in for the tensors;
    the admission and the program are monkeypatched at the module the site reads them from."""

    from types import SimpleNamespace

    from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module

    calls, marked, phases = [], [], []
    monkeypatch.setattr(fused.moe_combine, "admits", lambda *a, **k: True)
    monkeypatch.setattr(fused.moe_combine, "moe_combine", lambda *a, **k: calls.append((a, k)) or "program-partial")
    monkeypatch.setattr(moe_module.ttnn, "slice", lambda *a, **k: pytest.fail("the blocks path ran"), raising=False)
    self = SimpleNamespace(
        slab_combine_fused=True,
        expert_owner="owner",
        rows=2048,
        mesh_contract=SimpleNamespace(mark_local_partial=lambda partial, **k: marked.append((partial, k))),
        row_contract=SimpleNamespace(full_hidden=(1, 1, 2048, 2560)),
    )
    routing = SimpleNamespace(scores="scores", indices="indices")
    out = moe_module.Qwen38TTNNMoE._weighted_reduce_slab_blocks(self, "combine", "full_hidden", routing, phases.append)
    assert out == "program-partial"
    assert calls == [(("combine", "scores", "indices", "owner"), {"memory_config": moe_module.ttnn.DRAM_MEMORY_CONFIG})]
    assert marked == [
        ("program-partial", {"replicated_reference": "full_hidden", "expected_shape": (1, 1, 2048, 2560)})
    ]
    assert phases == ["before-selective-reduce", "after-selective-reduce"]
    # a bare namespace without the flag (the landing's aliasing test) takes the blocks path
    bare = SimpleNamespace(rows=512)
    assert getattr(bare, "slab_combine_fused", False) is False


def test_program_signature_is_the_site_call():
    import inspect

    params = list(inspect.signature(mc.moe_combine).parameters)
    assert params == ["combine", "scores", "indices", "owner", "memory_config"]
    assert list(inspect.signature(mc.admits).parameters) == params
