# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace as S

import pytest


class Tensor:
    def __init__(self, shape, rows=None):
        self.shape = shape
        self.rows = rows
        self.dead = False

    def deallocate(self, force):
        assert force is True and not self.dead
        self.dead = True


class Native:
    DRAM_MEMORY_CONFIG = object()

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []
        self.created = []

    def slice(self, x, start, end, **kw):
        assert not x.dead and kw == {"memory_config": self.DRAM_MEMORY_CONFIG}
        t = Tensor((1, 1, 128, 1024), list(range(start[2], end[2])))
        self.created.append(t)
        return t

    def matmul(self, x, w, **kw):
        assert not x.dead
        self.calls.append((x, w, kw))
        if self.fail == len(self.calls):
            raise RuntimeError("native failure")
        t = Tensor((*list(x.shape)[:-1], 2944), x.rows)
        self.created.append(t)
        return t

    def concat(self, x, **kw):
        assert all(not t.dead for t in x) and kw == {"dim": 2, "memory_config": self.DRAM_MEMORY_CONFIG}
        if self.fail == "concat":
            raise RuntimeError("concat")
        t = Tensor((1, 1, len(x) * 128, 2944), sum([t.rows for t in x], []))
        self.created.append(t)
        return t


def function(native):
    from models.demos.gpt_oss_120b_qb2.tt import attention_prefill

    # Keep the actual module binding restored even on operation failure.
    def run(*args):
        old = attention_prefill.ttnn
        attention_prefill.ttnn = native
        try:
            return attention_prefill._project_output(*args)
        finally:
            attention_prefill.ttnn = old

    return run


@pytest.mark.parametrize("batch", [2, 4, 8, 16, 32])
def test_exact_rows_kwargs_owned_lifetime(batch):
    n = Native()
    x = Tensor((1, 1, batch * 128, 1024))
    w = S(o_proj=object())
    cfg = object()
    dtype = object()
    y = function(n)(x, w, batch, dtype, cfg)
    assert y.rows == list(range(batch * 128)) and not x.dead and not y.dead
    assert len(n.calls) == batch and all(
        wt is w.o_proj and kw == {"dtype": dtype, "compute_kernel_config": cfg} for _, wt, kw in n.calls
    )
    assert all(t.dead for t in n.created if t is not y)


@pytest.mark.parametrize(
    "batch,seq,width", [(1, 128, 1024), (4, 256, 1024), (4, 64, 1024), (4, 128, 2048), (1, 1, 1024)]
)
def test_original_fallback(batch, seq, width):
    n = Native()
    x = Tensor((1, 1, batch * seq, width))
    w = S(o_proj=object())
    c = object()
    d = object()
    y = function(n)(x, w, batch, d, c)
    assert n.calls == [(x, w.o_proj, {"dtype": d, "compute_kernel_config": c})] and not x.dead and not y.dead


@pytest.mark.parametrize("fail", [1, 2, 3, 4, "concat"])
def test_failure_closes_only_owned(fail):
    n = Native(fail)
    x = Tensor((1, 1, 512, 1024))
    with pytest.raises(RuntimeError):  # allow-pytest.raises: CPU operation double, no device error.
        function(n)(x, S(o_proj=object()), 4, object(), object())
    assert not x.dead and all(t.dead for t in n.created)


def test_actual_bias_reduction_unchanged():
    tree = ast.parse((Path(__file__).parents[1] / "tt/attention_prefill.py").read_text())
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_finish_prefill")
    text = ast.unparse(f)
    assert "ttnn.add(tt_out, weights.o_proj_bias, output_tensor=tt_out)" in text
    assert "apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)" in text
    assert text.count("_project_output(") == 1
