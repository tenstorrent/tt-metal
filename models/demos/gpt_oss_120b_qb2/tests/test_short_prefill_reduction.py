# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as S

import pytest


class Tensor:
    def __init__(self, shape, rows=None):
        self.shape = shape
        self.rows = rows
        self.allocated = True
        self.frees = 0

    def is_allocated(self):
        return self.allocated

    def deallocate(self, force):
        assert force is True and self.allocated
        self.allocated = False
        self.frees += 1


class Native:
    DRAM_MEMORY_CONFIG = object()

    def __init__(self, failure=None):
        self.failure = failure
        self.created = []
        self.calls = []

    def slice(self, x, start, end, **kwargs):
        assert x.is_allocated() and kwargs == {"memory_config": self.DRAM_MEMORY_CONFIG}
        assert start == [0, 0, len(self.calls) * 128, 0]
        assert end == [1, 1, (len(self.calls) + 1) * 128, 2944]
        if self.failure == ("slice", len(self.calls)):
            raise RuntimeError("slice")
        t = Tensor((1, 1, 128, 2944), list(range(start[2], end[2])))
        self.created.append(t)
        return t

    def reduce(self, x, mesh, ccl, hidden):
        self.calls.append((x, mesh, ccl, hidden))
        n = len(self.calls)
        if self.failure == ("before", n):
            raise RuntimeError("before consumption")
        x.deallocate(True)
        if self.failure == ("after", n):
            raise RuntimeError("after consumption")
        y = Tensor((1, 1, x.shape[2], hidden), x.rows)
        self.created.append(y)
        return y

    def concat(self, parts, **kwargs):
        assert kwargs == {"dim": 2, "memory_config": self.DRAM_MEMORY_CONFIG}
        assert all(t.is_allocated() for t in parts)
        if self.failure == "concat":
            raise RuntimeError("concat")
        y = Tensor((1, 1, len(parts) * 128, 2880), sum((p.rows for p in parts), []))
        self.created.append(y)
        return y


def run(native, *args):
    from models.demos.gpt_oss_120b_qb2.tt import attention_prefill as module

    old_native, old_reduce = module.ttnn, module.apply_allreduce
    module.ttnn, module.apply_allreduce = native, native.reduce
    try:
        return module._reduce_output(*args)
    finally:
        module.ttnn, module.apply_allreduce = old_native, old_reduce


@pytest.mark.parametrize("batch", [2, 4, 8, 16, 32])
def test_dense_users_original_collective_and_consumption(batch):
    n = Native()
    x = Tensor((1, 1, batch * 128, 2944))
    mesh, ccl = S(tp=4, tp_axis=1), object()
    y = run(n, x, mesh, ccl, 2880, batch)
    assert y.rows == list(range(batch * 128)) and y.is_allocated()
    assert len(n.calls) == batch
    assert all(m is mesh and c is ccl and h == 2880 for _, m, c, h in n.calls)
    assert x.frees == 1 and all(t.frees == 1 for t in n.created if t is not y)


@pytest.mark.parametrize(
    "failure",
    [(phase, n) for phase in ("before", "after") for n in range(1, 5)] + [("slice", 0), ("slice", 2), "concat"],
)
def test_failure_frees_owned_inputs_and_outputs_once(failure):
    n = Native(failure)
    x = Tensor((1, 1, 512, 2944))
    with pytest.raises(RuntimeError):  # allow-pytest.raises: host operation doubles, no device failure.
        run(n, x, S(tp=4, tp_axis=1), object(), 2880, 4)
    assert x.frees == 1 and all(t.frees == 1 for t in n.created)


@pytest.mark.parametrize(
    "batch,rows,width,tp,axis,hidden",
    [
        (1, 128, 2944, 4, 1, 2880),
        (4, 1024, 2944, 4, 1, 2880),
        (4, 256, 2944, 4, 1, 2880),
        (4, 512, 2880, 4, 1, 2880),
        (4, 512, 2944, 1, 1, 2880),
        (4, 512, 2944, 4, 0, 2880),
        (4, 512, 2944, 4, 1, 4096),
    ],
)
def test_fallback_original_call_and_exact_return(batch, rows, width, tp, axis, hidden):
    n = Native()
    x = Tensor((1, 1, rows, width))
    mesh, ccl = S(tp=tp, tp_axis=axis), object()
    y = run(n, x, mesh, ccl, hidden, batch)
    assert n.calls == [(x, mesh, ccl, hidden)] and y is n.created[0] and y.is_allocated()
