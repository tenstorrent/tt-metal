# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ttnn op programs that kernels/step_accept.cpp and kernels/chain_inputs.cpp replaced, kept as the reference of
the device tests (test_spec_step.py, test_spec_chain.py), which run each kernel beside its op program on the same
inputs and compare every output word. Not used by the model.

  * OpGraphStepAccept.body: spec_accept.graph over ttnn ops (about 70 ops at B = 8, K = 3), the step trace of a
    one-enqueue step before the kernel;
  * OpGraphChain.body: spec_chain.chain_graph over ttnn ops (about 160 ops), the chain program before the kernel.

Both take the same input and output tensors as spec_accept.StepAccept and spec_chain.ChainInputs, plus their own
float32 TILE constants and candidates, and compute in float32 tiles.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_accept import graph
from models.demos.blackhole.qwen36.tt.spec_chain import chain_graph
from models.demos.blackhole.qwen36.tt.tp_common import aliases


def _replicated(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class _DeviceOps:
    """``graph``'s op set in ttnn: float32 TILE tensors; every intermediate is collected in ``owned``, except a tensor
    that aliases an input: a slice over a whole tensor (the ids or draws at B * T = 32) returns its input, which
    the step trace must not free."""

    def __init__(self, accept):
        self.__dict__.update(
            {n: getattr(accept, n) for n in ("ids", "draws", "step", "pt_rows", "iota_t", "iota_k", "user_rows")}
        )
        self.top, self.scratch = accept.c_top, accept.c_scratch
        self.inputs = [t for t in (self.ids, self.draws, self.step, self.pt_rows) if t is not None]
        self.inputs += [self.iota_t, self.iota_k, self.user_rows, self.top, self.scratch]
        self.owned = []

    def _keep(self, t):
        if not any(t is s or aliases(t, s) for s in self.inputs):
            self.owned.append(t)
        return t

    def _float(self, t):
        return self._keep(ttnn.typecast(self._keep(ttnn.to_layout(t, ttnn.TILE_LAYOUT)), ttnn.float32))

    def lanes(self, t, B, T):
        R = B * T
        return self._float(self._keep(ttnn.reshape(self._keep(ttnn.slice(t, (0, 0, 0, 0), (1, 1, 1, R))), (B, T))))

    def row_block(self, t, row, B, T):
        R = B * T
        return self._float(self._keep(ttnn.reshape(self._keep(ttnn.slice(t, (row, 0), (row + 1, R))), (B, T))))

    def rows(self, t, start, stop, n):
        return self._float(self._keep(ttnn.slice(t, (start, 0), (stop, n))))

    def table(self, t):
        return self._float(t)

    def transpose(self, x):
        return self._keep(ttnn.transpose(x, 0, 1))

    def slice(self, x, s, e):
        return self._keep(ttnn.slice(x, s, e))

    def reshape(self, x, shape):
        return self._keep(ttnn.reshape(x, shape))

    def add(self, a, b):
        return self._keep(ttnn.add(a, b))

    def sub(self, a, b):
        return self._keep(ttnn.subtract(a, b))

    def mul(self, a, b):
        return self._keep(ttnn.multiply(a, b))

    def minimum(self, a, b):
        return self._keep(ttnn.minimum(a, b))

    def rsub1(self, x):
        return self._keep(ttnn.rsub(x, 1.0))

    def eq(self, a, b):
        return self._keep(ttnn.eq(a, b, dtype=ttnn.float32))

    def lt(self, a, b):
        return self._keep(ttnn.lt(a, b, dtype=ttnn.float32))

    def cumprod(self, x):
        return self._keep(ttnn.cumprod(x, dim=1))

    def sum(self, x):
        return self._keep(ttnn.sum(x, dim=1, keepdim=True))

    def concat(self, xs, dim):
        return self._keep(ttnn.concat(xs, dim=dim))

    def pad_w(self, x, n, W):
        return x if n == W else self._keep(ttnn.pad(x, [(0, 0), (0, W - n)], value=0.0))


class OpGraphStepAccept:
    """The step trace as ttnn ops over the tensors of a spec_accept.StepAccept ``acc`` (its ids, draws, step buffer,
    table rows, pack and reseed table), with its own float32 constants. Writes ``pack`` and ``rs_pt``, by default
    those of ``acc``."""

    def __init__(self, acc, table_rows, scratch_block, pack=None, rs_pt=None):
        T = acc.K + 1
        self.acc, self.B, self.K, self.W = acc, acc.B, acc.K, acc.W
        self.pack = acc.pack if pack is None else pack
        self.rs_pt = acc.rs_pt if rs_pt is None else rs_pt
        f32 = lambda t: _replicated(acc.mesh, torch.as_tensor(t, dtype=torch.float32), ttnn.float32, ttnn.TILE_LAYOUT)
        self.iota_t = f32(torch.arange(T).reshape(1, T))
        self.iota_k = f32(torch.arange(acc.K).reshape(1, acc.K))
        self.user_rows = f32((torch.arange(acc.B) * T).reshape(acc.B, 1))
        self.c_top = f32([[int(table_rows) - 1]])
        self.c_scratch = f32([[int(scratch_block)]])
        self.buffers = [self.iota_t, self.iota_k, self.user_rows, self.c_top, self.c_scratch]

    def __getattr__(self, name):
        if name in ("ids", "draws", "step", "pt_rows"):
            return getattr(self.acc, name)
        raise AttributeError(name)

    def body(self):
        """Trace-capturable: writes ``pack`` and ``rs_pt`` in place."""
        o = _DeviceOps(self)
        pack, rs = graph(o, self.B, self.K, self.W)
        for out, dtype, dst in ((pack, ttnn.uint32, self.pack), (rs, ttnn.int32, self.rs_pt)):
            rm = ttnn.to_layout(o._keep(ttnn.typecast(out, dtype)), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy(rm, dst)
            o._keep(rm)
        for t in o.owned:
            ttnn.deallocate(t)

    def release(self):
        for t in self.buffers:
            ttnn.deallocate(t)
        self.buffers = []


class _ChainDeviceOps(_DeviceOps):
    """chain_graph's op set in ttnn (_DeviceOps plus the chain's own inputs)."""

    def __init__(self, chain):
        names = (
            "ids", "pack", "overlay", "cand", "draws", "seed_cand", "iota_t", "iota_k", "iota_b", "user_rows", "one",
            "head", "tail",
        )  # fmt: skip
        self.__dict__.update({n: getattr(chain, n) for n in names})
        self.top = chain.c_top
        self._drafts = chain.drafts
        self.inputs = [t for t in (*(getattr(chain, n) for n in names), self.top, *self._drafts) if t is not None]
        self.owned = []

    def draft(self, k, B):
        return self._float(self._keep(ttnn.reshape(self._drafts[k], (B, 1))))

    def relu(self, x):
        return self._keep(ttnn.relu(x))


class OpGraphChain:
    """The chain program as ttnn ops over the inputs of a spec_chain.ChainInputs ``chain`` (its ids, draws, pack,
    overlay and drafts), with its own float32 TILE constants and candidates (``stage``). Writes ``step``, ``ctrls``
    and ``seeds``, by default those of ``chain``."""

    def __init__(self, chain, table_rows, step=None, ctrls=None, seeds=None):
        B, K = chain.B, chain.K
        T = K + 1
        self.chain, self.B, self.K, self.nv_heads, self.groups = chain, B, K, chain.nv_heads, chain.groups
        self.step = chain.step if step is None else step
        self.ctrls = chain.ctrls if ctrls is None else list(ctrls)
        self.seeds = chain.seeds if seeds is None else seeds
        self.ctrl_words = int(self.ctrls[0].shape[-1])
        f32 = lambda t: _replicated(chain.mesh, torch.as_tensor(t, dtype=torch.float32), ttnn.float32, ttnn.TILE_LAYOUT)
        self.iota_t = f32(torch.arange(T).reshape(1, T))
        self.iota_k = f32(torch.arange(K).reshape(1, K))
        self.iota_b = f32(torch.arange(B).reshape(1, B))
        self.user_rows = f32((torch.arange(B) * T).reshape(B, 1))
        self.c_top = f32([[int(table_rows) - 1]])
        self.one = f32([[1.0]])
        self.head = f32(torch.zeros(1, 1 + B))
        rest = self.ctrl_words - 1 - B - 2 * B * self.nv_heads
        self.tail = f32(torch.zeros(1, rest)) if rest else None
        self.cand = f32(torch.zeros(T * B, 2 * self.nv_heads * self.groups))
        self.seed_cand = None if chain.seeds is None else f32(torch.zeros(T * B, T))
        consts = [self.iota_t, self.iota_k, self.iota_b, self.user_rows, self.c_top, self.one, self.head, self.tail]
        self.buffers = [t for t in (*consts, self.cand, self.seed_cand) if t is not None]

    def __getattr__(self, name):
        if name in ("ids", "draws", "pack", "overlay", "drafts"):
            return getattr(self.chain, name)
        raise AttributeError(name)

    def stage(self, cand, seed_cand=None):
        """Host -> device copies of the float32 candidates (the overlay is the chain's own)."""
        rep = ttnn.ReplicateTensorToMesh(self.chain.mesh)
        T = self.K + 1
        pairs = [(cand, self.cand)]
        if self.seed_cand is not None:
            pairs.append((torch.zeros(T * self.B, T) if seed_cand is None else seed_cand, self.seed_cand))
        self._refs = []
        for host, dev in pairs:
            h = torch.as_tensor(host).to(torch.float32)
            self._refs.append(ttnn.from_torch(h, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep))
            ttnn.copy_host_to_device_tensor(self._refs[-1], dev)

    def body(self):
        """Trace-capturable: writes ``step``, every ``ctrls`` page and, with a sampler, ``seeds`` in place."""
        o = _ChainDeviceOps(self)
        W = int(self.step.shape[-1])
        step, ctrls, seeds = chain_graph(o, self.B, self.K, W, self.groups, self.nv_heads, self.ctrl_words)
        for out, dtype, dst in (
            (step, ttnn.uint32, self.step),
            *((c, ttnn.int32, d) for c, d in zip(ctrls, self.ctrls)),
        ):
            rm = ttnn.to_layout(o._keep(ttnn.typecast(out, dtype)), ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy(rm, dst)
            o._keep(rm)
        if seeds is not None:
            # manual_seed takes the seeds as a 1-D uint32 row-major tensor.
            rm = o._keep(ttnn.to_layout(o._keep(ttnn.typecast(seeds, ttnn.uint32)), ttnn.ROW_MAJOR_LAYOUT))
            ttnn.copy(o._keep(ttnn.reshape(rm, (self.B * (self.K + 1),))), self.seeds)
        for t in o.owned:
            ttnn.deallocate(t)

    def release(self):
        for t in self.buffers:
            ttnn.deallocate(t)
        self.buffers = []
