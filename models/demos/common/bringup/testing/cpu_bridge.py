# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The CPU bridge (F46): how a device model runs a step deferred to op-gen (plan/op_request.py) on the host.

    bridge = CpuBridge(mesh, spec, "indexer_pool", ref.component(i, "indexer_pool"),
                       inputs=["replicate"], output="shard:1")
    y = bridge(ctx, x_device)      # a step fn like any other in the block graph (run_block overrides)

It brings the step's device inputs to the host (gathered per their mesh placement), runs the CPU reference step in
fp32, and puts the output back with the placement the next step expects. Placements: ``replicate`` (chip 0's copy),
``shard:<dim>`` (split over every chip in row-major mesh order), ``shard2d:<dim on mesh rows>,<dim on mesh cols>``
(either may be ``none``). ``to_ref`` / ``from_ref`` reshape between the device layout ([1, 1, S, H]) and the
reference's ([S, H]); the default drops leading size-1 dims on the way in and restores the first input's rank on the
way out.

Its transfers run inside ``host_transfers.bridged()``, so they never count in ``host_transfers_per_layer``, and its host
time is kept in STATS: ladder, profile and positions record ``deferred_cpu_steps`` and ``deferred_cpu_ms``. A bridge
is marked ``cpu_bridge = True``; the component test refuses one (a deferred step never passes its device gate), and the
swap harness runs a DEFERRED step on the reference itself.
"""

from __future__ import annotations

import time

from models.demos.common.bringup.testing.host_transfers import bridged


class BridgeStats:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.ms, self.calls, self.steps = 0.0, 0, set()

    def add(self, layer, step: str, ms: float) -> None:
        self.ms += ms
        self.calls += 1
        self.steps.add((layer, step))

    @property
    def step_names(self) -> list[str]:
        return sorted({s for _, s in self.steps})


STATS = BridgeStats()


def record(metrics, stats: BridgeStats = STATS) -> None:
    """deferred_cpu_steps: distinct (layer, step) the bridge ran; deferred_cpu_ms: its host time, both since reset."""
    metrics.record("deferred_cpu_steps", len(stats.steps))
    metrics.record("deferred_cpu_ms", round(stats.ms, 1))


def _drop_leading_ones(t):
    while t.dim() > 2 and t.shape[0] == 1:
        t = t.reshape(t.shape[1:])
    return t


class CpuBridge:
    cpu_bridge = True

    def __init__(
        self,
        mesh,
        spec,
        step: str,
        fn,
        inputs=("replicate",),
        output: str = "replicate",
        dtype=None,
        layout=None,
        memory_config=None,
        to_ref=None,
        from_ref=None,
        ctx_of=None,
        ttnn_module=None,
    ):
        if ttnn_module is None:
            import ttnn as ttnn_module
        self.ttnn, self.mesh, self.mesh_shape = ttnn_module, mesh, list(spec.mesh)
        self.step, self.fn, self.inputs, self.output = step, fn, list(inputs), output
        self.dtype, self.layout, self.memory_config = dtype, layout, memory_config
        self.to_ref = to_ref or _drop_leading_ones
        self.from_ref = from_ref
        self.ctx_of = ctx_of or (lambda ctx: ctx)

    # ---- placement
    def _axes(self, placement: str) -> tuple:
        """(tensor dim across mesh rows, across mesh cols); None = replicated along that mesh axis."""
        if placement == "replicate":
            return None, None
        if placement.startswith("shard:"):
            return "flat", int(placement.split(":")[1])
        if placement.startswith("shard2d:"):
            a, b = placement.split(":")[1].split(",")
            return tuple(None if x.strip() == "none" else int(x) for x in (a, b))
        raise ValueError(f"placement {placement!r}: replicate, shard:<dim> or shard2d:<dim>,<dim>")

    def to_host(self, x, placement: str):
        import torch

        ttnn = self.ttnn
        parts = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(x)]
        rows, cols = self.mesh_shape
        a, b = self._axes(placement)
        if a is None and b is None:
            return parts[0]
        if a == "flat":
            return torch.cat(parts, dim=b)
        grid = [parts[r * cols : (r + 1) * cols] for r in range(rows)]
        row = [torch.cat(g, dim=b) if b is not None else g[0] for g in grid]
        return torch.cat(row, dim=a) if a is not None else row[0]

    def to_device(self, y, placement: str, like):
        ttnn = self.ttnn
        a, b = self._axes(placement)
        if a is None and b is None:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        elif a == "flat":
            mapper = ttnn.ShardTensorToMesh(self.mesh, dim=b)
        else:
            mapper = ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh_shape), dims=(a, b))
        return ttnn.from_torch(
            y,
            dtype=self.dtype or like.dtype,
            layout=self.layout or like.layout,
            device=self.mesh,
            memory_config=self.memory_config or ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    # ---- the step
    def __call__(self, ctx, *xs):
        places = self.inputs if len(self.inputs) == len(xs) else self.inputs[:1] * len(xs)
        with bridged():
            t0 = time.perf_counter()
            host = [self.to_ref(self.to_host(x, p)) for x, p in zip(xs, places)]
            host = [h.float() if h.is_floating_point() else h for h in host]
            y = self.fn(self.ctx_of(ctx), *host)
            if self.from_ref is not None:
                y = self.from_ref(y)
            elif xs and len(xs[0].shape) > y.dim():
                y = y.reshape([1] * (len(xs[0].shape) - y.dim()) + list(y.shape))
            out = self.to_device(y, self.output, xs[0])
            STATS.add(getattr(ctx, "layer", None), self.step, (time.perf_counter() - t0) * 1e3)
        return out
