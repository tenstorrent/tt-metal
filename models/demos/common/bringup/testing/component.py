# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Component and swap tests. Both read golden inputs and compare with golden outputs; neither runs the CPU model
end to end.

component: one step of one layer alone. Metric ``<metric_prefix>`` (default ``pcc_<step>_L<layer>``).
swap:      the whole block of one layer, with the listed steps on the device and every other step on the CPU
           reference. Metrics ``pcc_swap_<boundary>`` for every boundary and ``pcc_swap_out``. When the block output
           drifts, the step swapped in last is the cause, because every earlier swap already passed its gate.

A step deferred to op-gen (task DEFERRED, F46) runs on the CPU reference in the swap test, as if it were not swapped.
Its own component test fails in device mode when the model hands it a CPU bridge: a bridged step is not on the device.
"""

from __future__ import annotations

import torch

from models.demos.common.bringup.reference.interface import run_block
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    impl_mode,
    reference_ctx,
    threshold,
)


def _step(ref, layer: int, name: str):
    for st in ref.block_graph(layer):
        if st.name == name:
            return st
    raise KeyError(f"layer {layer} has no step {name!r}; steps: {[s.name for s in ref.block_graph(layer)]}")


def module_under_test(s, ref, mesh, layer: int, name: str, mode: str | None = None):
    """fn(ref_ctx, dev_ctx, *inputs) for the chosen implementation of one step."""
    from models.demos.common.bringup.plan.op_request import deferred_steps

    mode = mode or impl_mode()
    cpu = ref.component(layer, name)
    if mode == "reference" or (mode == "device" and (s.block_type_of(layer), name) in deferred_steps(s)):
        if mode == "device":
            print(f"{name}: deferred to op-gen, runs on the CPU reference")
        return lambda rctx, dctx, *x: cpu(rctx, *x)
    if mode == "stub":
        return lambda rctx, dctx, *x: torch.zeros_like(cpu(rctx, *x))
    dev = s.hooks().device_component(mesh, s, layer, name)
    fn = lambda rctx, dctx, *x: dev(dctx, *x)  # noqa: E731
    fn.cpu_bridge = getattr(dev, "cpu_bridge", False)
    return fn


def run_component_test(
    s,
    step: str,
    layer: int | None = None,
    mesh=None,
    compare_mode: str | None = None,
    thr: float | None = None,
    metric: str | None = None,
) -> bool:
    ref_hooks = s.hooks()
    g, c = component_golden(s)
    layer = s.representative_layer(s.block_type_of(layer)) if layer is None else layer
    ref = ref_hooks.reference(s, layers=[layer], dtype=torch.float32)
    st = _step(ref, layer, step)
    gl = g.layer(c, layer)
    inputs = [gl[i].float() if gl[i].is_floating_point() else gl[i] for i in st.inputs]
    want = gl[st.output]
    from models.demos.common.bringup.plan.op_request import deferred_steps

    if impl_mode() == "device" and (s.block_type_of(layer), step) in deferred_steps(s):
        print(f"FAIL {step}: deferred to op-gen (task DEFERRED); it runs on the CPU until the op is delivered")
        return False
    fn = module_under_test(s, ref, mesh, layer, step)
    if getattr(fn, "cpu_bridge", False):
        print(f"FAIL {step}: device_component returned a CPU bridge; a deferred step is not on the device")
        return False
    out = fn(reference_ctx(ref, layer, g, c), device_ctx(layer, g, c), *inputs)
    mode = compare_mode or default_mode(want)
    _, ok = compare(
        metric or f"pcc_{step}_L{layer:02d}", out, want, mode, threshold(s, "component") if thr is None else thr
    )
    return ok


def run_swap_test(s, block_type: str, swapped: list[str], mesh=None, thr: float | None = None) -> bool:
    """Block of the block type's representative layer; ``swapped`` steps run under the chosen implementation."""
    g, c = component_golden(s)
    layer = s.representative_layer(block_type)
    ref = s.hooks().reference(s, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    overrides = {}
    for name in swapped:
        _step(ref, layer, name)
        mut = module_under_test(s, ref, mesh, layer, name)
        overrides[name] = lambda ctx, *x, mut=mut: mut(ctx, dctx, *x)
    gl = g.layer(c, layer)
    seen = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        rctx,
        gl["in"].float(),
        rec=lambda n, t: seen.__setitem__(n, t),
        overrides=overrides,
    )
    for n, t in seen.items():
        if n not in ("in", "out") and n in gl:
            compare(f"pcc_swap_{n}", t, gl[n], default_mode(gl[n]), 0.0)  # the trail, for diagnosis; not gated
    _, ok = compare("pcc_swap_out", seen["out"], gl["out"], "pcc", threshold(s, "block") if thr is None else thr)
    return ok
