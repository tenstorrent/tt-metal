# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the generic tests: spec, golden choice, the module under test, comparisons, mesh parameters.

The module under test is chosen by BRINGUP_IMPL (set by freeze):
    device     (default) the model's device implementation, from hooks.device_component / hooks.device_model
    reference  the CPU reference's own component: the test must PASS with it
    stub       zeros shaped like the reference output: the test must FAIL with it
    mutate:<kind>  the CPU reference with its output altered (testing/mutate.py, F49): proves a test catches a
               wrong module, on the CPU; BRINGUP_MUTATE_STEP names the one step to alter
    mutations  the freeze sweep of a component test with checks="auto" (F56, testing/component_checks.py): every
               standard mistake on the CPU in one process; the test must PASS (every mistake caught)

Device hooks a model provides (the implement role writes them):
    device_params(spec) -> dict                     mesh fixture params (fabric config, l1_small_size, ...)
    device_component(mesh, spec, layer, step) -> fn(ctx, *host inputs) -> host tensor
        ctx.extra: state_prefix ({name: tensor} from the golden, positions [0, prefix_len) valid; a fixed-size tensor
        (spec state.fixed) is the state as of prefix_len), prefix_len, max_seq
    device_model(mesh, spec, layers, lm_head=True) -> object with
        load_seconds, new_state(max_seq), embed(tokens), from_host(h [S, H]), layer(i, h, start, state),
        final_norm(h), to_host(h) -> [S, H], logits(hidden, rows) -> [len(rows), V], free(h), sync()
        state.load_prefix(layer, tensors, length), state.to_torch(layer, length) -> {name: tensor}
        (layer(i, h, 0, state) starts a new sequence: a fixed-size state resets there)
"""

from __future__ import annotations

import os

import torch

from models.demos.common.bringup.core import defaults, metrics
from models.demos.common.bringup.core.runs import IMPL_ENV
from models.demos.common.bringup.reference.golden import Golden, load_spec
from models.demos.common.bringup.reference.interface import Ctx

DEFAULT_THRESHOLDS = {
    k: defaults.get(f"thresholds.{k}") for k in ("component", "block", "layer", "state", "final_hidden", "top5")
}


def impl_mode() -> str:
    from models.demos.common.bringup.testing.mutate import kind_of

    mode = os.environ.get(IMPL_ENV, "device")
    if mode not in ("device", "reference", "stub", "mutations") and kind_of(mode) is None:
        raise ValueError(f"{IMPL_ENV}={mode!r}")
    return mode


def spec():
    return load_spec()


DEVICE_TEST_TIMEOUT_S = defaults.get("box.test_timeout_s")


def device_timeout(s):
    """pytest-timeout mark for the framework's device tests (ladder, contract, profile): the repo's pytest.ini default
    (300 s) is shorter than a full-target rung. Spec ``box.test_timeout_s``; hangs are still caught by run_safe_pytest.
    """
    import pytest

    return pytest.mark.timeout(int(s.get("box.test_timeout_s")))


def threshold(s, key: str) -> float:
    return s.threshold(key, DEFAULT_THRESHOLDS[key])


def component_golden(s) -> tuple[Golden, int]:
    """The rung and chunk component and swap tests read: the spec's ``tests.component_rung`` or the first rung with
    full dumps, and its last dumped chunk (start > 0, so stateful steps see a real prefix)."""
    name = s.get("tests.component_rung") or next((r["name"] for r in s.data["ladder"] if r.get("full_dumps")), None)
    if name is None:
        raise ValueError("no ladder rung with full_dumps; set tests.component_rung")
    g = Golden.for_rung(s, name)
    return g, int(s.get("tests.component_chunk", g.dumped_chunks[-1]))


def reference_ctx(ref, layer: int, g: Golden, c: int):
    start = c * g.chunk
    state = ref.new_state(g.seq)
    if start:
        ref.load_state(state, layer, g.state(layer, at=start), start)
    return ref.chunk_context(layer, start, g.chunk, state)


def device_ctx(layer: int, g: Golden, c: int) -> Ctx:
    start = c * g.chunk
    return Ctx(
        layer, start, g.chunk, None, {"state_prefix": g.state(layer, at=start), "prefix_len": start, "max_seq": g.seq}
    )


def as_float(t: torch.Tensor) -> torch.Tensor:
    return t.float() if t.is_floating_point() else t


def compare(name: str, got: torch.Tensor, want: torch.Tensor, mode: str, thr: float) -> tuple[float, bool]:
    """pcc: Pearson correlation. match: fraction of equal elements (integer outputs such as expert ids).
    topk_overlap: mean per-row fraction of the wanted set that the output contains (order-free)."""
    if got.numel() == want.numel():
        got = got.reshape(want.shape)  # device outputs may carry [1, 1, S, H] padding dims
    if got.shape != want.shape:
        v = 0.0
    elif mode == "pcc":
        v = metrics.pcc(as_float(got).float(), as_float(want).float())
    elif mode == "match":
        v = (got == want).float().mean().item()
    elif mode == "topk_overlap":
        w, gt = want.reshape(-1, want.shape[-1]), got.reshape(-1, got.shape[-1])
        v = (
            torch.tensor([len(set(a.tolist()) & set(b.tolist())) / len(set(b.tolist())) for a, b in zip(gt, w)])
            .mean()
            .item()
        )
    else:
        raise ValueError(f"compare mode {mode!r}")
    metrics.record(name, v)
    ok = v >= thr
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {mode}={v:.6f} (>= {thr}) shape {tuple(got.shape)}")
    return v, ok


def default_mode(t: torch.Tensor) -> str:
    return "pcc" if t.is_floating_point() else "match"


def device_params(s) -> dict:
    hooks = s.hooks()
    if hasattr(hooks, "device_params"):
        return hooks.device_params(s)
    import ttnn

    p = dict(s.get("box.device_params", {}))
    if "fabric_config" in p:
        p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    return p


def mesh_parametrize(fn):
    """Parametrize a pytest test with the spec's mesh shape and device params (the repo's mesh_device fixture).
    Under BRINGUP_IMPL=reference, stub or mutate:<kind> no device is opened: mesh_device is None."""
    import pytest

    if impl_mode() != "device":
        return pytest.mark.parametrize("mesh_device", [None], ids=["nodevice"])(fn)
    s = spec()
    fn = pytest.mark.parametrize("device_params", [device_params(s)], indirect=True, ids=["box"])(fn)
    return pytest.mark.parametrize("mesh_device", [tuple(s.mesh)], indirect=True, ids=["x".join(map(str, s.mesh))])(fn)
