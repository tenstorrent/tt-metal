# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Command 3 — the trace contract for the FLUX.2 Klein 9B VAE pipeline.

    ./python_env/bin/python -m pytest models/demos/flux_2_klein_9b_vae/tests/e2e/test_trace_contract.py -s

``PIPELINE_STAGES`` is ``["encode", "decode"]``: the diffusers config states the two phases
directly (``down_block_types`` = the compression stack, ``up_block_types`` = the expansion
stack). There is no autoregressive phase — ``AutoencoderKLFlux2`` has no ``generate()`` and no
KV cache — so no ``[prefill, decode]`` and no ``decode_prefill``/``decode_step``.

For each stage the perf engine binds four hooks on the pipeline OBJECT:
``<stage>_trace_setup(inputs)``, ``<stage>_trace_step()``, ``<stage>_trace_inputs()`` and
``<stage>_trace_items()``. This file checks the contract's shape (no device) and then its
behaviour (on device): a real capture/execute/release, a host-op-free forward, and the
``layers`` knob actually reducing what gets built.
"""
from __future__ import annotations

import inspect
import os

import pytest
import torch

import ttnn
from models.demos.flux_2_klein_9b.vae.tt import pipeline as pipeline_module
from models.demos.flux_2_klein_9b.vae.tt.pipeline import PIPELINE_STAGES, build_pipeline

_TP = int(os.environ.get("TT_HW_PLANNER_SHARD_TP", "8"))
_DP = int(os.environ.get("TT_HW_PLANNER_SHARD_DP", "1"))
_MESH = (_DP, _TP) if _DP > 1 else _TP

_TRACE_HOOK_SUFFIXES = ("_trace_setup", "_trace_step", "_trace_inputs", "_trace_items")

# recommended_trace_region_size() is a pure function of the pinned config, so it is safe at
# module scope — but device_params are built at COLLECTION time, before any device exists, so
# guard it and fall back to the measured literal rather than failing collection.
_DEFAULT_TRACE_REGION_SIZE = 23887872
try:
    _TRACE_REGION_SIZE = int(pipeline_module.recommended_trace_region_size())
except Exception:  # pragma: no cover - only when the helper needs a live device
    _TRACE_REGION_SIZE = _DEFAULT_TRACE_REGION_SIZE

_PLAIN_DEVICE_PARAMS = {"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}
_TRACE_DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "trace_region_size": _TRACE_REGION_SIZE,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
}


def _pipeline_class() -> type:
    """The resident pipeline class, resolved without instantiating it (no device needed)."""
    cls = getattr(pipeline_module, "TtFluxVaePipeline", None)
    if isinstance(cls, type):
        return cls
    candidates = [
        value
        for value in vars(pipeline_module).values()
        if isinstance(value, type)
        and getattr(value, "__module__", None) == pipeline_module.__name__
        and all(hasattr(value, name) for name in ("run_encode", "run_decode", "run_reconstruct"))
    ]
    assert candidates, (
        "tt/pipeline.py exposes no pipeline class (expected TtFluxVaePipeline, or any class defined "
        "there carrying run_encode/run_decode/run_reconstruct)."
    )
    return candidates[0]


def _required_parameters(func) -> list:
    signature = inspect.signature(func)
    return [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


# --------------------------------------------------------------------------------------
# Shape of the contract — no device
# --------------------------------------------------------------------------------------
def test_pipeline_stages_declared():
    """The stages are declared, and every stage carries all four hooks on the CLASS."""
    stages = list(PIPELINE_STAGES)
    print(f"[trace] PIPELINE_STAGES = {stages}", flush=True)
    assert stages == [
        "encode",
        "decode",
    ], f"PIPELINE_STAGES must be ['encode', 'decode'] for this VAE (no autoregressive phase), got {stages}"

    cls = _pipeline_class()
    print(f"[trace] pipeline class = {cls.__module__}.{cls.__name__}", flush=True)
    missing = []
    for stage in stages:
        for suffix in _TRACE_HOOK_SUFFIXES:
            hook = f"{stage}{suffix}"
            attr = getattr(cls, hook, None)
            if attr is None or not callable(attr):
                missing.append(hook)
    assert not missing, f"pipeline class {cls.__name__} is missing trace hooks: {missing}"

    assert callable(
        getattr(pipeline_module, "recommended_trace_region_size", None)
    ), "tt/pipeline.py must expose recommended_trace_region_size() so the harness can size the trace region"
    assert callable(getattr(pipeline_module, "build_pipeline", None)), "tt/pipeline.py must expose build_pipeline()"

    # build_pipeline must accept the knobs the perf harness passes.
    build_params = inspect.signature(build_pipeline).parameters
    for knob in ("model", "layers", "encode_layers", "decode_layers"):
        assert knob in build_params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in build_params.values()
        ), f"build_pipeline() accepts neither `{knob}` nor **kwargs"

    # AR contract is deliberately absent for this model.
    for absent in ("decode_prefill", "decode_step"):
        assert not hasattr(cls, absent), f"{absent} must not exist — AutoencoderKLFlux2 has no autoregressive stage"


def test_trace_inputs_are_zero_arg():
    """``<stage>_trace_inputs`` is the model-agnostic seam: it takes ONLY ``self``."""
    cls = _pipeline_class()
    for stage in PIPELINE_STAGES:
        hook = getattr(cls, f"{stage}_trace_inputs")
        params = list(inspect.signature(hook).parameters)
        print(f"[trace] {stage}_trace_inputs{tuple(params)}", flush=True)
        assert params == ["self"], (
            f"{stage}_trace_inputs must be zero-arg (only `self`) so the perf engine can call it with no "
            f"per-model knowledge; got {params}"
        )
        # _trace_step / _trace_items are called the same way: no required argument beyond self.
        for suffix in ("_trace_step", "_trace_items"):
            other = getattr(cls, f"{stage}{suffix}")
            required = _required_parameters(other)
            assert required == ["self"], f"{stage}{suffix} must be callable with no arguments, requires {required}"
        # _trace_setup takes exactly the value _trace_inputs returns: self + one argument.
        setup_required = _required_parameters(getattr(cls, f"{stage}_trace_setup"))
        assert (
            len(setup_required) == 2 and setup_required[0] == "self"
        ), f"{stage}_trace_setup must take exactly (self, inputs); requires {setup_required}"


# --------------------------------------------------------------------------------------
# Behaviour of the contract — on device
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [_TRACE_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_trace_capture_selftest(mesh_device):
    """Every stage really captures, executes and releases a trace, and the traced output PCCs."""
    print(f"[trace] trace_region_size={_TRACE_REGION_SIZE}", flush=True)
    pipeline = build_pipeline(mesh_device)
    result = pipeline.trace_capture_selftest(mesh_device)
    print(f"[trace] trace_capture_selftest -> {result!r}", flush=True)
    assert result is True, (
        "trace_capture_selftest did not return True — at least one of PIPELINE_STAGES could not be "
        "captured/executed/released, or its traced output failed the PCC check."
    )


@pytest.mark.parametrize("device_params", [_PLAIN_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_host_op_selftest(mesh_device):
    """The authoritative fully-on-device check: no host aten op may fire in any head."""
    pipeline = build_pipeline(mesh_device)
    verdict = pipeline.host_op_selftest()
    print(f"[host-op] reason: {verdict['reason']}", flush=True)
    print(f"[host-op] host_ops[:12] = {list(verdict['host_ops'])[:12]}", flush=True)
    print(f"[host-op] n_host_ops = {verdict.get('n_host_ops')}", flush=True)
    assert verdict["on_device"], (
        f"host compute in the forward — {verdict.get('n_host_ops')} host aten op(s): "
        f"{list(verdict['host_ops'])[:12]}"
    )


_SCALAR_TYPES = (str, bytes, int, float, bool, complex, type(None), torch.Tensor)


def _is_repeated_block_list(value) -> bool:
    """A plain list/tuple of >=1 same-typed non-scalar objects: the plan's repeated-block shape."""
    if not isinstance(value, (list, tuple)) or len(value) < 1:
        return False
    first = type(value[0])
    if issubclass(first, _SCALAR_TYPES):
        return False
    if isinstance(getattr(ttnn, "Tensor", None), type) and issubclass(first, ttnn.Tensor):
        return False
    return all(type(item) is first for item in value)


def _walk_repeated_lists(root, max_depth: int = 6) -> dict:
    """Fallback discovery: count every homogeneous list of built blocks hanging off the pipeline.

    ``hf`` is skipped — it is the untouched HF reference, not what the pipeline BUILT.
    """
    counts: dict = {}
    seen: set = set()
    stack = [(root, "", 0)]
    while stack:
        obj, path, depth = stack.pop()
        if depth > max_depth or id(obj) in seen:
            continue
        seen.add(id(obj))
        namespace = getattr(obj, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for key, value in list(namespace.items()):
            if key.startswith("__") or key == "hf":
                continue
            child = f"{path}.{key}" if path else key
            if _is_repeated_block_list(value):
                counts[child] = len(value)
                for index, item in enumerate(value):
                    stack.append((item, f"{child}[{index}]", depth + 1))
            elif hasattr(value, "__dict__") and not isinstance(value, _SCALAR_TYPES):
                stack.append((value, child, depth + 1))
    return counts


def _repeated_block_counts(pipeline):
    hook = getattr(pipeline, "repeated_block_counts", None)
    if callable(hook):
        counts = dict(hook())
        return counts, sum(int(v) for v in counts.values())
    counts = _walk_repeated_lists(pipeline)
    return counts, sum(int(v) for v in counts.values())


def _hf_structural_resnet_count(hf) -> int:
    """How many repeated resnets the FULL model has, straight off the HF reference."""
    total = 0
    for stack_name, blocks_name in (("encoder", "down_blocks"), ("decoder", "up_blocks")):
        stack = getattr(hf, stack_name, None)
        if stack is None:
            continue
        for block in list(getattr(stack, blocks_name, []) or []):
            total += len(list(getattr(block, "resnets", []) or []))
        mid = getattr(stack, "mid_block", None)
        if mid is not None:
            total += len(list(getattr(mid, "resnets", []) or []))
    return total


@pytest.mark.parametrize("device_params", [_PLAIN_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_layers_knob(mesh_device):
    """``layers=1`` really builds fewer repeated blocks; ``layers=None`` builds every layer."""
    full = build_pipeline(mesh_device, layers=None)
    full_counts, full_total = _repeated_block_counts(full)
    print(f"[layers] layers=None -> total={full_total} {full_counts}", flush=True)

    capped = build_pipeline(mesh_device, layers=1)
    capped_counts, capped_total = _repeated_block_counts(capped)
    print(f"[layers] layers=1    -> total={capped_total} {capped_counts}", flush=True)

    assert full_total > 0, (
        "no repeated blocks were discovered on the pipeline — expose repeated_block_counts() or hold "
        "each repeated stack as a plain list of same-typed elements"
    )
    assert capped_total < full_total, (
        f"layers=1 built {capped_total} repeated blocks, layers=None built {full_total} — the `layers` "
        "knob does not actually cap the built depth"
    )

    structural = _hf_structural_resnet_count(full.hf)
    print(f"[layers] HF reference repeated-resnet count = {structural}", flush=True)
    assert structural > 0, "could not read the repeated-resnet structure off pipeline.hf"
    assert full_total >= structural, (
        f"layers=None built only {full_total} repeated blocks but the HF reference has {structural} "
        "repeated resnets — layers=None must build EVERY layer"
    )
