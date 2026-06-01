import os
from contextlib import contextmanager
from typing import Any

import ttnn

# ``ttnn.ReadDeviceProfiler`` is a host call that syncs the device; it must never
# run inside a ``ttnn`` trace capture (which records device ops only and forbids
# host round-trips / syncs mid-capture). The traced decode path reuses several of
# the eager ``forward`` helpers below, so route every profiler read through this
# guard and silence it while a trace is being captured.
_IN_TRACE_CAPTURE = False


def _profile(device) -> None:
    if not _IN_TRACE_CAPTURE:
        ttnn.ReadDeviceProfiler(device)


# Tracy signposts let the (flat) device-op profile be sliced per decoder-layer
# sub-module (attention, MoE router/experts/shared, hyper-connection, norms): each
# ``_region`` emits a ``<NAME>_START`` / ``<NAME>_END`` host marker around the ops
# it issues. ``tracy`` only imports on a profiler-enabled build, so degrade to a
# no-op otherwise, and stay silent during a ttnn trace capture (no host calls).
try:
    from tracy import signpost as _tracy_signpost
except Exception:  # pragma: no cover - tracy missing on non-profiling builds
    _tracy_signpost = None

# Master switch for the per-module signposts. Defaults on (they are a no-op unless
# the run is captured under the Tracy profiler), but can be disabled to drop even
# the host-side call overhead: set ``DEEPSEEK_V4_SIGNPOSTS=0`` or call
# :func:`set_signposts_enabled(False)` at runtime.
_SIGNPOSTS_ENABLED = os.environ.get("DEEPSEEK_V4_SIGNPOSTS", "1") not in ("0", "", "false", "False")


def set_signposts_enabled(enabled: bool) -> None:
    """Enable/disable the per-module Tracy signposts at runtime."""
    global _SIGNPOSTS_ENABLED
    _SIGNPOSTS_ENABLED = bool(enabled)


def _signpost(header: str) -> None:
    if _SIGNPOSTS_ENABLED and _tracy_signpost is not None and not _IN_TRACE_CAPTURE:
        _tracy_signpost(header=header)


@contextmanager
def _region(name: str):
    """Wrap the enclosed ttnn ops in a Tracy ``<name>_START`` / ``<name>_END`` pair."""
    _signpost(f"{name}_START")
    try:
        yield
    finally:
        _signpost(f"{name}_END")


@contextmanager
def _trace_capture_guard():
    """Silence :func:`_profile` for the duration of a trace capture."""
    global _IN_TRACE_CAPTURE
    prev = _IN_TRACE_CAPTURE
    _IN_TRACE_CAPTURE = True
    try:
        yield
    finally:
        _IN_TRACE_CAPTURE = prev


# fp32 accumulation everywhere keeps the long (Dh=512) reductions and the
# softmax/RoPE chains from drifting under bf16; the per-layer PCC test needs it.
_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# The fused ``scaled_dot_product_attention_decode`` op must NOT run with
# ``fp32_dest_acc_en=True``: for this attention shape (head_dim=256, MQA with a
# single shared K==V head) that flag makes the kernel emit garbage (PCC ~0.36 vs
# the manual softmax). HiFi4 with bf16 dest accumulation matches the manual path
# at PCC ~0.9999. ``packer_l1_acc`` is safe to keep on.
_HIFI4_SDPA = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Additive-mask "-inf": a finite bf16-representable floor. Masked logits feed
# ``exp(x - max)`` which underflows to 0 for both this and a true ``-inf``, but
# the finite value avoids ``inf - inf -> NaN`` if a whole row were masked.
_MASK_NEG = -1.0e9


class DeepSeekV4Module:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
