# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""General, eval-independent up-front precompile collector for ANY ttnn pytest suite.

Phase 2 of the "marriage": drive mechanism #1's op-agnostic C++ collector
(``ttnn.graph.up_front_*``) across a whole pytest session via the call-phase
hookwrapper, so EVERY op of EVERY test body — multi-op bodies and C++
ProgramDescriptor-migrated ops alike — is stashed and deduped, then the distinct
set is JIT-compiled once, in parallel, at session end (warming the on-disk
``TT_METAL_CACHE``).

Why a hookwrapper (not eval's re-invocation): pytest runs each body through its
*own* fixture machinery, so this works for arbitrary tests without knowing their
signatures. ``begin_collect(clear=False)`` wraps ONLY the call phase, so
NO_DISPATCH never spans fixture setup/teardown (the device is created for real in
setup); ``clear=False`` makes the collector ACCUMULATE across tests.

TWO PASSES (required): under NO_DISPATCH each body runs "neutered" (mocked addr-0
buffers, nothing dispatched), so its asserts fail and are swallowed — pass 1 only
collects + compiles. Re-run the SAME suite over the SAME on-disk cache for the
real, now-warm results:

    # pass 1 — collect every op + parallel-compile the distinct set (warms the cache)
    # (run from the repo root; PYTHONPATH=$PWD ensures the local `tests` package wins
    #  over any foreign tt-metal checkout on an inherited PYTHONPATH — see split-checkout)
    UP_FRONT_COLLECT=1 TT_METAL_CACHE=/tmp/c  PYTHONPATH="$PWD" pytest -p tests.plugins.up_front_collect <tests>
    # pass 2 — real run, warm (plugin off)
    TT_METAL_CACHE=/tmp/c  pytest <tests>

Knobs:  UP_FRONT_COLLECT=1 (enable; no-op otherwise) ·
        UP_FRONT_COLLECT_WORKERS=N (0 => hardware_concurrency) ·
        UP_FRONT_COLLECT_DEVICE_ID=N (device for the session-end compile, default 0)

Limitations (see MODEL_PRECOMPILE_DESIGN.md §8.1):
  * Tests that drive trace/graph capture are SKIPPED — NO_DISPATCH blocks the real
    dispatch + buffer allocation that recording needs. They cold-compile in pass 2.
  * A body that reads a tensor back / branches on values mid-body breaks on addr-0:
    we capture the ops before that point; the rest cold-compile in pass 2.
  * Ops that pick their program from live allocator state see empty L1 under
    NO_DISPATCH and may collect a different variant than the real run uses.
"""

from __future__ import annotations

import contextlib
import inspect
import os

import pytest

_ACTIVE = os.environ.get("UP_FRONT_COLLECT") == "1"
_WORKERS = int(os.environ.get("UP_FRONT_COLLECT_WORKERS", "0"))  # 0 => hardware_concurrency
_DEVICE_ID = int(os.environ.get("UP_FRONT_COLLECT_DEVICE_ID", "0"))
# UP_FRONT_REAL_ALLOC=1: collect with REAL buffer addresses (dispatch still blocked) instead of
# addr-0 mocking, so address-baked / address-branched kernels (pool reader, move fwd/bwd) warm.
# Costs real device memory (~the real run's peak) — only use when the model fits.
_REAL_ALLOC = os.environ.get("UP_FRONT_REAL_ALLOC") == "1"
# UP_FRONT_FAST_COLLECT=1 (default): in the collect window (results thrown away under NO_DISPATCH),
# replace expensive HOST-side torch work — torch.randn -> zeros, the torch *reference* conv
# (torch.nn.functional.conv2d) -> shape-correct zeros, comp_pcc -> no-op. Values are irrelevant
# under NO_DISPATCH and collected programs depend only on ttnn op shapes/config, so this does NOT
# change what is collected — it stops pass 1 paying for the golden reference + PCC + RNG.
_FAST_COLLECT = os.environ.get("UP_FRONT_FAST_COLLECT", "1") == "1"
# UP_FRONT_FAST_COLLECT_SHALLOW=1 (default): within fast collect, make ttnn.from_torch allocate the
# ttnn tensor SHAPE-ONLY (skip the host tilize/convert/copy) — program build needs only the spec, not
# values. Removes the dominant collect cost on weight-heavy models (e.g. SDXL UNet weight prep, ~35s
# of from_torch). Keeps REAL torch tensors so host-side weight prep value reads still work (unlike
# meta). Set =0 to restore the original full from_torch inside fast collect.
_FAST_SHALLOW = os.environ.get("UP_FRONT_FAST_COLLECT_SHALLOW", "1") == "1"
# UP_FRONT_META_COLLECT=1 (opt-in): the GENERIC version of fast-collect. Instead of per-op torch
# swaps (randn/conv2d/layer_norm/...), run the throwaway body on the torch META device: tensors
# carry shape+dtype but NO storage, and the dispatcher propagates shapes through ANY torch op with
# no compute and no allocation. This is the host-side analogue of the mock ttnn device — it removes
# the host memory + compute that OOM'd full-suite collects. The one ttnn<->torch boundary is bridged
# by allocate_tensor_on_device (from_torch) + a meta stand-in (to_torch), valid because program
# capture depends only on shapes/layout, not values. Falls back to real per-input on any edge case.
_META_COLLECT = os.environ.get("UP_FRONT_META_COLLECT") == "1"
# UP_FRONT_FAKE_COLLECT=1 (opt-in): the MAINTAINED version of meta-collect. Run the throwaway body
# under torch's FakeTensorMode (the same metadata-only-tensor machinery torch.compile/Dynamo/export
# use). Like meta, every torch op becomes storage-less shape propagation with no per-op patching; but
# FakeTensorMode closes the edges that make bare meta collapse: allow_non_fake_inputs=True lets fake
# tensors mix with the REAL fixture/param tensors that meta refuses (suspected cause of the
# module-test 0-programs collapse), allow_fallback_kernels=True runs the real kernel for any op with
# no fake rule instead of raising, and a ShapeEnv makes .item()/.tolist() return a SymInt instead of
# raising (drops the meta _item/_tolist stand-ins). One ttnn boundary adapter remains: from_torch
# shape-only alloc (reused from fast-shallow, fake-aware), to_torch -> fake stand-in. See
# tests/plugins/META_VS_FAKE_COLLECT.md. Precedence: FAKE > META > FAST when more than one is set.
_FAKE_COLLECT = os.environ.get("UP_FRONT_FAKE_COLLECT") == "1"
# UP_FRONT_NULL_COMPUTE=1 (opt-in): the LIST-FREE generalization. A TorchDispatchMode that intercepts
# EVERY aten op: factories/loads (no tensor input) run REAL (allocation is fine), every op that
# CONSUMES tensors has its compute SKIPPED — output shape via the meta kernel, then REAL zeros of
# that shape are allocated. No op list, no size threshold. Effectively a meta tensor WITH
# real storage: shapes propagate via torch's own meta kernels, but no FLOPs are spent and storage is
# real, so the things that broke fake/meta all work — Parameter wrap, real .item() (no SymInt), and
# from_pretrained (real construction + skipped weight-copy => loads in ~1.5s). Same ttnn boundary
# adapter as fast/fake (shape-only from_torch + comp_pcc stub). See tests/plugins/META_VS_FAKE_COLLECT.md.
# Precedence: NULL > FAKE > META > FAST when more than one is set.
_NULL_COMPUTE = os.environ.get("UP_FRONT_NULL_COMPUTE") == "1"
# (settled A/B, matmul 845 bodies: zeros == empty wall, fewer swallows, identical coverage)
# UP_FRONT_COLLECT_NO_COMPILE=1: collect/dedup only, skip the session-end parallel compile. Used to
# isolate the body-run cost from the compile floor when benchmarking (e.g., fast vs meta collect).
_NO_COMPILE = os.environ.get("UP_FRONT_COLLECT_NO_COMPILE") == "1"

# Bodies that drive trace/graph capture can't run under NO_DISPATCH (it blocks the
# dispatch + alloc that recording needs). Heuristic: scan the test function source.
_CAPTURE_MARKERS = (
    "begin_trace_capture",
    "capture_trace",
    "execute_trace",
    "begin_graph_capture",
    "begin_mesh_trace_capture",
)

_stats = {"bodies": 0, "swallowed": 0, "skipped_capture": 0}


def _torch_to_ttnn_dtype(torch_dtype):
    import torch
    import ttnn

    return {
        torch.float32: ttnn.float32,
        torch.float64: ttnn.float32,
        torch.bfloat16: ttnn.bfloat16,
        torch.float16: ttnn.bfloat16,
        torch.int32: ttnn.uint32,
        torch.int64: ttnn.uint32,
        torch.int16: ttnn.uint16,
        torch.uint8: ttnn.uint8,
        torch.bool: ttnn.uint8,
    }.get(torch_dtype, ttnn.bfloat16)


@contextlib.contextmanager
def _shape_only_boundary(stats):
    """Patch the ttnn<->torch boundary to allocate device tensors SHAPE-ONLY during collect.

    Program build needs only the spec, so ttnn.from_torch and torch2tt_tensor (incl. their
    `from ... import` bindings) skip the host tilize/convert/copy — the dominant collect cost on
    weight-heavy bodies. Falls back to the real path for anything not safely shape-only
    (mesh_mapper, custom tile, host tensors, meta tensors). `stats` gets shallow/fallback counts.
    """
    import sys

    import ttnn

    real_from_torch = ttnn.from_torch

    def _shallow_from_torch(
        tensor,
        dtype=None,
        *,
        spec=None,
        tile=None,
        layout=None,
        device=None,
        memory_config=None,
        mesh_mapper=None,
        **kw,
    ):
        if (
            tensor is not None
            and device is not None
            and mesh_mapper is None
            and tile is None
            and not getattr(tensor, "is_meta", False)
        ):
            try:
                if spec is not None:
                    dt, lay, mc = spec.dtype, spec.layout, spec.memory_config
                    shp = ttnn.Shape(tuple(spec.shape))
                else:
                    dt = dtype or _torch_to_ttnn_dtype(tensor.dtype)
                    lay = layout or ttnn.ROW_MAJOR_LAYOUT
                    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
                    shp = ttnn.Shape(tuple(tensor.shape))
                t = ttnn.allocate_tensor_on_device(shp, dt, lay, device, mc)
                stats["shallow"] += 1
                return t
            except Exception:
                stats["fallback"] += 1
        return real_from_torch(
            tensor,
            dtype,
            spec=spec,
            tile=tile,
            layout=layout,
            device=device,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
            **kw,
        )

    # torch2tt_tensor builds device tensors outside from_torch (ttnn.Tensor(t).to(...)); same lever.
    t2tt_modules = []
    _ucf = sys.modules.get("models.common.utility_functions")
    real_t2tt = getattr(_ucf, "torch2tt_tensor", None) if _ucf is not None else None
    if real_t2tt is not None:
        _interleaved_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED)

        def _shallow_torch2tt(
            py_tensor, tt_device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=_interleaved_mc, tt_dtype=ttnn.bfloat16
        ):
            if tt_device is not None and not getattr(py_tensor, "is_meta", False):
                try:
                    size = list(py_tensor.size())
                    while len(size) < 4:
                        size.insert(0, 1)
                    t = ttnn.allocate_tensor_on_device(
                        ttnn.Shape(tuple(size)), tt_dtype, tt_layout, tt_device, tt_memory_config
                    )
                    stats["shallow"] += 1
                    return t
                except Exception:
                    stats["fallback"] += 1
            return real_t2tt(py_tensor, tt_device, tt_layout, tt_memory_config, tt_dtype)

        for m in list(sys.modules.values()):
            try:
                if getattr(m, "torch2tt_tensor", None) is real_t2tt:
                    t2tt_modules.append(m)
                    m.torch2tt_tensor = _shallow_torch2tt
            except Exception:
                pass

    ttnn.from_torch = _shallow_from_torch
    try:
        yield
    finally:
        ttnn.from_torch = real_from_torch
        for m in t2tt_modules:
            m.torch2tt_tensor = real_t2tt


@contextlib.contextmanager
def _stub_verifiers():
    """Fuse off the host-side numeric verifiers during collect (results are meaningless anyway).

    comp_pcc -> (True, ~1) and assert_numeric_metrics -> pass: a verifier failing on addr-0/zeros
    readback aborts the rest of a multi-op body, losing its ops. Both helpers may also be bound via
    `from ... import` in test modules, so patch every module attr pointing at the original.
    """
    import sys

    saved = []  # (module, attr, original)
    targets = {}
    for src, attr, stub in (
        ("tests.ttnn.utils_for_testing", "comp_pcc", lambda *a, **k: (True, 0.999999)),
        ("models.common.utility_functions", "comp_pcc", lambda *a, **k: (True, 0.999999)),
        ("tests.ttnn.utils_for_testing", "assert_numeric_metrics", lambda *a, **k: (True, "collect-stub")),
    ):
        mod = sys.modules.get(src)
        orig = getattr(mod, attr, None) if mod is not None else None
        if orig is not None:
            targets[(attr, orig)] = stub
    for (attr, orig), stub in targets.items():
        for m in list(sys.modules.values()):
            try:
                if getattr(m, attr, None) is orig:
                    saved.append((m, attr, orig))
                    setattr(m, attr, stub)
            except Exception:
                pass
    try:
        yield
    finally:
        for m, attr, orig in saved:
            setattr(m, attr, orig)


def _drives_capture(item) -> bool:
    fn = getattr(item, "function", None)
    if fn is None:
        return False
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    return any(m in src for m in _CAPTURE_MARKERS)


@contextlib.contextmanager
def _cheap_host_ops():
    """Swap expensive host-side torch work for cheap shape-correct stand-ins during collect.

    Only ttnn op shapes/config determine the collected programs, so zeroing the inputs, the
    torch reference conv, and the PCC check does not change WHAT is collected — it just removes
    the wasted golden-reference + PCC + RNG cost from the throwaway collect pass.
    """
    import torch
    import torch.nn.functional as F

    real_randn = torch.randn
    real_rand = torch.rand
    real_conv2d = F.conv2d
    real_layer_norm = F.layer_norm
    real_matmul = torch.matmul
    real_bmm = torch.bmm
    real_tensor_matmul = torch.Tensor.matmul
    real_tensor_rmatmul = torch.Tensor.__matmul__

    def _fast_randn(*size, **kw):
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            size = tuple(size[0])
        kw.pop("generator", None)
        return torch.zeros(*size, **kw)

    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    def _fast_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        sH, sW = _pair(stride)
        pH, pW = _pair(padding)
        dH, dW = _pair(dilation)
        N, C_out = input.shape[0], weight.shape[0]
        H, W = input.shape[-2], input.shape[-1]
        kH, kW = weight.shape[-2], weight.shape[-1]
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        return input.new_zeros((N, C_out, H_out, W_out))

    def _fast_layer_norm(input, *a, **k):
        return torch.zeros_like(input)  # layer_norm / rms_norm output shape == input shape

    def _fast_matmul(a, b):
        # Host golden reference matmul/@/.matmul()/bmm -> shape-correct zeros, NO GEMM compute.
        # Output shape via meta inference (no data) so all broadcasting rules stay exact. Only the
        # host reference uses these; ttnn.matmul is untouched, so the collected programs are unchanged.
        try:
            shp = tuple(real_matmul(a.to("meta"), b.to("meta")).shape)
            return torch.zeros(shp, dtype=getattr(a, "dtype", torch.bfloat16), device=getattr(a, "device", "cpu"))
        except Exception:
            return real_matmul(a, b)

    torch.randn = _fast_randn
    torch.rand = _fast_randn
    F.conv2d = _fast_conv2d
    F.layer_norm = _fast_layer_norm
    torch.matmul = _fast_matmul
    torch.bmm = lambda a, b, *x, **k: _fast_matmul(a, b)
    torch.Tensor.matmul = lambda self, other: _fast_matmul(self, other)
    torch.Tensor.__matmul__ = lambda self, other: _fast_matmul(self, other)
    boundary = _shape_only_boundary(_fast_stats) if _FAST_SHALLOW else contextlib.nullcontext()
    try:
        with boundary, _stub_verifiers():
            yield
    finally:
        torch.randn = real_randn
        torch.rand = real_rand
        F.conv2d = real_conv2d
        F.layer_norm = real_layer_norm
        torch.matmul = real_matmul
        torch.bmm = real_bmm
        torch.Tensor.matmul = real_tensor_matmul
        torch.Tensor.__matmul__ = real_tensor_rmatmul


_meta_stats = {"from_torch_alloc": 0, "from_torch_fallback": 0, "to_torch_meta": 0}
_fast_stats = {"shallow": 0, "fallback": 0}
_fake_stats = {"alloc": 0, "fallback": 0, "giveup": 0, "to_torch": 0}
_null_stats = {"skipped": 0, "ran_real": 0, "fallback": 0, "memo_fallback": 0, "shallow": 0}
# Aten overloads whose meta call raised (no meta kernel / data-dependent shape): the meta attempt
# costs an exception unwind per call, so memoize per overload — across bodies — and go straight to
# the fallback path. Skipping meta only changes cost (run real / placeholder), never collection.
_null_no_meta: set = set()


@contextlib.contextmanager
def _meta_host_ops():
    """Generic fake-torch collect: run the throwaway body on META tensors (no storage, no compute).

    Replaces the per-op swap list in _cheap_host_ops. Forces torch tensor creation onto the meta
    device so inputs AND every torch reference op become storage-less shape propagation (the
    dispatcher handles all ops generically — no per-op patching, no allocation -> no OOM). Bridges
    the ttnn boundary: from_torch(meta) -> allocate_tensor_on_device (shape-only, no host copy),
    to_torch -> meta stand-in. comp_pcc is a no-op. Any path that can't be meta'd falls back to a
    real shape-correct zeros for that one tensor, so the body still reaches the ttnn op to capture it.
    """
    import torch
    import ttnn

    real = {
        n: getattr(torch, n)
        for n in (
            "randn",
            "rand",
            "zeros",
            "ones",
            "empty",
            "full",
            "arange",
            "randint",
            "tensor",
            "as_tensor",
            "eye",
            "linspace",
            "zeros_like",
            "ones_like",
            "empty_like",
            "randn_like",
            "rand_like",
        )
        if hasattr(torch, n)
    }
    real_from_torch = ttnn.from_torch
    real_to_torch = ttnn.to_torch
    real_item = torch.Tensor.item
    real_tolist = torch.Tensor.tolist

    def _mk(orig):
        def wrapped(*a, **kw):
            kw.setdefault("device", "meta")
            kw.pop("generator", None)
            try:
                return orig(*a, **kw)
            except (TypeError, RuntimeError):
                kw.pop("device", None)  # creator doesn't accept device= (rare) -> real
                return orig(*a, **kw)

        return wrapped

    def _meta_from_torch(
        tensor, dtype=None, *, spec=None, layout=None, device=None, memory_config=None, mesh_mapper=None, **kw
    ):
        is_meta = tensor is not None and getattr(tensor, "is_meta", False)
        if is_meta and device is not None:
            try:
                if spec is not None:
                    dt, lay, mc = spec.dtype, spec.layout, spec.memory_config
                    shp = ttnn.Shape(tuple(spec.shape))
                else:
                    dt = dtype or _torch_to_ttnn_dtype(tensor.dtype)
                    lay = layout or ttnn.ROW_MAJOR_LAYOUT
                    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
                    shp = ttnn.Shape(tuple(tensor.shape))
                t = ttnn.allocate_tensor_on_device(shp, dt, lay, device, mc)
                _meta_stats["from_torch_alloc"] += 1
                return t
            except Exception:
                pass  # fall through to materialize
        if is_meta:
            # device=None (host tensor) or alloc failed -> materialize a real shape-correct zeros
            _meta_stats["from_torch_fallback"] += 1
            tensor = real["zeros"](tuple(tensor.shape), dtype=tensor.dtype)
        return real_from_torch(
            tensor,
            dtype,
            spec=spec,
            layout=layout,
            device=device,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
            **kw,
        )

    def _meta_to_torch(tensor, *a, **kw):
        try:
            _meta_stats["to_torch_meta"] += 1
            return real["zeros"](tuple(tensor.shape), device="meta")
        except Exception:
            return real_to_torch(tensor, *a, **kw)

    # .item()/.tolist() RAISE on meta tensors. A body that reads a scalar out of a tensor to drive
    # HOST-side control flow before its ttnn ops (e.g. test_layer_norm_with_padding does
    # `torch.randint(1, w+1, (1,)).item()` on line 1) would abort under meta collect BEFORE any ttnn
    # op is stashed -> the whole test (and its fill_pad/layer_norm kernels) goes uncollected. The
    # extracted value only feeds input *data*, never the ttnn program's shape/config (which is all
    # that determines the collected kernels), so a deterministic stand-in is safe and lets the body
    # reach its ops. Only meta tensors are intercepted; real tensors keep exact semantics.
    def _meta_item(self):
        if getattr(self, "is_meta", False):
            return 1
        return real_item(self)

    def _meta_tolist(self):
        if getattr(self, "is_meta", False):
            return real_tolist(real["zeros"](tuple(self.shape)))
        return real_tolist(self)

    for name, orig in real.items():
        setattr(torch, name, _mk(orig))
    ttnn.from_torch = _meta_from_torch
    ttnn.to_torch = _meta_to_torch
    torch.Tensor.item = _meta_item
    torch.Tensor.tolist = _meta_tolist
    try:
        with _stub_verifiers():
            yield
    finally:
        for name, orig in real.items():
            setattr(torch, name, orig)
        ttnn.from_torch = real_from_torch
        ttnn.to_torch = real_to_torch
        torch.Tensor.item = real_item
        torch.Tensor.tolist = real_tolist


@contextlib.contextmanager
def _fake_host_ops():
    """Generic fake-torch collect, the maintained sibling of _meta_host_ops (FakeTensorMode).

    Runs the throwaway body inside ``FakeTensorMode``: every torch op becomes storage-less shape
    propagation handled generically by ``__torch_dispatch__`` (no per-op swap list, like meta), but
    with the edges that make bare meta collapse closed — ``allow_non_fake_inputs`` (mix with the REAL
    fixture/param tensors meta refuses), ``allow_fallback_kernels`` (real kernel for an unimplemented
    op instead of a raise), and a ``ShapeEnv`` (``.item()``/``.tolist()`` yield a SymInt, not a raise).

    The one ttnn<->torch boundary is bridged exactly as fast-shallow does: ``from_torch`` allocates
    the ttnn tensor SHAPE-ONLY (program build needs only the spec), now fake-aware — a FakeTensor has
    no storage so ``real_from_torch`` can't consume it, and when shape-only is impossible
    (mesh_mapper/tile/symbolic dim) we ``unset_fake_temporarily`` to materialize a real shape-correct
    zeros (a plain ``torch.zeros`` here would itself be faked). ``to_torch`` returns a fake stand-in.
    Anything that can't be concretized (a data-dependent dim) is left to raise -> swallowed by the
    hookwrapper with the ops stashed so far kept, same graceful degradation as meta.
    """
    import torch
    import ttnn
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    real_from_torch = ttnn.from_torch
    real_to_torch = ttnn.to_torch

    def _concrete(shape):
        # Plain-int tuple; RAISES (GuardOnDataDependentSymNode) on an unbacked/symbolic dim.
        return tuple(int(d) for d in shape)

    def _fake_from_torch(
        tensor,
        dtype=None,
        *,
        spec=None,
        tile=None,
        layout=None,
        device=None,
        memory_config=None,
        mesh_mapper=None,
        **kw,
    ):
        # Shape-only allocate for ANY non-meta torch tensor (fake OR a real fixture/weight tensor) —
        # under collect only the spec matters, so this skips the host tilize/convert/copy that is the
        # dominant collect cost on weight-heavy models. mesh_mapper/tile/custom paths fall through.
        if (
            tensor is not None
            and device is not None
            and mesh_mapper is None
            and tile is None
            and not getattr(tensor, "is_meta", False)
        ):
            try:
                if spec is not None:
                    dt, lay, mc = spec.dtype, spec.layout, spec.memory_config
                    shp = ttnn.Shape(_concrete(spec.shape))
                else:
                    dt = dtype or _torch_to_ttnn_dtype(tensor.dtype)
                    lay = layout or ttnn.ROW_MAJOR_LAYOUT
                    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
                    shp = ttnn.Shape(_concrete(tensor.shape))
                t = ttnn.allocate_tensor_on_device(shp, dt, lay, device, mc)
                _fake_stats["alloc"] += 1
                return t
            except Exception:
                _fake_stats["fallback"] += 1
        # Couldn't shape-only. real_from_torch can't consume a FakeTensor (no storage) -> materialize a
        # real shape-correct zeros outside the mode. If the shape is symbolic we can't, so give up and
        # let real_from_torch raise (the hookwrapper swallows it, keeping ops stashed so far).
        if isinstance(tensor, FakeTensor):
            try:
                with unset_fake_temporarily():
                    tensor = torch.zeros(_concrete(tensor.shape), dtype=tensor.dtype)
            except Exception:
                _fake_stats["giveup"] += 1
        return real_from_torch(
            tensor,
            dtype,
            spec=spec,
            tile=tile,
            layout=layout,
            device=device,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
            **kw,
        )

    def _fake_to_torch(tensor, *a, **kw):
        # Under NO_DISPATCH the readback is meaningless; return a shape-correct stand-in. Inside the
        # mode torch.zeros is itself faked, so the result flows on as a (cheap) fake tensor.
        try:
            _fake_stats["to_torch"] += 1
            return torch.zeros(tuple(int(d) for d in tensor.shape))
        except Exception:
            return real_to_torch(tensor, *a, **kw)

    ttnn.from_torch = _fake_from_torch
    ttnn.to_torch = _fake_to_torch
    # allow_non_fake_inputs: mix with real fixture tensors. allow_fallback_kernels (default True):
    # run the real kernel for an op with no fake rule. shape_env: .item()/.tolist() -> SymInt not raise.
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    try:
        with _stub_verifiers(), fake_mode:
            yield  # body runs on fake tensors (no storage/compute); ttnn C++ ops still stash
    finally:
        ttnn.from_torch = real_from_torch
        ttnn.to_torch = real_to_torch


def _make_null_compute():
    """Build the _NullCompute TorchDispatchMode class (deferred so torch imports stay lazy)."""
    import torch
    from torch.utils._pytree import tree_flatten, tree_map
    from torch.utils._python_dispatch import TorchDispatchMode

    def _to_meta(x):
        return x.to("meta") if isinstance(x, torch.Tensor) else x

    class _NullCompute(TorchDispatchMode):
        """A meta tensor WITH real storage: shapes propagate via meta kernels, no FLOPs are spent.

        Intercepts EVERY aten op — no op list, no size threshold:
          * no tensor input (factory/load): run REAL, except randn/rand -> zeros (values are never
            load-bearing — masks/indices come from randint/arange, which stay real — and real RNG
            was the dominant residual cost vs fast).
          * tensor-consuming op: skip the compute — output SHAPE via the op's meta kernel, then a
            REAL zeros of that shape. zeros, not empty: deterministic values keep any unstubbed
            verifier/control-flow benign (zeros==zeros) at memset cost (measured: wall-neutral).
          * meta kernel raised (none registered / data-dependent shape): run the op for real, and
            memoize the overload so later calls skip the exception-per-call tax.
        Reentrancy is safe: the mode is suppressed inside its own __torch_dispatch__. Real storage
        is what meta/fake lacked — nn.Parameter wrap, real .item(), from_pretrained all work.
        """

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            tensor_args = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
            if not tensor_args:
                if func._overloadpacket._qualified_op_name in ("aten::randn", "aten::rand"):
                    _null_stats["skipped"] += 1
                    kwargs.pop("generator", None)
                    return torch.zeros(*args, **kwargs)
                _null_stats["ran_real"] += 1
                return func(*args, **kwargs)
            if func in _null_no_meta:
                _null_stats["memo_fallback"] += 1
                return func(*args, **kwargs)
            try:
                out = func(*tree_map(_to_meta, args), **tree_map(_to_meta, kwargs))
            except Exception:
                _null_no_meta.add(func)
                _null_stats["fallback"] += 1
                return func(*args, **kwargs)
            _null_stats["skipped"] += 1
            dev = tensor_args[0].device
            return tree_map(
                lambda m: torch.zeros(m.shape, dtype=m.dtype, device=dev) if isinstance(m, torch.Tensor) else m,
                out,
            )

    return _NullCompute


@contextlib.contextmanager
def _null_compute_host_ops():
    """List-free generic collect: _NullCompute over the body + the shared ttnn boundary patches."""
    import torch
    import ttnn

    real_to_torch = ttnn.to_torch

    def _null_to_torch(tensor, *a, **kw):
        try:
            return torch.zeros(tuple(int(d) for d in tensor.shape))
        except Exception:
            return real_to_torch(tensor, *a, **kw)

    ttnn.to_torch = _null_to_torch
    try:
        with _shape_only_boundary(_null_stats), _stub_verifiers(), _make_null_compute()():
            yield  # body runs with real storage but zero FLOPs; ttnn ops still stash
    finally:
        ttnn.to_torch = real_to_torch


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Run each body under one NO_DISPATCH collect window (call phase only)."""
    if not _ACTIVE:
        return (yield)

    import ttnn

    if _drives_capture(item):
        _stats["skipped_capture"] += 1
        return (yield)  # let it run normally; it cold-compiles in pass 2

    _stats["bodies"] += 1
    # UP_FRONT_LOG_SWALLOWED=1: per-body diagnostic — which tests threw under NO_DISPATCH and
    # how many ops each stashed before the throw (reveals coverage: a body that throws BEFORE
    # stashing its op is a real miss; one that throws on the addr-0 readback after stashing is fine).
    _log = os.environ.get("UP_FRONT_LOG_SWALLOWED")
    n_before = ttnn.graph.up_front_num_collected() if _log else 0
    swallowed = False
    exc_info = ""
    ttnn.graph.up_front_begin_collect(clear=False, real_alloc=_REAL_ALLOC)  # accumulate; wraps ONLY the body
    try:
        if _NULL_COMPUTE:
            with _null_compute_host_ops():
                return (yield)  # body runs with real storage but zero FLOPs; ttnn ops still stash
        if _FAKE_COLLECT:
            with _fake_host_ops():
                return (yield)  # body runs on FakeTensors (no storage/compute); ttnn ops still stash
        if _META_COLLECT:
            with _meta_host_ops():
                return (yield)  # body runs on meta (no storage/compute); ttnn ops still stash
        if _FAST_COLLECT:
            with _cheap_host_ops():
                return (yield)  # body runs with cheap host stand-ins; ttnn ops still stash
        return (yield)  # pytest runs the body with its real fixtures; ops stash into the collector
    except Exception as e:
        # Expected under NO_DISPATCH: a readback/assert on an addr-0 output. The
        # program was already stashed by the funnel before the failure. Swallow so
        # pass 1 stays green (its results are meaningless — pass 2 is the real run).
        _stats["swallowed"] += 1
        swallowed = True
        exc_info = f"{type(e).__name__}: {str(e)[:140]}"
        return None
    finally:
        ttnn.graph.up_front_end_collect()
        if _log:
            n_after = ttnn.graph.up_front_num_collected()
            print(
                f"UP_FRONT_BODY: {'SWALLOWED' if swallowed else 'ok       '} "
                f"stashed={n_after - n_before:<3} {item.nodeid}" + (f"  -> {exc_info}" if swallowed else ""),
                flush=True,
            )


def pytest_sessionstart(session):
    if not _ACTIVE:
        return
    import ttnn

    ttnn.graph.up_front_clear()  # clean slate before accumulating across the session


def pytest_sessionfinish(session, exitstatus):
    if not _ACTIVE:
        return
    import ttnn

    n_unique = ttnn.graph.up_front_num_unique()
    n_collected = ttnn.graph.up_front_num_collected()
    print(
        f"\nUP_FRONT_COLLECT: {n_collected} ops stashed across {_stats['bodies']} bodies "
        f"-> {n_unique} unique programs "
        f"(swallowed {_stats['swallowed']}, skipped-capture {_stats['skipped_capture']})",
        flush=True,
    )
    if _NULL_COMPUTE:
        print(
            f"UP_FRONT_COLLECT: null-compute — aten ops skipped={_null_stats['skipped']} "
            f"ran-real(factory)={_null_stats['ran_real']} fallback={_null_stats['fallback']} "
            f"memo-fallback={_null_stats['memo_fallback']}, from_torch shape-only={_null_stats['shallow']}",
            flush=True,
        )
    elif _FAKE_COLLECT:
        print(
            f"UP_FRONT_COLLECT: fake-collect boundary — from_torch alloc={_fake_stats['alloc']} "
            f"fallback={_fake_stats['fallback']} giveup={_fake_stats['giveup']}, "
            f"to_torch={_fake_stats['to_torch']}",
            flush=True,
        )
    elif _META_COLLECT:
        print(
            f"UP_FRONT_COLLECT: meta-collect boundary — from_torch alloc={_meta_stats['from_torch_alloc']} "
            f"fallback={_meta_stats['from_torch_fallback']}, to_torch meta={_meta_stats['to_torch_meta']}",
            flush=True,
        )
    elif _FAST_COLLECT and _FAST_SHALLOW:
        print(
            f"UP_FRONT_COLLECT: fast-shallow from_torch — shape-only={_fast_stats['shallow']} "
            f"fallback={_fast_stats['fallback']}",
            flush=True,
        )
    if n_unique == 0:
        print("UP_FRONT_COLLECT: nothing to compile", flush=True)
        return
    if _NO_COMPILE:
        print(f"UP_FRONT_COLLECT: NO_COMPILE set — collected {n_unique}, skipping compile", flush=True)
        return

    device = None
    try:
        device = ttnn.CreateDevice(device_id=_DEVICE_ID)
        try:
            device.enable_program_cache()
        except Exception:
            pass
        n_prog, n_err, used, wall = ttnn.graph.up_front_compile(device, _WORKERS, True)
        print(
            f"UP_FRONT_COLLECT: compiled {n_prog} programs in {wall:.1f}s "
            f"(workers={used}, errors={n_err}) -> on-disk JIT cache warm",
            flush=True,
        )
    except Exception as e:  # best-effort: never break the session
        print(f"UP_FRONT_COLLECT: compile skipped (best-effort) — {e!r}", flush=True)
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:
                pass
