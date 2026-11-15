"""Compatibility shim: provide ttnn.tools.trace_allocation_tracker on ttnn
builds that predate it. Both hooks are only active when
TT_METAL_TRACE_ALLOC_TRACKING=1; without it they are no-ops, so stubbing
preserves behavior on older ttnn wheels."""
import contextlib
import sys
import types

try:
    import ttnn.tools.trace_allocation_tracker  # noqa: F401
except ImportError:
    if "ttnn.tools" not in sys.modules:
        _tools = types.ModuleType("ttnn.tools")
        _tracker = types.ModuleType("ttnn.tools.trace_allocation_tracker")

        def _acknowledge_corruptible(value=None):
            pass

        @contextlib.contextmanager
        def _corruptible_allocation_scope(*args, **kwargs):
            yield

        _tracker.acknowledge_corruptible = _acknowledge_corruptible
        _tracker.corruptible_allocation_scope = _corruptible_allocation_scope
        _tools.trace_allocation_tracker = _tracker
        sys.modules["ttnn.tools"] = _tools
        sys.modules["ttnn.tools.trace_allocation_tracker"] = _tracker
