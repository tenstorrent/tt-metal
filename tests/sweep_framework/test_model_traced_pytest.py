# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generic pytest runner for model-traced sweep modules.

Auto-discovers sweep modules and parametrizes tests from vectors_export/.
Each vector becomes a pytest test case with per-test timeout and device
crash recovery.

Usage:
  # Run all modules:
  python -m pytest tests/sweep_framework/test_model_traced_pytest.py -v --timeout=30

  # Run specific module:
  python -m pytest tests/sweep_framework/test_model_traced_pytest.py -v --timeout=30 \
    -k "add_model_traced"

  # With tracer:
  python model_tracer/generic_ops_tracer.py \
    tests/sweep_framework/test_model_traced_pytest.py \
    -o trace.json -- --timeout=30 --module-name model_traced.add_model_traced

  # Filter by module (comma-separated):
  python -m pytest tests/sweep_framework/test_model_traced_pytest.py \
    --module-name "model_traced.add_model_traced,model_traced.silu_model_traced"
"""

import importlib
import json
import os
import sys
import time

import pytest

# Ensure sweep framework is importable
_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
if _SWEEP_DIR not in sys.path:
    sys.path.insert(0, _SWEEP_DIR)

from framework.serialize import deserialize_vector_structured
from framework.sweeps_logger import sweeps_logger as logger

try:
    import framework.tt_smi_util as tt_smi_util
except ImportError:
    tt_smi_util = None


# ── Custom CLI options ──────────────────────────────────────────────────────

# ── Vector loading ──────────────────────────────────────────────────────────

def _get_vector_source_dir(config):
    """Resolve the vectors directory."""
    explicit = config.getoption("--vector-source", None)
    if explicit:
        return explicit
    env = os.environ.get("TTNN_VECTORS_EXPORT_DIR")
    if env:
        return env
    return os.path.join(_SWEEP_DIR, "vectors_export")


def _load_vectors(vector_dir, module_name, suite_name):
    """Load vectors for a specific module from vectors_export/."""
    vectors = []
    for fname in sorted(os.listdir(vector_dir)):
        if not fname.startswith(module_name) or not fname.endswith(".json"):
            continue
        if fname == "generation_manifest.json":
            continue
        fpath = os.path.join(vector_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {fpath}: {e}")
            continue

        suite_data = data.get(suite_name, data)
        if not isinstance(suite_data, dict):
            continue

        for input_hash, vector in suite_data.items():
            if not isinstance(vector, dict):
                continue
            if vector.get("validity") == "VectorValidity.INVALID":
                continue
            vector["_input_hash"] = input_hash
            vector["_source_file"] = fname
            vectors.append(vector)

    return vectors


def _discover_modules(vector_dir, module_filter=None):
    """Discover available sweep modules from vector files."""
    modules = set()
    for fname in os.listdir(vector_dir):
        if not fname.endswith(".json") or fname == "generation_manifest.json":
            continue
        # e.g. "model_traced.add_model_traced.hw_wormhole_n300_1c.mesh_1x1.json"
        # module name is everything before the first ".hw_" or ".mesh_" or the stem
        base = fname.replace(".json", "")
        for sep in [".hw_", ".mesh_"]:
            idx = base.find(sep)
            if idx > 0:
                base = base[:idx]
                break
        modules.add(base)

    if module_filter:
        filter_set = {m.strip() for m in module_filter.split(",")}
        modules = {m for m in modules if m in filter_set}

    return sorted(modules)


def _load_sweep_module(module_name):
    """Import a sweep module and return its run() function."""
    module_path = f"sweeps.{module_name}"
    try:
        if module_path in sys.modules:
            mod = importlib.reload(sys.modules[module_path])
        else:
            mod = importlib.import_module(module_path)
        return mod
    except Exception as e:
        logger.error(f"Failed to import {module_path}: {e}")
        return None


def _extract_run_params(vector, module_run_func):
    """Extract positional and keyword args from a vector for the run() function."""
    import inspect

    sig = inspect.signature(module_run_func)
    params = list(sig.parameters.keys())

    kwargs = dict(vector)

    # Remove internal keys
    for key in ["_input_hash", "_source_file", "validity", "invalid_reason",
                "status", "sweep_name", "suite_name", "timestamp", "tag",
                "input_hash"]:
        kwargs.pop(key, None)

    # Extract positional params that run() expects explicitly
    positional = {}
    for p in params:
        if p in ("device", "self"):
            continue
        if p in kwargs:
            positional[p] = kwargs.pop(p)

    return positional, kwargs


# ── Device management ───────────────────────────────────────────────────────

class DeviceManager:
    """Manages mesh device with crash recovery."""

    def __init__(self):
        self.device = None
        self.arch_name = None
        self._reset_util = None
        self._l1_small_size = None

    def get_or_open(self, l1_small_size=None):
        if self.device is not None:
            if l1_small_size and l1_small_size != self._l1_small_size:
                self.close()
            else:
                return self.device

        import ttnn

        mesh_env = os.environ.get("MESH_DEVICE_SHAPE", "").strip()
        if mesh_env and "x" in mesh_env:
            rows, cols = (int(x) for x in mesh_env.split("x"))
            mesh_shape = ttnn.MeshShape(rows, cols)
        else:
            mesh_shape = ttnn.MeshShape(1, 1)

        kwargs = {}
        if l1_small_size:
            kwargs["l1_small_size"] = l1_small_size
        self._l1_small_size = l1_small_size

        self.device = ttnn.open_mesh_device(mesh_shape, **kwargs)
        self.arch_name = ttnn.get_arch_name()
        if tt_smi_util:
            self._reset_util = tt_smi_util.ResetUtil(self.arch_name)
        logger.info(f"Opened mesh device {mesh_shape} (l1_small_size={l1_small_size})")
        return self.device

    def reopen(self):
        """Close, reset, and reopen the device."""
        import ttnn

        logger.warning("Reopening mesh device after failure...")
        if self.device is not None:
            try:
                ttnn.close_mesh_device(self.device)
            except Exception:
                pass
            self.device = None

        if self._reset_util:
            try:
                self._reset_util.reset()
            except Exception as e:
                logger.error(f"tt-smi reset failed: {e}")

        time.sleep(1)
        return self.get_or_open()

    def close(self):
        import ttnn

        if self.device is not None:
            try:
                ttnn.close_mesh_device(self.device)
            except Exception:
                pass
            self.device = None


_device_mgr = DeviceManager()


# ── Test generation ─────────────────────────────────────────────────────────

def pytest_generate_tests(metafunc):
    """Dynamically parametrize test_sweep with vectors from all discovered modules."""
    if "module_name" not in metafunc.fixturenames:
        return

    config = metafunc.config
    vector_dir = _get_vector_source_dir(config)
    suite_name = config.getoption("--suite-name", "model_traced")
    module_filter = config.getoption("--module-name", None)

    if not os.path.isdir(vector_dir):
        pytest.skip(f"Vector directory not found: {vector_dir}")
        return

    modules = _discover_modules(vector_dir, module_filter)
    if not modules:
        pytest.skip(f"No modules found in {vector_dir}")
        return

    test_params = []
    test_ids = []

    for module_name in modules:
        vectors = _load_vectors(vector_dir, module_name, suite_name)
        for vector in vectors:
            input_hash = vector.get("_input_hash", "unknown")
            test_params.append((module_name, vector))
            test_ids.append(f"{module_name}[{input_hash[:16]}]")

    metafunc.parametrize("module_name,vector", test_params, ids=test_ids)


# ── The single test function ───────────────────────────────────────────────

@pytest.mark.timeout(30)
def test_sweep(module_name, vector):
    """Run a single sweep vector through its module's run() function."""
    # Import the module
    mod = _load_sweep_module(module_name)
    assert mod is not None, f"Failed to import {module_name}"
    assert hasattr(mod, "run"), f"Module {module_name} has no run() function"

    # Detect l1_small_size from module's mesh_device_fixture if available
    l1_small_size = None
    if hasattr(mod, "mesh_device_fixture"):
        import inspect
        src = inspect.getsource(mod.mesh_device_fixture)
        if "l1_small_size" in src:
            import re
            m = re.search(r"l1_small_size\s*=\s*(\d+)", src)
            if m:
                l1_small_size = int(m.group(1))

    device = _device_mgr.get_or_open(l1_small_size=l1_small_size)

    # Deserialize vector
    deserialized = deserialize_vector_structured(dict(vector))

    # Extract params
    positional, kwargs = _extract_run_params(deserialized, mod.run)

    # Remove 'device' from kwargs if present — we pass it explicitly
    kwargs.pop("device", None)
    positional.pop("device", None)

    try:
        result = mod.run(**positional, device=device, **kwargs)
    except RuntimeError as e:
        err_str = str(e)
        if "TIMEOUT" in err_str or "unrecoverable" in err_str:
            logger.warning(f"Device error on {module_name}: {e}")
            _device_mgr.reopen()
            pytest.fail(f"Device hang: {e}")
        raise
    except Exception:
        raise

    # Validate result
    if isinstance(result, (list, tuple)) and len(result) >= 1:
        pcc = result[0]
        if isinstance(pcc, tuple):
            passed, pcc_value = pcc
        elif isinstance(pcc, (int, float)):
            passed = float(pcc) >= 0.999
            pcc_value = pcc
        else:
            passed = bool(pcc)
            pcc_value = pcc
        assert passed, f"PCC check failed: {pcc_value}"


# ── Session cleanup ────────────────────────────────────────────────────────

def pytest_sessionfinish(session, exitstatus):
    _device_mgr.close()
