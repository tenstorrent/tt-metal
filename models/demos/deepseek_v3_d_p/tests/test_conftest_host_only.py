# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A host-only test in this tree collects and runs without querying the devices.

The tree's conftest and its modules import ttnn, so ttnn cannot be blocked here as it is for the root
conftest. The query is what opens the devices: the arch, device count or cluster type, read through
ttnn's device and cluster bindings. The test runs a host-only probe in a child pytest in which each of
those bindings records its caller and raises, so a collection-time query in this conftest fails it.
"""

import os
import subprocess
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Set only in the child's environment: the file that collects every device query the child makes.
_QUERY_LOG_ENV = "TT_DS_CONFTEST_DEVICE_QUERY_LOG"

# The bindings that enumerate the devices; every Python helper (is_blackhole, get_num_devices, ...) calls one.
_DEVICE_QUERIES = {
    "device": ("get_arch_name", "GetNumAvailableDevices", "GetNumPCIeDevices"),
    "cluster": ("get_cluster_type",),
}

_GUARD_AND_RUN_PYTEST = (
    "import os, sys, pytest; "
    "from models.demos.deepseek_v3_d_p.tests.test_conftest_host_only import _guard_device_queries; "
    f"_guard_device_queries(os.environ['{_QUERY_LOG_ENV}']); sys.exit(pytest.main(sys.argv[1:]))"
)


def _guard_device_queries(log_path):
    """Replace every reference to a device-query binding in the loaded ttnn modules with a recorder that raises."""
    import ttnn

    def recorder(name):
        def query(*args, **kwargs):
            with open(log_path, "a") as log:
                log.write(f"{name} called from:\n{''.join(traceback.format_stack())}\n")
            raise RuntimeError(f"{name} queried the devices in a host-only session")

        return query

    originals = {}
    for submodule, names in _DEVICE_QUERIES.items():
        for name in names:
            originals[id(getattr(getattr(ttnn._ttnn, submodule), name))] = recorder(name)
    # Aliases bound while ttnn imported (ttnn._get_arch_name, ttnn.GetNumAvailableDevices, ...) are replaced too.
    modules = [m for n, m in list(sys.modules.items()) if n == "ttnn" or n.startswith("ttnn.")]
    modules += [m for m in vars(ttnn._ttnn).values() if type(m) is type(ttnn)]
    for module in modules:
        for attr, value in list(vars(module).items()):
            if id(value) in originals:
                setattr(module, attr, originals[id(value)])


def test_host_only_probe():
    """A host-only test in this tree; run alone by the test below."""
    if os.environ.get(_QUERY_LOG_ENV):
        import ttnn

        # The guard lives only in the child's own interpreter; a process that does not inherit it, such as an
        # xdist worker, would let the probe pass even if the conftest queried the devices.
        assert ttnn._ttnn.device.get_arch_name.__name__ == "query", "the probe must run in the guarded process"
    assert int("42") == 42


def test_host_only_test_queries_no_device(tmp_path):
    query_log = tmp_path / "device_queries.log"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _GUARD_AND_RUN_PYTEST,
            f"{Path(__file__).relative_to(_REPO_ROOT)}::test_host_only_probe",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_REPO_ROOT,
        # PYTEST_ADDOPTS may carry -n N (CI sim legs set it), which would move the probe into xdist workers.
        env={**{k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}, _QUERY_LOG_ENV: str(query_log)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert not query_log.exists(), query_log.read_text()
    assert result.returncode == 0, output
    assert "1 passed" in output, output
