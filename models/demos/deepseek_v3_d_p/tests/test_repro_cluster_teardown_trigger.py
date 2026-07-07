# Throwaway diagnostic — see .github/workflows/repro-cluster-teardown-indexerror.yaml
# Trigger arm: three requires_mesh_topology-marked items force
# pytest_collection_modifyitems (conftest.py:91, "this opens a device") to
# open+close three throwaway clusters during collection, before any test
# body runs. Hypothesis: the subsequent ttnn.get_pcie_device_ids() call —
# the same call that is the first line of the real mesh_device fixture
# (root conftest.py:548) — then finds the PCIe device map torn down and
# raises IndexError: unordered_map::at.
import pytest


@pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="line")
def test_probe_a():
    pass


@pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="line")
def test_probe_b():
    pass


@pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="line")
def test_probe_c():
    pass


def test_query_after_probes():
    import ttnn

    device_ids = ttnn.get_pcie_device_ids()
    assert len(device_ids) > 0, f"expected at least one PCIe device id, got {device_ids}"
