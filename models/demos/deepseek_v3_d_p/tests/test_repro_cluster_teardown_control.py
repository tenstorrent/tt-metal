# Throwaway diagnostic — see .github/workflows/repro-cluster-teardown-indexerror.yaml
# Control arm: no requires_mesh_topology-marked items in this file, so
# pytest_collection_modifyitems (conftest.py:91) never opens a probe cluster
# during collection. This mirrors every passing CI job we inspected, where
# ttnn.get_pcie_device_ids() is the first-ever device call in the process.
import ttnn


def test_query_with_no_prior_probes():
    device_ids = ttnn.get_pcie_device_ids()
    assert len(device_ids) > 0, f"expected at least one PCIe device id, got {device_ids}"
