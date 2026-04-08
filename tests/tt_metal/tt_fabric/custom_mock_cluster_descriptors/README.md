# Mock Cluster Descriptors

This directory contains YAML files for mock cluster descriptors used in testing and simulation of multi-chip systems without actual hardware.

## Generating Mock Cluster Descriptors

To generate a cluster descriptor from real hardware:

> This must be done per host on a multi-host system to capture all ethernet connections on all hosts

1. Obtain the ClusterDescriptor instance, for example from `get_cluster()` or other means in your code.
2. Call `serialize_to_file` on the ClusterDescriptor object, optionally providing a destination file path. It will serialize to a YAML file and return the path.

If no destination is provided, it uses a default path.

This creates a YAML representation of the current cluster that can be used for mocking.

Alternatively, from the cluster object obtained via `get_cluster()`, you can call a serialization method if available (refer to UMD source for details).

### Multi-host Mock Rank Bindings File
For multi-host setups, create mapping YAML files that associate ranks to specific descriptor files, e.g.:

```yaml
rank_to_cluster_mock_cluster_desc:
  - rank: 0
    filename: "path/to/desc_rank_0.yaml"
  - rank: 1
    filename: "path/to/desc_rank_1.yaml"
```

See more examples in [custom_mock_cluster_descriptors](./)

### Generating Mock Clusters By Hand
Currently, to manually generate these mock clusters are quite tedious, and would not be possible for large multi-host systems because of the complexity of ethernet connections.

There currently is an Github issue tracking this: https://github.com/tenstorrent/tt-metal/issues/29062

## Enabling Mock Cluster Usage

To enable mock mode using these descriptors:

- Set the environment variable `TT_METAL_MOCK_CLUSTER_DESC_PATH` to the path of your mock cluster YAML file. This switches the runtime to mock mode using the specified descriptor.

For distributed/multi-host mocking in Python (e.g., via ttrun.py):

- Use the `--mock-cluster-rank-binding` option with a mapping YAML file as shown above.

Refer to unit tests like `test_multi_host.cpp` for usage examples in tests.

These mocks are useful for testing fabric routing, control plane, and distributed features without hardware.

For examples, see: [run_cpp_fabric_tests.sh](../../../scripts/run_cpp_fabric_tests.sh)
