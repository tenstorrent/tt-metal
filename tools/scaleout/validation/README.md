# Multi-Node Cluster Validation Tooling

This tool provides Tenstorrent Scaleout users with the ability to validate the connectivity and health of the Ethernet Interconnect backing their Multi-Node Cluster.

It performs Physical Discovery on the cluster, retrieving all Active Ethernet Connections between all visible chips.

The discovered state is then compared against an expected/golden representation of the cluster (passed in either as a Logical Cabling Descriptor and Deployment Descriptor, or as a Factory System Descriptor).

Any discrepancies, in the form of missing chips or connections are reported to the user.

Additionally, the user can choose to send point to point traffic across all discovered connections, and validate the health of the links.

For Multi-Node validation, this tool relies on MPI to be used as the distributed process launcher. In these cases, the user must wrap their validation call with mpirun, and pass in the appropriate hostfile or rankfile.

## Usage

Arguments required by the tool are captured below:

```
./build/tools/scaleout/run_cluster_validation --help

Utility to validate Ethernet Links and Connections for a Multi-Node TT Cluster
Compares live system state against the requested Cabling and Deployment Specifications

Arguments:
  --cabling-descriptor-path: Path to cabling descriptor
  --deployment-descriptor-path: Path to deployment descriptor
  --factory-descriptor-path: Path to factory descriptor
  --global-descriptor-path: Path to global descriptor (for cases where the user wants validate pregenerated cluster state against the expected state - live discovery will not be performed in this case)
  --output-path: Path to output directory
  --hard-fail: Fail on warning
  --log-ethernet-metrics: Log live ethernet statistics
  --print-connectivity: Print Ethernet Connectivity between ASICs
  --send-traffic: Send traffic across detected links
  --num-iterations: Number of iterations to send traffic
  --data-size: Data size (bytes) sent across each link per iteration
  --packet-size-bytes: Packet size (bytes) sent across each link
  --sweep-traffic-configs: Sweep pre-generated traffic configurations across detected links (stress testing)
  --help: Print usage information

To run on a multi-node cluster, use mpirun with a --hostfile option
```

Notes:
 - Validating connectivity requires the user to pass in an expected representation of the cluster. This can be done by specifying the `--cabling-descriptor-path` and `--deployment-descriptor-path` arguments
 - Alternatively, the user can pass in a Factory System Descriptor (FSD) file, which is a serialized representation of the expected cluster state. This can be done by specifying the `--factory-descriptor-path` argument
 - The descriptors mentioned above are deployment specific. Instructions for generating and using these descriptors can be found in the [Cabling Generator README](../README.md)
 - For cases where the user wants to run validation in an automated environment (CI/CD), the user can specify the `--hard-fail` argument to ensure that errors are thrown if missing connections or unhealthy links are detected
 - To validate the health of all discovered links, the user can choose to send point to point traffic across the interconnect, using the `--send-traffic` argument
 - Links can be stress tested by sweeping over pregenerated traffic configurations across the interconnect, using the `--sweep-traffic-configs` argument
 - Alternatively, for advanced use cases, the user can specify their own traffic pattern by passing in the data and packet size arguments, along with the number of iterations they want to send traffic for
 - Live Ethernet Metrics (after every iteration of sending traffic) can be logged to the terminal by using the `--log-ethernet-metrics` argument
 - Physical Connectivity between the ASICs can be logged to the terminal by using the `--print-connectivity` argument


## Extending to Multi-Node Clusters
As mentioned above, MPI can be used as a distributed process launcher for multi-node discovery and validation. This is fairly straightforward, once the user has an environment and MPI based networking setup on both hosts.

Example MPI Command:

```
mpirun --hostfile path/to/hostfile --deployment-specific-mpi-args ./build/tools/scaleout/run_cluster_validation --additional-arguments-depending-on-use-case
```

The user may need to pass in additional MPI arguments depending on their networking setup (example the NIC they want to bind traffic to, CPU binding policy, etc.).

The hostfile contains the list of hosts that the user wants to run validation on. An example hostfile is provided below (for a cluster of four hosts):
```
hostname0 slots=1
hostname1 slots=1
hostname2 slots=1
hostname3 slots=1
```

## Generated Output Files

The tool will generate the following output files to be viewed by the user:

- `global_system_descriptor_0.yaml`: The Global System Descriptor (GSD) for the cluster. This is a YAML file containing a serialized representation of the discovered cluster state. Compute Node and Cluster information are included in this file.
- `ethernet_metrics_report.csv`: This file contains Live Metrics collected across all links in the cluster, each time traffic was sent over the interconnect. This file is only generated when the `--log-ethernet-metrics` argument is used. Metrics include: Link Retrain Count, CRC Error Count, Corrected Codeword Count, Uncorrected Codeword Count, Mismatched Words

## Common Usage Patterns

Validate Connectivity Only (with expected cabling and deployment descriptors):
```
./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path <path_to_cabling_descriptor> --deployment-descriptor-path <path_to_deployment_descriptor>
```

Validate Connectivity Only (with expected factory system descriptor):
```
./build/tools/scaleout/run_cluster_validation --factory-descriptor-path <path_to_factory_descriptor>
```

Validate Connectivity and Send Traffic:
```
./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path <path_to_cabling_descriptor> --deployment-descriptor-path <path_to_deployment_descriptor> --send-traffic
```

Validate Connectivity and Run Stress Test:
```
./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path <path_to_cabling_descriptor> --deployment-descriptor-path <path_to_deployment_descriptor> --send-traffic --sweep-traffic-configs
```

Logging Live Ethernet Metrics and Printing Connectivity:
```
./build/tools/scaleout/run_cluster_validation --send-traffic --log-ethernet-metrics --print-connectivity
```

Stress Testing with Custom Traffic Configuration:
```
./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path <path_to_cabling_descriptor> --deployment-descriptor-path <path_to_deployment_descriptor> --send-traffic --data-size <data_size> --packet-size-bytes <packet_size_bytes> --num-iterations <num_iterations>
```

## Understanding the Terminal Output

If the cluster is in a healthy state, the user should see the following output:
```
| info     |     Distributed | ✓ All Detected Links are healthy.
```

In case the cluster is in an unhealthy state, the user will be which links are unhealthy, why they are unhealthy (missing connections, increasing Retrain Count, CRC Errors, etc.) and a failure report will be generated.

The log below is an example of a failure report that gets generated when links are unable to route traffic.

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              FAULTY LINKS REPORT                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
Total Faulty Link Occurrences: 40
Total Failure Instances: 40

Host                Tray  ASIC  Ch   Unique ID     Retrains    CRC Err       Uncorrected CW    Mismatch Words  Failure Type      Pkt Size    Data Size
---------------------------------------------------------------------------------------------------------------------------------------------------------
metal-wh-15         3     1     6    0x036190e058  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         3     1     0    0x036190e058  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         1     1     7    0x036190e019  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         1     1     0    0x036190e019  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         4     1     0    0x036190e00c  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         4     1     6    0x036190e00c  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         1     0     14   0x026190e019  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         1     0     6    0x026190e019  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         1     0     7    0x026190e019  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
metal-wh-15         3     1     1    0x036190e058  0           0x0           0x0               1               Data Mismatch     64 B        155648 B
```

If missing connections are detected, they will be logged to the terminal in the following format:
```
Channel Connections found in FSD but missing in GSD (11 connections):
  - PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=6}} <-> PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=6}}
  - PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=7}} <-> PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=7}}
  - PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=14}} <-> PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=14}}
  - PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=3, asic_channel=AsicChannel{asic_location=0, channel_id=15}} <-> PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=15}}
  - PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=4, asic_channel=AsicChannel{asic_location=0, channel_id=9}} <-> PhysicalChannelEndpoint{hostname='metal-wh-09', tray_id=4, asic_channel=AsicChannel{asic_location=1, channel_id=1}}
  - PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=14}} <-> PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=14}}
  - PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=1, asic_channel=AsicChannel{asic_location=0, channel_id=15}} <-> PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=15}}
  - PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=6}} <-> PhysicalChannelEndpoint{hostname='metal-wh-11', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=6}}
  - PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=7}} <-> PhysicalChannelEndpoint{hostname='metal-wh-11', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=7}}
  - PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=8}} <-> PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=1, channel_id=0}}
  - PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=0, channel_id=9}} <-> PhysicalChannelEndpoint{hostname='metal-wh-10', tray_id=2, asic_channel=AsicChannel{asic_location=1, channel_id=1}}

Port Connections found in FSD but missing in GSD (6 connections):
  - PhysicalPortEndpoint{hostname='metal-wh-09', aisle='A', rack=2, shelf_u=32, tray_id=1, port_type=QSFP_DD, port_id=1} <-> PhysicalPortEndpoint{hostname='metal-wh-09', aisle='A', rack=2, shelf_u=32, tray_id=4, port_type=QSFP_DD, port_id=1}
  - PhysicalPortEndpoint{hostname='metal-wh-09', aisle='A', rack=2, shelf_u=32, tray_id=3, port_type=WARP100, port_id=1} <-> PhysicalPortEndpoint{hostname='metal-wh-09', aisle='A', rack=2, shelf_u=32, tray_id=4, port_type=WARP100, port_id=1}
  - PhysicalPortEndpoint{hostname='metal-wh-09', aisle='A', rack=2, shelf_u=32, tray_id=4, port_type=TRACE, port_id=1} <-> PhysicalPortEndpoint{hostname='metal-wh-09', aisle='A', rack=2, shelf_u=32, tray_id=4, port_type=TRACE, port_id=2}
  - PhysicalPortEndpoint{hostname='metal-wh-10', aisle='A', rack=2, shelf_u=28, tray_id=1, port_type=WARP100, port_id=1} <-> PhysicalPortEndpoint{hostname='metal-wh-10', aisle='A', rack=2, shelf_u=28, tray_id=2, port_type=WARP100, port_id=1}
  - PhysicalPortEndpoint{hostname='metal-wh-10', aisle='A', rack=2, shelf_u=28, tray_id=2, port_type=TRACE, port_id=1} <-> PhysicalPortEndpoint{hostname='metal-wh-10', aisle='A', rack=2, shelf_u=28, tray_id=2, port_type=TRACE, port_id=2}
  - PhysicalPortEndpoint{hostname='metal-wh-10', aisle='A', rack=2, shelf_u=28, tray_id=2, port_type=QSFP_DD, port_id=1} <-> PhysicalPortEndpoint{hostname='metal-wh-11', aisle='A', rack=2, shelf_u=22, tray_id=2, port_type=QSFP_DD, port_id=1}
```

The logs above shows missing connections in a three host cluster (metal-wh-09, metal-wh-10, metal-wh-11). These connections have been specified in the Factory System Descriptor (FSD) but are not visible in the Global System Descriptor (GSD: Discovered State).

For most use cases, the user would be interested in the missing ports between Trays, since these correspond to either broken physical links or broken/disconnected cables.

Missing connections are physically specified in the following hierarchy:

```
Host (Hostname + Physical Location) -> Tray -> ASIC Location (Within the Tray) -> Port ID (Within the ASIC. This includes the cable type expected to be seen at this port).
```

To visualize the discovered connectivity, the user can use the `--print-connectivity` argument. This will print the connectivity between all ASICs and Compute Nodes in the cluster (grouped by hostname and cable type). Example output (for two connected compute nodes):


```
============================== HOST-LOCAL CONNECTIONS ===============================

  =============================== Hostname: metal-wh-09 ===============================

             ---------------------- Port Type: QSFP_DD ----------------------

 [metal-wh-09] Unique ID: 26190e0a4 Tray: 1, ASIC Location: 0, Ethernet Channel: 7
        Connected to [metal-wh-09] Unique ID: 26190e0cb Tray: 4, ASIC Location: 0, Ethernet Channel: 7

 [metal-wh-09] Unique ID: 26190e0a4 Tray: 1, ASIC Location: 0, Ethernet Channel: 6
        Connected to [metal-wh-09] Unique ID: 26190e0cb Tray: 4, ASIC Location: 0, Ethernet Channel: 6

 [metal-wh-09] Unique ID: 26190e0cb Tray: 4, ASIC Location: 0, Ethernet Channel: 7
        Connected to [metal-wh-09] Unique ID: 26190e0a4 Tray: 1, ASIC Location: 0, Ethernet Channel: 7

 [metal-wh-09] Unique ID: 26190e0cb Tray: 4, ASIC Location: 0, Ethernet Channel: 6
        Connected to [metal-wh-09] Unique ID: 26190e0a4 Tray: 1, ASIC Location: 0, Ethernet Channel: 6


  =============================== Hostname: metal-wh-10 ===============================

             ---------------------- Port Type: QSFP_DD ----------------------

 [metal-wh-10] Unique ID: 26190e008 Tray: 1, ASIC Location: 0, Ethernet Channel: 7
        Connected to [metal-wh-10] Unique ID: 26190e0d1 Tray: 4, ASIC Location: 0, Ethernet Channel: 7

 [metal-wh-10] Unique ID: 26190e008 Tray: 1, ASIC Location: 0, Ethernet Channel: 6
        Connected to [metal-wh-10] Unique ID: 26190e0d1 Tray: 4, ASIC Location: 0, Ethernet Channel: 6

 [metal-wh-10] Unique ID: 26190e0d1 Tray: 4, ASIC Location: 0, Ethernet Channel: 7
        Connected to [metal-wh-10] Unique ID: 26190e008 Tray: 1, ASIC Location: 0, Ethernet Channel: 7

 [metal-wh-10] Unique ID: 26190e0d1 Tray: 4, ASIC Location: 0, Ethernet Channel: 6
        Connected to [metal-wh-10] Unique ID: 26190e008 Tray: 1, ASIC Location: 0, Ethernet Channel: 6

 ============================== CROSS-HOST CONNECTIONS ===============================

  =============================== Hostname: metal-wh-09 ===============================

             ---------------------- Port Type: QSFP_DD ----------------------

 [metal-wh-09] Unique ID: 26190e003 Tray: 3, ASIC Location: 0, Ethernet Channel: 7
        Connected to [metal-wh-10] Unique ID: 26190e06f Tray: 3, ASIC Location: 0, Ethernet Channel: 7

 [metal-wh-09] Unique ID: 26190e003 Tray: 3, ASIC Location: 0, Ethernet Channel: 6
        Connected to [metal-wh-10] Unique ID: 26190e06f Tray: 3, ASIC Location: 0, Ethernet Channel: 6

  =============================== Hostname: metal-wh-10 ===============================

             ---------------------- Port Type: QSFP_DD ----------------------

 [metal-wh-10] Unique ID: 26190e06f Tray: 3, ASIC Location: 0, Ethernet Channel: 7
        Connected to [metal-wh-09] Unique ID: 26190e003 Tray: 3, ASIC Location: 0, Ethernet Channel: 7

 [metal-wh-10] Unique ID: 26190e06f Tray: 3, ASIC Location: 0, Ethernet Channel: 6
        Connected to [metal-wh-09] Unique ID: 26190e003 Tray: 3, ASIC Location: 0, Ethernet Channel: 6

```

## SuperPod and Multi-Mesh Fabric Testing

SuperPod testing goes beyond single-pod validation: it exercises the fabric across multiple pods (meshes) and requires matching Mesh Graph Descriptors, rank bindings, and MPI placement to your physical deployment.

### Mainlined Artifacts

The following artifacts are provided for minimal multi-pod (SuperPod) fabric testing:

- **Fabric test config (2D torus, multi-mesh):**
  `tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_multi_mesh.yaml`
  Defines fabric tests (e.g. RandomPairingMesh, AllToAllMesh, and 2D Torus XY variants) for Mesh and Torus topologies.

- **Minimal 4-mesh (e.g. 4× SuperPod, each pod 32×4) Mesh Graph Descriptor:**
  `tt_metal/fabric/mesh_graph_descriptors/dual_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`
  Describes four meshes (each 32×4, 4 hosts per mesh) with full 2D torus-style connectivity between meshes.

### Building Your Mesh Graph Descriptor

Your Mesh Graph Descriptor must match your physical SuperPod layout:

- **Number of meshes** — One graph instance per pod (or per logical mesh).
- **Shape of each mesh** — e.g. 32×4 (device dims) and host topology (e.g. 4×1) must match how many hosts and devices you have per pod.
- **Connectivity** — Inter-pod links (and optional torus wrap) must reflect how pods are wired (e.g. XY torus between pods).

Descriptor files live under `tt_metal/fabric/mesh_graph_descriptors/`. Use the mainlined `dual_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto` as a reference; adapt or add descriptors for different pod counts and shapes. See the [Cabling Generator README](../README.md) and fabric docs for generating or customizing descriptors from your deployment.

### Rank Files and Rank Binding Files

Fabric tests use **rank bindings** to map MPI ranks to (mesh_id, mesh_host_rank) and to point to the Mesh Graph Descriptor. You effectively **stamp out** one rank binding per process across all pods.

- **Rank bindings file** — YAML listing each rank’s `mesh_id`, `mesh_host_rank`, and optional `env_overrides`; plus top-level `mesh_graph_desc_path` pointing at your Mesh Graph Descriptor.
- **Rankfile** — OpenMPI rankfile that maps each rank to a (hostname, slot). Create one per deployment (do not commit; deployment-specific). **How to create one:** one line per MPI rank in order (rank 0, 1, 2, …), form `rank <N>=<hostname> slot=0`. Line count must equal the number of ranks in your rank bindings; hostnames and order must match your `--host` list. Example:

  ```
  rank 0=myhost-01 slot=0
  rank 1=myhost-02 slot=0
  rank 2=myhost-03 slot=0
  rank 3=myhost-04 slot=0
  ```

**Rank bindings** for 4 meshes × 4 hosts (16 ranks): see `tests/tt_metal/distributed/config/dual_32x4_quad_bh_galaxy_rank_bindings.yaml` for the full file. Excerpt:

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
  # ... one entry per rank (16 for 4 meshes × 4 hosts)
mesh_graph_desc_path: "tt_metal/fabric/mesh_graph_descriptors/dual_pod_32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
```

### Running Fabric Tests with tt-run

Use `tt-run` with a **rank-binding** file and MPI arguments that include your **rankfile** and **host list**. The test binary is `./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric`; pass the fabric test config via `--test_config`.

Example (use your own rankfile and host list; `--host` order must match the rankfile):

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/dual_32x4_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host <host0>,<host1>,... --map-by rankfile:file=<your_rankfile> --mca btl self,tcp --mca btl_tcp_if_include <nic> --bind-to none --tag-output" \
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_multi_mesh.yaml
```

Use the correct NIC for your environment (`btl_tcp_if_include`; e.g. `ens5f0np0` or `cnx1`).

For more on rankfiles and generating cluster descriptors from multiple hosts, see [README_generate_cluster_descriptors.md](../../scripts/scaleout/README_generate_cluster_descriptors.md).

### Validating the setup (without a SuperPod)

You can sanity-check the mainlined artifacts before running on real hardware. All commands below assume you run from the **repository root** (paths are relative to repo root).

1. **Build the test binary** (from repo root): `./build_metal.sh --build-tests` builds all tests; the fabric test binary is at `./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric`.

2. **Validate the fabric test config YAML** (parses and has expected structure):
   ```bash
   python3 -c "
   import yaml
   with open('tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_multi_mesh.yaml') as f:
       d = yaml.safe_load(f)
   assert 'Tests' in d and len(d['Tests']) >= 1
   print('YAML OK:', len(d['Tests']), 'test(s)')
   "
   ```

3. **Validate the mesh graph descriptor** (parses and loads):
   ```bash
   ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MeshGraphDescriptorTests.ParsesDualPod32x4QuadBhGalaxyTorusXY"
   ```
   Run from repo root so the descriptor path resolves.

4. **Validate the rank bindings YAML** (optional):
   ```bash
   python3 -c "
   import yaml
   with open('tests/tt_metal/distributed/config/dual_32x4_quad_bh_galaxy_rank_bindings.yaml') as f:
       d = yaml.safe_load(f)
   assert 'rank_bindings' in d and 'mesh_graph_desc_path' in d
   print('Rank bindings OK:', len(d['rank_bindings']), 'ranks')
   "
   ```

**Full end-to-end** validation requires either a 16-host SuperPod (run the `tt-run` example above) or a mock cluster: create a 16-rank mock cluster descriptor mapping and use `tt-run --mock-cluster-rank-binding <mapping> --rank-binding ...` with `--mpi-args "--allow-run-as-root"` to run 16 processes on one machine (see [custom_mock_cluster_descriptors/README.md](../../tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/README.md)).

## Directed Link Retrains (Wormhole Only)

The validation tool supports directed link recovery on Wormhole based clusters. If a missing connection is detected, the tool will attempt to recover the link by retraining the affected links.

Validation will then be performed again, with the tool trying to recover any leftover missing connections. This process will be repeated up to 5 times, after which the tool will fail if links are still missing.

This functionality is fairly limited in scope and can only be used to recover flaky links. Broken physical links or cables will not be recovered by this mechanism.

The user will be notified of all retrains attempted and the final state of the cluster post recovery.
