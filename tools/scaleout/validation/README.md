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
  Defines fabric tests (e.g. RandomPairing, AllToAll) for Mesh and 2D Torus XY topologies.

- **Minimal 4-mesh (e.g. 4× SuperPod, each pod 32×4) Mesh Graph Descriptor:**  
  `tt_metal/fabric/mesh_graph_descriptors/dual_pod_16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto`  
  Describes four meshes (each 32×4, 4 hosts per mesh) with full 2D torus-style connectivity between meshes.

### Building Your Mesh Graph Descriptor

Your Mesh Graph Descriptor must match your physical SuperPod layout:

- **Number of meshes** — One graph instance per pod (or per logical mesh).
- **Shape of each mesh** — e.g. 32×4 (device dims) and host topology (e.g. 4×1) must match how many hosts and devices you have per pod.
- **Connectivity** — Inter-pod links (and optional torus wrap) must reflect how pods are wired (e.g. XY torus between pods).

Descriptor files live under `tt_metal/fabric/mesh_graph_descriptors/`. Use the mainlined `dual_pod_16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto` as a reference; adapt or add descriptors for different pod counts and shapes. See the [Cabling Generator README](../README.md) and fabric docs for generating or customizing descriptors from your deployment.

### Rank Files and Rank Binding Files

Fabric tests use **rank bindings** to map MPI ranks to (mesh_id, mesh_host_rank) and to point to the Mesh Graph Descriptor. You effectively **stamp out** one rank binding per process across all pods.

- **Rank bindings file** — YAML listing each rank’s `mesh_id`, `mesh_host_rank`, and optional `env_overrides`; plus top-level `mesh_graph_desc_path` pointing at your Mesh Graph Descriptor.
- **Rankfile** — OpenMPI rankfile that maps each rank to a (hostname, slot). You must have **one rankfile entry per rank**; the order must match the rank bindings (rank 0, 1, …).

Example pattern for 4 meshes with 4 hosts per mesh (16 ranks), following the mainlined dual-pod style:

**Rank bindings** (excerpt; see `tests/tt_metal/distributed/config/dual_16x8_quad_bh_galaxy_rank_bindings.yaml` for the full file):

```yaml
rank_bindings:
  - rank: 0
    mesh_id: 0
    mesh_host_rank: 0
  - rank: 1
    mesh_id: 0
    mesh_host_rank: 1
  # ... one entry per rank (e.g. 16 for 4 meshes × 4 hosts)
mesh_graph_desc_path: "tt_metal/fabric/mesh_graph_descriptors/dual_pod_16x8_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
```

**Rankfile** — One line per rank, listing the host for that rank. A reference 16-rank rankfile for 4 meshes × 4 hosts is provided in the repo root as `8_glx_rankfile` (hosts bh-glx-d03u02/d03u08 through bh-glx-d10u02/d10u08). For other deployments, use the same format; example pattern:

```
rank 0=bh-glx-c01u08 slot=0
rank 1=bh-glx-c01u02 slot=0
rank 2=bh-glx-c02u02 slot=0
rank 3=bh-glx-c02u08 slot=0
rank 4=bh-glx-c05u08 slot=0
rank 5=bh-glx-c05u02 slot=0
rank 6=bh-glx-c06u08 slot=0
rank 7=bh-glx-c06u02 slot=0
rank 8=bh-glx-c03u08 slot=0
rank 9=bh-glx-c03u02 slot=0
rank 10=bh-glx-c04u08 slot=0
rank 11=bh-glx-c04u02 slot=0
# Add more lines for additional ranks (e.g. 12–15 for 4 meshes × 4 hosts)
```

Ensure the rankfile host list and count match your deployment and the rank bindings (same number of ranks in both).

### Running Fabric Tests with tt-run

Use `tt-run` with a **rank-binding** file and MPI arguments that include your **rankfile** and **host list**. The test binary is `./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric`; pass the fabric test config via `--test_config`.

Example (for 16 ranks using the in-repo rankfile and rank bindings; adjust host list and NIC to your cluster):

```bash
tt-run \
  --rank-binding tests/tt_metal/distributed/config/dual_16x8_quad_bh_galaxy_rank_bindings.yaml \
  --mpi-args "--host bh-glx-d03u08,bh-glx-d03u02,bh-glx-d04u02,bh-glx-d04u08,bh-glx-d05u08,bh-glx-d05u02,bh-glx-d06u02,bh-glx-d06u08,bh-glx-d07u08,bh-glx-d07u02,bh-glx-d08u02,bh-glx-d08u08,bh-glx-d09u08,bh-glx-d09u02,bh-glx-d10u02,bh-glx-d10u08 --rankfile 8_glx_rankfile --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0 --bind-to none --tag-output" \
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_bh_glx_2d_torus_multi_mesh.yaml
```

- The repo includes a 16-rank rankfile at `8_glx_rankfile` (repo root) matching the dual_16x8 bindings; the `--host` list above matches that rankfile’s hostnames.
- For other deployments, use your own rankfile and ensure `--host` lists the same hosts in the same order (one host per rank).
- Use the correct NIC interface for your environment (`btl_tcp_if_include`; e.g. `ens5f0np0` or `cnx1`).

For more on rankfiles and generating cluster descriptors from multiple hosts, see [README_generate_cluster_descriptors.md](../../scripts/scaleout/README_generate_cluster_descriptors.md).

## Directed Link Retrains (Wormhole Only)

The validation tool supports directed link recovery on Wormhole based clusters. If a missing connection is detected, the tool will attempt to recover the link by retraining the affected links.

Validation will then be performed again, with the tool trying to recover any leftover missing connections. This process will be repeated up to 5 times, after which the tool will fail if links are still missing.

This functionality is fairly limited in scope and can only be used to recover flaky links. Broken physical links or cables will not be recovered by this mechanism.

The user will be notified of all retrains attempted and the final state of the cluster post recovery.
