

![](images/image000.png)

# TT-Fabric Architecture Specification

Version 1.0

Revision 2.0

Authors: TT-Metalium Scale-Out Team

For questions and comments please use the [TT-Metalium Scale-Out Discord Server](https://discord.com/channels/863154240319258674/1321621251269328956)


## Table of Contests
[1. Overview](#overview)

[1.1. Operational Structure](#structure)

[1.1.1. Data Plane](#dataplane)

[1.1.2. Control Plane](#controlplane)

[1.1.2.1. Fabric Node Status Queues](#statusqueue)

[1.2. Some Additional Notes](#notes)

[2. TT-Fabric Network Layers](#fabric_layers)

[2.1. Layers 1, 2](#layer_12)

[2.2. TT-routing (Layer 3)](#layer_3)

[2.2.1. Routing Tables](#routing_tables)

[2.2.1.1. L0 Routing (Intra-Mesh)](#intramesh)

[2.2.1.1.1. L0 Routing Table Setup](#intramesh_setup)

[2.2.1.2. L1 Routing (Inter-Mesh)](#intermesh)

[2.2.1.2.1. L1 Routing Table Setup](#intermesh_setup)

[2.2.2. Routing Planes](#routing_planes)

[2.2.3. Automatic Traffic Rerouting](#rerouting)

[2.3. TT-transport (Layer 4)](#layer_4)

[2.3.1. Dateline Virtual Channel](#dvc)

[2.3.2. Control Virtual Channel](#cvc)

[2.4. TT-session (Layer 5)](#layer_5)

[3. Fabric Router](#router)

[3.1. Buffers and Virtual Channels](#rb_per_vc)

[3.1.1. 1D Line Virtual Channel](#1dlvc)

[3.1.2. 2D Mesh Virtual Channel](#2dmvc)

[4. Read/Write API Specification](#rw_api)

[4.1. Asynchronous Write](#async_wr)

[4.2. Asynchronous Multicast Write](#async_mcast_wr)

[4.3. Asynchronous Write Barrier](#async_wr_barrier)

[4.4. Asynchronous Read](#async_rd)

[4.5. Asynchronous Read Barrier](#async_rd_barrier)

[4.6. Asynchronous Atomic Increment](#async_atomic_inc)

[4.7. Asynchronous Atomic Read and Increment](#async_atomic_rd_inc)

[5. Sockets over TT-Fabric](#socket_api)

[6. Reliability](#reliability)

[6.1. Automatic Traffic Rerouting](#rerouting)

[7. Deadlock Avoidance and Mitigation](#deadlocks)

[7.1. Dimension Ordered Routing](#dim_order_routing)

[7.2. Edge Disjoint Routing](#disjoint_routing)

[7.3. Fabric Virtual Channels](#fab_vcs)

[7.4. Time To Live (TTL)](#ttl)

[7.5. Timeout](#timeout)

[7.6. Limitations](#limits)

[8. TT-Fabric Model](#model)

[8.1. Serialization and Visualization](#visualization)

[8.2. Data Plane Simulator](#simulator)

[8.3. Modelling External Disruptors and Buffer Limits](#disruptors)

[9. System Specification](#system_spec)

[9.1. System Components](#system_components)

[9.2. TG](#tg)

[9.3. Multi-Host TGG](#tgg)

[9.4. Quanta 2 Galaxy System](#ubb_galaxy)

[10. Resource Allocation](#resource_alloc)

[10.1. Available Dispatch Cores](#available_cores)

[10.2. Fast Dispatch and Fabric Kernel Resouces](#fd_and_fabric)

# 1 Overview <a id="overview"></a>

![](images/image001.png)

TT-fabric is a revolutionary approach to AI infrastructure, built around the Tenstorrent Galaxy as its core component.

* Tenstorrent Galaxy:
  + A comprehensive AI building block in a 6U form factor
  + Features (Blackhole):
    - Massive AI compute: 24 PFLOPs
    - Large high-speed memory: 1 TB capacity at 16 TB/s BW
    - Switch capabilities with industry-leading I/O: 11.5 TB/s
* TT-fabric leverages Galaxy to create a unified networking fabric that:
  + Integrates scale-up (building large servers) and scale-out (connecting large servers into a network) architectures into a single unified architecture
  + Provides a complete data-plane solution for deep learning AI training, covering:
    - Core Compute: Forward propagation and Backward propagation
    - Aggregation: Gradient reduction, Weight broadcasting, Sparse Embedding Gradient Sorting, Top-K expert sorting
    - Parameter server, optimizer compute and storage
* Key advantages:
  + Flexibility to build both powerful individual servers and expansive server networks
  + Entire back-end (data-plane) network can be constructed using only Galaxy boxes in a 4x8 configuration
  + Offers a scalable and unified approach to AI infrastructure design
  + 10x TCO advantages for AI data training data center design

The purpose of this document is to provide detailed architecture specification of TT-Fabric. TT-Fabric is Tenstorrent’s implementation of a mesh interconnect that enables AI accelerator devices to communicate with each other. TT-Fabric provides infrastructure to scale-up and scale-out AI accelerators into large meshes.

At its core TT-Fabric can be envisioned as a networking stack with multiple layers. The following diagram shows TT-Fabric stack side by side with other network stacks for reference.

![](images/image002.png)

## 1.1 Operational Structure <a id="structure"></a>

TT-Fabric is broadly structured to function in two operational planes. Data Plane and Control Plane as described in the following sections.

### 1.1.1 Data Plane <a id="dataplane"></a>

Data plane refers to the data communication network over which devices exchange information. In other words, Data Plane implements the layered TT-Fabric network stack shown in the figure above.

Data Plane uses the NOC for intra device routing. For inter device routing, data plane uses point-to-point ethernet link between the devices. Data transfer started by a fabric client may take several NOC and ethernet hops before it gets to its intended destination.

Any device-to-device communication in TT-Fabric requires a properly functioning Data Plane.

### 1.1.2 Control Plane <a id="controlplane"></a>

Control plane refers to a secondary interconnect that is used to launch and configure TT-Fabric software and firmware onto all the worker components. In other words, Control Plane is used to set up and launch the Data Plane.

The control plane is persistent, always functional and does not rely on a properly functioning Data Plane. If control plane access to a device is compromised, TT-Fabric cannot operate reliably and requires reboot of control plane to restore compromised connectivity.

On Wormhole (Quanta) and Blackhole devices, PCIe provides the control plane.

After launching TT-Fabric on devices, the control plane monitors the system. Any anomalies encountered by Fabric components are reported to control plane for user visibility and action.

Some examples of information generated by TT-Fabric are:

* Ethernet link down
* Observed data rates by different fabric routers
* Dropped packets
* Routing errors

Some examples of user or control plane actions based on TT-Fabric status are:

* Pause workloads on TT-Fabric errors
* Reconfigure routing tables when ethernet links are down
* Remove rows or columns of mesh devices if links or chips are not functional
* Relaunch workloads from a checkpoint after correcting TT-Fabric errors

#### 1.1.2.1 Fabric Node Status Queues<a id="statusqueue"></a>

The control plane sets up message queues in the DRAM of each fabric node. Fabric node workers push status and error messages as mentioned earlier into these queues for user visibility. The control plane retrieves messages from the queues and takes appropriate action.

All the messages received by control plane are also saved to disk to keep a log of fabric activity in case of malfunction. For a system comprising of thousands of chips, these logs help diagnose issues and root cause failures. Given the scale of the system, TT-Fabric will need some tools that scan the logs to identify specific physical areas of the fabric that might have encountered a problem.

## 1.2 Some Additional Notes <a id="notes"></a>

AI accelerators will be referred to as Devices for the remainder of this document. A device can be a Wormhole, Blackhole etc.

Our first TT-Fabric implementation is specific to Tenstorrent hardware and is intended to be deployable on current and future generations of Tenstorrent devices. TT-fabric is a software/firmware implementation that utilizes NOC and Ethernet capabilities of devices to build a mesh wide communication network.

TT-Fabric will be deployed on a mesh of homogeneous devices. This means that a cluster will not contain a mix of different device architectures. A Wormhole cluster will contain only wormhole devices.

TT-Fabric may support third party hardware in future but is currently outside the scope of this document.

Tenstorrent devices are connected via point-to-point ethernet links. While it is technically possible to have a mesh of non-homogeneous devices, we are deferring dealing with complications of connecting devices with disparate capabilities in a homogeneous fabric interconnect to a later time.

# 2 TT-Fabric Network Layers <a id="fabric_layers"></a>

This section describes the operation of different layers in TT-Fabric network stack.

## 2.1 Layers 1, 2 <a id="layer_12"></a>

Tenstorrent devices implement Layers 1 and 2 of the network stack in hardware.

At Layer 2, Tenstorrent ethernet controllers can operate in compliance mode where the payload is encapsulated in standard Ethernet frames as shown in the following figure.

![](images/image003.png)

In this mode Ethernet MAC drops packets that are received with errors. It is left to higher network layers such as layer 4 to implement data retransmission.

Alternatively, ethernet controllers can operate in TT-link mode. In this mode, ethernet controllers implement a Logical Link Control (LLC) sublayer of the standard OSI Layer 2. LLC is implemented in TT-link mode by adding a custom 16-Byte Tenstorrent TT-link header to user payload. This header implements two important LLC features.

* Payload destination address in Receiver’s TT-routing layer
* Go-Back-N ARQ protocol

The following diagram shows how TT-link mode encapsulates user payload into ethernet frame.

![](images/image004.png)

In TT-link mode, ethernet controller forwards the received data to TT-routing layer at the address specified in TT-link header as opposed to compliance mode, where data is left in ethernet controller’s receive buffer and requires Layer 3 software to read it out.

tx seq and rx seq fields in TT-link header are used by sender and receiver to communicate sequence numbers for transmitted and received ethernet frames. In case the sequence numbers diverge from expected values, because of MAC errors, Go-Back-N ARQ restarts transmission from the last successfully received ethernet frame.

TT-link mode ensures that TT-routing and upper layers never see data loss due to ethernet frame CRC errors. Any TT-routing payload that is accepted by TT-link layer is guaranteed to be delivered to TT-routing layer on the receiver side.

TT-link layer headers are automatically inserted on the send side and stripped when data is delivered to TT-routing on receiver.

TT-Fabric requires ethernet controllers to operate in TT-link mode.

## 2.2 TT-routing (Layer 3) <a id="layer_3"></a>

TT-Routing layer is the first software layer in TT-Fabric stack and is responsible for routing packets in the network. A packet requiring an inter-chip hop is forwarded to Layer 2 where it gets encapsulated by hardware into ethernet frames and forwarded over ethernet link.
A packet requiring an intra-chip hop is forded over NOC.

A device in the fabric network is addressed by MeshId and DeviceId fields in the packet header. Fabric router uses these fields to make routing decisions.

A packet's route can be determined by looking up routing tables in a router, or the packet can be source routed meaning the route is embedded into the packet header by a packet source. Within a mesh, packets are always source routed. At every hop, routers examine the packet header for source route fields and determine the necessary actions to process the packet. Source route in packet header is set by a worker that is injecting the packet into fabric. If the fabric contains a single mesh, then only the workers that connect to fabric to send traffic are the ones that populate source route in the packet header. The source route, once set by sending worker, never needs to be updated since a worker is always able to fully determine the route to any other device in the mesh.

When there are multiple meshes in fabric, then a packet's source route is set by the sending worker as well as at every other mesh entry point. For an inter-mesh packet the sending worker sets the packet source route to its local mesh's exit node. When the packet crosses the mesh boundary, the entry node fabric router in the next mesh calculates a new source route and updates the packet header. The new source route determines how the packet will traverse the new mesh. If the packet is destined for current mesh, the source route is the path to destination device in the current mesh. If the packet is destined for yet another mesh, the source route generated by the mesh entry node router is to an exit node of the current mesh. This exit node is along the path to the destination mesh specified in the packet header.

TT-Routing also implements automatic rerouting incase of an ethernet link failure. Rerouting is implemented within the TT-routing layer and is invisible to the upper layers of the stack.
TT-Fabric control plane monitors ethernet links and in case of an outage, it notifies the affected fabric routers to reroute their traffic over a different available router, that is connected in the same direction as the affected routers.
User workload on data plane encounters a temporary pause in activity. If TT-Fabric control plane and routers are able to setup a reroute, workload resumes operation at a lower data rate. In case alternate route is not possible, tt-fabric control plane can take other mitigating actions such as notifying the user application.


### 2.2.1 Routing Tables <a id="routing_tables"></a>

TT-Fabric allows a maximum scale out to 250,000 devices. Devices are connected in groups of meshes and we support upto 1024 meshes of 256 devices each.

A mesh is a fully and uniformaly connected grid of chips. Uniform connectivity means that all devices in the mesh have the same number of ethernet connections to all of their neighbors.

Inter-Mesh connectivity is provided through subset of devices called exit nodes. A mesh may be connected to multiple neighboring meshes in which case there are exit nodes providing routes to different neighboring meshes.

To support this topology, we need two levels of routing:

* L0 or Intra-Mesh routing
* L1 or Inter-Mesh routing

Fabric routers have fully instantiated routing tables indexed with Device Id or Mesh Id. This means that a router can look up a route to any Device or Mesh from its routing tables.

An intra-mesh routing table entry is a source route to a mesh destination. This destination can be the packet's final destination or it can be an exit node in case of in inter-mesh packet.

An inter-mesh routing table entry is an exit node in the current mesh.

#### 2.2.1.1 L0 Routing (Intra-Mesh) <a id="intramesh"></a>

When a packet’s destination is within the local mesh, the next hop is looked up from L0 routing table and packet is forwarded over specified ethernet port.

Every hop moves the packet towards the destination device in local mesh.

L0 routing table example:

| **DeviceId** | **Local Port** |
| --- | --- |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| ... |  |
| 1023 | 8 |

##### 2.2.1.1.1 L0 Routing Table Setup <a id="intramesh_setup"></a>

The following figure shows a 4 mesh cluster. Each mesh has 9 devices. Each device shows which of its ethernet ports are connected to its neighbor.

![](images/image009.png)

The L2 routing table for devices on Mesh 0 is shown below. Each row numbered 0 to 8 in Source Device group is the L0 routing table for the respective device on Mesh 0. This routing table is set up with dimension ordered routing. Packets go X dimension first then Y.

![](images/image010.png)

The following table shows how a packet sent by source Device 0 gets routed to destination Device 8 in Mesh 0.

<table>
  <tr>
    <th colspan="3">Route from Device 0 to Device 8</th>
  </tr>
  <tr>
    <th>Hop</th>
    <th>Sender Device-Port</th>
    <th>Receiver Device-Port</th>
  </tr>
  <tr>
    <td align="center">1</td>
    <td align="center">0-P2</td>
    <td align="center">1-P4</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">1-P2</td>
    <td align="center">2-P4</td>
  </tr>
  <tr>
    <td align="center">3</td>
    <td align="center">2-P1</td>
    <td align="center">5-P3</td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td align="center">5-P1</td>
    <td align="center">8-P3</td>
  </tr>
</table>

#### 2.2.1.2 L1 Routing (Inter-Mesh) <a id="intermesh"></a>

When a packet is not addressed to local mesh, the next hop is looked up from L1 routing table and packet is forwarded over specified ethernet port.

Every hop moves the packet towards an exit node on local mesh.

L1 routing table example:

| **MeshId** | **Local Port** |
| --- | --- |
| 0 | 4 |
| 1 | 4 |
| 2 | 9 |
| ... |  |
| 1023 | 9 |

##### 2.2.1.2.1 L1 Routing Table Setup <a id="intermesh_setup"></a>

For the same 4 Mesh cluster mentioned in previous section, L1 routing table for devices on each of the 4 meshes is shown in the following figure. Each row labeled 0 to 8 is L1 routing table for respective device in respective source mesh. Colored boxes identify the Exit nodes on each Mesh. To identify Mesh x Device x (Ethernet) Port x we can use a notation of MxDxPx.

From the 4-mesh topology presented earlier, we can see that M0D5 is an exit node and M0D5P2 is M0 to M1 ethernet link. Similarly, M1D3 is an exit node and M1D3P4 is M1 to M0 ethernet link. In the table below, these two ports are highlighted green to identify the route between M0 and M1. Any packet in M0 that has to be routed to M1 will be funneled towards M0D5. Similarly, any packet in M1 that has to be routed to M0 will be funneled towards M1D3.

There is no direct route between M0, M3 or M1, M2. Traffic from M0 to M3 will traverse either M1 or M2 on its way to M3. Which intermediate meshes are traversed depends on how we program L2 routing tables in the fabric routers. Fabric routers do not have any built-in bias towards picking paths.

![](images/image011.png)

### 2.2.2 Routing Planes <a id="routing_planes"></a>

Tenstorrent devices support multiple ethernet links for routing in any direction. On WH, for each of the East, West, North, South directions, we have 4 ethernet links. BH, similarly has 2, and potentially another 2 for Up or Down direction for 3D meshes. Multiple ethernet links means we have more than one TT-Fabric route to a destination. To support multiple routes while keeping the traffic patterns deterministic, we are introducing the concept of Routing Planes. The multiple ethernet ports in each direction are one to one mapped, such that the first ethernet port on any edge only routes over first ethernet port of the other edges. Similarly, the second ethernet port only routes over the second ethernet port of the other edges. This keeps the traffic contained within its routing plane and there is no cross talk between the routing planes. TT-Fabric clients can specify the routing plane when issuing transactions. In general, the number of routing planes is the same as the number of available parallel ethernet ports per direction on a device. For WH it is four while for BH it is two.

The following diagram shows how the four ethernet ports per direction on WH mesh form four separate routing planes. The diagram shows four WH devices.

![](images/image012.png)

## 2.3 TT-transport (Layer 4) <a id="layer_4"></a>

TT-Transport implements virtual channels that are used to carry packets in the routing netowrk. A virtual channel is composed of multiple buffers where each buffer holds packets packets from a dedicated source. Virtual channel buffer size depends upon amount of available SRAM space on fabric router. Traffic from multiple sources on a virtual channel serialized. All traffic on the same virtual channel is guaranteed to be ordered. TT-Fabric currently supports 1 user visible virtual channel per router.

### 2.3.1 Dateline Virtual Channel <a id="dvc"></a>

To avoid cyclic dependency deadlocks on Ring/Torus topologies, TT-Fabric has an internal dateline virtual channel. TT-Session APIs can only inject traffic into the data virtual channels. When packets cross the dateline, tt-fabric routers automatically switch the packet flow to the dateline virtual channel to avoid deadlock.

### 2.3.2 Control Virtual Channel <a id="cvc"></a>

TT-Fabric uses a dedicated virtual channel to route all control messages in the system. This prevents control traffic from interfering and competing with data traffic.

Control messages are small, fixed size packets and can be routed more efficiently by using a dedicated control virtual channel that is smaller than data virtual channel.

## 2.4 TT-session (Layer 5) <a id="layer_5"></a>

TT-Session layer provides the APIs for higher level user programs/kernels to connect and send data over TT-Fabric. Any data sent over fabric is encapsulated in independent, asynchronous packets. A packet specifies the complete destination memory address where the data must be written.

TT-session does not natively support synchronous transfer of data where data sender and data receiver need some kind of flow control.
Synchronous data transfer however is possible via sockets over fabric. We have implemented send/receive operations that use tt-fabric to exchange data as well as flow control messages. Sockets over tt-fabric are described later in this document.

# 3 Fabric Router <a id="router"></a>

Fabric Router implements all runtime TT-Fabric network stack functions. A fabric router can be mapped onto a single RISCV processor or the overall router functionality can be broken up into separate functions and maped onto multiple RISCV processors.
On Wromhole, Ethernet core RISCv performs all of the fabric netowrk stack functions. Blackhole Ethernet core supports 2 RISCv processors and fabric router functions are mapped to both processors.
The core function of a fabric router is to receive packets from ethernet and local device's worker and make a forwarding decision. A conbination of routing table lookup and source routing is used to make a forwarding decision. If a packet is to be consumed locally then the NOC command section of the packet header specifies how to process the packet payload locally.

Fabric routers can be built for different kinds of TT-Fabric topologies. A topology specifies how the TT-Fabric logical routing network should be created on a set of physically connected devices. TT-Fabric topology is specific to application requirements. For example the devcies may be physically connected in a 2D ethernet grid however the user application is such that the devices only communicate with other devices in the same row or column. In this case, Fabric routers only need to route traffic on one axis and packet flow does not switch from row to column or vice versa. We call this a 1D Fabric. TT-Fabric can also be launched in a 2D routing mode where turns between rows and columns are supported. In addition to these simple line and mesh topologies, TT-Fabric supports rings and 2D toruses. A ring is where the two endpoints of every line (in a 1D Fabric) are also connected. A 2D torus connects every row and column endpoints of a 2D mesh.

TT-Fabric topology can be an exact match of the physical connection topology, or it can be a functionally downshifted topology where all available ethernet links are not utilized.

The following diagrams show the supported 1D and 2D TT-Fabric topologies.

**8x4 Galaxy with TT-Fabric in 1D Line Topology**
```
                █  : Fabric Router
                ┄┄ : 1-D Row Line Fabric. There are 8 independent line fabric instances.
                ┇  : 1-D Column Line Fabric. There are 4 independent line fabric instances.
                ╔═════════════╦═════════════╦═════════════╦═════════════╗
                ║0            ║1            ║2            ║3            ║
                ║             ║             ║             ║             ║
                ║            █╬█┄┄┄┄┄┄┄┄┄┄┄█╬█┄┄┄┄┄┄┄┄┄┄┄█╬█            ║
                ║             ║             ║             ║             ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║4     █      ║5     █      ║6     █      ║7     █      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      ┇     █╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█     ┇      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║8     █      ║9     █      ║10    █      ║11    █      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      ┇     █╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█     ┇      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║12    █      ║13    █      ║14    █      ║15    █      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      ┇     █╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█     ┇      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║16    █      ║17    █      ║18    █      ║19    █      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      ┇     █╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█     ┇      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║20    █      ║21    █      ║22    █      ║23    █      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      ┇     █╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█     ┇      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║24    █      ║25    █      ║26    █      ║27    █      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      ┇     █╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█     ┇      ║
                ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║28    █      ║29    █      ║30    █      ║31    █      ║
                ║             ║             ║             ║             ║
                ║            █╬█┄┄┄┄┄┄┄┄┄┄┄█╬█┄┄┄┄┄┄┄┄┄┄┄█╬█            ║
                ║             ║             ║             ║             ║
                ║             ║             ║             ║             ║
                ╚═════════════╩═════════════╩═════════════╩═════════════╝
```

**8x4 Galaxy with TT-Fabric in 1D Ring Topology**
```
                █  : Fabric Router
                ┄┄ : 1-D Row Ring Fabric. There are 8 independent ring instances.
                ┇  : 1-D Column Ring Fabric. There are 4 independent ring instances.
            ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐
            ┊         ┌●            ┌○            ┌◎            ┌┅┅┅┅┅┅┅┅┅┅┅┅┐
            ┊         ┇             ┇             ┇             ┇         ┊  ┇
            ┊  ╔══════╬══════╦══════╬══════╦══════╬══════╦══════╬══════╗  ┊  ┇
            ┊  ║0     █      ║1     █      ║2     █      ║3     █      ║  ┊  ┇
            ┊  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ┊  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     ┇
               ║4     █      ║5     █      ║6     █      ║7     █      ║     ┇
            ★  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ★  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     ┇
               ║8     █      ║9     █      ║10    █      ║11    █      ║     ┇
            ☆  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ☆  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     ┇
               ║12    █      ║13    █      ║14    █      ║15    █      ║     ┇
            ✦  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ✦  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     ┇
               ║16    █      ║17    █      ║18    █      ║19    █      ║     ┇
            ✧  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ✧  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     ┇
               ║20    █      ║21    █      ║22    █      ║23    █      ║     ┇
            ✩  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ✩  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     ┇
               ║24    █      ║25    █      ║26    █      ║27    █      ║     ┇
            ✪  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ✪  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     ┇
               ║28    █      ║29    █      ║30    █      ║31    █      ║     ┇
            ✬  ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║  ✬  ┇
            └┄┄╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬█┄┄┄┄┄┇┄┄┄┄┄█╬┄┄┘  ┇
               ║      ┇      ║      ┇      ║      ┇      ║      ┇      ║     ┇
               ║      █      ║      █      ║      █      ║      █      ║     ┇
               ╚══════╬══════╩══════╬══════╩══════╬══════╩══════╬══════╝     ┇
                      ┇             ┇             ┇             ┇            ┇
                      └●            └○            └◎            └┅┅┅┅┅┅┅┅┅┅┅┅┘

```

**8x4 Galaxy with TT-Fabric in 2D Mesh Topology**
```
                █  : Fabric Router
                ── : 2-D Mesh Fabric. One fabric network connects all 32 devices.
                ╔═════════════╦═════════════╦═════════════╦═════════════╗
                ║0            ║1            ║2            ║3            ║
                ║             ║             ║             ║             ║
                ║      ┌─────█╬█─────┬─────█╬█─────┬─────█╬█─────┐      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║4     █      ║5     █      ║6     █      ║7     █      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      ├─────█╬█─────┼─────█╬█─────┼─────█╬█─────┤      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║8     █      ║9     █      ║10    █      ║11    █      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      ├─────█╬█─────┼─────█╬█─────┼─────█╬█─────┤      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║12    █      ║13    █      ║14    █      ║15    █      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      ├─────█╬█─────┼─────█╬█─────┼─────█╬█─────┤      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║16    █      ║17    █      ║18    █      ║19    █      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      ├─────█╬█─────┼─────█╬█─────┼─────█╬█─────┤      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║20    █      ║21    █      ║22    █      ║23    █      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      ├─────█╬█─────┼─────█╬█─────┼─────█╬█─────┤      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║24    █      ║25    █      ║26    █      ║27    █      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      ├─────█╬█─────┼─────█╬█─────┼─────█╬█─────┤      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      █      ║      █      ║      █      ║      █      ║
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣
                ║28    █      ║29    █      ║30    █      ║31    █      ║
                ║      │      ║      │      ║      │      ║      │      ║
                ║      └─────█╬█─────┴─────█╬█─────┴─────█╬█─────┘      ║
                ║             ║             ║             ║             ║
                ║             ║             ║             ║             ║
                ╚═════════════╩═════════════╩═════════════╩═════════════╝
```

**8x4 Galaxy with TT-Fabric in 2D Torus Topology**
```
                █  : Fabric Router
                ── : 2-D Torus Fabric. One fabric network connects all 32 devices.
                     All rows and columns are rings.
             ┌─────────────────────────────────────────────────────────────┐
             │         ┌─●           ┌─○           ┌─◎           ┌────────────┐
             │         │             │             │             │         │  │
             │  ╔══════╬══════╦══════╬══════╦══════╬══════╦══════╬══════╗  │  │
             │  ║0     █      ║1     █      ║2     █      ║3     █      ║  │  │
             │  ║      │      ║      │      ║      │      ║      │      ║  │  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     │
                ║4     █      ║5     █      ║6     █      ║7     █      ║     │
             ★  ║      │      ║      │      ║      │      ║      │      ║  ★  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     │
                ║8     █      ║9     █      ║10    █      ║11    █      ║     │
             ☆  ║      │      ║      │      ║      │      ║      │      ║  ☆  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     │
                ║12    █      ║13    █      ║14    █      ║15    █      ║     │
             ✦  ║      │      ║      │      ║      │      ║      │      ║  ✦  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     │
                ║16    █      ║17    █      ║18    █      ║19    █      ║     │
             ✧  ║      │      ║      │      ║      │      ║      │      ║  ✧  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     │
                ║20    █      ║21    █      ║22    █      ║23    █      ║     │
             ✩  ║      │      ║      │      ║      │      ║      │      ║  ✩  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     │
                ║24    █      ║25    █      ║26    █      ║27    █      ║     │
             ✪  ║      │      ║      │      ║      │      ║      │      ║  ✪  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╠══════╬══════╬══════╬══════╬══════╬══════╬══════╬══════╣     │
                ║28    █      ║29    █      ║30    █      ║31    █      ║     │
             ✬  ║      │      ║      │      ║      │      ║      │      ║  ✬  │
             └──╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬█─────┼─────█╬──┘  │
                ║      │      ║      │      ║      │      ║      │      ║     │
                ║      █      ║      █      ║      █      ║      █      ║     │
                ╚══════╬══════╩══════╬══════╩══════╬══════╩══════╬══════╝     │
                       │             │             │             │            │
                       └─●           └─○           └─◎           └────────────┘
```

## 3.1 Buffers and Virtual Channels <a id="rb_per_vc"></a>

A virtual channel is compirsed of several buffers used to transport incoming and outgoing fabric packets. Virtual channels buffers are classified as Sender Channels and Receiver Channels. Sender channels buffer all packets exiting a device through the router's ethernet link. Receiver Channels buffer all packets entering a device through the router's ethernet link.

TT-Fabric supports one user visible bidirectional virtual channel per fabric router. The number of virtual channels as well as the number of sender and receiver channels in a virtual channel depens on fabric topology. The number of slots in sender/receiver channel depend on amount of memory available for buffering.

### 3.1.1 1D Line Virtual Channel <a id="1dlvc"></a>
Basic architecture of a 1D line virtual channel is shown in the following diagram. In a 1D line, the outgoing traffic on a router is either passthrough packets from the fabric node's neighbor or the traffic originating from node's worker. Hence the router requires 2 Sender Channels. Fabric router round-robbins through the 2 sender channels and forwards packets over ethernet. A virtual channel only requires 1 Receiver channel. Fabric router examines the headers of packets arriving in the Receiver Channel to make processing decisions. A packet in the receiver channel might be passing through the router's node, or destined for the router's node or both (in the case of a multi-cast packet)

```
                                        1D LINE VIRTUAL CHANNEL
                                        ┌─────────────────────────────────────┐
                                        │  SENDER CHANNELS (2)                │
                                        │  ┌───────────────────────────────┐  │
                                        │  │ Sender Channel 0 (8 slots)    │  │
                                        │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │
                                    ╔═════▶┤ │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7├═▶═════════════╗
                                    ║   │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │      ║
                                    ║   │  └───────────────────────────────┘  │      ║
                                    ║   │  ┌───────────────────────────────┐  │      ║
                                    ║   │  │ Sender Channel 1 (8 slots)    │  │      ║
            ┌───────────────────┐   ║   │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │      ║
            │  Network-On-Chip  ├═▶═╩═════▶┤ │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7├═▶═════════════╩══════▶┌──────────────────┐
            └───────────────┬───┘       │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │              │  E T H E R N E T │◀═══▶
                            ▲           │  └───────────────────────────────┘  │      ╔═════◀═└──────────────────┘
                            ║           ├─────────────────────────────────────┤      ║
                            ║           │  RECEIVER CHANNEL (1)               │      ║
                            ║           │  ┌───────────────────────────────┐  │      ║
                            ║           │  │ Receiver Channel 0 (16 slots) │  │      ║
                            ║           │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │      ║
                            ╚═════════════◀┤ │ 0│ 1│ 2│ ┅│ ┅│ ┅│14│15├◀══════════════╝
                                        │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │
                                        │  └───────────────────────────────┘  │
                                        └─────────────────────────────────────┘

```

### 3.1.2 2D Mesh Virtual Channel <a id="2dmvc"></a>
Basic architecture of a 2D mesh virtual channel is shown in the following diagram. In a 2D mesh, the outgoing traffic on a router is either passthrough packets from three of the fabric node's neighbors or the traffic originating from node's worker. Hence the router requires 4 Sender Channels. Fabric router iterates over the channels and processes the packets similar to 1D topology.


```
                                        VIRTUAL CHANNEL
                                        ┌─────────────────────────────────────┐
                                        │  SENDER CHANNELS (4)                │
                                        │  ┌───────────────────────────────┐  │
                                        │  │ Sender Channel 0 (8 slots)    │  │
                                        │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │
                                    ╔═════▶┤ │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7├═▶══════════════╗
                                    ║   │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │       ║
                                    ║   │  └───────────────────────────────┘  │       ║
                                    ║   │  ┌───────────────────────────────┐  │       ║
                                    ║   │  │ Sender Channel 1 (8 slots)    │  │       ║
                                    ║   │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │       ║
                                    ╠═════▶┤ │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7├═▶══════════════╣
                                    ║   │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │       ║
            ┌───────────────────┐   ║   │  └───────────────────────────────┘  │       ╠══════▶┌──────────────────┐
            │  Network-On-Chip  ├═▶═╣   │  ┌───────────────────────────────┐  │       ║       │  E T H E R N E T │◀═══▶
            └───────────────┬───┘   ║   │  │ Sender Channel 2 (8 slots)    │  │       ║  ╔══◀═└──────────────────┘
                            ▲       ║   │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │       ║  ║
                            ║       ╠═════▶┤ │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7├═▶══════════════╣  ║
                            ║       ║   │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │       ║  ║
                            ║       ║   │  └───────────────────────────────┘  │       ║  ║
                            ║       ║   │  ┌───────────────────────────────┐  │       ║  ║
                            ║       ║   │  │ Sender Channel 3 (8 slots)    │  │       ║  ║
                            ║       ║   │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │       ║  ║
                            ║       ╚═════▶┤ │ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7├═▶══════════════╝  ║
                            ║           │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │          ║
                            ║           │  └───────────────────────────────┘  │          ║
                            ║           ├─────────────────────────────────────┤          ║
                            ║           │  RECEIVER CHANNEL (1)               │          ║
                            ║           │  ┌───────────────────────────────┐  │          ║
                            ║           │  │ Receiver Channel 0 (16 slots) │  │          ║
                            ║           │  │ ┌──┬──┬──┬──┬──┬──┬──┬──┐     │  │          ║
                            ╚═════════════◀┤ │ 0│ 1│ 2│ ┅│ ┅│ ┅│14│15├◀══════════════════╝
                                        │  │ └──┴──┴──┴──┴──┴──┴──┴──┘     │  │
                                        │  └───────────────────────────────┘  │
                                        └─────────────────────────────────────┘
```


The following table lists the sender/receiver channel counts for different fabric topologies.

TODO: Add a table here.


# 4 Read/Write API Specification <a id="rw_api"></a>

TT-Fabric provides Remote Direct Memory Access (RDMA) support to Write and Read data from any device connected to the fabric.

The following sections describe supported APIs and their operation.

## 4.1 Asynchronous Write <a id="async_wr"></a>
```
Point to API Header
```
Asynchronous write is used to write data to a remote receiver. Sender does not need to wait for all the data to be written to receiver. Data is guaranteed to be written ordered.

## 4.2 Asynchronous Atomic Increment <a id="async_atomic_inc"></a>
```
Point to API Header
```
Asynchronous atomic increment is used to atomically increment an address in a remote device.

## 4.3 Asynchronous Write Atomic Increment <a id="async_wr"></a>
```
Point to API Header
```
Asynchronous write atomic increment is a fusion of the two individual APIs. It is more efficient as both operations are achieved with a single fabric packet.
This command writes data to remote device and atomically increments the specified address in remote device.

## 4.4 Asynchronous Multicast Write <a id="async_mcast_wr"></a>
```
Point to API Header
```
Multicast write is used to write to more than one remote receiver with the same data. Multicast starts at the origin device of the multicast grid. The extent of multicast is specified by the number of hops around the origin device. All devices within the specified depth are written with single async multicast write command.

## 4.4 Scatter Write (a NOC command)
```
Point to API Header
```

# 5 Sockets over TT-Fabric <a id="socket_api"></a>

We have implemented sockets as send and receive operatoins that use tt-fabric asynchronous write APIs to implement flowcontroled data transfer between a sender and receiver.
TODO: Add more information on send/receive operations.

# 6 Reliability <a id="reliability"></a>
## 6.1 Automatic Traffic Rerouting <a id="rerouting"></a>

TT-Fabric supports device meshes that can scale up to hundreds of thousands of devices. On such a large scale, the probability of some ethernet links going down is non-negligible. An interconnect that does not implement link redundancy and is not able to work around some broken ethernet links will face frequent work interruptions and require a lot of system management calls. We intend to build redundancy into TT-Fabric network stack such that if some ethernet links on a fabric node go down, fabric can automatically reroute blocked traffic over an available ethernet link. If there is at least 1 available link in the same direction as the broken link, TT-Fabric's redundancy implementation will be completely transparent to workloads running on the system. End user applications may notice a temporary pause and lower data rates but should not otherwise require any intervention. TT-Fabric will also notify Control Plane of the rerouting status so that appropriate action may be taken on the system management front to service the broken ethernet links. User workload will be able to reach its next checkpoint without network interruption at degraded data rates. At that point Control plane can update routing tables to take out broken links from routing network. System maintenance can also be performed to fix ethernet link issues before resuming user work.

To support redundancy, each fabric router has an Ethernet Fallback Channel (EFC) that is brought into service by other fabric routers in a node when their dedicated ethernet links become unreliable or completely dysfunctional. EFC can be shared by multiple routers when multiple ethernet links lose connection. EFC is not virtualized and operates at Layer 2. When routers push data into EFC, special layer 2 headers are appended to traffic so that impacted fabric router’s native FVCs can be reliably connected to their receiver channels on either side of the broken ethernet links.

Fabric routers exchange credits with each for traffic flow control rather than the EFC so that EFC is available for all reroute traffic. Any FVC back pressure is kept within the fabric router buffers and does not propagate to EFC layer buffers.

The following diagram shows how traffic gets rerouted when Eth A link becomes inactive. Broken arrows show all the rerouted FVC traffic in both directions.

![](images/image013.png)

# 7 Deadlock Avoidance and Mitigation <a id="deadlocks"></a>

Like any other network, TT-Fabric faces deadlock hazards. Circular dependencies, resource contention, buffer exhaustion are some of the conditions that can lead to deadlocks in the routing network. We are building features into TT-Fabric to minimize chances of hitting deadlocks. In the event of a deadlock, TT-Fabric should be able to detect it, try to mitigate the effects and notify the Control Plane.

The following sections describe the TT-Fabric features for deadlock avoidance and mitigation.

## 7.1 Dimension Ordered Routing <a id="dim_order_routing"></a>

Cyclic dependency deadlock happens when a set of nodes form a cyclic traffic pattern. Each node’s outgoing traffic is waiting on resources in the next hop to make progress. Since the traffic pattern is a cycle, all nodes end up waiting for the next hop’s resource and form a routing deadlock.

Dimension ordered routing prevents cyclic dependency deadlocks by routing traffic in a way that does not form traffic cycles. In the routing table examples presented earlier in the document, we use X then Y dimension ordered routing when building routing tables.

The following diagram illustrates the cyclic dependency deadlock.

![](images/image017.png)

4 Devices are creating a cyclic traffic pattern as follows:

* D1 sending to D4
* D2 sending to D3
* D4 sending to D1
* D3 sending to D2

Traffic originating from D1, D2, D3, D4 is shown by yellow, blue, gray and orange segments respectively. After making the first hop to the next neighbor, route segments turn red. That is because incoming packets cannot make the routing turn in the desired direction. Turn is not possible because the outgoing router’s buffer is exhausted serving locally generated traffic. Each device’s traffic gets stuck in the next hop waiting for the router buffer to become available. No packet is able to make progress, and the system is in a deadlock.

Using dimension ordered routing where packets travel in X direction before turning in Y direction avoids this cyclic dependency deadlock as shown in the following diagram.

![](images/image018.png)

Packets from D1 to D4 and D4 to D1 follow original routes.

Packet from D2 to D3 and D3 to D2 are routed in X direction first thus avoiding deadlock.

TT-Fabric is not limited to just one kind of routing bias. Since routing tables are fully instantiated, any reasonable routing scheme can be devised and mapped onto fabric router tables.

## 7.2 Edge Disjoint Routing <a id="disjoint_routing"></a>

Edge disjoint routing uses different entry/exit nodes on network edges for traffic that is incoming/outgoing from current network. In TT-Fabric, we can set up L1 routing tables such that cross traffic between meshes uses different exit nodes. Opposite traffic flows will go through different fabric routers which can reduce chances of resource contention.

## 7.3 Fabric Virtual Channels <a id="fab_vcs"></a>

As stated earlier, FVCs guarantee independent progress of traffic relative to other FVCs. Traffic that is expected to contend for network or endpoint resources can be routed via unique FVCs so that one traffic stream does not get stuck behind other traffic that is stalled due to a stalled endpoint.

## 7.4 Time To Live (TTL) <a id="ttl"></a>

TT-Fabric may encounter packets that keep on circling the network and are not terminating. This can occur if the routing tables are misconfigured or corrupted. Such traffic can keep on living in the network forever and keep burning network resources. To avoid such patterns of traffic, TT-Fabric packets have a TTL parameter. On traffic initiating end or router, TTL is initialized to a conservative value that covers longest hop count any packet could encounter in TT-Fabric. At every network hop, fabric router decrements TTL by 1. Under normal conditions, a packet will reach its destination before TTL becomes 0 (expires). If for any reason a router sees a fabric packet with TTL parameter of 0, the packet is marked as expired, dropped, and drained from fabric. TT-Fabric also notifies Control Plane of the event.

![](images/image019.png)

The diagram above shows a packet that gets stuck in a routing loop. D1 sends a packet that gets routed to D5, D6, D7, D11, D10, D9, D5, D6, ...

Without any way for TT-Fabric to detect this anomaly, this packet will stay in this routing loop forever. In the above 4x4 grid example, D1 to D16 is the longest route which is 6 ethernet hops. If D1 sets the TTL parameter of its packet to 10 that is big enough to cover all legitimate routes in this grid.

With a TTL of 10 the packet hops are shown in the table below. As the packet loops, Device 11 will eventually see the expired TTL parameter in packet header and drop the packet.

| **Device** | **TTL** |
| --- | --- |
| 1 | 10 |
| 5 | 9 |
| 6 | 8 |
| 7 | 7 |
| 11 | 6 |
| 10 | 5 |
| 9 | 4 |
| 5 | 3 |
| 6 | 2 |
| 7 | 1 |
| 11 | 0  Fabric Router at Device 11 drops the packet with expired TTL |

## 7.5 Timeout <a id="timeout"></a>

Timeouts are TT-Fabric's last line of defense against deadlocks. Timeout is a detection mechanism rather than a prevention mechanism. Schemes mentioned in previous sections are meant to prevent or minimize deadlocks. If a routing deadlock slips through, TT-Fabric will detect it through timeout. If a packet head is not able to make progress through a fabric router within the specified timeout, it may indicate some deadlock due to resource contention, erroneous routing, stalled endpoint etc. Fabric router encountering routing timeout will drop the packet and drain its data from the fabric buffers. The Fabric router will also notify Control Plane of the event.

## 7.6 Limitations <a id="limits"></a>

TT-Fabric does not support end-to-end transmissions in case of dropped packets. The current fallback is to notify Control Plane and rely on host software managed mitigation. TT-Fabric can notify data senders of the dropped packets by sending a negative acknowledgement. Data retransmission is left to the TT-Fabric user’s discretion.

# 8 TT-Fabric Model <a id="model"></a>

TT-Fabric Model is a software functional model of all components of the Fabric. The purpose of the Fabric Model is to fully simulate the ethernet traffic in the Fabric. It will provide a ground truth for the state of any configuration of the Fabric, to help with debug and rerouting decisions. The Fabric Model will include a new piece of software to emulate the physical data plane, but otherwise shares the software components of the TT-Control Plane and Fabric Router.

## 8.1 Serialization and Visualization <a id="visualization"></a>

TT-Fabric Model will serialize these components of the Fabric:

* Physical state of the cluster (i.e. output of syseng tool create-ethernet-map)
  + Visualizer to help debug
* Intermesh and intra-mesh routing tables
  + Visualizer to help debug
* Routing buffers and VCs
  + Visualizer to help debug
* Packet traffic across data plane
  + We should be able to download traffic serialization from software simulator to run on hardware, and vice versa.

## 8.2 Data Plane Simulator <a id="simulator"></a>

The data plane simulator will model all paths ethernet packets may take across the hardware, except for NOC activity between a device and the Fabric. Key components:

* Device, APIs to assemble packet headers and send them to data plane
* Router/VC, APIs to model resource availability
* Data Plane, query serialized routing tables and physical state of machine provided by control plane, to move packets between devices
* Toggle in data plane to test redundancy links vs. preferred links
* Directed single threaded testing for buffer limits and rerouting
* Random testing with multi-threading. One thread per device to simulate requests to the Fabric and one thread for external disruptors.

## 8.3 Modelling External Disruptors and Buffer Limits <a id="disruptors"></a>

TT-Fabric Model will have hooks to simulate failed links, to trigger and verify rerouting in the control plane. It will also have SW APIs to simulate back-pressured buffers and VCs, to detect possible deadlock scenarios.

# 9 System Specification <a id="system_spec"></a>

To configure and launch TT-Fabric on a physical system we need to specify the hardware components of the system in a hierarchy that is modular and scalable. System specification should be in a format that can be parsed by the Control Plane. The following sections describe our current approach which is to use a yaml to describe the topology of a physical machine. The system specification file contains all the necessary information to enable a user to build their AI workload offline. Before launching the workload, Control Plane validates the physical connectivity of the system to ascertain that it conforms to the system specification yaml file.

## 9.1 System Components <a id="system_components"></a>

To be able to describe the topology of a physical system, we need to identify the basic building blocks that are used to build an AI machine. This section lists the building blocks for a physical Tenstorrent machine.

* **ChipSpec**: Single AI accelerator with ethernet capabilities. In a two-dimensional layout of chips, every chip can have one neighbor on each of its four sides. A chip connects to its neighbors through one or more unique ethernet ports. The four sides of a chip are identified as East, West, North and South.
* **Board**: Physical board that holds individual chips in a two-dimensional layout. Chips on a board have uniform number of ethernet connections between all neighbors.
* **Host**: An x86 host that provides Control Plane (PCIe) access to a Board.
* **Mesh**: A two-dimensional layout of fully connected boards.

Boards are fully connected when:

* + All the edge chips on one board are connected to the same number of edge chips on neighboring board.
  + The number of ethernet connections between two neighbor edge chips on two different boards is the same as number of ethernet connecters between neighbor chips on the same board.

If the boards are not fully connected, then one mesh can only have one such board.

A mesh is typically described as a Board and Host pair.

To support the current generation of Tenstorrent Wormhole Galaxy boards, a mesh is allowed to have a board without a host directly connected to it.

* **Graph**: A set of meshes that makes up the full system. Two neighboring meshes are not required to be fully connected. In addition, all the meshes are not required to be connected to each other. Traffic from a source mesh can traverse multiple other meshes before it reaches its destination mesh.

## 9.2 TG <a id="tg"></a>

![](images/image020.png)

<table>
  <tr>
    <th colspan="2">TG</th>
  </tr>
  <tr>
    <th>Chip</th>
    <td>
      Wormhole:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Ethernet Ports:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; N:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; E:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; S:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; W:4<br>
    </td>
  </tr>
  <tr>
    <th>Board</th>
    <td>
      Galaxy:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Wormhole<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 4x8 <br>
      N150Gateway:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Wormhole<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1 <br>
    </td>
  </tr>
  <tr>
    <th>
      Mesh<br>
      Notes: For a TG system, we only have CPU hosts connected to<br>
      N150 gateway cards that are connected to the Galaxy via<br>
      ethernet links. There is no direct PCI access to chips on the<br>
      Galaxy board.<br>
    </th>
    <td>
      0:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      1:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      2:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      3:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      4:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Galaxy<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[ ]]<br>
    </td>
  </tr>
  <tr>
    <th>Graph</th>
    <td>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {0, S0} <---> {4, N0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {0, S1} <---> {4, N4}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {1, S0} <---> {4, N8}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {1, S1} <---> {4, N12}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {2, S0} <---> {4, N16}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {2, S1} <---> {4, N20}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {3, S0} <---> {4, N24}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {3, S1} <---> {4, N28}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N0} <---> {0, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N4} <---> {0, S1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N8} <---> {1, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N12} <---> {1, S1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N16} <---> {2, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N20} <---> {2, S1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N24} <---> {3, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, N28} <---> {3, S1}<br>
    </td>
  </tr>

</table>


## 9.3 Multi-Host TGG <a id="tgg"></a>

![](images/image021.png)
<table>
  <tr>
    <th colspan="2">TG</th>
  </tr>
  <tr>
  <th>Chip</th>
    <td>
      Wormhole:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Ethernet Ports:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; N:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; E:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; S:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; W:4<br>
    </td>
  </tr>
  <tr>
    <th>Board</th>
    <td>
      Galaxy:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Wormhole<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 4x8 <br>
      N150Gateway:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Wormhole<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1 <br>
    </td>
  </tr>
  <tr>
    <th>
      Mesh<br>
      Notes: For a multi-host TGG system, we have two hosts that are <br>
      connected to N150 gateway cards. The Galaxy boards are connected <br>
      via long edge with external ethernet cables. There are only 2 <br>
      links per chip-to-chip connection, which is not uniform with the <br>
      chip-to-chip within the Galaxy board.
      <br>
    </th>
    <td>
      0:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      1:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      2:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      3:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost0&gt]]<br>
      4:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost1&gt]]<br>
      5:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost1&gt]]<br>
      6:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost1&gt]]<br>
      7:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: N150Gateway<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[&lthost1&gt]]<br>
      8:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Galaxy<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[ ]]<br>
      9:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Galaxy<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 1x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[ ]]<br>
    </td>
  </tr>
  <tr>
    <th>Graph</th>
    <td>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {0, S0} <---> {8, N0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {0, S1} <---> {8, N4}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {1, S0} <---> {8, N8}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {1, S1} <---> {8, N12}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {2, S0} <---> {8, N16}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {2, S1} <---> {8, N20}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {3, S0} <---> {8, N24}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {3, S1} <---> {8, N28}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N0} <---> {0, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N4} <---> {0, S1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N8} <---> {1, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N12} <---> {1, S1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N16} <---> {2, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N20} <---> {2, S1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N24} <---> {3, S0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {8, N28} <---> {3, S1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, S0} <---> {8, N0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {4, S1} <---> {8, N4}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {5, S0} <---> {8, N8}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {5, S1} <---> {8, N12}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {6, S0} <---> {8, N16}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {6, S1} <---> {8, N20}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {7, S0} <---> {8, N24}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {7, S1} <---> {8, N28}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S0} <---> {4, N0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S4} <---> {4, N1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S8} <---> {5, N0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S12} <---> {5, N1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S16} <---> {6, N0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S20} <---> {6, N1}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S24} <---> {7, N0}<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; {9, S28} <---> {7, N1}<br>
    </td>
  </tr>

</table>

## 9.4 Quanta 2 Galaxy System <a id="ubb_galaxy"></a>

<table>
  <tr>
    <th colspan="2">TG</th>
  </tr>
  <tr>
    <th>Chip</th>
    <td>
      Wormhole:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Ethernet Ports:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; N:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; E:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; S:4<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&bull; W:4<br>
    </td>
  </tr>
  <tr>
    <th>Board</th>
    <td>
      Galaxy:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Wormhole<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 4x8 <br>
    </td>
  </tr>
  <tr>
    <th>
      Mesh<br>
      Notes: For a Quanta Galaxy box, Galaxy to Galaxy connections <br>
      also have four edges, so we can represent two Galaxys as a 2x1 Mesh. <br>
      We will also have one CPU host per Galaxy box.
      <br>
    </th>
    <td>
      0:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Submodule: Galaxy<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Topology: 2x1<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&bull; Host Mapping: [[ ]]<br>
    </td>
  </tr>
  <tr>
    <th>Graph</th>
    <td>
    </td>
  </tr>

</table>

# 10 Resource Allocation <a id="resource_alloc"></a>
This seciton estimates the hardware resources required to implement different TT-Fabric workers.

## 10.1 Available Dispatch Cores <a id="available_cores"></a>

| Idle Eth Core | Worker (ROW) | Worker (COL) | Idle Eth Core |
| --- | --- | --- | --- |
| N300 L Chip | (1x8) 8 | (8x1) 8 | 10 |
| TG N150 Chip | (9x8) 72 | (9x8) 72 | 14 |
| TG Galaxy Chip | (2x8) 16 | (1x10) 10 | 0 |
| Quanta Galaxy Chip | (2x8) 16 | (1x10) 10 | 0 |
| Blackhole Chip |  |  |  |
| Galaxy Blackhole Chip |  |  |  |

## 10.2 Fast Dispatch and Fabric Kernel Resouces <a id="fd_and_fabric"></a>

![](images/image022.png)

* 10 idle eth cores on L chip, 10 idle eth cores on R chip
* Allows for 8x8 Tensix worker grid on both chips

![](images/image023.png)

* 10 idle eth cores on L chip, 10 idle eth cores on R chip
* Allows for 8x8 Tensix worker grid on both chips
* Lacking resources for a packetizer/socket startpoint on L chip

![](images/image024.png)

* 68 Tensix cores on N150 chip, 15 kernels on each of remote Galaxy chips
* Allows for 8x8 Tensix worker grid on Galaxy chips

![](images/image025.png)

* 68 Tensix cores on N150 chip, 10 kernels on each of remote Galaxy chips
* Allows for 10x7 Tensix worker grid on Galaxy chips
