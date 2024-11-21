# TT-Metalium for Beginners #
## 1. User Program Organization ##
TT-Metalium is a low-level programming model enabling kernel development on Tenstorrent or non-Tenstorrent hardware. It is for developers who customize models, write new models, or just run non-machine learning code. TT-Metalium is an open source, low-level AI hardware SDK with no black boxes, encrypted APIs, or hidden functions.

The user program organization is the main configuration program or host program. The host machine executes the program and defines collaboration between the host machine and the device. Kernels execute operations on Tensix cores that define computation, communication, and synchronization between the host machine and the device. User compilation is performed in four steps:
1. Main configuration program is compiled.
2. Host program code is executed to finish all kernel compilation.
3. Data movement kernels are compiled and compute kernels are pre-processed.
4. Compute kernels are compiled.

<a name=Fig1></a>![* *Figure 1. illustrates how the user program, host program, compiler, and API library function with components lower in the stack:*](tt-metal/docs/source/common/images/MfB-Fig1.png)
## 2. Tenstorrent Architecture ##
Tenstorrent is the future of AI architecture and parallel programming; it achieves high performance with AI model innovation while staying flexible and programmable. High performance enables innovation of future AI models and high-performance computing (HPC) applications without current architecture constraints. Tenstorrent architecture is designed to run the newest AI models efficiently, adapts to create and run new AI models and high-performance computing (HPC) applications. It is built for growth, scaling up to handle large AI workloads or can scale down to a few cores. The cost effective design uses simple and affordable components to keep costs low without sacrificing performance. It is purpose built for AI optimization, AI inference, and training from the ground up.
### 2.1 Mesh Architecture ###
All devices have both physical and logical mesh architecture. Physical mesh architecture refers to physical connections between components. Logical mesh architecture refers to how data flows between components:
- **Physical Tensix Core Mesh Architecture** - Physical 2D core mesh architecture has coordinates along the x and y axis to indicate the physical location of a component. Each core is assigned a unique mesh address. The physical core mesh architecture enables both Single Instruction Multiple Data (SIMD) and Multiple Instruction Multiple Data (MIMD) computation patterns. It has an adaptable communication topology for different computation patterns. Transformable memory organization allows for memory types like shared memory, distributed memory, or hybrid memory to be transformed.
- **Logical Tensix Core Mesh Architecture** - Logical 2D mesh architecture is used by the main configuration program. Only Tensix cores can be addressed, not DRAM cores. The main configuration program creates and dispatches data movement kernels. It allocates device memory for buffers and initializes buffers with source data.

<b name=Fig2></b>![* *Figure 2. illustrates the Logical-to-Physical 2D space mapping function:*](tt-metal/docs/source/common/images/MfB-Fig2.png)

Each Tensix Core has five single-threaded “baby” RISC-V processors. The core-to-thread mapping ratio is 1:1. Work for direct parallelization is broken across Tensix Cores with kernels dispatching directly. Kernels do not require context switching or complex scheduling and will run to completion without interruption. Performance analyses are simplified based on direct cycle counting of C/C++ kernels. Debugging is simplified via GDB step-through, breakpoints, and printf.
### 2.2 Ethernet Core Architecture ###
Ethernet cores are responsible for data movement between devices. There are two NoCs in an Ethernet core, each is a RISC-V that is programmable across multiple cores. All RISC-Vs are programmable in Ethernet Cores.

<c name=Fig3></c>![* *Figure 3. illustrates the architecture of Ethernet Cores and location in the larger architecture:*](tt-metal/docs/source/common/images/MfB-Fig3a.png)![](tt-metal/docs/source/common/images/MfB-Fig3b.png)
### 2.3 DRAM Core Architecture ###
DRAMs are responsible for moving data from Tensix Cores to off-chip locations. All RISC-Vs are programmable in DRAM Cores. Kernels for asynchronous pre-load spill to the DRAM Core when memory is running low so that the system will not crash or experience unnecessary errors. DRAMs are considered volatile, meaning saved data will be erased when powered off.

<d name=Fig4></d>![* *Figure 4. illustrates the architecture of DRAM Cores and location in the larger architecture:*](tt-metal/docs/source/common/images/MfB-Fig4.png)
### 2.4 NoC Architecture ###
NoCs have a torus topology. Each NoC has unidirectional rings in both red and blue indicating the movement of data. Routers move data around the red and blue rings. Each ring’s NoC Link BW is 32 B/c, each red and blue line represents 32B/c.

<e name=Fig5></e>![* *Figure 5. illustrates NoC architecture of routers and unidirectional data movement rings represented by the blue and red lines:*](tt-metal/docs/source/common/images/MfB-Fig5.png)
## 3. Program Model ##
### 3.1 Tensix Core Architecture ###
Tensix Cores consist of the following components:
- **Compute Components:**
  - **RISC-V** - A tiny, lightweight CPU that runs user kernels. All RISC-V CPUs within the Tensix Cores are programmable.
  - **Tile Math Engine** - A powerful tile-based math engine.
  - **Vectore Math Engine** - A fully programmable vector engine.
- **Data Movement Components:**
  - **ETH Controller** - Connects a core to the off-chip Ethernet.
  - **Router** - Connects a core to a NoC.
- **Storage Components:**
  - **DRAM Memory Bank Controller** - Off-chip memory.
  - **SRAM** - On-chip memory.

<f name=Fig6></f>![* *Figure 6. illustrates RISC-Vs in a Tensix Core and how they correspond to kernels:*](tt-metal/docs/source/common/images/MfB-Fig6a.png)![](tt-metal/docs/source/common/images/MfB-Fig6b.png)
### 3.2 Tensix Kernels ###
Kernels are a central component of any operating system, they perform processing, hardware management, and data movement tasks. Kernels transfer data to different parts of the system.

The following types of kernels are used in a Tensix Core:
- **Bare Metal C/C++ Kernels** - Kernels written in C or C++ for execution. Generally used to accelerate computation and processing performance.
- **Reader Kernel** - Kernel receiving data from a DRAM or SRAM buffer.
- **Writer Kernel** - Kernel sending data to a DRAM or SRAM buffer.
- **User Kernel Types:**
  - **Compute Kernels** - Kernels used for processing tasks or operations. Compute kernels will automatically generate the following types of kernels:
    - **Unpack Kernels** - Unpack kernels prepare data for operations to be performed by the math kernel.
    - **Math Kernels** - Math kernels are used for matrix multiplication (MatMul) and other mathematical tasks or operations.
    - **Pack Kernels** - Pack kernels wait for the end of the math kernel and prepare data to be moved to the next part of the system.
  - **Data Movement Kernels** - The first and fifth RISC-Vs on a Tensix Core that moves data between NoCs, memory buffers, and the compute kernel.
  - **Ethernet Data Movement Kernels** - Kernels that move data between cores.
- **Dispatch Kernels** - Kernels that determine where data is dispatched depending on the data’s priority and dispatch key.
- **Low-Level Kernels** - Kernels at the bottom of the software stack for t basic systems functions.

Compute kernels will automatically generate math, pack and unpack kernels. Unpack, Math, and Pack kernels execute in this order as data moves through a Tensix Core.

<g name=Fig7></g>![* *Figure 7. illustrates data movement and compute kernels in a Tensix Core. It shows data entering the first NoC, moving through SRAM buffers, the matrix engine, and vector engine:*](tt-metal/docs/source/common/images/MfB-Fig7.png)
### 3.3 Tensix Compute ###
In a Tensix core, RISC-Vs 1 and 5 are intended for data movement. RISC-Vs 2, 3, and 4, make up the compute kernel. Regardless of their role, all RISC-Vs have basic computing capabilities.

<h name=Fig8></h>![* *Figure 8. illustrates RISC-Vs 1 and 5 as data movement processors and RISC-Vs 2-4 as compute processors. Tile math and vector math engines make up the matrix engine:*](tt-metal/docs/source/common/images/MfB-Fig8.png)

The tile math engine uses a rich matrix ISA for MatMul, dot product, element-wise, and transpose operations. In Tensix cores compute instructions operate on a 32x32 matrix of scalars. These tiles operate on coarse data chunks of a single-threaded RISC-V to dispatch instructions. Data movement RISC-Vs issue asynchronous tile-sized data movement instructions to move data into a scratch SRAM, allowing many outstanding transfers generated by a single RISC-V data movement processor. The vector math engine uses a general purpose vector ISA for elementwise, sort, reshuffle, and LUT operations.

Three automatically compiled RISC-Vs make up the user compute kernel. Tile and vector math engines are an open source library of low level kernels with one API per math function. There are hundreds of tile and vector math low level kernels.

<i name=Fig9></i>![* *Figure 9. illustrates the user compute kernel instruction dispatch to the matrix engine:*](tt-metal/docs/source/common/images/MfB-Fig9.png)
### 3.4 Tensix Compute Schemes ###
Near memory computing (NMC) and SRAM use high bandwidth memory (HBM) and a high capacity SRAM in each Tensix core; high capacity allows the Tensix cores to achieve “silicon peak” performance. Single-level SRAMs are used as primary tensor storage, minimizing reliance on HBM. Tensix compute schemes efficiently implement distributed shared memory, optimizes data layout and movement, allows in-place computing on local SRAM data, and reduces unnecessary data movement. Native tile-based computing operates in a 32x32 matrix of tiles, uses single-threaded RISC-Vs, and supports concurrent data movement and computing. Explicit data movement is decoupled from computing allowing kernels to manage data transfer to the local SRAM. Explicit data movement is pre-planned and optimizes data movement. No caches or global crossbars are required.
### 3.5 Data Movement in the Tensix Core ###
Performance and efficiency of data movement in AI and HPC applications is as important as raw compute capacity. Data movement is explicit and separate from compute. Data movement kernels use the data movement engine in each Tensix core to move data from neighboring cores or off-chip DRAM to the local SRAM and trigger the compute engine. There are no caches, global crossbars, memory access, or other complex mechanisms used in traditional architecture that hide data movement. There are two data movement kernels in the Tensix core responsible for transferring data to NoCs. These kernels can read and write asynchronously and have access to all SRAM and DRAM memory banks.

<j name=Fig10></j>![* *Figure 10. illustrates the data movement kernel’s instruction dispatch to NoCs:*](tt-metal/docs/source/common/images/MfB-Fig10.png)

The following features are integrated into Tensix cores:
- **Independent NoCs** - 2
- **NoC Type** - 2-dimensional torus
- **NoC Link Width** - 64 bytes
- **NoC Link BW** - 83 GB/s
- **Tensix -> NoC I/O BW** - 665 GB/s
- **SRAM <-> NoC** - 333 GB/s
- **SRAM <_> NoC aggregate BW** - 47 TB/s

### 3.6 Memory Model ###
The TT-Metalium platform contains three memory regions:
- **Device DRAM** - Off-chip memory. Provides larger, off-chip storage for the system.
- **Device SRAM** - On-chip Memory. 1 MB SRAM memory (L1), a scratch pad accessible by all RISC-V processors and engines within the core.
- **Host Memory** - Off-chip memory bank residing in the host machine.

Use the following synchronization methods for data movement between memory regions:
- **Circular Buffers** - A circular buffer is a memory data structure with a fixed-size buffer used to store data in a continuous loop.
- **Memory Barriers** - Memory barriers act as hard stops for operations until any outstanding or incomplete operations are completed.
- **Semaphores** - Semaphores synchronize data movement through a single channel. They increment operations using kernels that track target values of operations. Once target values are met, subsequent operations are executed.

<k name=Fig11></k>![* *Figure 11. illustrates the memory model read/write capabilities of host kernels:](tt-metal/docs/source/common/images/MfB-Fig11.png)
#### 3.6.1 Device Memory Buffer Types ####
Tenstorrent devices support two device buffer types, DRAM and L1 SRAM, each device consists of memory banks. DRAM memory is based on low power and large capacity shared memory, accessible by cores with a unique address. DRAM memory provides extra storage space for large data sets. L1 SRAM memory is based on high-speed local memory. Each Tensix core has its own on-chip L1 SRAM memory enabling cores to perform in-place or near-memory compute operations. Physically separated core memory banks can be addressed by a unique address.
##### 3.6.1.1 L1 Interleaved Memory #####
L1 Interleaved memory distributes pages across all L1 banks sequentially and divides memory into multiple banks for fast access and improved bandwidth. It uses an address mapping generator to distribute consecutive addresses across memory banks. Storage cores are dedicated to L1 memory space, no compute cores or kernels run on them.
##### 3.6.1.2 L1 Sharded Memory #####
Sharding data falls into three strategies: height sharding, width sharding, and block sharding.
- **Height Sharding** - Height sharding splits data vertically across dimension 0. In a height sharded scheme the input matrix height is divided into contiguous segments, width is kept as full, and each shard is assigned to a different core.
- **Width Sharding** - Width sharding splits data horizontally across dimension 1. In a width sharded scheme the input matrix width is divided into segments, height is kept as full, and each core is assigned to a different segment.
- **Block Sharding** - Block sharding splits data vertically and horizontally across dimensions 0 and 1. A block sharded scheme divides height and width into submatrices.

<l name=Fig12></l>![* *Figure 12. illustrates data sharding types:*](tt-metal/docs/source/common/images/MfB-Fig12.png)
### 3.7 Synchronization Mechanisms ###
Synchronization mechanisms are techniques used to ensure operations can be executed simultaneously with consistent accuracy. Synchronization mechanisms can be both software and hardware based. These techniques allow for efficient processing on a multi-chip scale-out (sea of cores) of tenstorrent devices. The following synchronization mechanisms are utilized by TT-Metalium:
- **Circular Buffer** - A circular buffer is a memory data structure with a fixed-size buffer used to store data in a continuous loop.
- **Command Queue** - The command queue is the data exchange between the host and the device. The main configuration program queues up requests, waits for responses, then returns requested data.
- **Memory Barrier** - Memory barriers act as hard stops for operations until any outstanding or incomplete operations are completed.
- **Multicast** - Multicast operations are performed across chips and cores. Multiple Instruction Multiple Data (MIMD) allows multiple processors to function simultaneously and asynchronously. Multicast allows for efficient processing on multi-chip scale-out of tenstorrent devices. The alternative to multicast is a singlecast operation.
- **Semaphore** - Semaphores synchronize data movement through a single channel. Semaphores increment operations using kernels that track the target values of operations. Once the target values are met, subsequent operations are executed.
- **Singlecast** - Single Instruction Multiple Data (SIMD) allows processors to execute the same instruction on multiple data points simultaneously.

<m name=Fig13></m>![* *Figure 13. illustrates Multi-Chip scale-out or sea of cores:*](tt-metal/docs/source/common/images/MfB-Fig13.png)
### 3.8 Tensix Tensors ###
Tensors are data structures that represent and process data in the network and are stored in the SRAM buffer. Tensors can be structured in a row-major layout or a tiled layout:
- **Row-Major Tensor Layout** - Each row of the tensor corresponds to a page in the buffer.
- **Tiled Tensor Layout** - Pages represented as 2D tiles unconfined to a row.

Pages of a tensor are distributed across the memory of a device; memory on a device is partitioned into banks. Tensors have two storage methods that describe how it is mapped to memory banks, interleaved and sharded:
- **Interleaved** - Interleaved tensor pages are distributed across multiple banks. Interleaved tenors often have some fragmentation.
- **Sharded** - Sharded tensors are physically distributed across L1 memory banks of multiple cores. The tensor is divided into partitions called shards, each shard is placed in the L1 memory bank of a specific core.
### 3.9 Data Movement Patterns ###
Tenstorrent technology is built for AI data movement patterns. Data patterns in MatMuls, Convolutions, and Sharded Data Layouts are regular and mapped to the Mesh Architecture.

Memory and I/O components are listed with corresponding data movement patterns and BW:
- **SRAM** - Local/Shared - 94 TB/s
- **SRAM** - Neighbor (Halo) - 47 TB/s
- **SRAM** - Row/Column/Mesh Multicast - 24 TB/s
- **SRAM** - Gather/Scatter (3 hops) - 16 TB/s
- **SRAM** - Gather/Scatter (10 hops) - 5 TB/s
- **DRAM** - Row - 512 GB/s
- **Ethernet** - Column - 1 TB/s

<n name=Fig14></n>![* *Figure 14. illustrates data movement patterns in Tensix, SRAM, DRAM, and Ethernet cores:*](tt-metal/docs/source/common/images/MfB-Fig14.png)
## 4. Scale Out ##
### 4.1 Tensix Operations ###
The following are fundamental operations carried out by TT-Metalium:
- **Elementwise Ops** - Local on each element in a tensor data movement is not required.
- **MatMul Ops** - Performed across matrix rows and columns.
- **Reduction Ops** - Decomposed across matrix rows and columns and neighboring cores.
- **Temporal Ops** - Operations running sequentially using all available resources.
- **Window Based (stencil) Ops** - Convolution exchanging data with neighboring cores.

AI workloads operate on tensors and display locality and regularity in fundamental compute operations. Elementwise operations are local on each element in the tensor, and can be completed without data movement. MatMul operations communicate regularly across rows and columns of the matrix. Reduction operations are decomposed across dimensions like columns and rows, to nearby neighbors in the matrix. Window based (stencil) operations like convolutions, exchange data with their neighbors. Temporal operations run one after the other, each using all available resources. Every operation runs as fast as it can. Other operation layouts are not currently used in TT-Metalium.
### 4.2 Tensix Dataflow ###
Data in the Tensix architecture flows to and from certain devices, cores, and kernels. Data moves freely from the host machine to a device, from a device to a host machine, from core to core on the same device, or from kernel to kernel in the same core. Here are the ways in which data can flow through Tensix architecture:
- **Host to Device** - Data moves freely between the host machine and device. The main host machine memory is the source of data and sends it to the buffer in the core memory of the device. On-device memory can use either DRAM or SRAM cores.
- **Device to Host** - Data moves freely between the device and the host machine. The buffer is the data source for core memory on the device. Data is sent to the vector or address to be stored on the host machine.
- **On-device Core-to-core** - Core-to-core on the same device, cores can read or write:
  - **Read** -  Data is read freely between neighboring cores on a device. The source is a global 64 bit buffer address. It is sent to a local 32 bit circular address.
  - **Write** - Data is written to neighboring cores on the same device. In this case the source is a circular buffer in the cores SRAM memory. Data is written to a 64 bit buffer address in the buffer of a DRAM or SRAM memory core.
- **On-core Kernel-to-Kernel** - Data can move freely between kernels in the same core. Data is sent from the first kernel through the circular buffer in the SRAM memory of the Tensix core to the second kernel.
### 4.3 Compile-time Stack ###
The TT-Metalium software stack contains the compile-time stack and runtime support. The TT-Metalium compile-time stack compiles source code into the main configuration program, data sets, and compute kernels.
### 4.4 Runtime Support ###
Runtime support assembles kernels then dispatches those kernels to Tensix cores. The runtime stack will start and control kernel executions, initialize Tensix and DRAM memory banks, then collect results and stop all kernel executions when finished.

<o name=Fig15></o>![* *Figure 15. illustrates the Compile-time Stack and Runtime Support in the TT-Metalium Stack:*](tt-metal/docs/source/common/images/MfB-Fig15.png)
## 5. Glossary of Terms ##
To view the full Glossary of Terms used in TT-Metalium documentation see: [Glossary of Terms](tt-metal/GLOSSARY.md)
