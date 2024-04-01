# TT Architecture and Metalium Guide

* [TT Architecture and Metalium Overview](#tt-architecture-and-metalium-overiew)
* [TT Architecture deep dive](#tt-architecture-deep-dive)
* [Learn Metalium in 7 steps](#learn-metalium-in-7-steps)


## TT Architecture and Metalium Overview

Tenstorrent has built the future of AI architecture and parallel programming. 
It achieves high performance on current AI models but is flexible and programmable to enable invention of future AI models and HPC applications without the constraints of current architectures.
It is designed for both inference and training, and is from ground up designed for scale-out of AI workloads, while allowing to be scaled down to a couple of cores.
Additionally, it is built using cost-effective components: simple packages, GDDR memory and Ethernet.
This document desribes it. 

* [All you need is a Tensix core and a mesh](#all-you-need-is-a-tensix-core-and-a-mesh)
  - [Near Memory Compute](#near-memory-compute)
  - [Distributed Memory and In-Place Compute](#distributed-memory-and-in-place-compute)
  - [Explicit Data Movement](#explicit-data-movement)
  - [Native Tile-Based Compute](#native-tile-based-compute)
  - [Think Bare Metal Cores, Not Threads](#native-tile-processing)
* [Scalable Architecture](#scalable-architecture)
  - [Two levels of memory](#two-levels-of-memory)
  - [Compute Density via Scale-Out](#compute-desnsity-via-scale-out)
* [MIMD and Control of Both Compute and Data](#mimd-and-control-of-both-compute-and-data)
* [Everything is a RISCV kernel](#everything-is-a-riscv-kernel)
  - Bare Metal C/C++ kernels on RISCV 
  - [User Kernels: Explicit and Decoupled Data Movement and Compute](#user-kernels-explicit-and-decoupled-data-movement-and-compute)
    - Data Movement Kernels
    - Compute Kernels 
    - Ethernet Data Movement Kernels
    - Read-Compute-Write kernel pipeline
  - Dispatch Kernels
* [Efficiency of Tiled-Based Compute and Data Movement](#efficiency-of-tile-based-compute-and-data-movement)
* [Interleaved and Sharded Buffers](#interleaved-and-sharded-buffers)
* [Fast Kernel Dispatch](#fast-kernel-dispatch)
* [FAQ](#FAQ)
  - [What about CUDA, scalar threads, and caches?](#what-about-cuda-scalar-threads-and-caches)  
  - [What about HBM?](#what-about-HBM)
  - [For GPU, CPU, FPGA, TPU experts](#for-gpu-cpu-fpga-tpu-experts)
  - [First Principles Summary](#first-principles-summary)

### All you need is a Tensix core and a mesh 
 A Tensix Core is:
 - **5 small RISC-V processors** (aka "Baby RISCVs") that run C/C++ kernels and dispatch instructions to the compute and data movement engines
 - **1 MB SRAM memory**, a scratch pad accessible by all RISCVs and engines within the core
 - **Matrix engine (aka FPU)** that performs Matrix multiplication, elementwise, and dot product operations on small matricies (or tiles) of shape 32x32 and similar
 - **Vector engine (aka SFPU)** for vectorized kernels such as Top-k, Sort and special functions such as GELU, Exp, and Sqrt
 - **Data Movement engine** connected to 2 Networks on Chip (NoCs)

A chips is a collection of cores and I/O blocks, connected into a mesh via a NoC:
- **Tensix compute core** (each with local SRAM)
- **DRAM memory banks**
- **Ethernet cores** for chip-to-chip interconnect
- **PCIe link** for host interface
- **ARC core** for board management and control

<img width="900" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/78d64b36-bb68-4d41-b2ca-5e3ed7ccda8f">

#### Near Memory Compute
The high BW and large capacity SRAM in each Tensix core is a form of **near memory compute**. A Tensix core operating on its local SRAM achieves **"silicon peak"** of what current technology node allows for. 
Tensix cores are connected into a mesh via 2 NOCs, and each Tensix core can communicate with any other Tensix core in the mesh, and with off-chip DRAM, as well as Ethernet cores.

#### Distributed Memory and In-Place Compute
The mesh of Tensix cores architecture is the first one to efficiently implement distributed memory and enable programmers and compilers to optimize both layout and movement of the data. 
In many AI and HPC operations, such as as elementwise, the tensors can be laid out (ie "sharded") across SRAMs so that compute can operate on the local data **in-place** without any data movement. 
Further elaboration in [Scalable Architecture](#scalable-architecture) section.  

#### Explicit Data Movement
The performance and efficiency of data movement in AI and HPC application is as important as raw compute capacity of the math engines. 
In Tenix, data movemenet is explicit and decoupled from compute. The data movement kernels use the data movement engine in each Tensix to bring data from neighbouring cores or off-chip DRAM to the local SRAM of the Tensix core, and trigger the compute engine to operate on the data. The data movement in TT architecture can be pre-planned, optimized and debugged separately from the compute.
There is no caches, no global crossbars, no memory access coalesing or other complex mechanisms that are used in traditional architectures that hide the data movement from the programmer or compiler.
For deeper insight see section [User Kernels: Explicit and Decoupled Data Movement and Compute](#user-kernels-explicit-and-decoupled-data-movement-and-compute).

#### Native Tile-Based Compute
In Tensix, compute instructions operate on tiles -- 32x32 matrix of scalars. Operating on coarse chunks of data allows for use of a simple single-threaded RISCVs processors to dispatch these instructions. 
Similarly, data movement RISCVs issue asynchronous tile-sized data movement instructions to bring data into the scratch SRAM, allowing for large number of outstanding transfers generated by a single RISC-V data movement processor, concurrently with the compute engine. 

#### Think Bare Metal Cores, Not Threads
Each RISCV processor runs single-threaded and Core-to-Thread mapping is 1:1. Thus, the parallelization involves breaking the work across cores and dispatching the kernels directly to cores. This is in contrast to a complex thread scheduling scheme where a very large number of threads is time-slice scheduled onto a limited number of cores. As a result, in TT architecture there is no context switching or complex thread scheduling. Once the kernel is dispatched to a core it runs to completion without interruption or preemption by another thread. This simplifies reasoning about performance: it boils down to direct cycle couting of sections of a C/C++ kernel running on a bare metal RISCV core.
Equally important, it simplifies direct debug of kernels via gdb step-through, breakpoints, and printf from cores. 

### Scalable Architecture
AI workloads operate on tensors (N-dimensional data) and exhibit a high degree of locality and regularity in the fundamental compute operations:
- **Elementwise operations** are entirely local on each element in the tensor (in-place), and can be achieved without any data movement
- **Matrix multiplication operations** have regular communication across the rows and columns of the matrix
- **Reduction operation** can be decomposed across dimensions, such as columns, rows, and nearest neighbours in a matrix
- **Window based (stencil) operations** such as convolutions exchage data with their neigbours

These data movement patterns (local, row/column, nearest neighbour) are most efficiently implemented via a regular and scalable mesh architecture.

Tenstorrent architecture is a mesh of cores within a chip and mesh of chips at the cluster level. 
<img width="900" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/0f40ace9-e2b3-4740-a89c-3e8a3580da8a">
TODO: Describe Galaxy, break up the slide into two slides

#### Two levels of memory

TODO: Describe SRAM and DRAM as two levels of memory hierachy within the mesh, both distributed and explicit data movement.

#### Compute Density via Scale-Out

TODO: Describe that TT wins at scale-out, best compute density at the server and cluster level 

### MIMD and Control of Both Compute and Data

- "Program cores not threads" 

### Everything is a RISCV kernel

 - Bare Metal C/C++ kernels on RISCV 
  - User Kernels
    - Data Movement Kernels
    - Compute Kernels 
    - Ethernet Data Movement Kernels
  - Dispatch Kernels
  <img width="1176" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/d3c89155-6e4d-49cb-a95c-85654ac29e7d">
<img width="1171" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/73039d17-3bce-4ff5-b797-da1aa9b147c4">


### Efficiency of Tile-Based Compute and Data Movement

### Interleaved and Sharded Buffers

- Interleaved Buffers
- Sharded Buffers

### Fast Kernel Dispatch

### FAQ
#### Where is CUDA, scalar threads, and caches? 
#### Where is HBM?
#### For GPU, CPU, FPGA, TPU experts
 - GPU
 - CPU
 - FPGA
 - TPU
#### First Principles Summary
  



## TT Architecture deep dive

TODO: 1) TOC, write it 

## Learn Metalium in 7 steps

TODO: 1) TOC, 2) write it

### Eltwise Binary Compute Kernel

TODO: this is a placeholder

```cpp
#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1); // should be <= 8 in this kernel

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 =  tt::CB::c_out0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init();

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        // wait for a block of tiles in each of input CBs
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);

        tile_regs_acquire(); // acquire 8 tile registers
        // add a block of tiles
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            add_tiles(cb_in0, cb_in1, i, i, i);
        }
        tile_regs_commit(); // signal the packer

        tile_regs_wait(); // packer waits here
        // pack a block of tiles
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            pack_tile(i, cb_out0);
        }
        tile_regs_release(); // packer releases

        // pop a block of tiles from each of input CBs
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);

        // push a block of tiles to output CB
        cb_push_back(cb_out0, per_core_block_size);
    }

}
}
```

