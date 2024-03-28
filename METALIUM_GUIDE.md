## TT Architecture and Metalium Overview

Tenstorrent has built the future of AI architecture and programming. 
It achieves high performance on current models but is flexible and programmable to enable invention of future models without the constraints of current architectures.
It is designed for both inference and training, and is from ground up designed for scale-out of AI workloads, while allowing to be scaled down to a couple of cores.
Additionally, it is built using cost-effective components: simple packages, GDDR memory and Ethernet.
This document desribes how we did it. 

* [All you need is a Tensix core and a mesh](#all-you-need-is-a-tensix-core-and-a-mesh)
* [Scalable Architecture](#scalable-architecture)
  - Two levels of memory  
* [MIMD and Control of Both Compute and Data](#mimd-and-control-of-both-compute-and-data)
* [Everything is a RISCV kernel](#everything-is-a-riscv-kernel)
  - Bare Metal C/C++ kernels on RISCV 
  - User Kernels: Decoupled Data Movement and Compute
    - Data Movement Kernels
    - Compute Kernels 
    - Ethernet Data Movement Kernels
  - Dispatch Kernels
* [Native Support for Tiled Compute and Data](#native-support-for-tiled-compute-and-data)
* [Interleaved and Sharded Buffers](#interleaved-and-sharded-buffers)
* [Eltwise Binary Kernel](#eltwise-binary-kernel)

### All you need is a Tensix core and a mesh 
 - Tensix Core includes 5 small RISC-V processors (aka "Baby RISCVs"), a Matrix Engine, a Vector engine, and 1 MB scratch pad SRAM.   
 - Near Memory Compute in a Mesh

<img width="1167" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/78d64b36-bb68-4d41-b2ca-5e3ed7ccda8f">


### Scalable Architecture

AI workloads operate on tensors (N-dimensional data) and exhibit a high degree of locality and regularity in key compute operations: elementwise, matrix multiplications, reduction and convolutions.
Elementwise operations are entirely local on each element in the tensor, and can be achieved without any data movement. 
Matrix multiplication operation have regular communication across the rows and columns of the matrix. 
Reductions can be decomposed across dimensions: columns, rows and then nearest neighbours. 
Similarly window based (stencil) operations exchage data with their neigbours. 

These data movement patterns (local, row/column regular, nearest neighbour) are most efficiently implemented via regular and scalable mesh architecture.
Mesh has very high local SRAM BW -- at the speed of silicon.  
Tenstorrent architecture is a mesh of cores within a chip and mesh of chips at the cluster level. 

<img width="2141" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/0f40ace9-e2b3-4740-a89c-3e8a3580da8a">

### MIMD and Control of Both Compute and Data

- "Program cores not threads" , In TT Architecture the program specifies a kernel 

### Everything is a RISCV kernel

 - Bare Metal C/C++ kernels on RISCV 
  - User Kernels
  <img width="1176" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/d3c89155-6e4d-49cb-a95c-85654ac29e7d">

    - Data Movement Kernels
    - Compute Kernels 
    - Ethernet Data Movement Kernels
  - Dispatch Kernels

### Native Support for Tiled Compute and Data

### Interleaved and Sharded Buffers

- Interleaved Buffers
- Sharded Buffers

### Eltwise Binary Kernel

```
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



<img width="1470" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/6ea0cefc-6109-4579-8470-7a620f45b314">



<img width="1171" alt="image" src="https://github.com/tenstorrent-metal/tt-metal/assets/3885633/73039d17-3bce-4ff5-b797-da1aa9b147c4">
