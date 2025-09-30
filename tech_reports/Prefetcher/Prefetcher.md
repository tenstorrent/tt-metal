# Prefetcher Technical Report

## Table of Contents
- [Problem with Reading from DRAM](#problem-with-reading-from-dram)
- [Idea of the Prefetcher](#idea-of-the-prefetcher)
- [Sub-Devices: Concept and Usage](#sub-devices-concept-and-usage)
- [Prefetcher Architecture](#prefetcher-architecture)
- [Global Circular Buffer Implementation](#global-circular-buffer-implementation)
- [1D Ring Matmul Integration](#1d-ring-matmul-integration)
- [Performance Benefits](#performance-benefits)
- [Future Works](#future-works)

## Problem with Reading from DRAM

TODO: Add content describing the challenges and limitations of traditional DRAM reading approaches.

## Idea of the Prefetcher

TODO: Add high-level overview of the prefetcher concept and solution approach.

## Sub-Devices: Concept and Usage

Sub-devices enable partitioning the chip by grouping cores into isolated execution units with independent memory allocators and program execution tracking. This allows the prefetcher to run on dedicated cores while compute kernels execute independently on separate sub-devices, enabling true asynchronous data streaming without blocking the main computation pipeline.

For detailed information about sub-devices, their implementation, and APIs, see [SubDevices.md](../SubDevices/SubDevices.md).

## Prefetcher Architecture

The prefetcher implements a data movement op that efficiently fetches data from DRAM banks to receiver cores through global circular buffers. Currently, the prefetcher is used to prefetch weight tensors for Matmul operations only, though in theory it should support prefetching for other ops and activations as well. The reason we only use it for Matmul is because it has the largest weight tensors, and prefetching them provides the most benefit.

### API Overview and Setup Requirements

The prefetcher exposes an API to users at the TTNN level: `ttnn.dram_prefetcher(...)`

**API Interface:**


```python
# tt_tensors contains:
# - weight tensors for one layer (e.g., [ff1_weights, ff2_weights, ff3_weights, qkv_weights, ...])
# - address tensor containing DRAM addresses for all layers (last element in list)
tt_tensors = [weight_tensor_1, weight_tensor_2, ..., weight_tensor_n, address_tensor]

ttnn.dram_prefetcher(
    tt_tensors,                           # Weight tensors (1 layer) + address tensor (all layers)
    num_layers,                           # Number of decoder layers to prefetch through
    global_cb=global_circular_buffer,     # Global CB for prefetcher→matmul communication
    enable_performance_mode=True          # Skip NoC pointer updates for better performance (legacy flag for debugging purpose)
)
```

**Input Data Organization:**
The op expects two types of input data:

1. **Weight Tensors for One Decoder Layer**: In LLMs, there are several decoder layers, and within each decoder layer there are several Matmul operations that need to be prefetched. Between different layers, the Matmul tensor shapes are the same. We utilize this feature and only pass in the tensors for the first layer to reduce the number of runtime args needed to be passed to the kernel. These tensors must be width-sharded across DRAM banks and stored in DRAM with tile layout. The prefetcher supports multiple data types (bfloat4_b, bfloat8_b, bfloat16) with different tile sizes.

2. **Address/Configuration Tensor**: A config tensor containing the DRAM buffer addresses for ALL weight tensors across ALL layers. This tensor tells the prefetcher where to find each tensor for every layer, enabling it to fetch data for layers beyond the initial pattern. The tensor is height-sharded across prefetcher cores and stored in L1 for fast access, with each prefetcher core getting the complete address map.

Note: even though the prefetcher is designed based on the fact that each LLM has repeated decoder layers, it doesn't forbid the support of non-repeated decoder layers or other types of layers. Users just need to set num_layers=1 and then pass in all the tensors needed for prefetching.

**Global Circular Buffer Configuration:**
The global circular buffer serves as the communication bridge between prefetcher and consumer kernels. It requires a sender-receiver core mapping that defines which prefetcher cores send data to which consumer cores. The global CB buffer size ideally should buffer at least two tensors to avoid any stall on the consumer side (double buffering), although due to the fact that not all tensors are of the same size, in practice it needs to buffer more tensors to avoid any stall.

```python
# Map each prefetcher sender core to its corresponding receiver cores
sender_receiver_mapping = list(zip(prefetcher_sender_cores, matmul_receiver_cores))

# Create the global circular buffer
global_circular_buffer = ttnn.create_global_circular_buffer(
    device,
    sender_receiver_mapping,
    global_cb_size
)
```

**Sub-Device Setup:**
The op uses separate sub-devices for prefetcher and compute operations, enabling asynchronous execution. The prefetcher sub-device contains the sender cores running DRAM reader (issues NoC read to DRAM banks) and L1 writer (issues NoC writes to consumer cores) kernels, while the worker sub-device contains consumer cores running matmul and other operations.

```python
# Create sub-devices for independent execution
prefetcher_sub_device = ttnn.SubDevice([prefetcher_sender_core_range_set])
worker_sub_device = ttnn.SubDevice([matmul_worker_core_range_set])

# Create sub-device manager
sub_device_manager = device.create_sub_device_manager(
    [prefetcher_sub_device, worker_sub_device]
)

# Load the sub-device configuration onto the device
device.load_sub_device_manager(sub_device_manager)

# Set stall group to only wait on worker sub-device for synchronization
device.set_sub_device_stall_group([worker_sub_device_id])
```

This API design abstracts the complex multi-core coordination while providing users control over memory layouts and core allocation parameters.

### Physical Chip Layout and Core Allocation

The prefetcher leverages the spatial distribution of cores across the Tenstorrent chip to enable efficient data streaming:

![Galaxy Unharvested Chip Layout](images/galaxy_unharvested_layout.png)

**Core Allocation Strategy:**
- **Prefetcher Cores (Red W)**: Dedicated worker cores spatially distributed across the chip run the prefetcher kernels. These cores are placed near the DRAM banks to minimize NoC traffic congestion.

- **Matmul Cores (Purple W)**: Regular worker cores for matmul operations. Each prefetcher core issues NoC writes to the two Matmul cores on the right, placing them on the same row to reduce interference with other prefetch cores. For other device architectures, we can have one prefetcher core serving more or fewer Matmul cores to maximize the overall compute utilization.

- **Data Flow Path**: Data flows from DRAM banks → Prefetcher cores → Global Circular Buffers → Compute cores, with each stage operating on different physical regions of the chip.

### Kernel Architecture Overview

The prefetcher utilizes a reader-writer kernel architecture to efficiently stream data from DRAM to consumer cores:

**1. DRAM Reader Kernel** (`reader_dram.cpp`)
- Fetches data from DRAM banks into local triple-buffered circular buffer
- Each prefetcher core reads tensors from its assigned DRAM bank, processing each tensor block by block (block size and count are based on Matmul specifications)
- **Transaction ID Management**: Uses NoC transaction IDs to coordinate DRAM reads with the triple-buffered circular buffer. The kernel maintains `curr_block_trid` to track which transaction ID to use for each block, rotating through IDs 1→2→3→1→... to map reads to different blocks.

For detailed implementation of DRAM reading optimizations and transaction ID management, see [Saturating_DRAM_bandwidth.md](../Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md).

**2. L1 Writer Kernel** (`writer_l1.cpp`)
- Transfers data from local CB to global CB (remote consumers)
- Distributes each tensor block across multiple receiver cores
- Currently only supports slicing the tensor block on the width dimension; for one prefetcher core that serves multiple receiver cores, each receiver core gets a slice of the block
- Uses sender-side global CB operations (detailed in the next section)

## Global Circular Buffer Implementation

The global circular buffer is the core communication mechanism enabling efficient data transfer between prefetcher (sender) cores and matmul (receiver) cores. This section describes the kernel-side implementation, including configuration structures, initialization, operations, and synchronization mechanisms.

### 1. Initialization and Configuration

The global circular buffer initialization phase occurs when kernels start up and need to configure their remote CB interfaces. This process reads configuration data from L1 memory and sets up data structures that enable cross-core communication.

**Configuration Data Layout in L1 Memory:**
Each core's global CB configuration is stored in L1 memory as an array of `uint32_t` values accessed by index:
```cpp
// L1 configuration data layout (initialized on device)
volatile tt_l1_ptr uint32_t* l1_remote_cb_config_addr = /* ... */;

// Configuration elements accessed by array index:
const bool is_sender = l1_remote_cb_config_addr[0];              // 0=receiver, 1=sender
uint32_t num_receivers = l1_remote_cb_config_addr[1];            // Number of receiver cores
uint32_t fifo_start_addr = l1_remote_cb_config_addr[2];          // Base address of circular buffer
uint32_t fifo_size = l1_remote_cb_config_addr[3];               // Total size of circular buffer
uint32_t fifo_ptr = l1_remote_cb_config_addr[4];                // Initial read/write pointer
uint32_t remote_noc_xy_addr = l1_remote_cb_config_addr[5];      // Address containing NoC coordinates
uint32_t aligned_pages_sent_addr = l1_remote_cb_config_addr[6]; // Address for page synchronization counters
```

**Initialization Process (`setup_remote_cb_interfaces`):**
During kernel startup, each core reads its configuration from L1 and initializes the appropriate interface structure:

**For Sender Cores (Prefetcher):**
```cpp
RemoteSenderCBInterface& sender_cb_interface = get_remote_sender_cb_interface(cb_id);
sender_cb_interface.config_ptr = config_addr;                    // Pointer to L1 config
sender_cb_interface.fifo_start_addr = fifo_start_addr;          // Buffer start address
sender_cb_interface.fifo_wr_ptr = fifo_ptr;                     // Write pointer
sender_cb_interface.receiver_noc_xy_ptr = remote_noc_xy_addr;   // Receiver NoC coordinates
sender_cb_interface.aligned_pages_sent_ptr = aligned_pages_sent_addr; // Sent page counters
sender_cb_interface.num_receivers = num_receivers;             // Number of receivers
```

**For Receiver Cores (Matmul):**
```cpp
RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
receiver_cb_interface.config_ptr = config_addr;                 // Pointer to L1 config
receiver_cb_interface.fifo_start_addr = fifo_start_addr;       // Buffer start address
receiver_cb_interface.fifo_rd_ptr = fifo_ptr;                  // Read pointer
receiver_cb_interface.sender_noc_x = sender_noc_x;             // Sender NoC coordinates
receiver_cb_interface.sender_noc_y = sender_noc_y;
receiver_cb_interface.aligned_pages_acked_ptr = aligned_pages_acked_addr; // Acked page counters
```

**Key Initialization Steps:**
1. **Configuration Reading**: Each core reads its role (sender/receiver) and parameters from L1
2. **Interface Setup**: Appropriate CB interface structure is populated based on core role
3. **NoC Coordinate Mapping**: NoC coordinates are extracted for remote communication
4. **Synchronization Setup**: Page counter addresses are configured for flow control
5. **Initial Sizing**: The CB interface is sized with initial page size via `resize_remote_*_cb_interface`

**Note:**
The `config_ptr` stored in both `RemoteSenderCBInterface` and `RemoteReceiverCBInterface` serves a critical runtime purpose: we need to store the latest `fifo_wr_ptr` / `fifo_rd_ptr` value back to L1 memory so that the next op can read it back to initialize the local `RemoteSenderCBInterface` / `RemoteReceiverCBInterface`.

### 2. Dynamic Page Size Reconfiguration

On the consumer side, ops can **inplace** a local CB on top of the remote CB for local pointer updates and to consume different parts of the tensor. Due to the fact that each CB is required to be page-size aligned, the local CB size will not be the same as the global CB, but rather a size that is aligned to the current page size. Ideally, the consumer op will call cb_pop_front to access the tensor block by block without worrying about manual pointer updates.

When switching between tensors with different shapes or data formats, the global CB page size must be reconfigured to the new tensor block size. This process involves page alignment operations to ensure correct wraparound behavior when consumer performs the local CB pop.

![Global CB Tensor Allocation](images/global-cb-tensor-alloc.png)

**The Problem Without Alignment:**
If tensors are stored at non-aligned locations in the circular buffer, when the buffer wraps around to the top, a simple `cb_pop_front` operation on the consumer side will not land at the correct tensor boundary. The consumer would read from the middle of a tensor or across tensor boundaries, corrupting the data.

**Alignment:**
1. **Before New Tensor**: The write pointer may be at an arbitrary location after the previous tensor
2. **Alignment Step**: The pointer is advanced to the next page-aligned address that matches the new tensor's page size, i.e., (address - global CB start address) is always a multiple of the current page size.
3. **Tensor Storage**: The new tensor is stored starting at this aligned location
4. **Wraparound**: When the buffer wraps around, the consumer's `cb_pop_front` operations will correctly land at tensor boundaries

**Consumer Benefit:**
With page alignment, the consumer (matmul core) can perform simple circular buffer operations without worrying about tensor boundaries. Each `cb_pop_front` operation moves exactly one "page" (tensor block), and wraparound automatically lands at the start of the next tensor.

**Note:**
when the consumer does not rely on local CB pop but manually updates the read pointer, this will not be a problem and we do not need the extra page alignment feature. This is the case for the current Matmul implementation, where the compute kernel manually updates read pointers, so the feature is not required and can be disabled. However, when trying that on the LLama-70b model, the performance regressed by 1-2 tokens/second, and thus this feature is kept enabled. The potential cause could be that without alignment, less synchronization is needed and the prefetcher runs faster; however, a faster prefetcher can interfere with other ops more and cause a slowdown in other ops. Further experiments are needed to investigate disabling the alignment feature.

### 3. Sender-Side Operations

Prefetcher cores act as **senders** in the global CB system, writing data to be consumed by receiver cores. The sender-side operations provide flow control and data transfer capabilities.

#### 1. Dynamic CB Reconfiguration (`resize_remote_sender_cb_interface`)
```cpp
experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size_per_receiver, noc);
```
This API is called each time before switching to the next tensor. Since different tensors have different shapes, tile sizes, and data formats, each tensor has a different block size. The global CB page size needs to be reconfigured for each tensor to match the block size that will be sent to each receiver core.

**Functionality:**
- Updates the CB interface page size to match the new tensor's block size
- Adjusts read/write pointers to proper alignment
- If necessary, signals receivers over NoC about the configuration change
- Ensures proper wraparound behavior at the circular buffer boundary

#### 2. Flow Control (`remote_cb_reserve_back`)
```cpp
experimental::remote_cb_reserve_back(remote_cb_id, num_blocks);
```
Before sending data, the sender must ensure all receiver cores have available space in their portion of the global CB. This prevents overwriting data that receivers have not processed yet.

**Flow Control Mechanism:**
- Checks the credit counters (pages_sent vs pages_acked) for each receiver core
- Blocks until all receivers have enough free space for the incoming data
- Implements backpressure flow control
- Coordinates with receiver's `remote_cb_pop_front` to manage buffer space

**Functionality:**
- Calculates the number of aligned pages needed (accounting for wraparound)
- Polls each receiver's acknowledgment counter
- Blocks in a busy-wait loop until `(fifo_aligned_num_pages - (pages_sent - pages_acked)) >= num_pages_wait` for all receivers
- Ensures data integrity by preventing buffer overflow

#### 3. Data Transfer and Signaling (`remote_cb_push_back_and_write_pages`)
```cpp
experimental::remote_cb_push_back_and_write_pages<skip_ptr_update>(
    remote_cb_id,
    local_cb_addr,
    num_blocks,
    block_height_in_pages,
    coalesced_num_pages_per_row,
    coalesced_page_size,
    noc);
```
This is the core data movement operation that actually transfers the tensor block from the prefetcher's local CB to each receiver core's portion of the global CB.

**Parameters:**
- `block_height_in_pages`: Number of pages in the height dimension of each tensor block
- `coalesced_page_size`: Size of pages grouped together that can form one NoC write
- `coalesced_num_pages_per_row`: Number of NoC writes needed to be performed for each row of the block

**Functionality:**
- Performs coalesced NoC writes to distribute block slices to each receiver core
- Each receiver gets a horizontally-sliced portion of the tensor block
- Updates the sent page counters via NoC semaphore increment to signal receivers that new data is available
- Handles wraparound in the circular buffer automatically
- Advances the write pointer for the next operation

### 4. Receiver-Side Operations

Matmul cores act as **receivers** in the global CB system, consuming data streamed by the prefetcher. The receiver-side operations implement flow control and synchronization.

#### 1. Wait for Data Availability (`remote_cb_wait_front`)
```cpp
experimental::remote_cb_wait_front(remote_cb_id, num_pages);
```
Before consuming data from the global CB, the receiver must ensure that the prefetcher has sent the required pages. This prevents reading garbage or incomplete data.

**Flow Control Mechanism:**
- Checks the credit counters (pages_sent vs pages_acked) to determine how many pages are available
- Blocks until the sender has sent enough pages for consumption
- Implements backpressure by coordinating with sender's `remote_cb_reserve_back`

**Functionality:**
- Calculates the total number of pages needed, accounting for potential wraparound in the circular buffer
- Polls the `pages_sent` counter (written by sender) and compares it with local `pages_acked` counter
- Blocks in a busy-wait loop until `(pages_sent - pages_acked) >= num_pages_wait`
- Handles wraparound scenarios where the requested data exceeds the circular buffer boundary

#### 2. Release Consumed Data (`remote_cb_pop_front`)
```cpp
experimental::remote_cb_pop_front(remote_cb_id, num_pages, noc);
```
After consuming data from the global CB, the receiver must signal the sender that the space is now available for reuse. This implements the flow control mechanism that prevents the sender from overwriting data the receiver hasn't processed yet.

**Flow Control Mechanism:**
- Updates the read pointer and sends ack to the sender
- Enables the sender's `remote_cb_reserve_back` to unblock when waiting for free space

**Functionality:**
- Advances the `fifo_rd_ptr` by the number of pages consumed, handling wraparound if necessary
- Calculates the total number of pages to acknowledge (accounting for buffer wraparound)
- Sends ack to the sender's `pages_acked` counter via NoC atomic increment, signaling that this space is now free

## 1D Ring Matmul Integration

The matmul operation serves as the consumer side of the global circular buffer, implementing a ring-based data processing pattern for efficient multi-core execution. This section focuses on the global CB implementation and the integration with the prefetcher.

### Overview:

Matmul cores act as **receivers** in the global CB system, consuming data streamed by the prefetcher. Here is a diagram of a ring Matmul consisting of 4 cores, where each core starts at a different ring index. For example, core 0 starts at ring index 0 and iterates over the weight tensors in the global CB with an order 0→1→2→3, while core 1 starts at ring index 1 and iterates over the weight tensors in the global CB with an order 1→2→3→0.

![Matmul 1D Ring Example](images/matmul-1d-ring-example.png)

### Ring-Specific Circular Buffer Challenges

#### 1. Synchronization Requirements for Ring Topology

Since each core in the ring starts at a different block index, special synchronization is required:

**Wait for Full Tensor Before Computation:**
It is necessary to let Matmul wait for the entire tensor to arrive in the global CB before performing any computations. This is because the prefetcher is not aware of the out-of-order accessing of tensor blocks, and the `pages_sent` credit is in-order for each block. Performing any computations before the full tensor arrives might cause the use of garbage data.

**Release Full Tensor After Computation:**
The Matmul must pop out the tensor in the global CB only after all the blocks are processed, as `pages_ack` is also in-order for each block, and a premature ack can invalidate the wrong block index and cause the block to be overwritten.

#### 2. Tensor Split Detection and Handling

For tensors that do not wrap around the global CB boundary, each block is allocated contiguously for the tensor, and no special management is needed. For tensors that wrap around the global CB boundary, special pointer management is needed to ensure each core indexes into the global CB correctly. The matmul kernel detects and handles this automatically:

![Matmul 1D Ring Tensor Handling](images/matmul-1d-ring-tensor-handling.png)

**Tensor Split Scenarios:**

The image illustrates two critical scenarios the matmul kernel must handle:

1. **Tensor Split Detected**: When the current read pointer position plus the tensor size exceeds the circular buffer boundary, a split is detected. This triggers wraparound logic to ensure correct block access.

2. **Ring Index Wraparound**: In a 1D ring topology, each core starts at the Nth block, where N is its ring index. When tensor splits occur, the kernel must correctly calculate the start block index, the next block index, and the wrapped read pointer address. For example, core 3 in the diagram first starts at block 3; the next block will require a jump back to the base tensor address, and after processing block 1, it will require wraparound handling to the top of the global CB.

## Performance Benefits

The integrated prefetcher and matmul system provides several key performance advantages:

#### 1. Weight Prefetching
DRAM reads execute ahead of Matmul computation, removing the DRAM bandwidth bottleneck. By streaming data asynchronously, compute cores always have data ready for processing without stalling to wait for memory transfers.

#### 2. High Bandwidth Utilization
- **Coalesced NoC Writes**: Data is transferred in large, contiguous blocks to maximize NoC efficiency
- **Transaction ID Management**: Both DRAM reads and NoC writes use transaction IDs to pipeline operations and saturate available bandwidth
- **Spatial Distribution**: Prefetcher cores are positioned near DRAM banks to minimize NoC congestion

#### 3. Minimal Synchronization Overhead
- **Sender Side**: Only sends credits per block (not per page), reducing synchronization frequency
- **Receiver Side**: Only sends acknowledgments per tensor (not per block), minimizing NoC traffic for flow control
- **Coarse-Grained Coordination**: The credit system allows bulk data transfer while maintaining correctness

#### 4. Asynchronous Execution
- **Sub-Device Isolation**: Prefetcher and compute operations run on separate sub-devices with independent execution tracking
- **Non-Blocking Streaming**: Prefetcher continuously streams data while compute kernels process previous tensors

#### 5. Dynamic Page Size Flexibility
- **Runtime Reconfiguration**: Unlike local CBs which are constrained to a single page size throughout their lifetime, global CBs support dynamic page size changes at runtime
- **Multi-Tensor Support**: Different tensors with varying shapes, data formats (bfloat4_b, bfloat8_b, bfloat16), and tile sizes can be streamed through the same global CB without reallocation

This architecture enables efficient processing of large language models where weight tensors must be continuously streamed from DRAM to distributed compute cores with minimal latency and maximum throughput. The combination of spatial optimization, coarse-grained synchronization, asynchronous execution, and flexible page sizing creates a highly efficient and adaptable data streaming system.

## Future Works

There are several optimizations that can be adopted by the prefetcher:

1. **Skip NoC Counter Update**: Currently we rely on updating `posted` / `non-posted` NoC counters after issuing a NoC write, and flushing the NoC transaction to ensure the current packet has left L1. This can cause dozens of cycles of stalling, and updating the NoC counters also costs a few cycles. One way to work around this issue is to use transaction IDs for the NoC writes, similar to how it is done for DRAM reads. Instead of flushing the current block, we can flush on the transaction ID for the previous block.

1. **Use Stream Registers**: Currently we use L1 for `pages_sent` and `pages_ack`, and accessing L1 has worse latency compared to accessing stream registers. We should be able to switch to stream registers for the send/receive credits.

2. **Pause Prefetching**: Currently the prefetcher will send data whenever there is space available in the global CB, and this can cause NoC congestion if there are other ops using the same NoC, causing a slowdown. It would be beneficial in some cases to pause the prefetcher from consuming the NoC bandwidth and let other ops run faster. A scratch location in L1 or a stream register can be used to send this signal to the prefetcher from other ops, and the prefetcher will enter a while loop until that signal is cleared.

3. **Switch Allocation Strategy**: Currently we only use the aligned allocation strategy, which introduces gaps between the tensors. We should allow switching between compacted allocation and aligned allocation, as in some cases users might want to remove the gaps and handle pointer updates manually.
