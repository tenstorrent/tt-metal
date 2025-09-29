# Prefetcher Technical Report

## Table of Contents
- [Problem with Reading from DRAM](#problem-with-reading-from-dram)
- [Idea of the Prefetcher](#idea-of-the-prefetcher)
- [Sub-Devices: Concept and Usage](#sub-devices-concept-and-usage)
- [Prefetcher Architecture](#prefetcher-architecture)
- [Matmul 1D Ring Integration](#matmul-1d-ring-integration)

## Problem with Reading from DRAM

TODO: Add content describing the challenges and limitations of traditional DRAM reading approaches.

## Idea of the Prefetcher

TODO: Add high-level overview of the prefetcher concept and solution approach.

## Sub-Devices: Concept and Usage

Sub-devices enable partitioning the chip by grouping cores into isolated execution units with independent memory allocators and program execution tracking. This allows the prefetcher to run on dedicated cores while compute kernels execute independently on separate sub-devices, enabling true asynchronous data streaming without blocking the main computation pipeline.

For detailed information about sub-devices, their implementation, and APIs, see [SubDevices.md](../SubDevices/SubDevices.md).

## Prefetcher Architecture

The prefetcher implements a datamovement op that efficiently fetches data from DRAM banks to receiver cores through global circular buffers. The current usage is using prefetcher to prefetch weight tensors for Matmul operations only, it in theory should support prefetch for other ops and prefetch activations as well. The reason we only use it for Matmul is because it has the largest weight tensors and prefetch them gives the most benifit.

### API Overview and Setup Requirements

The prefetcher exposes an api to the users at the TTNN level: `ttnn.dram_prefetcher(...)`

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

1. **Weight Tensors for One Decoder Layer**: In LLMs there are several decoder layers and within each decoder layer there are several Matmul operations that needs to be prefetched, and between different layers the Matmul tensor shapes are the same, we utilize this feature and only pass in the tensors for the first layer to reduce the number of runtime args needed to be passed in to the kernel. These tensors must be width-sharded across DRAM banks and stored in DRAM with tile layout. The prefetcher supports multiple data types (bfloat4_b, bfloat8_b, bfloat16) with different tile sizes.

2. **Address/Configuration Tensor**: A config tensor containing the DRAM buffer addresses for ALL weight tensors across ALL layers. This tensor tells the prefetcher where to find each tensor for every layer, enabling it to fetch data for layers beyond the initial pattern. The tensor is height-sharded across prefetcher cores and stored in L1 for fast access, with each prefetcher core getting the complete address map. Note: even though prefetcher is designed based on the fact each LLM has repeated decoder layers, it doesn't forbid the support of non-repeated decoder layers, or other types of layers. User just need to set num_layers=1 and then pass in all the tensors needed to prefetch.

**Global Circular Buffer Configuration:**
The global circular buffer serves as the communication bridge between prefetcher and consumer kernels. It requires a sender-receiver core mapping that defines which prefetcher cores send data to which consumer cores. The global cb buffer size ideally should buffer at least two tensors to avoid any stall on the consumer side (double buffering), although due to the fact that not all tensors are of the same size, in practice need to buffer more tensors to avoid any stall.

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
The op uses separate sub-devices for prefetcher and compute operations, enabling true asynchronous execution. The prefetcher sub-device contains the sender cores running DRAM reader (issues NoC read to DRAM banks) and L1 writer (issues NoC writes to consumer cores) kernels, while the worker sub-device contains consumer cores running matmul and other operations.

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

- **Matmul Cores (Purple W)**: Regular worker cores for matmul operations. Each prefetcher core issues NoC writes to the two Matmul cores on the right, placing them on the same row to reduce interference with other prefetch cores. For other device architectures we can have one prefetch serving more or less Matmul cores, to maxmize the overall compute utilization.

- **Data Flow Path**: Data flows from DRAM banks → Prefetcher cores → Global Circular Buffers → Compute cores, with each stage operating on different physical regions of the chip.

### Kernel Overview

The prefetcher acts as a **sender** in the global circular buffer setup, utilizing a reader-writer kernel architecture.

1. **DRAM Reader Kernel** (`reader_dram.cpp`)
   - Fetches data from DRAM banks into local triple-buffered circular buffer
   - Each prefetcher core reads tensors from it's assigned DRAM bank, each tensor is read in block by block (the size of a block and number of blocks is based on the Matmul side specification)
   - **Transaction ID Management**: Uses NOC transaction IDs to coordinate DRAM reads with the triple-buffered circular buffer. The kernel maintains `curr_block_trid` to track which transaction ID to use for each block, rotating through IDs 1→2→3→1→... to map reads to different blocks.

   For detailed implementation of DRAM reading optimizations and transaction ID management, see [Saturating_DRAM_bandwidth.md](../Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md).


2. **L1 Writer Kernel** (`writer_l1.cpp`)
   - Transfers data from local CB to global CB (remote consumers)
   - Distributes each tensor block across multiple receiver cores, currently only supports slice the tensor block on width, for one prefetcher core that serves multiple receiver cores, each receiver core gets a slice of the block.

   **Key Operations and Their Purpose:**

   **1. Dynamic CB Reconfiguration (`resize_remote_sender_cb_interface`)**
   ```cpp
   experimental::resize_remote_sender_cb_interface<true>(remote_cb_id, curr_block_size_per_receiver, noc);
   ```
   - This API is called each time before we switch to the next tensor. Since different tensors have different shapes and tile sizes and data formats, this caused each tensor having different block size. The global CB page size needs to be reconfigured for each tensor to match the block size that will be sent to each receiver core.
   - For the detailed implementation, it updates the CB interface page size, adjusts read/write pointers to proper alignment, and if necessary, signals receivers over NOC about the configuration change.

   **2. Flow Control (`remote_cb_reserve_back`)**
   ```cpp
   experimental::remote_cb_reserve_back(remote_cb_id, num_blocks);
   ```
   - Before sending data, the sender must ensure all receiver cores have available space in their portion of the global CB. This prevents overwriting data that receivers have not processed yet.
   - Checks the credit counters (pages_sent vs pages_acked) for each receiver core. Blocks until all receivers have enough free space for the incoming data block. This implements backpressure flow control.

   **3. Data Transfer and Signaling (`remote_cb_push_back_and_write_pages`)**
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
   - This is the core data movement operation that actually transfers the tensor block from the prefetcher's local CB to each receiver core's portion of the global CB.
   - `block_height_in_pages` means the number of pages on the height dim of each tensor block. `coalesced_page_size` here is the pages that grouped together that can form one NoC writes. `coalesced_num_pages_per_row` means the number of NoC writes need to be performed for each row of the block.
   - functionality:
     - Performs coalesced NOC writes to distribute block slices to each receiver core
     - Each receiver gets a slice of the tensor block
     - Updates the sent page counters to signal receivers that new data is available
     - Handles wraparound in the circular buffer automatically


```cpp
// Core prefetcher flow in writer_l1.cpp
for (uint32_t layer = 0; layer < num_layers; layer++) {
    for (uint32_t t = 0; t < num_tensors; t++) {
        // Resize remote CB interface for current tensor
        experimental::resize_remote_sender_cb_interface<true>(
            remote_cb_id, curr_block_size_per_receiver, noc);

        for (uint32_t block = 0; block < num_blocks; ++block) {
            // Wait for local data from DRAM reader
            cb_wait_front(local_cb_id, max_block_num_tiles);

            // Reserve space on all receiver cores
            experimental::remote_cb_reserve_back(remote_cb_id, 1);

            // Push data to receivers with coalesced writes
            experimental::remote_cb_push_back_and_write_pages<skip_ptr_update>(
                remote_cb_id, local_cb_addr, 1,
                curr_block_height_in_tiles, curr_coalesced_num_pages,
                curr_coalesced_page_size, noc);

            // pop out local CB, signal reader core buffer is empty
            cb_pop_front(local_cb_id, max_block_num_tiles);
        }
    }
}
```

### Inside of Global CB
This section describes the kernel side implementation of global CB, config structures, data transfer, synchronization, etc.

#### 1. Global Circular Buffer Initialization

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
uint32_t remote_noc_xy_addr = l1_remote_cb_config_addr[5];      // Address containing NOC coordinates
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
sender_cb_interface.receiver_noc_xy_ptr = remote_noc_xy_addr;   // Receiver NOC coordinates
sender_cb_interface.aligned_pages_sent_ptr = aligned_pages_sent_addr; // Sent page counters
sender_cb_interface.num_receivers = num_receivers;             // Number of receivers
```

**For Receiver Cores (Matmul):**
```cpp
RemoteReceiverCBInterface& receiver_cb_interface = get_remote_receiver_cb_interface(cb_id);
receiver_cb_interface.config_ptr = config_addr;                 // Pointer to L1 config
receiver_cb_interface.fifo_start_addr = fifo_start_addr;       // Buffer start address
receiver_cb_interface.fifo_rd_ptr = fifo_ptr;                  // Read pointer
receiver_cb_interface.sender_noc_x = sender_noc_x;             // Sender NOC coordinates
receiver_cb_interface.sender_noc_y = sender_noc_y;
receiver_cb_interface.aligned_pages_acked_ptr = aligned_pages_acked_addr; // Acked page counters
```

**Key Initialization Steps:**
1. **Configuration Reading**: Each core reads its role (sender/receiver) and parameters from L1
2. **Interface Setup**: Appropriate CB interface structure is populated based on core role
3. **NOC Coordinate Mapping**: NOC coordinates are extracted for remote communication
4. **Synchronization Setup**: Page counter addresses are configured for flow control
5. **Initial Sizing**: The CB interface is sized with initial page size via `resize_remote_*_cb_interface`

**Config Pointer**
The `config_ptr` stored in both `RemoteSenderCBInterface` and `RemoteReceiverCBInterface` serves a critical runtime purpose: we need to store the latest `fifo_wr_ptr` / `fifo_rd_ptr` value back to L1 memory so that the next op can read it back to initilize the local `RemoteSenderCBInterface` / `RemoteReceiverCBInterface`.


#### 2. Dynamic Page Size Reconfiguration

On the consumer side, ops are required to inplace a local CB on top of the remote CB for local pointer update and consume different part of the tensor. Due to the fact that each CB are required to be page-size aligned, the local CB size will not be the same as global CB, but a size that is aligned to the current page size. Ideally, consumer op will call cb_pop_front to get to the tensor blocks incrementally, and not worrying about manual pointer updates.

When switching between tensors with different shapes or data formats, the global CB page size must be reconfigured to the new tensor block size. This process involves page alignment operations to ensure correct wraparound behavior when consumer performs the local CB pop.

![Global CB Tensor Allocation](images/global-cb-tensor-alloc.png)

**The Problem Without Alignment:**
If tensors are stored at non-aligned locations in the circular buffer, when the buffer wraps around to the top, a simple `cb_pop_front` operation on the consumer side will not land at the correct tensor boundary. The consumer would read from the middle of a tensor or across tensor boundaries, corrupting the data.

**How It Works:**
1. **Before New Tensor**: The write pointer may be at an arbitrary location after the previous tensor
2. **Alignment Step**: The pointer is advanced to the next page-aligned boundary that matches the new tensor's page size
3. **Tensor Storage**: The new tensor is stored starting at this aligned location
4. **Wraparound Guarantee**: When the buffer wraps around, the consumer's `cb_pop_front` operations will correctly land at tensor boundaries

**Consumer Benefit:**
With page alignment, the consumer (matmul core) can perform simple circular buffer operations without worrying about tensor boundaries. Each `cb_pop_front` operation moves exactly one "page" (tensor block), and wraparound automatically lands at the start of the next tensor.

Note: when consumer does not rely on local CB pop, but manually update read pointer then this will not be a problem and we do not need the extra page alignment feature. This is the case for the current Matmul implementation, where the compute kernel manually updates read pointers, so the feature is not required and can be disabled. However, when trying that on the LLama-70b model, the performance regressed by 1-2 token/second, and thus this feature is kept enabled. The potential cause could be without alignement, less synchronization is needed and prefetcher runs faster, however, faster prefetcher can interfere with other ops more and caused the slow down on other ops.


## 1D Ring Matmul Integration

The matmul operation serves as the consumer side of the global circular buffer, implementing a ring-based data processing pattern for efficient multi-core execution. This section will focus on the global CB implementation part and the integration with prefetcher.

### Overview:

Matmul cores act as **receivers** in the global CB system, consuming data streamed by the prefetcher:


### Mechanism: Advanced Circular Buffer Management

#### 1. Tensor Split Detection and Handling

Large tensors may wrap around the circular buffer boundary. The matmul kernel detects and handles this automatically:

```cpp
FORCE_INLINE bool is_tensor_split(uint32_t cb_id, uint32_t tensor_size_bytes) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t fifo_limit = local_cb.fifo_limit;
    // Check if tensor exceeds remaining buffer space
    bool split = (fifo_limit - fifo_rd_ptr) < tensor_size_bytes / L1_ALIGNMENT;
    return split;
}
```

When a tensor split is detected, the kernel handles wraparound correctly:

```cpp
FORCE_INLINE void calculate_next_block_index_and_update_rd_ptr(
    uint32_t cb_id, uint32_t num_blocks, uint32_t block_size_bytes,
    uint32_t curr_block_index, uint32_t cb_start_addr, uint32_t rd_ptr_start_addr,
    bool tensor_split, uint32_t* updated_block_index, uint32_t* updated_rd_ptr) {

    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t next_block_index = curr_block_index + 1;
    bool reach_limit = local_cb.fifo_rd_ptr == local_cb.fifo_limit;
    bool last_block = curr_block_index == (num_blocks - 1);

    if (tensor_split) {
        if (reach_limit) {
            local_cb.fifo_rd_ptr = cb_start_addr;  // Wrap to beginning
            if (last_block) {
                next_block_index = 0;
                next_fifo_rd_ptr = rd_ptr_start_addr;
            } else {
                next_fifo_rd_ptr = cb_start_addr + block_size_bytes_aligned;
            }
        }
        // ... additional wraparound logic
    }
}
```

#### 2. Ring Index Positioning

Each matmul core in the 1D ring processes different blocks. The kernel automatically positions the read pointer to the correct starting location:

```cpp
FORCE_INLINE void update_rd_ptr_to_ring_index(
    uint32_t cb_id, uint32_t block_size_bytes, uint32_t ring_index, bool tensor_split) {

    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);

    if (tensor_split) {
        // Handle wraparound for ring positioning
        if ((local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT) >= local_cb.fifo_limit) {
            uint32_t fifo_size = local_cb.fifo_size;
            uint32_t fifo_limit = local_cb.fifo_limit;
            uint32_t fifo_start_addr = fifo_limit - fifo_size;
            uint32_t fifo_size_skip_bytes = local_cb.fifo_rd_ptr - fifo_start_addr;

            // Calculate wrapped position using modulo arithmetic
            local_cb.fifo_rd_ptr = fifo_start_addr +
                (fifo_size_skip_bytes + ring_index * block_size_bytes / L1_ALIGNMENT) % local_cb.fifo_size;
        } else {
            local_cb.fifo_rd_ptr += ring_index * block_size_bytes / L1_ALIGNMENT;
        }
    } else {
        // Simple offset for non-split tensors
        local_cb.fifo_rd_ptr += ring_index * block_size_bytes / L1_ALIGNMENT;
    }
}
```

#### 3. Compute Kernel Integration

The compute kernel manages CB state throughout matmul execution:

```cpp
// From bmm_large_block_zm_fused_bias_activation_gathered.cpp
#ifdef ENABLE_GLOBAL_CB
    uint32_t in1_cb_start_addr = 0;
    uint32_t in1_rd_ptr_start_addr = 0;
    uint32_t curr_in1_block_index = 0;
    bool in1_tensor_split = 0;

    // Initialize CB interface state
    UNPACK((in1_cb_start_addr = get_local_cb_start_addr(in1_cb_id)));
    UNPACK((in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id)));
    UNPACK((curr_in1_block_index = ring_idx));
    UNPACK((in1_tensor_split = is_tensor_split(in1_cb_id, in1_tensor_size_bytes)));

    // Position to correct ring index
    UNPACK((update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_idx, in1_tensor_split)));
#endif

for (uint32_t block = 0; block < num_blocks; block++) {
    const uint32_t curr_ring_idx = (ring_idx + block) % ring_size;

    // Wait for data from global CB
    if constexpr (in1_is_dram) {
        cb_wait_front(in1_cb_id, in1_block_num_tiles);
    }

    #ifdef ENABLE_GLOBAL_CB
        // Calculate next block position in ring
        UNPACK((calculate_next_block_index_and_update_rd_ptr(
            in1_cb_id, num_blocks, in1_block_size_bytes,
            curr_in1_block_index, in1_cb_start_addr, in1_rd_ptr_start_addr,
            in1_tensor_split, &next_in1_block_index, &next_in1_rd_ptr_addr)));
    #endif

    // Perform matmul computation
    // ... computation logic ...

    #ifdef ENABLE_GLOBAL_CB
        // Update state for next iteration
        curr_in1_block_index = next_in1_block_index;
        UNPACK((update_local_cb_rd_ptr(in1_cb_id, next_in1_rd_ptr_addr)));
    #endif
}

#ifdef ENABLE_GLOBAL_CB
    // Reset for next tensor in sequence
    UNPACK((update_local_cb_rd_ptr(in1_cb_id, in1_rd_ptr_start_addr)));
    UNPACK((update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_size, in1_tensor_split)));
#endif
```

#### 4. Synchronization and Flow Control

The matmul system implements careful synchronization to ensure data availability and proper flow control:

```cpp
// Receiver-side flow control
FORCE_INLINE void remote_cb_wait_front(uint32_t cb_id, uint32_t num_pages) {
    RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(cb_id);
    uint32_t len_bytes = num_pages * remote_cb.fifo_page_size;
    uint32_t num_pages_wait = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;

    // Wait for sender (prefetcher) to provide enough data
    volatile tt_l1_ptr uint32_t* pages_acked_ptr = /* ... */;
    volatile tt_l1_ptr uint32_t* pages_sent_ptr = /* ... */;

    do {
        invalidate_l1_cache();
        pages_acked = *pages_acked_ptr;
        pages_sent = *pages_sent_ptr;
        num_pages_recv = pages_sent - pages_acked;
    } while (num_pages_recv < num_pages_wait);
}

// Signal completion to sender
FORCE_INLINE void remote_cb_pop_front(uint32_t cb_id, uint32_t num_pages, uint8_t noc) {
    RemoteReceiverCBInterface& remote_cb = get_remote_receiver_cb_interface(cb_id);
    // ... update read pointer ...

    uint32_t num_aligned_pages = len_bytes / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
    detail::update_pages_acked(remote_cb, num_aligned_pages, noc, false, write_at_cmd_buf);
}
```

### Performance Benefits

The prefetcher + matmul integration provides several key performance advantages:

1. **Weight Prefetching**: DRAM reads ahead of Matmul execution, removing the DRAM bandwidth limitation
2. **High Bandwidth Utilization**: Coalesced NOC writes maximize memory bandwidth
3. **Minimal Synchronization**: Only sync per tensor block
4. **Persistent State**: Continuous streaming across multiple program executions

This architecture enables efficient processing of large language models where weight tensors must be continuously streamed from DRAM to distributed compute cores with minimal latency and maximum throughput.
