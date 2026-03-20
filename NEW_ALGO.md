## core algorithm
new algo in tree_reduce_scatter_reader/writer/compute.cpp

only applies to ring topology.

from the point of view of device[x]:
```
step 0: D[x-3] -> D[x-2]    D[x-1] -> D[x]    D[x+1] <- D[x+2]    D[x+3] <- D[x+4]
step 1: D[x-3]    D[x-2] -----------> D[x]    D[x+1] <----------- D[x+3]    D[x+4]
step 2: D[x-3]    D[x-2]    D[x-1]    D[x] <- D[x+1]    D[x+2]    D[x+3]    D[x+4]
```

at each step, one device sends to neighbor, and receiving device waits and accumulates with its local slice.

the data shown above is just the slice of the tensor that device[x] needs.

you'll need to generalize this for any ring size (both odd and even sizes).

also, this procedure happens for every device in the ring. so the loop (within a single device) likely looks like:
```
for each step:
  work for device[x-3]: send tensor slice to neighbor device / receive and accumulate / do nothing
  work for device[x-2]: send tensor slice to neighbor device / receive and accumulate / do nothing
  work for device[x-1]: ...
  work for device[x] (this device): ...
  work for device[x+1]: ...
  work for device[x+2]: ...
  work for device[x+3]: ...
```

for every tensor slice perform temporary accumulation in intermediate_tensor, except for the one slice that should permanently accumulate on this device's output_tensor.

## design opens:
- what does semaphore syncs look like? how many semaphores should be created (minimize this)?

lets brainstorm together and keep iterating and refining on this.

## code style:
- keep loop structure clean and well commented.
- use sensible variable names
- assume a 4D tensor, the scatter dim can be anything


----------

Odd ring size:
```
step 0: D[x-4] -> D[x-3]    D[x-2] -> D[x-1]    D[x]    D[x+1] <- D[x+2]    D[x+3] <- D[x+4]
step 1: D[x-4]    D[x-3] -----------> D[x-1]    D[x]    D[x+1] <----------- D[x+3]    D[x+4]
step 2: D[x-4]    D[x-3]    D[x-2]    D[x-1] -> D[x] <- D[x+1]    D[x+2]    D[x+3]    D[x+4]
```

---

## Implementation Notes

### Files Created
- `tree_reduce_scatter_reader.cpp` - reads local/accumulated data + received data into CBs
- `tree_reduce_scatter_writer.cpp` - sends data to neighbors + stores reduced results
- `tree_reduce_scatter_compute.cpp` - performs tile addition

### Key Design Decisions

#### 1. Offset-based action determination
```cpp
int32_t offset = compute_offset(my_chip_id, slice, ring_size);
// offset in range [-(ring_size-1)/2, ring_size/2]
// Negative/zero offsets = left tree, positive offsets = right tree
```

#### 2. Unified formula for even/odd ring sizes
- Left tree root: `0` for even, `-1` for odd
- Right tree root: always `+1`
- Sender/receiver patterns use adjusted offsets relative to tree roots

#### 3. Double-buffered intermediate tensor
```
intermediate_tensor layout:
[buffer_0: slice_0, slice_1, ..., slice_{N-1}]
[buffer_1: slice_0, slice_1, ..., slice_{N-1}]

Step s uses:
- recv_buffer = step % 2       (where incoming data lands)
- accum_buffer = (step-1) % 2  (where previous accumulated result is)
```

#### 4. Semaphore scheme (2 semaphores per device)
- `forward_sem`: incremented when data arrives FROM device+1
- `backward_sem`: incremented when data arrives FROM device-1
- Receiver waits on appropriate semaphore before reading

### Completed Features

1. **TensorAccessor**: Used for input, intermediate, and output tensors (works with both interleaved and sharded)

2. **Fabric APIs**: Proper use of:
   - `FabricConnectionManager` for connection setup
   - `PacketHeaderPool::allocate_header()` for packet headers
   - `ccl_routing_utils::fabric_set_line_unicast_route()` for routing setup
   - `fabric_unicast_noc_unicast_write_with_state()` for data sends
   - `fabric_unicast_noc_unicast_atomic_inc_with_state()` for semaphore signals

3. **Scatter dimension support**: Compile-time `dim` parameter (1=C, 2=H, 3=W) with proper tile indexing

4. **Double-buffered intermediate**: `recv_buffer_idx = step % 2`, `accum_buffer_idx = (step-1) % 2`

### Remaining TODOs

#### 1. Odd ring_size final step: dual receive
At the final step for odd N, offset 0 receives from BOTH -1 and +1:
```cpp
// Current implementation only handles even ring_size
// For odd: need two sequential receives and reduces at final step
// Or implement 3-way add
```

#### 2. Host-side program factory
Need to create `tree_reduce_scatter_program_factory.hpp` with:
- CB allocation
- Kernel compilation with proper defines
- Runtime args setup
- Semaphore creation

### Verification checklist
- [ ] Even ring_size (8): trace D4's actions for all slices at each step
- [ ] Odd ring_size (9): trace D4's actions, especially final step
- [ ] Semaphore counts match expected receives
- [ ] Double-buffer indices don't cause read-after-write hazards
- [ ] Fabric routing info is correctly configured for bidirectional sends
