# Welcome
Thanks for participating in the Fabric/Multichip Programming Hackathon!

You chose the ops on fabric track (if not you may be lost :) )

This hackathon will be good for you if you are interested in writing multidevice ops
and programs!


# APIs
To make use of the fabric for your multi-device operation, you will use some or all of
the following APIs.

## Async Unicast Writes
- Unicast over the fabric and to a single NoC endpoint
```
void fabric_async_write(uint32_t routing,   // the network plane to use for this transaction
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size  // number of bytes to write to remote destination
)
```

## Async Multicast Writes
- multicast over the fabric to target multiple devices with a single NoC endpoint on
  within each targeted device
```
void fabric_async_write_multicast(
    uint32_t routing_plane,  // the network plane to use for this transaction
    uint32_t src_addr,       // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint16_t e_depth,
    uint16_t w_depth,
    uint16_t n_depth,
    uint16_t s_depth)
```

## Atomic Unicast Writes
- Unicast of the fabric to perform a noc atomic increment on a single NoC endpoint
```
void fabric_atomic_inc(
    uint32_t routing,   // the network plane to use for this transaction
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_addr,
    uint32_t atomic_inc,
    uint32_t wrap_boundary)
```

## Fused Async Write + Atomic Unicast Writes
- Unicast over the fabric and to a single NoC endpoint and perform a noc atomic increment on a single NoC endpoint
```
fabric_async_write_atomic_inc(
    uint32_t routing,   // the network plane to use for this transaction
    uint32_t src_addr,  // source address in sender’s memory
    uint16_t dst_mesh_id,
    uint16_t dst_dev_id,
    uint64_t dst_write_addr,
    uint64_t dst_atomic_addr,
    uint32_t size,  // number of bytes to write to remote destination
    uint32_t atomic_inc)
```

# Op Ideas
Feel free to build your own custom op but if you are looking for some ideas, take a
look here:

- Scatter(split) a tensor/buffer from a source device to multiple target devices
  - Each device gets a different piece of the input buffer

- Gather (combine) a tensor/buffer from multiple devices into a single buffer
  - Each device contributes a different piece of the output buffer

- Broadcast a tensor/buffer from a source device to all other devices
  - Each device gets a copy of the input buffer

- Implement a streaming operation from one device to another, with custom end-to-end flow
  control implemented by the user kernel(s)
