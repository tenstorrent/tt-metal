# CCL Performance Tuning Tips

The [CCL bandwidth report](./CCL_bandwidth.md) measures what `ttnn` collectives achieve out of the box, and explains what sets their ceiling and floor. This report covers the settings that remain in the caller's hands: how Fabric is configured, how Ops are dispatched, and what can be tuned.

## Fabric Configuration

Fabric is configured and initialized before `mesh_device` instantiation. The configurations are:

- `FABRIC_1D`
- `FABRIC_1D_RING`
- `FABRIC_2D`
- `FABRIC_2D_TORUS_X`
- `FABRIC_2D_TORUS_Y`
- `FABRIC_2D_TORUS_XY`

Choose a ring or torus configuration only along axes with wraparound links. Elsewhere it adds overhead to every packet with no benefit (see [What a ring costs](./CCL_bandwidth.md#what-a-ring-costs)).

A 2D mesh does not need a 2D Fabric. The 1D configurations build an independent line or ring along every row and every column, so CCLs can run on either axis. Choose a `FABRIC_2D` variant when traffic must turn between rows and columns, reaching devices off its own row or column.

```python
import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
```

Equivalently, via the `device_params` pytest fixture:

```python
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
```

## Use Trace Mode

In non-trace mode, multi-device Ops are dispatched to their respective devices sequentially. This induces cross-device delay (often called device skew), as later-dispatched devices don't start executing their kernels until they've received them. This skew exists for all Ops on a multi-device machine, however CCL Ops tend to "absorb" the delay because most execute an internal cross-device synchronization — in this case, the kernels don't complete for any device until the furthest-delayed device has completed (see [Per-invocation cost](./CCL_bandwidth.md#per-invocation-cost)).

Trace mode does not introduce such delays, since devices execute kernels without needing to wait for dispatch. See [Metal Trace](../AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md#1-metal-trace) for how to capture and replay a trace.

## Op-Specific Parameters

CCL Ops choose link count, topology and their internal settings automatically, tuned for the hardware. Leave them unset. The tuning parameters (`chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`) are experimental. You can try them to see if you do better for your shapes, but the defaults are the recommended path.

Some Ops under the `experimental` namespace take `num_links` and `topology`:

- **`num_links`**: the number of links per direction. If you set it, a rule of thumb is 2 for BH and 4 for WH, except 1 for T3K. The weakest hop on the axis sets the count for the whole collective, and it can differ between the axes of one system (see [Links available](./CCL_bandwidth.md#links-available)).
- **`topology`**: specifies the topology-specific algorithm. This should be the best algorithm allowed by the Fabric config (see [Fabric Configuration](#fabric-configuration)): `Ring` where the axis closes, `Linear` otherwise.

```python
output_tensor = ttnn.experimental.all_gather_async(
    input_tensor,
    dim=3,
    multi_device_global_semaphore=semaphore,
    num_links=4,
    topology=ttnn.Topology.Ring,
)
```

## Pre-Allocated Buffers

To guarantee correctness in an environment with device skew, CCL Ops must ensure that intra-device buffer destinations are safe to receive data and won't be overwritten by other Ops that may be running out of step with the sender device. To do this, all CCLs by default execute an internal global synchronization prior to executing Op logic, ensuring the destination buffer space is "owned" by the CCL.

Correctness can also be guaranteed by pre-allocating destination buffers at a global scope that won't ever get touched by other Ops. Most `experimental` CCLs provide optional parameters for passing in externally allocated output and, if necessary, intermediate buffers, as well as semaphores — and will automatically skip the initial sync when these inputs are present. Particularly for small data volumes, skipping this Fabric transaction can save notable time. Skipping it is safe only if the caller guarantees a buffer isn't reused while a peer is still reading it. One way to guarantee this is the round-robin pool below: with enough buffers in rotation, a device running ahead writes into a different buffer than the one its peers are still reading.

```python
# Pre-allocate a pool of semaphores + intermediate buffers, then round-robin reuse them across iterations
num_buffers = 8
semaphores = [ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0) for _ in range(num_buffers)]
intermediate_tensors = [
    ttnn.from_torch(
        torch.zeros(intermediate_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=intermediate_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
    )
    for _ in range(num_buffers)
]

out = ttnn.experimental.all_reduce_async(
    tt_input,
    intermediate_tensors[i % num_buffers],
    cluster_axis=cluster_axis,
    mesh_device=mesh_device,
    multi_device_global_semaphore=semaphores[i % num_buffers],
    memory_config=output_mem_config,
    topology=topology,
    num_links=num_links,
    subdevice_id=worker_sub_device_id,
)
```

`all_gather_async` similarly accepts a `persistent_output_buffer` argument directly:

```python
output_tensor = ttnn.experimental.all_gather_async(
    input_tensor,
    persistent_output_buffer=output_tensor,  # reused across iterations
    dim=3,
    multi_device_global_semaphore=semaphore,
    num_links=4,
    topology=ttnn.Topology.Ring,
)
```

## Custom Packet Size

By default, Fabric data transactions contain a payload of 4352 B (4 tiles of Bfp8_b). This can be parameterized, up to a hardware-dependent max of 15232 B for BH and 7616 B for WH. Sometimes, adjusting this parameter can improve CCL performance.

**Warning:** this parameter is global, set at init time, and affects every CCL in the model. A larger payload leaves less room for buffer slots in each router, so the maximum is not always the fastest choice (see [Packets and slots](./CCL_bandwidth.md#packets-and-slots)). The performance benefits and detriments of adjusting this parameter are algorithm- and shape-dependent, and help or hurt different CCLs in different ways. It is best to examine overall model CCL perf when adjusting this parameter, rather than focusing on a single Op.

```python
# Blackhole
router_config = ttnn.FabricRouterConfig()
router_config.max_packet_payload_size_bytes = 8192  # must be L1-aligned; capped per-arch (WH: 7616 B, BH: 15232 B)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router_config)
```

**Example (Blackhole):**

| Configuration | `all_gather` `[1,1,768,256]` |
|---|---|
| Naive | ~54 µs |
| Pre-allocated buffers + 8192 B packet size | ~45 µs |
