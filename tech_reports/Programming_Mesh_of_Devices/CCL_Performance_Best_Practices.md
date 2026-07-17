# CCL Performance Tuning Tips for tt-metal

## 0. Proper Initialization

Prior to running your models and CCL Ops, Fabric is configured and initialized before `mesh_device` instantiation. There are several available Fabric configurations:

- `FABRIC_1D`
- `FABRIC_1D_RING`
- `FABRIC_2D`
- `FABRIC_2D_TORUS_X`
- `FABRIC_2D_TORUS_Y`
- `FABRIC_2D_TORUS_XY`

Generally, `FABRIC_1D_RING` is the best choice for performance. Not all hardware and mesh configurations support ring — the supported Fabric configurations depend on the physical connectivity of the hardware. The `FABRIC_2D` variants are not generally used by `ttnn` CCLs and are necessarily less performant.


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

## 1. Use Trace Mode

In non-trace mode, multi-device Ops are dispatched to their respective devices sequentially. This induces cross-device delay (often called device skew), as later-dispatched devices don't start executing their kernels until they've received them. This skew exists for all Ops on a multi-device machine, however CCL Ops tend to "absorb" the delay because most execute an internal cross-device synchronization — in this case, the kernels don't complete for any device until the furthest-delayed device has completed.

Trace mode does not introduce such delays, since devices execute kernels without needing to wait for dispatch.

## 2. Op-Specific Parameters

There has been effort to auto-detect hardware configurations and hide some of the perf-specific CCL API parameters. CCL Ops that live under the global `ttnn` namespace should handle these details in an optimal way, automatically.

However, there are situations where CCLs under the `experimental` namespace need to be used, and there are a couple of parameters in these Ops that are worth paying attention to:

- **`num_links`** (or similar): set this to the hardware-specific EDM link capability. Rule of thumb: 2 for BH and 4 for WH, except 1 for T3K.
- **`topology`**: specifies the topology-specific algorithm. This should be the best algorithm allowed by the Fabric config (see [0. Proper Initialization](#0-proper-initialization)) — ideally `Ring`.

```python
output_tensor = ttnn.experimental.all_gather_async(
    input_tensor,
    dim=3,
    multi_device_global_semaphore=semaphore,
    num_links=4,
    topology=ttnn.Topology.Ring,
)
```

## 3. Pre-Allocated Buffers

To guarantee correctness in an environment with device skew, CCL Ops must ensure that intra-device buffer destinations are safe to receive data and won't be overwritten by other Ops that may be running out of step with the sender device. To do this, all CCLs by default execute an internal global synchronization prior to executing Op logic, ensuring the destination buffer space is "owned" by the CCL.

Correctness can also be guaranteed by pre-allocating destination buffers at a global scope that won't ever get touched by other Ops. Most CCLs provide optional parameters for passing in externally allocated output and, if necessary, intermediate buffers, as well as semaphores — and will automatically skip the initial sync when these inputs are present. Particularly for small data volumes, skipping this Fabric transaction can save notable time.

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

## 4. Custom Packet Size

By default, Fabric data transactions contain a payload of ~4.25 KB (4352 B — 4 tiles of Bfp8_b). This can be parameterized, up to a hardware-dependent max of ~14.9 KB for BH and ~7.4 KB for WH. Sometimes, adjusting this parameter can improve CCL performance.

**Warning:** this parameter is global, set at init time, and affects every CCL in the model. The performance benefits and detriments of adjusting this parameter are algorithm- and shape-dependent, and help or hurt different CCLs in different ways. It is best to examine overall model CCL perf when adjusting this parameter, rather than focusing on a single Op.

```python
router_config = ttnn.FabricRouterConfig()
router_config.max_packet_payload_size_bytes = 8192  # must be L1-aligned; capped per-arch (WH: 7616B, BH: 15232B)

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router_config)
```

**Example:**

| Configuration | `all_gather` `[1,1,768,256]` |
|---|---|
| Naive | ~54 µs |
| Pre-allocated buffers + 8 KB packet size | ~45 µs |
