# Hybrid Tensor and Data Parallelism Implementation

This short guide explains how to add hybrid tensor and data parallelism to your model using submesh tiling across a larger mesh.

## Overview of Changes

The main changes involve:

1. Creating multiple submeshes from the main mesh
2. Running the model on each submesh
3. Capturing and replaying a trace across all submeshes in parallel

## Key Implementation Details

### 1. Submesh Creation

```python
    # Work with submesh device as you would with a regular ttnn.MeshDevice
    submesh_devices: List[ttnn.MeshDevice] = mesh_device.create_submeshes((2, 4), ttnn.MeshType.Ring)
```

### 2. Compile & Run the Model on Each Submesh

```python
    # Run the model on each submesh
    for submesh_device in submesh_devices:
        model(..., device=submesh_device)
```

### 3. Capture & Replay the Trace

```python

    # Capture Model Trace spanning all submeshes
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for submesh_device in submesh_devices:
        model(..., device=submesh) # Run the model on each submesh
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    # Execute Model Trace across all submeshes in parallel
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)

```
