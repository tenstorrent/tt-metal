# Distributed Programming Examples

This directory contains examples of distributed programming model using the TT-Metalium API.

They are intended to be simple demonstrations for distributed program dispatch, distributed memory management, and end-to-end distributed program execution.

Users familiar with the single-device TT-Metal programming model will find the distributed programming model to be a natural extension.

MeshBuffer allocations use `ReplicatedBufferConfig` (the same local size on every device). Per-device host contents are owned by `DistributedHostBuffer` and transferred with `MeshCommandQueue::enqueue_write` / `enqueue_read`.
