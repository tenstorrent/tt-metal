# Distributed Tensor

APIs and constructs for Runtime Tensors whose data is placed across multiple
`MeshCoordinate`s on a `MeshDevice`.

## Background

These pieces were split out of `ttnn::Tensor` during Runtime Tensor lowering /
graduation so core host/device tensor APIs can graduate without taking on the
full distributed design.

Some ops and models already depend on the functionality in this folder, but
distributed tensor is still experimental and needs more design review.

## Current distributed tensor behavior in Runtime Tensor

Runtime Tensors are built on inherently distributed constructs (`MeshBuffer`
for `MeshDevice`, `DistributedHostBuffer` for `HostTensor`), so multi-device
capabilities are already present at the lower level. What remains are the
distributed-tensor-specific APIs and abstractions.

Limited H<->D data movement is supported in Runtime Tensor when the transfer is
"uniform"; see `is_uniform_write` in
[`distributed_tensor_apis.hpp`](distributed_tensor_apis.hpp) for more details.

## Main classes of distributed tensor functionality

Distributed tensor is mainly about expressing replication / sharding across
devices (`MeshCoordinate`s on a `MeshDevice`). That breaks into:

1. **Cross-device sharding: Splicing** — splitting the original data among
   multiple devices; currently handled by TTNN's distributed tensor, and will
   likely need further lowering.
2. **Cross-device sharding: Tracking** — recording that splicing / replication;
   currently `TensorTopology`. Note that topology is handled inconsistently
   across ops and tensor infrastructure; see
   [#41727](https://github.com/tenstorrent/tt-metal/issues/41727).
3. **Distributed H<->D transfer** — read/write distributed tensors according to
   the sharding / replication rules, including partial updates. The current
   Runtime Tensor implementation is restrictive, and what is provided here in
   [`distributed_tensor_apis.hpp`](distributed_tensor_apis.hpp) is
   underspecified.
4. **Host-side tensor transforms** — Runtime Tensor provides host transforms,
   but none process multi-device host data or respect sharding / replication
   rules.
