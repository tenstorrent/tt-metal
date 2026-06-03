# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python correctness sweeps for ttnn.D2HStreamService."""

import pytest
import torch

import ttnn

_DTYPE_TORCH = torch.int32
_DTYPE_TTNN = ttnn.uint32
_DTYPE_SIZE = 4
_IO_LOOPS = 10
_RANDINT_HIGH = 2**31


def _make_global_spec(shape: ttnn.Shape) -> ttnn.TensorSpec:
    return ttnn.TensorSpec(
        shape=shape,
        dtype=_DTYPE_TTNN,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _run_io_loop(
    service: ttnn.D2HStreamService,
    iter_mapper: ttnn.CppTensorToMesh,
    global_spec: ttnn.TensorSpec,
    shape_list: list,
    input_path: str,
    mesh_device,
    num_iters: int = _IO_LOOPS,
) -> None:
    # Drain the kernel's initial auto-iteration.
    service.read_from_tensor_bytes()
    service.barrier()

    for i in range(num_iters):
        # on each iteration: generate a random source tensor,
        # copy it to backing tensor on device, read_from_tensor() (tensor or bytes) to host,
        # barrier, assert readback matches expected host tensor
        gen = torch.Generator()
        gen.manual_seed(i)
        src_torch = torch.randint(low=0, high=_RANDINT_HIGH, size=shape_list, dtype=_DTYPE_TORCH, generator=gen)
        expected_host = ttnn.from_torch(src_torch, spec=global_spec, mesh_mapper=iter_mapper)
        ttnn.copy_host_to_device_tensor(expected_host, service.get_backing_tensor())
        ttnn.synchronize_device(mesh_device)

        if input_path == "tensor":
            read_host = ttnn.from_torch(
                torch.zeros(shape_list, dtype=_DTYPE_TORCH),
                spec=global_spec,
                mesh_mapper=iter_mapper,
            )
            service.read_from_tensor(read_host)
            service.barrier()
            _verify_readback(service, expected_host, read_host)
        else:
            raw = service.read_from_tensor_bytes()
            service.barrier()
            read_vec = torch.frombuffer(raw, dtype=torch.int32).clone()
            expected_vec = src_torch.view(-1)
            assert torch.equal(read_vec.to(torch.int64), expected_vec.to(torch.int64)), f"iter {i}: bytes mismatch"


def _verify_readback(service, expected_host, actual_host) -> None:
    expected_subs = ttnn.get_device_tensors(expected_host)
    actual_subs = ttnn.get_device_tensors(actual_host)
    assert len(actual_subs) == len(expected_subs)
    for i, (actual, expected) in enumerate(zip(actual_subs, expected_subs)):
        a_t = ttnn.to_torch(actual).view(-1).to(torch.int64)
        e_t = ttnn.to_torch(expected).view(-1).to(torch.int64)
        assert torch.equal(a_t, e_t), f"device {i}: contents mismatch"


@pytest.mark.parametrize(
    "shape_list, scratch_cb_pages, fifo_pages",
    [
        ([1, 1, 1, 640], 1, 1),
        ([1, 1, 16, 640], 4, 16),
        ([1, 1, 7, 640], 4, 8),
    ],
)
@pytest.mark.parametrize("input_path", ["tensor", "bytes"])
def test_d2h_stream_service_replicated_sweep(
    mesh_device,
    shape_list,
    scratch_cb_pages,
    fifo_pages,
    input_path,
):
    shape = ttnn.Shape(shape_list)
    per_row_bytes = shape_list[-1] * _DTYPE_SIZE
    global_spec = _make_global_spec(shape)

    placements = [ttnn.PlacementReplicate() for _ in range(mesh_device.shape.dims())]
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(placements=placements))

    service = ttnn.D2HStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=fifo_pages * per_row_bytes,
        scratch_cb_size_bytes=scratch_cb_pages * per_row_bytes,
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, input_path, mesh_device)
