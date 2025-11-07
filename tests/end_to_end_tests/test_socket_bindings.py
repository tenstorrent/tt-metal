# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Socket Python Bindings End-to-End Tests

Tests for actual socket send/receive operations using multiprocessing.
"""

import pytest
import torch
import ttnn
import multiprocessing as mp
import os
import sys


def sender_process(recv_ready_read, send_done_write, tensor_queue, tensor_shape, buffer_type, dtype):
    """Sender process that sends a tensor over socket"""
    try:
        # Set MPI rank for this process
        os.environ["OMPI_COMM_WORLD_RANK"] = "0"
        os.environ["OMPI_COMM_WORLD_SIZE"] = "2"

        # Wait for receiver to be ready (read from pipe)
        os.read(recv_ready_read, 1)

        # Initialize device
        device = ttnn.CreateDevice(device_id=0)
        ttnn.SetDefaultDevice(device)

        # Map ttnn dtype to torch dtype
        torch_dtype_map = {
            ttnn.float32: torch.float32,
            ttnn.bfloat16: torch.bfloat16,
            ttnn.bfloat8_b: torch.float32,
            ttnn.bfloat4_b: torch.float32,
            ttnn.int32: torch.int32,
            ttnn.uint32: torch.uint32,
        }
        torch_dtype = torch_dtype_map.get(dtype, torch.float32)

        # Create test tensor
        torch_input = torch.randn(tensor_shape, dtype=torch_dtype)

        # Convert to device tensor
        input_tensor = ttnn.from_torch(
            torch_input,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(buffer_type=buffer_type),
        )

        # Create socket configuration
        sender_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
        receiver_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1))

        socket_connection = ttnn.SocketConnection(sender_core, receiver_core)
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=buffer_type,
            fifo_size=1024,
        )
        socket_config = ttnn.SocketConfig([socket_connection], socket_mem_config)

        # Send tensor
        ttnn.experimental.send_async(input_tensor, device, socket_config)
        ttnn.synchronize_device(device)

        # Send tensor data via queue for verification
        tensor_queue.put(torch_input.cpu().numpy())

        # Signal completion (write to pipe)
        os.write(send_done_write, b"1")
        os.close(send_done_write)

        ttnn.close_device(device)
        return 0

    except Exception as e:
        print(f"Sender error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def receiver_process(send_ready_write, recv_done_read, tensor_queue, tensor_shape, buffer_type, dtype):
    """Receiver process that receives a tensor over socket"""
    try:
        # Set MPI rank for this process
        os.environ["OMPI_COMM_WORLD_RANK"] = "1"
        os.environ["OMPI_COMM_WORLD_SIZE"] = "2"

        # Initialize device
        device = ttnn.CreateDevice(device_id=0)
        ttnn.SetDefaultDevice(device)

        # Map ttnn dtype to torch dtype
        torch_dtype_map = {
            ttnn.float32: torch.float32,
            ttnn.bfloat16: torch.bfloat16,
            ttnn.bfloat8_b: torch.float32,
            ttnn.bfloat4_b: torch.float32,
            ttnn.int32: torch.int32,
            ttnn.uint32: torch.uint32,
        }
        torch_dtype = torch_dtype_map.get(dtype, torch.float32)

        # Create dummy tensor to receive into
        dummy_tensor = torch.zeros(tensor_shape, dtype=torch_dtype)

        output_tensor = ttnn.from_torch(
            dummy_tensor,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.MemoryConfig(buffer_type=buffer_type),
        )

        # Create socket configuration (must match sender)
        sender_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
        receiver_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1))

        socket_connection = ttnn.SocketConnection(sender_core, receiver_core)
        socket_mem_config = ttnn.SocketMemoryConfig(
            socket_storage_type=buffer_type,
            fifo_size=1024,
        )
        socket_config = ttnn.SocketConfig([socket_connection], socket_mem_config)

        # Signal that we're ready (write to pipe)
        os.write(send_ready_write, b"1")
        os.close(send_ready_write)

        # Receive tensor
        ttnn.experimental.recv_async(output_tensor, device, socket_config)
        ttnn.synchronize_device(device)

        # Convert back to torch
        received_tensor = ttnn.to_torch(output_tensor)

        # Get expected data from queue
        expected_tensor = torch.from_numpy(tensor_queue.get(timeout=10))

        # Verify tensor integrity
        assert received_tensor.shape == expected_tensor.shape, f"Shape mismatch"

        # For integer types, verify exact match; for floats, use tolerance
        if dtype in (ttnn.int32, ttnn.uint32):
            assert torch.equal(received_tensor, expected_tensor), f"Values mismatch for integer type"
        else:
            assert torch.allclose(received_tensor, expected_tensor, atol=1e-2), f"Values mismatch for float type"

        assert not torch.isnan(received_tensor).any(), f"Contains NaN"
        assert not torch.isinf(received_tensor).any(), f"Contains Inf"

        # Wait for sender to finish (read from pipe)
        os.read(recv_done_read, 1)
        os.close(recv_done_read)

        ttnn.close_device(device)
        return 0

    except Exception as e:
        print(f"Receiver error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


class TestSocketTensorTransfer:
    """Test actual socket tensor send/receive operations"""

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "tensor_shape",
        [
            [1, 1, 32, 256],  # Small tensor
            [1, 1, 64, 512],  # Medium tensor
            [1, 1, 128, 1024],  # Large tensor
        ],
    )
    @pytest.mark.parametrize(
        "buffer_type",
        [
            ttnn.BufferType.L1,
            ttnn.BufferType.DRAM,
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            ttnn.float32,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            ttnn.bfloat4_b,
            ttnn.int32,
            ttnn.uint32,
        ],
    )
    def test_socket_tensor_send_receive_multiprocess(self, tensor_shape, buffer_type, dtype):
        """Test sending and receiving tensors of different sizes and dtypes over socket using multiprocessing"""

        # Create queue for sharing tensor between processes
        tensor_queue = mp.Queue()

        # Create anonymous pipes
        # Pipe 1: receiver -> sender (ready signal)
        recv_ready_read, send_ready_write = os.pipe()
        # Pipe 2: sender -> receiver (done signal)
        send_done_read, send_done_write = os.pipe()

        try:
            # Start sender and receiver processes
            sender_proc = mp.Process(
                target=sender_process,
                args=(recv_ready_read, send_done_write, tensor_queue, tensor_shape, buffer_type, dtype),
            )
            receiver_proc = mp.Process(
                target=receiver_process,
                args=(send_ready_write, send_done_read, tensor_queue, tensor_shape, buffer_type, dtype),
            )

            # Close parent process copies of pipe file descriptors
            os.close(recv_ready_read)
            os.close(send_ready_write)
            os.close(send_done_read)
            os.close(send_done_write)

            # Start receiver first (it will wait for sender)
            receiver_proc.start()
            # Then start sender
            sender_proc.start()

            # Wait for both to complete (blocks until done or timeout)
            receiver_proc.join(timeout=60)
            sender_proc.join(timeout=60)

            # Verify processes actually finished
            assert not receiver_proc.is_alive(), "Receiver process timed out or is still running"
            assert not sender_proc.is_alive(), "Sender process timed out or is still running"

            # Check that both processes completed successfully
            assert receiver_proc.exitcode == 0, f"Receiver failed with exit code {receiver_proc.exitcode}"
            assert sender_proc.exitcode == 0, f"Sender failed with exit code {sender_proc.exitcode}"
        finally:
            # Ensure processes are terminated
            if "sender_proc" in locals() and sender_proc.is_alive():
                sender_proc.terminate()
            if "receiver_proc" in locals() and receiver_proc.is_alive():
                receiver_proc.terminate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
