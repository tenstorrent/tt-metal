import torch
import ttnn
import test
import pytest
import threading
import queue

from models.common.auto_compose import to_torch_auto_compose
from time import sleep, time

ttnn.set_printoptions(profile="short", sci_mode=False)
torch.set_printoptions(sci_mode=False)


def print_mesh(mesh_device):
    shape = mesh_device.shape
    for row in reversed(range(shape[0])):
        for col in range(shape[1]):
            print(f" {mesh_device.get_device_id(ttnn.MeshCoordinate(row, col))}", end="")
        print()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("submesh_shape", [(2, 2)])
@pytest.mark.parametrize(
    "num_links",
    [2],
)
@pytest.mark.parametrize("input_shape", [(1, 1, 1024, 1024)])
def test_copy_submesh(mesh_device, submesh_shape, num_links, input_shape):
    submesh1 = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
    submesh2 = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape), ttnn.MeshCoordinate(0, 2))
    print("Full mesh:")
    print_mesh(mesh_device)
    print("Submesh 1:")
    print_mesh(submesh1)
    print("Submesh 2:")
    print_mesh(submesh2)
    torch.manual_seed(time())
    input_tensor = torch.randint(-40, 50, input_shape, dtype=torch.bfloat16)
    weights_tensor = torch.randint(-40, 50, input_shape, dtype=torch.bfloat16)
    input_s1 = ttnn.from_torch(input_tensor, device=submesh1, layout=ttnn.Layout.TILE)
    weights_s1 = ttnn.from_torch(weights_tensor, device=submesh1, layout=ttnn.Layout.TILE)

    output_tensor = ttnn.allocate_tensor_on_device(input_s1.spec, submesh2)
    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(submesh1.shape):  # Iterates over all 16 device positions
        socket_connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),  # sender: device coord, core (0,0)
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),  # receiver: same layout
            )
        )
    socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
    socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
    sender_socket, receiver_socket = ttnn.create_socket_pair(
        submesh1,
        submesh2,
        socket_config,
    )
    ttnn.experimental.send_async(input_s1, sender_socket)
    ttnn.experimental.recv_async(output_tensor, receiver_socket)
    output = ttnn.matmul(input_s1, weights_s1)
    print("Output after matmul on submesh1:", output)
    composer_cfg = ttnn.MeshComposerConfig(dims=[0, 1], mesh_shape_override=ttnn.MeshShape(2, 2))
    torch_output1 = ttnn.to_torch(output_tensor, mesh_composer=ttnn.create_mesh_composer(submesh2, composer_cfg))
    assert torch.allclose(
        input_tensor, torch_output1
    ), f"Output tensor does not match input tensor.\nInput Tensor: {input_tensor}\nOutput Tensor: {torch_output1}"
    print("Output Tensor:", torch_output1)
    print("Cleaning up")


def run_submesh_ops(input_tensor_spec, run_device, socket_pair):
    sender_socket, receiver_socket = socket_pair
    if receiver_socket is not None:
        op_input_tensor = ttnn.allocate_tensor_on_device(input_tensor_spec, run_device)
        ttnn.experimental.recv_async(op_input_tensor, receiver_socket)
    else:
        op_input_tensor = input_tensor_spec
    output = op_input_tensor
    for _ in range(100):  # Simulate some work by running multiple iterations
        output = ttnn.plus_one(output)
    if sender_socket is not None:
        ttnn.experimental.send_async(output, sender_socket)
    return output


def run_submesh_first_stage(input_queue, input_tensor_spec, run_device, socket_pair, num_iterations):
    """First stage: receives from host FIFO, sends to next submesh."""
    sender_socket, _ = socket_pair

    for iteration in range(num_iterations):
        # Get input from host FIFO
        input_tensor = input_queue.get()
        if input_tensor is None:  # Sentinel value to stop
            break

        print(f"Stage 0: Processing iteration {iteration}")

        # Process the tensor
        output = input_tensor
        for _ in range(100):  # Simulate work
            output = ttnn.plus_one(output)

        # Send to next stage
        if sender_socket is not None:
            ttnn.experimental.send_async(output, sender_socket)


def run_submesh_middle_stage(input_tensor_spec, run_device, socket_pair, num_iterations, stage_id):
    """Middle stages: receive from previous, send to next submesh."""
    sender_socket, receiver_socket = socket_pair

    for iteration in range(num_iterations):
        # Receive from previous stage
        op_input_tensor = ttnn.allocate_tensor_on_device(input_tensor_spec, run_device)
        ttnn.experimental.recv_async(op_input_tensor, receiver_socket)

        print(f"Stage {stage_id}: Processing iteration {iteration}")

        # Process the tensor
        output = op_input_tensor
        for _ in range(100):  # Simulate work
            output = ttnn.plus_one(output)

        # Send to next stage
        if sender_socket is not None:
            ttnn.experimental.send_async(output, sender_socket)


def run_submesh_last_stage(output_queue, input_tensor_spec, run_device, socket_pair, num_iterations):
    """Last stage: receives from previous submesh, sends to host FIFO."""
    _, receiver_socket = socket_pair

    for iteration in range(num_iterations):
        # Receive from previous stage
        op_input_tensor = ttnn.allocate_tensor_on_device(input_tensor_spec, run_device)
        ttnn.experimental.recv_async(op_input_tensor, receiver_socket)

        print(f"Stage 3: Processing iteration {iteration}")

        # Process the tensor
        output = op_input_tensor
        for _ in range(100):  # Simulate work
            output = ttnn.plus_one(output)

        # Send to host FIFO
        output_queue.put((iteration, output))


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("num_submeshes, submesh_shape", [(4, (2, 1))])
@pytest.mark.parametrize("input_shape", [(1, 1, 1024, 1024)])
@pytest.mark.parametrize("num_iterations", [5])
def test_submesh_pipeline_sequential(mesh_device, num_submeshes, submesh_shape, input_shape, num_iterations):
    """Test submesh pipeline with sequential (single-threaded) execution."""
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(submesh_shape), ttnn.MeshCoordinate(0, i * submesh_shape[1]))
        for i in range(num_submeshes)
    ]
    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(submeshes[0].shape):
        socket_connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
            )
        )
    socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
    socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
    socket_pairs = {}
    for i in range(num_submeshes - 1):
        sender_socket, receiver_socket = ttnn.create_socket_pair(
            submeshes[i],
            submeshes[i + 1],
            socket_config,
        )
        socket_pairs[i] = (sender_socket, receiver_socket)

    print("\n=== Running sequential pipeline test ===")
    input_tensor = ttnn.from_torch(
        torch.zeros(input_shape, dtype=torch.uint32), layout=ttnn.Layout.ROW_MAJOR, device=submeshes[0]
    )
    input_tensor_spec = input_tensor.spec

    # Execute stages sequentially in reverse order (stage 3, 2, 1, 0)
    # This ensures sockets are set up before data flows through
    print("Executing stages sequentially...")
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration} ---")
        op4 = run_submesh_ops(input_tensor_spec, submeshes[3], (None, socket_pairs[2][1]))
        op3 = run_submesh_ops(input_tensor_spec, submeshes[2], (socket_pairs[2][0], socket_pairs[1][1]))
        op2 = run_submesh_ops(input_tensor_spec, submeshes[1], (socket_pairs[1][0], socket_pairs[0][1]))
        op1 = run_submesh_ops(input_tensor, submeshes[0], (socket_pairs[0][0], None))

    for submesh in submeshes:
        ttnn.synchronize_device(submesh)

    composer_cfg = ttnn.MeshComposerConfig(dims=[0, 1], mesh_shape_override=ttnn.MeshShape(2, 1))
    torch_output = ttnn.to_torch(op4, mesh_composer=ttnn.create_mesh_composer(submeshes[3], composer_cfg))

    print("Sequential pipeline output:", torch_output)
    print("=== Sequential pipeline test completed ===\n")


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}], indirect=True)
@pytest.mark.parametrize("num_submeshes, submesh_shape", [(4, (2, 1))])
@pytest.mark.parametrize("input_shape", [(1, 1, 1024, 1024)])
@pytest.mark.parametrize("num_iterations", [5])
def test_submesh_pipeline_multithreaded(mesh_device, num_submeshes, submesh_shape, input_shape, num_iterations):
    """Test submesh pipeline with multi-threaded execution and FIFOs for continuous data flow."""
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(submesh_shape), ttnn.MeshCoordinate(0, i * submesh_shape[1]))
        for i in range(num_submeshes)
    ]
    socket_connections = []
    for coord in ttnn.MeshCoordinateRange(submeshes[0].shape):
        socket_connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),  # sender: device coord, core (0,0)
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),  # receiver: same layout
            )
        )
    socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
    socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
    socket_pairs = {}
    for i in range(num_submeshes - 1):
        sender_socket, receiver_socket = ttnn.create_socket_pair(
            submeshes[i],
            submeshes[i + 1],
            socket_config,
        )
        socket_pairs[i] = (sender_socket, receiver_socket)

    # Create first input tensor to get spec
    initial_tensor = ttnn.from_torch(
        torch.zeros(input_shape, dtype=torch.uint32), layout=ttnn.Layout.ROW_MAJOR, device=submeshes[0]
    )
    input_tensor_spec = initial_tensor.spec

    # Create FIFOs for host-to-device and device-to-host communication
    input_queue = queue.Queue(maxsize=num_iterations + 1)
    output_queue = queue.Queue(maxsize=num_iterations + 1)

    print(f"\n=== Starting pipeline with {num_iterations} iterations ===")

    # Create and enqueue input tensors
    print(f"Enqueuing {num_iterations} input tensors...")
    input_tensors_torch = []
    for i in range(num_iterations):
        input_tensor_torch = torch.randint(0, 256, input_shape, dtype=torch.uint32)
        input_tensor = ttnn.from_torch(input_tensor_torch, layout=ttnn.Layout.ROW_MAJOR, device=submeshes[0])
        input_tensors_torch.append(input_tensor_torch)
        input_queue.put(input_tensor)

    # Create threads for each submesh operation with new pipeline functions
    thread1 = threading.Thread(
        target=run_submesh_first_stage,
        args=(input_queue, input_tensor_spec, submeshes[0], (socket_pairs[0][0], None), num_iterations),
    )
    thread2 = threading.Thread(
        target=run_submesh_middle_stage,
        args=(input_tensor_spec, submeshes[1], (socket_pairs[1][0], socket_pairs[0][1]), num_iterations, 1),
    )
    thread3 = threading.Thread(
        target=run_submesh_middle_stage,
        args=(input_tensor_spec, submeshes[2], (socket_pairs[2][0], socket_pairs[1][1]), num_iterations, 2),
    )
    thread4 = threading.Thread(
        target=run_submesh_last_stage,
        args=(output_queue, input_tensor_spec, submeshes[3], (None, socket_pairs[2][1]), num_iterations),
    )

    # Start all threads
    print("Starting all pipeline stages...")
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    # Collect results from output FIFO
    print("Collecting outputs...")
    results = []
    for i in range(num_iterations):
        iteration, output_tensor = output_queue.get()
        print(f"Received output for iteration {iteration}")
        results.append((iteration, output_tensor))

    # Wait for all threads to complete
    print("Waiting for all threads to complete...")
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    print(f"\n=== Pipeline completed {num_iterations} iterations ===")

    # Verify results
    composer_cfg = ttnn.MeshComposerConfig(dims=[0, 1], mesh_shape_override=ttnn.MeshShape(2, 1))
    print("\nVerifying results...")
    for iteration, output_tensor in results:
        torch_output = ttnn.to_torch(
            output_tensor, mesh_composer=ttnn.create_mesh_composer(submeshes[3], composer_cfg)
        ).to(torch.bfloat16)
        assert torch.allclose(
            torch_output, input_tensors_torch[iteration].to(torch.bfloat16) + 400
        ), f"Output tensor does not match expected value for iteration {iteration}.\nExpected: {input_tensors_torch[iteration][0] + 400}\nGot: {torch_output}"
        print(f"Iteration {iteration}: Output verified successfully.")

    print("Final output after pipeline:", torch_output)
