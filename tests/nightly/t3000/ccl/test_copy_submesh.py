import torch
import ttnn
import test
import pytest

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
