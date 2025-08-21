import ttnn
import torch


class TTModule:
    def __init__(self, mesh_device, state_dict):
        self.mesh_device = mesh_device

        # fc1
        w_fc1 = state_dict["fc1.weight"].T.view(1, 1, 1024, 128)  # transpose to (in_features, out_features)
        b_fc1 = state_dict["fc1.bias"]
        self.w_fc1 = ttnn.from_torch(w_fc1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        self.b_fc1 = ttnn.from_torch(b_fc1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        # fc2
        w_fc2 = state_dict["fc2.weight"].T.view(1, 1, 128, 96)
        b_fc2 = state_dict["fc2.bias"]
        self.w_fc2 = ttnn.from_torch(w_fc2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        self.b_fc2 = ttnn.from_torch(b_fc2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        # fc3
        w_fc3 = state_dict["fc3.weight"].T.view(1, 1, 96, 32)
        b_fc3 = state_dict["fc3.bias"]
        self.w_fc3 = ttnn.from_torch(w_fc3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

        self.b_fc3 = ttnn.from_torch(b_fc3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x is expected on device in TILE_LAYOUT
        # fc1
        breakpoint()
        out = ttnn.matmul(x, self.w_fc1)
        out = ttnn.add(out, self.b_fc1)
        out = ttnn.relu(out)

        # fc2
        out = ttnn.matmul(out, self.w_fc2)
        out = ttnn.add(out, self.b_fc2)
        out = ttnn.relu(out)

        # fc3
        out = ttnn.matmul(out, self.w_fc3)
        out = ttnn.add(out, self.b_fc3)
        out = ttnn.relu(out)

        # softmax output
        out = ttnn.softmax(out, dim=-1)
        return out


class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(1024, 128)
        self.fc2 = torch.nn.Linear(128, 96)
        self.fc3 = torch.nn.Linear(96, 32)

        # self.load_state_dict(state_dict)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = torch.nn.functional.relu(x)

        x = self.fc3(x)
        x = torch.nn.functional.relu(x)

        return torch.nn.functional.softmax(x)


import pytest


@pytest.mark.parametrize(
    "device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_run(
    device,
    reset_seeds,
):
    torch_input = torch.randn((1, 1, 1, 1024))
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    mnist_ref = MnistModel()
    mnist_ref.eval()
    state_dict = mnist_ref.state_dict()

    tt_model = TTModule(device, state_dict)

    tt_output = tt_model.forward(ttnn_input)
