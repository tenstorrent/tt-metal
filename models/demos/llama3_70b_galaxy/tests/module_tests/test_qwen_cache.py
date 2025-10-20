import pytest
import torch


@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_load_caches(mesh_device):
    keys_torch = torch.load("./keys.pt")
    values_torch = torch.load("./values.pt")
    k_heads_1BKD_torch = torch.load("./k_heads_1BKD.pt")
    v_heads_1BKD_torch = torch.load("./v_heads_1BKD.pt")

    k_head_slice = k_heads_1BKD_torch[:, 0, :, :]
    k_head_slice = k_head_slice[:, 0, :]

    keys_nonzero = keys_torch[torch.isfinite(keys_torch) & (keys_torch != 0)]

    # keys_ref = torch.load("./keys_ref.pt")

    breakpoint()
    print(keys_torch)
