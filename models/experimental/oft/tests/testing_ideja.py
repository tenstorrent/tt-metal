import torch


def test_group_norm_basic():
    # Create a tensor with shape (batch_size, num_channels, height, width)
    x = torch.randn(2, 6, 4, 4)
    # Apply GroupNorm with 3 groups
    group_norm = torch.nn.GroupNorm(num_groups=3, num_channels=6)
    output = group_norm(x)
    assert output.shape == x.shape
    # Check mean and std per group
    reshaped = output.view(2, 3, 2, 4, 4)
    means = reshaped.mean(dim=(2, 3, 4))
    stds = reshaped.std(dim=(2, 3, 4))
    # GroupNorm normalizes each group to mean~0, std~1
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-1)
    assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)


def test_group_norm_various_groups():
    x = torch.randn(1, 256, 159, 159)
    group_norm = torch.nn.GroupNorm(num_groups=16, num_channels=256)
    output = group_norm(x)

    x1 = x[:, :128, :, :]
    group_norm1 = torch.nn.GroupNorm(num_groups=8, num_channels=128)
    output1 = group_norm1(x1)

    x2 = x[:, 128:, :, :]
    group_norm2 = torch.nn.GroupNorm(num_groups=8, num_channels=128)
    output2 = group_norm2(x2)

    out = torch.cat((output1, output2), dim=1)
    assert torch.allclose(output, out, atol=1e-10)

    assert output.shape == x.shape
    # Check that output is finite
    assert torch.isfinite(output).all()
