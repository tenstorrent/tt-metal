import torch
import ttnn
from diffusers import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from models.tt_dit.models.vae.vae_wan2_1 import WanCausalConv3d as WanCausalConv3dTTNN
from models.common.metrics import compute_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor

# Test configuration constants
CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
BATCH_SIZE = 1
PCC_THRESHOLD = 0.99
# T must be >= temporal kernel_size for causal conv to work
H = 256  # Height
W = 320  # Width
# T will be set dynamically based on kernel_size


def test_wan_causal_conv3d():
    """Test WanCausalConv3dTTNN with Lingbot-VA model parameters."""

    print("Loading AutoencoderKLWan...")
    autoencoder_klwan = AutoencoderKLWan.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    autoencoder_klwan.eval()

    # Extract the actual conv_in layer to get its parameters
    conv_in_layer = autoencoder_klwan.encoder.conv_in
    actual_in_channels = conv_in_layer.in_channels
    actual_out_channels = conv_in_layer.out_channels
    actual_kernel_size = conv_in_layer.kernel_size
    actual_stride = conv_in_layer.stride
    actual_padding = conv_in_layer.padding

    # Get temporal kernel size (first dimension)
    temporal_kernel = actual_kernel_size[0] if isinstance(actual_kernel_size, tuple) else actual_kernel_size

    # Set T dynamically: must be >= temporal_kernel for causal conv
    T = max(temporal_kernel, 4)  # Use at least kernel_size, or 4 for a reasonable test

    print(
        f"Actual conv_in parameters: in_channels={actual_in_channels}, out_channels={actual_out_channels}, "
        f"kernel_size={actual_kernel_size}, stride={actual_stride}, padding={actual_padding}"
    )
    print(f"Using input dimensions: T={T}, H={H}, W={W}")

    wan_causal_conv3d_weights = conv_in_layer.state_dict()

    # Use actual parameters instead of hardcoded values
    wan_causal_conv3d = WanCausalConv3d(
        actual_in_channels,
        actual_out_channels,
        actual_kernel_size[0] if isinstance(actual_kernel_size, tuple) else actual_kernel_size,
        actual_stride[0] if isinstance(actual_stride, tuple) else actual_stride,
        actual_padding[0] if isinstance(actual_padding, tuple) else actual_padding,
    )
    wan_causal_conv3d.load_state_dict(wan_causal_conv3d_weights)
    wan_causal_conv3d.eval()

    torch.manual_seed(42)
    # Use actual_in_channels for input tensor
    input_tensor = torch.randn(BATCH_SIZE, actual_in_channels, T, H, W, dtype=torch.float32)
    input_tensor = input_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
    print("Running WanCausalConv3d forward pass...")
    with torch.no_grad():
        wan_causal_conv3d_out = wan_causal_conv3d(input_tensor)
    print("WanCausalConv3d output shape:", wan_causal_conv3d_out.shape)

    tt_wan_state_dict = {k: v for k, v in wan_causal_conv3d.state_dict().items()}
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )

    wan_causal_conv3d_ttnn = WanCausalConv3dTTNN(
        actual_in_channels,
        actual_out_channels,
        kernel_size=actual_kernel_size[0] if isinstance(actual_kernel_size, tuple) else actual_kernel_size,
        stride=actual_stride[0] if isinstance(actual_stride, tuple) else actual_stride,
        padding=actual_padding[0] if isinstance(actual_padding, tuple) else actual_padding,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    wan_causal_conv3d_ttnn.load_torch_state_dict(wan_causal_conv3d_weights)

    print("Running TTNN WanCausalConv3d forward pass...")
    tt_input_tensor = ttnn.from_torch(
        input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )

    # Convert from (B, C, T, H, W) to (B, T, H, W, C) format
    tt_input_tensor = ttnn.permute(tt_input_tensor, (0, 2, 3, 4, 1))

    # Pad input channels to match aligned_channels (32)
    # Input has 12 channels, needs to be padded to 32
    from models.tt_dit.utils.conv3d import aligned_channels

    actual_C = tt_input_tensor.shape[-1]
    aligned_C = aligned_channels(actual_C)
    if aligned_C != actual_C:
        # Pad the last dimension (channels) from actual_C to aligned_C
        pad_amount = aligned_C - actual_C
        tt_input_tensor = ttnn.pad(tt_input_tensor, [(0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)], 0.0)

    tt_wan_causal_conv3d_out = ttnn.permute(wan_causal_conv3d_ttnn(tt_input_tensor, logical_h=H), (0, 4, 1, 2, 3))
    print("Torch WanCausalConv3d output shape:", wan_causal_conv3d_out.shape)
    print("TTNN WanCausalConv3d output shape:", tt_wan_causal_conv3d_out.shape)

    tt_wan_causal_conv3d_out_torch = ttnn.to_torch(tt_wan_causal_conv3d_out)
    # ── 2. Compare outputs ──
    print("Comparing outputs...")
    pcc = compute_pcc(wan_causal_conv3d_out, tt_wan_causal_conv3d_out_torch)
    print("PCC:", pcc)
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc} is below threshold {PCC_THRESHOLD}"
    print("PCC is above threshold, test passed.")


if __name__ == "__main__":
    test_wan_causal_conv3d()
    exit(0)
