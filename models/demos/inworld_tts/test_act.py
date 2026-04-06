import torch
import torch.nn.functional as F
import ttnn


# ---------------------------------------------------------------------------
# SnakeBeta Activation
# ---------------------------------------------------------------------------
def snake_beta(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x).

    alpha, beta: [C] learnable parameters, broadcast over [B, C, T].
    """
    alpha = alpha.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
    beta = beta.unsqueeze(0).unsqueeze(-1)  # [1, C, 1]
    return x + (1.0 / beta) * torch.sin(alpha * x) ** 2


def ttnn_snake_beta(x, alpha, beta):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x).

    alpha, beta: [C] learnable parameters, broadcast over [B, C, T].
    """
    alpha = alpha.reshape([1, 1, 1, -1])
    beta = beta.reshape([1, 1, 1, -1])
    print(f"ttnn_snake_beta: x.shape={x.shape}, alpha.shape={alpha.shape}, beta.shape={beta.shape}")
    return x + (1.0 / beta) * ttnn.pow(ttnn.sin(alpha * x), 2)



# ---------------------------------------------------------------------------
# Anti-aliased SnakeBeta (Activation1d wrapper)
# ---------------------------------------------------------------------------
def activation1d_forward(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    up_filter: torch.Tensor,
    down_filter: torch.Tensor,
) -> torch.Tensor:
    """Anti-aliased SnakeBeta activation.

    Upsample 2x -> SnakeBeta -> Downsample 2x to avoid aliasing from nonlinearity.

    Args:
        x: [B, C, T]
        alpha: [C] SnakeBeta alpha parameter
        beta: [C] SnakeBeta beta parameter
        up_filter: [1, 1, K] FIR upsampling filter
        down_filter: [1, 1, K] FIR lowpass/downsampling filter
    Returns:
        [B, C, T]
    """
    B, C, T = x.shape

    # Prepare filters for depthwise conv: [C, 1, K]
    up_kernel = up_filter.squeeze(0).expand(C, -1, -1)  # [C, 1, K]
    down_kernel = down_filter.squeeze(0).expand(C, -1, -1)  # [C, 1, K]
    K = up_kernel.shape[-1]

    # For even-length FIR filters, use asymmetric padding to preserve length
    pad_left = K // 2
    pad_right = K // 2 - 1 if K % 2 == 0 else K // 2

    print(f"Upsample kernel size: {K}, padding: ({pad_left}, {pad_right})")
    # Upsample by 2: insert zeros between samples, then filter
    x_up = torch.zeros(B, C, T * 2, device=x.device, dtype=x.dtype)
    x_up[:, :, ::2] = x
    x_up = F.pad(x_up, (pad_left, pad_right))
    x_up = F.conv1d(x_up, up_kernel * 2.0, groups=C)


    # Apply SnakeBeta activation
    x_act = snake_beta(x_up, alpha, beta)

    # Downsample by 2: lowpass filter then take every 2nd sample
    x_down = F.pad(x_act, (pad_left, pad_right))
    x_down = F.conv1d(x_down, down_kernel, groups=C)
    x_down = x_down[:, :, ::2]

    return x_down


def activation1d_forward_ttnn(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    up_filter: torch.Tensor,
    down_filter: torch.Tensor,
    device,
) -> torch.Tensor:
    """Anti-aliased SnakeBeta activation using TTNN operations.

    TTNN implementation:
    - Upsample 2x: ttnn.conv_transpose2d (with height=1 to mimic conv_transpose1d)
    - SnakeBeta activation (on CPU for now - sin ops)
    - Downsample 2x: ttnn.conv1d with stride=2

    Args:
        x: [B, C, T] input tensor (torch)
        alpha: [C] SnakeBeta alpha parameter
        beta: [C] SnakeBeta beta parameter
        up_filter: [1, 1, K] FIR upsampling filter
        down_filter: [1, 1, K] FIR lowpass/downsampling filter
        device: TTNN device
    Returns:
        [B, C, T] output tensor (torch)
    """
    B, C, T = x.shape

    # Prepare filters
    up_kernel = up_filter.squeeze(0).expand(C, -1, -1)  # [C, 1, K]
    down_kernel = down_filter.squeeze(0).expand(C, -1, -1)  # [C, 1, K]
    K_up = up_kernel.shape[-1]
    K_down = down_kernel.shape[-1]

    pad_left = K_up // 2
    pad_right = K_up // 2 - 1 if K_up % 2 == 0 else K_up // 2


    # === UPSAMPLE using TTNN ConvTranspose2D (height=1 for 1D operation) ===
    # Reshape input for 2D conv: [B, C, T] -> [B, 1, T, C] (NHWC with H=1)
    x_nhwc = x.unsqueeze(1).permute(0, 1, 3, 2)  # [B, 1, T, C]
    x_ttnn = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Prepare weight for conv_transpose2d: [C, C, 1, K] (IOHW format)
    # For groups=C depthwise: input_channels=C, output_channels=C, groups=C
    # So each input channel has its own 1x1xK filter
    up_weight_2d = up_kernel.unsqueeze(2) * 2.0  # [C, 1, 1, K] -> add height dim
    up_weight_ttnn = ttnn.from_torch(up_weight_2d, dtype=ttnn.bfloat16)
    
    # Conv2d configuration
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.float32,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        act_block_h_override=32,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    
    # Padding for upsampling: only in width dimension
    pad_w = (K_up - 2) // 2
    
    # ConvTranspose2D with height=1, width=T, stride=(1,2) for 2x upsampling in width
    x_up_ttnn, [out_h, out_w], [up_weight_ret, _] = ttnn.conv_transpose2d(
        input_tensor=x_ttnn,
        weight_tensor=up_weight_ttnn,
        in_channels=C,
        out_channels=C,
        device=device,
        bias_tensor=None,
        kernel_size=(1, K_up),  # height=1, width=K
        stride=(1, 2),  # no stride in height, 2x in width for upsampling
        padding=(0, pad_w),  # no padding in height
        output_padding=(0, 0),
        batch_size=B,
        input_height=1,  # height dimension is 1
        input_width=T,  # width dimension is time
        conv_config=conv_config,
        compute_config=compute_config,
        groups=C,  # Depthwise
        mirror_kernel=False,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )
    
    print(f"ConvTranspose2D output: H={out_h}, W={out_w} (expected H=1, W~{T*2})")
    
    # Apply SnakeBeta activation (CPU - torch.sin not in TTNN)
    ttnn_alpha = ttnn.from_torch(alpha, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_beta = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    
    x_act = ttnn_snake_beta(x_up_ttnn, ttnn_alpha, ttnn_beta)

    # === DOWNSAMPLE using TTNN Conv1d with stride=2 ===
    # Padding for downsampling filter
    # pad_down_left = K_down // 2
    # pad_down_right = K_down // 2 - 1 if K_down % 2 == 0 else K_down // 2
    
    # Convert to TTNN NHWC format for conv1d: [B, 1, T, C]
    
    # Prepare downsampling kernel
    down_kernel_ttnn = ttnn.from_torch(down_kernel, dtype=ttnn.bfloat16)
    
    # Apply downsampling filter with stride=2 via TTNN
    x_down_ttnn, out_len, [down_kernel_ttnn, _] = ttnn.conv1d(
        input_tensor=x_act,
        weight_tensor=down_kernel_ttnn,
        in_channels=C,
        out_channels=C,
        device=device,
        bias_tensor=None,
        kernel_size=K_down,
        stride=2,  # Decimate by 2
        padding=(pad_left, pad_right),  # Already padded
        batch_size=B,
        input_length=x_act.shape[2],
        dtype=ttnn.bfloat16,
        conv_config=ttnn.Conv1dConfig(
            weights_dtype=ttnn.float32,
            deallocate_activation=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            act_block_h_override=32,
        ),
        compute_config=compute_config,
        groups=C,  # Depthwise
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    
    # Convert back to torch
    x_down = ttnn.to_torch(x_down_ttnn)
    if x_down.dim() == 4:
        x_down = x_down.squeeze(0).permute(0, 2, 1)  # Back to [B, C, T]
    
    
    return x_down

# ---------------------------------------------------------------------------
# Test/Comparison
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test parameters
    B, C, T = 1, 48, 100
    
    # Create sample input and parameters
    x = torch.randn(B, C, T)
    alpha = torch.randn(C) * 0.1 + 1.0
    beta = torch.randn(C) * 0.1 + 1.0
    
    # Create sample FIR filters (simple Gaussian-like for testing)
    K = 8
    up_filter = torch.randn(1, 1, K) * 0.1
    up_filter = up_filter / up_filter.sum()  # Normalize
    down_filter = torch.randn(1, 1, K) * 0.1
    down_filter = down_filter / down_filter.sum()  # Normalize
    
    # Open TTNN device
    print("Opening TTNN device...")
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    
    try:
        # Test both implementations
        print("\nTesting activation1d implementations...")
        print(f"Input shape: {x.shape}")
        
        # Original (zero-insertion + decimation) - PyTorch CPU
        y_original = activation1d_forward(x, alpha, beta, up_filter, down_filter)
        print(f"Original (PyTorch) output shape: {y_original.shape}")
        
        # TTNN implementation
        y_ttnn = activation1d_forward_ttnn(x, alpha, beta, up_filter, down_filter, device)
        print(f"TTNN output shape: {y_ttnn.shape}")
        
        
        # Compare TTNN vs Original
        print("\n" + "="*70)
        print("TTNN vs Original PyTorch:")
        print("="*70)
        if y_original.shape == y_ttnn.shape:
            diff = (y_original - y_ttnn).abs()
            print(f"  Mean absolute diff: {diff.mean().item():.6f}")
            print(f"  Max absolute diff: {diff.max().item():.6f}")
            print(f"  Relative error: {(diff / (y_original.abs() + 1e-8)).mean().item():.6f}")
            
            corr = torch.corrcoef(torch.stack([y_original.flatten(), y_ttnn.flatten()]))[0, 1]
            print(f"  Correlation: {corr.item():.6f}")
        else:
            print(f"  WARNING: Output shapes don't match!, Original: {y_original.shape}, TTNN: {y_ttnn.shape}")
        
        
    finally:
        # Clean up device
        ttnn.close_device(device)
        print("TTNN device closed.")
