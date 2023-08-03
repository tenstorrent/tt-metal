import tt_lib


def closestNumberDivisibleByTileSize(n):
    if n % 32 == 0:
        return n

    q = int(n / 32)
    num = 32 * (q + 1)

    return num


def WhisperPaddedLinear(
    in_features: int, out_features: int, torch_weight, torch_bias, device
):
    """
    Returns a function that performs a padded Linear operation with optional bias.

    ``weight`` must be padded inside linear if not divisible by 32.
    """
    out_mem_config_l1 = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)

    # Create weight tensor on host
    weight_on_host = tt_lib.tensor.Tensor(
        torch_weight.reshape(-1).tolist(),
        [1, 1, out_features, in_features],
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )
    # Pad on host
    input_tensor_start = [0, 0, 0, 0]
    out_features_pad = closestNumberDivisibleByTileSize(out_features)
    in_features_pad = closestNumberDivisibleByTileSize(in_features)
    output_tensor_shape = [1, 1, out_features_pad, in_features_pad]
    weight_on_host_pad = weight_on_host.pad(output_tensor_shape, input_tensor_start, 0)

    weight = weight_on_host_pad.to(tt_lib.tensor.Layout.TILE).to(device)

    if torch_bias is None:
        bias = None
    else:
        bias_on_host = tt_lib.tensor.Tensor(
            torch_bias.reshape(-1).tolist(),
            [1, 1, 1, out_features],
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        # Pad on host
        input_tensor_start = [0, 0, 0, 0]
        output_tensor_shape = [1, 1, 32, out_features_pad]
        bias_on_host_pad = bias_on_host.pad(output_tensor_shape, input_tensor_start, 0)

        bias = bias_on_host_pad.to(tt_lib.tensor.Layout.TILE).to(device)

    def linear_(activation):
        weight_T = tt_lib.tensor.transpose(weight)
        output = tt_lib.tensor.matmul(activation, weight_T)

        if bias is not None:
            output_plus_bias = tt_lib.tensor.bcast(
                output, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H, out_mem_config_l1
            )
            return output_plus_bias

        return output

    return linear_
