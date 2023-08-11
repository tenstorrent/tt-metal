import torch
import tt_lib as ttl
from loguru import logger
from models.utility_functions import unpad_from_zero, torch2tt_tensor, _nearest_32
from tt_lib.fused_ops.conv import conv as TtConv
from tt_lib.fallback_ops import fallback_ops


def is_conv_supported_on_device(conv_params):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    if K % 32 != 0 or dilation != 1 or groups != 1:
        logger.warning("DOES NOT HAVE SUPPORT FOR Conv with following parameters -")
        logger.warning(
            "K="
            + str(K)
            + " C="
            + str(C)
            + " R="
            + str(R)
            + " S="
            + str(S)
            + " U="
            + str(U)
            + " V="
            + str(V)
            + " PH="
            + str(P_H)
            + " PW="
            + str(P_W)
            + " dilation="
            + str(dilation)
            + " groups="
            + str(groups)
        )
        return False

    return True


def run_conv_on_device_wrapper(
    conv_weight, conv_params, device, conv_bias=None, channel_transpose=False
):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    conv_on_device = TtConv(conv_weight, conv_params, device, conv_bias)

    def run_conv_on_device(x):
        [N, C, H, W] = x.shape()
        if N == 1:
            return run_conv_on_device_batch_one(x)
        # need to move on CPU
        if isinstance(x, ttl.tensor.Tensor):
            xx = x.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
        else:
            xx = x

        def to_device(_):
            assert isinstance(_, torch.Tensor)
            return ttl.tensor.Tensor(
                _.reshape(-1).tolist(),
                [1, C, H, W],
                x.dtype(),
                ttl.tensor.Layout.ROW_MAJOR,
            )

        partial_convs = [
            run_conv_on_device_batch_one(to_device(xx[batch_idx, :, :, :]))
            for batch_idx in range(N)
        ]
        conv_concat_cpu = fallback_ops.concat(partial_convs,0)
        # return ttl.tensor.concat(partial_convs,0) # hit problem with autoformat for non-32 size N
        # concat on CPU for batch-size > 1
        return conv_concat_cpu
    
    def run_conv_on_device_batch_one(x):
        [N, C, H, W] = x.shape()
        if channel_transpose:
            # n c h w -> n h w c
            x = ttl.tensor.transpose_hc(x)
            x = ttl.tensor.transpose(x)  # wh

        logger.info("Running Conv with following parameters on device -")
        logger.info(
            "K="
            + str(K)
            + " C="
            + str(C)
            + " H="
            + str(H)
            + " W="
            + str(W)
            + " R="
            + str(R)
            + " S="
            + str(S)
            + " U="
            + str(U)
            + " V="
            + str(V)
            + " PH="
            + str(P_H)
            + " PW="
            + str(P_W)
            + " dilation="
            + str(dilation)
            + " groups="
            + str(groups)
        )

        logger.info("Going to run conv on tt device")
        x = conv_on_device(x)
        if channel_transpose:
            x = ttl.tensor.transpose(x)
            x = ttl.tensor.transpose_hc(x)

        logger.info("conv on tt device done")
        return x

    return run_conv_on_device
