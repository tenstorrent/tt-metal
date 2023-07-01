import torch
import tt_lib
from loguru import logger
from utility_functions_new import unpad_from_zero, torch2tt_tensor, _nearest_32
from tt_lib.fused_ops.conv import conv as TtConv



def is_conv_supported_on_device(conv_params):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    if (K%32 != 0 or dilation != 1 or groups != 1):
        logger.warning("DOES NOT HAVE SUPPORT FOR Conv with following parameters -")
        logger.warning("K="+str(K)+" C="+str(C)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
        return False

    return True

def can_run_conv_on_device(act_shape, conv_params):
    #return False
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    [N,C,H,W] = act_shape

    logger.info("Conv with following parameters -")
    logger.info("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))

    if (C % 32 != 0 or K%32 != 0 or dilation != 1 or groups != 1):
        return False

    return True

def run_conv_on_tt_device(x: torch.Tensor, conv_on_tt, conv_params, device, host):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    [N,C,H,W] = x.shape

    logger.info("Running Conv with following parameters on device -")
    logger.info("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))

    OH = ((int) ((H - R + 2 * P_H) / U)) + 1
    OW = ((int) ((W - S + 2 * P_W) / V)) + 1
    conv_as_mm_output_shape_unpadded = [1,1,OH*OW,K]
    x = torch2tt_tensor(x, device, tt_lib.tensor.Layout.CHANNELS_LAST, tt_lib.tensor.MemoryConfig(False))

    logger.info("Going to run conv on tt device")
    x = conv_on_tt(x)

    logger.info("conv on tt device done")
    x = unpad_from_zero(x, conv_as_mm_output_shape_unpadded, host)

    # Convert matmul output layout to conv output layout
    x = torch.transpose(x, 2, 3)
    assert(list(x.shape) == [1,1,K,(OH*OW)])

    return x.reshape([1,K,OH,OW])

def run_conv_on_device_wrapper(conv_weight, conv_params, device, host, conv_bias=None):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    conv_on_device = TtConv(conv_weight, conv_params, device, conv_bias)

    def run_conv_on_device(x: torch.Tensor):
        [N,C,H,W] = x.shape

        logger.info("Running Conv with following parameters on device -")
        logger.info("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))

        OH = ((int) ((H - R + 2 * P_H) / U)) + 1
        OW = ((int) ((W - S + 2 * P_W) / V)) + 1
        conv_as_mm_output_shape_unpadded = [1,1,OH*OW,K]
        x_shape_channel_padded = [N,_nearest_32(C),H,W]
        x = tt_lib.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
            ).pad(x_shape_channel_padded, (0,0,0,0), 0).to(tt_lib.tensor.Layout.CHANNELS_LAST).to(device, tt_lib.tensor.MemoryConfig(False))
        logger.info("Going to run conv on tt device")
        x = conv_on_device(x)

        logger.info("conv on tt device done")
        x = unpad_from_zero(x, conv_as_mm_output_shape_unpadded, host)

        # Convert matmul output layout to conv output layout
        x = torch.transpose(x, 2, 3)
        assert(list(x.shape) == [1,1,K,(OH*OW)])

        return x.reshape([1,K,OH,OW])
    return run_conv_on_device
