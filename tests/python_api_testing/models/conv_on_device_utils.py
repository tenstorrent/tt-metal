import torch
import tt_lib
from loguru import logger
from models.utility_functions import unpad_from_zero, torch2tt_tensor, _nearest_32
from tt_lib.fused_ops.conv import conv as TtConv

def is_conv_supported_on_device(conv_params):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]

    if (K%32 != 0 or dilation != 1 or groups != 1):
        logger.warning("DOES NOT HAVE SUPPORT FOR Conv with following parameters -")
        logger.warning("K="+str(K)+" C="+str(C)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
        return False

    return True

def run_conv_on_device_wrapper(conv_weight, conv_params, device, conv_bias=None):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    conv_on_device = TtConv(conv_weight, conv_params, device, conv_bias)

    def run_conv_on_device(x):
        [N,C,H,W] = x.shape()

        logger.debug("Running Conv with following parameters on device -")
        logger.debug("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
        logger.debug("Going to run conv on tt device")
        x = conv_on_device(x)
        logger.debug("conv on tt device done")
        return x
    return run_conv_on_device
