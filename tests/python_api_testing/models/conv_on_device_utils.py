import torch
import numpy as np
from libs import tt_lib as ttl
from utility_functions import pad_by_zero, unpad_from_zero, torch2tt_tensor
from python_api_testing.fused_ops.conv import conv as TtConv
from libs.tt_lib.utils import (
    _nearest_32 as nearest_32,
)

def is_conv_supported_on_device(conv_params):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    if (C % 32 != 0 or K%32 != 0 or dilation != 1 or groups != 1):
        print("DOES NOT HAVE SUPPORT FOR Conv with following parameters -")
        print("K="+str(K)+" C="+str(C)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
        return False
    return True

def can_run_conv_on_device(act_shape, conv_params):
    #return False
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    [N,C,H,W] = act_shape
    print("Conv with following parameters -")
    print("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
    if (C % 32 != 0 or K%32 != 0 or dilation != 1 or groups != 1):
        return False
    return True

def run_conv_on_tt_device(x: torch.Tensor, conv_on_tt, conv_params, device, host):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    [N,C,H,W] = x.shape
    print("Running Conv with following parameters on device -")
    print("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
    OH = ((int) ((H - R + 2 * P_H) / U)) + 1
    OW = ((int) ((W - S + 2 * P_W) / V)) + 1
    conv_as_mm_output_shape_unpadded = [1,1,OH*OW,K]
    x = torch2tt_tensor(x, device, ttl.tensor.Layout.CHANNELS_LAST, ttl.tensor.MemoryConfig(False, 0))
    print("Going to run conv on tt device")
    x = conv_on_tt(x)
    print("conv on tt device done")
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
        print("Running Conv with following parameters on device -")
        print("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))
        OH = ((int) ((H - R + 2 * P_H) / U)) + 1
        OW = ((int) ((W - S + 2 * P_W) / V)) + 1
        conv_as_mm_output_shape_unpadded = [1,1,OH*OW,K]
        x = torch2tt_tensor(x, device, ttl.tensor.Layout.CHANNELS_LAST, ttl.tensor.MemoryConfig(False, 0))
        print("Going to run conv on tt device")
        x = conv_on_device(x)
        print("conv on tt device done")
        x = unpad_from_zero(x, conv_as_mm_output_shape_unpadded, host)
        # Convert matmul output layout to conv output layout
        x = torch.transpose(x, 2, 3)
        assert(list(x.shape) == [1,1,K,(OH*OW)])
        return x.reshape([1,K,OH,OW])
    return run_conv_on_device
