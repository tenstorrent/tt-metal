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

def run_conv_on_device_wrapper(conv_weight, conv_params, device, host, conv_bias=None):
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    conv_on_device = TtConv(conv_weight, conv_params, device, conv_bias)

    def run_conv_on_device(x: torch.Tensor):
        [N,C,H,W] = x.shape

        logger.info("Running Conv with following parameters on device -")
        logger.info("K="+str(K)+" C="+str(C)+" H="+str(H)+" W="+str(W)+" R="+str(R)+" S="+str(S)+" U="+str(U)+" V="+str(V)+" PH="+str(P_H)+" PW="+str(P_W)+" dilation="+str(dilation)+" groups="+str(groups))

        OH = ((int) ((H - R + 2 * P_H) / U)) + 1
        OW = ((int) ((W - S + 2 * P_W) / V)) + 1
        conv_output_shape = [1,K,OH,OW]
        x_shape_channel_padded = [N,_nearest_32(C),H,W]

        x = tt_lib.tensor.Tensor(x.contiguous().to(torch.bfloat16))
        x = x.pad(x_shape_channel_padded, (0,0,0,0), 0).to(tt_lib.tensor.Layout.CHANNELS_LAST).to(device)
        logger.info("Going to run conv on tt device")
        x = conv_on_device(x)
        logger.info("conv on tt device done")
        x = x.to(host)
        assert(x.shape() == conv_output_shape)
        assert(x.layout() == tt_lib.tensor.Layout.CHANNELS_LAST)

        # Copy output to host and convert tt tensor to pytorch tensor
        conv_output_shape_cl = [1,OH,OW,K]
        x = torch.tensor(x.data()).reshape(conv_output_shape_cl)
        x = torch.transpose(x, 2, 3)
        x = torch.transpose(x, 1, 2)

        assert(list(x.shape) == [1,K,OH,OW])

        return x
    return run_conv_on_device
