import math
import struct

import torch
import numpy as np

def _nearest_32(x):
    return math.ceil(x / 32) * 32

def pad_activation(x):
    """
    This function pads an activation with 0s as a pre-preprocessing step to tilization.

    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"
    if len(x.shape) == 1: # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, :x.shape[0]] = x
    elif len(x.shape) == 2: # (batch, num features)
        padded_tensor = torch.zeros(x.shape[0], 1, 32, nearest_32(x.shape[1]))
        padded_tensor[:, 0, 0, :x.shape[1]] = x
    elif len(x.shape) == 3: # (batch, num features y, num features x)
        padded_tensor = torch.zeros(x.shape[0], 1, nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., 0, :x.shape[-2], :x.shape[-1]] = x
    else: # (batch, num channels, num features y, num features x)
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., :x.shape[-2], :x.shape[-1]] = x
    return padded_tensor

def pad_weight(x):
    """
    This function pads a weight/bias with 0s as a pre-preprocessing step to tilization.

    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of padding on device
    """
    nearest_32 = _nearest_32

    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"

    if len(x.shape) == 1: # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
        padded_tensor[:, 0, 0, :x.shape[0]] = x
    elif len(x.shape) == 2: # (r_features, c_features)
        padded_tensor = torch.zeros(1, 1, nearest_32(x.shape[0]), nearest_32(x.shape[1]))
        padded_tensor[:, 0, :x.shape[0], :x.shape[1]] = x
    else:
        padded_tensor = torch.zeros(*x.shape[:-2], nearest_32(x.shape[-2]), nearest_32(x.shape[-1]))
        padded_tensor[..., :x.shape[-2], :x.shape[-1]] = x

    return padded_tensor

def channels_last(x):
    """
    This function converts a row-major tensor to channels last.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    """
    nearest_32 = _nearest_32

    assert isinstance(x, (torch.Tensor, np.ndarray)), "Input to this function must be an instance of torch.Tensor or np.array"
    assert len(x.shape) == 4, "Only 4D tensors suppported"

    ret_shape = [x.shape[0], x.shape[2], x.shape[3], x.shape[1]]
    if isinstance(x, torch.Tensor):
        ret = torch.zeros(np.prod(x.shape))
    else:
        ret = np.zeros(np.prod(x.shape))

    idx = 0
    for n in range(x.shape[0]):
        for h in range(x.shape[2]):
            for w in range(0, x.shape[3]):
                for c in range(0, x.shape[1]):
                    ret[idx] = x[n][c][h][w]
                    idx+=1

    return ret.reshape(ret_shape)

def convert_weights_2d_matrix(weights, w_shape):
    """
    :param weights: Input PyTorch Tensor
    :type weights: class:`torch.Tensor`
    """
    ret_shape = [1,1,w_shape[0],w_shape[1]*w_shape[2]*w_shape[3]]
    if isinstance(weights, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for k in range(w_shape[0]):
        for r in range(w_shape[2]):
            for s in range(w_shape[3]):
                for c in range(w_shape[1]):
                    ret[idx] = weights[k][c][r][s]
                    idx+=1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape).transpose(2,3)

def convert_act_2d_matrix(activation, kernel_y, kernel_x, stride_y, stride_x, pad_y, pad_x):
    """
    :param activation: Input PyTorch Tensor
    :type activation: class:`torch.Tensor`
    """
    N = activation.shape[0]
    C = activation.shape[1]
    H = activation.shape[2]
    W = activation.shape[3]

    OH = (int) ((H - kernel_y + 2*pad_y) // stride_y) + 1
    OW = ((W - kernel_x + 2*pad_x) // stride_x) + 1
    nrows = OH*OW
    ncols = C*kernel_x*kernel_y
    ret_shape = [1,N,nrows,ncols]
    if isinstance(activation, torch.Tensor):
        ret = torch.zeros(np.prod(ret_shape))
    else:
        ret = np.zeros(np.prod(ret_shape))
    idx = 0
    for n in range(N):
        for h in range(-1*pad_y, H+pad_y-kernel_y+1, stride_y):
            for w in range(-1*pad_x, W+pad_x-kernel_x+1, stride_x):
                for r in range(kernel_y):
                    for s in range(kernel_x):
                        for c in range(C):
                            h_offs = h+r
                            w_offs = w+s
                            pad = h_offs < 0 or h_offs >= H or w_offs < 0 or w_offs >= W
                            ret[idx] = 0 if pad else activation[n][c][h_offs][w_offs]
                            idx+=1
    assert idx == np.prod(ret_shape)
    return ret.reshape(ret_shape)

def tilize(x):
    """
    This function tilizes a tensor. The last two tensor dims must be divisible by 32, after which this function
    produces row major tiles and creates faces. The output of this function is a flattened list that
    we can send to the device.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """
    nearest_32 = _nearest_32

    assert isinstance(x, (torch.Tensor, np.ndarray)), "Input to this function must be an instance of torch.Tensor or np.array"
    assert len(x.shape) == 4, "Only 4D tensors suppported"
    assert (x.shape[-2] % 32) == 0 and (x.shape[-1] % 32) == 0, "The last two dimensions of the tensor must be divisible by 32"

    if isinstance(x, torch.Tensor):
        ret = torch.zeros(np.prod(x.shape))
    else:
        ret = np.zeros(np.prod(x.shape))

    idx = 0
    for B in range(x.shape[0]):
        for C in range(x.shape[1]):
            for H in range(0, x.shape[2], 32):
                for W in range(0, x.shape[3], 32):
                    unfaced_tile = x[B, C, H:H + 32, W:W + 32]

                    face0 = unfaced_tile[:16, :16]
                    face1 = unfaced_tile[:16, 16:]
                    face2 = unfaced_tile[16:, :16]
                    face3 = unfaced_tile[16:, 16:]

                    for face in (face0, face1, face2, face3):
                        ret[idx:idx + 256] = face.reshape(-1)
                        idx += 256

    return ret.reshape(x.shape)

def tilize_to_list(x):
    """
    Tilize a PyTorch and then return the values as a flat list. The last two
    tensor dims must be divisible by 32, after which this function produces row
    major tiles and creates faces.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """

    return tilize(x).reshape(-1).tolist()

def untilize(x):
    """
    This function untilizes a tensor to row major format.

    :param x: Input PyTorch Tensor
    :type x: class:`torch.Tensor`

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """
    nearest_32 = _nearest_32

    assert isinstance(x, (torch.Tensor, np.ndarray)), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) == 4, "Only 4D tensors suppported"
    assert (x.shape[-2] % 32) == 0 and (x.shape[-1] % 32) == 0, "The last two dimensions of the tensor must be divisible by 32"

    if isinstance(x, torch.Tensor):
        ret = torch.zeros(x.shape)
    else:
        ret = np.zeros(x.shape)

    for B in range(x.shape[0]):
        for C in range(x.shape[1]):
            x_hw = x[B,C,:].reshape(-1)
            hw = 0
            for h in range(0, x.shape[2], 32):
                for w in range(0, x.shape[3], 32):
                    f_tile = x_hw[hw:hw+256].reshape(16, 16)
                    ret[B, C, h:h+16, w:w+16] = f_tile

                    f_tile = x_hw[hw+256:hw+512].reshape(16, 16)
                    ret[B, C, h:h+16, w+16:w+32] = f_tile

                    f_tile = x_hw[hw+512:hw+768].reshape(16, 16)
                    ret[B, C, h+16:h+32, w:w+16] = f_tile

                    f_tile = x_hw[hw+768:hw+1024].reshape(16, 16)
                    ret[B, C, h+16:h+32, w+16:w+32] = f_tile
                    hw += 1024 # traverse tiles in RM-order

    return ret


def print_diff_argmax(a, b, annotation = ""):
    """
    Prints out the value of both tensors at a point where the absolute difference is the largest.
    """
    absdiff = (a-b).abs()
    argmax = absdiff.argmax().item()
    diff = absdiff.reshape(-1)[argmax]
    rela = a.abs()/(torch.max(a.abs(), b.abs()))
    relb = b.abs()/(torch.max(a.abs(), b.abs()))
    print("Abs diff=", diff, " at ", argmax, " --- ", annotation)
    print("  (a=", a.reshape(-1)[argmax].item(), ")")
    print("  (b=", b.reshape(-1)[argmax].item(), ")")
    print("  Rel a=", rela.reshape(-1)[argmax], " at ", argmax)
    print("  Rel b=", relb.reshape(-1)[argmax], " at ", argmax)
    return diff.item()


def tt2torch(ttx):
    """
    Converts an llbuda tiled tensor to torch tensor.
    """
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    host = ttm.device.GetHost()
    shp = ttx.shape()
    tt_out = ttx.to(host)
    torch_out = untilize(torch.Tensor(tt_out.data()).reshape(shp))
    return torch_out


def tt2torch_rm(ttx):
    """
    Converts an llbuda row-major tensor to torch tensor.
    """
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    host = ttm.device.GetHost()
    shp = ttx.shape()
    tt_out = ttx.to(host)
    torch_out = torch.Tensor(tt_out.data()).reshape(shp)
    return torch_out


def divup(a, b):
    return (a+b-1)//b


def roundup(a, b):
    result = divup(a, b)*b
    return result


def roundup32(a):
    return roundup(a, 32)


def float_to_bits(x):
    s = struct.pack('>f', x)
    return struct.unpack('>l', s)[0]
