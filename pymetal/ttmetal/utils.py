import math
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
