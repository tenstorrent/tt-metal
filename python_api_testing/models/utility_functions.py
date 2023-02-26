import math

import torch
import numpy as np

from gpai import gpai

def nearest_32(x):
    return math.ceil(x / 32) * 32

def pad_activation(x):
    """
    This function pads an activation with 0s as a pre-preprocessing step to tilization.

    In the 2d case, it pads a vector to the right with 0s, and in the 2+d case,
    it pads the bottom and right corners of the last two dimensions.

    WARNING: This function should eventually be retired in favour of padding on device
    """
    assert isinstance(x, torch.Tensor), "Input to this function must be an instance of torch.Tensor"
    assert len(x.shape) >= 1 and len(x.shape) <= 4, "Only tensors with dimension 1-4 supported"
    if len(x.shape) == 1: # (num_features,)
        padded_tensor = torch.zeros(1, 1, 32, nearest_32(x.shape[0]))
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

    WARNING: This function should eventually be retired in favour of padding on device
    """
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

    WARNING: This function should eventually be retired in favour of fully tilizing on device.
    """
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
    return tilize(x).reshape(-1).tolist()

def untilize(x):
    """
    This function untilizes a tensor to row major format.
    """
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


def is_close(a, b, rtol=1e-2, atol=1e-2, max_mag = 2.0, max_mag_fraction = 0.02):
    """
    A variant of np.isclose with logging.
    """
    absdiff = (a-b).abs()
    reldiff1 = (a.abs() / b.abs()) - 1.0
    reldiff2 = (a.abs()+1.0) / (b.abs()+1.0) - 1.0 # in case b.abs() is 0
    reldiff_or = torch.logical_or(reldiff1.abs()<rtol, reldiff2.abs()<rtol)
    max_mag_ok = (absdiff<max_mag*max_mag_fraction)

    or_abs_rel = torch.logical_or( absdiff<atol, reldiff_or )
    or_abs_rel = torch.logical_or(or_abs_rel, max_mag_ok)
    debug_index = or_abs_rel.to(torch.int32).argmin().item()
    if not or_abs_rel.reshape(-1)[debug_index]:
        print("isclose mismatch at index=", debug_index)
        print(a.reshape(-1)[debug_index])
        print(b.reshape(-1)[debug_index])
        print("reldiff1=", reldiff1.reshape(-1)[debug_index])
        print("reldiff2=", reldiff2.reshape(-1)[debug_index])
        print("absdiff=", absdiff.reshape(-1)[debug_index])
    return torch.all( or_abs_rel )

def print_diff_argmax(a, b):
    """
    Prints out the value of both tensors at a point where the absolute difference is the largest.
    """
    absdiff = (a-b).abs()
    argmax = absdiff.argmax().item()
    diff = absdiff.reshape(-1)[argmax]
    rela = a.abs()/(torch.max(a.abs(), b.abs()))
    relb = b.abs()/(torch.max(a.abs(), b.abs()))
    print("Abs diff=", diff, " at ", argmax)
    print("Rel a=", rela.reshape(-1)[argmax], " at ", argmax)
    print("Rel b=", relb.reshape(-1)[argmax], " at ", argmax)
    return diff.item()


def get_oom_of_float(float_lst):
    """
    Given a list of floats, returns a list of the order or magnitudes
    of the floats. Useful when you want to make sure that even if your
    tt outputs don't match pytorch all that well, they are at least
    on the same order of magnitude
    """
    ooms = []
    for el in float_lst:
        str_el = str(el)
        if "e" in str_el:
            oom = int(str_el.split("e")[1])
        elif str_el[:2] == "0.":
            str_el = str_el.split(".")[1]

            oom = -1
            for e in str_el:
                if e != "0":
                    break
                oom -= 1
        else:
            oom = len(str_el.split(".")[0])

        ooms.append(oom)

    return ooms

def get_FR():
    # TODO(AP): a hacky workflow where we manually set force recompile counter before every kernel from python
    return gpai.device.GetForceRecompiles()

def set_FR(new_val):
    # TODO(AP): a hacky workflow where we manually set force recompile counter before every kernel from python
    host = gpai.device.SetForceRecompiles(new_val)
    print("Force recompiles=", get_FR())


def tt2torch(ttx):
    """
    Converts an llbuda tiled tensor to torch tensor.
    """
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    host = gpai.device.GetHost()
    shp = ttx.shape()
    tt_out = ttx.to(host)
    torch_out = untilize(torch.Tensor(tt_out.data()).reshape(shp))
    return torch_out

def tt2torch_rm(ttx):
    """
    Converts an llbuda row-major tensor to torch tensor.
    """
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    host = gpai.device.GetHost()
    shp = ttx.shape()
    tt_out = ttx.to(host)
    torch_out = torch.Tensor(tt_out.data()).reshape(shp)
    return torch_out
