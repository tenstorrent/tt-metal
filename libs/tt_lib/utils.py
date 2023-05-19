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
    HT = a.shape[-2] // 32
    WT = a.shape[-1] // 32
    hwt = argmax//1024
    wt = hwt % WT
    ht = hwt // WT
    h = (argmax % 1024) // 32
    w = (argmax % 1024) % 32
    print("Abs diff=", diff, " at ", argmax, " --- ", annotation, "HTWT=", ht, wt, "HW=", h, w)
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

def read_conv_act_into_mm_act_block(conv_act, in0_address_map_index, address_map, in0_block_h, in0_block_w):
    mm_act_block_shape = [1,1,in0_block_h*32, in0_block_w*32]
    mm_act_block_size = in0_block_h*in0_block_w*1024
    mm_act_block = torch.zeros(mm_act_block_size, dtype=torch.bfloat16).float()
    total_read_size = 0
    num_reads = 0
    while(total_read_size != mm_act_block_size):
        src_address = address_map[in0_address_map_index]
        dst_address = address_map[in0_address_map_index+1]
        read_size = address_map[in0_address_map_index+2]
        pad = address_map[in0_address_map_index+3]
        for s in range(read_size):
            assert(dst_address+s < mm_act_block_size)
            if pad:
                mm_act_block[dst_address+s] = 0
            else:
                assert(src_address+s < len(conv_act))
                mm_act_block[dst_address+s] = conv_act[src_address+s]
        total_read_size += read_size
        in0_address_map_index += 4
        num_reads += 1
    return (mm_act_block.reshape(mm_act_block_shape), in0_address_map_index)

def read_conv_weight_into_mm_act_block(mm_weight, in1_row_start_tile_id,
                                        in1_current_block_start_tile_id, in1_tile_stride_h,
                                        in1_block_stride_h, in1_block_h, in1_block_w):
    mm_weight_block_shape = [1,1,in1_block_h*32, in1_block_w*32]
    mm_weight_block_size = in1_block_h*in1_block_w*1024
    mm_weight_block = torch.zeros(mm_weight_block_size, dtype=torch.bfloat16).float()
    dst_address = 0
    for weight_tile_h in range(in1_block_h):
        in1_tile_id = in1_row_start_tile_id
        for weight_tile_w in range(in1_block_w):
            src_address = in1_tile_id * 1024; # Tile size = 1024
            for s in range(1024):
                assert(src_address+s < len(mm_weight))
                assert(dst_address+s < mm_weight_block_size)
                mm_weight_block[dst_address+s] = mm_weight[src_address+s]
            in1_tile_id += 1
            dst_address += 1024
        in1_row_start_tile_id += in1_tile_stride_h
    in1_current_block_start_tile_id += in1_block_stride_h
    return (mm_weight_block.reshape(mm_weight_block_shape), in1_row_start_tile_id, in1_current_block_start_tile_id)

def blocked_mm_with_conv_act(conv_act,
                            mm_weight,
                            address_map,
                            num_blocks_in0_h,
                            num_blocks_in0_w,
                            num_blocks_in1_w,
                            in0_block_h,
                            in0_block_w,
                            in1_block_w,
                            in1_tile_stride_h,
                            in1_block_stride_h,
                            in1_block_stride_w):
    # in0 refers to conv activation tensor
    # in1 refers to conv weight tensor
    mm_output_shape = [1,1,num_blocks_in0_h*in0_block_h*32,num_blocks_in1_w*in1_block_w*32]
    ret = torch.zeros(mm_output_shape, dtype=torch.bfloat16).float()
    mm_output_block_shape = [1,1,in0_block_h*32, in1_block_w*32]
    in0_address_map_start_block_index = 0
    in0_address_map_index = 0
    in1_block_h = in0_block_w
    for block_in0_h in range(num_blocks_in0_h):
        # Reset in1 (weight) to the starting tile in this column
        in1_start_tile_id = 0
        in0_address_map_start_block_index = in0_address_map_index
        for block_in1_w in range(num_blocks_in1_w):
            in1_current_block_start_tile_id = in1_start_tile_id
            # Reset in0 (activation) to starting block in this row
            in0_address_map_index = in0_address_map_start_block_index
            output_block = torch.zeros(mm_output_block_shape, dtype=torch.bfloat16).float()
            for block_in0_w in range(num_blocks_in0_w):
                in1_row_start_tile_id = in1_current_block_start_tile_id
                (mm_act_block, in0_address_map_index) = read_conv_act_into_mm_act_block(conv_act, in0_address_map_index,
                                                    address_map, in0_block_h, in0_block_w)
                (mm_weight_block, in1_row_start_tile_id, in1_current_block_start_tile_id) = read_conv_weight_into_mm_act_block(mm_weight, in1_row_start_tile_id,
                                                    in1_current_block_start_tile_id, in1_tile_stride_h,
                                                    in1_block_stride_h, in1_block_h, in1_block_w)
                # Untilize weight block (this CPU reference does matmul on untilized blocks)
                mm_weight_block = untilize(mm_weight_block)
                for out_h_block in range(in0_block_h*32):
                    for out_w_block in range(in1_block_w*32):
                        output_block[0][0][out_h_block][out_w_block] += torch.dot(mm_act_block[0,0,out_h_block,:].reshape(-1), mm_weight_block[0,0,:,out_w_block].reshape(-1))
            in1_start_tile_id += in1_block_stride_w
            start_oh = block_in0_h * in0_block_h * 32
            start_ow = block_in1_w * in1_block_w * 32
            end_oh = start_oh + (in0_block_h * 32)
            end_ow = start_ow + (in1_block_w * 32)
            ret[0,0,start_oh:end_oh,start_ow:end_ow] = output_block

    return ret


def is_close(a, b, rtol=8e-2, atol=8e-2, max_mag=4.0, max_mag_fraction=0.02):
    """
    An improved variant of isclose, taking into account max magnitude in the sum, with logging.
    """
    absdiff = (a - b).abs()
    reldiff1 = (a.abs() / b.abs()) - 1.0
    reldiff2 = (a.abs() + 1.0) / (b.abs() + 1.0) - 1.0  # in case b.abs() is 0
    reldiff_or = torch.logical_or(reldiff1.abs() < rtol, reldiff2.abs() < rtol)
    max_mag_ok = absdiff < max_mag * max_mag_fraction

    or_abs_rel = torch.logical_or(absdiff < atol, reldiff_or)
    or_abs_rel = torch.logical_or(or_abs_rel, max_mag_ok)
    debug_index = or_abs_rel.to(torch.int32).argmin().item()
    if not or_abs_rel.reshape(-1)[debug_index]:
        print("****   is_close mismatch at index=", debug_index)
        print(a.reshape(-1)[debug_index])
        print(b.reshape(-1)[debug_index])
        print("****    reldiff1=", reldiff1.reshape(-1)[debug_index])
        print("****    reldiff2=", reldiff2.reshape(-1)[debug_index])
        print("****    absdiff=", absdiff.reshape(-1)[debug_index])
        HT = a.shape[-2] // 32
        WT = a.shape[-1] // 32
        hwt = debug_index//1024
        wt = hwt % WT
        ht = hwt // WT
        h = (debug_index % 1024) // 32
        w = (debug_index % 1024) % 32
        print("****    at ", debug_index, " --- ", "HTWT=", ht, wt, "HW=", h, w)
    return torch.all(or_abs_rel)
