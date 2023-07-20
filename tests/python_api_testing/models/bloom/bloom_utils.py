import torch
import json
import tt_lib


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = tt_lib.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    ).to(tt_device)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = tt_lib.device.GetHost()
    tt_output = tt_tensor.to(host)
    if tt_output.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(tt_lib.tensor.Layout.ROW_MAJOR)
    dtype = {
        tt_lib.tensor.DataType.FLOAT32:   torch.float,
        tt_lib.tensor.DataType.BFLOAT16:  torch.bfloat16,
        tt_lib.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_tensor.dtype()]

    py_output = torch.frombuffer(tt_output.data(), dtype=dtype).reshape(tt_output.shape())
    return py_output


def tt_const_tensor(value, shape, device):
    pytorch_const = torch.full(shape, value)
    tt_const = torch2tt_tensor(pytorch_const, device)
    return tt_const


def pad_input_tensor(tensor, value, multiple):
    len = tensor.shape[1]

    if len % multiple == 0:
        return tensor

    padded_len = ((len // multiple) + 1) * multiple

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def tt_matmul(t1, t2, device, on_torch=False):
    if on_torch:
        t1 = tt2torch_tensor(t1)
        t2 = tt2torch_tensor(t2)

        res = torch.matmul(t1, t2)
        return torch2tt_tensor(res, device)
    else:
        return tt_lib.tensor.bmm(t1, t2)


def tt_bmm(t1, t2, device, on_torch=False):
    if on_torch:
        return tt_matmul(t1, t2, device)
    else:
        return tt_lib.tensor.bmm(t1, t2)


def read_model_config(json_file):
    # read file
    with open(json_file, "r") as myfile:
        data = myfile.read()

    # parse file
    obj = json.loads(data)
    return obj
