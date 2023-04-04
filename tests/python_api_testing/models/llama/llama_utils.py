import torch
import json
from libs import tt_lib as ttl
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm


def torch2tt_tensor(py_tensor:torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = ttl.tensor.Tensor(
        py_tensor.reshape(-1).tolist(),
        size,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE).to(tt_device)

    return tt_tensor


def tt2torch_tensor(output):
    host = ttl.device.GetHost()
    tt_output = output.to(host).to(ttl.tensor.Layout.ROW_MAJOR)
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output


def tt_const_tensor(value, shape, device):
    if (len(shape)==4):
        number_tensor = torch.full(shape, value)
        tt_number_tensor = tilize_to_list(number_tensor)
        tt_number_tensor = ttl.tensor.Tensor(tt_number_tensor, number_tensor.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
        return tt_number_tensor
    else:
        s1 = 1
        s2 = shape[0]
        s3 = shape[1]
        s4 = shape[2]
        number_tensor = torch.full([s1, s2, s3, s4], value)
        tt_number_tensor = tilize_to_list(number_tensor)
        tt_number_tensor = ttl.tensor.Tensor(tt_number_tensor, number_tensor.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

        return tt_number_tensor


def read_model_config(json_file):
    # read file
    with open(json_file, 'r') as myfile:
        data=myfile.read()

    # parse file
    obj = json.loads(data)
    return obj


def print_corr_coef(x: torch.Tensor, y: torch.Tensor):
    x = torch.reshape(x, (-1, ))
    y = torch.reshape(y, (-1, ))

    input = torch.stack((x, y))

    corrval = torch.corrcoef(input)
    print(f"Corr coef:")
    print(f"{corrval}")
