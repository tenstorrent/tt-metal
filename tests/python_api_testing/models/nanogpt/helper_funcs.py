import torch
import tt_lib

def tt_linear(x, weight, bias=None):
    #weight = tt_lib.tensor.transpose(weight)
    x = tt_lib.tensor.matmul(x, weight)
    if bias is not None:
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
    return x

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
