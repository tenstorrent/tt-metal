import ll_buda_bindings.ll_buda_bindings._C as _C

def test_bcast_add():

    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    data = list(range(1024))
    t0 = _C.tensor.Tensor(data, [1, 1, 32, 32], _C.tensor.Layout.TILE, device)
    t1 = _C.tensor.Tensor(data, [1, 1, 32, 32], _C.tensor.Layout.TILE, device)

    t2 = _C.tensor.bcast(t0, t1, _C.tensor.BcastOpMath.ADD, _C.tensor.BcastOpDim.H)

    t2.print(_C.tensor.Layout.ROW_MAJOR)
    _C.device.CloseDevice(device)
    return

if __name__ == "__main__":
    test_bcast_add()