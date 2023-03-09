import ttmetal

def test_add():

    device = ttmetal.device.CreateDevice(ttmetal.device.Arch.GRAYSKULL, 0)
    ttmetal.device.InitializeDevice(device)
    data = list(range(1024))
    t0 = ttmetal.tensor.Tensor(data, [1, 1, 32, 32], ttmetal.tensor.DataType.BFLOAT16, ttmetal.tensor.Layout.TILE, device)
    t1 = ttmetal.tensor.Tensor(data, [1, 1, 32, 32], ttmetal.tensor.DataType.BFLOAT16, ttmetal.tensor.Layout.TILE, device)

    t2 = ttmetal.tensor.add(t0, t1)

    t2.print(ttmetal.tensor.Layout.TILE)
    ttmetal.device.CloseDevice(device)
    return

if __name__ == "__main__":
    test_add()
