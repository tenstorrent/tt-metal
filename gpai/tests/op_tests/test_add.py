from gpai import gpai

def test_add():

    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    data = list(range(1024))
    t0 = gpai.tensor.Tensor(data, [1, 1, 32, 32], gpai.tensor.DataFormat.FLOAT32, gpai.tensor.Layout.TILE, device)
    t1 = gapai.tensor.Tensor(data, [1, 1, 32, 32], gpai.tensor.DataFormat.FLOAT32, gpai.tensor.Layout.TILE, device)

    t2 = gpai.tensor.add(t0, t1)

    t2.print(gpai.tensor.Layout.TILE)
    gpai.device.CloseDevice(device)
    return

if __name__ == "__main__":
    test_add()
