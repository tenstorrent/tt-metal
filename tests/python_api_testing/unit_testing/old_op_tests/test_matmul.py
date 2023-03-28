import pytest
from libs import tt_lib
import torch
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

@pytest.mark.skip(reason="This is an old test that needs to be deleted")
def test_matmul():

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()
    tt_lib.device.StartDebugPrintServer(device)

    torch.manual_seed(1234)

    batch = 3
    M = 32
    K = 64
    N = 96

    A = torch.randn((batch,1,M,K))
    B = torch.randn((1,1,K,N)) - 0.95

    t0 = tt_lib.tensor.Tensor(tilize_to_list(A), [batch, 1, M, K], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    t1 = tt_lib.tensor.Tensor(tilize_to_list(B), [1, 1, K, N], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)

    t2 = tt_lib.tensor.matmul(t0, t1)
    assert(t2.shape() == [batch, 1, M, N])
    tt_host_rm = t2.to(host).data()
    pyt_got_back = torch.Tensor(tt_host_rm).reshape((batch,1,M,N))
    pyt_got_back_rm = untilize(pyt_got_back)

    ref_bmm = torch.matmul(A.reshape(batch,M,K), B.reshape(1, K, N))
    ref_bmm = ref_bmm.reshape(batch, 1, M, N)
    maxmag = ref_bmm.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(pyt_got_back_rm, ref_bmm, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())

    # test bmm variant without bcast
    # batch*C are treated as batch
    C = 2
    A = torch.randn((batch,C,M,K))
    B = torch.randn((batch,C,K,N)) - 0.95

    t0 = tt_lib.tensor.Tensor(tilize_to_list(A), [batch, C, M, K], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
    t1 = tt_lib.tensor.Tensor(tilize_to_list(B), [batch, C, K, N], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)

    t2 = tt_lib.tensor.bmm(t0, t1)
    assert(t2.shape() == [batch, C, M, N])
    tt_host_rm = t2.to(host).data()
    pyt_got_back = torch.Tensor(tt_host_rm).reshape((batch,C,M,N))
    pyt_got_back_rm = untilize(pyt_got_back)

    ref_bmm = torch.bmm(A.reshape(batch*C,M,K), B.reshape(batch*C, K, N))
    ref_bmm = ref_bmm.reshape(batch, C, M, N)
    maxmag = ref_bmm.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(pyt_got_back_rm, ref_bmm, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())

    tt_lib.device.CloseDevice(device)
    return

if __name__ == "__main__":
    test_matmul()
