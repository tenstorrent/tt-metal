import ll_buda_bindings.ll_buda_bindings._C as _C
import torch
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close
#from python_api_testing.models.utility_functions import enable_compile_cache, enable_binary_cache

def test_matmul():

    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    _C.device.StartDebugPrintServer(device)

    torch.manual_seed(1234)

    batch = 3
    M = 32
    K = 64
    N = 96

    #enable_compile_cache()

    A = torch.randn((batch,1,M,K))
    B = torch.randn((1,1,K,N)) - 0.95

    t0 = _C.tensor.Tensor(tilize_to_list(A), [batch, 1, M, K], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    t1 = _C.tensor.Tensor(tilize_to_list(B), [1, 1, K, N], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)

    t2 = _C.tensor.matmul(t0, t1)
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

    t0 = _C.tensor.Tensor(tilize_to_list(A), [batch, C, M, K], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    t1 = _C.tensor.Tensor(tilize_to_list(B), [batch, C, K, N], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)

    t2 = _C.tensor.bmm(t0, t1)
    assert(t2.shape() == [batch, C, M, N])
    tt_host_rm = t2.to(host).data()
    pyt_got_back = torch.Tensor(tt_host_rm).reshape((batch,C,M,N))
    pyt_got_back_rm = untilize(pyt_got_back)

    ref_bmm = torch.bmm(A.reshape(batch*C,M,K), B.reshape(batch*C, K, N))
    ref_bmm = ref_bmm.reshape(batch, C, M, N)
    maxmag = ref_bmm.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(pyt_got_back_rm, ref_bmm, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())

    _C.device.CloseDevice(device)
    return

if __name__ == "__main__":
    test_matmul()
