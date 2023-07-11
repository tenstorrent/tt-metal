import torch
import tt_lib as ttl
from tests.python_api_testing.models.utility_functions_new import (
    comp_allclose_and_pcc,
    comp_pcc,
)
from loguru import logger
import pytest


@pytest.mark.parametrize(
    "input_shape, num_features",
    ((torch.Size([1, 2, 3, 4]), 2), (torch.Size([2, 6, 12, 6]), 6)),
)
@pytest.mark.parametrize("momentum", (0.1, 0.2))
@pytest.mark.parametrize("eps, affine, track_running_stats", ((1e-5, True, True),))
@pytest.mark.parametrize("on_device", (False, True))
def test_BatchNorm_fallback(
    input_shape,
    num_features,
    eps,
    momentum,
    affine,
    track_running_stats,
    on_device,
):
    torch.manual_seed(1234)
    host = ttl.device.GetHost()
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    x = torch.randn(input_shape).bfloat16().float()
    w = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    b = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    r_m = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    r_v = torch.randn([1, 1, 1, num_features]).bfloat16().float()
    n_b_t = torch.randint(0, 200, [1, 1, 1, 1]).bfloat16().float()
    pt_nn = torch.nn.BatchNorm2d(
        num_features, eps, momentum, affine, track_running_stats
    )
    pt_nn.weight = torch.nn.Parameter(w.reshape([num_features]))
    pt_nn.bias = torch.nn.Parameter(b.reshape([num_features]))
    pt_nn.running_mean = r_m.reshape([num_features])
    pt_nn.running_var = r_v.reshape([num_features])
    pt_nn.n_b_t = torch.tensor(n_b_t.item())
    pt_out = pt_nn(x)

    # Test on host RM
    t0 = ttl.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        t0 = t0.to(device)

    w0 = ttl.tensor.Tensor(
        w.reshape(-1).tolist(),
        w.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        w0 = w0.to(device)

    b0 = ttl.tensor.Tensor(
        b.reshape(-1).tolist(),
        b.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        b0 = b0.to(device)

    r_m0 = ttl.tensor.Tensor(
        r_m.reshape(-1).tolist(),
        r_m.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        r_m0 = r_m0.to(device)

    r_v0 = ttl.tensor.Tensor(
        r_v.reshape(-1).tolist(),
        r_v.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )
    if on_device:
        r_v0 = r_v0.to(device)

    # Scaler must remain on host
    n_b_t0 = ttl.tensor.Tensor(
        n_b_t.reshape(-1).tolist(),
        n_b_t.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    )

    tt_nn = ttl.fallback_ops.BatchNorm2d(
        w0,
        b0,
        r_m0,
        r_v0,
        n_b_t0,
        num_features,
        eps,
        momentum,
        affine,
        track_running_stats,
    )
    t1 = tt_nn(t0)

    output = torch.Tensor(t1.to(host).to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        t1.shape()
    )
    comp_pass, _ = comp_pcc(pt_out, output, 0.9999)
    _, comp_out = comp_allclose_and_pcc(pt_out, output)
    logger.info(comp_out)
    assert comp_pass

    del t1

    ttl.device.CloseDevice(device)
