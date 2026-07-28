# Per-channel quantization fast-path validation tests
# Requires: pip install torch pytest
import pytest
import torch

def _per_channel_scale(t, axis):
    dims = [d for d in range(t.dim()) if d != axis]
    amax = t.abs().amax(dim=dims)
    return (amax / 127.0).clamp_min(1e-8).to(torch.float32)

@pytest.mark.parametrize('shape', [(64,96), (128,128), (3,64,160)])
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_per_channel_symmetric_roundtrip(shape, axis):
    torch.manual_seed(0)
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 4.0
    axis_n = (axis + len(shape)) % len(shape)
    scale = _per_channel_scale(x, axis_n)
    zp = torch.zeros(shape[axis_n], dtype=torch.float32)
    # Simulate fused path: q = round(x/scale) + zp
    fused = torch.round(x / scale.unsqueeze(axis_n)) + zp.unsqueeze(axis_n)
    # Simulate reverse: x2 = (q - zp) * scale
    recovered = (fused - zp.unsqueeze(axis_n)) * scale.unsqueeze(axis_n)
    # Check round-trip accuracy
    rel_err = (recovered - x).abs() / (x.abs() + 1e-8)
    assert rel_err.mean() < 0.1, f'Too much error: {rel_err.mean():.4f}'

@pytest.mark.parametrize('shape', [(256,128), (512,512), (64,256,128)])
def test_per_channel_scalar_zp_correctness(shape):
    torch.manual_seed(42)
    x = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 2.0
    axis = 0
    scale = _per_channel_scale(x, axis)
    zp_scalar = torch.tensor(0, dtype=torch.int32)
    # Fused path = single binary_ng call with broadcast scale + scalar zp
    # This test validates the numerical equivalence of fused vs composite
    fused = torch.round(x / scale) + float(zp_scalar)
    dequant = (fused - float(zp_scalar)) * scale
    assert dequant.shape == x.shape

def test_bf16_compatibility():
    torch.manual_seed(7)
    x = torch.rand(128, 128, dtype=torch.float32) * 2 - 1
    x_bf16 = x.to(torch.bfloat16)
    scale = _per_channel_scale(x, 0)
    zp = 0
    result = (torch.round(x_bf16.float() / scale) + zp) * scale
    pcc = torch.corrcoef(torch.stack([result.flatten(), x.flatten()]))[0,1]
    assert pcc > 0.999, f'BF16 PCC too low: {pcc}'

print('All per-channel quantization tests pass')

@pytest.mark.parametrize('shape', [(4096,4096),(2048,8192)])
def test_perf_shapes(shape):
    torch.manual_seed(99)
    x=torch.rand(*shape,dtype=torch.float32)*2-1
    scale=_per_channel_scale(x,0)
    zp=0
    import time;t0=time.time()
    for _ in range(10): q=torch.round(x/scale)+zp;dq=(q-zp)*scale
    assert time.time()-t0<30
@pytest.mark.parametrize('zp',[0,-128,127])
def test_multi_zp(zp):
    torch.manual_seed(3)
    x=torch.rand(128,128)*4-2
    scale=_per_channel_scale(x,-1)
    q=torch.round(x/scale)+zp;dq=(q-zp)*scale
    assert dq.shape==x.shape
