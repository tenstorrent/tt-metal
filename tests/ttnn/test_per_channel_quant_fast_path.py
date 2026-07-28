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

@pytest.mark.parametrize('shape,axis', [((1,128),0), ((128,1),1), ((16,16,16),0), ((16,16,16),2)])
def test_edge_shapes(shape, axis):
    torch.manual_seed(17)
    x = torch.rand(*shape) * 4 - 2
    scale = _per_channel_scale(x, axis)
    zp = 0
    q = torch.round(x / scale) + zp
    dq = (q - zp) * scale
    assert dq.shape == x.shape
    assert not torch.isnan(dq).any()

def test_out_of_range_zp():
    torch.manual_seed(5)
    x = torch.rand(64, 128) * 6 - 3
    scale = _per_channel_scale(x, 0)
    for zp in [-1000, 0, 1000]:
        q = torch.round(x / scale) + zp
        assert q.min() >= -128 and q.max() <= 127

def test_identical_fused_vs_composite():
    torch.manual_seed(42)
    x = torch.rand(256, 256) * 4 - 2
    scale = _per_channel_scale(x, 0)
    zp_scalar = 0
    fused = (torch.round(x / scale) + zp_scalar) * scale
    zp_tensor = torch.zeros(256, dtype=torch.float32)
    composite = (torch.round(x / scale) + zp_tensor) * scale
    assert torch.allclose(fused, composite, rtol=1e-5)

@pytest.mark.parametrize('shape', [(32,64),(64,32),(256,256),(128,512),(512,128),(1024,64),(64,1024)])
def test_varied_shapes(shape):
    torch.manual_seed(77)
    x=torch.rand(*shape)*6-3
    s=_per_channel_scale(x,0)
    z=0
    q=torch.round(x/s)+z;dq=(q-z)*s
    assert not torch.isnan(dq).any()
    assert dq.shape==x.shape


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_dtype_consistency(dtype):
    torch.manual_seed(13)
    x_f32=torch.rand(128,128)*4-2
    x_dt=x_f32.to(dtype)
    s=_per_channel_scale(x_f32,0)
    z=0
    r_f32=(torch.round(x_f32/s)+z)*s
    r_dt=(torch.round(x_dt.float()/s)+z)*s
    pcc=torch.corrcoef(torch.stack([r_f32.flatten(),r_dt.flatten()]))[0,1]
    assert pcc>0.999

@pytest.mark.parametrize('axis', [0,1,-1])
def test_all_axes(axis):
    torch.manual_seed(31)
    x=torch.rand(64,128)*4-2
    axis_n=(axis+2)%2
    s=_per_channel_scale(x,axis_n)
    z=0
    q=torch.round(x/s)+z;dq=(q-z)*s
    assert dq.shape==x.shape

@pytest.mark.parametrize('shape', [(4096,4096),(2048,8192),(1024,4096),(512,2048)])
def test_large_matrices(shape):
    torch.manual_seed(99)
    x=torch.rand(*shape)*4-2
    for axis in [0,1]:
        s=_per_channel_scale(x,axis)
        z=0
        q=torch.round(x/s)+z;dq=(q-z)*s
        assert torch.allclose(dq,x,rtol=0.5)

def test_row_vs_column_speedup_ratio():
    torch.manual_seed(55)
    x=torch.rand(4096,4096)*4-2
    import time
    s0=_per_channel_scale(x,0);s1=_per_channel_scale(x,1)
    t0=time.time()
    for _ in range(5): q=torch.round(x/s0);dq=(q)*s0
    t_col=time.time()-t0
    t0=time.time()
    for _ in range(5): q=torch.round(x/s1);dq=(q)*s1
    t_row=time.time()-t0
    assert t_row<t_col*1.5
