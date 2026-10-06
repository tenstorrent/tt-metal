# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58724, #58723): the production ops whose compute kernels can opt in to the per-tile source
bank hand-off, at the shapes of their tests and model callers, for the device profiler (eb_prof_plugin repeats each test
function). Every case calls the op the way its tree test or model does; the tree test helpers are reused where they take a
device. Lives outside the tree; the device is opened here with an l1_small_size every case accepts."""
import pytest
import torch
import ttnn


@pytest.fixture(scope="module")
def device():
    from tests.tests_common.cache_entries_counter import CacheEntriesCounter

    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    dev.cache_entries_counter = CacheEntriesCounter(dev)
    yield dev
    ttnn.close_device(dev)


# ---- depthwise conv1d (compute_depthwise_conv1d.cpp): dest-reuse add per tap, standard mul ----
MAMBA = [(1, 5120, 5120, 32, 4, 1, 3, 5120), (1, 5120, 5120, 1024, 4, 1, 3, 5120), (1, 2560, 2560, 1027, 4, 1, 0, 2560)]


@pytest.mark.parametrize("shape", MAMBA, ids=["c5120_l32", "c5120_l1024", "c2560_l1027"])
def test_conv1d_mamba(device, shape):
    from tests.ttnn.unit_tests.operations.conv.test_conv1d import run_conv

    b, oc, ic, length, k, s, p, g = shape
    run_conv(device, ttnn.MathFidelity.LoFi, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, b, oc, ic, length, k, s, p,
             True, None, groups=g)


@pytest.mark.parametrize("channels, path", [(512, "coalesced"), (1280, "non-coalesced")])
def test_conv1d_depthwise_hifi4(device, channels, path):
    from tests.ttnn.nightly.unit_tests.operations.conv.test_conv1d import test_conv1d_depthwise_multi_height_block

    test_conv1d_depthwise_multi_height_block(device, channels, path)


# ---- KDA qkv_causal_conv1d_silu: dest-reuse add per tap ----
KDA = [((512, 512, 512), 1536, 64), ((1024, 1024, 1024), 768, 64), ((512, 256, 128), 896, 64), ((512, 512, 1536), 512, 2048)]


@pytest.mark.parametrize("widths, chunk, seq", KDA, ids=["single_block", "multiple_blocks", "asymmetric", "qwen36_t2048"])
def test_kda_qkv_conv1d_silu(device, widths, chunk, seq):
    from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
        make_actual_start,
        qkv_device_inputs,
        qkv_reference,
    )

    (inputs, history, taps), (input_tt, history_tt, taps_tt) = qkv_device_inputs(device, sequence=seq, widths=widths)
    out = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        input_tt, history_tt, *taps_tt, *widths, actual_start=make_actual_start(device, 0), predecessor_carry=history_tt,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=chunk),
    )
    ref = qkv_reference(inputs.float(), history.float(), tuple(t.float() for t in taps), widths)
    for o, r in zip(out, ref):
        got = ttnn.to_torch(o).float()
        assert torch.corrcoef(torch.stack([got.flatten(), r.flatten()]))[0, 1] > 0.999


# ---- deepseek_v3_b1 RMSNorm micro op (rmsnorm.hpp: mul_reuse_dest) ----
@pytest.mark.parametrize("width", [7168, 1536, 512])
@pytest.mark.parametrize("use_fp32", [True, False])
def test_dsv3_rmsnorm(device, width, use_fp32):
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_rmsnorm import test_rmsnorm

    test_rmsnorm(device, width, 1e-6, use_fp32)


# ---- group_norm: welford DRAM (welford_groupnorm.cpp), tile-reduction DRAM (groupnorm.cpp), sharded (groupnorm_sharded_v2,
# welford_groupnorm_sharded_v2) ----
SDXL_VAE = [(1, 512, 256, 256, 4), (1, 256, 512, 512, 12), (1, 128, 1024, 1024, 32)]


@pytest.mark.parametrize("n, c, h, w, nob", SDXL_VAE, ids=["vae_65536x512", "vae_262144x256", "vae_1048576x128"])
def test_group_norm_dram_welford(device, n, c, h, w, nob):
    from tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM import run_group_norm_DRAM

    run_group_norm_DRAM(device, n, c, h, w, 32, nob, 8, 8, "two_pass", True, perf_test_mode=True)


GN_DRAM_TILE = [(1, 512, 64, 64, 32, 1, 8, 8), (1, 1152, 128, 128, 32, 2, 8, 4), (1, 512, 128, 128, 32, 2, 8, 4)]


@pytest.mark.parametrize("shape", GN_DRAM_TILE, ids=["sd14_vae_512x64x64", "sdxl_refiner_1152x128x128", "512x128x128"])
def test_group_norm_dram_tile_reduction(device, shape):
    from tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM import run_group_norm_DRAM

    n, c, h, w, g, nob, cy, cx = shape
    run_group_norm_DRAM(device, n, c, h, w, g, nob, cy, cx, "tile_reduction", True, perf_test_mode=True)


SDXL_UNET = [(1, 1280, 64, 64), (1, 320, 128, 128), (1, 640, 64, 64), (1, 2560, 32, 32)]


@pytest.mark.parametrize("shape", SDXL_UNET, ids=["unet_1280x64x64", "unet_320x128x128", "unet_640x64x64", "unet_2560x32x32"])
def test_group_norm_sharded_sdxl(device, shape):
    from tests.ttnn.unit_tests.operations.fused.test_group_norm import test_sdxl_base_group_norm_bh

    test_sdxl_base_group_norm_bh(device, shape, True, perf_test_mode=True)


@pytest.mark.parametrize("shape", [(1, 320, 128, 128), (1, 640, 64, 64), (1, 512, 128, 128)], ids=["320x128x128", "640x64x64", "512x128x128_tile"])
def test_group_norm_sharded_welford(device, shape):
    from tests.ttnn.unit_tests.operations.fused.test_group_norm import run_sdxl_base_group_norm_test

    n, c, h, w = shape
    layout = ttnn.TILE_LAYOUT if c == 512 else ttnn.ROW_MAJOR_LAYOUT
    run_sdxl_base_group_norm_test(device, n, c, h, w, True, layout, layout != ttnn.TILE_LAYOUT, True, perf_test_mode=True)


# ---- layer_norm / rms_norm, interleaved (layernorm.cpp: square at HiFi4), large tensor (layernorm_large_tensor.cpp),
# welford large tensor (layernorm_large_tensor_welford.cpp) ----
@pytest.mark.parametrize("b, h, w", [(1, 32, 2880), (1, 32, 4096), (1, 128, 4096)], ids=["gptoss_decode", "llama_decode", "h128_w4096"])
def test_rms_norm_interleaved(device, b, h, w):
    from tests.ttnn.unit_tests.operations.fused.test_rms_norm import test_rms_norm

    test_rms_norm(device, b, h, w)


@pytest.mark.parametrize("h, w", [(32, 1280), (1504, 1280), (128, 4096)], ids=["whisper_dec", "whisper_enc", "h128_w4096"])
def test_layer_norm_interleaved(device, h, w):
    from tests.ttnn.unit_tests.operations.fused.test_layer_norm import test_layer_norm_with_weight_and_bias

    test_layer_norm_with_weight_and_bias(device, h, w, False)


@pytest.mark.parametrize("h, w", [(32, 8192), (128, 16384)], ids=["h32_w8192", "h128_w16384"])
def test_layer_norm_large(device, h, w):
    from tests.ttnn.unit_tests.operations.fused.test_layer_norm import test_large_layer_norm_with_weight_and_bias

    test_large_layer_norm_with_weight_and_bias(device, h, w, False)


@pytest.mark.parametrize("width, has_residual", [(16384, False), (16384, True)], ids=["w16384", "w16384_residual"])
def test_layer_norm_welford_large(device, width, has_residual):
    from tests.ttnn.nightly.unit_tests.operations.fused.test_two_pass_layer_norm import test_layer_norm_welford_large_offset

    test_layer_norm_welford_large_offset(device, width, has_residual)


# ---- softmax large tensor (softmax_large_tensor.cpp) ----
@pytest.mark.parametrize("b, h, w", [(1, 32, 32000), (1, 128, 128000)], ids=["32x32000", "128x128000"])
def test_softmax_large(device, b, h, w):
    from tests.ttnn.unit_tests.operations.fused.test_softmax import test_large_softmax

    test_large_softmax(device, b, h, w, -1)


def test_softmax_large_causal(device):
    from tests.ttnn.unit_tests.operations.fused.test_softmax import test_scale_mask_softmax_causal_large_kernel

    test_scale_mask_softmax_causal_large_kernel(device)


# ---- rotary embedding llama, prefill (rotary_embedding_llama.cpp: two multiplies per tile at HiFi4) ----
@pytest.mark.parametrize("seq, nh, nkv", [(128, 32, 8), (1024, 32, 8), (2048, 8, 1)], ids=["s128_32h", "s1024_32h", "s2048_8h"])
def test_rotary_llama_prefill(device, seq, nh, nkv):
    from tests.ttnn.nightly.unit_tests.operations.experimental.test_rotary_embedding_llama import (
        run_test_rotary_embedding_llama,
    )

    run_test_rotary_embedding_llama(device, 1, seq, 0.9997, nh, nkv, 128, max(4096, seq), ttnn.bfloat16)


# ---- hardswish (kernel_lib DestReuseBinary<Mul> at HiFi4) ----
@pytest.mark.parametrize("shape", [(3, 128, 32), (1, 3, 320, 384), (1, 1, 1024, 1024)], ids=["3x128x32", "1x3x320x384", "1024x1024"])
def test_hardswish(device, shape):
    x = torch.empty(shape, dtype=torch.bfloat16).uniform_(-3, 3)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.hardswish(t)).float()
    ref = torch.nn.functional.hardswish(x.float())
    assert torch.allclose(out, ref, atol=0.05, rtol=0.02)


# ---- prod (prod_all.cpp, prod_nc.cpp: chained dest-reuse multiplies at HiFi4) ----
@pytest.mark.parametrize("shape", [[1, 1, 32, 32], [1, 4, 32, 32], [16, 16]], ids=["1x1x32x32", "1x4x32x32", "16x16"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_prod_all(device, shape, dtype):
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    f = torch.tensor((1.5, 0.75, 1.25, 0.625, 2.0, 0.5))
    n = 1
    for d in shape:
        n *= d
    x = f[torch.randint(0, 6, (n,), generator=torch.Generator().manual_seed(7))].reshape(shape).to(tdt)
    out = ttnn.prod(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))
    assert torch.isfinite(ttnn.to_torch(out).float()).all()


@pytest.mark.parametrize("shape", [[2, 3, 191, 223], [9, 16, 415, 607], [8, 8, 127, 127]], ids=["2x3", "9x16", "8x8"])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_prod_nc(device, shape, dim, dtype):
    tdt = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    f = torch.tensor((1.5, 0.75, 1.25, 0.625, 2.0, 0.5))
    n = 1
    for d in shape:
        n *= d
    x = f[torch.randint(0, 6, (n,), generator=torch.Generator().manual_seed(7))].reshape(shape).to(tdt)
    out = ttnn.prod(ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device), dim=dim, keepdim=True)
    ref = torch.prod(x.double(), dim=dim, keepdim=True)
    assert torch.allclose(ttnn.to_torch(out).double(), ref, rtol=0.05, atol=1e-3, equal_nan=True)


# ---- batch_norm with fp32 accumulation off (batch_norm_kernel.cpp: dest-reuse multiplies) ----
@pytest.mark.parametrize("shape", [(3, 17, 47, 32), (8, 64, 128, 128)], ids=["3x17x47x32", "8x64x128x128"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_batch_norm_fpu(device, shape, training):
    n, c, h, w = shape
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.bfloat16) * 5 + 5
    mean = torch.rand((1, c, 1, 1), dtype=torch.bfloat16) * 6 + 4
    var = torch.rand((1, c, 1, 1), dtype=torch.bfloat16) * 16 + 4
    wt = torch.rand((1, c, 1, 1), dtype=torch.bfloat16) * 6 + 4
    bs = torch.rand((1, c, 1, 1), dtype=torch.bfloat16) * 6 + 4
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    out = ttnn.batch_norm(tt(x), running_mean=tt(mean), running_var=tt(var), weight=tt(wt), bias=tt(bs), training=training,
                          compute_kernel_config=cfg)
    ref = torch.nn.functional.batch_norm(x.float(), mean.flatten().float().clone(), var.flatten().float().clone(),
                                         wt.flatten().float(), bs.flatten().float(), training=training)
    got = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.99


# ---- SDXL transformer block layer_norm on an L1 interleaved tensor (layernorm.cpp: square at HiFi2), as tt_transformerblock.py ----
@pytest.mark.parametrize("h, w", [(4096, 640), (1024, 1280)], ids=["4096x640", "1024x1280"])
def test_layer_norm_sdxl_l1(device, h, w):
    torch.manual_seed(0)
    x = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    g = torch.rand((w,), dtype=torch.bfloat16)
    b = torch.rand((w,), dtype=torch.bfloat16)
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tg = ttnn.from_torch(g, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.layer_norm(tx, weight=tg, bias=tb, epsilon=1e-5, compute_kernel_config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG,
                          program_config=ttnn.LayerNormDefaultProgramConfig(legacy_reduction=True))
    ref = torch.nn.functional.layer_norm(x.float(), [w], g.float(), b.float(), eps=1e-5)
    got = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.999


# ---- bge_m3 balanced layernorm on Blackhole p100a (custom op compute.cpp: residual add, square at HiFi2), S512 ----
@pytest.mark.parametrize("batch", [8, 16])
def test_bge_balanced_layernorm(device, batch):
    from models.demos.wormhole.bge_m3.tt.custom_ops.balanced_layernorm.op import bge_balanced_layernorm

    torch.manual_seed(0)
    shape = (batch, 1, 512, 1024)
    x = torch.rand(shape, dtype=torch.bfloat16)
    r = torch.rand(shape, dtype=torch.bfloat16)
    g = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    b = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rm = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = bge_balanced_layernorm(tt(x), tt(r), rm(g), rm(b), eps=1e-5, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
    ref = torch.nn.functional.layer_norm((x + r).float(), [1024], g.flatten().float(), b.flatten().float(), eps=1e-5)
    got = ttnn.to_torch(out).float()
    assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.999


# ---- rotary_embedding_hf, prefill on interleaved tensors (rotary_embedding_hf.cpp: two multiplies per tile at HiFi4) ----
@pytest.mark.parametrize("shape", [(1, 71, 128, 64), (1, 32, 128, 128), (32, 71, 32, 64)], ids=["1x71x128x64", "1x32x128x128", "32x71x32x64"])
def test_rotary_hf_prefill(device, shape):
    from tests.tt_eager.python_api_testing.unit_testing.misc.test_rotary_embedding_hf import test_rotary_embedding_hf_prefill

    w, z, y, x = shape
    test_rotary_embedding_hf_prefill(w, z, y, x, 2048, False, False, ttnn.bfloat16, ttnn.bfloat16, device)


# ---- depthwise conv1d, coalesced read path at a K7 width the audio models use (HiFi4 default compute config) ----
@pytest.mark.parametrize("prepare_weights", [False])
def test_conv1d_coalesced_k7(device, prepare_weights):
    from tests.ttnn.unit_tests.operations.conv.test_conv1d import test_with_prepare_weights

    test_with_prepare_weights(device, 2, 512, 512, 1024, 7, 1, 3, 512, prepare_weights)
