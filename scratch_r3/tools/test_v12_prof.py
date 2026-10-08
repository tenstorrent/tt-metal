# Round 3 matmul, second review of #58712, #58713 (2026-10-07): production matmuls under the device profiler (one launch
# per repetition, tools/mm_prof_plugin.py), each with the program config its model sets (or ttnn's auto config):
# - kl: one-tile sub blocks at in0_block_w 8 (the k loop): Llama 3.1 8B FF2 at 128 tokens with the p100a config
#   (tt_transformers get_mlp_ff2_prg_config, dram_shard_grid_width 7), Qwen3.6 27B GDN out-projection in decode
#   (gdn_out_decode_1d_progcfg, TP 4 per device: 32 x 1536 x 5120).
# - es: one-tile k 1 rows (tt_transformers PREFILL_MLP_W*_PRG_CONFIG_128) and 2x2 sub blocks with a 32-bit DEST (auto
#   configs with even per-core blocks).
# - gdn: ttnn.transformer.chunk_gated_delta_rule at the Qwen3.6 27B TP 4 prefill slice (T 2048, Hk 4, Hv 12, 128/128,
#   flat q/k/v as the model passes them, the fused path); raw outputs under V12_OUT for a bit comparison across farms.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.path.isfile(_os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""))):
    _sys.exit("not under hwlock")
import os

import pytest
import torch
import ttnn

BF, B8, B4, F32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.float32
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    ttnn.SetDefaultDevice(dev)
    yield dev
    ttnn.close_device(dev)


def _ckc(dev, fid, fp32, approx=False, l1acc=True):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=FID[fid], math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc
    )


def _pc2d(grid, ibw, sh, sw, pcm, pcn, act=None, fb=False):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
        per_core_M=pcm, per_core_N=pcn, transpose_mcast=False, fused_activation=act, fuse_batch=fb,
    )


def _pc1d(grid, ibw, sh, sw, pcm, pcn, fb=True, mcast_in0=True):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=ibw, out_subblock_h=sh, out_subblock_w=sw,
        per_core_M=pcm, per_core_N=pcn, fuse_batch=fb, mcast_in0=mcast_in0,
    )


SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
# (name, M, K, N, in0 dtype, in1 dtype, out dtype, fidelity, fp32 dest, approx, program config or None for auto)
MM_CASES = [
    ("kl_llama8b_ff2_p100a_128", 128, 14336, 4096, B8, B8, BF, "HiFi2", False, False, lambda: _pc2d((8, 8), 8, 1, 1, 1, 19)),
    ("kl_qwen36_gdn_out_decode", 32, 1536, 5120, F32, B8, BF, "HiFi2", True, True, lambda: _pc1d((11, 3), 8, 1, 1, 1, 5)),
    ("es_mlp_w1_128", 128, 4096, 14336, BF, B8, BF, "LoFi", False, False, lambda: _pc2d((8, 8), 1, 1, 1, 1, 56, act=SILU)),
    ("es_mlp_w2_128", 128, 14336, 4096, BF, B8, B8, "LoFi", False, False, lambda: _pc2d((8, 8), 1, 1, 1, 1, 16)),
    ("es_auto_1k_4k_4k_bfp8_hifi2_fp32", 1024, 4096, 4096, BF, B8, BF, "HiFi2", True, False, None),
    ("es_auto_2k_4k_1k_bf16_hifi4_fp32", 2048, 4096, 1024, BF, BF, BF, "HiFi4", True, False, None),
    ("es_auto_2k_4k_4k_fp32_bfp8_lofi_fp32", 2048, 4096, 4096, F32, B8, BF, "LoFi", True, False, None),
]


@pytest.mark.parametrize("case", MM_CASES, ids=[c[0] for c in MM_CASES])
def test_v12_mm(device, case):
    name, m, k, n, d0, d1, do, fid, fp32, approx, pc = case
    torch.manual_seed(0)
    a = ttnn.from_torch(torch.randn(1, 1, m, k) * 0.1, dtype=d0, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(1, 1, k, n) * 0.1, dtype=d1, layout=ttnn.TILE_LAYOUT, device=device)
    kw = {"program_config": pc()} if pc else {}
    out = ttnn.matmul(a, b, compute_kernel_config=_ckc(device, fid, fp32, approx), dtype=do, **kw)
    ttnn.synchronize_device(device)
    if os.environ.get("V12_OUT"):
        torch.save(ttnn.to_torch(out).contiguous().view(torch.int16), os.path.join(os.environ["V12_OUT"], f"{name}.pt"))
    out.deallocate()
    a.deallocate()
    b.deallocate()


_GDN = {}


def test_v12_gdn(device):
    from tests.ttnn.unit_tests.operations.transformers.test_chunk_gated_delta_rule import _const_tiles

    B, T, Hk, Hv, D = 1, 2048, 4, 12, 128
    if not _GDN:
        torch.manual_seed(0)
        dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        q, k = torch.randn(B, T, Hk * D), torch.randn(B, T, Hk * D)
        v = torch.randn(B, T, Hv * D)
        beta = torch.sigmoid(torch.randn(B, T, Hv))
        g = -0.5 * torch.nn.functional.softplus(torch.randn(B, T, Hv))
        s0 = 0.05 * torch.randn(B, Hv, D, D)
        _GDN["in"] = (dev(q, BF), dev(k, BF), dev(v, BF), dev(g, F32), dev(beta, F32), dev(s0, F32))
        _GDN["const"] = _const_tiles(device)
    q, k, v, g, beta, s0 = _GDN["in"]
    eye, tril, ones, masks = _GDN["const"]
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        q, k, v, g, beta, scale=D**-0.5, initial_state=s0, output_final_state=True, chunk_size=32,
        output_head_major=True, eye=eye, tril=tril, ones=ones, masks=masks,
    )
    ttnn.synchronize_device(device)
    if os.environ.get("V12_OUT"):
        torch.save(
            (ttnn.to_torch(o).contiguous().view(torch.int32), ttnn.to_torch(fs).contiguous().view(torch.int32)),
            os.path.join(os.environ["V12_OUT"], "gdn_t2048.pt"),
        )
    o.deallocate()
    fs.deallocate()
