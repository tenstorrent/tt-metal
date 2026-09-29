# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The three GDN prefill conv paths (KDA fused op, native ttnn.conv1d, FIR) against one torch reference.

Random data, no checkpoint. Data is replicated to every device of the TP mesh and asserted on device 0.
Run:
    MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_gdn_conv_paths.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp, replicate_to_device
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, kda_channel_chunk_size, kda_conv_prefill
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir

K = 4  # conv kernel width; the carry holds K-1 = 3 rows
PCC_VS_REF = 0.999
PCC_KDA_VS_NATIVE = 0.9999

# (T, kd, vd): the 27B TP-4 production shape (C = 2560) and a small one (C = 192).
SHAPES = [
    pytest.param(2048, 512, 1536, id="T2048-kd512-vd1536"),
    pytest.param(64, 64, 64, id="T64-kd64-vd64"),
]


def _ref_conv(x, hist, w):
    """Depthwise causal conv + SiLU in fp32: silu(sum_j w[:, j] * xpad[t + j]) with the K-1 history rows
    prepended, so tap j multiplies input row t - 3 + j. x [1,T,C], hist [1,3,C] bf16; w [C,K] bf16."""
    T = x.shape[1]
    xp = torch.cat([hist.float(), x.float()], dim=1)  # [1, K-1+T, C]
    return F.silu(sum(w[:, j].float() * xp[:, j : j + T, :] for j in range(K)))


def _split_qkv(t, kd, vd):
    return t[..., :kd], t[..., kd : 2 * kd], t[..., 2 * kd :]


def _dev0(t):
    """Device 0's copy of a replicated mesh tensor, as a torch tensor."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _pcc(golden, calculated):
    return comp_pcc(golden, calculated)[1]


def _random_inputs(T, C, seed):
    torch.manual_seed(seed)
    x = torch.randn(1, T, C).to(torch.bfloat16)
    hist = torch.randn(1, K - 1, C).to(torch.bfloat16)  # nonzero carry: exercises the history rows
    w = (torch.randn(C, K) * 0.3).to(torch.bfloat16)  # taps [C, K]; tap j multiplies row t-3+j
    return x, hist, w


def _to_l1(mesh, x):
    """qkv as the layer hands it to the conv: bf16 TILE in L1, replicated."""
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _taps(mesh, w):
    """The KDA / FIR tap contract: four [1, 1, C] bf16 TILE tensors in kernel-position order."""
    C = w.shape[0]
    return [replicate_to_device(mesh, w[:, j].reshape(1, 1, C).contiguous()) for j in range(K)]


def _actual_start(mesh):
    return ttnn.from_torch(
        torch.tensor([0], dtype=torch.int64),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _native_stub(mesh, w):
    """A TPGatedDeltaNet shell carrying only what _conv1d_prefill reads. conv_w1d is built exactly as the
    layer's loader does: host ROW_MAJOR [nd*C, 1, K] sharded on dim 0, here the same [C, 1, K] per device."""
    C = w.shape[0]
    nd = mesh.get_num_devices()
    stub = TPGatedDeltaNet.__new__(TPGatedDeltaNet)
    stub.mesh, stub.K, stub.qkv_dim_tp, stub._conv1d_wprep = mesh, K, C, None
    w1d = w.reshape(C, 1, K).contiguous().repeat(nd, 1, 1)
    stub.tw = {
        "conv_w1d": ttnn.from_torch(
            w1d,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
    }
    return stub


def _slice3(conv, T, kd, vd):
    """q | k | v column split of a [1, T, C] conv output (the non-fused paths' epilogue)."""
    C = 2 * kd + vd
    q = ttnn.slice(conv, (0, 0, 0), (1, T, kd))
    k = ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd))
    v = ttnn.slice(conv, (0, 0, 2 * kd), (1, T, C))
    ttnn.deallocate(conv)
    return q, k, v


def _check_contract(name, q, k, v, new_state, T, kd, vd, x):
    """Output contract shared by every path: q/k/v [1,T,kd|kd|vd] and new_state [1,3,C], all TILE in DRAM;
    new_state is bit-exactly the last K-1 rows of the bf16 input."""
    C = 2 * kd + vd
    for label, t, shape in (
        ("q", q, (1, T, kd)),
        ("k", k, (1, T, kd)),
        ("v", v, (1, T, vd)),
        ("new_state", new_state, (1, 3, C)),
    ):
        assert tuple(t.shape) == shape, f"{name} {label}: shape {tuple(t.shape)} != {shape}"
        assert t.layout == ttnn.TILE_LAYOUT, f"{name} {label}: layout {t.layout}"
        assert t.memory_config().buffer_type == ttnn.BufferType.DRAM, f"{name} {label}: {t.memory_config()}"
    assert torch.equal(_dev0(new_state), x[:, T - (K - 1) :, :]), f"{name}: new_state != last {K - 1} input rows"


def _compare(name, got, ref):
    """PCC (asserted) and max-abs (logged) of q/k/v against the reference triple. got/ref: fp32 torch."""
    pccs = [_pcc(r, g) for r, g in zip(ref, got)]
    mads = [(g - r).abs().max().item() for r, g in zip(ref, got)]
    logger.info(
        f"{name:7s}: pcc q/k/v {pccs[0]:.6f} {pccs[1]:.6f} {pccs[2]:.6f} | max-abs q/k/v "
        f"{mads[0]:.3e} {mads[1]:.3e} {mads[2]:.3e}"
    )
    for label, p in zip("qkv", pccs):
        assert p >= PCC_VS_REF, f"{name} {label}: pcc {p:.6f} < {PCC_VS_REF}"
    return pccs, mads


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("T, kd, vd", SHAPES)
def test_conv_paths_match_reference(mesh_device, T, kd, vd, reset_seeds, ensure_gc, request):
    """KDA fused op, native ttnn.conv1d (the layer's _conv1d_prefill) and the FIR on the same L1 qkv and the
    same nonzero TILE carry: each within PCC of the fp32 reference, KDA and native agree more tightly, and
    all three honour the same output contract."""
    mesh = mesh_device
    C = 2 * kd + vd
    x, hist, w = _random_inputs(T, C, seed=223)
    ref = [r.contiguous() for r in _split_qkv(_ref_conv(x, hist, w), kd, vd)]

    qkv = _to_l1(mesh, x)
    carry = replicate_to_device(mesh, hist)  # TILE DRAM, as the layer's conv_carry
    taps = _taps(mesh, w)
    start = _actual_start(mesh)
    stub = _native_stub(mesh, w)

    def run_kda():
        return kda_conv_prefill(qkv, T, carry, taps, (kd, kd, vd), start)

    def run_native():
        conv, ns = TPGatedDeltaNet._conv1d_prefill(stub, qkv, T, carry)
        return (*_slice3(conv, T, kd, vd), ns)

    def run_fir():
        conv, ns = _causal_conv1d_fir(
            qkv,
            None,
            None,
            K,
            mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            conv_state=carry,
            weight_taps=taps,
            bias_dev=None,
            valid_len=None,
        )
        return (*_slice3(conv, T, kd, vd), ns)

    outs = {}
    for name, fn in (("kda", run_kda), ("native", run_native), ("fir", run_fir)):
        q, k, v, ns = fn()
        _check_contract(name, q, k, v, ns, T, kd, vd, x)
        outs[name] = [_dev0(t).float() for t in (q, k, v)]
        for t in (q, k, v, ns):
            ttnn.deallocate(t)
        _compare(name, outs[name], ref)

    kn_pccs = [_pcc(a, b) for a, b in zip(outs["kda"], outs["native"])]
    kn_mad = max((a - b).abs().max().item() for a, b in zip(outs["kda"], outs["native"]))
    logger.info(f"kda vs native: pcc q/k/v {kn_pccs[0]:.6f} {kn_pccs[1]:.6f} {kn_pccs[2]:.6f} | max-abs {kn_mad:.3e}")
    assert min(kn_pccs) >= PCC_KDA_VS_NATIVE, f"kda vs native pcc {min(kn_pccs):.6f} < {PCC_KDA_VS_NATIVE}"


@torch.no_grad()
@parametrize_mesh_tp()
def test_kda_carry_across_chunks(mesh_device, reset_seeds, ensure_gc, request):
    """Two consecutive T-row chunks through kda_conv_prefill: chunk 1 from an all-zero ROW_MAJOR history (the
    layer's _kda_zero_history), chunk 2 from chunk 1's TILE new_state. Both must match the reference computed
    over the concatenated 2T rows."""
    mesh = mesh_device
    T, kd, vd = 2048, 512, 1536
    C = 2 * kd + vd
    x, _, w = _random_inputs(2 * T, C, seed=224)
    zeros = torch.zeros(1, K - 1, C, dtype=torch.bfloat16)
    ref_full = _ref_conv(x, zeros, w)  # [1, 2T, C]

    history = ttnn.from_torch(
        zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    taps = _taps(mesh, w)
    start = _actual_start(mesh)

    for ci in range(2):
        xc = x[:, ci * T : (ci + 1) * T, :]
        qkv = _to_l1(mesh, xc)
        q, k, v, ns = kda_conv_prefill(qkv, T, history, taps, (kd, kd, vd), start)
        ttnn.deallocate(qkv)
        _check_contract(f"kda chunk{ci}", q, k, v, ns, T, kd, vd, xc)
        got = [_dev0(t).float() for t in (q, k, v)]
        for t in (q, k, v):
            ttnn.deallocate(t)
        ref = [r.contiguous() for r in _split_qkv(ref_full[:, ci * T : (ci + 1) * T, :], kd, vd)]
        _compare(f"chunk{ci}", got, ref)
        history = ns  # TILE: chunk 2 takes the TILE -> ROW_MAJOR branch of kda_conv_prefill


@pytest.mark.parametrize(
    "channels, expected",
    [(2560, 512), (1280, 320), (5120, 512), (96, 96), (64, 64)],
)
def test_kda_channel_chunk_size(channels, expected):
    """Largest tile-aligned divisor of the channel count not above the cap (pure python)."""
    assert kda_channel_chunk_size(channels) == expected


def test_kda_channel_chunk_size_rejects_unaligned(expect_error):
    with expect_error(ValueError, "no tile-aligned channel chunk divides 100"):
        kda_channel_chunk_size(100)
