# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview iHC (#65891) at prefill shapes on the TP32 prefill mesh (#65884).

The prefill layout (#65884) on the physical 8x4 Blackhole torus (``single_bh_galaxy_torus_xy``): the residual's hidden
dimension is split over mesh axis 1 (4 chips); heads (8 per chip) and the prefill sequence are on axis 0 (8 chips), so
the residual is ``[S / 8, 6144 / 4]`` per chip. iHC is per token, so it runs with ``MeshPlan(mesh, hidden_axis=1)``
(hidden_tp = 4, the decode tests' rows4-sc1 case) and each chip simply holds T = S/8 of the tokens; the sequence split
never meets iHC's single all-reduce, which runs over the 4 chips of one mesh row.

Checked against the pure-torch reference (models/experimental/hy4_preview/reference/ihc.py, bit-exact with HF) with
the checkpoint's own iHC parameters for the attention and FFN sites of layers 0 and 1 and for the head. Streams are
fp32; the wrapped sublayer runs BF16 through ``sublayer_dtype``, as the prefill block does.

Two all-reduce paths: ``ttnn_all_reduce`` is the decode module's own ``MeshPlan`` (``ttnn.all_reduce``); ``prefill_ccl``
is ``tt/hy4/ihc.py``'s ``PrefillMeshPlan`` (all-gather on the prefill's persistent CCL semaphores + a local fp32 sum),
which the prefill block uses because ``ttnn.all_reduce`` allocates its semaphores in the prefill's 1536 B L1_SMALL.
The ``prefill_ccl`` case opens the mesh with that L1_SMALL size, as the prefill does; ``ttnn_all_reduce`` OOMs there
(TT_FATAL, L1_SMALL), so its case opens the default size.

Reported per case: PCC of the output streams, of the stream update (out - x, which the residual would otherwise
hide), of the sublayer input y, wall-clock per iHC site (device-synchronised, after a warm-up), and DRAM per chip.

Env: HY4_HF_MODEL=<tencent/Hy4-preview checkpoint dir> (the real-weight cases skip without it).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.hy4.ihc import ihc_plan
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCHead, IHCWrap
from models.experimental.hy4_preview.tests.common import hy4_tensors, pcc
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.ihc import TtIHCHead, TtIHCWrap

HIDDEN_AXIS = 1  # the residual's hidden: 4 chips
SEQ_AXIS = 0  # heads and the prefill sequence: 8 chips
PCC_STREAMS = 0.9999
PCC_UPDATE = 0.999
TIMING_ITERS = 5
L1_SMALL = 1536  # the prefill's (test_prefill_transformer_chunked.GLM_L1_SMALL_SIZE)

# Tokens per chip: S / 8. 512 = one 4K chunk; 2048 = 16K; 4096 = 32K.
T_PER_CHIP = [512, 2048, 4096]

SITES = [
    ("L0.attn", "model.layers.0.hc_attn_layer.hc_pre"),
    ("L0.ffn", "model.layers.0.hc_mlp_layer.hc_pre"),
    ("L1.attn", "model.layers.1.hc_attn_layer.hc_pre"),
    ("L1.ffn", "model.layers.1.hc_mlp_layer.hc_pre"),
]


def _streams(T, cfg, seed):
    """Synthetic residual streams [1, T, n, C]: four different streams with a few large outlier channels, the shape
    the real residual has after a few layers (|x| up to ~1e2 on a handful of hidden channels)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, cfg.n, cfg.dim, generator=g)
    x *= torch.tensor([1.0, 0.7, 1.3, 0.5]).view(1, 1, cfg.n, 1)
    hot = torch.randint(0, cfg.dim, (8,), generator=g)
    x[..., hot] *= 40.0
    return x


def _pack(x, tp):
    """[1, T, n, C] -> [1, 1, T, n*C] in chip-major-then-stream column order, so a contiguous 1/tp split of the last
    dim hands each chip its own hidden slice of every stream (models/experimental/hy4_preview/tests/test_ihc.py)."""
    _, T, n, C = x.shape
    return x.reshape(1, T, n, tp, C // tp).permute(0, 1, 3, 2, 4).reshape(1, 1, T, n * C)


def _unpack(d, tp, n):
    _, _, T, _ = d.shape
    return d.reshape(1, T, tp, n, -1).permute(0, 1, 3, 2, 4).reshape(1, T, n, -1)


def _mapper(mesh_device):
    """Hidden (dim 3) over mesh axis 0, tokens (dim 2) over mesh axis 1."""
    dims = [None, None]
    dims[HIDDEN_AXIS], dims[SEQ_AXIS] = 3, 2
    return ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))


def _composer(mesh_device):
    dims = [None, None]
    dims[HIDDEN_AXIS], dims[SEQ_AXIS] = 3, 2
    return ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))


def _up(mesh_device, t):
    return ttnn.from_torch(
        t.float().contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.float32,
        mesh_mapper=_mapper(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _down(mesh_device, t):
    return ttnn.to_torch(t, mesh_composer=_composer(mesh_device)).float()


def _dram_mb(mesh_device):
    """DRAM allocated per chip (MB; the allocator is lock-step, so one chip stands for all)."""
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return v.total_bytes_allocated_per_bank * v.num_banks / 2**20


def _timed(mesh_device, fn):
    out = fn()  # warm-up (program compile)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(TIMING_ITERS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / TIMING_ITERS * 1e3


def _ref_sublayer(y):
    """What the device sublayer computes: SiLU in BF16 (y is cast to BF16 going in)."""
    return torch.nn.functional.silu(y.to(torch.bfloat16).float()).to(torch.bfloat16).float()


@pytest.mark.parametrize(
    "mesh_device, device_params, reduce",
    [
        # The prefill's 1536 B L1_SMALL with the prefill block's all-reduce.
        pytest.param(
            (8, 4), torus_xy_device_params(l1_small_size=L1_SMALL), "prefill_ccl", id="8x4-hidden1-prefill_ccl"
        ),
        # The decode module's ttnn.all_reduce: it OOMs in 1536 B of L1_SMALL, so this case opens the default size.
        pytest.param((8, 4), torus_xy_device_params(), "ttnn_all_reduce", id="8x4-hidden1-ttnn_all_reduce"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("t_per_chip", T_PER_CHIP, ids=[f"T{t}" for t in T_PER_CHIP])
@pytest.mark.timeout(0)
def test_hy4_ihc_prefill_real_weights(mesh_device, device_params, t_per_chip, reduce):
    torch.manual_seed(0)
    cfg = IHCConfig()
    n, C = cfg.n, cfg.dim
    if reduce == "prefill_ccl":
        topo = per_axis_topology(device_params["fabric_config"])
        plan = ihc_plan(
            mesh_device, HIDDEN_AXIS, num_links=2, topology=topo[HIDDEN_AXIS] if isinstance(topo, tuple) else topo
        )
    else:
        plan = MeshPlan(mesh_device, hidden_axis=HIDDEN_AXIS)
    tp, sp = plan.hidden_tp, mesh_device.shape[SEQ_AXIS]
    assert tp == 4 and sp == 8, f"expected the 8x4 mesh with hidden over axis 1, got {tuple(mesh_device.shape)}"
    T = t_per_chip * sp

    x = _streams(T, cfg, seed=t_per_chip)
    dram0 = _dram_mb(mesh_device)
    tt_x = _up(mesh_device, _pack(x, tp))
    dram_x = _dram_mb(mesh_device) - dram0
    rows = []

    for name, prefix in SITES:
        fn, base, scale = hy4_tensors([f"{prefix}.hc_fn", f"{prefix}.hc_base", f"{prefix}.hc_scale"])
        ref = IHCWrap(cfg).eval()
        ref.fn.data, ref.base.data, ref.scale.data = fn, base, scale
        with torch.no_grad():
            ref_y, ref_post = ref.hc_pre(x)
            ref_out = ref.hc_post(_ref_sublayer(ref_y), x, ref_post)

        d0 = _dram_mb(mesh_device)
        wrap = TtIHCWrap(plan, cfg, fn, base, scale)
        dram_w = _dram_mb(mesh_device) - d0

        captured = {}

        def sublayer(h, keep=True):
            if keep:
                captured["y"] = _down(mesh_device, h)
            return ttnn.silu(h)

        tt_out = wrap.forward(tt_x, sublayer, sublayer_dtype=ttnn.bfloat16)
        dev_out = _unpack(_down(mesh_device, tt_out), tp, n)
        dram_out = _dram_mb(mesh_device) - d0 - dram_w
        ttnn.deallocate(tt_out)
        dev_y = captured["y"].reshape(1, T, C)

        p_streams = pcc(ref_out, dev_out)
        p_update = pcc(ref_out - x, dev_out - x)
        p_y = pcc(ref_y, dev_y)
        ms = _timed(mesh_device, lambda: wrap.forward(tt_x, lambda h: sublayer(h, keep=False), ttnn.bfloat16))
        rows.append((name, p_streams, p_update, p_y, ms, dram_w, dram_out))
        logger.info(
            f"[ihc prefill] {reduce} T/chip={t_per_chip} {name}: streams pcc={p_streams:.6f} update pcc={p_update:.6f} "
            f"y pcc={p_y:.6f} max|d|={(ref_out - dev_out).abs().max():.3e} | {ms:.2f} ms/site | "
            f"weights {dram_w:.2f} MB, out {dram_out:.2f} MB per chip"
        )
        assert p_streams >= PCC_STREAMS, f"{name}: streams pcc {p_streams}"
        assert p_update >= PCC_UPDATE, f"{name}: update pcc {p_update}"
        assert p_y >= PCC_UPDATE, f"{name}: y pcc {p_y}"
        del wrap

    # The sublayer alone, to separate it from the iHC cost in the per-site timing above.
    h_bf = ttnn.typecast(ttnn.slice(tt_x, [0, 0, 0, 0], [1, 1, t_per_chip, C // tp]), ttnn.bfloat16)
    ms_silu = _timed(mesh_device, lambda: ttnn.silu(h_bf))
    ttnn.deallocate(h_bf)

    # Head
    hfn, hbase, hscale = hy4_tensors(
        ["model.hc_head.hc_head_fn", "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale"]
    )
    ref_h = IHCHead(cfg).eval()
    ref_h.fn.data, ref_h.base.data, ref_h.scale.data = hfn, hbase, hscale
    with torch.no_grad():
        ref_head = ref_h(x)
    d0 = _dram_mb(mesh_device)
    head = TtIHCHead(plan, cfg, hfn, hbase, hscale)
    dram_hw = _dram_mb(mesh_device) - d0
    tt_head = head(tt_x)
    dev_head = _down(mesh_device, tt_head).reshape(1, T, C)
    ttnn.deallocate(tt_head)
    p_head = pcc(ref_head, dev_head)
    ms_head = _timed(mesh_device, lambda: head(tt_x))
    logger.info(
        f"[ihc prefill] {reduce} T/chip={t_per_chip} head: pcc={p_head:.6f} | {ms_head:.2f} ms | weights {dram_hw:.2f} MB"
    )

    table = "\n".join(
        f"  {r[0]:8s} streams {r[1]:.6f}  update {r[2]:.6f}  y {r[3]:.6f}  {r[4]:7.2f} ms  "
        f"fn/base/scale {r[5]:.2f} MB  out {r[6]:.2f} MB"
        for r in rows
    )
    logger.info(
        f"[ihc prefill] SUMMARY {reduce} mesh {tuple(mesh_device.shape)} hidden_axis={HIDDEN_AXIS} T/chip={t_per_chip} "
        f"(S={T}); streams in {dram_x:.2f} MB/chip; silu sublayer alone {ms_silu:.2f} ms\n{table}\n"
        f"  head     pcc {p_head:.6f}  {ms_head:7.2f} ms  weights {dram_hw:.2f} MB"
    )
    assert p_head >= PCC_UPDATE, f"head pcc {p_head}"
