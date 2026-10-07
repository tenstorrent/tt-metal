# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-chip test for TtFlatRoutedExpert (the flat_routed_expert op) at the routed-expert shapes of the models
that run it: Kimi-K2.7, GLM-5.3 and Kimi-K3.

Several local experts share one dispatch buffer laid out the way offset_cumsum lays it out (each expert's region
starts at a 32-row boundary), with ragged counts including an empty expert, and every expert's rows are graded
against TorchExpert. The second case builds the weights once from torch, then a second module from the per-expert
cache alone, which is the path a model run with a warm cache takes.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, ACTIVATION_SITU, TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_flat_routed_expert import TtFlatRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]

# (id, routed-expert K, I, device activation, torch activation). K3's routed experts run in the LatentMoE space.
MODELS = [
    (
        "kimi_k2_7",
        KimiK27Config.EMB_SIZE,
        KimiK27Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
        ACTIVATION_SILU,
    ),
    (
        "glm_53",
        GLM53Config.EMB_SIZE,
        GLM53Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.Silu,
        ACTIVATION_SILU,
    ),
    (
        "kimi_k3",
        KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        KimiK3Config.MOE_INTERMEDIATE_SIZE,
        ttnn.RoutedExpertActivation.SituGlu,
        ACTIVATION_SITU,
    ),
]

# Tensor-parallel slices of the K2 expert: the op splits these over 2 (I 512) and 3 (I 1024) core subgrids, a
# layout the three model shapes (all one subgrid) never reach. Not shipped by any model today; covered so the layout
# and the kernels stay right for them.
SUBGRID_SHAPES = [
    ("k2_tp4_i512", KimiK27Config.EMB_SIZE, 512, ttnn.RoutedExpertActivation.Silu, ACTIVATION_SILU),
    ("k2_tp2_i1024", KimiK27Config.EMB_SIZE, 1024, ttnn.RoutedExpertActivation.Silu, ACTIVATION_SILU),
]
SHAPES = MODELS + SUBGRID_SHAPES

# Per local expert token counts: ragged, one empty, one past a 128-row sub-block, one not a multiple of 32.
COUNTS = [200, 0, 37, 512, 1, 131]
MAX_TOKENS = 640  # dispatch capacity per expert; the op needs >= 256


_PREFIX = "layer_0.routed_expert"


def _write_expert_cache(weights, mesh_device, weights_dtype, cache_dir):
    """The per-expert routed-expert cache, as TtMoe.build_ttnn_cache writes it (TtRoutedExpert's tensorbins)."""
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

    TtRoutedExpert.build_ttnn_cache(weights, len(weights), mesh_device, weights_dtype, cache_dir, _PREFIX)


def _up(v, a):
    return -(-v // a) * a


def _bf4(w_out_in):
    """The weight as the device holds it: bfloat4_b tiles of the [in, out] matrix, back in HF [out, in] layout. The
    shared exponents are per tile row, so quantizing the transposed matrix is what makes the reference exact."""
    t = ttnn.from_torch(w_out_in.T.contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().T.contiguous()


def run_flat_routed_expert(mesh_device, emb_dim, hidden_dim, activation, torch_activation, cache_dir=None):
    E = len(COUNTS)
    regions = []
    r = 0
    for c in COUNTS:
        regions.append(r)
        r += _up(c, 32)
    rows = _up(r, 32) + 32

    torch.manual_seed(0)
    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim) * 0.02,
        }
        for _ in range(E)
    ]
    x = torch.zeros(rows, emb_dim)
    for c, r0 in zip(COUNTS, regions):
        x[r0 : r0 + c] = torch.randn(c, emb_dim)

    kw = dict(
        experts_per_chip=E,
        num_routed_experts=E,
        dispatch_group_size=1,
        num_dispatch_groups=1,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=MAX_TOKENS,
        weights_dtype=ttnn.bfloat4_b,
        weight_cache_path=cache_dir,
        cache_name_prefix=None if cache_dir is None else _PREFIX,
        activation=activation,
    )
    if cache_dir is None:
        expert = TtFlatRoutedExpert(mesh_device, torch_weights=weights, **kw)
    else:
        # From the per-expert cache alone, the path a model run with a warm routed-expert cache takes. The module
        # only reads caches; the test writes this one (to its own tmp dir) the way TtMoe's cache build does.
        _write_expert_cache(weights, mesh_device, ttnn.bfloat4_b, cache_dir)
        expert = TtFlatRoutedExpert(mesh_device, torch_weights=None, **kw)

    def u32(values):
        return ttnn.from_torch(
            torch.tensor([values], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_x = ttnn.from_torch(
        x,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_y = expert(tt_x, u32(COUNTS), u32(regions))
    y = ttnn.to_torch(tt_y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:rows].float()

    for e, (c, r0) in enumerate(zip(COUNTS, regions)):
        if c == 0:
            continue
        with torch.no_grad():
            ref = TorchExpert(
                emb_dim,
                hidden_dim,
                {k: _bf4(w) for k, w in weights[e].items()},
                activation=torch_activation,
                situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
                situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
            )(x[r0 : r0 + c])
        out = y[r0 : r0 + c]
        assert torch.isfinite(out).all(), f"expert {e}: non-finite output"
        _, pcc = comp_pcc(ref, out)
        logger.info(f"expert {e} ({c} tokens): PCC {pcc:.5f}")
        # Graded against the bf4 weights the device holds, so what is left is x's bfp8 packing and accumulation.
        assert pcc >= 0.999, f"expert {e} ({c} tokens): PCC {pcc:.5f}"


@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("model, emb_dim, hidden_dim, activation, torch_activation", SHAPES, ids=[m[0] for m in SHAPES])
@pytest.mark.skipif(not is_blackhole(), reason="flat_routed_expert is Blackhole-only")
def test_flat_routed_expert(mesh_device, device_params, model, emb_dim, hidden_dim, activation, torch_activation):
    run_flat_routed_expert(mesh_device, emb_dim, hidden_dim, activation, torch_activation)


@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("model, emb_dim, hidden_dim, activation, torch_activation", MODELS, ids=[m[0] for m in MODELS])
@pytest.mark.skipif(not is_blackhole(), reason="flat_routed_expert is Blackhole-only")
def test_flat_routed_expert_from_cache(
    mesh_device, device_params, model, emb_dim, hidden_dim, activation, torch_activation, tmp_path
):
    run_flat_routed_expert(mesh_device, emb_dim, hidden_dim, activation, torch_activation, cache_dir=tmp_path)


@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("model, emb_dim, hidden_dim, activation, torch_activation", MODELS, ids=[m[0] for m in MODELS])
@pytest.mark.skipif(not is_blackhole(), reason="flat_routed_expert is Blackhole-only")
def test_flat_routed_expert_placeholder(
    mesh_device, device_params, model, emb_dim, hidden_dim, activation, torch_activation
):
    """No torch weights and no cache (the perf legs): placeholder weights allocated in the op's layout must match
    the real layout tensor for tensor, and the op must run on them."""
    E = len(COUNTS)
    kw = dict(
        experts_per_chip=E,
        num_routed_experts=E,
        dispatch_group_size=1,
        num_dispatch_groups=1,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=MAX_TOKENS,
        weights_dtype=ttnn.bfloat4_b,
        activation=activation,
    )
    weights = [
        {
            "gate_proj": torch.zeros(hidden_dim, emb_dim),
            "up_proj": torch.zeros(hidden_dim, emb_dim),
            "down_proj": torch.zeros(emb_dim, hidden_dim),
        }
        for _ in range(E)
    ]
    real = TtFlatRoutedExpert(mesh_device, torch_weights=weights, **kw)
    placeholder = TtFlatRoutedExpert(mesh_device, torch_weights=None, **kw)
    for n in ("w_gu", "w_d", "w_rd"):
        a, b = getattr(real, n), getattr(placeholder, n)
        assert (a is None) == (b is None), n
        if a is not None:
            assert tuple(a.shape) == tuple(b.shape), f"{n}: {tuple(a.shape)} vs {tuple(b.shape)}"
            assert a.dtype == b.dtype and a.memory_config() == b.memory_config(), n

    regions = []
    r = 0
    for c in COUNTS:
        regions.append(r)
        r += _up(c, 32)
    rows = _up(r, 32) + 32

    def u32(values):
        return ttnn.from_torch(
            torch.tensor([values], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_x = ttnn.from_torch(
        torch.zeros(rows, emb_dim),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    y = placeholder(tt_x, u32(COUNTS), u32(regions))
    ttnn.synchronize_device(mesh_device)
    assert tuple(y.shape)[-2:] == (rows, emb_dim)


def _golden_flat_layout(weights, lay, dtype, mesh_device):
    """The flat layout built the slow, obvious way, as an independent check on flat_tile_maps + the tile gather:
    quantize each projection in its plain [K, N] layout, cut the per-core regions out of the dequantized values tile
    by tile, stack them into bank columns and quantize again (exact: the values are already representable)."""
    kblk = 8
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    tiles = lambda w: w.view(w.shape[0] // 32, 32, w.shape[1] // 32, 32).permute(0, 2, 1, 3)
    T_l = [(tiles(q(w["gate_proj"])), tiles(q(w["up_proj"])), tiles(q(w["down_proj"]))) for w in weights]
    it = T_l[0][0].shape[1]
    RG, NP = lay["rg"], lay["np"]
    gu = []
    for r in range(lay["n_rd"]):
        blocks = []
        for Wg, Wu, _ in T_l:
            for c in range(lay["nk_gu"]):
                for pl in range(RG):
                    cols = [((r % lay["n_rd_sg"]) * RG + pl) * NP + p for p in range(NP)]
                    ks = slice(c * kblk, (c + 1) * kblk)
                    blocks.append(torch.stack([w[ks, c_] for c_ in cols for w in (Wg, Wu)], dim=1).reshape(-1, 32, 32))
        gu.append(torch.cat(blocks))

    def kd_of(pcd):
        return max(k for k in (8, 4, 2, 1) if k * pcd <= 16 and it % k == 0)

    dn = []
    for pcd, col0 in zip(lay["pcds"], lay["col0s"]):
        kd = kd_of(pcd)
        dn.append(
            torch.cat(
                [
                    Wd[c * kd : (c + 1) * kd, col0 : col0 + pcd].reshape(-1, 32, 32)
                    for _, _, Wd in T_l
                    for c in range(it // kd)
                ]
            )
        )
    n_max = max(r.shape[0] for r in dn)
    dn = [torch.cat([r, torch.zeros(n_max - r.shape[0], 32, 32)]) for r in dn]
    regions = {"w_gu": gu, "w_d": dn}
    if lay["rdown"]:
        kd_r, n_rdn, pcd_r, rem = lay["kd_r"], lay["n_rdn"], lay["pcd_r"], lay["rem_cols"]
        regions["w_rd"] = [
            torch.cat(
                [
                    Wd[c * kd_r : (c + 1) * kd_r, rem + (i % n_rdn) * pcd_r : rem + (i % n_rdn + 1) * pcd_r].reshape(
                        -1, 32, 32
                    )
                    for _, _, Wd in T_l
                    for c in range(lay["nblk_r"])
                ]
            )
            for i in range(n_rdn * lay["nsg"])
        ]
    banks = lay["banks"]
    out = {}
    for name, regs in regions.items():
        per = -(-len(regs) // banks)
        regs = list(regs) + [torch.zeros_like(regs[0])] * (per * banks - len(regs))
        host = torch.cat(
            [torch.cat([regs[b + h * banks] for h in range(per)]).reshape(-1, 32) for b in range(banks)], dim=1
        )
        out[name] = ttnn.to_torch(ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    return out


@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("model, emb_dim, hidden_dim, activation, torch_activation", SHAPES, ids=[m[0] for m in SHAPES])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b], ids=["bf4", "bf8"])
@pytest.mark.skipif(not is_blackhole(), reason="flat_routed_expert is Blackhole-only")
def test_flat_routed_expert_layout_parity(
    mesh_device, device_params, model, emb_dim, hidden_dim, activation, torch_activation, weights_dtype, tmp_path
):
    """The tile-gather layout (from torch weights, and from the per-expert cache alone) equals the golden layout
    value for value, every tensor, padding included; and building from the cache writes nothing into it."""
    from models.demos.deepseek_v3_d_p.tt.moe.tt_flat_routed_expert import flat_routed_expert_supported

    why = flat_routed_expert_supported(
        mesh_device, activation, weights_dtype, emb_dim, hidden_dim, MAX_TOKENS, False, 3, 3
    )
    if why is not None:
        pytest.skip(f"the op has no layout here: {why}")
    E = 3
    torch.manual_seed(1)
    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim) * 0.02,
        }
        for _ in range(E)
    ]
    kw = dict(
        experts_per_chip=E,
        num_routed_experts=E,
        dispatch_group_size=1,
        num_dispatch_groups=1,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=MAX_TOKENS,
        weights_dtype=weights_dtype,
        activation=activation,
    )
    from_torch = TtFlatRoutedExpert(mesh_device, torch_weights=weights, **kw)
    _write_expert_cache(weights, mesh_device, weights_dtype, tmp_path)
    snapshot = lambda: sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.iterdir())
    before = snapshot()
    from_cache = TtFlatRoutedExpert(
        mesh_device, torch_weights=None, weight_cache_path=tmp_path, cache_name_prefix=_PREFIX, **kw
    )
    assert snapshot() == before, "TtFlatRoutedExpert wrote into the cache directory"

    golden = _golden_flat_layout(weights, from_torch.plan, weights_dtype, mesh_device)
    for name, ref in golden.items():
        for label, module in (("torch", from_torch), ("cache", from_cache)):
            got = ttnn.to_torch(getattr(module, name)).float().reshape(ref.shape)
            assert torch.equal(got, ref), f"{name} ({label}): max |diff| {(got - ref).abs().max().item()}"


@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="flat_routed_expert is Blackhole-only")
def test_flat_routed_expert_supported(mesh_device, device_params):
    """The fallback decision: every shipped shape is accepted, and a shape the op's planner has no layout for (an I
    and dtype with no gate/up split that fits) is refused with a reason rather than failing later in the
    constructor."""
    from models.demos.deepseek_v3_d_p.tt.moe.tt_flat_routed_expert import flat_routed_expert_supported

    def reason(
        emb_dim, hidden_dim, activation=ttnn.RoutedExpertActivation.Silu, epc=8, max_tokens=MAX_TOKENS, dtype=None
    ):
        return flat_routed_expert_supported(
            mesh_device, activation, dtype or ttnn.bfloat4_b, emb_dim, hidden_dim, max_tokens, False, epc, 8 * epc
        )

    for name, emb_dim, hidden_dim, activation, _ in SHAPES:
        assert reason(emb_dim, hidden_dim, activation) is None, name
    # bfp8 weights at I 1024 leave the planner no gate/up split that fits L1 (bfp4 has one).
    assert reason(KimiK27Config.EMB_SIZE, 1024, dtype=ttnn.bfloat8_b) is not None
    assert reason(KimiK27Config.EMB_SIZE, 2040) is not None  # not whole tiles
    assert reason(KimiK27Config.EMB_SIZE, 2048, max_tokens=128) is not None
    assert reason(KimiK27Config.EMB_SIZE, 2048, epc=65) is not None
