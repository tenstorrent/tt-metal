# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The MoE block's stages on a Galaxy for the op-level checks (test_experts_combine_checks, test_sparse_dispatch_2d_checks,
test_moe_tail_checks): a case's block (combine snap on, q 2^-14), seeded random weights, and the real upstream ops run
once to make the op under test's inputs - moe_head -> sparse_dispatch_2d -> routed_experts_combine -> moe_tail, as
MoEBlock's forward runs them (each in a call's own L1 arena, freed after).

A case is a model (the block tests' MODELS: GLM-5.3, Kimi-K3, DeepSeek V4.1-Flash) on a mesh at a chunk, with the
expected device times of the experts + combine. A routing is how the tokens spread over the experts:
  * uniform: random router weights, a near-zero bias;
  * zipf: a router bias falling as rank^-1.2 over a random order of the experts (a few experts take most tokens);
  * hotchip: a flatter bias (rank^-0.8) with the top ranks all on one chip's experts: on DeepSeek V4.1-Flash 8x4 5120
    the hottest chip gets ~3.7x the mean chip's rows (zipf: ~2.4x), on the sigmoid-routed models ~7x (zipf: ~3.5x).

Weights: seeded random, one chip's experts repeated on every chip (all distinct experts would take the host ~40-120 GB:
a row sent to the right local expert on the wrong chip is not caught by the combine's check; it is in the block's
accuracy test). A case's weights are built on the host once a module (the host_weights fixture - import it into a test
module: the experts packed and converted to bfp4, the reference's bfp4-grid copies, the head's and a latent tail's fp32
inputs, x); every test reopens the mesh and only uploads them. 4x4 cases run on half a Galaxy, set up as the block
tests' 4x4 runs; a case whose block needs splits is skipped (one head call a chunk here)."""

from dataclasses import dataclass
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.tt.kimi_k3.moe_block_ffn import block_dims
from models.demos.deepseek_v3_d_p.tt.moe_impl import moe_block_words_slab
from block_cases import MESHES, MODELS
from models.demos.moe_block.tt.flat_gather import flat_weights_module
from models.demos.moe_block.tt.moe_block import MoEBlock

mb = ttnn.experimental.moe_block
fw = flat_weights_module()

SNAP = 2.0**-14
CALLS = 10  # (determinism calls; perf: op_checks.PERF_RUNS)
SAMPLE = 64  # tokens a chip checked against torch
PCC = 0.99
ROUTINGS = ("uniform", "zipf", "hotchip")


# ---------------------------------------------------------------------------------------------------- test configs
@dataclass(frozen=True)
class Case:
    model: str  # test_moe_block_models.MODELS key
    mesh: str = "8x4"  # test_moe_block_models.MESHES key
    chunk: int = 5120
    perf_margin: float = 0.05  # (the baselines: experts_combine_perf_baselines.json)
    seed: int = 5

    @property
    def id(self):
        return f"{self.model}-{self.mesh}-{self.chunk}"


CASES = [
    Case("glm53", "8x4", 5120),
    Case("kimi_k3", "8x4", 5120),
    Case("dsv41_flash", "8x4", 5120),
    Case("glm53", "4x4-y", 5120),
    Case("kimi_k3", "4x4-y", 2560),
    # MiniMax-M3 / M3.1's production stage (one [2, 4] tray, Fabric2D, no wrap: test_moe_block_models.M3_PRODUCTION):
    # one 5120-token chunk a forward, four 1024-token chunks packed in a 4096-token forward, 2048-token chunks
    Case("minimax_m3", "2x4", 5120),
    Case("minimax_m3", "2x4", 4096),
    Case("minimax_m3", "2x4", 2048),
]


def _params(routings):
    out = []
    for c in CASES:
        shape, make_params, topology = MESHES[c.mesh]
        dp = make_params(**MODELS[c.model].mesh_params())
        mark = pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology)
        for r in routings:
            out.append(pytest.param(c, r, shape, dp, marks=mark, id=f"{c.id}-{r}" if r else c.id))
    return out


ARGS = "case, routing, mesh_device, device_params"
INDIRECT = ["mesh_device", "device_params"]
PARAMS = pytest.mark.parametrize(ARGS, _params(ROUTINGS), indirect=INDIRECT)
CASE_PARAMS = pytest.mark.parametrize(ARGS, _params((None,)), indirect=INDIRECT)


def _bias(E, routing, seed, chips):
    """The router's bias for a routing (the pick is on score + bias; the gate weights are the scores alone). Chip c
    holds experts [c E/chips, (c + 1) E/chips)."""
    g = torch.Generator().manual_seed(seed)
    if routing == "uniform":
        return torch.randn(E, generator=g) * 0.01
    rank = torch.empty(E)
    if routing == "zipf":
        rank[torch.randperm(E, generator=g)] = torch.arange(1, E + 1, dtype=torch.float32)
        return 0.5 * rank**-1.2
    assert routing == "hotchip", routing
    epc = E // chips
    hot = int(torch.randint(chips, (1,), generator=g))
    order = torch.randperm(E, generator=g)
    order = torch.cat([hot * epc + torch.randperm(epc, generator=g), order[(order // epc) != hot]])
    rank[order] = torch.arange(1, E + 1, dtype=torch.float32)
    return 0.5 * rank**-0.8


def _q4(w):
    """w on the bfp4 grid (the flat expert's weights as the op holds them), as bf16 (exact: 3 mantissa bits)."""
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).to(torch.bfloat16)


@dataclass
class HostWeights:
    """A case's weights on the host. The seeded torch experts are kept only until they are packed (the first test's
    FlatRoutedExpert) and put on the bfp4 grid (the first reference)."""

    local: list  # per local expert (Wg [Hr, I], Wu [Hr, I], Wd [I, Hr]) fp32, until packed and wq are made
    head: tuple  # the head's fp32 host tensors: shared gate, up, down; latent down (None: no latent); router
    x: torch.Tensor  # [1, 1, chunk, H] bf16
    tail: tuple = None  # a latent model's tail: fp32 host latent_up [H, Hr], latent_norm [Hr] (None: no latent)
    packed: object = None  # fw.PackedFlatWeights
    wq: list = None  # per local expert (g, u, d) on the bfp4 grid, bf16

    @staticmethod
    def build(case, d, EPC):
        """The seeded draws in the order the test always made them."""
        g = torch.Generator().manual_seed(case.seed)
        r = lambda *s, sc: torch.randn(*s, generator=g) * sc  # noqa: E731
        Hr, I, H, Is = d.routed_width, d.routed_expert_hidden_dim, d.emb_dim, d.shared_expert_hidden_dim
        local = [(r(Hr, I, sc=Hr**-0.5), r(Hr, I, sc=Hr**-0.5), r(I, Hr, sc=I**-0.5)) for _ in range(EPC)]
        f32 = lambda t: None if t is None else ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731
        shared = (r(Is, H, sc=H**-0.5), r(Is, H, sc=H**-0.5), r(H, Is, sc=Is**-0.5))
        router = r(d.num_experts, H, sc=H**-0.5)
        latent_down = r(Hr, H, sc=H**-0.5) if d.latent_dim else None
        head = (*(f32(t) for t in shared), f32(latent_down), f32(router))
        x = torch.randn(1, 1, case.chunk, H, generator=g).to(torch.bfloat16)
        tail = (f32(r(H, Hr, sc=Hr**-0.5)), f32(1.0 + r(Hr, sc=0.1))) if d.latent_dim else None
        return HostWeights(local, head, x, tail)

    def drop_local(self):
        if self.packed is not None and self.wq is not None:
            self.local = None


@pytest.fixture(scope="module")
def host_weights():
    """case key -> HostWeights, built on a case's first test."""
    return {}


# ---------------------------------------------------------------------------------------------------- the op
class BlockStages:
    """The block's resources and plan for a case (combine snap on), the routed experts, a head weight set a routing,
    and x. route(r): the MoE head + dispatch for routing r, their outputs kept (head, xg); dispatch(): the dispatch
    alone again on the kept head; __call__(): routed_experts_combine on them (y_own); tail(y_own): moe_tail on y_own
    and the kept head's shared expert output."""

    def __init__(self, mesh, case, cache, layout=None):
        """layout: the flat expert's FlatLayoutOptions (e.g. x_bf16 / h_bf16), None: the block's default."""
        self.mesh, self.case = mesh, case
        model = MODELS[case.model]
        self.dims, self.act = block_dims(model.cfg)
        d = self.dims
        R, C = tuple(mesh.shape)
        self.N = R * C
        self.E, self.K, self.H, self.Hr = d.num_experts, d.top_k, d.emb_dim, d.routed_width
        self.I, self.EPC, self.S = d.routed_expert_hidden_dim, d.num_experts // self.N, case.chunk // self.N
        cfg = mb.MoEBlockConfig(chunk_tokens=case.chunk, num_links=2, combine_snap=SNAP, persist_l1_small=True)
        cfg.words_slab_bytes = moe_block_words_slab(model.cfg)
        self.dtypes = mb.MoEDtypes()
        self.res = mb.MoEBlockResources(mesh, d, cfg, self.dtypes)
        self.block = MoEBlock(mesh, self.res, dtypes=self.dtypes, layout=layout)
        kw = self.block.expert_kw

        # (the packed weights follow the flat plan, which the layout's x / h formats can change: one cache entry each)
        key = f"{case.id}-seed{case.seed}" + (f"-{layout!r}" if layout is not None else "")
        if key not in cache:
            cache[key] = HostWeights.build(case, d, self.EPC)
        self.hw = hw = cache[key]
        self.experts = fw.FlatRoutedExpert(
            mesh, hw.packed or [hw.local] * self.N, m=kw["max_tokens_per_expert"], H=self.Hr, I=self.I,
            gids=[list(range(c * self.EPC, (c + 1) * self.EPC)) for c in range(self.N)], n_global=self.E,
            act=self.act, pin=1, layout=kw["layout"], keep_packed=True,
        )  # fmt: skip
        hw.packed = self.experts.packed
        hw.drop_local()

        self.head_w = {}  # routing -> the head's weights (shared expert + router, not checked here; a bias a routing)
        self.x = ttnn.from_torch(
            hw.x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
        )
        self.head = self.xg = None
        self.w_up = None
        if self.block.splits != 1:
            pytest.skip(f"{case.id}: {self.block.splits} splits (these tests run one head call a chunk)")

    def route(self, routing):
        """The MoE head + dispatch for a routing (MoEBlock's sequence up to the op): the op's inputs, kept. The L1
        arena lives only while the ops run, as in MoEBlock's forward (any other op's static CBs clash with it)."""
        blk = self.block
        if routing not in self.head_w:
            bias = _bias(self.E, routing, self.case.seed + ROUTINGS.index(routing), self.N)
            self.head_w[routing] = mb.prepare_moe_head_weights(
                *self.hw.head, ttnn.from_torch(bias, dtype=ttnn.float32), resources=self.res, dtypes=self.dtypes,
                accumulators=blk.accumulators,
            )  # fmt: skip
        arena = ttnn.allocate_tensor_on_device(blk.arena_spec, self.mesh)
        self.head = mb.moe_head(
            self.x, self.head_w[routing], resources=self.res, dtypes=self.dtypes, accumulators=blk.accumulators,
            arena=arena,
        )  # fmt: skip
        self.xg = mb.sparse_dispatch_2d(self.head.dispatch_buffer, self.head.route_words, self.res, arena=arena)
        ttnn.deallocate(arena)
        ttnn.synchronize_device(self.mesh)
        self.routing = routing

    def _arena(self):
        return ttnn.allocate_tensor_on_device(self.block.arena_spec, self.mesh)

    def dispatch(self):
        """sparse_dispatch_2d again on the kept head's xbuf + route (in a call's own arena): xbuf filled in place."""
        arena = self._arena()
        xg = mb.sparse_dispatch_2d(self.head.dispatch_buffer, self.head.route_words, self.res, arena=arena)
        ttnn.deallocate(arena)
        return xg

    def tail(self, y_own):
        """moe_tail on y_own and the kept head's shared expert output (no arena: MoEBlock runs it after the arena is
        freed); a latent model's W_up prepared on the first call. Returns [1, 1, S/SP, H/TP] (a new tensor a call)."""
        if self.w_up is None and self.hw.tail is not None:
            self.w_up = mb.prepare_moe_tail_weights(*self.hw.tail, resources=self.res, row_blocks=self.head.row_blocks)
        return mb.moe_tail(
            y_own, self.head.shared_expert_output, resources=self.res, w_up=self.w_up, row_blocks=self.head.row_blocks
        )

    def __call__(self):
        """The op in a call's own arena (allocated, run, freed - MoEBlock's forward), y_own returned."""
        h, ex = self.head, self.experts
        arena = self._arena()
        y = mb.routed_experts_combine(
            self.xg, h.counts, h.regions, h.token_index, h.table, h.plan, ex.gidx, ex.w_gu, ex.w_d, self.res,
            arena=arena, **self.block.expert_kw,
        )  # fmt: skip
        ttnn.deallocate(arena)
        return y

    # ---------------------------------------------------------------------------------------------- host side
    def routing_on_host(self):
        """(ids [T, K], gate weights [T, K], the dispatched own rows [T, Hr]) in token order (chip me's S tokens at
        me S ..), as the head gave them."""
        K, S = self.K, self.S
        rows = torch.cat(
            [
                ttnn.to_torch(t).reshape(-1, 2 * K).to(torch.int64)[:S]
                for t in ttnn.get_device_tensors(self.head.routing_rows)
            ]
        )
        ids = rows[:, :K] & 0xFFFF
        ws = ((rows[:, K:] & 0xFFFF) << 16).to(torch.int32).view(torch.float32)
        xin = []
        for me, t in enumerate(ttnn.get_device_tensors(self.xg)):
            xin.append(ttnn.to_torch(t).reshape(-1, self.Hr)[me * S : (me + 1) * S].float())
        return ids, ws, torch.cat(xin)

    def log_load(self, ids):
        per_expert = torch.bincount(ids.reshape(-1), minlength=self.E).float()
        per_chip = per_expert.reshape(self.N, self.EPC).sum(-1)
        top = per_expert.sort(descending=True).values
        logger.info(
            f"{self.case.id} {self.routing}: rows an expert max {int(top[0])} / mean {per_expert.mean():.0f} "
            f"(top 8 take {top[:8].sum() / top.sum():.0%}), rows a chip max {int(per_chip.max())} / mean "
            f"{per_chip.mean():.0f}"
        )

    def reference(self, ids, ws, xin):
        """[T', Hr] fp32: each token's K experts on its dispatched row, bfp4 weights, x its gate weight, summed."""
        hw = self.hw
        if hw.wq is None:
            hw.wq = [tuple(_q4(w) for w in e) for e in hw.local]
            hw.drop_local()
        out = torch.zeros(xin.shape[0], self.Hr)
        le = ids % self.EPC  # (every chip's local expert le is the same expert)
        for e in range(self.EPC):
            t, k = (le == e).nonzero(as_tuple=True)
            if len(t):
                g, u, dn = (w.float() for w in hw.wq[e])
                y = fw.act_ref(xin[t] @ g, xin[t] @ u, self.act) @ dn
                out.index_add_(0, t, y * ws[t, k].unsqueeze(1))
        return out

    def y_on_host(self, y):
        """y_own -> [T, Hr] in token order."""
        return torch.cat([ttnn.to_torch(t).reshape(-1, self.Hr)[: self.S] for t in ttnn.get_device_tensors(y)]).float()

    def check(self, y, label):
        """y_own against torch on SAMPLE tokens a chip (all finite); returns y_own on the host."""
        ids, ws, xin = self.routing_on_host()
        self.log_load(ids)
        got = self.y_on_host(y)
        assert torch.isfinite(got).all(), f"{label}: non-finite outputs"
        g = torch.Generator().manual_seed(77)
        pick = torch.cat([me * self.S + torch.randperm(self.S, generator=g)[:SAMPLE] for me in range(self.N)])
        want = self.reference(ids[pick], ws[pick], xin[pick])
        ok, pcc = comp_pcc(want, got[pick], PCC)
        n = min(SAMPLE, self.S)
        worst = min(
            (float(comp_pcc(want[i * n : (i + 1) * n], got[pick][i * n : (i + 1) * n], PCC)[1]), i)
            for i in range(self.N)
        )
        logger.info(f"ACCURACY {label}: pcc {pcc} over {len(pick)} tokens, worst chip {worst[1]} pcc {worst[0]:.5f}")
        assert ok, f"{label}: pcc {pcc} < {PCC}"
        assert worst[0] >= PCC, f"{label}: chip {worst[1]} pcc {worst[0]:.5f} < {PCC}"
        return got

    def close(self):
        self.res.release()


def skip_unless_galaxy():
    if not is_blackhole():
        pytest.skip("Blackhole only")
