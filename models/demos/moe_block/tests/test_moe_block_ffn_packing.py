# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The model adapter (TtK3MoEBlockFfn) keeps a layer's prepared weights by their packing - host only, no device.

The chunk picks the block's plans, and the head's weights are packed for its plan's layout (GX x GY matmul cores, MT,
the K step KB): a layer whose calls change chunk must get weights packed for each block's plan, not the first block's.
The ops are replaced by fakes that record the layout each bundle was packed for, so one adapter run on two blocks is
compared with a fresh adapter for each. test_moe_block_models.test_moe_block_chunk_switch is the same check on a Galaxy,
on the block's outputs."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt.kimi_k3 import moe_block_ffn as mbf

H, IS, E, LAT = 64, 32, 32, 48  # (small dims: the fakes only read them)


def _lay(GY=10, MT=2, KB=8, NCH=1, NIT=1, GX=11):
    return SimpleNamespace(
        engine_columns=GX,
        engine_rows=GY,
        row_tiles_per_core=MT,
        k_step_tiles=KB,
        row_chunks=NCH,
        router_iterations=NIT,
        compact_reduce_scatter_zones=False,
    )


FLAT = dict(
    banks=8,
    n_rd=8,
    n_rd_sg=8,
    rg=2,
    np=1,
    nu=0,
    nk_gu=4,
    nd=10,
    pcds=[2] * 10,
    col0s=list(range(0, 20, 2)),
    wd_rot_tiles=96,
    mt=4,
)


class _FakeOps:
    """The adapter's moe_block calls: the head's plan is the block's (resources.head_lay); each prepared part records
    the layout it was packed for."""

    def __init__(self):
        self.heads, self.tails, self.routed, self.plans = [], 0, [], 0

    def plan_moe_head(self, resources, dtypes, accumulators, l1_free=None):
        self.plans += 1
        return SimpleNamespace(
            ok=True, planned=[SimpleNamespace(layout=resources.head_lay, predicted_us=1.0)], reason="", note=""
        )

    def prepare_moe_head_weights(self, gate, up, down, latent_down, rw, rb, *, resources, dtypes, accumulators):
        lay = resources.head_lay
        packed = (
            "head",
            lay.engine_columns,
            lay.engine_rows,
            lay.row_tiles_per_core,
            lay.k_step_tiles,
            round(float(gate.sum()), 3),
        )
        self.heads.append(packed)
        return packed

    def prepare_moe_tail_weights(self, latent_up, latent_norm, *, resources):
        self.tails += 1
        return ("w_up", round(float(latent_up.sum()), 3))

    def flat_routed_expert_plan(self, mesh, Hr, I, epc, E, m, *, weights_bf8, pin, layout):
        return SimpleNamespace(as_dict=lambda: dict(layout))


@pytest.fixture
def ops(monkeypatch):
    fake = _FakeOps()
    monkeypatch.setattr(mbf.ttnn.experimental, "moe_block", fake, raising=False)
    monkeypatch.setattr(mbf.ttnn, "from_torch", lambda t, dtype=None: t)

    def routed(self, block, sources=None):  # (the flat_gather layout: the plan's packing fields)
        lay = self._routed_layout(block)
        packed = ("routed",) + tuple(str(lay[k]) for k in self._ROUTED_PACKING)
        fake.routed.append(packed)
        return packed, packed, packed

    monkeypatch.setattr(mbf.TtK3MoEBlockFfn, "_routed", routed)
    return fake


def _cfg(latent: bool):
    dims = SimpleNamespace(
        num_experts=E,
        emb_dim=H,
        latent_dim=LAT if latent else None,
        routed_width=LAT if latent else H,
        routed_expert_hidden_dim=IS,
    )
    return SimpleNamespace(__name__="fake", moe_block_dims=lambda: (dims, "silu"))


def _sd(latent: bool):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    sd = {
        "gate_weights": {"weight": r(E, H), "e_score_correction_bias": r(E)},
        "shared_expert_weights": {"gate_proj": r(IS, H), "up_proj": r(IS, H), "down_proj": r(H, IS)},
        "routed_expert_weights": [None] * E,  # (not read: _routed is the fake's)
    }
    if latent:
        sd["latent_weights"] = {"down_proj": r(LAT, H), "up_proj": r(H, LAT), "norm": r(LAT)}
    return sd


def _block(head_lay, flat=None, accumulators=16):
    return SimpleNamespace(
        resources=SimpleNamespace(head_lay=head_lay),
        dtypes=SimpleNamespace(shared_weights="WeightDtype.Bf8"),
        accumulators=accumulators,
        expert_kw={"max_tokens_per_expert": 5120, "layout": dict(flat or FLAT)},
    )


def _ffn(latent: bool):
    mesh = SimpleNamespace(shape=(8, 4))
    return mbf.TtK3MoEBlockFfn(mesh, _cfg(latent), layer_idx=1, state_dict=_sd(latent))


def _same(a, b):
    return (a.head, a.gate_up, a.down, a.w_up) == (b.head, b.gate_up, b.down, b.w_up)


@pytest.mark.parametrize("latent", [False, True], ids=["glm", "k3-latent"])
def test_two_head_plans_one_adapter_vs_fresh(ops, latent):
    """Two blocks whose head plans pack differently (GY 10 MT 2 KB 8 / GY 5 MT 4 KB 12): one adapter called on A, B,
    A again gives each block the weights a fresh adapter makes for it."""
    a, b = _block(_lay(GY=10, MT=2, KB=8)), _block(_lay(GY=5, MT=4, KB=12))
    ffn = _ffn(latent)
    wa, wb, wa2 = ffn._weights(a), ffn._weights(b), ffn._weights(a)
    fa, fb = _ffn(latent)._weights(a), _ffn(latent)._weights(b)
    assert wb.head[2:5] == (5, 4, 12), f"block B got head weights packed for {wb.head[1:5]}"
    assert _same(wa, fa) and _same(wb, fb), "the shared adapter's weights differ from fresh adapters'"
    assert wa2 is wa, "block A's weights were not reused"
    assert ops.plans == 4, f"{ops.plans} head plans looked up: a block's later calls re-planned"  # (A, B; fa, fb)
    assert wb.gate_up is wa.gate_up, "the routed experts (same flat plan) were laid out again"
    if latent:
        assert wb.w_up is wa.w_up, "the tail's W' (chunk-independent) was prepared again"


def test_same_packing_reused(ops):
    """Blocks whose plans differ only where the packing does not (row chunks, router iterations, accumulators) share
    one prepared head."""
    ffn = _ffn(False)
    w1 = ffn._weights(_block(_lay(NCH=1, NIT=1), accumulators=16))
    w2 = ffn._weights(_block(_lay(NCH=5, NIT=5), accumulators=12))
    assert w2.head is w1.head and w2.gate_up is w1.gate_up
    assert len(ops.heads) == 1 and len(ops.routed) == 1


def test_routed_layout_change(ops):
    """A block whose flat plan lays the experts out differently gets its own routed weights; the head (same plan) is
    kept."""
    ffn = _ffn(True)
    w1 = ffn._weights(_block(_lay()))
    w2 = ffn._weights(_block(_lay(), flat={**FLAT, "rg": 4, "n_rd_sg": 4}))
    w3 = ffn._weights(_block(_lay(), flat={**FLAT, "mt": 2}))  # (mt: not a packing field)
    assert w2.gate_up is not w1.gate_up and w2.head is w1.head and w2.w_up is w1.w_up
    assert w3 is w1
    assert len(ops.routed) == 2 and len(ops.heads) == 1 and ops.tails == 1
