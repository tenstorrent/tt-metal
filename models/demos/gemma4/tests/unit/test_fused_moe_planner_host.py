# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the custom MoE programs are selected, never asserted into, on configurations they cannot take.
``RouterTopk.supports`` (a tile multiple of experts up to 256, k <= 32, packed rows need k a multiple of 8) gates the
fused decode router through ``MoEBlock.fused_decode_ok`` / ``fused_packed_routing_ok`` and the layer's fused norm
chain; ``ExpertGatherSmall.supports`` gates the small-row prefill gather; the ring combine passes main's
``fwd_reduction`` only when the shared constructor has it. CI reduced-expert configs (max(4, chips)) fall back to
the shared chain. No device, no conftest."""

from types import SimpleNamespace

from models.demos.gemma4.custom_ops.expert_gather_small import ExpertGatherSmall
from models.demos.gemma4.custom_ops.router_topk import RouterTopk
from models.demos.gemma4.tt.experts.dispatch_prefill import combine2d_extra_kwargs, combine_fwd_reduction
from models.demos.gemma4.tt.moe import MoEBlock


def test_router_program_contract_over_expert_counts():
    assert [RouterTopk.supports(e, 8) for e in (4, 32, 128, 256, 288)] == [False, True, True, True, False]
    assert RouterTopk.supports(128, 32) and not RouterTopk.supports(128, 33) and not RouterTopk.supports(128, 0)
    # packed rows of 2k bf16 must be 32-byte aligned: k a multiple of 8
    assert RouterTopk.supports(128, 8, packed=True) and not RouterTopk.supports(128, 2, packed=True)
    assert not RouterTopk.supports(4, 8, packed=True)


def test_small_gather_contract():
    assert ExpertGatherSmall.supports(128, 32, 8, 2688)
    assert not ExpertGatherSmall.supports(4, 1, 2, 2688)  # the CI reduced-expert count
    assert not ExpertGatherSmall.supports(128, 32, 8, 2690)  # hidden not a tile multiple
    assert not ExpertGatherSmall.supports(128, 32, 8, 8192)  # a row must fit the L1 staging (2H <= 8192)
    assert not ExpertGatherSmall.supports(128, 256, 8, 2688)  # more local experts than experts


def _moe(num_experts, top_k, fused=True, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("GEMMA4_MOE_FUSED", "1" if fused else "0")
    return SimpleNamespace(
        num_experts=num_experts,
        top_k=top_k,
        mesh_config=None,
        experts=SimpleNamespace(ep=1, moe_decode=None),
        router=SimpleNamespace(per_expert_scale=object()),
    )


def test_fused_decode_falls_back_below_the_tile_multiple(monkeypatch):
    ok = [MoEBlock.fused_decode_ok(_moe(e, 8, monkeypatch=monkeypatch)) for e in (4, 32, 128, 256, 288)]
    assert ok == [False, True, True, True, False], ok
    assert MoEBlock.fused_packed_routing_ok(_moe(128, 8, monkeypatch=monkeypatch))
    assert not MoEBlock.fused_packed_routing_ok(_moe(128, 2, monkeypatch=monkeypatch))
    assert not MoEBlock.fused_packed_routing_ok(_moe(4, 8, monkeypatch=monkeypatch))
    # the flag off, lanes, EP > 1, the decode kernel or no per-expert scale keep the shared chain as before
    assert not MoEBlock.fused_decode_ok(_moe(128, 8, fused=False, monkeypatch=monkeypatch))
    m = _moe(128, 8, monkeypatch=monkeypatch)
    m.experts.ep = 2
    assert not MoEBlock.fused_decode_ok(m)
    m = _moe(128, 8, monkeypatch=monkeypatch)
    m.router.per_expert_scale = None
    assert not MoEBlock.fused_decode_ok(m) and not MoEBlock.fused_packed_routing_ok(m)


def test_ring_combine_passes_fwd_reduction_only_when_the_signature_has_it(monkeypatch):
    class WithoutIt:
        def __init__(self, mesh_device, experts_per_chip, num_experts_per_tok, seq_len_per_chip, cluster_axis=0):
            pass

    class WithIt:
        def __init__(
            self,
            mesh_device,
            experts_per_chip,
            num_experts_per_tok,
            seq_len_per_chip,
            fwd_reduction: int,
            cluster_axis=0,
        ):
            pass

    monkeypatch.delenv("GEMMA4_MOE_COMBINE_FWD_REDUCTION", raising=False)
    assert combine2d_extra_kwargs(WithoutIt) == {}
    assert combine2d_extra_kwargs(WithIt) == {"fwd_reduction": 1}
    monkeypatch.setenv("GEMMA4_MOE_COMBINE_FWD_REDUCTION", "4")
    assert combine_fwd_reduction() == 4 and combine2d_extra_kwargs(WithIt) == {"fwd_reduction": 4}
    monkeypatch.setenv("GEMMA4_MOE_COMBINE_FWD_REDUCTION", "0")
    assert combine_fwd_reduction() == 1  # the op requires >= 1
