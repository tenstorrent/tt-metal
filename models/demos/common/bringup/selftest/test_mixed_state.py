# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F50: mixed per-layer state. Layers of different block types carry different state tensors (spec
state.by_block_type), and fixed-size tensors (state.fixed: a recurrent state, a conv tail) are snapshotted at chunk
starts, because the state at a chunk start cannot be sliced out of the final one (GLM-5.3: KDA + sparse MLA).
CPU only, on the fixture with layer 1 as a linear recurrence."""

import pytest
import torch

from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.plan.memory import GB, check_plan
from models.demos.common.bringup.reference import check_hf, check_reference, generate_golden, prompt
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.selftest.test_plan import TENSORS, plan
from models.demos.common.bringup.testing.component import run_component_test, run_swap_test
from models.demos.common.bringup.testing.ladder import run_ladder

MIXED = {
    "block_types": {"attn": {"layers": [0, 2]}, "rec": {"layers": [1]}},
    "state": {
        "kind": "mixed",
        "tensors": ["key", "value", "recurrent"],
        "by_block_type": {"attn": ["key", "value"], "rec": ["recurrent"]},
        "fixed": ["recurrent"],
    },
    "fixture": {"recurrent_layers": [1]},
}


@pytest.fixture
def mspec(fx, monkeypatch):
    def make(**over):
        p = fx(**{**MIXED, **over})
        monkeypatch.setenv("BRINGUP_SPEC", p)
        prompt.build(Spec.load(p))
        generate_golden.main(["--spec", p, "--rung", "s256"])
        generate_golden.main(["--spec", p, "--rung", "s512"])
        return Spec.load(p)

    return make


def reference_state_at(s, layer, at):
    ref = fixture_model.Reference(s)
    st = ref.new_state(s.rung("s256")["seq"])
    ref.forward_chunk(Golden.for_rung(s, "s256").tokens()[:at], 0, st)
    return ref.state_tensors(st, layer, at)


def test_spec_names_per_block_type(fx):
    s = Spec.load(fx(**MIXED))
    assert s.validate() == []
    assert s.state_names(0) == ["key", "value"] and s.state_names(1) == ["recurrent"] and s.state_fixed == ["recurrent"]
    assert Spec.load(fx()).state_names(1) == ["key", "value"] and Spec.load(fx()).state_fixed == []


@pytest.mark.parametrize(
    "state,needle",
    [
        ({"by_block_type": {"attn": ["key"], "nope": ["key"]}}, "nope is not a block type"),
        ({"by_block_type": {"attn": ["key"]}}, "misses block types ['rec']"),
        ({"by_block_type": {"attn": ["kv"], "rec": ["recurrent"]}}, "['kv'] not in state.tensors"),
        ({"fixed": ["conv"]}, "state.fixed: ['conv']"),
    ],
)
def test_spec_validation(fx, state, needle):
    errs = Spec.load(fx(**{**MIXED, "state": {**MIXED["state"], **state}})).validate()
    assert any(needle in e for e in errs), errs


def test_hf_parity_with_a_recurrent_layer(fx):
    check_hf.main(["--spec", fx(**MIXED), "--seq", "128"])
    assert min(v for k, v in got().items() if k.startswith("pcc_")) > 0.99999


def test_graph_replay_starts_from_the_recurrent_state_before_the_last_chunk(fx):
    """Before F50 the replay's prefix was sliced from the final state: exact for a KV cache, wrong for a recurrence."""
    check_reference.main(["--spec", fx(**MIXED), "--seq", "256", "--chunk", "64"])
    m = got()
    assert m["pcc_hidden"] > 0.999999 and m["pcc_state_min"] > 0.999999
    assert m["graph_errors"] == 0 and m["boundaries_missing"] == 0 and m["graph_replay_maxabs"] == 0.0


def test_golden_stores_per_layer_names_and_snapshots(mspec):
    s = mspec()
    g = Golden.for_rung(s, "s256")
    assert g.verify() and set(g.state(0)) == {"key", "value"} and set(g.state(1)) == {"recurrent"}
    # full dumps: a snapshot before every chunk; s256 is the contract rung: one more after seq - 32 tokens
    assert g.meta["state_snapshots"] == [0, 64, 128, 192, 224]
    for at in (64, 128, 224):
        want = reference_state_at(s, 1, at)["recurrent"]
        assert torch.equal(g.state(1, at=at)["recurrent"], want.to(g.state(1, at=at)["recurrent"].dtype))
    assert not torch.equal(g.state(1, at=128)["recurrent"], g.state(1)["recurrent"])
    assert torch.equal(g.state(0, at=128)["key"], g.state(0)["key"])  # growing tensors: the consumer slices
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        g.state(1, at=100)
    g512 = Golden.for_rung(s, "s512")
    assert g512.meta["state_snapshots"] == [384]  # last chunk only (no full dumps, not the contract rung)


def test_golden_without_fixed_state_is_unchanged(fx, monkeypatch):
    p = fx()
    prompt.build(Spec.load(p))
    generate_golden.main(["--spec", p, "--rung", "s256"])
    g = Golden.for_rung(Spec.load(p), "s256")
    assert g.meta["state_snapshots"] == [] and not list((g.dir / "kv_cache").glob("*_at_*"))


@pytest.mark.parametrize("impl,noise_level,passes", [("device", 1e-3, True), ("stub", 0.0, False)])
def test_component_of_the_recurrent_step_gets_the_state_at_the_chunk_start(
    mspec, monkeypatch, impl, noise_level, passes
):
    from models.demos.common.bringup.core.runs import IMPL_ENV

    s = mspec()
    monkeypatch.setitem(fixture_model.NOISE, "value", noise_level)
    monkeypatch.setenv(IMPL_ENV, impl)
    assert run_component_test(s, "attention", layer=1) is passes
    if passes:
        assert got()["pcc_attention_L01"] > 0.999


def test_swap_on_the_recurrent_block(mspec, monkeypatch):
    s = mspec()
    monkeypatch.setitem(fixture_model.NOISE, "value", 0.0)
    assert run_swap_test(s, "rec", ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp"])
    assert got()["pcc_swap_out"] > 0.99999


@pytest.mark.parametrize("rung", ["s256", "last"])
def test_ladder_with_mixed_state(mspec, monkeypatch, rung):
    s = mspec()
    monkeypatch.setitem(fixture_model.NOISE, "value", 0.0)
    out = run_ladder(s, rung, None)
    m = got()
    assert not out["failed"], out
    assert m["pcc_layer_L01"] > 0.9999 and m["pcc_state_recurrent_L01"] > 0.9999 and "pcc_state_key_L01" not in m
    assert m["pcc_state_key_L00"] > 0.9999


def test_memory_of_a_fixed_size_state(fx):
    s = Spec.load(fx(box={"mesh": [1, 4], "chip_dram_gb": 0.01}))
    st = [
        {
            "layers": "0,2",
            "what": "latent",
            "heads_per_chip": 1,
            "head_dim": 64,
            "tensors": 1,
            "dtype": "bf16",
            "seq_divisor": 4,
            "kv_heads": 1,
            "replicated": True,
        },
        {
            "layers": "1",
            "what": "recurrent",
            "heads_per_chip": 2,
            "head_dim": 64 * 64,
            "tensors": 1,
            "dtype": "fp32",
            "per_token": False,
            "kv_heads": 8,
        },
    ]
    r = check_plan(s, plan(state=st), TENSORS, {"num_key_value_heads": 4})
    by = {x["group"]: x["gb"] * GB for x in r["rows"]}
    assert r["errors"] == [], r["errors"]
    assert by["state: latent layers 0,2"] == 2 * 64 * 2 * (256 // 4)
    assert by["state: recurrent layers 1"] == 2 * 64 * 64 * 4
    st[1]["heads_per_chip"] = 1  # 1 x 4 chips < its own 8 heads
    assert any("KV heads" in e for e in check_plan(s, plan(state=st), TENSORS, {"num_key_value_heads": 4})["errors"])
