# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Hy4 weight loader skeleton against the real checkpoint (index + shard headers; one expert slab is read)."""

import json
import os

import pytest

from models.demos.blackhole.hy4_preview import ckpt_stamp
from models.demos.blackhole.hy4_preview.config import DEFAULT_CKPT
from models.demos.blackhole.hy4_preview.weights import (
    GATE_UP_ORDERS,
    Hy4Weights,
    all_logical_names,
    checkpoint_key,
    layer_names,
    logical_name,
    read_header,
)

CKPT = os.environ.get("HY4_CKPT", DEFAULT_CKPT)
needs_ckpt = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "model.safetensors.index.json")), reason="checkpoint not mounted"
)


@pytest.fixture(scope="module")
def w():
    return Hy4Weights(CKPT)


def test_name_map_examples():
    assert logical_name("model.layers.3.self_attn.q_a_proj.weight") == "layers.3.attn.q_a_proj"
    assert logical_name("model.layers.1.mlp.experts.gate_up_proj") == "layers.1.moe.experts.gate_up"
    assert logical_name("model.mtp_layers.0.eh_proj.weight") == "mtp.0.eh_proj"
    assert logical_name("model.mtp_layers.0.self_attn.indexer.wk.weight") == "mtp.0.indexer.wk"
    assert logical_name("model.hc_head.hc_head_fn") == "hc_head.fn"
    assert logical_name("lm_head.weight") == "lm_head"
    with pytest.raises(KeyError):  # allow-pytest.raises: CPU tests run without the root conftest
        logical_name("model.layers.3.self_attn.mystery.weight")


@needs_ckpt
def test_every_index_key_maps_and_round_trips(w):
    assert w.unmapped_keys() == []
    for k in w.weight_map:
        assert checkpoint_key(logical_name(k)) == k


@needs_ckpt
def test_index_equals_expected_inventory(w):
    """The set of checkpoint keys is exactly the config-derived inventory (hc/indexer presence rules, MTP extras)."""
    want = {checkpoint_key(n) for n in all_logical_names(w.cfg)}
    have = set(w.weight_map)
    assert sorted(want - have) == [] and sorted(have - want) == []
    assert len(have) == 2006


@needs_ckpt
def test_layer_inventory_rules(w):
    cfg = w.cfg
    l0, l1, l2 = (set(layer_names(cfg, i)) for i in (0, 1, 2))
    assert "layers.0.mlp.gate_proj" in l0 and "layers.0.indexer.wk" in l0 and "layers.0.moe.router.weight" not in l0
    assert "layers.1.moe.experts.gate_up" in l1 and "layers.1.indexer.wk" in l1
    assert (
        "layers.2.indexer.wk" not in l2 and "layers.2.moe.experts.down" in l2
    )  # shared-indexer layer: no indexer tensors
    mtp = set(layer_names(cfg, 78))
    assert "mtp.0.eh_proj" in mtp and "mtp.0.indexer.wk" in mtp and not any(n.startswith("mtp.0.hc.") for n in mtp)


@needs_ckpt
@pytest.mark.parametrize("layer", [0, 1, 2, 78])
def test_shard_headers_match_config(w, layer):
    """Layers 0 (dense), 1 (full indexer, MoE), 2 (shared indexer, MoE) and the MTP layer: dtype and shape of every tensor from shard headers."""
    names = layer_names(w.cfg, layer)
    assert w.check_headers(names) == []


@needs_ckpt
def test_global_headers(w):
    cfg = w.cfg
    assert w.check_headers(["embed", "lm_head", "final_norm"]) == []
    assert tuple(w.header("hc_head.fn")["shape"]) == (cfg.hc_mult, cfg.hc_flat_dim)  # DSV4-like [4, 24576]
    assert tuple(w.header("hc_head.base")["shape"]) == (cfg.hc_mult,)
    assert tuple(w.header("hc_head.scale")["shape"]) == (1,)


@needs_ckpt
def test_header_reader_does_not_need_data(w):
    h = read_header(w.shard_of("layers.1.moe.experts.gate_up"))
    t = h[checkpoint_key("layers.1.moe.experts.gate_up")]
    s, e = t["data_offsets"]
    assert e - s == 256 * 4096 * 6144 * 2  # bf16


@needs_ckpt
def test_lazy_expert_split_reads_one_slab(w):
    gu = w.load_expert_fused(1, 5, "gate_up")
    assert tuple(gu.shape) == (4096, 6144)
    gate, up = w.split_gate_up(gu)
    assert tuple(gate.shape) == tuple(up.shape) == (2048, 6144)
    assert (gate == gu[:2048]).all() and (up == gu[2048:]).all()  # default order gate_first (UNVERIFIED)
    for order in GATE_UP_ORDERS:
        w2 = Hy4Weights(CKPT, w.cfg, gate_up_order=order)
        g2, u2 = w2.split_gate_up(gu)
        assert g2.shape == u2.shape == (2048, 6144)
    assert (Hy4Weights(CKPT, w.cfg, "up_first").split_gate_up(gu)[0] == gu[2048:]).all()
    assert (Hy4Weights(CKPT, w.cfg, "interleaved").split_gate_up(gu)[1] == gu[1::2]).all()
    down = w.load_expert_fused(1, 5, "down")
    assert tuple(down.shape) == (6144, 2048)
    with pytest.raises(IndexError):  # allow-pytest.raises: CPU tests run without the root conftest
        w.load_expert_fused(1, 256, "down")
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU tests run without the root conftest
        Hy4Weights(CKPT, w.cfg, gate_up_order="bogus")


@needs_ckpt
def test_expert_slab_is_distinct_per_expert(w):
    a = w.load_expert_fused(1, 0, "down")
    b = w.load_expert_fused(1, 1, "down")
    assert not (a == b).all()


@needs_ckpt
def test_ckpt_stamp(tmp_path):
    ident = ckpt_stamp.checkpoint_identity(CKPT)
    assert ident["config_sha256"].startswith("a2c0dc5e")  # decision D0-1
    assert ident["index_sha256"].startswith("2a0954dc")
    root = str(tmp_path / "cache")
    ckpt_stamp.ensure_stamp(root, CKPT)
    with open(os.path.join(root, ckpt_stamp.STAMP)) as f:
        rec = json.load(f)
    assert rec["ckpt_id"] == ident["ckpt_id"] and rec["ckpt_dir"] == CKPT
    ckpt_stamp._checked.clear()
    ckpt_stamp.ensure_stamp(root, CKPT)  # same checkpoint: fine
    rec["ckpt_id"] = "0" * 64
    with open(os.path.join(root, ckpt_stamp.STAMP), "w") as f:
        json.dump(rec, f)
    ckpt_stamp._checked.clear()
    with pytest.raises(RuntimeError):  # allow-pytest.raises: CPU tests run without the root conftest
        ckpt_stamp.ensure_stamp(root, CKPT)
    ckpt_stamp.ensure_stamp(None, CKPT)
    ckpt_stamp.ensure_stamp("0", CKPT)
