# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""The converter packs a layer the way the device modules expect (host only, no card).

A tiny synthetic checkpoint holds one local and one global layer under the publisher's tensor names, filled from
the same random HF state dict the attention tests use. ``convert_layer_dense`` writes it; every attention and
router tensor must equal, bit for bit, what the modules' own HF packing helpers (``host_weights_from_hf``,
``prepare_moe_weights``) make from that state dict. This catches a q/gate/rel/sink column order or conv tap-order
change in the converter, which otherwise only the full-model golden test would see.

Run without a card by pointing ttnn at a mock cluster:
    TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml \
        pytest --noconftest -c /dev/null models/demos/inkling2_mini/tests/test_convert_pack.py
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn")
safetensors_torch = pytest.importorskip("safetensors.torch")

from models.demos.inkling2_mini.convert import convert as C
from models.demos.inkling2_mini.convert.source import Checkpoint
from models.demos.inkling2_mini.tests.test_attention import random_state_dict
from models.demos.inkling2_mini.tt.attention import host_weights_from_hf
from models.demos.inkling2_mini.tt.moe import prepare_moe_weights

H = C.HIDDEN

# publisher checkpoint name (under model.llm.layers.N.) -> HF attention state-dict name
# (transformers' inkling2_loop conversion mapping)
ATTN = {
    "attn.wq_du.weight": "q_proj.linear.weight",
    "attn.gate_attn.weight": "o_proj.gate_proj.weight",
    "attn.wr_du.weight": "relative_position_bias.linear.weight",
    "attn.attn_sink_proj.weight": "sink_proj.weight",
    "attn.wo_ud.weight": "o_proj.linear.weight",
    "attn.q_l2_clip.scale": "q_proj.norm.scale",
    "attn.rel_l2_clip.scale": "relative_position_bias.norm.scale",
    "attn.rel_logits_proj.proj": "relative_position_bias.proj",
}
for _kv in "kv":
    ATTN[f"attn.w{_kv}_dv.weight"] = f"{_kv}_proj.linear.weight"
    ATTN[f"attn.{_kv}_sconv.weight"] = f"{_kv}_proj.sconv.conv1d.weight"  # local
    ATTN[f"attn.{_kv}_sconv.weight_proj"] = f"{_kv}_proj.sconv.weight_proj.weight"  # global
    ATTN[f"attn.{_kv}_sconv.static_w"] = f"{_kv}_proj.sconv.static_weight"  # global


def synthetic_layer(layer, g):
    """Publisher-named tensors of one dense layer + the HF tensors they came from."""
    is_global = layer in C.GLOBAL_LAYERS
    sd = random_state_dict(is_global, seed=layer + 1)
    p = f"{C.P}layers.{layer}."
    ck = {p + k: sd[v] for k, v in ATTN.items() if v in sd}
    norms = ["q_input_norm", "k_input_norm", "v_input_norm"] if is_global else ["attn_norm"]
    for n in norms + ["mlp_norm"]:
        ck[f"{p}{n}.weight"] = (1 + 0.1 * torch.randn(H, generator=g)).bfloat16()
    for n in ("attn_sconv", "mlp_sconv"):
        ck[f"{p}{n}.weight"] = torch.randn(H, 1, 4, generator=g)
    if layer > 0:
        ck[f"{C.P}attn_res.attn_queries.{layer - 1}"] = torch.randn(3 if is_global else 1, H, generator=g)
    ck[f"{C.P}attn_res.mlp_queries.{layer}"] = torch.randn(H, generator=g)
    ck[p + "mlp.gate.weight"] = torch.randn(C.N_EXPERTS, H, generator=g) / H**0.5
    ck[p + "mlp.gate.bias"] = 0.05 * torch.randn(C.N_EXPERTS, generator=g)
    return {k: v.contiguous() for k, v in ck.items()}, sd


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    g = torch.Generator().manual_seed(0)
    src, out = tmp_path_factory.mktemp("ckpt"), tmp_path_factory.mktemp("cache")
    tensors, sds = {}, {}
    for layer in (0, 5):  # local, global
        t, sds[layer] = synthetic_layer(layer, g)
        tensors |= t
    safetensors_torch.save_file(tensors, str(src / "model.safetensors"))
    w = C.Writer(out)
    ck = Checkpoint(src)
    for layer in sds:
        C.convert_layer_dense(ck, w, layer)
    got = {(e["layer"], e["name"]): ttnn.to_torch(ttnn.load_tensor(out / e["file"])) for e in w.entries}
    return got, tensors, sds


@pytest.mark.parametrize("layer", [0, 5], ids=["local", "global"])
def test_attention_packing(converted, layer):
    got, _, sds = converted
    want = host_weights_from_hf(sds[layer], layer in C.GLOBAL_LAYERS)
    assert set(want) <= {n.removeprefix("attn.") for (L, n) in got if L == layer}
    for name, t in want.items():
        g = got[(layer, f"attn.{name}")]
        w = ttnn.to_torch(t)
        assert g.shape == w.shape, (name, g.shape, w.shape)
        assert torch.equal(g.float(), w.float()), name


@pytest.mark.parametrize("layer", [0, 5], ids=["local", "global"])
def test_router_packing(converted, layer):
    got, tensors, _ = converted
    p = f"{C.P}layers.{layer}."
    want = prepare_moe_weights(
        tensors[p + "mlp_norm.weight"],
        tensors[p + "mlp.gate.weight"],
        tensors[p + "mlp.gate.bias"],
        torch.zeros(1, 64, 32),  # experts are not packed by convert_layer_dense
        torch.zeros(1, 32, 32),
    )
    for name in ("mlp_norm", "router.w", "router.bias"):
        g, w = got[(layer, name)], ttnn.to_torch(want[name])
        assert g.shape == w.shape and torch.equal(g.float(), w.float()), name
