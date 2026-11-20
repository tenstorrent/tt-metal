# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the standalone decoder-layer reference and the HF recording harness
(tenstorrent/tt-metal#65893).

reference/decoder.py is what the device decoder layer is validated against on small configs, and
reference/hf_harness.py records the goldens the checkpoint tests use. This file pins both:

  * a small HYV4Model (dense + full indexer, MoE + full indexer, MoE + shared indexer: the three
    block types of layers 0-5) against the reference, layer by layer on HF's own layer inputs and
    as a free-running stack, on every recorded stage;
  * the condition under which the reference's dense attention is HF's attention: the DSA picks
    cover every visible key up to index_topk tokens and stop covering them one token later;
  * the same model in the checkpoint's dtype layout (BF16 with HF's fp32 exceptions) against the
    fp32 reference holding BF16 streams, which is the noise floor a device comparison against a
    BF16 golden inherits;
  * the reference with checkpoint weights against the cached goldens of the real model.

Pure CPU (torch vs torch), no device. Skipped when the installed transformers predates ``hy_v4``.
"""

import os
from pathlib import Path

import pytest
import torch

from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig
from models.experimental.hy4_preview.reference.decoder import (
    STAGES,
    DecoderConfig,
    Hy4DecoderLayer,
    Hy4DecoderStack,
    checkpoint_names,
    dense_prefill_inputs,
)
from models.experimental.hy4_preview.reference.ihc import IHCConfig
from models.experimental.hy4_preview.reference.moe import MoEConfig

hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config  # noqa: E402

HIDDEN = 96
NUM_LAYERS = 3
MLP_TYPES = ["dense", "sparse", "sparse"]
INDEXER_TYPES = ["full", "full", "shared"]
LAYER_IDS = ["dense-full", "moe-full", "moe-shared"]


def _config(index_topk=64) -> DecoderConfig:
    return DecoderConfig(
        attention=AttentionConfig(
            hidden_size=HIDDEN,
            num_heads=4,
            q_lora_rank=48,
            kv_lora_rank=32,
            qk_nope_head_dim=24,
            qk_rope_head_dim=8,
            v_head_dim=16,
        ),
        moe=MoEConfig(hidden_size=HIDDEN, num_experts=32, top_k=4, intermediate_size=48, dense_intermediate_size=160),
        ihc=IHCConfig(dim=HIDDEN),
        index_topk=index_topk,
    )


def _hf_config(cfg: DecoderConfig) -> HYV4Config:
    a, m, h = cfg.attention, cfg.moe, cfg.ihc
    return HYV4Config(
        vocab_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=cfg.hidden_size,
        intermediate_size=m.dense_intermediate_size,
        moe_intermediate_size=m.intermediate_size,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=a.num_heads,
        q_lora_rank=a.q_lora_rank,
        kv_lora_rank=a.kv_lora_rank,
        qk_nope_head_dim=a.qk_nope_head_dim,
        qk_rope_head_dim=a.qk_rope_head_dim,
        v_head_dim=a.v_head_dim,
        n_routed_experts=m.num_experts,
        num_experts_per_tok=m.top_k,
        n_shared_experts=m.n_shared_experts,
        routed_scaling_factor=m.routed_scaling_factor,
        swiglu_limit=m.swiglu_limit,
        mlp_layer_types=list(MLP_TYPES),
        indexer_types=list(INDEXER_TYPES),
        index_topk=cfg.index_topk,
        index_head_dim=16,
        index_n_heads=4,
        hc_mult=h.n,
        hc_magnitude=h.magnitude,
        hc_eps=h.eps,
        rms_norm_eps=cfg.norm_eps,
        max_position_embeddings=4096,
        rope_parameters={"rope_type": "default", "rope_theta": a.rope_theta},
        attn_implementation="eager",
    )


def _randomize(model, seed):
    """Parameters at the scale a trained model has, every one away from its trivial init: linear
    outputs of order one, norm weights and sinks spread around their init, iHC logits of order one
    with distinct pre / post scales, and a router bias large enough to change which experts win."""
    g = torch.Generator().manual_seed(seed)
    tensors = dict(model.named_parameters())
    tensors.update(model.named_buffers())
    with torch.no_grad():
        for name, p in tensors.items():
            leaf = name.rsplit(".", 1)[-1]
            if "inv_freq" in name:
                continue
            if leaf in ("fn", "hc_fn"):
                value = 0.05 * torch.randn(p.shape, generator=g)
            elif leaf in ("scale", "hc_scale"):
                value = torch.tensor([1.0, 0.5])[: p.numel()]
            elif leaf == "e_score_correction_bias":
                value = 0.1 * torch.randn(p.shape, generator=g)
            elif leaf in ("base", "hc_base", "sinks"):
                value = torch.randn(p.shape, generator=g)
            elif p.dim() == 1:  # norm weights (and the indexer's LayerNorm bias)
                value = 1.0 + 0.5 * torch.randn(p.shape, generator=g)
            elif leaf == "weight" and name.endswith("embed_tokens.weight"):
                value = torch.randn(p.shape, generator=g)
            else:
                value = torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5
            p.copy_(value)
    return model


def _hf_model(cfg: DecoderConfig, seed=11):
    return _randomize(hy_v4.HYV4Model(_hf_config(cfg)).eval(), seed)


def _reference_layer(cfg: DecoderConfig, model, layer: int) -> Hy4DecoderLayer:
    tensors = hf_harness.layer_tensors(model.state_dict(), layer)
    ref = Hy4DecoderLayer(cfg, dense=cfg.is_dense(layer)).eval()
    return ref.load({name: tensors[name].float().clone() for name in ref.parameter_names})


def _reference_stack(cfg: DecoderConfig, model) -> Hy4DecoderStack:
    stack = Hy4DecoderStack(cfg, NUM_LAYERS).eval()
    for index in range(NUM_LAYERS):
        stack.layers[index] = _reference_layer(cfg, model, index)
    return stack


def _embeds(seq_len, seed=12):
    return torch.randn(seq_len, HIDDEN, generator=torch.Generator().manual_seed(seed))


def _relative(a, b):
    """max|a - b| relative to max|b|."""
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


# ---------------------------------------------------------------------------------------------
# fp32: the reference is HF
# ---------------------------------------------------------------------------------------------
def test_names_cover_the_hf_layer():
    """Every tensor an HF layer holds is either taken by the reference or belongs to the indexer."""
    cfg = _config()
    model = _hf_model(cfg)
    for layer in range(NUM_LAYERS):
        tensors = hf_harness.layer_tensors(model.state_dict(), layer)
        taken = set(Hy4DecoderLayer(cfg, dense=cfg.is_dense(layer)).parameter_names)
        assert taken <= set(tensors), sorted(taken - set(tensors))
        left = {name for name in tensors if name not in taken}
        assert all(name.startswith("self_attn.indexer.") for name in left), sorted(left)
        assert bool(left) == (INDEXER_TYPES[layer] == "full")


@pytest.mark.parametrize("layer", range(NUM_LAYERS), ids=LAYER_IDS)
@pytest.mark.parametrize("seq_len", [1, 12, 40], ids=["S1", "S12", "S40"])
def test_layer_matches_hf(layer, seq_len):
    """One layer on the input HF's own layer saw: every stage within 1e-5 of HF, relative."""
    cfg = _config()
    model = _hf_model(cfg)
    golden = hf_harness.record_model(model, inputs_embeds=_embeds(seq_len))
    hf = golden["layers"][layer]
    assert hf["dsa_dropped"] == 0

    ref = _reference_layer(cfg, model, layer)
    rec = {}
    with torch.no_grad():
        out = ref(hf["streams_in"], *dense_prefill_inputs(cfg, seq_len), rec)
    assert out.dtype == hf["streams_out"].dtype and out.shape == hf["streams_out"].shape
    worst = {key: _relative(rec[key], hf[key]) for key in STAGES}
    assert max(worst.values()) <= 1e-5, worst
    assert _relative(out, hf["streams_out"]) <= 1e-5
    # the layer is not a pass-through: each sublayer moves the streams
    assert _relative(hf["streams_mid"], hf["streams_in"]) > 0.05
    assert _relative(hf["streams_out"], hf["streams_mid"]) > 0.05
    if not ref.dense:
        # Handed HF's routing, the layer is still HF; handed another routing, it follows that one.
        routing = ref.mlp.dense_weights(hf["router_weights"], hf["router_indices"])
        with torch.no_grad():
            forced = ref(hf["streams_in"], *dense_prefill_inputs(cfg, seq_len), expert_weights=routing)
            moved = ref(hf["streams_in"], *dense_prefill_inputs(cfg, seq_len), expert_weights=routing.roll(1, dims=1))
        assert _relative(forced, hf["streams_out"]) <= 1e-5
        assert _relative(moved, hf["streams_out"]) > 1e-2


@pytest.mark.parametrize("seq_len", [1, 12, 40], ids=["S1", "S12", "S40"])
def test_stack_matches_hf(seq_len):
    """The three layers free-running from the embedding output: the compounded error stays within 1e-5."""
    cfg = _config()
    model = _hf_model(cfg)
    embeds = _embeds(seq_len)
    golden = hf_harness.record_model(model, inputs_embeds=embeds)
    records = []
    with torch.no_grad():
        out = _reference_stack(cfg, model)(embeds, record=records)
    for layer, (rec, hf) in enumerate(zip(records, golden["layers"])):
        worst = {key: _relative(rec[key], hf[key]) for key in STAGES}
        assert max(worst.values()) <= 1e-5, (layer, worst)
        if "router_indices" in hf:
            _, picks = _reference_layer(cfg, model, layer).mlp.route(hf["mlp_in"])
            assert torch.equal(picks.sort(dim=-1).values, hf["router_indices"].sort(dim=-1).values)
    assert _relative(out, golden["layers"][-1]["streams_out"]) <= 1e-5


def test_recorder_takes_token_ids():
    """The recorder run from token ids records the embedding it looked up, and the streams entering
    layer 0 are that embedding in each of the n streams."""
    cfg = _config()
    model = _hf_model(cfg)
    ids = torch.randint(3, 64, (20,), generator=torch.Generator().manual_seed(13))
    golden = hf_harness.record_model(model, input_ids=ids)
    assert torch.equal(golden["embedding"], model.embed_tokens.weight[ids])
    first = golden["layers"][0]["streams_in"]
    assert first.shape == (20, cfg.ihc.n, HIDDEN)
    assert all(torch.equal(first[:, i], golden["embedding"]) for i in range(cfg.ihc.n))
    assert torch.equal(golden["positions"], torch.arange(20))


# ---------------------------------------------------------------------------------------------
# When dense attention is HF's attention
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("extra", [0, 1, 24], ids=["S=topk", "S=topk+1", "S=topk+24"])
def test_dense_attention_is_exact_up_to_index_topk(extra):
    """Up to index_topk tokens every query picks every key and the dense reference is HF. One token
    more and the last query loses a visible key: the picks start to matter, on the full layers and
    on the shared layer that reuses them, and the reference refuses the prompt."""
    cfg = _config(index_topk=16)
    seq_len = cfg.index_topk + extra
    model = _hf_model(cfg)
    embeds = _embeds(seq_len)
    golden = hf_harness.record_model(model, inputs_embeds=embeds)
    dropped = [rec["dsa_dropped"] for rec in golden["layers"]]
    for rec in golden["layers"]:
        assert rec["topk_indices"].shape == (seq_len, cfg.index_topk)
    # the shared layer attends with the picks of the full layer below it
    assert torch.equal(golden["layers"][2]["topk_indices"], golden["layers"][1]["topk_indices"])

    stack = _reference_stack(cfg, model)
    if extra == 0:
        assert dropped == [0, 0, 0]
        with torch.no_grad():
            out = stack(embeds)
        assert _relative(out, golden["layers"][-1]["streams_out"]) <= 1e-5
        return

    # query q sees q + 1 keys and keeps index_topk of them
    expected = sum(max(0, q + 1 - cfg.index_topk) for q in range(seq_len))
    assert dropped == [expected] * NUM_LAYERS
    with pytest.raises(AssertionError, match="index_topk"):  # allow-pytest.raises: host-side check
        stack(embeds)
    # Forcing the dense reference through anyway shows the difference is real, not a formality.
    stack.cfg = _config(index_topk=seq_len)
    with torch.no_grad():
        out = stack(embeds)
    assert _relative(out, golden["layers"][-1]["streams_out"]) > (1e-2 if extra > 1 else 1e-5)


# ---------------------------------------------------------------------------------------------
# The checkpoint's dtype layout: BF16 with HF's fp32 exceptions
# ---------------------------------------------------------------------------------------------
def _as_checkpoint_dtypes(model):
    """BF16 everywhere except what HF keeps in fp32 (``_keep_in_fp32_modules_strict``), which is how
    from_pretrained leaves the real model."""
    keep = set(model._keep_in_fp32_modules_strict)
    model.to(torch.bfloat16)
    tensors = dict(model.named_parameters())
    tensors.update(model.named_buffers())
    for name, p in tensors.items():
        if keep & set(name.split(".")):
            p.data = p.data.float()
    return model


@pytest.mark.parametrize("seq_len", [40], ids=["S40"])
def test_bf16_model_noise_floor(seq_len):
    """HF in the checkpoint's dtypes against the fp32 reference holding BF16 streams. They differ by
    BF16 rounding inside HF's sublayers, so this is not an equality: it measures how closely an
    exact fp32 implementation can track a BF16 golden, layer by layer and free-running."""
    cfg = _config()
    model = _as_checkpoint_dtypes(_hf_model(cfg))
    embeds = _embeds(seq_len).to(torch.bfloat16)
    golden = hf_harness.record_model(model, inputs_embeds=embeds)
    assert golden["layers"][0]["streams_in"].dtype == torch.bfloat16
    assert golden["layers"][0]["attn_post"].dtype == torch.float32

    stack = _reference_stack(cfg, model)
    inputs = dense_prefill_inputs(cfg, seq_len)
    free = []
    with torch.no_grad():
        stack(embeds, record=free)
    for layer, hf in enumerate(golden["layers"]):
        assert hf["dsa_dropped"] == 0
        rec = {}
        with torch.no_grad():
            out = stack.layers[layer](hf["streams_in"], *inputs, rec)
        assert out.dtype == torch.bfloat16
        update = _pcc(out.float() - hf["streams_in"].float(), hf["streams_out"].float() - hf["streams_in"].float())
        stages = {key: _pcc(rec[key].float(), hf[key].float()) for key in STAGES}
        free_pcc = _pcc(free[layer]["streams_out"].float(), hf["streams_out"].float())
        print(f"bf16 noise floor layer {layer}: update pcc={update:.6f} free-running pcc={free_pcc:.6f} {stages}")
        assert min(stages.values()) >= 0.999, stages
        assert update >= 0.999 and free_pcc >= 0.999


# ---------------------------------------------------------------------------------------------
# The real model: cached goldens of layers 0-5, and the reference with checkpoint weights
# ---------------------------------------------------------------------------------------------
def _golden():
    golden = hf_harness.find_golden()
    if golden is None:
        pytest.skip(f"no golden under {hf_harness.GOLDEN_ROOT}: record one with reference/hf_harness.py")
    return golden


def test_golden_is_complete_and_dense():
    """The cached golden of the real layers 0-5: every stage is there, the streams enter layer 0 as
    the embedding and chain from layer to layer, and no layer's DSA picks dropped a visible key."""
    golden = _golden()
    meta = golden["meta"]
    S, n, hidden = meta["seq_len"], IHCConfig().n, IHCConfig().dim
    assert meta["num_layers"] == hf_harness.DEFAULT_NUM_LAYERS and S <= meta["index_topk"]
    assert meta["mlp_layer_types"] == ["dense"] + ["sparse"] * 5
    assert meta["indexer_types"] == ["full", "full", "shared", "shared", "shared", "full"]
    assert golden["embedding"].shape == (S, hidden) and golden["embedding"].dtype == torch.bfloat16
    layers = golden["layers"]
    assert all(torch.equal(layers[0]["streams_in"][:, i], golden["embedding"]) for i in range(n))
    for index, rec in enumerate(layers):
        assert rec["dsa_dropped"] == 0, f"layer {index} dropped {rec['dsa_dropped']} visible keys"
        assert rec["topk_indices"].shape == (S, S)
        for key in STAGES:
            expected = (S, n) if key.endswith("_post") else (S, n, hidden) if key.startswith("streams") else (S, hidden)
            assert rec[key].shape == expected, (index, key, rec[key].shape)
            assert torch.isfinite(rec[key].float()).all(), (index, key)
        assert ("router_indices" in rec) == (index > 0)
        if index:
            assert torch.equal(rec["streams_in"], layers[index - 1]["streams_out"])


@pytest.mark.parametrize("layer", range(hf_harness.DEFAULT_NUM_LAYERS), ids=lambda layer: f"layer{layer}")
def test_reference_matches_golden(layer):
    """The fp32 reference with the checkpoint's weights, fed the golden input of the layer, against
    what the real BF16 model produced. One layer at a time: a MoE layer is 39 GB in fp32."""
    golden = _golden()
    hf = golden["layers"][layer]
    if not os.environ.get(hf_harness.HY4_HF_MODEL_ENV):
        pytest.skip(f"{hf_harness.HY4_HF_MODEL_ENV} is not set")
    available_gb = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) / 2**20
    if available_gb < 120:
        pytest.skip(f"only {available_gb:.0f} GB of host memory available")

    cfg = DecoderConfig()
    dense = cfg.is_dense(layer)
    ref = Hy4DecoderLayer(cfg, dense=dense, init=False).eval()
    names = checkpoint_names(layer, dense)
    for name, short in zip(names, ref.parameter_names):  # one at a time: adopt, do not hold two copies
        ref.load_one(short, hf_harness.checkpoint_tensors([name])[0])

    S = hf["streams_in"].shape[0]
    rec = {}
    with torch.no_grad():
        out = ref(hf["streams_in"], *dense_prefill_inputs(cfg, S), rec)
    update = _pcc(out.float() - hf["streams_in"].float(), hf["streams_out"].float() - hf["streams_in"].float())
    stages = {key: round(_pcc(rec[key].float(), hf[key].float()), 6) for key in STAGES}
    worst = {key: f"{_relative(rec[key].float(), hf[key].float()):.2e}" for key in STAGES}
    print(f"reference vs golden layer {layer}: update pcc={update:.6f}\n  pcc {stages}\n  max rel err {worst}")
    if not dense:
        _, picks = ref.mlp.route(hf["mlp_in"].float())
        same = (picks.sort(dim=-1).values == hf["router_indices"].sort(dim=-1).values).all(dim=-1).float().mean()
        print(f"  router picks identical for {same:.4f} of {S} tokens (reference routed on the golden mlp_in)")
        assert same >= 0.95
    assert min(stages.values()) >= 0.999, stages
    assert update >= 0.999
