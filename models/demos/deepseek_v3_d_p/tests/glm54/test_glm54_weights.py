# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GLM-5.4 weight tiers (tests/glm54_weights.py) and the synthetic damp gates.

The tier-2 tests read the GLM-5.3 checkpoint, golden trace and TTNN cache on Weka (read only) and skip where they
are absent; a skip is not evidence for them.
"""

import math
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_3.block import glm_decoder_layer_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import (
    LoopPass2MLAReference,
    damp_gate_reference,
    glm54_pass2_layer_reference,
    glm54_stack_reference,
    pass1_indices,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import (
    GATE_BETA_CENTER,
    GATE_LOGIT_STD,
    GLM54_SYNTHETIC_SEED,
    synthetic_gate_weights,
    synthetic_gates,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config, glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tests.glm54_weights import (
    GLM53_CHECKPOINT,
    GLM53_TTNN_CACHE,
    GLM54ReducedConfig,
    glm54_checkpoint_layer,
    glm54_reduced_config,
    glm54_tier1_weights,
    glm54_tier2_weights,
    tier2_provenance,
    tier2_sources,
)
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter

HIDDEN = GLM54Config.EMB_SIZE
GOLDEN = Path(GLM53Adapter.test_prefill_trace_default) / "decoder_io"
needs_checkpoint = pytest.mark.skipif(
    not (tier2_sources()[0] / "model.safetensors.index.json").is_file(), reason="GLM-5.3 checkpoint absent"
)
_SOURCE_ENV = (GLM53Adapter.env_var, "PREFILL_HF_MODEL", "PREFILL_TTNN_CACHE")
needs_golden = pytest.mark.skipif(not GOLDEN.is_dir(), reason="GLM-5.3 golden trace absent")


def _golden_rows(layer: int, rows) -> torch.Tensor:
    """Rows of the vLLM decoder output of ``layer`` (GLM-5.3, 56320-token prompt) as float32 [len(rows), hidden]."""
    name = f"decoder_output_layer_{layer}"
    with safe_open(str(GOLDEN / name / "rows_00000000_00056320.safetensors"), "pt") as f:
        s = f.get_slice(name)
        if isinstance(rows, slice):
            return s[rows].float()
        return torch.stack([s[int(i) : int(i) + 1][0] for i in rows]).float()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.detach().flatten().double(), b.detach().flatten().double()]))[0, 1])


# --- synthetic gates ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("per_channel", [False, True], ids=["scalar", "per_channel"])
def test_gate_weights_shape_dtype_and_seed(per_channel, expect_error):
    width = HIDDEN if per_channel else 1
    g1 = synthetic_gate_weights(HIDDEN, 1, seed=3, per_channel=per_channel)
    assert {k: (tuple(v.shape), v.dtype) for k, v in g1.items()} == {
        "norm_weight": ((HIDDEN,), torch.bfloat16),
        "proj_weight": ((width, HIDDEN), torch.bfloat16),
        "proj_bias": ((width,), torch.bfloat16),
    }
    again = synthetic_gate_weights(HIDDEN, 1, seed=3, per_channel=per_channel)
    assert all(torch.equal(g1[k], again[k]) for k in g1)
    other_seed = synthetic_gate_weights(HIDDEN, 1, seed=4, per_channel=per_channel)
    assert not torch.equal(g1["proj_weight"], other_seed["proj_weight"])
    pass2 = synthetic_gate_weights(HIDDEN, 2, seed=3, per_channel=per_channel)
    assert not torch.equal(g1["proj_weight"], pass2["proj_weight"])  # A1b: own weights per pass
    with expect_error(ValueError, "loop pass must be one of"):
        synthetic_gate_weights(HIDDEN, 3, seed=3)


def test_gate_bias_is_the_logit_of_the_centre():
    # logit(0.2) = ln(1/4) = -1.3863, logit(0.9) = ln(9) = 2.1972; bf16 keeps 8 significant bits.
    b1 = synthetic_gate_weights(HIDDEN, 1, seed=0)["proj_bias"].float()
    b2 = synthetic_gate_weights(HIDDEN, 2, seed=0)["proj_bias"].float()
    assert abs(float(b1) - (-1.3863)) < 0.01 and abs(float(b2) - 2.1972) < 0.01
    assert GATE_BETA_CENTER == {1: 0.2, 2: 0.9}


def test_gate_spread_on_independent_unit_rms_inputs():
    """Inputs with no shared direction: the logit is N(bias, GATE_LOGIT_STD^2) per token, so the median beta is
    the centre and +-2 sigma of pass 1 spans about 0.11..0.33."""
    x = torch.randn(4096, HIDDEN, generator=torch.Generator().manual_seed(9))
    for k, center in GATE_BETA_CENTER.items():
        g = synthetic_gate_weights(HIDDEN, k, seed=GLM54_SYNTHETIC_SEED)
        _, beta = damp_gate_reference(x, x, eps=1e-5, **g)
        logit = torch.logit(beta.double()).flatten()
        assert abs(float(logit.std()) - GATE_LOGIT_STD) < 0.03
        assert abs(float(beta.median()) - center) < 0.02
    lo, hi = 1 / (1 + math.exp(1.3863 + 2 * GATE_LOGIT_STD)), 1 / (1 + math.exp(1.3863 - 2 * GATE_LOGIT_STD))
    assert 0.11 < lo < 0.12 and 0.32 < hi < 0.34  # 0.112, 0.330


@needs_golden
def test_gate_beta_on_glm53_activations():
    """The documented tier-2 range (synthetic_gates docstring): gate 1 on x0 = layer-25 output with h1 = layer-53
    output, gate 2 on x1 = gate-1 output, 8192 rows spread over the 56320-token golden prompt."""
    rows = torch.linspace(0, 56319, 8192).long()
    x0, h1 = _golden_rows(25, rows), _golden_rows(53, rows)
    g1, g2 = synthetic_gates(HIDDEN, 2, seed=GLM54_SYNTHETIC_SEED)
    x1, beta1 = damp_gate_reference(x0, h1, eps=1e-5, **g1)
    _, beta2 = damp_gate_reference(x1, x1, eps=1e-5, **g2)  # beta2 reads x1 only; h2 is not in any trace
    q = torch.tensor([0.02, 0.5, 0.98])
    b1, b2 = torch.quantile(beta1.flatten(), q).tolist(), torch.quantile(beta2.flatten(), q).tolist()
    print(f"beta1 q02/50/98 {[round(v, 3) for v in b1]}, beta2 {[round(v, 3) for v in b2]}")
    assert 0.1 <= b1[0] and b1[2] <= 0.3 and 0.15 <= b1[1] <= 0.25
    assert 0.8 <= b2[0] and b2[2] <= 0.97 and 0.87 <= b2[1] <= 0.93


# --- tier 1 ---------------------------------------------------------------------------------------------------


def test_reduced_config_roles():
    c = glm54_reduced_config(1024)
    assert (c.num_hidden_layers, c.first_k_dense_replace, c.loop_first_layer, c.loop_last_layer) == (4, 1, 1, 2)
    assert c.indexer_types == ["full", "full", "shared", "full"]
    assert c.loop_count == 2 and c.loop_window_size == 128 and c.hidden_size == HIDDEN
    assert GLM54ReducedConfig.indexer_types() == c.indexer_types
    # Everything except the layer count and roles is GLM-5.4's own (production widths for the device).
    full = vars(glm_5_4_hf_config(1024))
    roles = {"num_hidden_layers", "first_k_dense_replace", "indexer_types", "loop_first_layer", "loop_last_layer"}
    assert {k: v for k, v in vars(c).items() if k not in roles} == {k: v for k, v in full.items() if k not in roles}


def test_tier1_weights_run_the_reference_stack():
    """Tier 1 loads into the GLM-5.4 reference stack (2 chunks, window 128 crossing the chunk boundary) in the
    TtPrefillTransformer state-dict form, deterministically, and the gates damp as configured."""
    seq, chunk = 256, 128
    c = glm54_reduced_config(seq)
    w = glm54_tier1_weights(c)
    layers = w.state_dict["layers"]
    assert set(w.state_dict) == {"embed_weight", "layers"} and len(layers) == 4
    assert "ffn_weights" in layers[0] and all("routed_expert_weights" in layers[i] for i in (1, 2, 3))
    assert len(layers[1]["routed_expert_weights"]) == GLM54Config.NUM_ROUTED_EXPERTS
    assert not any(k.startswith("indexer.") for k in layers[2]["mla_weights"])  # shared layer owns no indexer
    assert any(k.startswith("indexer.") for k in layers[1]["mla_weights"])
    assert not torch.equal(layers[1]["gate_weights"]["weight"], layers[2]["gate_weights"]["weight"])
    assert w.provenance["seed"] == GLM54_SYNTHETIC_SEED and w.provenance["gates"]["gate_seeds"] == [100002, 100003]

    # Input: rows of the synthetic embedding, as a device run feeds the stack.
    tokens = torch.randint(0, c.vocab_size, (seq,), generator=torch.Generator().manual_seed(5))
    x = w.state_dict["embed_weight"][tokens].unsqueeze(0)
    r1 = glm54_stack_reference(c, list(layers), w.gates, x, chunk_size=chunk)
    r2 = glm54_stack_reference(c, list(layers), w.gates, x, chunk_size=chunk)
    assert torch.equal(r1.output, r2.output)
    assert r1.output.shape == (1, seq, HIDDEN) and torch.isfinite(r1.output).all()
    cp = r1.checkpoints
    assert cp["L1.pass2.kvpe"].shape == (seq, c.kv_lora_rank + c.qk_rope_head_dim)
    for k, center in GATE_BETA_CENTER.items():
        beta = cp[f"gate{k}.beta"]
        print(f"tier 1 gate {k}: beta min/median/max {beta.min():.3f}/{beta.median():.3f}/{beta.max():.3f}")
        assert abs(float(beta.median()) - center) < 0.15 and float(beta.std()) > 0.005
    # The gate moves x by beta toward the pass output: neither input nor output is passed through unchanged.
    assert not torch.equal(cp["gate1.out"], cp["pass1.out"]) and not torch.equal(cp["gate1.out"], cp["pass1.in"])


# --- tier 2 ---------------------------------------------------------------------------------------------------


def test_tier2_provenance_follows_the_runner_environment(monkeypatch, tmp_path):
    """Provenance records the checkpoint and cache a run actually reads: the runner's env overrides, in its order
    ($GLM53_HF_MODEL before $PREFILL_HF_MODEL), and as many gates as the config's loop passes."""

    def fake_checkpoint(name: str, text: str) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(text)
        (d / "model.safetensors.index.json").write_text("{}")
        return d

    for name in _SOURCE_ENV:
        monkeypatch.delenv(name, raising=False)
    prefill = fake_checkpoint("GLM-5.3-fp8-prefill1", "a")
    monkeypatch.setenv("PREFILL_HF_MODEL", str(prefill))
    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path / "cache"))
    assert tier2_sources() == (prefill, tmp_path / "cache")
    c = glm_5_4_hf_config()
    c.loop_count = 1  # not GLM54Config.LOOP_COUNT (2)
    p = tier2_provenance(c)
    assert p["checkpoint"] == str(prefill) and p["checkpoint_dir_tag"] == "prefill1"
    assert p["config_sha256"] == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"  # sha256(b"a")
    assert p["ttnn_cache"] == str(tmp_path / "cache") and len(p["gates"]["gate_seeds"]) == 1
    own = fake_checkpoint("GLM-5.3-fp8-own2", "b")
    monkeypatch.setenv(GLM53Adapter.env_var, str(own))
    assert tier2_sources()[0] == own


@needs_checkpoint
def test_tier2_provenance(monkeypatch):
    for name in _SOURCE_ENV:
        monkeypatch.delenv(name, raising=False)
    c = glm_5_4_hf_config()
    p = tier2_provenance(c)
    assert p["checkpoint"] == GLM53_CHECKPOINT and p["checkpoint_dir_tag"] == "aca966e4"
    assert p["ttnn_cache"] == GLM53_TTNN_CACHE
    assert len(p["config_sha256"]) == 64 and len(p["index_sha256"]) == 64
    w = glm54_tier2_weights(c)
    assert w.state_dict == {} and len(w.gates) == 2 and w.provenance == p
    assert all(
        torch.equal(a["proj_weight"], b["proj_weight"]) for a, b in zip(w.gates, synthetic_gates(HIDDEN, 2, seed=1))
    )


@needs_checkpoint
@needs_golden
def test_tier2_loop_layer_in_the_reference():
    """GLM-5.4 loop layer 26 from the GLM-5.3 checkpoint: pass 1 on vLLM's own layer-26 input reproduces vLLM's
    layer-26 output (the loaded tensors are the ones vLLM ran), and the same weights run loop pass 2 behind the
    tier-2 gate 1."""
    seq, layer = 512, GLM54Config.LOOP_FIRST_LAYER
    c = glm_5_4_hf_config(seq)
    w = glm54_checkpoint_layer(c, layer)
    assert set(w) == {"attn_norm_weight", "ffn_norm_weight", "mla_weights", "gate_weights"} | {
        "shared_expert_weights",
        "routed_expert_weights",
    }
    moe = {k: w[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")}
    x0 = _golden_rows(layer - 1, slice(0, seq)).to(torch.bfloat16).unsqueeze(0)
    mla = SparseMLAReference(c, w["mla_weights"], seq_len=seq)
    h, kvpe, topk = glm_decoder_layer_reference(
        c,
        None,
        w["attn_norm_weight"],
        w["ffn_norm_weight"],
        x0,
        seq,
        moe_weights=moe,
        return_indexer_topk=True,
        mla_ref=mla,
    )
    want = _golden_rows(layer, slice(0, seq))
    p = _pcc(h.float(), want)
    print(f"tier 2 layer {layer} pass 1 vs vLLM: PCC {p:.5f}")
    assert p > 0.99

    # Pass 2 of the same layer through the shared weights, fed x1 = gate 1(x0, h1) with h1 = vLLM's layer-53 output.
    gate1 = glm54_tier2_weights(c).gates[0]
    x1, _ = damp_gate_reference(x0.float(), _golden_rows(53, slice(0, seq)).unsqueeze(0), eps=c.rms_norm_eps, **gate1)
    attn = LoopPass2MLAReference(mla, window=c.loop_window_size, latent_dim=c.kv_lora_rank)
    out = glm54_pass2_layer_reference(
        c,
        attn,
        w["attn_norm_weight"],
        w["ffn_norm_weight"],
        x1.to(torch.bfloat16),
        kvpe_a=kvpe,
        indices_a=pass1_indices(topk, 0, c.index_topk),
        moe_weights=moe,
    )
    assert out.output.shape == (1, seq, HIDDEN) and torch.isfinite(out.output).all()


@pytest.mark.skipif(not tier2_sources()[1].is_dir(), reason="GLM-5.3 TTNN cache absent")
def test_tier2_ttnn_cache_holds_every_layer():
    """The device side of tier 2: the GLM-5.3 TTNN cache for a Blackhole 8x4 stage (SP on axis 0, TP 4) holds every
    trunk tensor of layers 0-77, which GLM-5.4 reuses as is (pass 2 reads loop layers 26-53 again)."""
    from models.demos.deepseek_v3_d_p.tt.runners.weight_categories import missing_trunk_categories
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE

    cache = tier2_sources()[1] / "glm_5_3_bh_32dev" / "8x4"
    if not cache.is_dir():
        pytest.skip(f"{cache} absent")
    c = glm_5_4_hf_config()
    missing = missing_trunk_categories(
        cache,
        num_layers=GLM54Config.NUM_LAYERS,
        first_layer_idx=0,
        experts_per_chip=GLM54Config.NUM_ROUTED_EXPERTS // 32,
        model_cfg=GLM54Config,
        routed_expert_weights_dtype=DEFAULT_ROUTED_EXPERT_WEIGHTS_DTYPE,
        is_first_rank=True,
        is_last_rank=False,
        mtp_levels=0,
        kv_only_last_layer=False,
        hf_config=c,
        tp_factor=4,
    )
    assert missing == {}, missing
