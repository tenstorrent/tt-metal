# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GLM-5.4 weight tiers (tests/glm54_weights.py) and the synthetic damp gates.

The tier-2 tests read the GLM-5.3 checkpoint, golden trace and TTNN cache on Weka (read only) and skip where they
are absent; a skip is not evidence for them.
"""

import copy
import math
import os
import time
import types
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
from models.demos.deepseek_v3_d_p.tests import glm54_weights as gw
from models.demos.deepseek_v3_d_p.tests import glm_synthetic_weights
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


# --- tier-1 disk cache -------------------------------------------------------------------------------------------


def _tiny_tier1_config(monkeypatch):
    """The reduced stack's layer roles at toy widths (and few, narrow experts), so the generator runs in a second."""
    monkeypatch.setattr(GLM54ReducedConfig, "NUM_ROUTED_EXPERTS", 4)
    monkeypatch.setattr(GLM54ReducedConfig, "MOE_INTERMEDIATE_SIZE", 32)
    c = glm54_reduced_config(64)
    for k, v in dict(hidden_size=64, intermediate_size=96, vocab_size=128, q_lora_rank=32, kv_lora_rank=32).items():
        setattr(c, k, v)
    for k, v in dict(num_attention_heads=2, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16).items():
        setattr(c, k, v)
    c.index_n_heads, c.index_head_dim = 2, 16
    return c


def _flat(w) -> list[torch.Tensor]:
    out = []

    def walk(x):
        if isinstance(x, torch.Tensor):
            out.append(x)
        elif isinstance(x, dict):
            for k in sorted(x):
                walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk([w.state_dict, w.gates])
    return out


def _same(a, b) -> bool:
    fa, fb = _flat(a), _flat(b)
    return len(fa) == len(fb) and all(x.dtype == y.dtype and torch.equal(x, y) for x, y in zip(fa, fb))


def test_tier1_cache_key_tracks_generator_config_and_seed(monkeypatch, tmp_path):
    """The key moves with every input of the tier-1 bytes (seed, a config field, a model constant, a generator module's
    source, a generator function's source, the torch version) and not with the fields the generator does not read."""
    c = glm54_reduced_config(5120)
    base = gw.tier1_cache_key(c)
    assert gw.tier1_cache_key(c) == base

    def moved(**config_changes) -> bool:
        changed = copy.deepcopy(c)
        for k, v in config_changes.items():
            setattr(changed, k, v)
        return gw.tier1_cache_key(changed) != base

    assert gw.tier1_cache_key(c, seed=GLM54_SYNTHETIC_SEED + 1) != base
    assert moved(hidden_size=c.hidden_size + 64)
    assert moved(indexer_types=["full"] * c.num_hidden_layers)
    assert moved(mla_use_output_gate=True)  # a field the generator reads with a default, absent from the config
    assert not moved(max_seq_len=3 * 5120, rope_scaling={**c.rope_scaling, "original_max_position_embeddings": 1})
    assert not moved(loop_window_format="bf16")

    with monkeypatch.context() as m:
        m.setattr(GLM54ReducedConfig, "NUM_ROUTED_EXPERTS", 8)
        assert gw.tier1_cache_key(c) != base
    with monkeypatch.context() as m:
        edited = tmp_path / "glm_synthetic_weights.py"
        edited.write_text(Path(glm_synthetic_weights.__file__).read_text() + "\n# edited\n")
        m.setattr(glm_synthetic_weights, "__file__", str(edited))
        assert gw.tier1_cache_key(c) != base
    with monkeypatch.context() as m:

        def indexer_layer_is_reused(config, layer_idx):  # same name, other source
            return False

        m.setattr(gw, "_TIER1_GENERATOR_FUNCTIONS", (gw._generate_tier1, indexer_layer_is_reused))
        assert gw.tier1_cache_key(c) != base
    with monkeypatch.context() as m:
        m.setattr(torch, "__version__", "0.0.0")
        assert gw.tier1_cache_key(c) != base
    assert gw.tier1_cache_key(c) == base


def test_tier1_key_excluded_fields_are_not_read(monkeypatch):
    """Every hf-config field left out of the key is unread by the generator (recorded on a toy-width config)."""
    reads = set()

    class Recorder(types.SimpleNamespace):
        def __getattribute__(self, name):
            reads.add(name)
            return object.__getattribute__(self, name)

    gw._generate_tier1(Recorder(**vars(_tiny_tier1_config(monkeypatch))), GLM54_SYNTHETIC_SEED)
    assert {"hidden_size", "indexer_types", "num_hidden_layers"} <= reads  # the recorder sees the reads
    assert not reads & gw.TIER1_KEY_EXCLUDED_FIELDS, reads & gw.TIER1_KEY_EXCLUDED_FIELDS


def test_tier1_cache_round_trip_and_bounds(monkeypatch, tmp_path):
    """A miss writes one entry that a hit returns bit-identical without generating; entries beyond the cap are evicted
    least recently used first (a hit counts as use); a write that would leave the disk under the floor is skipped and
    the generated weights are still returned."""
    c = _tiny_tier1_config(monkeypatch)
    monkeypatch.setenv("TT_GLM54_CACHE_DIR", str(tmp_path))
    root = gw.tier1_cache_dir()
    expected = {seed: gw.Glm54Weights(*gw._generate_tier1(c, seed)) for seed in (1, 2, 3)}
    size = gw._tensor_bytes(gw._generate_tier1(c, 1))

    first = gw.glm54_tier1_weights(c, seed=1)
    assert _same(first, expected[1]) and [p.name for p in root.glob("*.pt")] == [f"{gw.tier1_cache_key(c, 1)}.pt"]
    assert first.provenance["cache_key"] == gw.tier1_cache_key(c, 1)
    del first  # no live object: the next request reads the disk entry
    assert gw.tier1_cache_key(c, 1) not in gw._LIVE_TIER1
    with monkeypatch.context() as m:
        m.setattr(gw, "_generate_tier1", lambda *a: pytest.fail("a hit must not generate"))
        assert _same(gw.glm54_tier1_weights(c, seed=1), expected[1])

    # Cap: two entries. Seed 2 fits; seed 1 is used again; seed 3 then evicts seed 2 (least recently used).
    monkeypatch.setenv("TT_GLM54_TIER1_CACHE_GB", str(2.5 * size / 1e9))
    t = time.time()
    os.utime(root / f"{gw.tier1_cache_key(c, 1)}.pt", (t - 30, t - 30))
    assert _same(gw.glm54_tier1_weights(c, seed=2), expected[2])
    os.utime(root / f"{gw.tier1_cache_key(c, 2)}.pt", (t - 20, t - 20))
    gw.glm54_tier1_weights(c, seed=1)  # hit: seed 1 becomes the most recently used
    assert _same(gw.glm54_tier1_weights(c, seed=3), expected[3])
    assert {p.name for p in root.glob("*.pt")} == {f"{gw.tier1_cache_key(c, s)}.pt" for s in (1, 3)}
    assert not list(root.glob(".*.tmp"))

    # Disk floor: no write, weights still returned.
    monkeypatch.setattr(gw, "TIER1_CACHE_MIN_FREE_GB", 1e9)
    assert _same(gw.glm54_tier1_weights(c, seed=2), expected[2])
    assert not (root / f"{gw.tier1_cache_key(c, 2)}.pt").exists()


def _count_generations(monkeypatch) -> list:
    calls = []
    generate = gw._generate_tier1

    def counted(config, seed):
        calls.append(seed)
        return generate(config, seed)

    monkeypatch.setattr(gw, "_generate_tier1", counted)
    return calls


def test_tier1_cache_write_failure_leaves_nothing(monkeypatch, tmp_path):
    """A write that fails partway (torch.save raising RuntimeError, as it does on a full disk) is 'not cached': the
    generated weights are returned and neither an entry nor a temporary file stays behind."""
    c = _tiny_tier1_config(monkeypatch)
    monkeypatch.setenv("TT_GLM54_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(gw, "TIER1_CACHE_MIN_FREE_GB", 0)  # the write is attempted whatever this host has free
    expected = gw.Glm54Weights(*gw._generate_tier1(c, 1))
    saves = []

    def failing_save(obj, f):
        saves.append(f)
        Path(f).write_bytes(b"partial" * 1000)
        raise RuntimeError("[enforce fail at inline_container.cc] PytorchStreamWriter failed writing file")

    monkeypatch.setattr(torch, "save", failing_save)
    assert _same(gw.glm54_tier1_weights(c, seed=1), expected)
    assert len(saves) == 1, "the write was not attempted"
    assert list(gw.tier1_cache_dir().iterdir()) == []


def test_tier1_one_object_per_key_while_held(monkeypatch, tmp_path):
    """While a caller holds the weights, the same key returns that object: no second generation when the entry could
    not be written (as test_glm54_loop_on_device's inline check asks again), no second mapping on a hit. Once
    released, the next request generates or reads the disk again."""
    c = _tiny_tier1_config(monkeypatch)
    monkeypatch.setenv("TT_GLM54_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(gw, "TIER1_CACHE_MIN_FREE_GB", 1e9)  # nothing is written
    calls = _count_generations(monkeypatch)
    w = gw.glm54_tier1_weights(c, seed=1)
    assert gw.glm54_tier1_weights(c, seed=1) is w and calls == [1]
    assert gw.glm54_tier1_weights(c, seed=2) is not w and calls == [1, 2]
    del w
    gw.glm54_tier1_weights(c, seed=1)
    assert calls == [1, 2, 1]


def test_tier1_cache_tolerates_concurrent_eviction(monkeypatch, tmp_path):
    """An entry evicted by another process between lookup and load is a miss, not an error; eviction skips entries
    that vanish while it scans."""
    c = _tiny_tier1_config(monkeypatch)
    monkeypatch.setenv("TT_GLM54_CACHE_DIR", str(tmp_path))
    assert gw._load(tmp_path / "absent.pt") is None
    w = gw.glm54_tier1_weights(c, seed=1)
    path = gw.tier1_cache_dir() / f"{gw.tier1_cache_key(c, 1)}.pt"
    del w
    real_load = torch.load

    def load_after_eviction(f, **kw):  # the other process wins the race
        Path(f).unlink()
        return real_load(f, **kw)

    calls = _count_generations(monkeypatch)
    with monkeypatch.context() as m:
        m.setattr(torch, "load", load_after_eviction)
        m.setattr(gw, "_store", lambda *a: False)
        gw.glm54_tier1_weights(c, seed=1)
    assert calls == [1] and not path.exists()

    real_stat = Path.stat
    (gw.tier1_cache_dir() / "gone.pt").write_bytes(b"x")

    def stat_vanishing(self, *a, **kw):
        if self.name == "gone.pt":
            raise FileNotFoundError(self)
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", stat_vanishing)
    gw._evict(gw.tier1_cache_dir(), 0)


def test_tier2_provenance_follows_the_runner_environment(monkeypatch, tmp_path):
    """Provenance records the checkpoint and cache a run actually reads: the runner's env overrides, in its order
    ($PREFILL_HF_MODEL before $GLM53_HF_MODEL), and as many gates as the config's loop passes."""

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
    assert tier2_sources()[0] == prefill  # $PREFILL_HF_MODEL wins, as in the runner's hf_model_path
    monkeypatch.delenv("PREFILL_HF_MODEL")
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
