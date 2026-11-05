# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only oracles for the sparse-MLA CPU reference (no device).

The device tests trust ``SparseMLAReference`` as ground truth, so it is pinned here against
independent implementations:

  1. MLA / gate oracle -- at ``seq <= index_topk`` the indexer keeps every causal key, so sparse MLA is
     dense MLA. For Kimi-K3.1 the sparse reference must then match the vendored upstream K3 attention
     (``KimiMLAAttention``: NoPE + output gate), which shares no code with ``MLACPU``.
  2. Pruning oracle -- at ``seq > index_topk`` (index_topk shrunk so it stays cheap on CPU) the
     reference's selection must be the top-k of a brute-force lightning-indexer score written straight
     from the formula, and the sparse output must equal dense attention masked to exactly those keys.
     Oracle 1 cannot see a selection bug; this one can.
  3. Key-set regression -- ungated variants (DS-V3.2, GLM) must not start requiring ``g_proj``.
"""

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference, random_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiMLAAttention
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import kimi_k3_1_hf_config
from tests.ttnn.utils_for_testing import comp_pcc

ORACLE_PCC = 0.9999


def _k31_config(seq_len, topk=None):
    cfg = kimi_k3_1_hf_config(max_seq=seq_len)
    if topk is not None:
        cfg = copy.copy(cfg)
        cfg.index_topk = topk
    return cfg


def _hidden(seq_len, hidden, seed=7):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hidden, dtype=torch.bfloat16)


def _upstream_k3(cfg, weights, mask=None):
    """Vendored upstream K3 attention loaded with the same canonical weights (HF names)."""
    attn = KimiMLAAttention(cfg, layer_idx=0)
    sd = {k: v for k, v in weights.items() if not k.startswith("indexer.")}
    missing, unexpected = attn.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, strict=False)
    assert not unexpected, unexpected
    assert not [m for m in missing if m != "rotary_emb"], f"upstream K3 attention left weights at init: {missing}"
    return attn.eval().to(torch.bfloat16)


def _causal(seq_len):
    return torch.full((seq_len, seq_len), float("-inf")).triu_(1)


def test_kimi_k31_reference_matches_upstream_k3_when_dense():
    """seq <= index_topk: sparse reference == upstream K3 gated NoPE MLA."""
    seq_len = 256
    cfg = _k31_config(seq_len)
    assert seq_len <= cfg.index_topk
    weights = random_mla_weights(cfg, seed=11)
    assert "g_proj.weight" in weights
    x = _hidden(seq_len, cfg.hidden_size)

    ref = SparseMLAReference(cfg, weights, seq_len=seq_len)
    out = ref.forward(x)
    assert ref.indexer_topk.shape[-1] == seq_len, "inert top-k must select every causal key"

    with torch.no_grad():
        up = _upstream_k3(cfg, weights)(x, attention_mask=_causal(seq_len)[None, None].to(torch.bfloat16))
    ok, pcc = comp_pcc(up.float(), out.float(), ORACLE_PCC)
    assert ok, f"K3.1 sparse reference vs upstream K3 (dense regime): {pcc}"

    # The gate is live: dropping it must move the output well below the bar.
    ungated = copy.copy(cfg)
    ungated.mla_use_output_gate = False
    no_gate = SparseMLAReference(ungated, weights, seq_len=seq_len).forward(x)
    ok_ng, _ = comp_pcc(up.float(), no_gate.float(), ORACLE_PCC)
    assert not ok_ng, "output gate had no effect -- the oracle would not catch a missing gate"


def _brute_force_index_scores(cfg, weights, x):
    """Lightning-indexer score from the formula (fp32, NoPE): sum_h relu(q_h . k_t) * w_h * H^-0.5 * D^-0.5."""
    xf = x.float()[0]
    qr = F.linear(xf, weights["q_a_proj.weight"].float())
    qr = qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + 1e-6) * weights["q_a_layernorm.weight"].float()
    qr = qr.to(torch.bfloat16).float()
    h, d = cfg.index_n_heads, cfg.index_head_dim
    q = F.linear(qr, weights["indexer.wq_b.weight"].float()).view(-1, h, d)  # [S, H, D]
    k = F.linear(xf, weights["indexer.wk.weight"].float())  # [S, D]
    k = F.layer_norm(
        k, (d,), weights["indexer.k_norm.weight"].float(), weights["indexer.k_norm_bias.weight"].float(), 1e-6
    )
    w = F.linear(xf, weights["indexer.weights_proj.weight"].float()) * (h**-0.5) * (d**-0.5)  # [S, H]
    scores = torch.einsum("shd,td->sht", q, k).relu()
    return torch.einsum("sht,sh->st", scores, w) + _causal(x.shape[1])  # [S, T]


def test_kimi_k31_reference_pruning_matches_brute_force_and_masked_dense():
    """seq > index_topk: selection == brute-force top-k, and sparse == dense masked to that selection."""
    seq_len, topk = 256, 48
    cfg = _k31_config(seq_len, topk=topk)
    weights = random_mla_weights(cfg, seed=13)
    x = _hidden(seq_len, cfg.hidden_size, seed=3)

    ref = SparseMLAReference(cfg, weights, seq_len=seq_len)
    out = ref.forward(x)
    sel = ref.indexer_topk[0]  # [S, topk]
    assert sel.shape == (seq_len, topk)
    assert (sel >= 0).all() and (sel < seq_len).all(), "reference selected an out-of-range / sentinel key"

    # Selection: the reference keeps the brute-force top-k, up to near-ties at the k-th score (bf16 in the
    # reference vs fp32 here). Tie-tolerant: every kept key scores >= the best dropped causal key - tol.
    brute = _brute_force_index_scores(cfg, weights, x)
    rows = torch.arange(topk, seq_len)  # rows with more causal keys than topk actually prune
    for s in rows.tolist():
        kept = torch.zeros(seq_len, dtype=torch.bool)
        kept[sel[s]] = True
        causal = torch.arange(seq_len) <= s
        assert not kept[~causal].any(), f"row {s}: a future key was selected"
        kept_min = brute[s][kept].min()
        dropped = brute[s][causal & ~kept]
        tol = 1e-2 * brute[s][causal].abs().max()
        assert dropped.numel() == 0 or dropped.max() <= kept_min + tol, f"row {s}: selection is not the top-{topk}"
    assert rows.numel() > 0, "pruning regime was not exercised"

    # Sparse == upstream dense attention masked to exactly the reference's selection.
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask.scatter_(-1, sel, 0.0)
    mask = mask + _causal(seq_len)
    with torch.no_grad():
        up = _upstream_k3(cfg, weights)(x, attention_mask=mask[None, None].to(torch.bfloat16))
    ok, pcc = comp_pcc(up.float(), out.float(), ORACLE_PCC)
    assert ok, f"K3.1 sparse reference vs masked-dense upstream K3: {pcc}"


def test_ungated_reference_does_not_require_g_proj():
    """Config-dependent key set: GLM (ungated) builds without g_proj, and its random weights carry none."""
    cfg = glm_5_3_hf_config(max_seq=128)
    weights = random_mla_weights(cfg, seed=1)
    assert "g_proj.weight" not in weights
    SparseMLAReference(cfg, weights, seq_len=128)  # must not raise


def test_gated_reference_requires_g_proj(expect_error):
    cfg = _k31_config(128)
    weights = random_mla_weights(cfg, seed=1)
    weights.pop("g_proj.weight")
    with expect_error(KeyError, "g_proj"):
        SparseMLAReference(cfg, weights, seq_len=128)


def test_reference_cache_key_tracks_semantics():
    """The on-disk reference cache key changes with any field the reference reads (K3.1 vs K3-style NoPE
    toggles, top-k), and is stable for an unchanged config."""
    from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import reference_cache_key

    base = _k31_config(256)
    assert reference_cache_key(base) == reference_cache_key(_k31_config(256))
    for field, value in (("mla_use_output_gate", False), ("index_topk", 1024), ("index_rope_dim", None)):
        other = copy.copy(base)
        setattr(other, field, value)
        assert reference_cache_key(other) != reference_cache_key(base), field
    assert reference_cache_key(glm_5_3_hf_config()) != reference_cache_key(base)


def test_pretrained_loader_reads_multimodal_key_root(tmp_path):
    """Kimi-K3 / K3.1 checkpoints nest the LM under ``language_model.``; the loader must map that root (incl. the
    gate and the indexer LayerNorm bias) to the canonical names, exactly as it maps ``model.layers.N``."""
    from safetensors.torch import save_file

    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import pretrained_mla_weights

    cfg = _k31_config(128)
    canon = random_mla_weights(cfg, seed=5)
    to_hf = {"indexer.k_norm_bias.weight": "indexer.k_norm.bias"}
    shard = {
        f"language_model.model.layers.3.self_attn.{to_hf.get(k, k)}": v.to(torch.bfloat16).contiguous()
        for k, v in canon.items()
    }
    shard["language_model.model.layers.4.self_attn.q_a_proj.weight"] = torch.zeros(1, 1, dtype=torch.bfloat16)
    path = tmp_path / "shard.safetensors"
    save_file(shard, str(path))
    loaded = pretrained_mla_weights(cfg, layer=3, checkpoint_path=str(path))
    assert set(loaded) == set(canon), sorted(set(loaded) ^ set(canon))
    for k, v in canon.items():
        assert torch.equal(loaded[k].to(torch.bfloat16), v.to(torch.bfloat16)), k
    SparseMLAReference(cfg, loaded, seq_len=128)  # gated + indexer key set is complete


def test_kimi_k31_pretrained_layer_matches_upstream_k3_when_dense():
    """Real Kimi-K3.1 layer 3 (first DSA layer) weights: dense-regime sparse reference == upstream K3 attention.
    Uses the cached range-read extract; skips when it is not present locally (fetch it with
    reference.cpu_deepseek_v32.weights.extract_layer_attention("moonshotai/Kimi-K3.1-Pro", 3))."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import pretrained_mla_weights

    seq_len = 256
    cfg = _k31_config(seq_len)
    try:
        weights = pretrained_mla_weights(cfg, layer=3, checkpoint_path=_cached_k31_layer3())
    except LocalEntryNotFoundError as e:
        import pytest

        pytest.skip(str(e))
    x = _hidden(seq_len, cfg.hidden_size, seed=21)
    out = SparseMLAReference(cfg, weights, seq_len=seq_len).forward(x)
    with torch.no_grad():
        up = _upstream_k3(cfg, weights)(x, attention_mask=_causal(seq_len)[None, None].to(torch.bfloat16))
    ok, pcc = comp_pcc(up.float(), out.float(), ORACLE_PCC)
    assert ok, f"K3.1 layer-3 sparse reference vs upstream K3 (dense regime): {pcc}"


def _cached_k31_layer3():
    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import extract_layer_attention

    return extract_layer_attention("moonshotai/Kimi-K3.1-Pro", 3, local_files_only=True)


def test_extract_layer_attention_range_reads_only_the_layer(tmp_path, monkeypatch, expect_error):
    """Offline: the range-read extractor parses the safetensors header, pulls exactly the mapped layer tensors
    (dropping unmapped keys and other layers), caches per revision, and rejects a layer with no MLA tensors."""
    import json

    import fsspec
    import huggingface_hub
    import huggingface_hub.constants  # explicit: patched below, independent of setattr order
    from safetensors.torch import save_file

    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import weights as W

    cfg = _k31_config(128)
    canon = random_mla_weights(cfg, seed=9)
    to_hf = {"indexer.k_norm_bias.weight": "indexer.k_norm.bias"}
    root = "language_model.model.layers"
    shard_a = {f"{root}.3.self_attn.{to_hf.get(k, k)}": v.to(torch.bfloat16).contiguous() for k, v in canon.items()}
    shard_a[f"{root}.3.self_attn.unmapped_buffer"] = torch.zeros(4, dtype=torch.int64)  # unsupported dtype, unused
    shard_b = {f"{root}.2.self_attn.o_proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}  # a KDA-like layer
    snap = tmp_path / "snapshots" / "rev123"
    snap.mkdir(parents=True)
    save_file(shard_a, str(tmp_path / "a.safetensors"))
    save_file(shard_b, str(tmp_path / "b.safetensors"))
    weight_map = {k: str(tmp_path / "a.safetensors") for k in shard_a} | {
        k: str(tmp_path / "b.safetensors") for k in shard_b
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(snap / "model.safetensors.index.json"))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "cache"))
    local = fsspec.filesystem("file")

    import os

    path = W.extract_layer_attention("org/model", 3, fs=local)
    assert "rev123" in path
    assert [p.name for p in Path(path).parent.iterdir()] == [Path(path).name], "temp file left behind"
    # Same creation mode as any 0o666 file in that directory (umask and default ACLs applied), not mkstemp's 0600.
    probe = Path(path).parent / "mode_probe"
    os.close(os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
    expected_mode = os.stat(probe).st_mode & 0o777
    probe.unlink()
    assert os.stat(path).st_mode & 0o777 == expected_mode, "extract must keep the shareable creation mode"
    from safetensors.torch import load_file

    raw = load_file(path)
    # Exactly the mapped layer-3 tensors: no unmapped key, no other layer, byte-exact contents.
    assert set(raw) == {k for k in shard_a if not k.endswith("unmapped_buffer")}, sorted(set(raw) ^ set(shard_a))
    for k, v in raw.items():
        assert torch.equal(v, shard_a[k]), k
    loaded = W.load_attention_state_dict(path, 3)
    assert len(loaded) == len(canon)
    assert W.extract_layer_attention("org/model", 3, local_files_only=True, fs=local) == path  # cache hit
    with expect_error(ValueError, "no MLA attention tensors"):
        W.extract_layer_attention("org/model", 2, fs=local)
