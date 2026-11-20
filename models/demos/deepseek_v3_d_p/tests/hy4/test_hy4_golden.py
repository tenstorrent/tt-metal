# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the Hy4-preview golden-trace tooling (#65919), on a random tiny checkpoint.

* the streaming reference matches HF ``HYV4ForCausalLM`` (skipped unless transformers ships ``hy_v4``);
* it is blocking-invariant and causal (a prefix's golden is the long run's first rows);
* a written trace is read back by the prefill producer's own golden loaders, and a simulated device
  cache in the device frame scores PCC 1.0 through the producer's compare path;
* the device index-key frame is exactly what a weight-permuted indexer emits;
* MXFP8 dequantization, the top-k agreement metric and the comparison CLI.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.hy4 import compare as hy4_compare
from models.demos.deepseek_v3_d_p.reference.hy4.checkpoint import (
    Hy4Checkpoint,
    dequant_mxfp8,
    random_checkpoint_tensors,
    save_random_checkpoint,
)
from models.demos.deepseek_v3_d_p.reference.hy4.config import Hy4RefConfig, tiny_random_config
from models.demos.deepseek_v3_d_p.reference.hy4.golden import (
    GoldenTraceWriter,
    half_split_to_interleaved_perm,
    index_k_device_perm,
    index_k_device_to_hf,
    index_k_hf_to_device,
    topk_agreement,
)
from models.demos.deepseek_v3_d_p.reference.hy4.hf_check import _Capture, check, hf_hy_v4_available
from models.demos.deepseek_v3_d_p.reference.hy4.model import Hy4PrefillReference, rope_cos_sin
from models.demos.deepseek_v3_d_p.reference.hy4.paths import HY4_HF_MODEL_CANDIDATES

SEQ = 72  # > tiny index_topk (16): the sparse selection drops positions


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("hy4_tiny")
    save_random_checkpoint(d, tiny_random_config(), seed=0)
    return Hy4Checkpoint(d)


@pytest.fixture(scope="module")
def token_ids(tiny_ckpt):
    return torch.randint(3, tiny_ckpt.config.vocab_size, (SEQ,), generator=torch.Generator().manual_seed(1))


def _run(ckpt, ids, **kw):
    cap = _Capture()
    Hy4PrefillReference(ckpt, prefetch=False, **kw).run(ids, cap)
    return cap


@pytest.fixture(scope="module")
def tiny_capture(tiny_ckpt, token_ids):
    return _run(tiny_ckpt, token_ids)


# ------------------------------------------------------------------------------- reference fidelity
@pytest.mark.skipif(not hf_hy_v4_available(), reason="transformers in this env has no hy_v4 (needs >= 5.23)")
def test_reference_matches_hf_hy_v4(tiny_ckpt):
    assert check(tiny_ckpt.model_dir, SEQ, None) == []


def test_reference_is_block_size_invariant(tiny_ckpt, token_ids, tiny_capture):
    other = _run(tiny_ckpt, token_ids, attn_block=5, proj_block=7, index_budget_bytes=1)
    for layer, ref in tiny_capture.layers.items():
        got = other.layers[layer]
        torch.testing.assert_close(got["kvpe"], ref["kvpe"], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(got["streams"], ref["streams"], rtol=1e-4, atol=1e-4)
        if ref["topk"] is not None:
            assert torch.equal(got["topk"], ref["topk"]), f"layer {layer} top-k depends on blocking"
    torch.testing.assert_close(other.logits, tiny_capture.logits, rtol=1e-4, atol=1e-4)


def test_prefix_golden_equals_long_run_rows(tiny_ckpt, token_ids, tiny_capture):
    """Causality: the n-token prompt's KV / index keys / top-k are the long run's first n rows, which is
    what lets one 128K pass emit the 4K and 32K traces."""
    n = 40
    short = _run(tiny_ckpt, token_ids[:n])
    for layer, ref in tiny_capture.layers.items():
        got = short.layers[layer]
        torch.testing.assert_close(got["kvpe"], ref["kvpe"][:n], rtol=1e-5, atol=1e-5)
        if ref["index_k"] is not None:
            torch.testing.assert_close(got["index_k"], ref["index_k"][:n], rtol=1e-5, atol=1e-5)
            assert topk_agreement(ref["topk"][:n], got["topk"]).min() == 1.0


def test_sparse_selection_is_active(tiny_capture, tiny_ckpt):
    """Late rows select exactly index_topk positions out of more than that many causal keys."""
    k = tiny_ckpt.config.index_topk
    tk = tiny_capture.layers[0]["topk"]
    assert tk.shape == (SEQ, k)
    assert (tk[-1] >= 0).sum() == k and SEQ > k
    assert (tk[3] >= 0).sum() == 4 and (tk[3][4:] == -1).all()  # row 3 has 4 causal keys


# ------------------------------------------------------------------------- trace <-> producer loaders
@pytest.fixture(scope="module")
def tiny_trace(tmp_path_factory, tiny_ckpt, token_ids, tiny_capture):
    out = tmp_path_factory.mktemp("trace")
    w = GoldenTraceWriter(out, tiny_ckpt.config, token_ids, rows_per_shard=32)
    for layer in sorted(tiny_capture.layers):
        c = tiny_capture.layers[layer]
        w.on_layer(layer, kvpe=c["kvpe"], index_k=c["index_k"], topk=c["topk"])
    w.on_final(logits=tiny_capture.logits, position=SEQ - 1)
    w.finalize()
    return out


def test_trace_metadata(tiny_trace, tiny_ckpt, token_ids):
    meta = json.loads((tiny_trace / "metadata.json").read_text())
    assert meta["complete"] and meta["token_ids"] == token_ids.tolist()
    assert meta["full_indexer_layers"] == tiny_ckpt.config.full_indexer_layers == [0, 1, 5]
    assert meta["frames"]["index_k"]["frame"] == "device"
    # Hy4's checkpoint pairs RoPE interleaved; the golden records it so a stale (half-split) trace is recognisable.
    assert meta["frames"]["kvpe"]["pe_rope"] == "interleaved" and meta["frames"]["index_k"]["rope"] == "interleaved"


def test_producer_loaders_read_the_trace(tiny_trace, tiny_ckpt, tiny_capture, token_ids):
    from models.demos.common.prefill.runners.runner_utils import load_trace_golden_span, load_trace_token_ids
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import (
        _load_golden_index_k,
        _load_golden_kv_post,
        index_golden_present,
        kvpe_golden_present,
    )

    cfg = tiny_ckpt.config
    assert load_trace_golden_span(tiny_trace) == (0, SEQ)
    assert load_trace_token_ids(tiny_trace) == token_ids.tolist()
    for layer, c in tiny_capture.layers.items():
        assert kvpe_golden_present(tiny_trace, layer)
        # windowed read that crosses the 32-row shard boundary
        torch.testing.assert_close(_load_golden_kv_post(tiny_trace, layer, 30, start=20), c["kvpe"][20:50])
        assert index_golden_present(tiny_trace, layer) == cfg.is_full_indexer(layer)
        if cfg.is_full_indexer(layer):
            got = _load_golden_index_k(tiny_trace, layer, SEQ)
            torch.testing.assert_close(got, index_k_hf_to_device(c["index_k"], cfg.qk_rope_head_dim))


def test_producer_compare_path_scores_a_device_frame_cache_at_one(tiny_trace, tiny_ckpt, tiny_capture):
    """Feed the producer's own comparison a cache laid out the way the device writes it: KVPE with an
    interleaved pe half (the reference's own: Hy4's checkpoint pairing, so no re-base,
    ``Hy4Adapter.kv_pe_interleave = False``); index keys in the device frame, Hadamard-rotated (the producer
    de-rotates)."""
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.hy4_preview import Hy4Adapter
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_index_k, _load_golden_kv_post
    from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs
    from tt_py_test_utils_common.utils_for_testing import comp_pcc

    cfg = tiny_ckpt.config
    L = cfg.kv_lora_rank
    assert Hy4Adapter.kv_pe_interleave is False
    for layer, c in tiny_capture.layers.items():
        device_kvpe = c["kvpe"]
        nope, pe = cache_half_pccs(
            _load_golden_kv_post(tiny_trace, layer, SEQ), device_kvpe, L, pe_interleave=Hy4Adapter.kv_pe_interleave
        )
        assert nope > 0.99999 and pe > 0.99999, (layer, nope, pe)
        if cfg.is_full_indexer(layer):
            had = normalized_hadamard_matrix(cfg.index_head_dim).float()
            device_ik = index_k_hf_to_device(c["index_k"], cfg.qk_rope_head_dim) @ had  # stored rotated
            dev_ik = (device_ik @ had).to(torch.bfloat16)  # what the producer does on read-back
            _, p = comp_pcc(_load_golden_index_k(tiny_trace, layer, SEQ), dev_ik)
            assert p > 0.999, (layer, p)


# ---------------------------------------------------------------------------------- key frames
def test_frame_permutation_matches_repo_conventions():
    from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_perm_matrix
    from models.demos.deepseek_v3_d_p.utils.test_utils import interleave_pe

    x = torch.randn(5, 64)
    perm = half_split_to_interleaved_perm(64)
    torch.testing.assert_close(x[:, perm], interleave_pe(x))
    torch.testing.assert_close(x[:, perm], x @ interleaved_perm_matrix(64))
    k = torch.randn(5, 128)
    torch.testing.assert_close(index_k_device_to_hf(index_k_hf_to_device(k, 64), 64), k)
    # Interleaved (current): the device frame only moves the rope block to the front. half_split: the frame of traces
    # written before the fix, still readable.
    assert index_k_device_perm(128, 64).tolist() == list(range(64, 128)) + list(range(64))
    assert index_k_device_perm(8, 4, "half_split").tolist() == [4, 6, 5, 7, 0, 1, 2, 3]
    torch.testing.assert_close(index_k_device_to_hf(index_k_hf_to_device(k, 64, "half_split"), 64, "half_split"), k)
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc

    assert torch.equal(index_k_device_perm(128, 64), hc.index_k_device_perm(128, 64))  # the reader's frame


def _rope_interleaved(x, cos, sin):
    a, b = x[..., 0::2], x[..., 1::2]
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    c, s = cos.view(shape), sin.view(shape)
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


def test_device_index_frame_is_a_weight_permutation(tiny_ckpt, tiny_capture, token_ids):
    """A device indexer that ropes the FIRST R dims interleaved, fed wk / k_norm / wq_b permuted by
    ``index_k_device_perm``, writes exactly ``index_k_hf_to_device(key)`` of the checkpoint's math (rope on the LAST
    R dims, interleaved) and scores identically."""
    cfg = tiny_ckpt.config
    R, DI, NH = cfg.qk_rope_head_dim, cfg.index_head_dim, cfg.index_n_heads
    t = random_checkpoint_tensors(cfg, seed=0)
    x = torch.randn(SEQ, cfg.hidden_size, generator=torch.Generator().manual_seed(3))
    qr = torch.randn(SEQ, cfg.q_lora_rank, generator=torch.Generator().manual_seed(4))
    cos, sin = rope_cos_sin(torch.arange(SEQ), R, cfg.rope_theta)
    a = "model.layers.0.self_attn.indexer."
    wk, wq = t[a + "wk.weight"], t[a + "wq_b.weight"]
    nw, nb = t[a + "k_norm.weight"], t[a + "k_norm.bias"]

    from models.demos.deepseek_v3_d_p.reference.hy4.model import apply_rope_interleaved

    def hf(v):  # checkpoint frame: rope the last R dims, interleaved
        return torch.cat([v[..., : DI - R], apply_rope_interleaved(v[..., DI - R :], cos, sin)], dim=-1)

    k_hf = hf(F.layer_norm(x @ wk.T, (DI,), nw, nb, cfg.rms_norm_eps))
    q_hf = hf((qr @ wq.T).view(SEQ, NH, DI))

    P = index_k_device_perm(DI, R)

    def dev(v):  # device frame: rope the first R dims, interleaved
        return torch.cat([_rope_interleaved(v[..., :R], cos, sin), v[..., R:]], dim=-1)

    k_dev = dev(F.layer_norm(x @ wk[P].T, (DI,), nw[P], nb[P], cfg.rms_norm_eps))
    q_dev = dev((qr @ wq.view(NH, DI, -1)[:, P].reshape(NH * DI, -1).T).view(SEQ, NH, DI))

    torch.testing.assert_close(k_dev, index_k_hf_to_device(k_hf, R), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        torch.einsum("shd,td->sht", q_dev, k_dev), torch.einsum("shd,td->sht", q_hf, k_hf), rtol=1e-4, atol=1e-4
    )


# ------------------------------------------------------------------------------------- MXFP8, metrics
def _quantize_mxfp8(w: torch.Tensor, block: int = 32):
    blocks = w.float().reshape(*w.shape[:-1], -1, block)
    amax = blocks.abs().amax(-1).clamp(min=2.0**-120)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(e).unsqueeze(-1)).reshape(w.shape).to(torch.float8_e4m3fn)
    return q, (e + 127).to(torch.uint8)


def test_mxfp8_checkpoint_dequant(tmp_path):
    from safetensors.torch import save_file

    cfg = tiny_random_config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **cfg.to_hf_dict(),
                "quantization_config": {"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP8"}},
            }
        )
    )
    w = torch.randn(64, 96)
    ex = torch.randn(4, 32, 64)
    qw, sw = _quantize_mxfp8(w)
    qe, se = _quantize_mxfp8(ex)
    save_file(
        {
            "a.weight": qw,
            "a.weight_scale": sw,
            "model.layers.1.mlp.experts.gate_up_proj": qe,
            "model.layers.1.mlp.experts.gate_up_proj_scale": se,
            "b.weight": torch.randn(3, 3, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    ck = Hy4Checkpoint(tmp_path)
    assert ck.precision == "mxfp8"
    got = ck.get("a.weight", torch.float32)
    torch.testing.assert_close(got, dequant_mxfp8(qw, sw).float())
    assert (got - w).abs().max() <= w.abs().max() * 2**-3  # e4m3: 3 mantissa bits
    assert ck.get("model.layers.1.mlp.experts.gate_up_proj").dtype == torch.bfloat16
    # bf16 dequant is exact: an e4m3 value times a power of two fits bf16
    exact = qe.float().reshape(4, 32, 2, 32) * torch.exp2(se.float() - 127).unsqueeze(-1)
    assert torch.equal(ck.get("model.layers.1.mlp.experts.gate_up_proj").float(), exact.reshape(4, 32, 64))
    assert ck.get("b.weight").dtype == torch.bfloat16


def test_topk_agreement_metric():
    ref = torch.tensor([[0, 1, 2, -1], [3, 2, 1, 0], [-1, -1, -1, -1]])
    other = torch.tensor([[2, 1, 0, -1], [3, 9, 8, -1], [5, -1, -1, -1]])
    torch.testing.assert_close(topk_agreement(ref, other), torch.tensor([1.0, 0.25, 1.0], dtype=torch.float64))
    narrow = torch.tensor([[1, 0], [3, 2], [-1, -1]])  # HF returns min(topk, S) columns
    torch.testing.assert_close(topk_agreement(narrow, ref), torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))


# ----------------------------------------------------------------------------------- comparison CLI
def test_compare_traces_and_log(tmp_path, tiny_trace, tiny_ckpt, tiny_capture, token_ids):
    # a perturbed copy (e.g. a lower-precision checkpoint): PCC < 1 but recorded per layer
    noisy = tmp_path / "noisy"
    w = GoldenTraceWriter(noisy, tiny_ckpt.config, token_ids, index_k_frame="hf")
    g = torch.Generator().manual_seed(9)
    for layer in sorted(tiny_capture.layers):
        c = tiny_capture.layers[layer]
        w.on_layer(
            layer,
            kvpe=c["kvpe"] + 0.05 * (layer + 1) * torch.randn(c["kvpe"].shape, generator=g),
            index_k=c["index_k"],
            topk=c["topk"],
        )
    w.finalize()
    rows = hy4_compare.compare_traces(tiny_trace, noisy)
    assert [r["layer"] for r in rows] == list(range(6))
    assert rows[0]["kvpe_nope"] > rows[-1]["kvpe_nope"]  # drift with depth is visible
    full = [r for r in rows if "index_k" in r]
    assert [r["layer"] for r in full] == [0, 1, 5]
    assert all(r["index_k"] > 0.99999 and r["topk_agree_min"] == 1.0 for r in full)  # frames reconciled

    log = tmp_path / "producer.log"
    log.write_text(
        "[producer] slot 0 layer  0 KV PCC: nope=0.99950 pe=0.99900\n"
        "[producer] slot 0 layer 77 KV PCC: nope=0.98500 pe=0.99100\n"
        "[producer] slot 0 layer  5 index PCC: 0.99700\n"
    )
    assert hy4_compare.main(["log", str(log), "--out", str(tmp_path / "rep")]) == 1  # layer 77 < 0.99
    rep = json.loads((tmp_path / "rep.json").read_text())
    assert rep["summary"]["min_kvpe_nope"] == {"value": 0.985, "layer": 77}
    assert (tmp_path / "rep.csv").read_text().startswith("layer,")


def test_compare_dump_follows_the_golden_rope_pairing(
    tmp_path, tiny_trace, tiny_ckpt, tiny_capture, token_ids, expect_error
):
    """A device KV dump (interleaved pe) against a current golden compares directly; against a trace written before the
    interleaved fix (``pe_rope: half_split``, pe stored half-split) the golden's pe is re-based first."""
    L = tiny_ckpt.config.kv_lora_rank
    dump = tmp_path / "dump"
    dump.mkdir()
    for layer in (0, 1):
        torch.save(tiny_capture.layers[layer]["kvpe"].clone(), dump / f"device_kv_layer{layer}_slot0.pt")
    rows = hy4_compare.compare_dump(tiny_trace, dump)
    assert all(r["kvpe_pe"] > 0.99999 and r["kvpe_nope"] > 0.99999 for r in rows), rows

    old = tmp_path / "old"
    w = GoldenTraceWriter(old, tiny_ckpt.config, token_ids, rope_pairing="half_split")
    to_half = torch.argsort(half_split_to_interleaved_perm(tiny_ckpt.config.qk_rope_head_dim))
    for layer in (0, 1):
        kv = tiny_capture.layers[layer]["kvpe"]
        w.on_layer(layer, kvpe=torch.cat([kv[:, :L], kv[:, L:][:, to_half]], dim=-1))
    w.finalize()
    rows = hy4_compare.compare_dump(old, dump)
    assert all(r["kvpe_pe"] > 0.99999 for r in rows), rows
    with expect_error(SystemExit, "different pairings"):
        hy4_compare.compare_traces(tiny_trace, old)


# ------------------------------------------------------------------------------------- real config
def _real_config_dir():
    for c in HY4_HF_MODEL_CANDIDATES:
        if (Path(c) / "config.json").exists():
            return Path(c)
    return None


@pytest.mark.skipif(_real_config_dir() is None, reason="Hy4-preview checkpoint not staged")
def test_real_config_layer_maps():
    cfg = Hy4RefConfig.from_pretrained(_real_config_dir())
    full = cfg.full_indexer_layers
    assert len(full) == 21 and full[:3] == [0, 1, 5] and full[-1] == 77
    assert full == [0] + list(range(1, 78, 4))
    assert [i for i in range(78) if cfg.is_dense(i)] == [0]
    assert (cfg.vocab_size, cfg.routed_scaling_factor, cfg.swiglu_limit, cfg.rope_theta) == (120832, 2.827, 10.0, 1e7)
    assert cfg.kvpe_dim == 576 and cfg.index_head_dim == 128 and cfg.index_topk == 2048
    assert (cfg.rms_norm_eps, cfg.low_rank_norm_eps) == (1e-5, 1e-6)


# ------------------------------------------------------------------------------------- norm epsilons
def test_low_rank_norm_eps_is_hf_rmsnorm_default():
    """HF builds q_a_layernorm / kv_a_layernorm as ``HYV4RMSNorm(dim)`` -- the class default eps (1e-6),
    not ``config.rms_norm_eps`` (1e-5), which only the decoder/final/iHC norms and the indexer k_norm take."""
    assert Hy4RefConfig().low_rank_norm_eps == 1e-6
    assert "low_rank_norm_eps" not in tiny_random_config().to_hf_dict()  # HF hard-codes it, never reads it
    if hf_hy_v4_available():
        import inspect

        from transformers.models.hy_v4 import modeling_hy_v4 as hf

        default = inspect.signature(hf.HYV4RMSNorm.__init__).parameters["eps"].default
        assert default == Hy4RefConfig().low_rank_norm_eps
        cfg = tiny_random_config()
        attn = hf.HYV4Attention(hf.HYV4Config(**cfg.to_hf_dict()), layer_idx=0)
        assert attn.q_a_layernorm.variance_epsilon == cfg.low_rank_norm_eps
        assert attn.kv_a_layernorm.variance_epsilon == cfg.low_rank_norm_eps


def test_reference_norm_eps_wiring(tiny_ckpt, token_ids, monkeypatch):
    """Every RMSNorm the reference applies uses the eps HF uses for that norm (distinct eps values here so
    a swapped argument cannot pass)."""
    import dataclasses

    from models.demos.deepseek_v3_d_p.reference.hy4 import model as ref_model

    cfg = dataclasses.replace(tiny_ckpt.config, rms_norm_eps=3e-5, low_rank_norm_eps=7e-7).truncated(2)
    w0 = ref_model.load_layer_weights(tiny_ckpt, cfg, 0)
    by_width = {
        cfg.q_lora_rank: ("q_a_layernorm", w0["q_a_layernorm"]),
        cfg.kv_lora_rank: ("kv_a_layernorm", w0["kv_a_layernorm"]),
    }
    seen = {}
    real = ref_model.rms_norm

    def spy(x, weight, eps):
        name = by_width.get(x.shape[-1], ("hidden",))[0]
        seen.setdefault(name, set()).add(eps)
        return real(x, weight, eps)

    monkeypatch.setattr(ref_model, "rms_norm", spy)
    Hy4PrefillReference(tiny_ckpt, cfg, prefetch=False).run(token_ids, _Capture(), num_layers=2)
    assert seen == {"q_a_layernorm": {7e-7}, "kv_a_layernorm": {7e-7}, "hidden": {3e-5}}, seen
