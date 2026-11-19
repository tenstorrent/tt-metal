# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the embedding / final norm / LM head reference must track HF transformers' hy_v4.

reference/lm_head.py is what the device tests validate against; this pins it to HF. A small
HYV4ForCausalLM is run whole and its two ends are compared with the reference:

    the input of decoder layer 0   ==  ihc_expand(Hy4Embedding(input_ids))
    the logits                     ==  Hy4LMHead(IHCHead(the output of the last decoder layer))

so the order (embedding, expansion, layers, stream readout, final norm, LM head) is HF's own. Run
in fp32 and as HF loads the BF16 checkpoint (a BF16 model whose iHC parameters and LM head stay
fp32). Pure CPU (torch vs torch), no device. Skipped when the installed transformers predates
``hy_v4``.

``test_lm_head_precision_formats`` is the host half of the LM-head precision question: what the
device's number formats cost the checkpoint's head in exact arithmetic. The device half
(tests/test_embedding_lm_head.py ``test_lm_head_precision``) adds the device's matmul.
"""

import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCHead, ihc_expand
from models.experimental.hy4_preview.reference.lm_head import (
    Hy4Embedding,
    Hy4LMHead,
    LMHeadConfig,
    VocabShards,
    bfp8_values,
    format_agreement,
    logit_agreement,
    sampling_parameters,
    stand_in_readouts,
)

hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config  # noqa: E402

SMALL = LMHeadConfig(hidden_size=96, vocab_size=640)


def _hf_model(cfg: LMHeadConfig, bf16: bool, seed: int):
    """A two-layer HYV4ForCausalLM with random parameters, norm weight away from its trivial init."""
    hf_cfg = HYV4Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        rms_norm_eps=cfg.norm_eps,
        pad_token_id=cfg.vocab_size - 1,
        bos_token_id=0,
        eos_token_id=1,
        intermediate_size=160,
        moe_intermediate_size=48,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        q_lora_rank=48,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=64,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=4096,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = hy_v4.HYV4ForCausalLM(hf_cfg).eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # The default init (std 0.006) leaves every logit near zero; give the two tables and the
        # head readout a trained scale so a wrong order or a dropped norm weight shows.
        model.model.embed_tokens.weight.copy_(torch.randn(model.model.embed_tokens.weight.shape, generator=g))
        model.model.norm.weight.copy_(1.0 + 0.5 * torch.randn(cfg.hidden_size, generator=g))
        model.lm_head.weight.copy_(torch.randn(model.lm_head.weight.shape, generator=g) * cfg.hidden_size**-0.5)
        model.model.hc_head.hc_fn.copy_(torch.randn(model.model.hc_head.hc_fn.shape, generator=g) * 0.05)
        model.model.hc_head.hc_base.copy_(torch.randn(model.model.hc_head.hc_base.shape, generator=g))
        model.model.hc_head.hc_scale.fill_(0.7)
    if bf16:
        # What from_pretrained does with the BF16 checkpoint: everything BF16 except the modules
        # named in _keep_in_fp32_modules_strict.
        keep = hy_v4.HYV4ForCausalLM._keep_in_fp32_modules_strict
        model.to(torch.bfloat16)
        for name, p in model.named_parameters():
            if any(part in keep for part in name.split(".")):
                p.data = p.data.float()
        assert model.lm_head.weight.dtype == torch.float32
        assert model.model.embed_tokens.weight.dtype == model.model.norm.weight.dtype == torch.bfloat16
    return hf_cfg, model


def _references(cfg: LMHeadConfig, hf_cfg, model):
    """The reference modules holding the HF model's parameters, in the HF model's dtypes."""
    embedding = Hy4Embedding(cfg).eval()
    head = Hy4LMHead(cfg).eval()
    readout = IHCHead(
        IHCConfig(
            dim=cfg.hidden_size,
            n=hf_cfg.hc_mult,
            eps=hf_cfg.hc_eps,
            norm_eps=hf_cfg.rms_norm_eps,
            magnitude=hf_cfg.hc_magnitude,
        )
    ).eval()
    embedding.weight.data = model.model.embed_tokens.weight.data.clone()
    head.norm_weight.data = model.model.norm.weight.data.clone()
    head.weight.data = model.lm_head.weight.data.clone()
    readout.fn.data = model.model.hc_head.hc_fn.data.clone()
    readout.base.data = model.model.hc_head.hc_base.data.clone()
    readout.scale.data = model.model.hc_head.hc_scale.data.clone()
    return embedding, readout, head


@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("seq_len", [1, 24], ids=["S1", "S24"])
def test_model_ends_match_hf(bf16, seq_len):
    cfg = SMALL
    hf_cfg, model = _hf_model(cfg, bf16, seed=3)
    embedding, readout, head = _references(cfg, hf_cfg, model)
    input_ids = torch.randint(0, cfg.vocab_size, (2, seq_len), generator=torch.Generator().manual_seed(5))

    seen = {}
    hooks = [
        model.model.layers[0].register_forward_pre_hook(lambda m, args: seen.__setitem__("streams_in", args[0])),
        model.model.hc_head.register_forward_pre_hook(lambda m, args: seen.__setitem__("streams_out", args[0])),
        model.model.hc_head.register_forward_hook(lambda m, args, out: seen.__setitem__("readout", out)),
        model.model.norm.register_forward_hook(lambda m, args, out: seen.__setitem__("normed", out)),
    ]
    with torch.no_grad():
        hf_logits = model(input_ids=input_ids).logits
        streams_in = ihc_expand(embedding(input_ids), hf_cfg.hc_mult)
        r_readout = readout(seen["streams_out"])
        r_normed = head.norm(r_readout)
        r_logits = head(r_readout)
    for hook in hooks:
        hook.remove()

    dtype = torch.bfloat16 if bf16 else torch.float32
    assert streams_in.dtype == seen["streams_in"].dtype == dtype
    assert torch.equal(streams_in, seen["streams_in"]), "embedding + expansion differ from HF"
    assert torch.equal(r_readout, seen["readout"]), "the stream readout differs from HF"
    assert r_normed.dtype == seen["normed"].dtype == dtype
    assert torch.equal(r_normed, seen["normed"]), "the final norm differs from HF"
    assert r_logits.dtype == hf_logits.dtype == torch.float32, "HF computes the logits in fp32 either way"
    assert torch.equal(r_logits, hf_logits), f"max|Δ|={(r_logits - hf_logits).abs().max().item():.3e}"
    if bf16:
        # as_hf_bf16 must turn an fp32-held head (BF16 values, as the device tests build it) into this one.
        widened = Hy4LMHead(cfg).eval()
        widened.norm_weight.data = head.norm_weight.data.float()
        widened.weight.data = head.weight.data
        with torch.no_grad():
            twin = widened.as_hf_bf16()
            assert twin.norm_weight.dtype == torch.bfloat16 and twin.weight is widened.weight
            assert torch.equal(twin(seen["readout"]), hf_logits)
    # The check above must not pass on dead logits.
    assert hf_logits.std() > 0.1 and hf_logits.argmax(-1).unique().numel() > 1


def test_norm_weight_and_eps_are_used():
    """The pieces the bit-exact test could not tell apart from an identity must change the logits."""
    cfg = SMALL
    head = Hy4LMHead(cfg).eval()
    g = torch.Generator().manual_seed(7)
    x = 1e-3 * torch.randn(4, cfg.hidden_size, generator=g)  # small enough for eps to matter
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=g))
        base = head(x)
        head.norm_weight.mul_(2.0)
        assert torch.allclose(head(x), 2.0 * base, rtol=1e-5)
        head.norm_weight.div_(2.0)
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        expected = x / torch.sqrt(rms**2 + cfg.norm_eps)
        assert torch.allclose(head.norm(x), expected, rtol=1e-5)
        assert (head.norm(x).pow(2).mean(-1).sqrt() < 0.5).all(), "eps should dominate at this scale"


def test_lm_head_fp32_is_unconditional_in_hf():
    """config.json carries ``enable_lm_head_fp32``; HF never reads it. The fp32 head comes from
    _keep_in_fp32_modules_strict and from the cast in HYV4ForCausalLM.forward."""
    assert "enable_lm_head_fp32" not in inspect.getsource(hy_v4)
    assert "lm_head" in hy_v4.HYV4ForCausalLM._keep_in_fp32_modules_strict
    assert ".to(dtype=self.lm_head.weight.dtype)" in inspect.getsource(hy_v4.HYV4ForCausalLM.forward)


@pytest.mark.parametrize("vocab, chips", [(120832, 32), (640, 32), (640, 1)])
def test_vocab_shards(vocab, chips):
    """Every token id is one chip's row and a zero row on every other chip; the head split is the
    same cut of the vocab."""
    shards = VocabShards(vocab, chips)
    g = torch.Generator().manual_seed(11)
    ids = torch.cat([torch.tensor([0, vocab - 1]), torch.randint(0, vocab, (200,), generator=g)])
    if chips > 1:  # both sides of every chip boundary
        edges = shards.offsets()[1:]
        ids = torch.cat([ids, edges, edges - 1])
    assert torch.equal(shards.token_id(shards.chip(ids), shards.row(ids)), ids)

    rows = shards.table_rows_for(ids)  # [chips, N]
    assert rows.shape == (chips, ids.numel()) and rows.min() >= 0 and rows.max() <= shards.rows_per_chip + 1
    real = (rows >= 1) & (rows <= shards.rows_per_chip)
    assert (real.sum(0) == 1).all(), "a token id must be a real row on exactly one chip"
    assert torch.equal(real.float().argmax(0), shards.chip(ids))
    assert torch.equal(rows[shards.chip(ids), torch.arange(ids.numel())] - 1, shards.row(ids))

    hidden = 8
    table = torch.randn(vocab, hidden, generator=g)
    split = shards.split_table(table)
    assert split.shape == (chips, shards.table_rows, hidden)
    looked_up = torch.stack([split[c][rows[c]] for c in range(chips)]).sum(0)
    assert torch.equal(looked_up, table[ids]), "the per-chip lookups must sum to the embedding"

    head = shards.split_head(table)
    x = torch.randn(3, hidden, generator=g)
    assert torch.equal(torch.cat([x @ head[c] for c in range(chips)], dim=-1), x @ table.T)

    # An id outside the vocab is refused, as F.embedding refuses it, rather than read as a zero row.
    for bad in (-1, vocab, vocab + shards.rows_per_chip):
        with pytest.raises(IndexError):  # allow-pytest.raises: host-side check
            torch.nn.functional.embedding(torch.tensor([bad]), table)
        with pytest.raises(AssertionError, match="outside"):  # allow-pytest.raises: host-side check
            shards.table_rows_for(torch.tensor([0, bad]))


@pytest.mark.parametrize("vocab, chips", [(120832, 32), (4096, 32), (4096, 1)])
@pytest.mark.parametrize("ties", [False, True], ids=["distinct", "ties"])
def test_sharded_top_k_is_exact(vocab, chips, ties):
    """The selection tt/lm_head.py runs on device, in torch: every chip's top 32 with the token
    ids riding along, concatenated in chip order, then a top k of the 32 x chips candidates. For
    k <= 32 that is the top k of the whole vocab, ties included; past 32 it is not."""
    shards = VocabShards(vocab, chips)
    rows, per_chip = 5, 32
    g = torch.Generator().manual_seed(13)
    logits = 3.0 * torch.randn(rows, vocab, generator=g)
    if ties:  # a grid coarse enough that the leading values repeat, within a chip and across chips
        logits = (logits * 2).round() / 2
        assert (torch.topk(logits, 2, dim=-1).values.diff(dim=-1) == 0).any()
    logits[0, :40] = 50.0 + torch.arange(40.0)  # 40 of the global top 40 on chip 0
    local = logits.reshape(rows, chips, shards.rows_per_chip)
    values, slots = torch.topk(local, per_chip, dim=-1)  # [rows, chips, 32]
    ids = shards.token_id(torch.arange(chips)[None, :, None], slots)
    values, ids = values.reshape(rows, -1), ids.reshape(rows, -1)
    at = torch.arange(rows)[:, None]
    for k in (1, 5, 32):
        top, where = torch.topk(values, k, dim=-1)
        picked = ids[at, where]
        assert torch.equal(top, torch.topk(logits, k, dim=-1).values)
        assert torch.equal(logits[at, picked], top), "an id does not name its value"
        assert all(row.unique().numel() == k for row in picked)
    if chips > 1:  # why the device module stops at 32: the 33rd..40th of row 0 never left chip 0
        top = torch.topk(values, 40, dim=-1).values
        assert not torch.equal(top[0], torch.topk(logits, 40, dim=-1).values[0])


def test_sampling_parameters():
    """What the on-device sampler is handed: one value per row, k inside the candidate row, a
    probability, and the reciprocal of a positive temperature. The op itself checks none of it."""
    k, p, inv_t = sampling_parameters(5, 0.9, 0.5, rows=32)
    assert k.dtype == torch.int32 and k.shape == p.shape == inv_t.shape == (32,)
    assert (k == 5).all() and torch.allclose(p, torch.full((32,), 0.9)) and (inv_t == 2.0).all()
    per_row_k = torch.arange(32) + 1
    k, p, inv_t = sampling_parameters(per_row_k, torch.linspace(0, 1, 32), torch.full((32,), 4.0), rows=32)
    assert torch.equal(k, per_row_k.to(torch.int32)) and p[0] == 0 and p[-1] == 1 and (inv_t == 0.25).all()
    assert torch.equal(sampling_parameters(8, 1.0, 1.0, rows=4, max_top_k=8)[0], torch.full((4,), 8, dtype=torch.int32))
    bad = [
        dict(k=0, p=1.0, temperature=1.0),  # the kernel would draw from no candidates
        dict(k=33, p=1.0, temperature=1.0),  # past the 32 candidates of a row
        dict(k=120832, p=1.0, temperature=1.0),  # "no top-k" spelled as the vocab size
        dict(k=-1, p=1.0, temperature=1.0),  # would wrap to 4294967295 as uint32
        dict(k=2.5, p=1.0, temperature=1.0),
        dict(k=1, p=-0.1, temperature=1.0),
        dict(k=1, p=1.5, temperature=1.0),
        dict(k=1, p=1.0, temperature=0.0),  # greedy is k = 1; 1 / 0 would reach the device as inf
        dict(k=1, p=1.0, temperature=-1.0),
        dict(k=1, p=1.0, temperature=float("inf")),
        dict(k=1, p=float("nan"), temperature=1.0),
        dict(k=torch.ones(31), p=1.0, temperature=1.0),  # neither one value nor one per row
        dict(k=torch.cat([torch.ones(31), torch.tensor([40.0])]), p=1.0, temperature=1.0),  # one bad row
    ]
    for kwargs in bad:
        with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
            sampling_parameters(rows=32, **kwargs)


def test_bfp8_values():
    """The BFP8 model: a run of 16 shares its largest exponent and keeps 7 mantissa bits."""
    run = torch.tensor([1.0, -1.0, 0.5, 0.75, 1.984375, 1.9921875, 0.0078125, 0.01171875] + [0.0] * 8)
    # exponent 0 -> step 2^-6: 1.9921875 = 127.5 steps ties to even (128) and saturates at 127;
    # 0.0078125 = half a step ties to 0; 0.01171875 = 0.75 of a step rounds to one step
    want = torch.tensor([1.0, -1.0, 0.5, 0.75, 1.984375, 1.984375, 0.0, 0.015625] + [0.0] * 8)
    assert torch.equal(bfp8_values(run), want)
    g = torch.Generator().manual_seed(0)
    t = torch.randn(4, 64, generator=g) * torch.tensor([1e-3, 1.0, 37.0, 1e4])[:, None]
    q = bfp8_values(t)
    step = torch.exp2(torch.floor(torch.log2(t.reshape(4, 4, 16).abs().amax(-1, keepdim=True))) - 6)
    assert ((q - t).reshape(4, 4, 16).abs() <= step / 2).all(), "an element moved by more than half a step"
    assert torch.equal(bfp8_values(q), q), "BFP8 values must survive a second conversion"
    assert torch.equal(bfp8_values(torch.zeros(2, 32)), torch.zeros(2, 32))


def _checkpoint(names):
    """Checkpoint tensors by name from $HY4_HF_MODEL, as fp32; skip when it is not there."""
    safetensors = pytest.importorskip("safetensors")
    root = os.environ.get("HY4_HF_MODEL")
    if not root or not (Path(root) / "model.safetensors.index.json").is_file():
        pytest.skip("HY4_HF_MODEL does not point at the tencent/Hy4-preview checkpoint")
    weight_map = json.loads((Path(root) / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for name in names:
        shard = Path(root) / weight_map[name]
        if not shard.is_file():
            pytest.skip(f"{shard.name} (holds {name}) is not downloaded")
        with safetensors.safe_open(str(shard), framework="pt") as f:
            out.append(f.get_tensor(name).float())
    return out


def test_lm_head_precision_formats():
    """What the device's number formats cost the checkpoint's LM head, in exact host arithmetic.

    The reference is HF's recipe (BF16 normed state, fp32 head and logits) on stand-in readouts.
    Against it: the exact fp32 arithmetic (how far HF's own BF16 rounding of the state moves the
    logits, the yardstick), BF16 logits (the fp32 sums rounded to BF16), and a BFP8 head in the
    per-chip layout the device holds, with fp32 and with BF16 logits. A head held in fp32 or in
    BF16 has the checkpoint's numbers either way, so with fp32 logits both reproduce the
    reference exactly here; on device they differ from it only by the matmul's arithmetic.
    """
    cfg = LMHeadConfig()
    tokens = 256
    head = Hy4LMHead(cfg, init=False).eval()
    embedding, head.weight.data, head.norm_weight.data = _checkpoint(
        ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    )
    shards = VocabShards(cfg.vocab_size, 32)
    with torch.no_grad():
        assert torch.equal(head.weight.bfloat16().float(), head.weight), "the checkpoint's head is not BF16-valued"
        stand_ins = stand_in_readouts(embedding, head.weight, tokens, seed=5)
        del embedding
        held = bfp8_values(shards.split_head(head.weight))  # [chips, hidden, rows]: exponents shared along the vocab
        bfp8_weight = held.transpose(1, 2).reshape(cfg.vocab_size, cfg.hidden_size)
        del held
        rel = ((bfp8_weight - head.weight).norm() / head.weight.norm()).item()
        logger.info(f"formats | BFP8 head: relative weight error {rel:.3e}")
        hf = head.as_hf_bf16()

        results = {}
        for kind, x in stand_ins.items():
            h = hf.norm(x.bfloat16())
            ref = hf.logits(h)
            top = torch.topk(ref, 6, dim=-1).values
            margin1, margin5 = top[:, 0] - top[:, 1], top[:, 4] - top[:, 5]
            below = ", ".join(
                f"<{m}: {(margin1 < m).float().mean().item():.4f}" for m in (0.005, 0.01, 0.02, 0.05, 0.1)
            )
            logger.info(
                f"formats | reference | {kind}: logit std {ref.std().item():.3f}, top-1 logit mean "
                f"{top[:, 0].mean().item():.2f}, top-1 margin median {margin1.median().item():.4f} min "
                f"{margin1.min().item():.5f}, 5th-6th margin median {margin5.median().item():.4f}; rows with "
                f"top-1 margin {below}"
            )
            bfp8_logits = h.float() @ bfp8_weight.T
            variants = {
                "fp32 state (exact arithmetic)": head(x),
                "bf16 logits": ref.bfloat16().float(),
                "bfp8 head, fp32 logits": bfp8_logits,
                "bfp8 head, bf16 logits": bfp8_logits.bfloat16().float(),
            }
            for name, logits in variants.items():
                a = results[name, kind] = logit_agreement(ref, logits)
                logger.info(f"formats | {name} | {kind}: {format_agreement(a)}")
    for (name, kind), a in results.items():
        assert a["pcc"] >= 0.999, f"{name} | {kind}: logits pcc {a['pcc']:.6f}"
