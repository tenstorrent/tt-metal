# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's own ``inference/model.py`` as a CPU fp32 reference.

Its tilelang ``kernel`` module is replaced by torch: the activation-quantization
simulations become no-ops (the TT port does not reproduce them) and ``sparse_attn`` a
gather + softmax with the attention sink. Weights are loaded dequantized, so every
``Linear`` takes the plain ``F.linear`` path.
"""

import dataclasses
import importlib.util
import json
import sys
import types

import torch

from models.experimental.deepseek_v41_flash.tt.config import dequantized


def _sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """``q`` ``[b, s, h, d]`` over the ``kv`` ``[b, n, d]`` rows ``topk_idxs`` ``[b, s, k]`` names (-1: none)."""
    rows = kv[torch.arange(kv.shape[0])[:, None, None], topk_idxs.clamp(min=0).long()]  # [b, s, k, d]
    scores = torch.einsum("bshd,bskd->bshk", q.float(), rows.float()) * softmax_scale
    scores = scores.masked_fill((topk_idxs < 0)[:, :, None], float("-inf"))
    sink = attn_sink.float().view(1, 1, -1, 1).expand(*scores.shape[:-1], 1)
    probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs, rows.float()).to(q.dtype)


def _hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """``mixes`` ``[b, s, (2 + hc) * hc]`` -> ``pre`` / ``post`` ``[b, s, hc]`` and the doubly stochastic
    ``comb`` ``[b, s, hc, hc]``, as the tilelang kernel computes them."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _unsupported(*args, **kwargs):
    raise NotImplementedError("quantized GEMMs are not part of the reference")


def load_reference(snapshot_dir):
    """Import ``<snapshot_dir>/inference/model.py`` with torch stand-ins for ``kernel``."""
    kernel = types.ModuleType("kernel")
    kernel.act_quant = kernel.fp4_act_quant = lambda *args, **kwargs: None
    kernel.sparse_attn = _sparse_attn
    kernel.hc_split_sinkhorn = _hc_split_sinkhorn
    kernel.fp4_gemm = kernel.fp8_gemm = _unsupported
    sys.modules["kernel"] = kernel
    sys.path.insert(0, str(snapshot_dir / "inference"))
    spec = importlib.util.spec_from_file_location("deepseek_v41_reference", snapshot_dir / "inference" / "model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    model.default_dtype = torch.float32
    return model


def model_args(model, snapshot_dir, max_seq: int, max_batch_size: int = 1):
    """``model.ModelArgs`` from ``inference/config.json``, keeping only the dataclass fields."""
    with open(snapshot_dir / "inference" / "config.json") as f:
        cfg = json.load(f)
    fields = {f.name for f in dataclasses.fields(model.ModelArgs)}
    return model.ModelArgs(
        **{k: v for k, v in cfg.items() if k in fields}, max_batch_size=max_batch_size, max_seq_len=max_seq
    )


def reference_attentions(model, snapshot_dir, loader, layer_ids, max_seq: int) -> dict:
    """fp32 ``model.Attention`` per layer id, with the checkpoint's (dequantized) weights."""
    args = model_args(model, snapshot_dir, max_seq)
    layers = {}
    for i in layer_ids:
        attn = model.Attention(i, args)
        attn.load_state_dict({k: dequantized(loader, f"layers.{i}.attn.{k}")() for k in attn.state_dict()})
        layers[i] = attn.float().eval()
    return layers


class _LazyExpert(torch.nn.Module):
    """``model.Expert`` built from the checkpoint each time it is called, then dropped: the 384
    experts of a layer are ~54 GB in fp32, and a few tokens route to only a handful."""

    def __init__(self, build):
        super().__init__()
        self.build = build

    def forward(self, *args):
        return self.build()(*args)


def reference_moe(model, snapshot_dir, loader, layer_id: int, prefix: str | None = None):
    """fp32 ``model.MoE`` of layer ``layer_id``, with the checkpoint's (dequantized) weights under ``prefix``
    (default ``layers.{layer_id}.ffn.``)."""
    args = dataclasses.replace(model_args(model, snapshot_dir, max_seq=1), expert_dtype=None)
    prefix = prefix or f"layers.{layer_id}.ffn."
    with torch.device("meta"):
        moe = model.MoE(layer_id, args)
    for module, name in ((moe.gate, "gate"), (moe.shared_experts, "shared_experts")):
        module.load_state_dict(
            {k: dequantized(loader, f"{prefix}{name}.{k}")() for k in module.state_dict()}, assign=True
        )

    def expert(i):
        def build():
            with torch.device("meta"):
                e = model.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
            e.load_state_dict(
                {k: dequantized(loader, f"{prefix}experts.{i}.{k}")() for k in e.state_dict()}, assign=True
            )
            return e

        return _LazyExpert(build)

    moe.experts = torch.nn.ModuleList(expert(i) for i in range(moe.n_routed_experts))
    return moe.eval()


def reference_blocks(model, snapshot_dir, loader, layer_ids, max_seq: int) -> dict:
    """fp32 ``model.Block`` per layer id: :func:`reference_attentions` and :func:`reference_moe` for its
    sublayers, the checkpoint's hyper-connection and norm weights for the rest.

    The blocks have no Engram (``block.engram`` is ``None``); like ``Transformer.forward``, the caller
    applies :func:`reference_engram` to the streams before an Engram layer's block.
    """
    args = model_args(model, snapshot_dir, max_seq)
    attns = reference_attentions(model, snapshot_dir, loader, layer_ids, max_seq)

    class Block(model.Block):
        """``model.Block`` around the prebuilt attention, so the meta device below never reaches it."""

        def attention_cls(self, layer_id, args):
            return attns[layer_id]

    blocks = {}
    for i in layer_ids:
        with torch.device("meta"):
            block = Block(i, args)
        block.ffn = reference_moe(model, snapshot_dir, loader, i)
        own = [k for k in block.state_dict() if not k.startswith(("attn.", "ffn."))]
        block.load_state_dict({k: dequantized(loader, f"layers.{i}.{k}")() for k in own}, strict=False, assign=True)
        blocks[i] = block.eval()
    return blocks


def reference_dspark(model, snapshot_dir, loader, max_seq: int):
    """The checkpoint's DSpark drafter in fp32: ``model.DSparkBlock`` stages ``mtp.0..`` with the tied
    ``embed`` / ``head``, driven by ``Transformer.forward_spec``.

    Returns a namespace with ``mtp`` and ``hc_mult`` (what ``forward_spec`` reads) and
    ``spec(input_ids, main_hidden, start_pos)``. ``main_hidden`` ``[b, 1, len(target_layer_ids) * dim]``
    is the stream mean at the target layers' input for position ``start_pos``; at ``start_pos == 0`` it
    may cover the whole prompt and only seeds the window.
    """
    args = model_args(model, snapshot_dir, max_seq)
    embed = model.ParallelEmbedding(args.vocab_size, args.dim)
    embed.weight = torch.nn.Parameter(dequantized(loader, "embed.weight")(), requires_grad=False)
    head = model.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
    head.weight = torch.nn.Parameter(dequantized(loader, "head.weight")(), requires_grad=False)
    stages = []
    for stage in range(args.n_mtp_layers):
        layer_id = args.n_layers + stage
        prefix = f"mtp.{stage}."
        attn = model.DSparkAttention(layer_id, args)
        attn.load_state_dict({k: dequantized(loader, f"{prefix}attn.{k}")() for k in attn.state_dict()})
        attn = attn.float().eval()

        class Block(model.DSparkBlock):
            """``model.DSparkBlock`` around the prebuilt attention, so the meta device below never reaches it."""

            def attention_cls(self, layer_id, args, attn=attn):
                return attn

        with torch.device("meta"):
            block = Block(layer_id, args)
        block.ffn = reference_moe(model, snapshot_dir, loader, layer_id, prefix=f"{prefix}ffn.")
        own = [k for k in block.state_dict() if not k.startswith(("attn.", "ffn.", "embed.", "head."))]
        block.load_state_dict({k: dequantized(loader, prefix + k)() for k in own}, strict=False, assign=True)
        block.embed, block.head = embed, head
        block.temperature = 0  # greedy, as the device drafts
        stages.append(block.eval())
    drafter = types.SimpleNamespace(mtp=torch.nn.ModuleList(stages), hc_mult=args.hc_mult)
    drafter.spec = lambda input_ids, main_hidden, start_pos: model.Transformer.forward_spec(
        drafter, input_ids, main_hidden, start_pos
    )
    return drafter


def reference_engram_hash(model, snapshot_dir, tokenizer, max_seq: int, max_batch_size: int = 1):
    """The checkpoint's ``NgramHashState``: token history plus the n-gram hash of every Engram layer."""
    args = model_args(model, snapshot_dir, max_seq, max_batch_size)
    return model.NgramHashState(args, model.EngramLayout.from_args(args), tokenizer)


def reference_engram_embedding(model, layout, layer_id: int, weight: torch.Tensor, scale: torch.Tensor):
    """Layer ``layer_id``'s ``model.ParallelEngramEmbedding`` over a caller-provided table.

    ``weight`` ``[rows, head_dim]`` fp8 E4M3 and ``scale`` ``[rows, head_dim / 32]`` E8M0 are the
    checkpoint's tensors as stored (~100 GB), so the module is built on the meta device rather than
    allocating its own copy.
    """
    rows = layout.num_embeddings[layout.layer_ids.index(layer_id)]
    assert tuple(weight.shape) == (rows, layout.head_dim), (tuple(weight.shape), rows, layout.head_dim)
    with torch.device("meta"):
        embed = model.ParallelEngramEmbedding(rows, layout.head_dim)
    embed.weight = torch.nn.Parameter(weight, requires_grad=False)
    embed.scale = torch.nn.Parameter(scale, requires_grad=False)
    return embed.eval()


class _Fp32(torch.nn.Module):
    """``module``'s output as fp32: the bf16 rows then meet the fp32 ``wkv`` in ``F.linear`` (exact)."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args):
        return self.module(*args).float()


class GivenRows(torch.nn.Module):
    """Stands in for ``Engram.embed`` over rows looked up elsewhere: ignores the hash ids and returns ``rows``
    ``[B, L, n_hash_cols * head_dim]`` as fp32 ``[B, L, n_hash_cols, head_dim]``, so a test needs no ~100 GB table."""

    def __init__(self, rows: torch.Tensor, head_dim: int):
        super().__init__()
        self.rows = rows.float().unflatten(-1, (-1, head_dim))

    def forward(self, hash_ids):
        return self.rows


def reference_engram_with_embed(model, snapshot_dir, loader, layer_id: int, embed: torch.nn.Module):
    """fp32 ``model.Engram`` of layer ``layer_id``: the checkpoint's dequantized ``wkv`` / ``q_weight`` /
    ``k_weight``, and ``embed`` (hash ids -> fp32 ``[B, L, n_hash_cols, head_dim]`` rows) as its table lookup."""
    args = model_args(model, snapshot_dir, max_seq=1)
    layout = model.EngramLayout.from_args(args)
    with torch.device("meta"):
        engram = model.Engram(args, layer_id, layout)
    engram.embed = embed
    prefix = f"layers.{layer_id}.engram."
    engram.wkv.weight = torch.nn.Parameter(dequantized(loader, prefix + "wkv.weight")(), requires_grad=False)
    for name in ("q_weight", "k_weight"):
        setattr(engram, name, torch.nn.Parameter(dequantized(loader, prefix + name)().float(), requires_grad=False))
    return engram.eval()


def reference_engram(model, snapshot_dir, loader, layer_id: int, weight: torch.Tensor, scale: torch.Tensor):
    """:func:`reference_engram_with_embed` with the table as :func:`reference_engram_embedding` over the caller's
    ``weight`` / ``scale``."""
    layout = model.EngramLayout.from_args(model_args(model, snapshot_dir, max_seq=1))
    embed = _Fp32(reference_engram_embedding(model, layout, layer_id, weight, scale))
    return reference_engram_with_embed(model, snapshot_dir, loader, layer_id, embed)
