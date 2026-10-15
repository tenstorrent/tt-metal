# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The oracle for DeepSeek-V4.1-Flash: DeepSeek's own inference code shipped with the checkpoint, on the CPU.

transformers has no deepseek_v41, and the checkpoint (476 GB: 272 GB of fp4 experts, 197 GB of engram tables) does not
fit in host RAM unpacked, so the stock loader cannot run it. Instead:

* ``inference/model.py`` and ``inference/engram.py`` are imported unmodified from the checkpoint directory, with
  ``reference/kernels.py`` (a CPU port of ``inference/kernel.py``: same fp8 / fp4 quantize-dequantize, fp32 GEMMs)
  installed as their ``kernel`` module, and model.py's own GEMMs (``F.linear``, the two ``torch.einsum``
  contractions) run in the same fixed tiles as the reference's (kernels.py "GEMM tiling": the GPU kernels tile too;
  identical tiling makes oracle and reference agree to the bit, where the fp8 / fp4 quantizers would otherwise turn
  last-bit summation differences into whole quantization steps);
* the model is built as ``generate.py`` builds it (``ModelArgs`` from ``inference/config.json``, batch 1), text only
  (no vision tower, no DSpark / MTP layers), with two storage substitutions made while it is built:
  routed experts are ``LazyExpert``s (no parameters: each forward reads the packed fp4 weights of that one expert from
  the checkpoint and runs model.py's own ``Expert.forward`` on them; a pre-hook on each MoE reads and dequantizes the
  experts it will run a few ahead, on a thread pool), and the engram tables are memory-mapped
  (``MemmapEngramEmbedding``: the rows looked up are read, dequantized as ``ParallelEngramEmbedding`` does);
* every other weight is loaded as ``convert.py`` prepares it for one rank: names unchanged, ``wo_a`` dequantized to
  bf16 (exact), everything else in its stored dtype, cast into the parameter's dtype;
* ``dtype`` is torch's default dtype while it builds and runs: bf16 is the GPU configuration (``generate.py``); fp32
  (the parity check) keeps every activation in fp32, and the few parameters model.py hard-codes as bf16 (wo_a, the
  ratio-1 compressor, indexer wk / weights_proj; all bf16 in the checkpoint) are widened so the fp32 GEMMs run;
* ``forward`` is model.py's ``Transformer.forward`` without sampling, with every position's logits and, after each
  block, a tap (``taps[i]``, what ``hooks.hf_layers`` returns) on the block output packed like the reference's
  boundary: [S, 4 * H + 4] = the 4 streams, then the pre weights the next block collapses with;
* ``generate`` is greedy and runs like ``generate.py``: prefill from position 0, then one token per step.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.deepseek_v41_flash_d_p.reference import kernels
from models.demos.deepseek_v41_flash_d_p.reference.weights import (
    Checkpoint,
    EngramTable,
    ExpertPrefetcher,
    fp8_block_dequant,
)

FP4 = torch.float4_e2m1fn_x2


@contextmanager
def default_dtype(dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _load_file(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Proxy(types.ModuleType):
    """A module that is ``base`` except for the attributes given."""

    def __init__(self, base, **override):
        super().__init__(base.__name__)
        self._base = base
        self.__dict__.update(override)

    def __getattr__(self, name):
        return getattr(self._base, name)


@lru_cache(maxsize=None)
def inference_module(model_path: str):
    """model.py from ``<checkpoint>/inference``, importing our CPU kernels as ``kernel`` (sys.modules restored), its
    GEMMs (``F.linear``, ``torch.einsum``) tiled as the reference's are (kernels.py "GEMM tiling")."""
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import inference_dir

    d = inference_dir(model_path)
    bare = ("kernel", "engram", "image_processor", "vision")
    saved = {n: sys.modules.get(n) for n in bare}
    try:
        sys.modules["kernel"] = kernels
        for n in bare[1:]:
            sys.modules[n] = _load_file(f"ds41_inference_{n}", os.path.join(d, f"{n}.py"))
        mod = _load_file("ds41_inference_model", os.path.join(d, "model.py"))
        mod.F = _Proxy(F, linear=kernels.linear)
        mod.torch = _Proxy(torch, einsum=kernels.einsum)
        return mod
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


class LazyExpert(nn.Module):
    """A routed expert whose packed fp4 weights stay in the checkpoint; forward = model.py's Expert.forward."""

    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit: float = 0.0):
        super().__init__()
        assert dtype == FP4
        self.swiglu_limit = swiglu_limit
        self.where = None  # (mod, checkpoint, layer, expert id), set after the model is built
        self.prefetch: ExpertPrefetcher | None = None  # set per MoE forward (moe_prefetch)

    def forward(self, x, weights=None):
        mod, ck, layer, e = self.where
        packed = self.prefetch.get(e) if self.prefetch is not None else None
        if packed is None:
            packed = {n: (w, s, None) for n, (w, s) in ck.expert(layer, e).items()}
        ws = {}
        for n, (w, s, w32) in packed.items():
            t = w.view(FP4)
            t.scale = s  # model.py's Linear attaches the scale to the weight; linear() reads weight.scale
            t.dequant = w32  # the same weight dequantized ahead of time (kernels.fp4_gemm uses it if present)
            ws[n] = t
        shim = SimpleNamespace(
            w1=lambda v: mod.linear(v, ws["w1"]),
            w2=lambda v: mod.linear(v, ws["w2"]),
            w3=lambda v: mod.linear(v, ws["w3"]),
            swiglu_limit=self.swiglu_limit,
        )
        return mod.Expert.forward(shim, x, weights)


def moe_prefetch(ck: Checkpoint, layer: int):
    """A forward pre-hook on a MoE: route (the MoE's own gate, deterministic) to learn which experts will run, and
    read + dequantize them ahead in that order (weights.ExpertPrefetcher); the numbers are unchanged."""

    def hook(moe, args):
        x = args[0]
        _, idx = moe.gate(x.reshape(-1, moe.dim))
        pf = ExpertPrefetcher(ck, layer, sorted(set(idx.flatten().tolist())))
        for e in pf.order:
            moe.experts[e].prefetch = pf

    return hook


class MemmapEngramEmbedding(nn.Module):
    """ParallelEngramEmbedding (one rank) over the memory-mapped table: rows dequantized on lookup, cast to bf16."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings, self.dim = num_embeddings, dim
        self.table: EngramTable | None = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.table.num_rows == self.num_embeddings
        return self.table.rows(indices).to(torch.bfloat16)


class Tap(nn.Module):
    def forward(self, x):
        return x


class DS41Oracle(nn.Module):
    """model.py's Transformer, driven as a causal LM (``forward(input_ids).logits``, greedy ``generate``)."""

    def __init__(self, mod, model, dtype, eos_token_id: int):
        super().__init__()
        self.mod, self.model, self.dtype, self.eos = mod, model, dtype, eos_token_id
        self.taps = nn.ModuleList(Tap() for _ in model.layers)

    @torch.inference_mode()
    def _run(self, input_ids: torch.Tensor, start_pos: int, full_logits: bool) -> torch.Tensor:
        """Transformer.forward (text only, no MTP hiddens, no sampling) with a tap after each block."""
        m = self.model
        with default_dtype(self.dtype):
            hashes = m.engram_hash(input_ids, start_pos, None) if m.engram_hash is not None else None
            h = m.embed(input_ids)
            h = h.unsqueeze(2).repeat(1, 1, m.hc_mult, 1)
            pre_mix = self.mod.make_identity_pre_mix(h, m.hc_mult)
            for i, layer in enumerate(m.layers):
                if layer.engram is not None:
                    h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
                h, pre_mix = layer(h, start_pos, pre_mix, None)
                self.taps[i](torch.cat([h.flatten(2), pre_mix.to(h.dtype)], dim=-1))
            h = layer.hc_pre(h, pre_mix)
            return m.head(m.norm(h), full_logits=full_logits)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **_):
        assert input_ids.shape[0] == 1, "batch 1"
        if attention_mask is not None:
            assert bool(attention_mask.all()), "padding not supported"
        return SimpleNamespace(logits=self._run(input_ids, 0, full_logits=True))

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, do_sample=False, **_):
        """Greedy: prefill the prompt from position 0, then decode one token per step (inference generate.py)."""
        assert not do_sample and input_ids.shape[0] == 1
        if attention_mask is not None:
            assert bool(attention_mask.all()), "padding not supported"
        ids = input_ids[0].tolist()
        logits = self._run(input_ids, 0, full_logits=False)
        for step in range(max_new_tokens):
            nxt = int(logits[0].argmax(-1))
            ids.append(nxt)
            if nxt == self.eos or step == max_new_tokens - 1:
                break
            logits = self._run(torch.tensor([[nxt]]), len(ids) - 1, full_logits=False)
        return torch.tensor([ids])


def build_oracle(model_path: str, num_layers: int | None = None, dtype=torch.bfloat16, max_seq_len: int = 8192):
    """DeepSeek's inference Transformer (first num_layers blocks, all if None) on the checkpoint, on the CPU."""
    from transformers import AutoTokenizer

    mod = inference_module(model_path)
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import inference_config

    raw = inference_config(model_path)  # a scaled checkpoint: the source's with the cut counts
    args = mod.ModelArgs(**raw)
    args.max_batch_size = 1
    args.max_seq_len = max_seq_len
    args.n_layers = num_layers or raw["n_layers"]
    args.dspark_block_size = 0  # no DSpark / MTP layers
    args.vision_n_layers = 0  # no vision tower
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    real_expert, real_engram = mod.Expert, mod.ParallelEngramEmbedding

    def expert_factory(dim, inter_dim, dtype=None, swiglu_limit=0.0):
        if dtype == FP4:
            return LazyExpert(dim, inter_dim, dtype, swiglu_limit)
        return real_expert(dim, inter_dim, dtype=dtype, swiglu_limit=swiglu_limit)

    try:
        mod.Expert, mod.ParallelEngramEmbedding = expert_factory, MemmapEngramEmbedding
        with default_dtype(dtype):
            model = mod.Transformer(args, tokenizer)
    finally:
        mod.Expert, mod.ParallelEngramEmbedding = real_expert, real_engram

    ck = Checkpoint(model_path)
    for i, layer in enumerate(model.layers):
        for e, ex in enumerate(layer.ffn.experts):
            ex.where = (mod, ck, i, e)
        layer.ffn.register_forward_pre_hook(moe_prefetch(ck, i))
        if layer.engram is not None:
            layer.engram.embed.table = EngramTable(ck, i)

    loaded = set()
    with torch.no_grad():
        for name, p in model.state_dict(keep_vars=True).items():
            if not isinstance(p, nn.Parameter):
                continue  # buffers: caches and tables model.py computes itself
            if name.endswith("wo_a.weight"):  # convert.py: dequantized to bf16 (exact)
                src = fp8_block_dequant(ck.view(name), ck.view(name[: -len("weight")] + "scale"), torch.bfloat16)
            else:
                src = ck.view(name)
            assert tuple(src.shape) == tuple(p.shape), (name, src.shape, p.shape)
            p.data.copy_(src)
            loaded.add(name)
            if dtype == torch.float32 and p.dtype == torch.bfloat16:
                p.data = p.data.float()  # hard-coded bf16 Linear in model.py: widened for the fp32 run
    params = {n for n, p in model.named_parameters()}
    assert params == loaded, sorted(params ^ loaded)[:10]
    for p in model.parameters():
        p.requires_grad_(False)
    return DS41Oracle(mod, model.eval(), dtype, tokenizer.eos_token_id)
