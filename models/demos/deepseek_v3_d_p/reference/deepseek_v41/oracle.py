# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trusted, cached CPU expected results for DeepSeek-V4.1 prefill (bead §6, tt-metal_tracker-8y7.7).

The vendored reference (``model.py``) is the semantic authority; this module only builds it, observes it
with forward hooks, and caches what it saw. The reference runs single-shot (``start_pos == 0``) and is
never called with ``start_pos > 0``: chunked expectations are slices of the single-shot result
(``chunk_expectations``, graph.md §4).

Typical use::

    spec = real_spec((2, 3), seq_len=2048)                   # V4.1 layers 2 -> 3, real dims, synthetic
    spec = real_spec((0, 2, 3), 2048, checkpoint=HF_SNAPSHOT)  # the same with checkpoint weights
    spec = small_spec(seq_len=41)                           # small dims, all six block types + DSpark
    model = build_reference(spec)                           # e.g. for device weights
    result = oracle(spec, random_tokens(spec), model)       # cached captures and state
    chunks = chunk_expectations(result, chunk_len=5120, pad_multiple=2 * 32 * sp)

Weights. Every model unit (``embed``, ``layers.<id>``, ``norm``, ``head``, ``mtp.<k>``) is filled by
``testing.init_weights`` from its own seed, derived from (seed, unit name, V4.1 layer id). A layer's
synthetic weights therefore do not depend on which other layers the schedule holds. Checkpoint weights
are read from the HF shards as stored (FP8/FP4 + E8M0 scales) and converted like upstream ``convert.py``
(``wo_a`` dequantized to bf16, FP4 experts viewed as ``float4_e2m1fn_x2``). Synthetic Engram tables are
row-sparse: row ``r`` of layer ``l`` is drawn from its own seed, and only the rows a prompt hashes to are
materialized (a real table has 384M rows); lookups still run through the reference module.

Result layout (all tensors batch-squeezed, keyed by V4.1 layer id; the small schedule's ids are its
positions)::

    tokens [S] int64, logits [vocab] fp32 (last position)
    meta: layer_ids, compress_ratio{id}, window, kv_sources, index_sources, candidate_source, seq_len
    blocks[id]: x_in [S,4,D] (after Engram), pre_in [S,4] fp32, attn_in [S,D] (normed), attn_out [S,D],
                ffn_in [S,D] (normed), ffn_out [S,D], x_out [S,4,D], pre_out [S,4] (next block's pre_mix),
                window_kv [S,head_dim] (post-RoPE, post-FP8-QDQ window KV of every position);
                Engram layers also engram_in [S,4,D] and engram_hash_ids [S,n_hash_cols]
    shared[id] (as published by that source's attention):
                KV sources: compress_kv [S//r,head_dim] (RoPE + FP4 QDQ), index_k [S//r,index_head_dim]
                index sources: topk_idxs [S,min(index_topk,S//r)] int32 compressed row ids (window offset
                removed), -1 = unused slot
                candidate source: candidates [S,S//r] bool
    state (after the prompt): window{id} [window,head_dim] ring (slot = pos % window);
                carry{id} {kv, score} [ratio,head_dim] fp32 (ratio-2 KV sources; slots >= S % ratio are
                0 / -inf); engram_tokens [S] compressed token ids (Engram schedules);
                main_hidden [S,n_taps*D] and dspark_window{k} [window,head_dim] (DSpark schedules)

Disk cache (``CACHE_DIR``, env ``TT_V41_ORACLE_CACHE``). Results are keyed by the model args, layer ids,
seed, weights source (``synthetic`` or the checkpoint revision), tokenizer source, a sha256 of the prompt
tokens, a digest of the vendored reference sources (model/kernel_cpu/engram/testing .py), the torch
version and ``PACKAGE_VERSION``; synthetic weights per unit by unit name, parameter signature, seed and
the same digest. Invalidation: editing a vendored reference file or changing torch invalidates
automatically; bump ``PACKAGE_VERSION`` whenever this module changes what is captured or how weights are
derived; delete the directory to reclaim space (files are ``oracle-*.pt`` and ``weights-*.pt``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.engram import EngramLayout, compute_hash_multipliers
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.testing import (
    StubTokenizer,
    init_weights,
    prefill,
    quantize_fp8_rows,
    small_model_args,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig, V41BlockType

PACKAGE_VERSION = 1
CACHE_DIR = Path(os.environ.get("TT_V41_ORACLE_CACHE", Path.home() / ".cache" / "tt-v41-oracle"))
HF_SNAPSHOT = Path(
    os.environ.get(
        "TT_V41_HF_SNAPSHOT",
        Path.home()
        / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots"
        / "dba1be0a40aa45a94ad051997016db3960a90277",
    )
)
_HERE = Path(__file__).parent
_REFERENCE_SOURCES = ("model.py", "kernel_cpu.py", "engram.py", "testing.py")
# per-token captures, sliced by chunk_expectations
_TOKEN_KEYS = (
    "x_in",
    "pre_in",
    "attn_in",
    "attn_out",
    "ffn_in",
    "ffn_out",
    "x_out",
    "pre_out",
    "window_kv",
    "engram_in",
    "engram_hash_ids",
)


@dataclass(frozen=True)
class OracleSpec:
    """What determines a reference model: its args, the checkpoint layer id of each backbone position,
    the full model those layers come from (Engram hashing depends on it), the seed of synthetic weights,
    and the checkpoint snapshot (None = synthetic weights)."""

    args: v41.ModelArgs
    layer_ids: tuple[int, ...]
    model_args: v41.ModelArgs
    seed: int = 0
    checkpoint: Path | None = None


def released_args(seq_len: int) -> v41.ModelArgs:
    """The released V4.1-Flash args (vendored ``config.json``), text only, batch 1, greedy."""
    cfg = {
        k: tuple(v) if isinstance(v, list) else v for k, v in json.loads((_HERE / "config.json").read_text()).items()
    }
    return replace(v41.ModelArgs(**cfg), max_batch_size=1, max_seq_len=seq_len, temperature=0.0, vision_n_layers=0)


def real_spec(
    layers: tuple[int, ...],
    seq_len: int,
    *,
    dspark: bool = False,
    candidate_topk_blocks: int | None = None,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> OracleSpec:
    """A reference made of V4.1 layers ``layers`` (execution order) at real dims.

    Every layer keeps its V4.1 role: compress ratio (0 = sliding window only, base RoPE), KV / index /
    candidate source, Engram (layers 1, 14). A layer that reads shared state needs the layers producing
    it in ``layers``; ``dspark`` adds the 3 DSpark layers and needs their tap layers 37-39.
    ``candidate_topk_blocks`` overrides 2048 (at short prompts every block is a candidate otherwise).
    """
    cfg = DeepSeekV41FlashConfig
    layers = tuple(layers)
    if not layers or list(layers) != sorted(set(layers)) or not 0 <= layers[0] <= layers[-1] < cfg.NUM_LAYERS:
        raise ValueError(f"layers must be strictly increasing V4.1 backbone layer ids, got {layers}")
    needed = set()
    for layer in layers:
        if cfg.compress_ratio(layer):
            needed |= {cfg.kv_source(layer), cfg.index_source(layer)}
        if cfg.block_type(layer) == V41BlockType.CANDIDATE_INDEX_SOURCE:
            needed.add(cfg.CANDIDATE_SOURCE_LAYER)
    if dspark:
        needed |= set(cfg.DSPARK_TARGET_LAYER_IDS)
    if missing := sorted(needed - set(layers)):
        raise ValueError(f"layers {layers} read shared state or taps of layers {missing}, which must be included")

    full = released_args(seq_len)
    pos = {layer: i for i, layer in enumerate(layers)}

    def positions(ids):
        return tuple(pos[i] for i in ids if i in pos)

    engram = [i for i, layer in enumerate(full.engram_layer_ids) if layer in pos]
    args = replace(
        full,
        n_layers=len(layers),
        n_mtp_layers=cfg.NUM_DSPARK_LAYERS if dspark else 0,
        compress_ratios=tuple(cfg.compress_ratio(i) for i in layers) + (0,) * (cfg.NUM_DSPARK_LAYERS if dspark else 0),
        kv_source_layers=positions(full.kv_source_layers),
        index_source_layers=positions(full.index_source_layers),
        candidate_source_layer=pos.get(full.candidate_source_layer, -1),
        candidate_topk_blocks=candidate_topk_blocks or full.candidate_topk_blocks,
        engram_layer_ids=tuple(pos[full.engram_layer_ids[i]] for i in engram),
        engram_num_embeddings=tuple(full.engram_num_embeddings[i] for i in engram),
        dspark_block_size=full.dspark_block_size if dspark else 0,
        dspark_target_layer_ids=positions(full.dspark_target_layer_ids) if dspark else (),
    )
    return OracleSpec(args, layers, full, seed, None if checkpoint is None else Path(checkpoint))


def small_spec(seq_len: int, *, seed: int = 0, **overrides) -> OracleSpec:
    """The small-dims schedule of ``testing.small_model_args`` (six block types, Engram, one DSpark
    layer); layer ids are positions. Synthetic weights only."""
    args = small_model_args(max_seq_len=seq_len, **overrides)
    return OracleSpec(args, tuple(range(args.n_layers)), args, seed)


def random_tokens(spec: OracleSpec, seq_len: int | None = None, seed: int = 1) -> torch.Tensor:
    """[1, S] uniformly random token ids (S defaults to the spec's max_seq_len)."""
    seq_len = seq_len or spec.args.max_seq_len
    return torch.randint(0, spec.args.vocab_size, (1, seq_len), generator=torch.Generator().manual_seed(seed))


def text_tokens(seq_len: int) -> torch.Tensor:
    """[1, seq_len] V4.1-tokenized real text: A Tale of Two Cities from its first chapter (the book the
    tt_transformers accuracy references use), for accuracy gates that score predictions of real text."""
    import bz2

    from transformers import AutoTokenizer

    book = bz2.open(_HERE.parents[3] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2", "rt").read()
    text = book[book.index("It was the best of times") :][: 12 * seq_len]  # > 1 character per token
    ids = AutoTokenizer.from_pretrained(HF_SNAPSHOT)(text, add_special_tokens=False)["input_ids"]
    if len(ids) < seq_len:
        raise ValueError(f"{len(ids)} tokens of text, {seq_len} requested")
    return torch.tensor(ids[:seq_len], dtype=torch.int64)[None]


# ---------------------------------------------------------------------------------------------- model


def _digest(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _reference_digest(synthetic: bool = True) -> str:
    """Digest of the reference sources a result depends on. ``testing.py`` only builds synthetic weights, so
    results from checkpoint weights exclude it (a synthetic-init change must not invalidate real results)."""
    h = hashlib.sha256()
    for name in _REFERENCE_SOURCES:
        if synthetic or name != "testing.py":
            h.update((_HERE / name).read_bytes())
    return h.hexdigest()[:16]


def _uses_real_tokenizer(spec: OracleSpec) -> bool:
    return spec.model_args.vocab_size == DeepSeekV41FlashConfig.VOCAB_SIZE


def _tokenizer(spec: OracleSpec):
    if not spec.args.engram_layer_ids:
        return None
    if not _uses_real_tokenizer(spec):
        return StubTokenizer(spec.args.vocab_size)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.checkpoint or HF_SNAPSHOT)


def _source_ids(spec: OracleSpec) -> dict:
    """The weights and tokenizer identities that enter every cache key."""
    weights = "synthetic" if spec.checkpoint is None else f"hf:{spec.checkpoint.name}"
    tokenizer = f"hf:{(spec.checkpoint or HF_SNAPSHOT).name}" if _uses_real_tokenizer(spec) else "stub"
    return {"weights": weights, "tokenizer": tokenizer}


def _units(model: v41.Transformer, spec: OracleSpec) -> list[tuple[str, torch.nn.Module]]:
    """(checkpoint name, module) of every weight unit, backbone layers under their checkpoint ids."""
    units = [("embed", model.embed)]
    units += [(f"layers.{spec.layer_ids[i]}", layer) for i, layer in enumerate(model.layers)]
    units += [("norm", model.norm), ("head", model.head)]
    units += [(f"mtp.{k}", blk) for k, blk in enumerate(model.mtp)]
    return units


class _Detached:
    """Within the block, hide the shared embed/head that the DSpark blocks register as submodules."""

    def __init__(self, module: torch.nn.Module):
        self.module, self.saved = module, {}

    def __enter__(self) -> torch.nn.Module:
        for name in ("embed", "head"):
            if isinstance(getattr(self.module, name, None), torch.nn.Module):
                self.saved[name] = getattr(self.module, name)
                setattr(self.module, name, None)
        return self.module

    def __exit__(self, *exc):
        for name, mod in self.saved.items():
            setattr(self.module, name, mod)


def _unit_seed(seed: int, name: str) -> int:
    # 32 bits: torch's CPU generator only uses the low 32 bits of a seed
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16)


def _signature(module: torch.nn.Module) -> list:
    return [(n, tuple(p.shape), str(p.dtype)) for n, p in module.named_parameters()]


def _attach_scales(module: torch.nn.Module) -> None:
    # load_state_dict(assign=True) drops the `.scale` attribute Linear attaches to its weight
    for mod in module.modules():
        if isinstance(mod, v41.Linear) and mod.scale is not None:
            mod.weight.scale = mod.scale


def _synthetic_unit(name: str, module: torch.nn.Module, seed: int) -> None:
    path = CACHE_DIR / f"weights-{_digest(name, _signature(module), seed, _reference_digest(), PACKAGE_VERSION)}.pt"
    if path.is_file():
        module.load_state_dict(torch.load(path, mmap=True), assign=True)
        _attach_scales(module)
        return
    # a top-level unit is wrapped under its name, so init_weights sees e.g. "norm" as a norm owner
    init_weights(module if "." in name else torch.nn.ModuleDict({name: module}), _unit_seed(seed, name))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(module.state_dict(), tmp)
    tmp.replace(path)


def _checkpoint_unit(name: str, module: torch.nn.Module, checkpoint: Path) -> None:
    """Load a unit from the HF shards, converted as upstream convert.py does for one rank."""
    from safetensors import safe_open

    if name.startswith("layers.") and getattr(module, "engram", None) is not None:
        raise NotImplementedError(f"{name}: checkpoint Engram tables (384M rows) are not loaded by the oracle")
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = name + "."
    by_file: dict[str, list[str]] = {}
    for full_name, file in weight_map.items():
        if full_name.startswith(prefix):
            by_file.setdefault(file, []).append(full_name)
    if not by_file:
        raise KeyError(f"no tensors named {prefix}* in {checkpoint}")
    if missing := sorted(f for f in by_file if not (checkpoint / f).is_file()):
        raise FileNotFoundError(f"{name} needs shards {missing}, which are not in {checkpoint}")
    sd = {}
    for file, names in by_file.items():
        with safe_open(checkpoint / file, framework="pt") as f:
            for full_name in names:
                sd[full_name[len(prefix) :]] = f.get_tensor(full_name)
    for key in [k for k in sd if k.endswith("wo_a.weight")]:
        w, s = sd[key], sd.pop(key.replace("weight", "scale"))
        bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
        w = w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float() * s[:, None, :, None].float()
        sd[key] = w.flatten(2, 3).flatten(0, 1).bfloat16()
    for key in sd:
        if ".experts." in key and "shared" not in key and sd[key].dtype == torch.int8:
            sd[key] = sd[key].view(torch.float4_e2m1fn_x2)
    if name.startswith("mtp."):
        sd.pop("embed.weight", None), sd.pop("head.weight", None)  # tied to the backbone's (convert.py)
    if getattr(getattr(module, "ffn", None), "gate", None) is not None and module.ffn.gate.bias_vl is None:
        sd.pop("ffn.gate.bias_vl", None)  # the VL routing bias only acts on image tokens (text-only oracle)
    module.load_state_dict(sd, strict=True)  # copy_: converts e.g. bf16 head -> fp32 like load_model


def synthetic_engram_rows(
    seed: int, layer_id: int, rows: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows ``rows`` of the synthetic Engram table of checkpoint layer ``layer_id``, in the table format
    (float8_e4m3fn [n, head_dim], E8M0 [n, head_dim/32]). Each row is N(0, 1) (as testing.init_weights
    draws tables) from its own generator, so it depends only on (seed, layer, row)."""
    w = torch.empty(len(rows), head_dim, dtype=torch.float32)  # independent of the caller's default dtype
    for i, r in enumerate(rows.tolist()):
        row_seed = _unit_seed(seed, f"layers.{layer_id}.engram.embed.{r}")
        torch.randn(head_dim, generator=torch.Generator().manual_seed(row_seed), out=w[i])
    return quantize_fp8_rows(w)


def _make_engram_row_sparse(engram: v41.Engram) -> None:
    """Replace the table by an empty one; ``load_engram_rows`` fills the rows a prompt needs, and a
    pre-hook maps global hash ids onto them before the reference lookup runs."""
    emb = engram.embed
    emb.weight = torch.nn.Parameter(torch.empty(0, emb.dim, dtype=torch.float8_e4m3fn), requires_grad=False)
    emb.scale = torch.nn.Parameter(
        torch.empty(0, emb.dim // emb.block_size, dtype=v41.scale_dtype), requires_grad=False
    )
    emb.oracle_rows = torch.empty(0, dtype=torch.int64)

    def remap(mod, args):
        ids = args[0]
        local = torch.searchsorted(mod.oracle_rows, ids.contiguous()).clamp_max(max(len(mod.oracle_rows) - 1, 0))
        if len(mod.oracle_rows) == 0 or not torch.equal(mod.oracle_rows[local], ids):
            raise RuntimeError("Engram rows were not loaded for this prompt (call load_engram_rows first)")
        return (local,)

    emb.register_forward_pre_hook(remap)


def load_engram_rows(model: v41.Transformer, spec: OracleSpec, tokens: torch.Tensor) -> None:
    """Materialize the synthetic Engram rows ``tokens`` hash to (before the prefill runs)."""
    if model.engram_hash is None:
        return
    with torch.inference_mode():
        hashes = model.engram_hash(tokens, 0)  # [B, S, n_engram_layers, n_hash_cols], deterministic
    for pos, layer in enumerate(model.layers):
        if layer.engram is None:
            continue
        emb = layer.engram.embed
        rows = torch.unique(hashes[:, :, layer.engram.layer_hash_index].flatten().clone())
        q, s = synthetic_engram_rows(spec.seed, spec.layer_ids[pos], rows, emb.dim)
        emb.weight = torch.nn.Parameter(q, requires_grad=False)
        emb.scale = torch.nn.Parameter(s, requires_grad=False)
        emb.vocab_end_idx = len(rows)
        emb.oracle_rows = rows


def _align_engram_hash(hash_state: v41.NgramHashState, spec: OracleSpec) -> None:
    """Give a schedule's Engram layers the hash of their checkpoint layers: primes, offsets and
    multipliers come from the full model's layout, selected for this schedule's Engram layers."""
    full = EngramLayout.from_args(spec.model_args)
    local = [spec.layer_ids[p] for p in spec.args.engram_layer_ids]
    sel = [full.layer_ids.index(layer) for layer in local]
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in full.primes]
    offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
    multipliers = compute_hash_multipliers(
        full.layer_ids, full.max_ngram_size, spec.model_args.engram_compressed_vocab_size
    )
    hash_state.primes = torch.tensor(full.primes)[sel]
    hash_state.offsets = torch.tensor(offsets)[sel]
    hash_state.multipliers = multipliers[sel]


def build_reference(spec: OracleSpec) -> v41.Transformer:
    """The reference model for ``spec`` with synthetic (seeded, disk-cached) or checkpoint weights."""
    with v41.set_dtype(torch.bfloat16):
        model = v41.Transformer(spec.args, _tokenizer(spec))
    if model.engram_hash is not None:
        _align_engram_hash(model.engram_hash, spec)
    for layer in model.layers:
        if layer.engram is not None:
            _make_engram_row_sparse(layer.engram)
    for name, module in _units(model, spec):
        with _Detached(module) as unit:
            if spec.checkpoint is None:
                _synthetic_unit(name, unit, spec.seed)
            else:
                _checkpoint_unit(name, unit, spec.checkpoint)
    return model.eval()


# --------------------------------------------------------------------------------------------- oracle


def _reset_state(model: v41.Transformer) -> None:
    """Fresh-request state, so ring slots / carries a prompt does not write are defined (0, -inf)."""
    for mod in model.modules():
        for name in ("window_kv_cache", "compress_kv_cache", "kv_state"):
            if isinstance(getattr(mod, name, None), torch.Tensor):
                getattr(mod, name).zero_()
        if isinstance(mod, v41.Compressor) and hasattr(mod, "score_state"):
            mod.score_state.fill_(-torch.inf)
        if isinstance(mod, v41.Indexer) and mod.owns_k:
            mod.k_cache.zero_()
    v41.shared_attn.__init__()


def _meta(spec: OracleSpec) -> dict:
    a, ids = spec.args, spec.layer_ids
    return {
        "layer_ids": list(ids),
        "compress_ratio": {ids[p]: a.compress_ratios[p] for p in range(a.n_layers)},
        "window": a.window_size,
        "kv_sources": [ids[p] for p in a.kv_source_layers],
        "index_sources": [ids[p] for p in a.index_source_layers],
        "candidate_source": ids[a.candidate_source_layer] if a.candidate_source_layer >= 0 else None,
        "engram_layers": [ids[p] for p in a.engram_layer_ids],
    }


@torch.no_grad()
def _run(model: v41.Transformer, spec: OracleSpec, tokens: torch.Tensor) -> dict:
    """Single-shot prefill of ``tokens`` [1, S] with block / shared-state / final-state captures."""
    seq = tokens.size(1)
    _reset_state(model)
    if spec.checkpoint is None:
        load_engram_rows(model, spec, tokens)
    blocks: dict = {lid: {} for lid in spec.layer_ids}
    shared: dict = {}
    hooks = []

    def add(mod, pre=None, post=None):
        if pre:
            hooks.append(mod.register_forward_pre_hook(pre))
        if post:
            hooks.append(mod.register_forward_hook(post))

    for pos, layer in enumerate(model.layers):
        lid, attn, rec = spec.layer_ids[pos], layer.attn, blocks[spec.layer_ids[pos]]
        stash = {}

        def block_pre(mod, args, rec=rec):
            rec["x_in"], rec["pre_in"] = args[0][0].clone(), args[2][0].clone()

        def block_post(mod, args, out, rec=rec):
            rec["x_out"], rec["pre_out"] = out[0][0].clone(), out[1][0].clone()

        def attn_pre(mod, args, rec=rec):
            rec["attn_in"] = args[0][0].clone()

        def kv_norm_post(mod, args, out, stash=stash):
            stash["kv"] = out  # RoPE and FP8 QDQ then run in place on this tensor

        def attn_post(mod, args, out, rec=rec, stash=stash, lid=lid):
            rec["attn_out"] = out[0].clone()
            rec["window_kv"] = stash.pop("kv")[0].clone()
            ratio, pub = mod.compress_ratio, {}
            if mod.is_kv_source:
                pub["compress_kv"] = mod.compress_kv_cache[0, : seq // ratio].clone()
                pub["index_k"] = mod.indexer.k_cache[0, : seq // ratio].clone()
            if mod.is_index_source:
                raw = v41.shared_attn.topk_idxs[0]
                pub["topk_idxs"] = torch.where(raw >= 0, raw - seq, -1).int()  # window offset = S
                if mod.indexer.is_candidate_source:
                    pub["candidates"] = v41.shared_attn.candidates[0].clone()
            if pub:
                shared[lid] = pub

        def ffn_pre(mod, args, rec=rec):
            rec["ffn_in"] = args[0][0].clone()

        def ffn_post(mod, args, out, rec=rec):
            rec["ffn_out"] = out[0].clone()

        def engram_pre(mod, args, rec=rec):
            rec["engram_in"], rec["engram_hash_ids"] = args[0][0].clone(), args[1][0].clone()

        add(layer, block_pre, block_post)
        add(attn, attn_pre, attn_post)
        add(attn.kv_norm, post=kv_norm_post)
        add(layer.ffn, ffn_pre, ffn_post)
        if layer.engram is not None:
            add(layer.engram, pre=engram_pre)
    try:
        logits, main_hidden = prefill(model, tokens)
    finally:
        for h in hooks:
            h.remove()

    state: dict = {"window": {}, "carry": {}}
    for pos, layer in enumerate(model.layers):
        lid, attn = spec.layer_ids[pos], layer.attn
        state["window"][lid] = attn.window_kv_cache[0].clone()
        if attn.compressor is not None and attn.compress_ratio > 1:
            state["carry"][lid] = {
                "kv": attn.compressor.kv_state[0].clone(),
                "score": attn.compressor.score_state[0].clone(),
            }
    if model.engram_hash is not None:
        state["engram_tokens"] = model.engram_hash.cache[0, :seq].clone()
    if len(model.mtp):
        state["main_hidden"] = main_hidden[0].clone()
        state["dspark_window"] = {k: blk.attn.window_kv_cache[0].clone() for k, blk in enumerate(model.mtp)}
    return {
        "tokens": tokens[0].clone(),
        "logits": logits[0].float(),
        "meta": _meta(spec) | {"seq_len": seq},
        "blocks": blocks,
        "shared": shared,
        "state": state,
    }


def cache_path(spec: OracleSpec, tokens: torch.Tensor) -> Path:
    """Disk location of the result for (spec, tokens); see the module docstring for the key."""
    key = _digest(
        asdict(spec.args),
        asdict(spec.model_args),
        list(spec.layer_ids),
        spec.seed,
        _source_ids(spec),
        hashlib.sha256(tokens.to(torch.int64).contiguous().numpy().tobytes()).hexdigest(),
        list(tokens.shape),
        _reference_digest(synthetic=spec.checkpoint is None),
        torch.__version__,
        PACKAGE_VERSION,
    )
    return CACHE_DIR / f"oracle-{key}.pt"


def oracle(spec: OracleSpec, tokens: torch.Tensor, model: v41.Transformer | None = None) -> dict:
    """Single-shot expected results for ``tokens`` [1, S] (S <= spec.args.max_seq_len), from the disk
    cache or computed (with ``model``, else a freshly built reference) and stored."""
    if tokens.dim() != 2 or tokens.size(0) != 1 or not 0 < tokens.size(1) <= spec.args.max_seq_len:
        raise ValueError(f"tokens must be [1, S] with 0 < S <= {spec.args.max_seq_len}, got {list(tokens.shape)}")
    path = cache_path(spec, tokens)
    if path.is_file():
        return torch.load(path)
    result = _run(model if model is not None else build_reference(spec), spec, tokens)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(result, tmp)
    tmp.replace(path)
    return result


@torch.no_grad()
def tail_logits(
    spec: OracleSpec,
    tokens: torch.Tensor,
    count: int,
    model: v41.Transformer | None = None,
    noise: tuple[float, float, int] | None = None,
) -> torch.Tensor:
    """fp32 logits ``[count, vocab]`` of the last ``count`` positions of a single-shot prefill of ``tokens``
    [1, S]: a prompt is teacher-forced by construction, so these are next-token predictions at ``count``
    positions. ``noise = (output_rel, input_rel, seed)`` adds Gaussian noise of RMS ``output_rel`` x the output's
    RMS to the output of every backbone attention and MoE (component-level error) and of RMS ``input_rel`` x the
    input's RMS to every attention and MoE input (the input-level sensitivity that flips top-k selection and
    expert routing): the floor an implementation whose components meet their bars is gated against. Cached next
    to the ``oracle`` result of the same (spec, tokens), keyed additionally by ``count`` and ``noise``."""
    if not 0 < count <= tokens.size(1):
        raise ValueError(f"count must be in (0, {tokens.size(1)}], got {count}")
    base = cache_path(spec, tokens)
    tag = f"-tail{count}" + (f"-noise-out{noise[0]:g}-in{noise[1]:g}-{noise[2]}" if noise else "")
    path = base.with_name(base.stem + tag + ".pt")
    if path.is_file():
        return torch.load(path)
    model = model if model is not None else build_reference(spec)
    _reset_state(model)
    if spec.checkpoint is None:
        load_engram_rows(model, spec, tokens)
    captured = {}

    def capture(mod, args):  # the backbone's head call comes first (DSpark calls the head again afterwards)
        captured.setdefault("x", args[0][0, -count:].clone())

    hooks = [model.head.register_forward_pre_hook(capture)]
    if noise is not None:
        output_rel, input_rel, seed = noise
        gen = torch.Generator().manual_seed(seed)

        def perturbed(t, rel):
            return (t.float() + torch.randn(t.shape, generator=gen) * rel * t.float().pow(2).mean().sqrt()).to(t.dtype)

        def perturb_output(mod, args, out):
            return perturbed(out, output_rel)

        def perturb_input(mod, args):
            return (perturbed(args[0], input_rel), *args[1:])

        for layer in model.layers:
            for m in (layer.attn, layer.ffn):
                hooks += [m.register_forward_pre_hook(perturb_input), m.register_forward_hook(perturb_output)]
    try:
        prefill(model, tokens)
    finally:
        for h in hooks:
            h.remove()
    # one row at a time, as the reference head projects the last position (bit-identical to its logits)
    weight = model.head.weight.float()
    logits = torch.cat([torch.nn.functional.linear(row[None].float(), weight) for row in captured["x"]])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(logits, tmp)
    tmp.replace(path)
    return logits


# ------------------------------------------------------------------------------------ chunk contract


def window_ring(window_kv: torch.Tensor, end: int, window: int) -> torch.Tensor:
    """The window ring after positions [0, end): slot p % window holds position p's window KV for the
    last ``window`` positions; slots never written stay 0 (graph.md §4 rule 1)."""
    ring = window_kv.new_zeros(window, window_kv.size(-1))
    pos = torch.arange(max(0, end - window), end)
    ring[pos % window] = window_kv[pos]
    return ring


def chunk_expectations(result: dict, chunk_len: int, pad_multiple: int) -> list[dict]:
    """What a chunked device prefill must match, per chunk, given the single-shot ``result``.

    Chunk c covers valid tokens [start, start + length) and runs padded to ``padded_length`` (a
    multiple of ``pad_multiple``, e.g. 2*32*SP); positions >= length write nothing (rule 8). Per chunk:
    ``blocks[id]`` per-token captures sliced to the valid tokens; ``rows[src]`` the compressed-KV and
    index-K rows [row_start, row_end) = [start//r, (start+length)//r) that the chunk writes, and nothing
    beyond row_end; ``topk_idxs[id]`` / ``candidates[id]`` rows of the chunk's queries (candidates over
    the row_end visible columns; top-k: compare the sets of valid ids, the single-shot width is
    min(topk, S//r)); ``window[id]`` the ring after the chunk; ``engram_history`` the compressed ids of
    the last 3 real tokens. The last chunk adds ``logits`` and the final ``state`` (carries, rings,
    DSpark), which equal the single-shot ones.
    """
    meta, seq = result["meta"], result["meta"]["seq_len"]
    ratios = [r for r in meta["compress_ratio"].values() if r]
    if pad_multiple <= 0 or chunk_len % pad_multiple or any(pad_multiple % r for r in ratios):
        raise ValueError(
            f"chunk_len ({chunk_len}) must be a multiple of pad_multiple ({pad_multiple}), itself a multiple "
            f"of every compress ratio {sorted(set(ratios))}"
        )
    chunks = []
    for index, start in enumerate(range(0, seq, chunk_len)):
        end = min(start + chunk_len, seq)
        length = end - start
        c = {
            "index": index,
            "start": start,
            "length": length,
            "padded_length": -(-length // pad_multiple) * pad_multiple,
            "blocks": {
                lid: {k: v[start:end] for k, v in rec.items() if k in _TOKEN_KEYS}
                for lid, rec in result["blocks"].items()
            },
            "rows": {},
            "topk_idxs": {},
            "candidates": {},
            "window": {
                lid: window_ring(rec["window_kv"], end, meta["window"]) for lid, rec in result["blocks"].items()
            },
        }
        for lid, pub in result["shared"].items():
            r = meta["compress_ratio"][lid]
            if "compress_kv" in pub:
                lo, hi = start // r, end // r
                c["rows"][lid] = {
                    "row_start": lo,
                    "row_end": hi,
                    "compress_kv": pub["compress_kv"][lo:hi],
                    "index_k": pub["index_k"][lo:hi],
                }
            if "topk_idxs" in pub:
                c["topk_idxs"][lid] = pub["topk_idxs"][start:end]
            if "candidates" in pub:
                c["candidates"][lid] = pub["candidates"][start:end, : end // r]
        if "engram_tokens" in result["state"]:
            c["engram_history"] = result["state"]["engram_tokens"][max(0, end - 3) : end]
        if end == seq:
            c["logits"], c["state"] = result["logits"], result["state"]
        chunks.append(c)
    return chunks


# ----------------------------------------------------------------------------------- vision (ViT + aligner)
#
# ``vision_oracle`` runs the reference image encoder (``Transformer.encode_image``: ``vision.ViT`` then
# ``vision.Aligner``) on one image's patches. Weights are synthetic (per-tensor seeded) or the checkpoint's
# ``vision.*`` / ``aligner.*`` tensors (shard 1, stored bf16, used as stored). Results are cached as
# ``vision-*.pt``, keyed by the vision args, weights source, a sha256 of the patches, the grid, a digest of
# ``vision.py``, the torch version and ``PACKAGE_VERSION``.

_VISION_SOURCES = ("vision.py",)


def vision_args() -> v41.ModelArgs:
    """The released V4.1-Flash args (vendored ``config.json``) with the vision tower enabled."""
    cfg = {
        k: tuple(v) if isinstance(v, list) else v for k, v in json.loads((_HERE / "config.json").read_text()).items()
    }
    return v41.ModelArgs(**cfg)


def _vision_modules(args: v41.ModelArgs):
    from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import vision

    with v41.set_dtype(torch.bfloat16):  # as upstream builds the model (norm weights stay fp32)
        return vision.ViT(args), vision.Aligner(args)


def vision_weight_shapes(args: v41.ModelArgs) -> dict[str, torch.Size]:
    """Checkpoint name -> shape of every ViT / aligner tensor, from the reference modules."""
    vit, aligner = _vision_modules(args)
    return {f"{p}{k}": v.shape for p, m in (("vision.", vit), ("aligner.", aligner)) for k, v in m.state_dict().items()}


def synthetic_vision_weights(args: v41.ModelArgs, seed: int = 0) -> dict[str, torch.Tensor]:
    """bf16 ViT / aligner weights by checkpoint name: norms 1 + 0.1 N(0,1), biases 0.02 N(0,1), matrices
    N(0,1) / sqrt(fan_in). Each tensor is drawn from its own seed (seed, name), so it does not depend on
    the other tensors or on ``vision_n_layers``."""
    out = {}
    for name, shape in vision_weight_shapes(args).items():
        t = torch.randn(shape, generator=torch.Generator().manual_seed(_unit_seed(seed, name)), dtype=torch.float32)
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("norm.weight"):
            t = 1 + 0.1 * t
        elif name.endswith(".bias"):
            t = 0.02 * t
        else:
            t = t * shape[-1] ** -0.5
        out[name] = t.to(torch.bfloat16)
    return out


def checkpoint_vision_weights(checkpoint: Path = HF_SNAPSHOT) -> dict[str, torch.Tensor]:
    """The checkpoint's ViT / aligner tensors by name, as stored."""
    from safetensors import safe_open

    names = vision_weight_shapes(vision_args())
    weight_map = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    if missing := sorted(n for n in names if n not in weight_map):
        raise KeyError(f"{checkpoint} has no tensors {missing[:3]}...")
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    if absent := sorted(f for f in by_file if not (Path(checkpoint) / f).is_file()):
        raise FileNotFoundError(f"vision weights need shards {absent}, which are not in {checkpoint}")
    out = {}
    for file, file_names in by_file.items():
        with safe_open(Path(checkpoint) / file, framework="pt") as f:
            out |= {n: f.get_tensor(n) for n in file_names}
    return out


def build_vision_reference(args: v41.ModelArgs, weights: dict[str, torch.Tensor]):
    """Reference ``(ViT, Aligner)`` holding ``weights`` (by checkpoint name), converted to the modules'
    dtypes as upstream ``load_state_dict`` does."""
    vit, aligner = _vision_modules(args)
    for prefix, module in (("vision.", vit), ("aligner.", aligner)):
        module.load_state_dict({k: weights[prefix + k].to(v.dtype) for k, v in module.state_dict().items()})
    return vit.eval(), aligner.eval()


def synthetic_image(width: int, height: int, seed: int = 0) -> bytes:
    """A deterministic PNG with structure (colour gradients, a disc) plus mild noise."""
    import io

    from PIL import Image

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    disc = np.hypot(x - width * 0.4, y - height * 0.6) < min(width, height) * 0.25
    img = np.stack([255 * x / width, 255 * y / height, 128 + 100 * disc], axis=-1)
    img += torch.randn(img.shape, generator=torch.Generator().manual_seed(seed)).numpy() * 8
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def image_patches(image: bytes, args: v41.ModelArgs | None = None) -> tuple[torch.Tensor, int, int]:
    """``(patches [n_h*n_w, 3, p, p] bf16, n_h, n_w)`` of an encoded image, via ``image_processor.load_image``."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import image_processor

    patches, n_h, n_w, _, _ = image_processor.load_image({"data": image}, args or vision_args())
    return patches, n_h, n_w


def random_patches(n_h: int, n_w: int, args: v41.ModelArgs | None = None, seed: int = 0) -> torch.Tensor:
    """``[n_h*n_w, 3, p, p]`` bf16 N(0, 1) patches of a direct ``n_h x n_w`` grid."""
    p = (args or vision_args()).vision_patch_size
    return torch.randn(n_h * n_w, 3, p, p, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _vision_digest() -> str:
    h = hashlib.sha256()
    for name in _VISION_SOURCES:
        h.update((_HERE / name).read_bytes())
    return h.hexdigest()[:16]


def vision_cache_path(
    args: v41.ModelArgs, patches: torch.Tensor, n_h: int, n_w: int, seed: int, checkpoint: Path | None
) -> Path:
    fields = {k: v for k, v in asdict(args).items() if k.startswith("vision_") or k == "dim"}
    key = _digest(
        "vision",
        fields,
        "synthetic" if checkpoint is None else f"hf:{Path(checkpoint).name}",
        seed if checkpoint is None else None,
        hashlib.sha256(patches.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        [str(patches.dtype), *patches.shape],
        n_h,
        n_w,
        _vision_digest(),
        torch.__version__,
        PACKAGE_VERSION,
    )
    return CACHE_DIR / f"vision-{key}.pt"


@torch.no_grad()
def vision_oracle(
    patches: torch.Tensor,
    n_h: int,
    n_w: int,
    *,
    args: v41.ModelArgs | None = None,
    seed: int = 0,
    checkpoint: Path | None = None,
) -> dict:
    """Expected image encoder outputs for ``patches`` [n_h*n_w, 3, p, p] of an ``n_h x n_w`` grid, from the
    disk cache or computed and stored: ``hidden`` [n_h*n_w, vision_dim] bf16 (ViT output, row-major patch
    order), ``aligned`` [ceil(n_h/r)*ceil(n_w/r), dim] bf16 (aligner output, reading order; the grid is
    zero-padded to multiples of r), ``meta``. Weights are the synthetic ones of ``seed`` or, with
    ``checkpoint``, the stored ones."""
    args = args or vision_args()
    if patches.dim() != 4 or patches.size(0) != n_h * n_w or n_h <= 0 or n_w <= 0:
        raise ValueError(f"patches must be [n_h*n_w = {n_h * n_w}, 3, p, p], got {list(patches.shape)}")
    path = vision_cache_path(args, patches, n_h, n_w, seed, checkpoint)
    if path.is_file():
        return torch.load(path)
    weights = synthetic_vision_weights(args, seed) if checkpoint is None else checkpoint_vision_weights(checkpoint)
    vit, aligner = build_vision_reference(args, weights)
    with v41.set_dtype(torch.bfloat16):
        hidden = vit(patches, n_h, n_w)
        aligned = aligner(hidden, n_h, n_w)
    r = args.vision_downsample_ratio
    result = {
        "hidden": hidden.clone(),
        "aligned": aligned.clone(),
        "meta": {"n_h": n_h, "n_w": n_w, "n_llm_h": -(-n_h // r), "n_llm_w": -(-n_w // r)},
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(result, tmp)
    tmp.replace(path)
    return result
