"""Weights plumbing for the TT side: the HF "view" directory tt_transformers.ModelArgs reads (language_model config +
tokenizer files + safetensors symlinks), the sliced LM head / embedding tables, and the depth/DiT/cond/vocoder loaders.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import torch

from models.autoports.minimaxai_minimax_music3.config import (
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    HF_REVISION,
    SEMANTIC_VOCAB_SIZE,
    SLICED_VOCAB,
    SLICED_VOCAB_PADDED,
)


def cache_root() -> Path:
    p = Path(os.environ.get("TT_DIT_CACHE_DIR", Path.home() / ".cache" / "minimax_music3"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_hf_view(snapshot: os.PathLike, view_dir: os.PathLike | None = None) -> Path:
    """A directory ModelArgs(HF_MODEL=...) can read: language_model/config.json (+ generation_config), the shards and
    index (symlinked), and the tokenizer files from tokenizer/. Named "MiniMax-Music3-LM" so base_model_name is stable.
    """
    snapshot = Path(snapshot)
    view_dir = Path(view_dir or os.environ.get("MUSIC3_HF_VIEW") or (cache_root() / "hf_view" / "MiniMax-Music3-LM"))
    view_dir.mkdir(parents=True, exist_ok=True)
    lm = snapshot / "language_model"
    for src in sorted(lm.iterdir()):
        dst = view_dir / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                dst.write_bytes(src.read_bytes())
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        src, dst = snapshot / "tokenizer" / name, view_dir / name
        if src.exists() and not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                dst.write_bytes(src.read_bytes())
    cfg = json.load(open(view_dir / "config.json"))
    assert cfg["model_type"] == "qwen3" and cfg["vocab_size"] > AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE, cfg.get(
        "vocab_size"
    )
    return view_dir


def sliced_rows(full: torch.Tensor) -> torch.Tensor:
    """[V, D] -> [SLICED_VOCAB_PADDED, D]: rows 0..16383 = semantic codes, 16384 = end-of-audio, rest zero."""
    out = torch.zeros(SLICED_VOCAB_PADDED, full.shape[1], dtype=full.dtype)
    out[:SEMANTIC_VOCAB_SIZE] = full[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE]
    out[SEMANTIC_VOCAB_SIZE] = full[AUDIO_END_TOKEN_ID]
    return out


def load_lm_tables(snapshot: os.PathLike) -> Dict[str, torch.Tensor]:
    """embed_tokens + lm_head (bf16, full vocab) straight from the shards, without loading the 36 layers."""
    from safetensors import safe_open

    idx = json.load(open(Path(snapshot) / "language_model" / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(str(Path(snapshot) / "language_model" / idx[key]), framework="pt", device="cpu") as f:
            out[key] = f.get_tensor(key)
    return out


def load_depth_state(snapshot: os.PathLike) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(Path(snapshot) / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors"), device="cpu")


def load_dit_state(snapshot: os.PathLike, num_layers: int | None = None) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    sd = {}
    for shard in sorted(Path(snapshot, "transformer").glob("*.safetensors")):
        sd.update(load_file(str(shard), device="cpu"))
    if num_layers is not None:
        sd = {
            k: v for k, v in sd.items() if not k.startswith("transformer_blocks.") or int(k.split(".")[1]) < num_layers
        }
    return sd


def weight_report(snapshot: os.PathLike) -> dict:
    """Stage-02 evidence: per-component tensor counts, dtypes, parameter totals."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    rep = {"snapshot": str(snapshot), "components": {}}
    for sub in ("language_model", "transformer", "rvq_depth_decoder", "condition_encoder", "vocoder"):
        n, params, dtypes, bytes_ = 0, 0, {}, 0
        for shard in sorted((snapshot / sub).glob("*.safetensors")):
            bytes_ += os.path.getsize(os.path.realpath(shard))
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    sl = f.get_slice(k)
                    shape = sl.get_shape()
                    n += 1
                    numel = 1
                    for s in shape:
                        numel *= s
                    params += numel
                    dtypes[sl.get_dtype()] = dtypes.get(sl.get_dtype(), 0) + 1
        rep["components"][sub] = {"tensors": n, "params": params, "dtypes": dtypes, "bytes": bytes_}
    tables = load_lm_tables(snapshot)
    rep["lm_head_shape"] = list(tables["lm_head.weight"].shape)
    rep["embed_shape"] = list(tables["model.embed_tokens.weight"].shape)
    rep["tied"] = bool(torch.equal(tables["lm_head.weight"][:4096], tables["model.embed_tokens.weight"][:4096]))
    rep["sliced_head_rows"] = SLICED_VOCAB
    rep["revision"] = HF_REVISION
    return rep


# ---------------------------------------------------------------------------------------------------------------
# The SLICED HF view: a real HF model directory whose embed_tokens / lm_head carry only the 16 385 rows the c0 head
# can ever un-mask (padded to 16 416), so tt_transformers' standard HF path builds correctly-sized device tables.
# Written once (stage 02) into the cache root; shards that do not hold the two tables are symlinked.
# ---------------------------------------------------------------------------------------------------------------
SLICED_VIEW_NAME = "MiniMax-Music3-LM-sliced"
_TABLES = ("model.embed_tokens.weight", "lm_head.weight")


def sliced_view_dir() -> Path:
    return Path(os.environ.get("MUSIC3_HF_VIEW") or (cache_root() / "hf_view" / SLICED_VIEW_NAME))


def ensure_sliced_view(snapshot: os.PathLike, view_dir: os.PathLike | None = None, log=print) -> Path:
    from safetensors import safe_open
    from safetensors.torch import save_file

    snapshot = Path(snapshot)
    view = Path(view_dir or sliced_view_dir())
    marker = view / ".sliced_view_complete"
    if marker.exists() and (view / "model.safetensors.index.json").exists():
        return view
    view.mkdir(parents=True, exist_ok=True)
    lm = snapshot / "language_model"
    idx = json.load(open(lm / "model.safetensors.index.json"))
    wm: Dict[str, str] = dict(idx["weight_map"])
    table_shards = {wm[k] for k in _TABLES}
    new_map: Dict[str, str] = {}
    for shard in sorted(set(wm.values())):
        dst = view / shard
        if shard in table_shards:
            log(f"sliced view: rewriting {shard} without the vocab tables")
            with safe_open(str(lm / shard), framework="pt", device="cpu") as f:
                keep = {k: f.get_tensor(k) for k in f.keys() if k not in _TABLES}
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            save_file(keep, str(dst), metadata={"format": "pt"})
            for k in keep:
                new_map[k] = shard
        else:
            if not dst.exists():
                dst.symlink_to((lm / shard).resolve())
            for k, v in wm.items():
                if v == shard:
                    new_map[k] = shard
    tables = load_lm_tables(snapshot)
    sliced = {k: sliced_rows(tables[k]).contiguous() for k in _TABLES}
    save_file(sliced, str(view / "model-sliced.safetensors"), metadata={"format": "pt"})
    for k in _TABLES:
        new_map[k] = "model-sliced.safetensors"
    total = sum(os.path.getsize(os.path.realpath(view / s)) for s in set(new_map.values()))
    json.dump(
        {
            "metadata": {"total_size": total, "sliced_vocab": SLICED_VOCAB, "source_revision": HF_REVISION},
            "weight_map": new_map,
        },
        open(view / "model.safetensors.index.json", "w"),
        indent=1,
    )
    cfg = json.load(open(lm / "config.json"))
    cfg["vocab_size"] = SLICED_VOCAB_PADDED
    cfg["_music3_sliced_vocab"] = {
        "rows": SLICED_VOCAB,
        "semantic_offset": AUDIO_CODE_OFFSET,
        "end_token_id": AUDIO_END_TOKEN_ID,
        "end_row": SEMANTIC_VOCAB_SIZE,
    }
    json.dump(cfg, open(view / "config.json", "w"), indent=2)
    gen = lm / "generation_config.json"
    (view / "generation_config.json").write_text(gen.read_text() if gen.exists() else "{}")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        src, dst = snapshot / "tokenizer" / name, view / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    marker.write_text("ok\n")
    log(f"sliced view ready: {view}")
    return view
