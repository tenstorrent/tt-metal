"""Weights for Fish S2 Pro on TT: HF fish-speech checkpoint -> tt_transformers state dict + fast-decoder
dict + codebook table + codec state, and the synthesized HF "view" directory ModelArgs needs.

Key facts (verified, see doc/): the checkpoint ships 358 tensors under `text_model.model.*` (slow tower)
and `audio_decoder.*` (fast tower + the SLOW tower's codebook input table). Attention is stored fused as
`wqkv` [q_dim+2*kv_dim, dim]; tt_transformers' Attention wants wq/wk/wv split. The LM head is TIED to the
input embedding (no lm_head tensor). Fish RoPE is the interleaved-pair (Meta) convention, i.e. exactly
tt_transformers' default RotarySetup: NO reverse_permute, NO use_hf_rope.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from models.autoports.fishaudio_s2_pro.config import HF_REPO_ID, HF_REVISION, S2Config

SLOW_PREFIX = "text_model.model."
FAST_PREFIX = "audio_decoder."
CODEBOOK_KEY = "audio_decoder.codebook_embeddings.weight"
STALE_CODEC_SUFFIXES = ("causal_mask", "freqs_cis")


def resolve_snapshot(repo_id: str = None, revision: str = None, allow_download: bool = True) -> Path:
    """Locate the HF snapshot dir. Offline-first (the tt-model container mounts the host HF cache)."""
    repo_id = repo_id or os.environ.get("HF_MODEL") or HF_REPO_ID
    if os.path.isdir(repo_id):
        return Path(repo_id)
    revision = revision or os.environ.get("FISH_S2_WEIGHTS_REVISION") or HF_REVISION
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, revision=revision, local_files_only=True))
    except Exception:
        if not allow_download:
            raise
        return Path(snapshot_download(repo_id, revision=revision))


def load_fish_state_dict(snapshot: os.PathLike, device="cpu") -> Dict[str, torch.Tensor]:
    """All LM tensors (bf16) from the safetensors shards, keyed by their HF names."""
    from safetensors.torch import load_file

    sd: Dict[str, torch.Tensor] = {}
    for shard in sorted(Path(snapshot).glob("model-*.safetensors")):
        sd.update(load_file(str(shard), device=device))
    if not sd:
        raise FileNotFoundError(f"no model-*.safetensors under {snapshot}")
    return sd


def split_wqkv(w: torch.Tensor, cfg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = torch.split(w, [cfg.q_dim, cfg.kv_dim, cfg.kv_dim], dim=0)
    return q.contiguous(), k.contiguous(), v.contiguous()


def slow_tower_state_dict(sd: Dict[str, torch.Tensor], cfg: S2Config) -> Dict[str, torch.Tensor]:
    """tt_transformers internal ("meta") naming for the slow tower.

    text_model.model.embeddings.weight -> tok_embeddings.weight AND output.weight (tied)
    text_model.model.layers.N.attention.wqkv.weight -> layers.N.attention.{wq,wk,wv}.weight
    everything else: prefix stripped, names already match (attention.wo, q_norm, k_norm, attention_norm,
    ffn_norm, feed_forward.w1/w2/w3, norm).
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith(SLOW_PREFIX):
            continue
        nk = k[len(SLOW_PREFIX) :]
        if nk == "embeddings.weight":
            out["tok_embeddings.weight"] = v
            out["output.weight"] = v  # tied head; LMHead pads to padded_vocab_size itself
        elif nk.endswith(".attention.wqkv.weight"):
            base = nk[: -len("wqkv.weight")]
            q, kk, vv = split_wqkv(v, cfg.slow)
            out[base + "wq.weight"], out[base + "wk.weight"], out[base + "wv.weight"] = q, kk, vv
        else:
            out[nk] = v
    expected = (
        2 + 1 + cfg.slow.n_layer * 11
    )  # tok_embeddings+output, norm; per layer wq wk wv wo q_norm k_norm attention_norm ffn_norm w1 w2 w3
    assert len(out) == expected, f"slow tower: got {len(out)} keys, expected {expected}"
    return out


def fast_tower_state_dict(
    sd: Dict[str, torch.Tensor], cfg: S2Config, split_qkv: bool = True
) -> Dict[str, torch.Tensor]:
    """audio_decoder.* minus the codebook table, with wqkv optionally split. Keys:
    embeddings.weight, layers.N.attention.{wq,wk,wv|wqkv}.weight, layers.N.attention.wo.weight,
    layers.N.{attention_norm,ffn_norm}.weight, layers.N.feed_forward.{w1,w2,w3}.weight, norm.weight, output.weight"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not k.startswith(FAST_PREFIX) or k == CODEBOOK_KEY:
            continue
        nk = k[len(FAST_PREFIX) :]
        if split_qkv and nk.endswith(".attention.wqkv.weight"):
            base = nk[: -len("wqkv.weight")]
            q, kk, vv = split_wqkv(v, cfg.fast)
            out[base + "wq.weight"], out[base + "wk.weight"], out[base + "wv.weight"] = q, kk, vv
        else:
            out[nk] = v
    return out


def codebook_table(sd: Dict[str, torch.Tensor], cfg: S2Config) -> torch.Tensor:
    """[num_codebooks*codebook_size, dim] slow-input table; row = code_i + i*codebook_size."""
    t = sd[CODEBOOK_KEY]
    assert tuple(t.shape) == (cfg.num_codebooks * cfg.codebook_size, cfg.slow.dim), t.shape
    return t


def load_codec_state(snapshot: os.PathLike) -> Dict[str, torch.Tensor]:
    """codec.pth without the stale non-persistent buffers older code saved (~305 MB of masks/freqs)."""
    sd = torch.load(Path(snapshot) / "codec.pth", map_location="cpu", weights_only=True, mmap=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return {k: v for k, v in sd.items() if not k.endswith(STALE_CODEC_SUFFIXES)}


def ensure_hf_view(snapshot: os.PathLike, view_dir: os.PathLike = None) -> Path:
    """Directory tt_transformers.ModelArgs can read: a Qwen3-shaped config.json + the real tokenizer files.

    ModelArgs requires HF_MODEL to point at a dir with a transformers-parseable config and tokenizer files
    (model_config.py: _set_hf_params / create_tokenizer). The real config is model_type fish_qwen3_omni.
    """
    snapshot = Path(snapshot)
    view_dir = Path(
        view_dir
        or os.environ.get("FISH_S2_HF_VIEW")
        or (Path(os.environ.get("TT_DIT_CACHE_DIR", Path.home() / ".cache" / "fish_s2_pro")) / "hf_view" / "s2-pro")
    )
    view_dir.mkdir(parents=True, exist_ok=True)
    src_cfg = Path(__file__).resolve().parents[1] / "model_params" / "s2-pro" / "config.json"
    cfg = json.load(open(src_cfg))
    real = json.load(open(snapshot / "config.json"))
    tc = real["text_config"]
    # keep the view honest against the real snapshot
    assert (
        cfg["hidden_size"] == tc["dim"]
        and cfg["num_hidden_layers"] == tc["n_layer"]
        and cfg["vocab_size"] == tc["vocab_size"]
    )
    (view_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        s, d = snapshot / name, view_dir / name
        if s.exists() and not d.exists():
            try:
                d.symlink_to(s.resolve())
            except OSError:
                d.write_bytes(s.read_bytes())
    gen = view_dir / "generation_config.json"
    if not gen.exists():
        gen.write_text(
            json.dumps(
                {"eos_token_id": real.get("eos_token_id", 151645), "pad_token_id": real.get("pad_token_id", 151669)}
            )
        )
    return view_dir


def remap_report(snapshot: os.PathLike) -> dict:
    """Stage-02 evidence: counts, shapes, param totals."""
    cfg = S2Config.from_snapshot(snapshot)
    sd = load_fish_state_dict(snapshot)
    slow, fast, cb = slow_tower_state_dict(sd, cfg), fast_tower_state_dict(sd, cfg), codebook_table(sd, cfg)
    codec = load_codec_state(snapshot)
    n = lambda d: int(sum(v.numel() for v in d.values()))
    return {
        "hf_keys": len(sd),
        "hf_total_params": n(sd),
        "slow_keys": len(slow),
        "slow_params_incl_tied_head_once": n({k: v for k, v in slow.items() if k != "output.weight"}),
        "fast_keys": len(fast),
        "fast_params": n(fast),
        "codebook_table_shape": list(cb.shape),
        "codec_tensors": len(codec),
        "codec_params": n(codec),
        "slow_sample_keys": sorted(slow)[:12],
        "fast_sample_keys": sorted(fast)[:8],
        "config": cfg.as_dict(),
    }
