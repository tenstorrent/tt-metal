# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-text KDA layer inputs: English prose, tokenized, embedded and normalized for the layer under test.

Random hidden states keep every decay channel away from its extremes; trained KDA gates on real text reach
the bounded gate's cap (per-32-token-chunk |G_last| = 160 on Kimi-K3 at Pride and Prejudice tokens [776, 808),
tt_metal_tracker-g1b.4.10). A text input is the layer's ``input_layernorm`` applied to the token embeddings.
For the first decoder layer that is the exact layer input; for a later layer it omits the earlier layers'
residual contributions, so it is a real-text proxy, not the model's activation.

Building an input reads the pinned tokenizer and embedding rows from Hugging Face (range reads of the
embedding shard when it is not local), so it runs only in the CPU preparation step; device tests load the
cached tensor and fail fast on a miss.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.deepseek_v3_d_p.utils.oracle_cache import oracle_cache_root, publish_once

# Covers the stored text input: corpus body extraction, tokenizer construction, the window, the embedding
# lookup and the RMSNorm below. Bump when any of them changes the stored tensor; the cache is shared by every
# worktree (utils/oracle_cache.py), so an unmerged branch bumps to a value no other branch uses.
TEXT_INPUT_VERSION = 1

# Project Gutenberg #1342, Pride and Prejudice (the en-prose corpus of tt-work kda_decay_range_k3_glm.py).
CORPUS_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
CORPUS_SHA256 = "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"

# Kimi-K3 tokenizer split pattern (tokenization_kimi.py TikTokenTokenizer.pat_str at the pinned revision).
_KIMI_PATTERN = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


@dataclass(frozen=True)
class TextInputModel:
    """Where one model's tokenizer, embedding and layer norms come from."""

    repo: str
    revision: str
    model_root: str
    tokenizer: str
    # First corpus token of the window. Chosen so the model's strongest measured decay window lands on a
    # 32-token chunk boundary inside SP rank 1 (640 tokens per rank).
    window_start: int
    # GLM (mHC): the layer input is the input_layernorm of the attention hyper-connection collapse of
    # hc_mult copies of the embedding (the residual streams at layer 0).
    hyper_connection_streams: bool = False


TEXT_INPUT_MODELS = {
    # Pride and Prejudice tokens [776, 808) drive K3 heads 17/48 to |G_last| = 160 (layers.0 gate); start 8
    # puts them at positions [768, 800), an aligned chunk.
    "kimi_k3": TextInputModel(
        repo="moonshotai/Kimi-K3",
        revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        model_root="language_model.model.",
        tokenizer="tiktoken",
        window_start=8,
    ),
    # GLM layer 0 reaches per-chunk |G_last| 157.3 on the first 4096 en-prose tokens (g1b.4.10).
    "glm_5_3_flash": TextInputModel(
        repo="zai-org/GLM-5.3-Flash",
        revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        model_root="model.language_model.",
        tokenizer="tokenizers",
        window_start=0,
        hyper_connection_streams=True,
    ),
}


def text_input_cache_path(model: str, layer_idx: int, tokens: int) -> Path:
    """Shared oracle-cache path of one model/layer/length text input (keyed by the producer's full identity)."""
    spec = TEXT_INPUT_MODELS[model]
    name = (
        f"v{TEXT_INPUT_VERSION}-{spec.revision[:12]}-layer{layer_idx}-pg1342-{CORPUS_SHA256[:12]}"
        f"-start{spec.window_start}-tokens{tokens}.pt"
    )
    return oracle_cache_root() / model / "text_input" / name


def load_text_input(model: str, layer_idx: int, tokens: int) -> torch.Tensor | None:
    """Return the cached ``[1, tokens, hidden]`` bf16 text input, or None when it was not prepared."""
    path = text_input_cache_path(model, layer_idx, tokens)
    if not path.is_file():
        return None
    hidden = _load_checked(path)["hidden"]
    logger.info(f"text input {model} layer {layer_idx} T={tokens} cache hit: {path}")
    return hidden


def _load_checked(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert _sha256(payload["hidden"]) == payload["sha256"], f"text input checksum mismatch: {path}"
    return payload


def build_text_input(model: str, layer_idx: int, tokens: int, checkpoint_dir: Path) -> torch.Tensor:
    """Build, cache and return the text input (CPU preparation only: needs network for the tokenizer)."""
    path = text_input_cache_path(model, layer_idx, tokens)
    start = time.perf_counter()

    def produce() -> dict[str, torch.Tensor | str]:
        token_ids = _token_ids(model, tokens)
        hidden = _build_hidden(model, layer_idx, token_ids, checkpoint_dir)
        return {"hidden": hidden, "sha256": _sha256(hidden), "token_ids": torch.tensor(token_ids)}

    payload, produced = publish_once(path, produce, torch.save, _load_checked)
    verb = "built" if produced else "published by another producer, loaded"
    logger.info(
        f"text input {model} layer {layer_idx} T={tokens} {verb} in {time.perf_counter() - start:.1f} s: {path}"
    )
    return payload["hidden"]


def _sha256(hidden: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(hidden.contiguous().view(torch.uint8).numpy())).hexdigest()


def _token_ids(model: str, tokens: int) -> list[int]:
    spec = TEXT_INPUT_MODELS[model]
    token_ids = _encoder(spec)(corpus_body())[spec.window_start : spec.window_start + tokens]
    if len(token_ids) != tokens:
        raise ValueError(f"corpus has {len(token_ids)} tokens after {spec.window_start}, need {tokens}")
    return token_ids


def _build_hidden(model: str, layer_idx: int, token_ids: list[int], checkpoint_dir: Path) -> torch.Tensor:
    spec = TEXT_INPUT_MODELS[model]
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    embeddings = _embedding_rows(spec, checkpoint_dir, token_ids)
    layer_prefix = f"{spec.model_root}layers.{layer_idx}."
    norm_weight = input_norm_weight(model, layer_idx, checkpoint_dir)
    if spec.hyper_connection_streams:
        if layer_idx != 0:
            raise ValueError("hyper-connection text inputs are exact only for layer 0")
        embeddings = _hyper_connection_collapse(
            embeddings,
            *(
                _checkpoint_tensor(spec, checkpoint_dir, f"{layer_prefix}hc_attn_{name}")
                for name in ("fn", "base", "scale")
            ),
            text_config,
        )
    return _rms_norm(embeddings, norm_weight, text_config["rms_norm_eps"]).unsqueeze(0).contiguous()


def input_norm_weight(model: str, layer_idx: int, checkpoint_dir: Path) -> torch.Tensor:
    """The layer's ``input_layernorm`` weight ``w`` [hidden] (bf16) from the pinned checkpoint.

    Every input the layer receives in the model is ``w * u`` with per-token RMS(u) <= 1 (the normalized residual
    stream), so ``w`` bounds the reachable layer input.
    """
    spec = TEXT_INPUT_MODELS[model]
    return _checkpoint_tensor(spec, checkpoint_dir, f"{spec.model_root}layers.{layer_idx}.input_layernorm.weight")


def chunk_decay_extremes(hidden: torch.Tensor, weights: dict[str, torch.Tensor], config) -> dict[str, float]:
    """Largest per-32-token-chunk |G_last| of the layer gate on ``hidden`` and where it occurs."""
    from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_gate_reference

    x = hidden[0].float()
    raw = (x @ weights["f_a_proj.weight"].float().T) @ weights["f_b_proj.weight"].float().T
    gate = kda_gate_reference(
        raw.view(1, -1, config.num_heads, config.head_k_dim),
        weights["A_log"],
        weights["dt_bias"],
        config.gate_lower_bound,
    )[0]
    usable = gate.shape[0] // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    chunk = gate[:usable].reshape(-1, ttnn.TILE_SIZE, config.num_heads, config.head_k_dim).sum(1).abs()
    index = int(chunk.argmax())
    chunk_index, remainder = divmod(index, config.num_heads * config.head_k_dim)
    head, channel = divmod(remainder, config.head_k_dim)
    return {
        "max_chunk_abs_G_last": float(chunk.max()),
        "chunk": chunk_index,
        "head": head,
        "channel": channel,
        "chunk_pairs_over_144": int((chunk > 144).sum()),
        "chunk_pairs_weak_below_2^-9": int((chunk < 2.0**-9).sum()),
        "chunk_pairs": chunk.numel(),
    }


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """KimiRMSNorm / Glm5NextTextRMSNorm: fp32 normalize, cast to the input dtype, scale (bf16 result)."""
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _hyper_connection_collapse(
    embeddings: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, text_config: dict
) -> torch.Tensor:
    """Glm5NextTextHyperConnection pre-mix collapse of hc_mult identical streams (transformers@26c0a7fd)."""
    streams_count, eps = text_config["hc_mult"], text_config["rms_norm_eps"]
    streams = embeddings.unsqueeze(1).expand(-1, streams_count, -1)
    flat = streams.reshape(streams.shape[0], -1).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre_logits = (flat @ fn.float().T)[:, :streams_count]
    pre = torch.sigmoid(pre_logits * scale.float()[0] + base.float()[:streams_count]) + text_config["hc_eps"]
    return (pre.unsqueeze(-1) * streams).sum(dim=1).to(embeddings.dtype)


def corpus_body() -> str:
    """Body of the pinned corpus (after its "*** START OF" line), downloaded once into the shared oracle cache."""
    path = oracle_cache_root() / "text_corpus" / Path(CORPUS_URL).name

    def download() -> bytes:
        from huggingface_hub import get_session

        response = get_session().get(CORPUS_URL, follow_redirects=True)
        response.raise_for_status()
        _check_corpus(response.content, CORPUS_URL)
        return response.content

    publish_once(path, download, lambda data, file: file.write_bytes(data), lambda file: None)
    _check_corpus(path.read_bytes(), path)
    raw = path.read_text(encoding="utf-8")
    return raw[raw.index("\n", raw.index("*** START OF")) + 1 :]


def _check_corpus(content: bytes, source: object) -> None:
    digest = hashlib.sha256(content).hexdigest()
    if digest != CORPUS_SHA256:
        raise ValueError(f"corpus {source} sha256 {digest} != pinned {CORPUS_SHA256}")


def _encoder(spec: TextInputModel) -> Callable[[str], list[int]]:
    from huggingface_hub import hf_hub_download

    if spec.tokenizer == "tiktoken":
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        ranks = load_tiktoken_bpe(hf_hub_download(spec.repo, "tiktoken.model", revision=spec.revision))
        encoding = tiktoken.Encoding(name=spec.repo, pat_str=_KIMI_PATTERN, mergeable_ranks=ranks, special_tokens={})
        return encoding.encode_ordinary
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(spec.repo, "tokenizer.json", revision=spec.revision))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def _shard_of(checkpoint_dir: Path, key: str) -> str:
    with (checkpoint_dir / "model.safetensors.index.json").open(encoding="utf-8") as index_file:
        return json.load(index_file)["weight_map"][key]


def _checkpoint_tensor(spec: TextInputModel, checkpoint_dir: Path, key: str) -> torch.Tensor:
    shard = checkpoint_dir / _shard_of(checkpoint_dir, key)
    if not shard.exists():
        raise FileNotFoundError(f"{key} lives in {shard.name}, which is not in {checkpoint_dir}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _embedding_rows(spec: TextInputModel, checkpoint_dir: Path, token_ids: list[int]) -> torch.Tensor:
    """Embedding rows of ``token_ids`` (in order), read locally or by HTTP range reads of the pinned shard."""
    key = f"{spec.model_root}embed_tokens.weight"
    filename = _shard_of(checkpoint_dir, key)
    unique = sorted(set(token_ids))
    if (checkpoint_dir / filename).exists():
        with safe_open(checkpoint_dir / filename, framework="pt", device="cpu") as handle:
            table = handle.get_slice(key)[:][unique]
    else:
        table = _remote_rows(spec, filename, key, unique)
    position = {token: row for row, token in enumerate(unique)}
    return table[[position[token] for token in token_ids]]


_SAFETENSOR_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def _remote_rows(spec: TextInputModel, filename: str, key: str, rows: list[int]) -> torch.Tensor:
    from huggingface_hub import get_session, hf_hub_url

    session = get_session()
    response = session.get(
        hf_hub_url(spec.repo, filename, revision=spec.revision), headers={"Range": "bytes=0-7"}, follow_redirects=True
    )
    response.raise_for_status()
    url = str(response.url)

    def read(offset: int, size: int) -> bytes:
        for attempt in range(5):
            try:
                reply = session.get(url, headers={"Range": f"bytes={offset}-{offset + size - 1}"})
                reply.raise_for_status()
                if len(reply.content) != size:
                    raise IOError(f"short read {len(reply.content)} != {size}")
                return reply.content
            except Exception:  # noqa: BLE001 - transient CDN errors are retried
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")

    header_size = struct.unpack("<Q", response.content)[0]
    meta = json.loads(read(8, header_size))[key]
    dtype = _SAFETENSOR_DTYPES[meta["dtype"]]
    width = meta["shape"][1]
    row_bytes = width * torch.tensor([], dtype=dtype).element_size()
    base = 8 + header_size + meta["data_offsets"][0]
    spans, first, last = [], rows[0], rows[0]
    for row in rows[1:]:
        if row - last > 8:
            spans.append((first, last))
            first = row
        last = row
    spans.append((first, last))

    def fetch(span: tuple[int, int]) -> tuple[int, torch.Tensor]:
        low, high = span
        raw = read(base + low * row_bytes, (high - low + 1) * row_bytes)
        return low, torch.frombuffer(bytearray(raw), dtype=dtype).reshape(high - low + 1, width)

    table = {}
    with ThreadPoolExecutor(16) as pool:
        for low, block in pool.map(fetch, spans):
            for offset in range(block.shape[0]):
                table[low + offset] = block[offset].clone()
    logger.info(f"text input: {len(rows)} embedding rows of {filename} in {len(spans)} range reads")
    return torch.stack([table[row] for row in rows])
