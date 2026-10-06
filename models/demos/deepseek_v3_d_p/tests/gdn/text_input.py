# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-text inputs of the first Qwen GDN layer: English prose, tokenized, embedded and normalized as layer 0 sees it.

Seeded random hidden states miss both decay extremes of the trained GDN gates: on real text a few layer-0 V heads
stay in the weak band (|G_last| < 2^-9) for most 32-token chunks, 3-11 % of (chunk, head) entries against about 0 %
for random inputs on 27B / 35B-A3B (tt_metal_tracker-g1b.5.16), while the full-forgetting heads reach |G_last| in the
thousands (g1b.5.12). Layer 0 is the first GDN layer of every target model and its input depends only on the token,
so the exact layer input needs only the token embeddings and layer 0's input normalization
(transformers@56d3afc0):

* Qwen3.5 / Qwen3.5-MoE (27B, 35B-A3B, 2.4T): ``input_layernorm(embed_tokens(ids))`` (``Qwen3_5RMSNorm``, ``1 + w``).
* Qwen4-Exp (Flash-Next): the embedding repeated into ``hc_count`` hyper-connection streams, then layer 0's
  ``attn_hyper_connection`` input mix (``Qwen4ExpTextGatedResidual``); layer 0 has no per-layer embedding when
  ``1`` is not in ``ple_layer_ids``.

Building an input reads the pinned tokenizer, the embedding rows of the window's tokens and layer 0's norm tensors by
HTTP range reads of the pinned checkpoint (nothing is written to the weights directory), so it runs only in the CPU
preparation step. The result goes to the shared CPU oracle cache; device tests load it and fail fast on a miss.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import (
    QWEN_FIRST_GDN_LAYER,
    QWEN_GDN_MODELS,
    qwen_model_config,
)
from models.demos.deepseek_v3_d_p.tests.gdn.checkpoint_utils import gdn_model_root
from models.demos.deepseek_v3_d_p.tests.gdn.hub_shard import HubSafetensorsShard
from models.demos.deepseek_v3_d_p.tests.kda.text_input import CORPUS_SHA256, corpus_body
from models.demos.deepseek_v3_d_p.utils.oracle_cache import oracle_cache_root, publish_once

# Covers the stored text input: corpus body extraction (tests/kda/text_input.py), the tokenizer, the window and the
# layer-0 input computation below. Bump when any of them changes the stored tensor; the cache is shared by every
# worktree (utils/oracle_cache.py), so an unmerged branch bumps to a value no other branch uses.
TEXT_INPUT_VERSION = 1
# First corpus token of the window (the start of the body, as the g1b.5.12 / g1b.5.16 decay measurements).
WINDOW_START = 0


def gdn_text_input_cache_path(model: str, layer_idx: int, tokens: int) -> Path:
    """Shared oracle-cache path of one model / layer / length text input."""
    revision = QWEN_GDN_MODELS[model].revision
    name = (
        f"v{TEXT_INPUT_VERSION}-{revision[:12]}-layer{layer_idx}-pg1342-{CORPUS_SHA256[:12]}"
        f"-start{WINDOW_START}-tokens{tokens}.pt"
    )
    return oracle_cache_root() / "gdn" / model / "text_input" / name


def load_gdn_text_input(model: str, layer_idx: int, tokens: int) -> torch.Tensor | None:
    """Return the cached ``[1, tokens, hidden]`` bf16 text input, or None when it was not prepared."""
    path = gdn_text_input_cache_path(model, layer_idx, tokens)
    if not path.is_file():
        return None
    hidden = _load_checked(path)["hidden"]
    logger.info(f"GDN text input {model} layer {layer_idx} T={tokens} cache hit: {path}")
    return hidden


def build_gdn_text_input(model: str, layer_idx: int, tokens: int) -> torch.Tensor:
    """Build, cache and return the text input (CPU preparation only: reads the hub)."""
    if layer_idx != QWEN_FIRST_GDN_LAYER:
        raise ValueError(f"a GDN text input is the exact layer input only for layer {QWEN_FIRST_GDN_LAYER}")
    path = gdn_text_input_cache_path(model, layer_idx, tokens)
    start = time.perf_counter()

    def produce() -> dict:
        token_ids = _token_ids(model, tokens)
        hidden = _layer0_input(model, token_ids).unsqueeze(0).contiguous()
        return {"hidden": hidden, "sha256": _sha256(hidden), "token_ids": torch.tensor(token_ids)}

    payload, produced = publish_once(path, produce, torch.save, _load_checked)
    verb = "built" if produced else "published by another producer, loaded"
    logger.info(f"GDN text input {model} layer {layer_idx} T={tokens} {verb} in {time.perf_counter() - start:.1f} s")
    return payload["hidden"]


def qwen_input_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int | None = None) -> torch.Tensor:
    """``Qwen3_5RMSNorm`` / ``Qwen4ExpTextRMSNorm``: FP32 normalize (per ``group`` features), ``* (1 + w)``, cast back."""
    xf = x.float()
    if group is not None:
        xf = xf.reshape(*xf.shape[:-1], -1, group)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        out = out.flatten(-2)
    return (out * (1.0 + weight.float())).to(x.dtype)


def hyper_connection_input_mix(
    embeddings: torch.Tensor,
    norm_weight: torch.Tensor,
    mix_down: torch.Tensor,
    mix_up: torch.Tensor,
    streams: int,
    eps: float,
) -> torch.Tensor:
    """Layer-0 ``Qwen4ExpTextGatedResidual`` mixed input of ``streams`` copies of the embedding (dtype-preserving)."""
    hidden = embeddings.shape[-1]
    normed = qwen_input_norm(embeddings.repeat(1, streams), norm_weight, eps, group=hidden)
    weight = F.silu(F.linear(normed, mix_down) / streams)
    weight = torch.sigmoid(F.linear(weight, mix_up)).unflatten(-1, (streams, hidden))
    return (weight * normed.unflatten(-1, (streams, hidden))).mean(dim=-2)


def _layer0_input(model: str, token_ids: list[int]) -> torch.Tensor:
    from huggingface_hub import hf_hub_download

    source = QWEN_GDN_MODELS[model]
    config = qwen_model_config(model)
    text_config = config.get("text_config", config)
    root = gdn_model_root(config)
    index_path = hf_hub_download(source.repo, "model.safetensors.index.json", revision=source.revision)
    weight_map = json.loads(Path(index_path).read_text(encoding="utf-8"))["weight_map"]
    shards: dict[str, HubSafetensorsShard] = {}

    def shard(key: str) -> HubSafetensorsShard:
        if weight_map[key] not in shards:
            shards[weight_map[key]] = HubSafetensorsShard(source.repo, source.revision, weight_map[key])
        return shards[weight_map[key]]

    embed_key = f"{root}embed_tokens.weight"
    unique = sorted(set(token_ids))
    table = shard(embed_key).rows(embed_key, unique)
    position = {token: row for row, token in enumerate(unique)}
    embeddings = table[[position[token] for token in token_ids]]
    layer = f"{root}layers.{QWEN_FIRST_GDN_LAYER}."
    eps = text_config["rms_norm_eps"]
    streams = text_config.get("hc_count")
    if streams is None:
        keys = [f"{layer}input_layernorm.weight"]
    else:
        if QWEN_FIRST_GDN_LAYER + 1 in (text_config.get("ple_layer_ids") or ()):
            raise ValueError(f"{model}: layer {QWEN_FIRST_GDN_LAYER} adds a per-layer embedding, not modelled here")
        names = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")
        keys = [f"{layer}attn_hyper_connection.{name}" for name in names]
    tensors = [shard(key).tensor(key) for key in keys]
    logger.info(
        f"GDN text input {model}: {len(unique)} embedding rows and {keys} by range reads, "
        f"{sum(s.bytes_read for s in shards.values()) / 2**20:.1f} MiB"
    )
    if streams is None:
        return qwen_input_norm(embeddings, *tensors, eps)
    return hyper_connection_input_mix(embeddings, *tensors, streams=streams, eps=eps)


def _token_ids(model: str, tokens: int) -> list[int]:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    source = QWEN_GDN_MODELS[model]
    tokenizer = Tokenizer.from_file(hf_hub_download(source.repo, "tokenizer.json", revision=source.revision))
    token_ids = tokenizer.encode(corpus_body(), add_special_tokens=False).ids[WINDOW_START : WINDOW_START + tokens]
    if len(token_ids) != tokens:
        raise ValueError(f"corpus has {len(token_ids)} tokens after {WINDOW_START}, need {tokens}")
    return token_ids


def _sha256(hidden: torch.Tensor) -> str:
    return hashlib.sha256(memoryview(hidden.contiguous().view(torch.uint8).numpy())).hexdigest()


def _load_checked(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert _sha256(payload["hidden"]) == payload["sha256"], f"GDN text input checksum mismatch: {path}"
    return payload
