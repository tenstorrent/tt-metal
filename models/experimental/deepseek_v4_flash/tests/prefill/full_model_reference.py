# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Streamed fp32 CPU reference of the whole DeepSeek-V4-Flash network, on the real checkpoint.

The reference model (``DeepseekV4ForCausalLM``, the standalone HF copy under
``models.demos.deepseek_v3_d_p.reference.deepseek_v4``) holds ~280B parameters, more than fits in host
memory as fp32. So this streams: one decoder layer at a time is built, given its real (dequantized)
checkpoint weights, run over the residual streams, and freed (a layer is ~38 GB of host memory while
resident). Every layer's output streams, the embedding and the final logits are cached on disk, keyed by
the prompt tokens, so the expensive pass runs once and the device verification test only reads it. No
device is touched: it can be run ahead of time, alone::

    python -m models.experimental.deepseek_v4_flash.tests.prefill.full_model_reference --len 2048

The CSA lightning indexer keeps each query's top ``index_topk`` compressed entries. The prefill model runs
CSA *without* the indexer (every visible entry), which equals the model exactly while a layer has at most
``index_topk`` entries (2048 tokens). Past that the reference is made dense the same way -- ``index_topk``
is raised so ``top_k = min(index_topk, entries)`` keeps everything -- so it stays comparable to the device
at any length, though for prompts over 2048 tokens neither is the model's exact answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_DENSE_INDEX_TOPK = 1 << 30


def default_reference_dir() -> Path:
    """Where reference bundles are cached (``DEEPSEEK_V4_VERIFY_REF_DIR``, else next to the weight cache)."""
    if "DEEPSEEK_V4_VERIFY_REF_DIR" in os.environ:
        return Path(os.environ["DEEPSEEK_V4_VERIFY_REF_DIR"])
    base = os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache")
    return Path(base) / "prefill_reference"


def reference_config(loader: DeepseekV4WeightLoader, seq_len: int) -> DeepseekV4Config:
    """The checkpoint's config for the reference, made dense-CSA-equivalent when the prompt needs it."""
    cfg = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    cfg._attn_implementation = "eager"
    if seq_len // min(cfg.compress_rates.values()) > cfg.index_topk:
        cfg.index_topk = _DENSE_INDEX_TOPK
    return cfg


def _sliding_causal_mask(seq_len: int, sliding_window: int) -> torch.Tensor:
    """Additive ``[1, 1, S, S]`` mask: query ``i`` sees the ``sliding_window`` tokens ending at ``i``."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len).masked_fill(~keep, torch.finfo(torch.float32).min)
    return mask.view(1, 1, seq_len, seq_len)


def _weight(loader: DeepseekV4WeightLoader, name: str) -> torch.Tensor:
    return dequantize_weight(loader.get_tensor(name), loader.get_scale(name)).to(torch.float32)


def load_reference_layer(
    cfg: DeepseekV4Config, loader: DeepseekV4WeightLoader, layer_idx: int
) -> DeepseekV4DecoderLayer:
    """Layer ``layer_idx`` with its real weights: every parameter by HF name, the packed experts assembled."""
    layer = DeepseekV4DecoderLayer(cfg, layer_idx).eval()
    state = layer.state_dict()
    num_experts = state["mlp.experts.gate_up_proj"].shape[0]

    def w(name: str) -> torch.Tensor:
        return _weight(loader, f"layers.{layer_idx}.{name}")

    sd: dict[str, torch.Tensor] = {}
    for key in state:
        if key == "mlp.experts.gate_up_proj":
            sd[key] = torch.stack(
                [
                    torch.cat([w(f"mlp.experts.{e}.gate_proj.weight"), w(f"mlp.experts.{e}.up_proj.weight")], dim=0)
                    for e in range(num_experts)
                ]
            )  # [E, 2I, D]
        elif key == "mlp.experts.down_proj":
            sd[key] = torch.stack([w(f"mlp.experts.{e}.down_proj.weight") for e in range(num_experts)])  # [E, D, I]
        elif key.endswith("tid2eid"):  # the frozen token-id -> expert-id table (an integer buffer)
            sd[key] = loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").to(state[key].dtype)
        else:
            sd[key] = w(key)
    layer.load_state_dict(sd, strict=True, assign=True)
    return layer


class ReferenceStore:
    """The cached reference for one prompt: embedding streams, each layer's output, the logits."""

    def __init__(self, directory: Path, num_layers: int):
        self.dir = directory
        self.num_layers = num_layers

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.pt"

    def has(self, name: str) -> bool:
        return self.path(name).exists()

    def get(self, name: str) -> torch.Tensor:
        return torch.load(self.path(name), map_location="cpu", weights_only=True)

    def put(self, name: str, tensor: torch.Tensor) -> None:
        tmp = self.path(name).with_suffix(".tmp")
        torch.save(tensor, tmp)
        tmp.rename(self.path(name))  # a file is either complete or absent

    def layer_input(self, layer_idx: int) -> torch.Tensor:
        """The streams ``[1, T, hc, D]`` entering layer ``layer_idx`` in the reference chain."""
        return self.get("embedding" if layer_idx == 0 else f"layer_{layer_idx - 1:02d}")

    def layer_output(self, layer_idx: int) -> torch.Tensor:
        return self.get(f"layer_{layer_idx:02d}")

    def logits(self) -> torch.Tensor:
        return self.get("logits")

    def complete(self) -> bool:
        return self.has("logits") and all(self.has(f"layer_{i:02d}") for i in range(self.num_layers))


def store_for(ids: torch.Tensor, num_layers: int, cfg: DeepseekV4Config, root: Optional[Path] = None) -> ReferenceStore:
    """The store for these token ids (keyed by the ids, the layer count and the CSA mode)."""
    key = hashlib.sha1(ids.to(torch.int64).numpy().tobytes() + f"|{num_layers}|{cfg.index_topk}".encode()).hexdigest()
    directory = (root or default_reference_dir()) / f"T{ids.shape[1]}_L{num_layers}_{key[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    return ReferenceStore(directory, num_layers)


@torch.no_grad()
def run_reference(
    loader: DeepseekV4WeightLoader,
    cfg: DeepseekV4Config,
    ids: torch.Tensor,
    store: ReferenceStore,
    progress: Optional[Callable[[str], None]] = None,
) -> ReferenceStore:
    """Fill ``store`` for ``ids`` ``[1, T]`` (resuming after the last cached layer). Returns ``store``."""
    note = progress or (lambda message: logger.info(message))
    if store.complete():
        note(f"reference already cached in {store.dir}")
        return store
    seq_len = ids.shape[1]
    started = time.perf_counter()
    position_ids = torch.arange(seq_len).unsqueeze(0)
    mask = _sliding_causal_mask(seq_len, cfg.sliding_window)
    rotary = DeepseekV4RotaryEmbedding(cfg)

    if not store.has("embedding"):
        note("reference: embedding")
        table = _weight(loader, "embed_tokens.weight")
        embedded = table[ids]  # [1, T, D]
        store.put("embedding", embedded.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous())
        del table
    first_missing = next((i for i in range(store.num_layers) if not store.has(f"layer_{i:02d}")), store.num_layers)
    streams = store.layer_input(first_missing) if first_missing < store.num_layers else None

    position_embeddings = {
        kind: rotary(
            streams if streams is not None else torch.zeros(1, seq_len, 1, 1),
            position_ids=position_ids,
            layer_type=kind,
        )
        for kind in ("main", "compress")
    }
    for li in range(first_missing, store.num_layers):
        t0 = time.perf_counter()
        layer = load_reference_layer(cfg, loader, li)
        t_load = time.perf_counter() - t0
        streams = layer(
            streams,
            input_ids=ids,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=None,
        )
        assert torch.isfinite(streams).all(), f"reference layer {li} produced non-finite values"
        store.put(f"layer_{li:02d}", streams)
        del layer
        elapsed = time.perf_counter() - started
        note(
            f"reference layer {li + 1}/{store.num_layers} ({cfg.layer_types[li]}, {cfg.mlp_layer_types[li]}): "
            f"weights {t_load:.1f}s, total {time.perf_counter() - t0:.1f}s, std {streams.std():.3f} "
            f"[{elapsed / 60:.1f} min so far]"
        )

    if not store.has("logits"):
        note("reference: hyper head, final norm, lm_head")
        head = DeepseekV4HyperHead(cfg)
        head.load_state_dict(
            {p: _weight(loader, f"hc_head.{p}") for p in ("hc_fn", "hc_base", "hc_scale")}, assign=True
        )
        norm = DeepseekV4RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        norm.load_state_dict({"weight": _weight(loader, "norm.weight")}, assign=True)
        final = store.layer_output(store.num_layers - 1) if streams is None else streams
        hidden = norm(head(final))
        store.put("logits", F.linear(hidden, _weight(loader, "lm_head.weight")))
    (store.dir / "meta.json").write_text(json.dumps({"tokens": seq_len, "layers": store.num_layers}))
    note(f"reference done in {(time.perf_counter() - started) / 60:.1f} min -> {store.dir}")
    return store


if __name__ == "__main__":  # pre-generate the reference for the demo's prompt, host only
    import argparse

    from transformers import AutoTokenizer

    from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import _DEFAULT_MODEL_DIR
    from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
        _DEFAULT_PROMPT_FILE,
        _build_prompt_ids,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--len", type=int, default=2048, help="prompt tokens (a multiple of 128)")
    parser.add_argument("--layers", type=int, default=None, help="only the first N layers")
    args = parser.parse_args()

    ckpt = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(ckpt.snapshot_dir)
    prompt_ids, _ = _build_prompt_ids(tok, json.loads(Path(_DEFAULT_PROMPT_FILE).read_text())[0], args.len)
    config = reference_config(ckpt, args.len)
    n_layers = min(args.layers or config.num_hidden_layers, config.num_hidden_layers)
    token_ids = torch.tensor(prompt_ids).unsqueeze(0)
    run_reference(ckpt, config, token_ids, store_for(token_ids, n_layers, config))
