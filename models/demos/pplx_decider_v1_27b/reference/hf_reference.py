# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-streamed HuggingFace reference for perplexity-ai/pplx-decider-v1-27b.

The checkpoint is ~54 GB of BF16 and the host has ~61 GB of RAM, so this module never
builds the full model. It builds one HF ``Qwen3_5DecoderLayer`` (or the embedding,
final norm or readout) at a time from the snapshot safetensors, loading only that
module's tensors with ``strict=True``.

Reference forward being reproduced (snapshot ``source/src/autojev/model.py``):
``Qwen3_5Model(...).last_hidden_state[:, -1]`` -> ``readout`` (Linear 5120->255, no bias)
-> mask by option count -> ``/ temperature`` -> softmax. For text-only inputs the
interleaved mRoPE reduces to plain 1D RoPE on the first 64 dims of each head, which is
what ``Qwen3_5TextRotaryEmbedding`` computes with 1D positions.

CLI (writes layer inputs of a real prompt to the golden directory)::

    python -m models.demos.pplx_decider_v1_27b.reference.hf_reference \
        --seq-len 8192 --layers 0 3 61 63 --out /local/ttuser/gtobar/artifacts/pplx_decider/goldens
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import cached_property
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding

MODEL_ID = "perplexity-ai/pplx-decider-v1-27b"
REVISION = "b01a5cbaca5391f73bd55103d4f27e8982cd5e60"
TEXT_PREFIX = "language_model."
LAYER_PREFIX = TEXT_PREFIX + "layers.{}."
NUM_OPTIONS = 255
DEFAULT_GOLDEN_DIR = Path(
    os.environ.get("PPLX_DECIDER_GOLDEN_DIR", "/local/ttuser/gtobar/artifacts/pplx_decider/goldens")
)


def snapshot_path() -> Path:
    """Resolve the immutable snapshot directory (env override, else the HF cache)."""
    override = os.environ.get("PPLX_DECIDER_SNAPSHOT")
    if override:
        return Path(override)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))


class SnapshotReader:
    """Reads individual tensors from the sharded snapshot; nothing is cached in RAM."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else snapshot_path()
        self.weight_map: dict[str, str] = json.loads((self.path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    @cached_property
    def text_config(self):
        config = AutoConfig.from_pretrained(self.path, local_files_only=True).text_config
        config._attn_implementation = "sdpa"
        return config

    @cached_property
    def decision_config(self) -> dict:
        return json.loads((self.path / "decision_config.json").read_text())

    def tensor(self, key: str) -> torch.Tensor:
        with safe_open(self.path / self.weight_map[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def tensors_with_prefix(self, prefix: str) -> dict[str, torch.Tensor]:
        keys = [k for k in self.weight_map if k.startswith(prefix)]
        if not keys:
            raise KeyError(f"No snapshot tensors with prefix '{prefix}'")
        by_shard: dict[str, list[str]] = {}
        for key in keys:
            by_shard.setdefault(self.weight_map[key], []).append(key)
        out = {}
        for shard, shard_keys in by_shard.items():
            with safe_open(self.path / shard, framework="pt", device="cpu") as f:
                for key in shard_keys:
                    out[key[len(prefix) :]] = f.get_tensor(key)
        return out

    def layer_state_dict(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Layer-local dict whose keys match ``Qwen3_5DecoderLayer.state_dict()``."""
        return self.tensors_with_prefix(LAYER_PREFIX.format(layer_idx))

    def embedding_weight(self) -> torch.Tensor:
        return self.tensor(TEXT_PREFIX + "embed_tokens.weight")

    def final_norm_weight(self) -> torch.Tensor:
        return self.tensor(TEXT_PREFIX + "norm.weight")

    def readout_weight(self) -> torch.Tensor:
        with safe_open(self.path / "readout.safetensors", framework="pt", device="cpu") as f:
            return f.get_tensor("weight")

    @property
    def temperature(self) -> float:
        return float(self.decision_config["temperature"])


def build_decoder_layer(reader: SnapshotReader, layer_idx: int, dtype=torch.float32) -> Qwen3_5DecoderLayer:
    """One HF decoder layer with exactly its own weights (strict load)."""
    layer = Qwen3_5DecoderLayer(reader.text_config, layer_idx)
    layer.load_state_dict(reader.layer_state_dict(layer_idx), strict=True)
    return layer.to(dtype).eval()


def build_embedding(reader: SnapshotReader, dtype=torch.float32) -> torch.nn.Embedding:
    cfg = reader.text_config
    module = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    module.load_state_dict({"weight": reader.embedding_weight()}, strict=True)
    return module.to(dtype).eval()


def build_final_norm(reader: SnapshotReader, dtype=torch.float32) -> Qwen3_5RMSNorm:
    cfg = reader.text_config
    module = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    module.load_state_dict({"weight": reader.final_norm_weight()}, strict=True)
    return module.to(dtype).eval()


def build_readout(reader: SnapshotReader, dtype=torch.float32) -> torch.nn.Linear:
    module = torch.nn.Linear(reader.text_config.hidden_size, NUM_OPTIONS, bias=False)
    module.load_state_dict({"weight": reader.readout_weight()}, strict=True)
    return module.to(dtype).eval()


def rotary_cos_sin(config, seq_len: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """HF position embeddings for text-only positions ``start..start+seq_len``; each [1, S, 64] fp32."""
    rope = Qwen3_5TextRotaryEmbedding(config)
    positions = torch.arange(start, start + seq_len)[None]
    return rope(torch.empty(1, seq_len, 1, dtype=torch.float32), positions)


@torch.no_grad()
def layer_forward(layer: Qwen3_5DecoderLayer, x: torch.Tensor, position_embeddings) -> torch.Tensor:
    """HF decoder-layer prefill, batch 1, no padding, no cache (``use_cache=False`` in the app)."""
    return layer(x, position_embeddings=position_embeddings, attention_mask=None, position_ids=None)


@torch.no_grad()
def layer_submodule_outputs(layer: Qwen3_5DecoderLayer, x: torch.Tensor, config) -> dict[str, torch.Tensor]:
    """Every intermediate the TT module tests compare against, from the HF submodules."""
    pe = rotary_cos_sin(config, x.shape[1])
    normed = layer.input_layernorm(x)
    if layer.layer_type == "linear_attention":
        mixer = layer.linear_attn(hidden_states=normed, cache_params=None, attention_mask=None)
    else:
        mixer, _ = layer.self_attn(hidden_states=normed, position_embeddings=pe, attention_mask=None)
    h = x + mixer
    mlp_in = layer.post_attention_layernorm(h)
    mlp_out = layer.mlp(mlp_in)
    return {
        "input_norm": normed,
        "mixer": mixer,
        "post_norm": mlp_in,
        "mlp": mlp_out,
        "layer": h + mlp_out,
    }


def golden_prompt_ids(reader: SnapshotReader, seq_len: int) -> torch.Tensor:
    """Deterministic real-text prompt: the chat template around snapshot text files, first ``seq_len`` ids."""
    tokenizer = AutoTokenizer.from_pretrained(reader.path, local_files_only=True)
    sources = [
        reader.path / "README.md",
        reader.path / "source" / "README.md",
        reader.path / "source" / "src" / "autojev" / "model.py",
        reader.path / "source" / "src" / "autojev" / "data.py",
        reader.path / "source" / "src" / "autojev" / "evaluate.py",
        reader.path / "source" / "src" / "autojev" / "train.py",
    ]
    text = "\n\n".join(p.read_text() for p in sources)
    ids: list[int] = []
    while len(ids) < seq_len:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        text = text + "\n\n" + text
    return torch.tensor(ids[:seq_len], dtype=torch.int64)[None]


def golden_input_path(out_dir: Path, name: str, seq_len: int) -> Path:
    return Path(out_dir) / f"S{seq_len}" / f"{name}.pt"


@torch.no_grad()
def stream_layer_inputs(
    reader: SnapshotReader,
    ids: torch.Tensor,
    capture: list[int],
    out_dir: Path,
    *,
    through_final: bool = False,
) -> None:
    """Run the HF text stack one layer at a time in fp32 and save the input of each captured layer.

    Saves ``ids`` and ``L{i}_input`` (fp32 [1,S,5120]); with ``through_final`` also ``final_input``
    (the output of the last layer, i.e. the input of the final norm).
    """
    seq_len = ids.shape[1]
    target = golden_input_path(out_dir, "ids", seq_len).parent
    target.mkdir(parents=True, exist_ok=True)
    torch.save(ids.clone(), golden_input_path(out_dir, "ids", seq_len))
    cfg = reader.text_config
    last = cfg.num_hidden_layers - 1 if through_final else max(capture)
    embedding = reader.embedding_weight()
    x = embedding[ids[0]].to(torch.float32)[None]
    del embedding
    pe = rotary_cos_sin(cfg, seq_len)
    for i in range(last + 1):
        if i in capture:
            torch.save(x.clone(), golden_input_path(out_dir, f"L{i}_input", seq_len))
        start = time.time()
        layer = build_decoder_layer(reader, i)
        x = layer_forward(layer, x, pe)
        del layer
        logger.info(f"layer {i:2d} ({cfg.layer_types[i]}) S={seq_len} done in {time.time() - start:.1f}s")
    if through_final:
        torch.save(x.clone(), golden_input_path(out_dir, "final_input", seq_len))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 3, 61, 63])
    parser.add_argument("--through-final", action="store_true", help="also run to the last layer for the final norm")
    parser.add_argument("--out", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    reader = SnapshotReader()
    ids = golden_prompt_ids(reader, args.seq_len)
    stream_layer_inputs(reader, ids, args.layers, args.out, through_final=args.through_final)


if __name__ == "__main__":
    main()
