# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline, deterministic target-shape weights from recorded checkpoint stats.

No HF Hub calls or checkpoint reads occur here. BF16 tensors are generated
straight into their final storage, without a full FLOAT32 intermediate. Seeds
are derived from canonical checkpoint names, so filtering/reordering generation
does not change any tensor. The 102GB PLE embedding table is never synthesized;
full PLE decoder acceptance belongs to the separate real-table harness.
"""

import hashlib
import json
from pathlib import Path

import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

EVIDENCE = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder"
REVISION = "812d96e10a0dd82b69d749e68693698cf6c36daa"


def target_config():
    """The saved actual HF text configuration; never shrink its dimensions/context."""
    config = Qwen4ExpTextConfig(**json.loads((EVIDENCE / "target_text_config.json").read_text()))
    config._attn_implementation = "eager"
    if (config.hidden_size, config.hc_count, config.max_position_embeddings) != (2560, 4, 262144):
        raise ValueError("Saved target configuration does not match the pinned model")
    return config


def recorded_stats():
    report = json.loads((EVIDENCE / "weight_stats.json").read_text())
    if report["revision"] != REVISION or report["std_correction"] != 0:
        raise ValueError("Unexpected weight-stat provenance or variance convention")
    return report["tensors"]


def synthetic_tensor(record):
    """Allocate exactly the recorded shape/dtype; use a name-stable local RNG."""
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}.get(record["dtype"])
    if dtype is None:
        raise ValueError(f'Cannot sample semantic integer buffer from mean/std: {record["name"]}')
    seed = int.from_bytes(hashlib.sha256(record["name"].encode()).digest()[:8], "little") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor = torch.empty(record["shape"], dtype=dtype, device="cpu")
    if record["std"] == 0:
        return tensor.fill_(record["mean"])
    return tensor.normal_(mean=record["mean"], std=record["std"], generator=generator)


def layer_weights(layer_idx, *, component=None):
    """Return canonical layer-local keys. PLE table/integer hashing state excluded.

    Full decoder sampling is supported for layers0/3. For layer1, explicitly
    request component='ple.'; this samples its floating arithmetic weights only,
    retaining the actual config shapes without allocating a fake lookup table.
    """
    if layer_idx not in (0, 3) and not (layer_idx == 1 and component == "ple."):
        raise ValueError("Supported offline fixtures: full layers0/3 or layer1 PLE arithmetic")
    prefix = f"model.language_model.layers.{layer_idx}."
    records = [record for record in recorded_stats() if record["name"].startswith(prefix)]
    if component is not None:
        records = [record for record in records if record["name"].startswith(prefix + component)]
    result = {}
    for record in records:
        local = record["name"].removeprefix(prefix)
        if ".ngram_embedding.shard_" in local or record["dtype"] == "torch.int64":
            continue
        result[local] = synthetic_tensor(record)
    if not result:
        raise ValueError("No matching recorded tensors")
    return result


def embedding_fixture(tokens, *, seed):
    """PLE arithmetic-only input distribution, drawn from recorded shard0 stats."""
    record = next(
        record
        for record in recorded_stats()
        if record["name"].endswith("layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight")
    )
    return torch.empty((1, tokens, 2560), dtype=torch.bfloat16).normal_(
        mean=record["mean"], std=record["std"], generator=torch.Generator().manual_seed(seed)
    )
