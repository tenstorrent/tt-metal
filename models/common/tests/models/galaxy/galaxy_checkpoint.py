# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load a *layer subset* of a large HF causal LM without materializing all of it.

Milestone B's one-block qualification needs the real checkpoint's layer 0 and
nothing else. The obvious way to get it -
``AutoModelForCausalLM.from_pretrained(...)`` followed by truncating
``model.layers`` - reads and allocates the whole checkpoint first: for
``meta-llama/Llama-3.3-70B-Instruct`` that is 141 GB of NFS I/O and a 141 GB host
allocation to keep 1/80th of it. At ~460 MB/s that is five to six minutes per
process, and the Milestone B house rule is three fresh processes per claim.

This helper reads only the safetensors shards that actually hold the tensors of
the requested layers, plus the embedding, final norm and LM head. For Llama-3.3
that is 3 shards of 30, about 12 GB, roughly 25 s. The resulting module is a
genuine ``LlamaForCausalLM``/``Qwen3ForCausalLM`` holding the checkpoint's real
tensors; it is not a synthetic or randomly initialized stand-in.

Two properties are preserved deliberately, because Milestone A's post-mortem
turned on both:

* the returned module is the *only* source of both the TT weights and the
  reference logits, so a weight-conversion error cannot cancel itself out
  across the two sides of a comparison; and
* the config is the checkpoint's own config with ``num_hidden_layers`` reduced,
  so the rotary embedding - including Llama 3 scaling - is constructed from the
  real parameters rather than from defaults.

A checkpoint that cannot be resolved raises ``CheckpointUnavailable``; callers
turn that into ``pytest.skip``. Inventing weights instead is what the brief
forbids.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


class CheckpointUnavailable(RuntimeError):
    """The checkpoint could not be resolved from local storage."""


def local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def resolve_snapshot(hf_model: str, *, revision: str | None = None) -> Path:
    """Return the local snapshot directory for ``hf_model``.

    Only the small metadata files are requested, so this never triggers a weight
    download; if the weights are absent the shard check below reports it.
    """

    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                hf_model,
                revision=revision,
                allow_patterns=["config.json", "*.index.json", "generation_config.json"],
                local_files_only=local_files_only(),
            )
        )
    except BaseException as error:  # noqa: BLE001 - any resolution failure is unavailability
        raise CheckpointUnavailable(f"cannot resolve {hf_model!r}: {error}") from error


def _weight_map(snapshot: Path) -> dict[str, str]:
    index = snapshot / "model.safetensors.index.json"
    if not index.exists():
        raise CheckpointUnavailable(f"no safetensors index under {snapshot}")
    return json.loads(index.read_text())["weight_map"]


def _wanted_names(weight_map: dict[str, str], layer_indices: Sequence[int], extras: Iterable[str]) -> dict[str, str]:
    prefixes = tuple(f"model.layers.{index}." for index in layer_indices)
    wanted = {name: shard for name, shard in weight_map.items() if name.startswith(prefixes)}
    for name in extras:
        if name in weight_map:
            wanted[name] = weight_map[name]
    return wanted


def _renumber(name: str, layer_indices: Sequence[int]) -> str:
    """Map ``model.layers.<original>.`` onto its position in the subset."""

    for position, index in enumerate(layer_indices):
        prefix = f"model.layers.{index}."
        if name.startswith(prefix):
            return f"model.layers.{position}." + name[len(prefix) :]
    return name


def load_layer_subset_causal_lm(
    hf_model: str,
    *,
    layer_indices: Sequence[int] = (0,),
    revision: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Return a causal LM holding only ``layer_indices`` of ``hf_model``.

    ``layer_indices`` are positions in the original checkpoint; in the returned
    module they are renumbered to ``0..len(layer_indices) - 1`` and
    ``config.num_hidden_layers`` is set accordingly.
    """

    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    layer_indices = tuple(layer_indices)
    snapshot = resolve_snapshot(hf_model, revision=revision)
    weight_map = _weight_map(snapshot)

    try:
        config = AutoConfig.from_pretrained(hf_model, revision=revision, local_files_only=local_files_only())
    except BaseException as error:  # noqa: BLE001
        raise CheckpointUnavailable(f"cannot read the config of {hf_model!r}: {error}") from error

    out_of_range = [index for index in layer_indices if not 0 <= index < int(config.num_hidden_layers)]
    if out_of_range:
        raise ValueError(f"layer indices {out_of_range} are outside the checkpoint's {config.num_hidden_layers}")

    extras = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
    wanted = _wanted_names(weight_map, layer_indices, extras)
    shards = sorted(set(wanted.values()))
    absent = [shard for shard in shards if not (snapshot / shard).exists()]
    if absent:
        raise CheckpointUnavailable(f"{hf_model!r} shards are not materialized locally: {absent}")

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        loaded = load_file(str(snapshot / shard))
        for name, source in wanted.items():
            if source == shard:
                state[_renumber(name, layer_indices)] = loaded[name].to(torch_dtype)
        del loaded
        gc.collect()

    subset_config = AutoConfig.from_pretrained(hf_model, revision=revision, local_files_only=local_files_only())
    subset_config.num_hidden_layers = len(layer_indices)
    # `from_config` on the reduced config allocates only the subset, so the peak
    # host footprint is the subset itself rather than the whole checkpoint.
    model = AutoModelForCausalLM.from_config(subset_config, torch_dtype=torch_dtype)
    if getattr(subset_config, "tie_word_embeddings", False) and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if not name.endswith(".rotary_emb.inv_freq")]
    missing = [name for name in missing if not name.endswith(".rotary_emb.inv_freq")]
    if missing or unexpected:
        raise CheckpointUnavailable(
            f"layer subset of {hf_model!r} did not load cleanly: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    del state
    gc.collect()
    return model


__all__ = [
    "CheckpointUnavailable",
    "load_layer_subset_causal_lm",
    "local_files_only",
    "resolve_snapshot",
]
