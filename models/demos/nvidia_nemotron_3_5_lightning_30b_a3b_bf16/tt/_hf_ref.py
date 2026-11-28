# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-A access: the HuggingFace reference for
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

Two jobs, and ONLY these two:

  1. hand the TT pipeline the torch submodules it extracts weights from at
     BUILD time (allowed HF usage #2), and
  2. compute the GOLDEN output for the PCC gate inside
     `tt/pipeline.py::_hf_reference_text_generation` (allowed HF usage #3).

Nothing here is ever called from the TT hot path.

DEPTH CAP
---------
The checkpoint is 31.58e9 parameters. Its 23 MoE blocks alone hold
128 x 2 x 2688 x 1856 x 23 = 29.4e9 parameters (58.8 GB bf16); expert-parallel
at TP=2 halves that to ~29 GB *per chip* against ~12 GB of Wormhole DRAM. No
TP degree this model permits (TP>2 is blocked by num_key_value_heads=2, see
kernel_findings.json) makes a resident 52-layer build fit on 4 chips.

So the on-device gate runs a DEPTH-CAPPED model, and the golden is the SAME
checkpoint capped to the SAME depth -- TT and HF compute the same function, so
the PCC comparison is exact-in-scope rather than approximate. `layers=None`
still means every layer; the caller chooses.

The first 7 blocks are `[mamba, moe, mamba, moe, mamba, attention, moe]` --
the shortest prefix carrying all three block types with enough of each to host
every graduated stub.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt._hf_compat import install_hf_compat

HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

# `layers_block_type[:7] == [mamba, moe, mamba, moe, mamba, attention, moe]`
DEFAULT_GATE_LAYERS = 7

_DEMO_DIR = Path(__file__).resolve().parents[1]
_CACHE_DIR = Path(os.environ.get("TT_NEMOTRON_REF_CACHE", _DEMO_DIR / "_captured" / "_hf_ref"))

_MODEL_CACHE: dict[int | None, object] = {}
_TOK = None


def get_tokenizer():
    """The Source-A tokenizer used to build every input in this package."""
    global _TOK
    if _TOK is None:
        install_hf_compat()
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    return _TOK


def _truncate(model, layers: int):
    """Cap the decoder stack to `layers` blocks, in place, and free the rest.

    Everything that is NOT a repeated block -- embeddings, final norm, lm_head --
    stays intact, so a capped build still exercises every distinct op the full
    model runs, just fewer times.
    """
    import torch.nn as nn

    cfg = model.config
    full = len(model.model.layers)
    if layers is None or layers >= full:
        return model
    keep = list(model.model.layers[:layers])
    model.model.layers = nn.ModuleList(keep)
    cfg.num_hidden_layers = layers
    cfg.layers_block_type = list(cfg.layers_block_type)[:layers]
    gc.collect()
    return model


def _cache_path(layers: int) -> Path:
    return _CACHE_DIR / f"depth{layers}"


# The census file this model already ships (written by the planner's Step-1 census). It records the
# real checkpoint size and layer count, so the reference-precision decision is keyed to THIS model's
# own numbers, not a value typed into source. Named, not hardcoded per instance.
_CENSUS_FILE = "perf_target_inputs.json"


def _mem_available_bytes():
    """Host memory free for a new allocation right now, from MemAvailable, or None if unreadable.
    Deliberately a few lines of /proc/meminfo rather than importing the perf-automation tool's
    probe, so this model package carries no dependency on that tool. MemAvailable (not MemFree)
    already accounts for reclaimable cache, which is what decides whether the next alloc succeeds."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001 -- unreadable => caller treats as "cannot size", keeps its default
        return None
    return None


def _checkpoint_bf16_bytes_and_layers():
    """(checkpoint bytes, total decoder layers) for THIS model, or (None, None) if unobtainable.

    Sourced from the model's OWN facts, no architecture-specific arithmetic and no perf-tool import.
    Two sources, both keyed to this checkpoint: first the census file a run may have written next to
    the demo (cheap when present), then -- reliably, since it is always there at build time -- the
    checkpoint the build is about to load from: total bytes from the safetensors index's
    metadata.total_size, layer count from the model config. The second source matters because the
    census file is a runtime artifact that a fresh build may not have yet."""
    import json

    try:
        d = json.loads((_DEMO_DIR / _CENSUS_FILE).read_text())
        wb = d.get("weight_bytes")
        ly = d.get("layers") or (d.get("blocks") or {}).get("backbone", {}).get("layers")
        if isinstance(wb, (int, float)) and wb > 0 and isinstance(ly, (int, float)) and ly > 0:
            return (int(wb), int(ly))
    except Exception:  # noqa: BLE001 -- fall through to the checkpoint itself
        pass

    try:
        install_hf_compat()
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoConfig

        total_layers = int(AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True).num_hidden_layers)
        idx = try_to_load_from_cache(HF_MODEL_ID, "model.safetensors.index.json")
        if isinstance(idx, str):
            total_size = json.loads(Path(idx).read_text()).get("metadata", {}).get("total_size")
            if isinstance(total_size, (int, float)) and total_size > 0 and total_layers > 0:
                return (int(total_size), total_layers)
    except Exception:  # noqa: BLE001 -- unsized => caller keeps its default
        pass
    return (None, None)


def _mem_safety_margin() -> float:
    """Dimensionless multiplier from the checkpoint's steady-state bytes to a load's PEAK host use
    (the from_pretrained transient sits above the resident weights). A ratio, not a byte count, and
    tunable from measurement -- it assumes nothing about the machine. Default 1.7 is the measured
    peak/model ratio on this box (fp32 full-depth build ~217 GB vs ~128 GB steady, 2026-09-21)."""
    try:
        return max(1.0, float(os.environ.get("PERF_MCP_MEM_SAFETY_MARGIN", "1.7")))
    except Exception:  # noqa: BLE001
        return 1.7


def _mem_usable_fraction() -> float:
    """Fraction of currently-available memory a build may plan to use, leaving the rest as headroom
    for estimate error and for other processes growing while the build runs (a neighbor took ~74 GB
    mid-run on 2026-09-21). A dimensionless ratio in (0, 1], tunable via PERF_MCP_MEM_USABLE_FRACTION;
    default 0.8 leaves 20%. Not a byte count, so it makes no assumption about machine size."""
    try:
        return min(1.0, max(0.1, float(os.environ.get("PERF_MCP_MEM_USABLE_FRACTION", "0.8"))))
    except Exception:  # noqa: BLE001
        return 0.8


def _fp32_reference_fits(layers):
    """(fits, est_fp32_gb, usable_gb): would a float32 reference of `layers` blocks (None == all) fit
    within the usable share of host memory? Depth-aware -- a shallow build's footprint is scaled by
    its share of the layers, so a gate build fits where the full-depth build does not. The estimate
    is the checkpoint's steady bytes x2 (fp32) x a peak/steady margin; it must fit inside available x
    a usable fraction, not all of available, so a build that only just fits at idle is not attempted.
    `fits` is None when the model cannot be sized or memory cannot be read (caller keeps default)."""
    bf16_bytes, total_layers = _checkpoint_bf16_bytes_and_layers()
    avail = _mem_available_bytes()
    if not bf16_bytes or not total_layers or not avail:
        return (None, None, None)
    frac = 1.0 if layers is None else min(1.0, max(1, int(layers)) / total_layers)
    fp32_need = bf16_bytes * 2 * frac * _mem_safety_margin()
    usable = avail * _mem_usable_fraction()
    return (fp32_need <= usable, fp32_need / 1e9, usable / 1e9)


def choose_reference_dtype(layers):
    """Pick the reference build's precision for `layers` blocks (None == all) and return
    (torch dtype, one-line reason). fp32 when it fits host memory, else bf16 -- so a large full-depth
    reference cannot OOM the host while a shallow gate build keeps full precision. Explicit signals
    win: PERF_MCP_LOW_MEM_REFERENCE=1 forces bf16, PERF_MCP_FORCE_FP32_REFERENCE=1 forces fp32. When
    the model cannot be sized, keep fp32 (the historical default) rather than invent a number."""
    depth = "all" if layers is None else int(layers)
    if os.environ.get("PERF_MCP_LOW_MEM_REFERENCE") == "1":
        return torch.bfloat16, f"bf16 (PERF_MCP_LOW_MEM_REFERENCE=1; layers={depth})"
    if os.environ.get("PERF_MCP_FORCE_FP32_REFERENCE") == "1":
        return torch.float32, f"fp32 (PERF_MCP_FORCE_FP32_REFERENCE=1; layers={depth})"
    fits, est_gb, avail_gb = _fp32_reference_fits(layers)
    if fits is None:
        return torch.float32, f"fp32 (model unsized; layers={depth}, no memory comparison)"
    if fits:
        return torch.float32, f"fp32 fits (layers={depth}, est {est_gb:.0f} GB <= usable {avail_gb:.0f} GB)"
    return torch.bfloat16, f"bf16 to fit memory (layers={depth}, fp32 est {est_gb:.0f} GB > usable {avail_gb:.0f} GB)"


def load_reference(layers: int | None = DEFAULT_GATE_LAYERS, dtype=torch.float32):
    """Return the HF reference model capped to `layers` blocks (None == all 52).

    A depth-capped model is cached to disk on first use so later runs skip the
    66 GB full-checkpoint read.
    """
    key = layers
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    install_hf_compat()
    from transformers import AutoConfig, AutoModelForCausalLM

    ckpt = _cache_path(layers) if layers is not None else None
    if ckpt is not None and (ckpt / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            ckpt, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    else:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        cfg._attn_implementation = "eager"
        # SAME TWO FIELDS _truncate() SETS, MOVED BEFORE CONSTRUCTION. _truncate() used to be the
        # only place that shrank num_hidden_layers/layers_block_type, but it ran on an
        # already-fully-built model -- from_pretrained had by then read and materialized all 52
        # layers' weights from the checkpoint (~66 GB bf16) before the other 45 were ever
        # discarded, which is what OOM-killed the host on this model's first (cache-miss) load.
        # Shrinking the SAME config fields first means from_pretrained only builds and fills the
        # `layers` blocks that survive anyway -- the kept layers end up byte-identical (same
        # checkpoint, same indices), so this changes nothing about the PCC golden or the roofline
        # census (agent/weight_census.py reads the checkpoint files directly, never this object).
        if layers is not None:
            full_layer_count = cfg.num_hidden_layers
            if layers < full_layer_count:
                cfg.num_hidden_layers = layers
                cfg.layers_block_type = list(cfg.layers_block_type)[:layers]
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, config=cfg, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model = _truncate(model, layers)  # no-op safety net: cfg already matches `layers`
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt, safe_serialization=True)

    model.config._attn_implementation = "eager"
    model.eval()
    if dtype is not None:
        model = model.to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    _MODEL_CACHE[key] = model
    return model


def block_types(model) -> list[str]:
    return [blk.block_type for blk in model.model.layers]


if __name__ == "__main__":  # prime the depth-capped cache
    import argparse

    ap = argparse.ArgumentParser(description="Prime the depth-capped HF reference cache.")
    ap.add_argument("--layers", type=int, default=DEFAULT_GATE_LAYERS)
    a = ap.parse_args()
    m = load_reference(a.layers, dtype=None)
    print(f"[ref] depth={len(m.model.layers)} block_types={block_types(m)}")
    print(f"[ref] params={sum(p.numel() for p in m.parameters())/1e9:.2f}B cached at {_cache_path(a.layers)}")
