# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 weights for tests until the official weights are published: two tiers, each with its provenance.

Tier 1 (accuracy): every weight seeded random, at a reduced config (``glm54_reduced_config``): GLM-5.3 widths, so
device modules run their production shapes, and 4 layers holding every GLM-5.4 layer role:

    L0 prelude (dense FFN, full indexer) | L1 loop (MoE, full) L2 loop (MoE, shared) | L3 coda (MoE, full)

The loop L1-L2 runs twice; the first coda layer owns an indexer, as L54 does at full scale. Layer weights come
from the GLM-5.3 generator (``glm_synthetic_weights``), drawn per global layer index, with one routed/shared
expert set reused by every MoE layer (each keeps its own router, norms and MLA, so a layer mix-up still shows) to
hold host RAM at one MoE layer (~19 GB). The same dicts feed ``glm54_stack_reference`` and
``TtPrefillTransformer``. Tier 1 checks wiring and conformance only (synthetic data does not show numeric
sensitivity).

Tier 2 (performance): every shared tensor is GLM-5.3's FP8 checkpoint, GLM-5.4 layer L (either loop pass) =
GLM-5.3 layer L; only the damp gates are synthetic (``reference.glm_5_4.synthetic_gates``, centred so beta1 ~0.2,
beta2 ~0.9). Device runs read those tensors from the GLM-5.3 TTNN cache through the runner, unchanged: its
directory is ``{cache}/glm_5_3_{arch}_{N}dev/...``. ``MLAPrefillAdapter.weight_cache_path`` names it after the
adapter, so a GLM-5.4 adapter must keep the ``glm_5_3`` name or override that path: otherwise it creates
``{cache}/glm_5_4_{arch}_{N}dev`` on Weka, finds every weight category missing, and the runner rebuilds and writes a
full cache there (``weight_categories``), breaking the read-only rule for the GLM-5.3 checkpoint and cache. Reference
checks load one layer with ``glm54_checkpoint_layer``. Nothing here writes to the checkpoint or the cache.

``tier2_sources`` resolves the checkpoint and cache the way the runner does (``$PREFILL_HF_MODEL``, else
``$GLM53_HF_MODEL``, else the adapter default; ``$PREFILL_TTNN_CACHE``, else the adapter default), and every tier-2
function reads its paths from it, so the recorded provenance is the source actually used.

``tier1_provenance`` / ``tier2_provenance`` return what a run must record to be reproducible: generator,
seeds and constants for synthetic tensors; checkpoint path, revision and file digests for real ones.

Tier-1 disk cache. Generating tier 1 takes ~80 s of host CPU, which device tests used to spend holding the device
lock, so ``glm54_tier1_weights`` keeps one file per key under ``tier1_cache_dir()`` (``$TT_GLM54_CACHE_DIR``, default
``~/.cache/tt-glm54``, then ``tier1-weights``; never Weka) and memory-maps it on a hit (~0.1 s). The key
(``tier1_key_inputs``) is everything the bytes depend on: the generator's source (whole files plus the functions it
calls elsewhere), the torch version, every hf-config field except those the generator does not read
(``TIER1_KEY_EXCLUDED_FIELDS``, so runs at different sequence lengths share one entry), the model constants and the
seed. A miss generates, writes the entry atomically (~23 GB, ~30 s) and returns the mapped copy. Disk use is
bounded: before a write, least recently used entries are deleted until the total fits ``$TT_GLM54_TIER1_CACHE_GB``
(default 50, two entries), and the write is skipped (with a warning; the generated weights are still returned) when
it would leave less than ``TIER1_CACHE_MIN_FREE_GB`` free on the disk. Fill the cache outside the device lock with
``python -m models.demos.deepseek_v3_d_p.tests.glm54_weights`` (host only, opens no device).
"""

import contextlib
import hashlib
import inspect
import json
import os
import shutil
import sys
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import reference as _mla_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import synthetic_gates as _synthetic_gates_module
from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import (
    GATE_BETA_CENTER,
    GATE_LOGIT_STD,
    GATE_NORM_GAIN_STD,
    GLM54_SYNTHETIC_SEED,
    SYNTHETIC_GATES_VERSION,
    gate_seed,
    synthetic_gates,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config, glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tests import glm_synthetic_weights as _layer_generator_module
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_transformer_state_dict
from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter

# GLM-5.4 = GLM-5.3 + loop fields (A8): tier 2 reads GLM-5.3's checkpoint and TTNN cache.
GLM53_CHECKPOINT = GLM53Adapter.hf_model_default
GLM53_TTNN_CACHE = GLM53Adapter.ttnn_cache_default
# Every per-layer category of weight_categories (no embedding / lm_head / mtp).
LAYER_CATEGORIES = ("norms", "mla", "dense_ffn", "moe_gate", "shared_experts", "routed_experts")


class GLM54ReducedConfig(GLM54Config):
    """Model constants of the tier-1 reduced stack (``model_cfg`` for TtPrefillTransformer / TtPrefillBlock)."""

    NUM_LAYERS = 4
    NUM_DENSE_LAYERS = 1
    LOOP_FIRST_LAYER = 1
    LOOP_LAST_LAYER = 2
    INDEX_TOPK_PATTERN = "FFSF"


def glm54_reduced_config(max_seq_len: int):
    """GLM-5.4 hf config of the tier-1 reduced stack (see the module docstring), at ``max_seq_len``."""
    m = GLM54ReducedConfig
    c = glm_5_4_hf_config(max_seq_len)
    c.num_hidden_layers = m.NUM_LAYERS
    c.first_k_dense_replace = m.NUM_DENSE_LAYERS
    c.indexer_types = m.indexer_types()
    c.loop_first_layer, c.loop_last_layer = m.LOOP_FIRST_LAYER, m.LOOP_LAST_LAYER
    return c


@dataclass
class Glm54Weights:
    """One weight set: ``state_dict`` in TtPrefillTransformer form (``embed_weight``, ``layers``: one
    TtPrefillBlock dict per layer, loop layers once), ``gates`` (one dict per loop pass, as
    ``glm54_stack_reference`` takes) and the ``provenance`` to record with any result built on it."""

    state_dict: dict
    gates: list[dict]
    provenance: dict = field(default_factory=dict)


def _gate_provenance(seed: int, loop_count: int) -> dict:
    return {
        "generator": "reference.glm_5_4.synthetic_gates.synthetic_gate_weights",
        "version": SYNTHETIC_GATES_VERSION,
        "seed": seed,
        "gate_seeds": [gate_seed(seed, k) for k in range(1, loop_count + 1)],
        "beta_center": {k: GATE_BETA_CENTER[k] for k in range(1, loop_count + 1)},
        "logit_std": GATE_LOGIT_STD,
        "norm_gain_std": GATE_NORM_GAIN_STD,
        "width": "scalar (per token)",
    }


def tier1_provenance(seed: int = GLM54_SYNTHETIC_SEED) -> dict:
    """What determines the tier-1 weights' bytes."""
    m = GLM54ReducedConfig
    return {
        "tier": 1,
        "config": "tests.glm54_weights.glm54_reduced_config",
        "layers": f"{m.NUM_LAYERS} (dense < {m.NUM_DENSE_LAYERS}, loop {m.LOOP_FIRST_LAYER}..{m.LOOP_LAST_LAYER}, "
        f"indexer {m.INDEX_TOPK_PATTERN})",
        "layer_generator": "tests.glm_synthetic_weights.glm_random_transformer_state_dict(share_experts=True)",
        "seed": seed,
        "gates": _gate_provenance(seed, m.LOOP_COUNT),
    }


def _generate_tier1(config, seed: int) -> tuple[dict, list[dict]]:
    """The tier-1 draw: (state_dict, gates). Its source is part of the cache key."""
    state_dict = glm_random_transformer_state_dict(
        config, GLM54ReducedConfig, config.num_hidden_layers, seed=seed, share_experts=True
    )
    return state_dict, synthetic_gates(config.hidden_size, config.loop_count, seed=seed)


# Bump when the entry layout changes.
TIER1_CACHE_FORMAT = 1
# hf-config fields the tier-1 generator does not read (test_glm54_weights records the reads), left out of the key.
TIER1_KEY_EXCLUDED_FIELDS = frozenset({"max_seq_len", "rope_scaling", "loop_window_format"})
# Where the tier-1 bytes come from: these modules hashed whole, plus single functions of larger modules.
_TIER1_GENERATOR_MODULES = (_layer_generator_module, _synthetic_gates_module, _mla_reference)
_TIER1_GENERATOR_FUNCTIONS = (_generate_tier1, indexer_layer_is_reused)
TIER1_CACHE_MIN_FREE_GB = 30


def tier1_cache_dir() -> Path:
    return Path(os.environ.get("TT_GLM54_CACHE_DIR") or Path.home() / ".cache" / "tt-glm54") / "tier1-weights"


def _tier1_cache_cap_bytes() -> int:
    return int(float(os.environ.get("TT_GLM54_TIER1_CACHE_GB", "50")) * 1e9)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tier1_key_inputs(config, seed: int = GLM54_SYNTHETIC_SEED) -> dict:
    """Everything the tier-1 bytes depend on (``tier1_cache_key`` hashes it; stored in each entry)."""
    m = GLM54ReducedConfig
    return {
        "format": TIER1_CACHE_FORMAT,
        "generator": {
            **{mod.__name__: _sha256_text(Path(mod.__file__).read_text()) for mod in _TIER1_GENERATOR_MODULES},
            **{f.__qualname__: _sha256_text(inspect.getsource(f)) for f in _TIER1_GENERATOR_FUNCTIONS},
        },
        "torch": torch.__version__,
        "config": {k: v for k, v in sorted(vars(config).items()) if k not in TIER1_KEY_EXCLUDED_FIELDS},
        "model_cfg": {k: getattr(m, k) for k in dir(m) if k.isupper()},
        "seed": seed,
    }


def _tier1_key_and_inputs(config, seed: int) -> tuple[str, str]:
    """(key, the JSON of ``tier1_key_inputs`` it hashes)."""
    inputs = json.dumps(tier1_key_inputs(config, seed), sort_keys=True, default=repr)
    return _sha256_text(inputs)[:24], inputs


def tier1_cache_key(config, seed: int = GLM54_SYNTHETIC_SEED) -> str:
    return _tier1_key_and_inputs(config, seed)[0]


def _tensor_bytes(obj) -> int:
    """Bytes ``torch.save`` writes for the tensors in ``obj``: each storage once (MoE layers share their experts)."""
    storages = {}

    def walk(x):
        if isinstance(x, torch.Tensor):
            st = x.untyped_storage()
            storages[st.data_ptr()] = st.nbytes()
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(obj)
    return sum(storages.values())


def _evict(root: Path, budget: int) -> None:
    """Delete least recently used entries until the entries total at most ``budget`` bytes; delete writers'
    temporary files older than an hour (crashed writers)."""
    for tmp in root.glob(".*.tmp"):
        with contextlib.suppress(FileNotFoundError):
            if time.time() - tmp.stat().st_mtime > 3600:
                tmp.unlink(missing_ok=True)
    entries = []  # (mtime, size, path); another process may evict concurrently
    for p in root.glob("*.pt"):
        with contextlib.suppress(FileNotFoundError):
            st = p.stat()
            entries.append((st.st_mtime, st.st_size, p))
    total = sum(size for _, size, _ in entries)
    for _, size, p in sorted(entries):
        if total <= budget:
            break
        total -= size
        p.unlink(missing_ok=True)
        logger.info(f"tier-1 cache: evicted {p.name}")


def _store(path: Path, entry: dict) -> bool:
    """Write ``entry`` to ``path`` atomically within the cache bounds; False (and a warning) when it is skipped."""
    root, size, cap = path.parent, _tensor_bytes(entry), _tier1_cache_cap_bytes()
    root.mkdir(parents=True, exist_ok=True)
    if size > cap:
        logger.warning(f"tier-1 cache: entry ({size / 1e9:.1f} GB) exceeds the cap ({cap / 1e9:.1f} GB); not cached")
        return False
    _evict(root, cap - size)
    free = shutil.disk_usage(root).free
    if free - size < TIER1_CACHE_MIN_FREE_GB * 1e9:
        logger.warning(
            f"tier-1 cache: writing {size / 1e9:.1f} GB would leave {(free - size) / 1e9:.1f} GB free on {root} "
            f"(< {TIER1_CACHE_MIN_FREE_GB} GB); not cached"
        )
        return False
    tmp = root / f".{path.name}.{os.getpid()}.tmp"
    try:
        torch.save(entry, tmp)
        os.replace(tmp, path)
    except Exception as e:  # torch.save reports I/O failures as RuntimeError
        logger.warning(f"tier-1 cache: write to {path} failed ({e!r}); not cached")
        return False
    finally:
        tmp.unlink(missing_ok=True)  # no-op after the rename; also runs on KeyboardInterrupt
    return True


def _load(path: Path) -> dict | None:
    """The entry at ``path``, memory-mapped; None when it is absent (or evicted by another process meanwhile)."""
    try:
        entry = torch.load(path, mmap=True, weights_only=True)
    except FileNotFoundError:
        return None
    with contextlib.suppress(FileNotFoundError):
        os.utime(path)  # recency for eviction
    return entry


# One object per key per process while a caller holds it: a second request (a test's CPU check after its device
# phase, a later test in the session) neither maps the entry again nor regenerates when the entry was not written.
_LIVE_TIER1: "weakref.WeakValueDictionary[str, Glm54Weights]" = weakref.WeakValueDictionary()


def glm54_tier1_weights(config, seed: int = GLM54_SYNTHETIC_SEED) -> Glm54Weights:
    """Seeded random weights for ``glm54_reduced_config``: embedding, 4 layers (MoE layers share one expert
        set), and the scalar synthetic gates. About 23 GB, served from the tier-1 disk cache (module docstring)
        when present; the returned tensors are then memory-mapped (copy on write). While a caller holds the returned
    object, the same key returns that object."""
    if config.num_hidden_layers != GLM54ReducedConfig.NUM_LAYERS:
        raise ValueError(f"tier 1 is the {GLM54ReducedConfig.NUM_LAYERS}-layer reduced stack, got {config}")
    key, key_inputs = _tier1_key_and_inputs(config, seed)
    live = _LIVE_TIER1.get(key)
    if live is not None:
        return live
    path = tier1_cache_dir() / f"{key}.pt"
    t0 = time.perf_counter()
    entry = _load(path)
    if entry is not None:
        logger.info(f"tier-1 weights: cache hit {path} ({time.perf_counter() - t0:.1f} s)")
    else:
        state_dict, gates = _generate_tier1(config, seed)
        logger.info(f"tier-1 weights: cache miss {key}, generated in {time.perf_counter() - t0:.1f} s")
        entry = {"key_inputs": key_inputs, "state_dict": state_dict, "gates": gates}
        t1 = time.perf_counter()
        if _store(path, entry):
            # Page-cache backed instead of ~23 GB of process memory (unless another process evicted it already).
            entry = _load(path) or entry
            logger.info(f"tier-1 weights: cached to {path} in {time.perf_counter() - t1:.1f} s")
    w = Glm54Weights(entry["state_dict"], entry["gates"], {**tier1_provenance(seed), "cache_key": key})
    _LIVE_TIER1[key] = w
    return w


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tier2_sources() -> tuple[Path, Path]:
    """(checkpoint, TTNN cache root) a tier-2 run reads, resolved as the GLM-5.3 runner resolves them
    (``MLAPrefillAdapter.build_runtime``'s ``hf_model_path`` / ``MLAPrefillAdapter.weight_cache_path``)."""
    checkpoint = os.environ.get("PREFILL_HF_MODEL") or os.environ.get(GLM53Adapter.env_var) or GLM53_CHECKPOINT
    cache = os.environ.get("PREFILL_TTNN_CACHE", GLM53_TTNN_CACHE)
    return Path(checkpoint), Path(cache)


def tier2_provenance(config, seed: int = GLM54_SYNTHETIC_SEED) -> dict:
    """What determines the tier-2 weights: the GLM-5.3 checkpoint in use (path, revision tag of its directory name,
    digests of its config and tensor index), the TTNN cache root device runs read, and the synthetic gates of
    ``config.loop_count`` passes."""
    ckpt, cache = tier2_sources()
    return {
        "tier": 2,
        "checkpoint": str(ckpt),
        "checkpoint_dir_tag": ckpt.name.rsplit("-", 1)[-1],
        "config_sha256": _sha256(ckpt / "config.json"),
        "index_sha256": _sha256(ckpt / "model.safetensors.index.json"),
        "ttnn_cache": str(cache),
        "layer_map": "GLM-5.4 layer L, any loop pass = GLM-5.3 layer L",
        "gates": _gate_provenance(seed, config.loop_count),
    }


def glm54_tier2_weights(config, seed: int = GLM54_SYNTHETIC_SEED) -> Glm54Weights:
    """Tier 2: the synthetic gates; ``state_dict`` is empty, the runner's cache-only form (shared tensors come from
    the GLM-5.3 TTNN cache on device, or ``glm54_checkpoint_layer`` for a reference check)."""
    gates = synthetic_gates(config.hidden_size, config.loop_count, seed=seed)
    return Glm54Weights({}, gates, tier2_provenance(config, seed))


def glm54_checkpoint_layer(config, layer_idx: int, *, categories=LAYER_CATEGORIES) -> dict:
    """GLM-5.4 layer ``layer_idx`` from the tier-2 checkpoint (``tier2_sources``) as one TtPrefillBlock dict (FP8
    dequantized to bf16), holding ``categories`` (the FFN categories that do not apply to the layer are dropped).
    Read only. ``config`` is the full-size GLM-5.4 hf config (``glm_5_4_hf_config``): its layer kinds must be the
    checkpoint's.

    A MoE layer with routed experts is ~19 GB of host RAM and minutes of reading; pass fewer categories when the
    check does not need them. A ``shared``-indexer layer's MLA dict carries no indexer weights."""
    from models.demos.deepseek_v3_d_p.tt.runners.weight_categories import load_trunk_state_dict

    if config.num_hidden_layers != GLM54Config.NUM_LAYERS:
        raise ValueError("glm54_checkpoint_layer needs the full-size config (the checkpoint's layer kinds)")
    moe = layer_idx >= config.first_k_dense_replace
    skip = {"dense_ffn"} if moe else {"moe_gate", "shared_experts", "routed_experts"}
    sd = load_trunk_state_dict(
        tier2_sources()[0],
        config,
        GLM54Config,
        missing={c: [layer_idx] for c in categories if c not in skip},
        first_layer_idx=layer_idx,
        num_layers=1,
    )
    return sd["layers"][0]


__all__ = [
    "GLM53_CHECKPOINT",
    "GLM53_TTNN_CACHE",
    "GLM54ReducedConfig",
    "Glm54Weights",
    "LAYER_CATEGORIES",
    "glm54_checkpoint_layer",
    "glm54_reduced_config",
    "glm54_tier1_weights",
    "glm54_tier2_weights",
    "tier1_cache_dir",
    "tier1_cache_key",
    "tier1_key_inputs",
    "tier1_provenance",
    "tier2_provenance",
    "tier2_sources",
]


if __name__ == "__main__":
    # Fill the tier-1 cache outside the device lock (host only: generation imports ttnn but opens no device).
    w = glm54_tier1_weights(glm54_reduced_config(8192))  # any length: max_seq_len is not in the key
    entry = tier1_cache_dir() / f"{w.provenance['cache_key']}.pt"
    print(f"tier-1 cache {'ready' if entry.is_file() else 'NOT written'}: {entry}")
    sys.exit(0 if entry.is_file() else 1)
