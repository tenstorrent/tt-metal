# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight initialization and loading for the CPU reference layers (``IndexerCPU`` /
``MLACPU`` in ``model.py``).

Two entry points, both via the free function ``initialize_weights(module, ...)``:
  * random init (default) — normal(0, 0.02) on every ``Linear``; norms keep ones.
  * pretrained load — read a layer's attention tensors from DeepSeek-V3.2-Exp
    safetensors shard file(s), map HF (transformers-style) names to our module
    names (mirrors ``inference/convert.py``), and dequantize fp8 → bf16.

Pretrained shards are fetched just-in-time (and cached) by ``load_pretrained_hf`` /
``resolve_layer_shards``. This module depends on ``model.py`` (for the module/layer
types); ``model.py`` does not import it, so there is no circular import.
"""

import torch
import torch.nn as nn
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import BLOCK_SIZE, IndexerCPU, Linear

DEFAULT_REPO = "deepseek-ai/DeepSeek-V3.2-Exp"
_INDEX_FILE = "model.safetensors.index.json"

# Single source of truth for MLA/indexer weight naming across the conventions in play:
#   * canonical / device  — what ttMLA + SparseMLAReference consume. Equals the HF-checkpoint name for
#     every tensor EXCEPT the indexer LayerNorm bias, which the device keeps as a separate
#     ``indexer.k_norm_bias.weight`` tensor (γ and β split), where the checkpoint/CPU store it as the
#     LayerNorm's native ``indexer.k_norm.bias``.
#   * CPU port            — MLACPU/IndexerCPU module parameter names (``wq_a``, ``q_norm``, ...).
# CANONICAL_TO_CPU is the authority; the HF-checkpoint reader map (_HF_TO_MLA) is derived from it.
CANONICAL_TO_CPU = {
    "q_a_proj.weight": "wq_a.weight",
    "q_a_layernorm.weight": "q_norm.weight",
    "q_b_proj.weight": "wq_b.weight",
    "kv_a_proj_with_mqa.weight": "wkv_a.weight",
    "kv_a_layernorm.weight": "kv_norm.weight",
    "kv_b_proj.weight": "wkv_b.weight",
    "o_proj.weight": "wo.weight",
    "indexer.wq_b.weight": "indexer.wq_b.weight",
    "indexer.wk.weight": "indexer.wk.weight",
    "indexer.k_norm.weight": "indexer.k_norm.weight",
    "indexer.k_norm_bias.weight": "indexer.k_norm.bias",
    "indexer.weights_proj.weight": "indexer.weights_proj.weight",
}

# Kimi-K3 gated MLA only: the output gate. Kept OUT of CANONICAL_TO_CPU so ungated variants (DS-V3.2, GLM)
# never require it; callers pick the key set for a config with canonical_to_cpu(gated=...).
GATE_CANONICAL_TO_CPU = {"g_proj.weight": "g_proj.weight"}


def canonical_to_cpu(gated: bool = False) -> dict[str, str]:
    """Canonical -> CPU-port name map for one MLA flavour: ``CANONICAL_TO_CPU`` plus ``g_proj`` when gated."""
    return {**CANONICAL_TO_CPU, **GATE_CANONICAL_TO_CPU} if gated else dict(CANONICAL_TO_CPU)


# HF (transformers-style) submodule name -> CPU param prefix, derived from CANONICAL_TO_CPU (mirrors
# inference/convert.py). The HF-checkpoint key equals the canonical key for every tensor except the
# indexer LayerNorm bias, which rides under the ``indexer.k_norm`` submodule (so it's excluded here —
# its prefix already comes from the ``indexer.k_norm.weight`` row). fp8 weights carry a companion
# ``*.weight_scale_inv``.
_HF_TO_MLA = {
    canon[: -len(".weight")]: cpu[: -len(".weight")]
    for canon, cpu in canonical_to_cpu(gated=True).items()
    if canon != "indexer.k_norm_bias.weight"
}


def _dequant_fp8(weight: torch.Tensor, scale_inv: torch.Tensor, block: int = BLOCK_SIZE) -> torch.Tensor:
    """
    Reconstruct a bf16 weight from a blockwise fp8 (e4m3) checkpoint tensor and
    its per-(block x block) scale, matching model.py:weight_dequant. ``scale_inv``
    has shape ``[ceil(O/block), ceil(I/block)]``.
    """
    w = weight.float()
    s = scale_inv.repeat_interleave(block, 0)[: w.shape[0]].repeat_interleave(block, 1)[:, : w.shape[1]]
    return (w * s).to(torch.bfloat16)


def _attention_prefixes(layer: int) -> tuple[str, ...]:
    """Checkpoint key roots for one layer's attention: text-only models use ``model.layers.N``; multimodal
    wrappers (Kimi-K3 / K3.1: ``KimiK3ForConditionalGeneration``) nest the LM under ``language_model.``."""
    return (f"model.layers.{layer}.self_attn.", f"language_model.model.layers.{layer}.self_attn.")


def load_attention_state_dict(shard_paths, layer: int) -> dict:
    """
    Read one transformer layer's attention (MLA + indexer) tensors from the given
    DeepSeek-V3.2-Exp safetensors shard file(s) and return a state dict keyed to
    ``MLACPU``'s parameter names (indexer params keep the ``indexer.`` prefix).

    fp8 weights are dequantized to bf16; norms and the indexer's ``weights_proj``
    are cast to fp32 to match the module dtypes. Shard paths come from
    ``resolve_layer_shards`` (which downloads on demand).
    """
    from safetensors import safe_open

    if isinstance(shard_paths, str):
        shard_paths = [shard_paths]
    prefixes = _attention_prefixes(layer)
    state = {}
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                prefix = next((p for p in prefixes if k.startswith(p)), None)
                if prefix is None:
                    continue
                if k.endswith(".weight_scale_inv"):
                    continue
                sub, field = k[len(prefix) :].rsplit(".", 1)  # ("q_a_proj","weight") / ("indexer.wk","weight")
                if sub not in _HF_TO_MLA:
                    continue
                t = f.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn:
                    t = _dequant_fp8(t, f.get_tensor(k.replace(".weight", ".weight_scale_inv")))
                key = f"{_HF_TO_MLA[sub]}.{field}"
                # Match module dtypes: norms + weights_proj fp32, linear weights bf16.
                t = t.float() if ("norm" in key or "weights_proj" in key) else t.to(torch.bfloat16)
                state[key] = t
    if not state:
        raise ValueError(f"No layer-{layer} attention tensors found in {shard_paths}.")
    return state


def _check_pretrained_load(missing, unexpected, what: str):
    """Validate a strict=False load_state_dict result for a pretrained load."""
    if missing:
        raise RuntimeError(f"{what}: parameters missing from checkpoint: {missing}")
    if unexpected:
        logger.warning(f"{what}: ignoring unexpected checkpoint keys: {unexpected}")


def init_random(module: nn.Module):
    """
    Random-init every ``Linear`` in ``module`` (normal 0, 0.02); norms keep their
    default (ones) value. Works for both ``IndexerCPU`` and ``MLACPU`` (whose walk
    also covers the nested indexer, in registration order).
    """
    name = type(module).__name__
    logger.info(f"Initializing {name} weights randomly for testing")
    for _, m in module.named_modules():
        if isinstance(m, Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.02)
    logger.info(f"✓ {name} random weight initialization complete")


def load_pretrained(module: nn.Module, shard_paths, layer: int = 0):
    """
    Load pretrained weights for ``layer`` into ``module`` from DeepSeek-V3.2-Exp
    safetensors shard file(s). ``MLACPU`` gets MLA + nested indexer; a standalone
    ``IndexerCPU`` gets just the indexer tensors (``indexer.`` prefix stripped).
    """
    name = type(module).__name__
    logger.info(f"Loading pretrained {name} weights (layer {layer}) from {shard_paths}")
    sd = load_attention_state_dict(shard_paths, layer)
    if isinstance(module, IndexerCPU):
        sd = {k[len("indexer.") :]: v for k, v in sd.items() if k.startswith("indexer.")}
    missing, unexpected = module.load_state_dict(sd, strict=False)
    _check_pretrained_load(missing, unexpected, f"{name} layer {layer}")
    logger.info(f"✓ Loaded pretrained {name} weights")


def initialize_weights(
    module: nn.Module,
    layer: int = None,
    checkpoint_path=None,
    repo: str = DEFAULT_REPO,
    token: str = None,
    local_files_only: bool = False,
):
    """
    The single weight-init entry point. Modes:

    * **random** (default) — neither ``layer`` nor ``checkpoint_path`` given.
    * **pretrained from HF** — ``layer`` given: resolve + (JIT) download that layer's
      shard(s) and load them. ``local_files_only=True`` uses the cache only and raises
      ``LocalEntryNotFoundError`` if a shard is absent (so callers can skip).
    * **pretrained from a local shard** — ``checkpoint_path`` (a shard path or list)
      for ``layer`` (default 0); takes precedence over HF resolution.
    """
    if checkpoint_path is not None:
        load_pretrained(module, checkpoint_path, layer or 0)
    elif layer is not None:
        load_pretrained_hf(module, layer, repo, token, local_files_only)
    else:
        init_random(module)


def _layer_cache_path(repo: str, revision: str, layer: int):
    from pathlib import Path

    from huggingface_hub.constants import HF_HUB_CACHE

    # Keyed by the checkpoint REVISION too, so a repo update never reuses an older extract.
    root = Path(HF_HUB_CACHE) / "tt_attention_extracts" / repo.replace("/", "--") / revision
    return root / f"layer{layer}_self_attn.safetensors"


_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
}


def _wanted_attention_key(key: str, prefix: str) -> bool:
    """Only the tensors the loader consumes: mapped MLA / indexer submodules and their fp8 scales."""
    sub = key[len(prefix) :].rsplit(".", 1)[0]
    return sub in _HF_TO_MLA


def _read_safetensors_subset(f, keys) -> dict:
    """Read `keys` from an open (seekable) safetensors file object using the header's byte offsets."""
    import json
    import struct

    (header_len,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(header_len))
    base = 8 + header_len  # data_offsets are relative to the end of the header
    out = {}
    for k in keys:
        meta = header[k]
        dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"{k}: unsupported safetensors dtype {meta['dtype']} for an attention extract")
        begin, end = meta["data_offsets"]
        f.seek(base + begin)
        raw = bytearray(f.read(end - begin))
        out[k] = torch.frombuffer(raw, dtype=dtype).reshape(meta["shape"]).clone()
    return out


def extract_layer_attention(repo: str, layer: int, token: str = None, local_files_only: bool = False, fs=None):
    """Fetch ONLY ``layer``'s attention (MLA + indexer) tensors from ``repo`` into one small local safetensors file.

    Range-reads each tensor's bytes from the remote shards (safetensors headers carry byte offsets), so a
    multi-TB checkpoint whose shards are tens of GB each (Kimi-K3.1: 96 shards) costs only the layer's
    attention bytes (~0.5 GB for a K3.1 DSA layer). Cached under the HF cache, per checkpoint revision;
    returns the local path. ``fs`` overrides the remote filesystem (tests).
    """
    import json
    import os
    import uuid
    from pathlib import Path

    from huggingface_hub import HfFileSystem, hf_hub_download
    from safetensors.torch import save_file

    # The (small) index pins the revision; with local_files_only it must already be cached too.
    index = hf_hub_download(repo, _INDEX_FILE, token=token, local_files_only=local_files_only)
    revision = Path(index).parent.name  # .../snapshots/<revision>/model.safetensors.index.json
    out = _layer_cache_path(repo, revision, layer)
    if out.exists():
        return str(out)
    if local_files_only:
        from huggingface_hub.errors import LocalEntryNotFoundError

        raise LocalEntryNotFoundError(f"no cached layer-{layer} attention extract for {repo}@{revision} at {out}")

    weight_map = json.load(open(index))["weight_map"]
    by_shard = {}
    for k, shard in weight_map.items():
        prefix = next((p for p in _attention_prefixes(layer) if k.startswith(p)), None)
        if prefix is not None and _wanted_attention_key(k, prefix):
            by_shard.setdefault(shard, []).append(k)
    found = {k for keys in by_shard.values() for k in keys}
    if not any(k.endswith("self_attn.q_a_proj.weight") for k in found):
        raise ValueError(f"layer {layer} of {repo} has no MLA attention tensors (a non-MLA layer, e.g. KDA?)")

    fs = fs or HfFileSystem(token=token)
    tensors = {}
    logger.info(f"Range-reading {len(found)} layer-{layer} attention tensors of {repo} from {len(by_shard)} shard(s)")
    for shard, keys in sorted(by_shard.items()):
        # Pinned to the index's revision, so a repo update mid-extract cannot mix revisions under one cache key.
        # Buffered random access (block_size=0 would stream and could not seek).
        with fs.open(f"{repo}@{revision}/{shard}" if isinstance(fs, HfFileSystem) else shard, "rb") as f:
            tensors.update(_read_safetensors_subset(f, keys))
    out.parent.mkdir(parents=True, exist_ok=True)
    # A unique temp file per writer + atomic replace: concurrent extractors never expose a partial file and
    # never rename each other's temp away.
    # O_EXCL on a unique name (no shared state); mode 0o666 lets the kernel apply the umask, so a shared HF cache
    # stays readable by other users (mkstemp would create 0600) without touching the process-wide umask.
    tmp = str(out.parent / f".{out.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        try:
            publish_mode = os.fstat(fd).st_mode & 0o777  # umask/ACL-derived; the process umask is untouched
        finally:
            os.close(fd)
        save_file(tensors, tmp)
        os.chmod(tmp, publish_mode)  # save_file re-creates the file 0600; restore the shareable mode
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    logger.info(f"Saved {len(tensors)} layer-{layer} attention tensors to {out}")
    return str(out)


def resolve_layer_shards(
    layer: int = 0,
    repo: str = DEFAULT_REPO,
    token: str = None,
    local_files_only: bool = False,
    extract_only: bool = False,
):
    """
    Return local path(s) to the safetensors shard(s) that hold ``layer``'s attention
    (MLA + indexer) weights. Missing files are downloaded just-in-time, in full, via
    ``hf_hub_download`` (cached, atomic) — a warning is emitted first since the shards
    are multi-GB. With ``local_files_only=True`` only cached files are used and
    ``LocalEntryNotFoundError`` is raised if a shard is absent. ``extract_only=True`` fetches just the
    layer's attention tensors instead (``extract_layer_attention``), for checkpoints whose shards are too
    large to download whole.
    """
    import json

    if extract_only:
        return [extract_layer_attention(repo, layer, token=token, local_files_only=local_files_only)]

    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    index = hf_hub_download(repo, _INDEX_FILE, token=token, local_files_only=local_files_only)
    weight_map = json.load(open(index))["weight_map"]
    prefixes = _attention_prefixes(layer)
    shards = sorted({weight_map[k] for k in weight_map if k.startswith(prefixes)})
    if not shards:
        raise ValueError(f"No attention tensors for layer {layer} in {repo} index.")

    if not local_files_only:
        missing = [s for s in shards if not isinstance(try_to_load_from_cache(repo, s), str)]
        if missing:
            logger.warning(
                f"Downloading {len(missing)} shard(s) for layer {layer} from {repo} "
                f"({', '.join(missing)}); this can take a while — the shards are multi-GB."
            )
    return [hf_hub_download(repo, s, token=token, local_files_only=local_files_only) for s in shards]


def load_pretrained_hf(
    module: nn.Module, layer: int = 0, repo: str = DEFAULT_REPO, token: str = None, local_files_only: bool = False
):
    """
    Resolve (downloading if needed) and load ``layer``'s pretrained weights from the
    DeepSeek-V3.2-Exp HF repo into ``module``. Convenience wrapper over
    ``resolve_layer_shards`` + ``load_pretrained``.
    """
    load_pretrained(module, resolve_layer_shards(layer, repo, token, local_files_only), layer)
