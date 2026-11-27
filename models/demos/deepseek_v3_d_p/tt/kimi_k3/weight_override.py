# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in weight override for Kimi-K3.1 prefill: chosen layers take GPTQ'd tensors instead of the checkpoint's.

Off unless K3_WEIGHT_OVERRIDE_DIR is set (production: every hook below is one module-global check). The directory is
the GPTQ tools' pipeline output (tt-blaze `tools/k31_gptq/pipeline.py --out`), read as is:

    <dir>/layer_LL/DONE                       the layer is complete (only layers with it are overridden)
    <dir>/layer_LL/carriers.safetensors       bf16 carriers on their grid, HF [out, in], keyed by checkpoint name
    <dir>/layer_LL/carriers_alt.safetensors   "<...>.b_proj.weight:joint": KDA b_proj on prefill's block grouping
    <dir>/layer_LL/carriers_folded.safetensors  routed_expert_up_proj with routed_expert_norm folded in (on the grid)
    <dir>/layer_LL/experts/shard_*.safetensors  routed experts w1 / w3 / w2 (bf4 grid), by checkpoint name

Keys may carry any root ("language_model.model.", "model."): a tensor is matched by `layers.L.<rest>`.

    K3_WEIGHT_OVERRIDE_DIR      the directory above
    K3_WEIGHT_OVERRIDE_LAYERS   comma list of layers to override (default: every layer_LL with DONE); each must have DONE
    K3_WEIGHT_OVERRIDE_VERIFY   "1": check that every bf4 / bf8 conversion made while an overridden layer is built is the
                                identity on its input (no re-rounding), and log a per-layer tally at exit
    K3_WEIGHT_OVERRIDE_VERIFY_OUT   also write that tally (json) here

How a layer is overridden (so no stale tensorbin can stand in for a carrier):
* every read of the dequantized checkpoint for that layer (`weights.load_tensors`, the KDA loader, the bf8 routed
  expert reader) returns the carrier where the override has one, the checkpoint's tensor otherwise;
* the layer is built from that state dict with the TTNN weight cache bypassed (`TtKimiK3Transformer` passes no
  `weight_cache_path` for it), so nothing is read from or written to the cache for it;
* its routed experts are laid out from the carrier shards (bf4, or bf8 where K3_ACC_ROUTED_BF8 covers the layer: the
  same values), not from TtMoe's bf4 cache;
* KDA b_proj takes the ":joint" copy (prefill packs 16 consecutive heads of a TP rank per block; the per-element copy
  would be re-rounded), and routed_expert_up_proj the folded carrier with routed_expert_norm replaced by ones (prefill
  folds that gamma in fp32 before packing bf8, so the unfolded carrier would be re-rounded).

Inexact by design in prefill (logged, not refused): KDA f_b_proj (prefill multiplies it by exp(A_log) and rounds to bf16
on device after loading it exactly).
"""

from __future__ import annotations

import atexit
import os
import re
import threading
from pathlib import Path

import torch
from loguru import logger

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
JOINT = ":joint"
LATENT_UP = "block_sparse_moe.routed_expert_up_proj.weight"
LATENT_NORM = "block_sparse_moe.routed_expert_norm.weight"
INEXACT_IN_PREFILL = ("self_attn.f_b_proj.weight",)

OVERRIDE = None  # the active WeightOverride, None in production
_INIT = False


def active():
    """The active override (K3_WEIGHT_OVERRIDE_DIR set), else None."""
    global OVERRIDE, _INIT
    if not _INIT:
        _INIT = True
        root = os.environ.get("K3_WEIGHT_OVERRIDE_DIR")
        if root:
            OVERRIDE = WeightOverride(Path(root))
    return OVERRIDE


def split_key(key: str):
    """(layer, layer-relative name) of a checkpoint key, or None for a non-layer key."""
    m = _LAYER_RE.search(key)
    return (int(m.group(1)), m.group(2)) if m else None


def lookup(key: str):
    """The override's tensor for this full checkpoint key, or None (no override, other layer, or not overridden)."""
    ov = active()
    if ov is None:
        return None
    parts = split_key(key)
    return None if parts is None else ov.tensor(*parts)


def has_layer(layer_idx: int) -> bool:
    ov = active()
    return ov is not None and int(layer_idx) in ov.layers


class WeightOverride:
    def __init__(self, root: Path):
        self.root = root
        done = {int(p.parent.name.split("_")[1]) for p in root.glob("layer_*/DONE")}
        want = [x.strip() for x in os.environ.get("K3_WEIGHT_OVERRIDE_LAYERS", "").split(",") if x.strip()]
        if want:
            layers = {int(x) for x in want}
            missing = sorted(layers - done)
            assert not missing, f"K3_WEIGHT_OVERRIDE_LAYERS {missing} have no DONE in {root}"
        else:
            layers = done
        self.layers = layers
        self.verify = os.environ.get("K3_WEIGHT_OVERRIDE_VERIFY", "") not in ("", "0")
        self._dense: dict = {}  # layer -> {rel name: bf16 tensor}
        self._experts: dict = {}  # layer -> {rel name: shard path}
        self._lock = threading.Lock()
        self.tally: dict = {}  # layer -> {"exact": n, "inexact": [(what, shape, n differing)]}
        self.building = None  # the layer being built (verify)
        logger.warning(f"K3_WEIGHT_OVERRIDE: layers {sorted(layers)} from {root} (verify {self.verify})")

    def _layer_dir(self, layer: int) -> Path:
        return self.root / f"layer_{layer:02d}"

    def dense(self, layer: int) -> dict:
        """Every non-expert carrier of an override layer, by layer-relative name (as the checkpoint names it)."""
        from safetensors import safe_open

        with self._lock:
            got = self._dense.get(layer)
            if got is not None:
                return got
            got, folded = {}, False
            d = self._layer_dir(layer)
            for f in ("carriers", "carriers_alt", "carriers_folded"):  # (later files win: joint b_proj, folded up)
                path = d / f"{f}.safetensors"
                if not path.is_file():
                    continue
                with safe_open(str(path), framework="pt", device="cpu") as h:
                    for k in h.keys():
                        parts = split_key(k)
                        assert parts and parts[0] == layer, f"{path}: key {k} is not layer {layer}'s"
                        rel = parts[1]
                        if f == "carriers_alt":
                            if not rel.endswith(JOINT):  # (bf4 alternates of MLA o_proj / kv_b: not used here)
                                continue
                            rel = rel[: -len(JOINT)]  # (b_proj on prefill's grouping replaces the per-element copy)
                        elif f == "carriers_folded":
                            if rel != LATENT_UP:  # (decode's folded copies: prefill applies those gammas itself)
                                continue
                            folded = True
                        got[rel] = h.get_tensor(k)
            if folded:
                # The folded carrier already holds routed_expert_norm, and prefill folds that norm again: give it ones.
                got[LATENT_NORM] = torch.ones(got[LATENT_UP].shape[1], dtype=torch.bfloat16)
            elif LATENT_UP in got:
                logger.warning(
                    f"K3_WEIGHT_OVERRIDE: layer {layer} latent up has no folded carrier (prefill re-rounds it)"
                )
            for rel in INEXACT_IN_PREFILL:
                if rel in got:
                    logger.warning(f"K3_WEIGHT_OVERRIDE: layer {layer} {rel} is folded on device (inexact by design)")
            self._dense[layer] = got
            logger.warning(f"K3_WEIGHT_OVERRIDE: layer {layer}: {len(got)} dense tensors")
            return got

    def expert_index(self, layer: int) -> dict:
        """{layer-relative expert key: shard file} of an override layer (empty when it has no expert shards)."""
        from safetensors import safe_open

        with self._lock:
            got = self._experts.get(layer)
            if got is not None:
                return got
            got = {}
            for path in sorted((self._layer_dir(layer) / "experts").glob("shard_*.safetensors")):
                with safe_open(str(path), framework="pt", device="cpu") as h:
                    for k in h.keys():
                        parts = split_key(k)
                        assert parts and parts[0] == layer, f"{path}: key {k} is not layer {layer}'s"
                        got[parts[1]] = path
            self._experts[layer] = got
            if got:
                logger.warning(f"K3_WEIGHT_OVERRIDE: layer {layer}: {len(got)} routed expert tensors")
            return got

    def has_experts(self, layer: int) -> bool:
        return int(layer) in self.layers and bool(self.expert_index(int(layer)))

    def tensor(self, layer: int, rel: str):
        if layer not in self.layers:
            return None
        if rel.startswith("block_sparse_moe.experts."):
            path = self.expert_index(layer).get(rel)
            if path is None:
                return None
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as h:
                for k in h.keys():
                    if k.endswith(f"layers.{layer}.{rel}"):
                        return h.get_tensor(k)
            return None
        return self.dense(layer).get(rel)

    # --- verification (K3_WEIGHT_OVERRIDE_VERIFY) ---

    def check(self, what: str, x: torch.Tensor, dtype) -> None:
        """Record whether packing `x` to `dtype` (bfloat4_b / bfloat8_b, TILE) is the identity: the host packer's round
        trip, ttnn's own code. Called on the host tensor a module hands the packer, in the layout it hands it."""
        import ttnn

        if not self.verify or self.building is None or dtype not in (ttnn.bfloat4_b, ttnn.bfloat8_b):
            return
        src = x.detach().float().contiguous()
        t = self.tally.setdefault(self.building, {"exact": 0, "inexact": [], "unchecked": []})
        if src.dim() < 2 or src.shape[-1] % 32 or src.shape[-2] % 32:
            t["unchecked"].append((what, tuple(src.shape)))  # (tile padding on the device side: not modelled here)
            return
        pack = _ORIG.get("from_torch", ttnn.from_torch)
        back = ttnn.to_torch(pack(src.reshape(-1, src.shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT))
        back = back.float().reshape(src.shape)
        n = int((back != src).sum())
        if n == 0:
            t["exact"] += 1
        else:
            t["inexact"].append((what, tuple(src.shape), str(dtype).split(".")[-1], n))

    def verify_report(self) -> dict:
        return {
            layer: {"exact": t["exact"], "inexact": t["inexact"], "unchecked": t["unchecked"]}
            for layer, t in sorted(self.tally.items(), key=lambda kv: str(kv[0]))
        }


def _report_at_exit():
    """K3_WEIGHT_OVERRIDE_VERIFY: log the tally (the MoE weights are made at the first forward, so this runs at exit) and
    write it to K3_WEIGHT_OVERRIDE_VERIFY_OUT when that is set."""
    ov = OVERRIDE
    if ov is None or not ov.verify:
        return
    rep = ov.verify_report()
    for layer, t in rep.items():
        logger.warning(
            f"K3_WEIGHT_OVERRIDE_VERIFY layer {layer}: {t['exact']} bf4/bf8 conversions exact, "
            f"{len(t['inexact'])} re-rounded {t['inexact']}, {len(t['unchecked'])} unchecked {t['unchecked']}"
        )
    out = os.environ.get("K3_WEIGHT_OVERRIDE_VERIFY_OUT")
    if out:
        import json

        Path(out).write_text(json.dumps({str(k): v for k, v in rep.items()}, indent=1))


atexit.register(_report_at_exit)


class building:
    """`with weight_override.building(layer):` - bf4/bf8 conversions inside are checked (K3_WEIGHT_OVERRIDE_VERIFY)."""

    def __init__(self, layer):
        self.layer = layer

    def __enter__(self):
        ov = active()
        self.prev = None
        if ov is not None and ov.verify and has_layer(self.layer):
            self.prev, ov.building = ov.building, self.layer
            _patch_converters()
        return self

    def __exit__(self, *exc):
        ov = active()
        if ov is not None and ov.verify and has_layer(self.layer):
            ov.building = self.prev


_PATCHED = False
_ORIG: dict = {}  # the unpatched converters (check() packs with these)


def _patch_converters():
    """Route ttnn.from_torch / ttnn.as_tensor through `check` while a layer is being built (verify mode only)."""
    global _PATCHED
    if _PATCHED:
        return
    import ttnn

    _PATCHED = True
    orig_from, orig_as = ttnn.from_torch, ttnn.as_tensor
    _ORIG.update(from_torch=orig_from, as_tensor=orig_as)

    def from_torch(tensor, *args, **kwargs):
        ov = active()
        if ov is not None and ov.building is not None and isinstance(tensor, torch.Tensor):
            ov.check("from_torch", tensor, kwargs.get("dtype", args[0] if args else None))
        return orig_from(tensor, *args, **kwargs)

    def as_tensor(tensor, *args, **kwargs):
        ov = active()
        if ov is not None and ov.building is not None and isinstance(tensor, torch.Tensor):
            ov.check(f"as_tensor {kwargs.get('cache_file_name', '')}", tensor, kwargs.get("dtype"))
        return orig_as(tensor, *args, **kwargs)

    ttnn.from_torch, ttnn.as_tensor = from_torch, as_tensor
