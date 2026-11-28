# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Load a Hugging Face reference model without exceeding host RAM.

Generated bring-up tests compare a TT port against the HF reference on CPU. Loading the whole
reference is fine for most models and stays the default. A model whose loaded weights exceed the
host's available RAM (e.g. ~314B parameters = ~628 GB in bf16 on a 566 GB host) was OOM-killed
instead. For those, a component test loads the reference with only as many repeated blocks as the
component needs, through HF's own loader (so its key renames and weight conversions still apply).

Nothing here names a model, a stage or a layer: sizes come from the model's own parameter count,
the repeated block is found by walking the module tree, and the config field that sets its length
is discovered by trying each candidate on a weightless skeleton.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Fraction of the host's currently available RAM a whole-model load may use (config, not code).
RAM_FRACTION_ENV = "TT_HF_REFERENCE_RAM_FRACTION"
_DEFAULT_RAM_FRACTION = 0.9

_DTYPE_BYTES = {"float32": 4, "float": 4, "bfloat16": 2, "float16": 2, "half": 2}


def host_ram_available_bytes() -> int:
    """Bytes the kernel reports as available to a new allocation (MemAvailable); 0 if unknown."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _ram_fraction() -> float:
    try:
        return float(os.environ.get(RAM_FRACTION_ENV, _DEFAULT_RAM_FRACTION))
    except ValueError:
        return _DEFAULT_RAM_FRACTION


def _dtype_bytes(dtype: Any) -> int:
    name = str(dtype).replace("torch.", "")
    return _DTYPE_BYTES.get(name, 2)


def _is_meta(model) -> bool:
    try:
        return any(p.device.type == "meta" for p in model.parameters())
    except Exception:
        return False


@functools.lru_cache(maxsize=16)
def _skeleton_cached(model_id: str, class_names: Tuple[str, ...], overrides: Tuple[Tuple[str, Any], ...]):
    return _build_skeleton(model_id, class_names, dict(overrides))


def _load_config(model_id: str, overrides: Optional[Dict[str, Any]] = None, trust_remote_code: bool = True, **kwargs):
    import transformers

    cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kwargs)
    for dotted, value in (overrides or {}).items():
        _set_config_path(cfg, dotted, value)
    return cfg


def _build_skeleton(
    model_id: str,
    class_names: Sequence[str],
    overrides: Dict[str, Any],
    trust_remote_code: bool = True,
    **config_kwargs,
):
    import transformers

    cfg = _load_config(model_id, overrides, trust_remote_code=trust_remote_code, **config_kwargs)
    try:
        from accelerate import init_empty_weights
    except Exception:  # noqa: BLE001
        init_empty_weights = None
    last_err: Optional[BaseException] = None
    for name in class_names:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            if init_empty_weights is None:
                return cls.from_config(cfg, trust_remote_code=trust_remote_code)
            # Parameters on `meta` (no memory); buffers stay real so computed tables keep their values.
            with init_empty_weights():
                return cls.from_config(cfg, trust_remote_code=trust_remote_code)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    # The ORIGINAL error, unwrapped: callers branch on its type (e.g. ImportError -> CPU-compat retry).
    raise last_err or RuntimeError(f"none of {list(class_names)} exists in this transformers")


def build_hf_skeleton(
    model_id: str, class_names: Sequence[str] = ("AutoModel",), trust_remote_code: bool = True, **config_kwargs
):
    """The model's full module tree with NO weight memory (parameters on `meta`), from its config.
    The first of ``class_names`` that builds wins (same cascade the callers' loaders use)."""
    return _build_skeleton(model_id, class_names, {}, trust_remote_code=trust_remote_code, **config_kwargs)


def loaded_weight_bytes(model_id: str, class_names: Sequence[str], dtype: Any = "bfloat16") -> int:
    """Bytes the reference occupies once loaded at ``dtype``: the model's own parameter count times
    the dtype width (a checkpoint's file size can be half of that, e.g. fp8-stored weights)."""
    try:
        sk = _skeleton_cached(model_id, tuple(class_names), ())
        return sum(p.numel() for p in sk.parameters()) * _dtype_bytes(dtype)
    except Exception:  # noqa: BLE001 -- unknown size: treat as fitting (the previous behavior)
        return 0


def fits_in_host_ram(model_id: str, class_names: Sequence[str], dtype: Any = "bfloat16") -> bool:
    need = loaded_weight_bytes(model_id, class_names, dtype)
    avail = host_ram_available_bytes()
    if not need or not avail:
        return True
    return need <= avail * _ram_fraction()


def _gb(n: int) -> str:
    return f"{n / 1e9:.0f} GB"


def require_whole_model_fits(model_id: str, class_names: Sequence[str], dtype: Any = "bfloat16") -> None:
    """Raise a clear error up front when a WHOLE-model CPU reference cannot fit host RAM, instead of
    letting the OOM killer end the process minutes into the load."""
    if fits_in_host_ram(model_id, class_names, dtype):
        return
    need = loaded_weight_bytes(model_id, class_names, dtype)
    avail = host_ram_available_bytes()
    raise MemoryError(
        f"{model_id}: the whole-model CPU reference needs ~{_gb(need)} of RAM at {dtype}, but this host "
        f"has ~{_gb(avail)} available ({RAM_FRACTION_ENV}={_ram_fraction()}). Per-component tests load "
        "only the blocks they need; a whole-model comparison needs a host with more RAM."
    )


# ----------------------------------------------------------------------------- path / config helpers


def _tokens(dotted: str) -> List[Any]:
    out: List[Any] = []
    for part in dotted.replace("[", ".").replace("]", "").split("."):
        if part:
            out.append(int(part) if part.isdigit() else part)
    return out


def _is_container(mod) -> bool:
    import torch

    return isinstance(mod, (torch.nn.ModuleList, torch.nn.Sequential))


def _outer_containers(model) -> Dict[str, int]:
    """Every OUTERMOST repeated-block container (a ModuleList/Sequential with no container above it)
    and its length. Only these are ever shrunk: a container nested inside a block (e.g. a MoE layer's
    experts) is sized by fields that also shape sibling weights, so shrinking it breaks loading."""
    out: Dict[str, int] = {}
    for name, mod in model.named_modules():
        if not name or not _is_container(mod) or len(mod) < 2:
            continue
        if any(name.startswith(c + ".") for c in out):
            continue
        out[name] = len(mod)
    return out


def _blocks_needed(model, dotted: str, containers: Dict[str, int]) -> Dict[str, int]:
    """{outer container: blocks needed} for one path: index+1 where the path indexes into it, 1 where
    the path ends on it (the tests then use element 0)."""
    toks = _tokens(dotted)
    out: Dict[str, int] = {}
    for c in containers:
        ct = _tokens(c)
        if toks[: len(ct)] != ct:
            continue
        nxt = toks[len(ct)] if len(toks) > len(ct) else None
        out[c] = (nxt + 1) if isinstance(nxt, int) else 1
    return out


def _int_config_fields(cfg, prefix: str = "") -> Iterable[Tuple[str, int]]:
    """Every int field of the config, recursing into sub-configs, as (dotted_path, value)."""
    try:
        items = vars(cfg).items()
    except TypeError:
        return
    for key, val in items:
        if key.startswith("_"):
            continue
        path = f"{prefix}{key}"
        if isinstance(val, bool):
            continue
        if isinstance(val, int):
            yield path, val
        elif hasattr(val, "to_dict") and hasattr(val, "model_type"):
            yield from _int_config_fields(val, path + ".")


def _set_config_path(cfg, dotted: str, value: Any) -> None:
    *parents, last = dotted.split(".")
    obj = cfg
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, last, value)


def _container_len(model, container_path: str) -> Optional[int]:
    cur = model
    for tok in _tokens(container_path):
        try:
            cur = cur[tok] if isinstance(tok, int) else getattr(cur, tok)
        except Exception:
            return None
    try:
        return len(cur)
    except TypeError:
        return None


def truncation_overrides(model_id: str, class_names: Sequence[str], paths: Sequence[str]) -> Dict[str, int]:
    """Config overrides that keep only the repeated blocks ``paths`` need (outermost containers no path
    touches keep one). For each container, the config field that sets its length is DISCOVERED: every int field equal to the
    container's length is tried on a weightless skeleton, and the one that actually shrinks that
    container is used. Returns {} when nothing can be shrunk."""
    sk = _skeleton_cached(model_id, tuple(class_names), ())
    cfg = sk.config
    containers = _outer_containers(sk)
    # Containers no path needs keep one block; the rest keep as many as the deepest path needs.
    need: Dict[str, Tuple[int, int]] = {c: (n, 1) for c, n in containers.items()}
    for path in paths:
        for container, blocks in _blocks_needed(sk, path, containers).items():
            length, have = need[container]
            need[container] = (length, max(blocks, have))
    overrides: Dict[str, int] = {}
    fields = list(_int_config_fields(cfg))
    for container, (length, blocks) in need.items():
        if blocks >= length:
            continue
        for field, val in fields:
            if val != length or field in overrides:
                continue
            try:
                trial = _skeleton_cached(
                    model_id, tuple(class_names), tuple(sorted({**overrides, field: blocks}.items()))
                )
            except Exception:  # noqa: BLE001 -- this field cannot take that value; try the next
                continue
            if _container_len(trial, container) == blocks:
                overrides[field] = blocks
                break
    return overrides


def load_hf_reference(
    model_id: str,
    class_names: Sequence[str],
    needed_paths: Sequence[str] = (),
    dtype: Any = "bfloat16",
    **from_pretrained_kwargs,
) -> Tuple[Any, Optional[BaseException], Dict[str, int]]:
    """(model, last_error, overrides). The whole model when it fits host RAM (the previous behavior,
    unchanged) or when no ``needed_paths`` are given; otherwise only the repeated blocks those paths
    need. ``model`` is None when every class failed (``last_error`` says why)."""
    import transformers

    kwargs = dict(trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    kwargs.update(from_pretrained_kwargs)
    overrides: Dict[str, int] = {}
    if needed_paths and not fits_in_host_ram(model_id, class_names, dtype):
        overrides = truncation_overrides(model_id, class_names, needed_paths)
        if overrides:
            need = loaded_weight_bytes(model_id, class_names, dtype)
            print(
                f"[reference] {model_id}: whole model ~{_gb(need)} > available RAM "
                f"~{_gb(host_ram_available_bytes())}; loading only the blocks this test needs ({overrides})",
                flush=True,
            )
            kwargs["config"] = _load_config(model_id, overrides)
    last_err: Optional[BaseException] = None
    for name in class_names:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            return model, None, overrides
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    return None, last_err, overrides
