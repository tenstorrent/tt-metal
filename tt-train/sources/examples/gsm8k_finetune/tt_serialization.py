# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pathlib
import json
import tempfile
import datetime
import numpy as np
from typing import Optional, Dict
import ttml
import os
import hashlib
import logging


# -----------------------------
# Checkpoint helpers (MODEL ONLY)
# -----------------------------
def _ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def json_sha1(d) -> str:
    # Accept dict or list for convenience
    s = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _atomic_write_bytes(path: str, writer_fn):
    """
    Atomically write binary content to `path`.
    `writer_fn` takes an open binary file handle and must write all bytes to it.
    """
    p = pathlib.Path(path)
    _ensure_dir(str(p.parent))
    with tempfile.NamedTemporaryFile(
        mode="wb", delete=False, dir=str(p.parent), prefix=p.name + ".", suffix=".tmp"
    ) as f:
        tmp_name = f.name
        writer_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_name, str(p))  # atomic on POSIX


def _atomic_write_text(path: str, writer_fn):
    """
    Atomically write text content to `path`.
    `writer_fn` takes an open text file handle and must write all text to it.
    """
    p = pathlib.Path(path)
    _ensure_dir(str(p.parent))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(p.parent),
        prefix=p.name + ".",
        suffix=".tmp",
    ) as f:
        tmp_name = f.name
        writer_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_name, str(p))  # atomic on POSIX


def _write_manifest(ckpt_dir: str, manifest: dict):
    def _writer(fh):
        json.dump(manifest, fh, indent=2)

    _atomic_write_text(_manifest_path(ckpt_dir), _writer)


def _save_model_npz(
    model, ckpt_dir: str, step: int, extra: Optional[dict] = None
) -> str:
    _ensure_dir(ckpt_dir)
    arrays = _collect_param_arrays(model)
    ckpt_name = _default_ckpt_name(step)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    def _writer(fh):
        # Write the .npz into the already-open file handle (binary)
        np.savez_compressed(fh, **arrays)

    _atomic_write_bytes(ckpt_path, _writer)

    # update manifest
    man = _read_manifest(ckpt_dir)
    hist = man.setdefault("history", [])
    entry = {
        "file": ckpt_name,
        "step": int(step),
        "saved_at": _now_iso(),
        "num_params": len(arrays),
        "sha1_of_names": json_sha1(sorted(list(arrays.keys()))),
    }
    if extra:
        entry.update({"extra": extra})
    hist.append(entry)
    man["last"] = entry
    _write_manifest(ckpt_dir, man)
    return ckpt_path


def _manifest_path(ckpt_dir: str):
    return os.path.join(ckpt_dir, "manifest.json")


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _list_checkpoints(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return []
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".npz")]
    return sorted(files)


def _default_ckpt_name(step: int) -> str:
    return f"model_step_{step:08d}.npz"


def _write_manifest(ckpt_dir: str, manifest: dict):
    def _writer(fh):
        json.dump(manifest, fh, indent=2)

    _atomic_write_text(_manifest_path(ckpt_dir), _writer)


def _read_manifest(ckpt_dir: str) -> dict:
    try:
        with open(_manifest_path(ckpt_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _collect_param_arrays(model) -> Dict[str, np.ndarray]:
    arrs = {}
    for name, tensor in model.parameters().items():
        arr = tensor.to_numpy()  # Expect numpy array
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Parameter '{name}' to_numpy() did not return a numpy array."
            )
        arrs[name] = arr
    return arrs


def _load_npz_arrays(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _guess_latest_ckpt_path(ckpt_dir: str) -> Optional[str]:
    files = _list_checkpoints(ckpt_dir)
    return os.path.join(ckpt_dir, files[-1]) if files else None


def _set_param_array_autotensor(param, arr: np.ndarray):
    """
    Construct a TTML tensor from numpy with TILED layout + BFLOAT16 dtype,
    then assign it into `param` in-place using `param.assign(src)`.
    """
    # Ensure float base dtype convertible to BF16
    if arr.dtype not in (np.float32, np.float16):
        arr = arr.astype(np.float32, copy=False)

    # Optional: check shape against current parameter snapshot
    try:
        cur = param.to_numpy()
        if cur.shape != arr.shape:
            print(
                f"[load] shape mismatch for param: {cur.shape} (model) vs {arr.shape} (ckpt). Assigning anyway."
            )
    except Exception:
        logging.warning(
            f"Could not fetch current shape for param during load. Assigning anyway."
        )

    # Create source tensor in the required layout + dtype, then assign
    src = ttml.autograd.Tensor.from_numpy(
        arr,
        ttml.Layout.TILE,  # TILED layout
        ttml.autograd.DataType.BFLOAT16,  # BF16 dtype
    )
    param.assign(src)

    # Clean up graph refs (best-effort)
    try:
        ttml.autograd.AutoContext.get_instance().reset_graph()
    except Exception as e:
        # Non-critical: errors while cleaning up graph refs are ignored (best-effort)
        logging.warning(f"Failed to reset TTML graph during param assignment: {e}")


def _load_model_from_npz(model, ckpt_path: str, strict: bool = False):
    loaded = _load_npz_arrays(ckpt_path)
    params = model.parameters()
    missing, loaded_ok = [], 0

    for name, param in params.items():
        if name not in loaded:
            missing.append(name)
            continue
        arr = loaded[name]
        _set_param_array_autotensor(param, arr)
        loaded_ok += 1

    if strict:
        extras = [k for k in loaded.keys() if k not in params]
        if extras:
            raise RuntimeError(f"Checkpoint has unexpected params: {extras}")

    if missing:
        print(
            f"[load] Missing {len(missing)} params (kept existing): first 5: {missing[:5]}"
        )
    print(f"[load] Loaded {loaded_ok} parameters from {ckpt_path}")
