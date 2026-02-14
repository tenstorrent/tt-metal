import errno
import fcntl
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

import ttnn
from models.demos.deepseek_v3.utils.cache.cache import CacheKey, _safe_name_from_manifest


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _exclusive_lock(path: Path, *, ensure_dir: Path | None = None, timeout_s: float | None = None, poll_s: float = 0.1):
    if ensure_dir is not None:
        ensure_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    with os.fdopen(fd, "r+") as f:
        acquired = False
        if timeout_s is None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            acquired = True
        else:
            deadline = time.monotonic() + max(timeout_s, 0.0)
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError as e:
                    if e.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(poll_s)
        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _atomic_write(final_path: Path, write_fn: Callable[[Path], None]) -> None:
    if final_path.suffix:
        tmp_name = f"{final_path.stem}.tmp.{os.getpid()}{final_path.suffix}"
    else:
        tmp_name = f"{final_path.name}.tmp.{os.getpid()}"
    tmp_path = final_path.with_name(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


class OnDiskCacheStorage:
    def __init__(
        self,
        cache_root: str | Path,
        device: ttnn.Device | ttnn.MeshDevice,
        *,
        lock_timeout_s: float | None = 10.0,
        lock_poll_s: float = 0.1,
    ):
        self.cache_root = Path(cache_root)
        self.device = device
        self.lock_timeout_s = lock_timeout_s
        self.lock_poll_s = lock_poll_s

    def _ensure_entry_consistency(self, key: CacheKey, tensor_path: Path, manifest_path: Path) -> tuple[bool, bool]:
        tensor_exists = tensor_path.is_file()
        manifest_exists = manifest_path.is_file()
        if tensor_exists and not manifest_exists:
            raise RuntimeError(
                f"Cache tensor exists but manifest is missing for fingerprint {key.fingerprint}. "
                "This may indicate a partial write or concurrent writer."
            )
        if manifest_exists and not tensor_exists:
            raise RuntimeError(
                f"Cache manifest exists but tensor is missing for fingerprint {key.fingerprint}. "
                "This may indicate a partial write or concurrent writer."
            )
        return tensor_exists, manifest_exists

    def _read_manifest_payload(self, key: CacheKey, manifest_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            raise ValueError(f"Invalid manifest file: {manifest_path}")
        fingerprint = payload.get("fingerprint")
        if fingerprint != key.fingerprint:
            raise RuntimeError(
                f"Manifest fingerprint mismatch: expected {key.fingerprint}, got {fingerprint}. "
                f"Manifest path: {manifest_path}"
            )
        return payload

    def _tensor_path(self, key: CacheKey) -> Path:
        safe_name = _safe_name_from_manifest(key.manifest)
        return self.cache_root / f"{safe_name}__{key.fingerprint}.tensorbin"

    def _manifest_path(self, key: CacheKey) -> Path:
        safe_name = _safe_name_from_manifest(key.manifest)
        return self.cache_root / f"{safe_name}__{key.fingerprint}.manifest.json"

    def _lock_path(self, key: CacheKey) -> Path:
        safe_name = _safe_name_from_manifest(key.manifest)
        return self.cache_root / f"{safe_name}__{key.fingerprint}.lock"

    def set(self, key: CacheKey, tensor: ttnn.Tensor) -> None:
        tensor_path = self._tensor_path(key)
        manifest_path = self._manifest_path(key)
        lock_path = self._lock_path(key)

        with _exclusive_lock(
            lock_path, ensure_dir=self.cache_root, timeout_s=self.lock_timeout_s, poll_s=self.lock_poll_s
        ) as acquired:
            if not acquired:
                logger.warning(
                    f"Cache write skipped due to lock timeout: fingerprint={key.fingerprint} path={lock_path}"
                )
                return
            created_at = None
            if manifest_path.is_file():
                existing_payload = self._read_manifest_payload(key, manifest_path)
                created_at = existing_payload.get("created_at")

            _atomic_write(tensor_path, lambda p: ttnn.dump_tensor(p, tensor))

            now = _utc_now_iso()
            if created_at is None:
                created_at = now
            manifest_payload = key.manifest.to_dict()
            manifest_payload["fingerprint"] = key.fingerprint
            manifest_payload["created_at"] = created_at
            manifest_payload["last_modified"] = now
            _atomic_write(
                manifest_path,
                lambda p: p.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"),
            )

    def has(self, key: CacheKey) -> bool:
        tensor_path = self._tensor_path(key)
        manifest_path = self._manifest_path(key)
        tensor_exists, manifest_exists = self._ensure_entry_consistency(key, tensor_path, manifest_path)
        return tensor_exists and manifest_exists

    def get(self, key: CacheKey) -> ttnn.Tensor:
        tensor_path = self._tensor_path(key)
        manifest_path = self._manifest_path(key)
        tensor_exists, manifest_exists = self._ensure_entry_consistency(key, tensor_path, manifest_path)
        if not tensor_exists or not manifest_exists:
            raise KeyError(key.fingerprint)
        self._read_manifest_payload(key, manifest_path)
        return ttnn.load_tensor(tensor_path, device=self.device)
