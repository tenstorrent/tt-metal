# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load validated BEVFormer JSON configs from bundled files, local paths or HTTPS URLs."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .schema import ModelConfig, SUPPORTED_VARIANTS


_ACTIVE_CONFIG: ModelConfig | None = None


def builtin_config_path(variant: str) -> Path:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"variant is {variant!r}, expected one of {SUPPORTED_VARIANTS}")
    return Path(__file__).with_name(f"bevformer_{variant}.json")


def _github_raw_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == "github.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repository, _, revision, *path = parts
            return f"https://raw.githubusercontent.com/{owner}/{repository}/{revision}/{'/'.join(path)}"
        if "tree" in parts:
            raise ValueError("--config URL must identify one JSON file, not a GitHub directory")
    return url


def _verify_sha256(content: bytes, expected: str | None):
    if expected is None:
        return
    if len(expected) != 64 or any(character not in "0123456789abcdefABCDEF" for character in expected):
        raise ValueError("config SHA-256 must contain exactly 64 hexadecimal characters")
    actual = hashlib.sha256(content).hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"config SHA-256 mismatch: expected {expected.lower()}, got {actual}")


def _url_bytes(url: str, sha256: str | None, cache_dir: str | Path | None, refresh: bool) -> bytes:
    url = _github_raw_url(url)
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("remote configs must use HTTPS")
    cache_root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "tt-metal" / "bevformer"
    cache_key = hashlib.sha256(url.encode()).hexdigest()
    cache_path = cache_root / f"{cache_key}.cache"
    if cache_path.is_file() and not refresh:
        content = cache_path.read_bytes()
    else:
        request = Request(url, headers={"User-Agent": "tt-metal-bevformer-demo"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - HTTPS is enforced above.
            content = response.read()
        _verify_sha256(content, sha256)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
    _verify_sha256(content, sha256)
    return content


def read_source_bytes(
    source: str | Path,
    *,
    sha256: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> tuple[bytes, str]:
    source_string = str(source)
    if urlparse(source_string).scheme:
        content = _url_bytes(source_string, sha256, cache_dir, refresh)
        origin = source_string
    else:
        path = Path(source)
        content = path.read_bytes()
        origin = str(path)
        _verify_sha256(content, sha256)
    return content, origin


def load_model_config(
    source: str | Path,
    *,
    sha256: str | None = None,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> ModelConfig:
    content, origin = read_source_bytes(
        source,
        sha256=sha256,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    try:
        raw = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{origin}: expected a normalized BEVFormer JSON config") from error
    try:
        return ModelConfig.from_dict(raw)
    except ValueError as error:
        raise ValueError(f"{origin}: {error}") from error


def load_builtin_config(variant: str) -> ModelConfig:
    return load_model_config(builtin_config_path(variant))


def activate_config(config: ModelConfig):
    """Select a config before importing the compatibility ``model_config`` module."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config


def selected_config() -> ModelConfig:
    if _ACTIVE_CONFIG is not None:
        return _ACTIVE_CONFIG
    source = os.environ.get("BEVFORMER_CONFIG")
    if source:
        return load_model_config(source)
    return load_builtin_config(os.environ.get("BEVFORMER_VARIANT", "base"))
