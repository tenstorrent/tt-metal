# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which transformers release a model needs, discovered rather than pinned by hand.

A model whose architecture (``model_type``) the installed transformers does not register cannot be
loaded natively. These helpers read the model's ``model_type`` from the model itself and find the
LOWEST newer transformers release whose own wheel registers it and that installs on this Python --
the smallest step off tt-metal's pin. The bring-up pre-flight (cli._preflight_load_with_autofix)
installs it and re-runs.
"""
from __future__ import annotations

import re


def wheel_registers_model_type(wheel_url: str, model_type: str, timeout: int = 120) -> bool:
    """Whether a transformers wheel's OWN auto-config table registers ``model_type``.

    Read straight from the published wheel (no install, no import), so the answer is the
    release's, not a guess from version numbers or a hand-kept list."""
    import tempfile
    import urllib.request
    import zipfile

    try:
        with tempfile.TemporaryFile() as tmp:
            with urllib.request.urlopen(wheel_url, timeout=timeout) as r:
                while chunk := r.read(1 << 20):
                    tmp.write(chunk)
            tmp.seek(0)
            with zipfile.ZipFile(tmp) as zf:
                # The auto package's (model_type, ConfigClass) table: which file holds it moved between
                # releases (configuration_auto.py -> auto_mappings.py), so read the whole auto package.
                text = "\n".join(
                    zf.read(n).decode("utf-8", "replace")
                    for n in zf.namelist()
                    if "/models/auto/" in n and n.endswith(".py")
                )
    except Exception:  # noqa: BLE001 -- unreadable/offline: treat as "not shown to register"
        return False
    return re.search(r"""\(\s*["']%s["']\s*,\s*["']\w+Config["']""" % re.escape(model_type), text) is not None


def release_registering(model_type: str, lister=None, prober=None) -> str:
    """Lowest transformers release NEWER than the installed one that registers ``model_type`` and
    whose ``requires_python`` admits this interpreter; "" when none (or PyPI is unreachable).

    The installed release is the floor, so this only ever moves forward, and only as far as the
    model needs -- the smallest step off tt-metal's pin. Candidates are bisected (an architecture,
    once added, stays) and every answer is verified against that release's own wheel."""
    import platform

    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    from .module_tree import _installed_version, _pypi_json

    releases = (lister or (lambda: _pypi_json("transformers", timeout=15).get("releases", {})))()
    try:
        have = Version(_installed_version("transformers"))
    except InvalidVersion:
        have = Version("0")
    py = platform.python_version()
    cands = []
    for raw, files in (releases or {}).items():
        try:
            ver = Version(raw)
        except InvalidVersion:
            continue
        if ver.is_prerelease or ver <= have:
            continue
        wheel = next((f for f in files if f.get("packagetype") == "bdist_wheel" and not f.get("yanked")), None)
        if not wheel:
            continue
        try:
            if wheel.get("requires_python") and not SpecifierSet(wheel["requires_python"]).contains(py):
                continue
        except InvalidSpecifier:
            continue
        cands.append((ver, wheel.get("url", "")))
    cands.sort()
    probe = prober or wheel_registers_model_type
    lo, hi, found = 0, len(cands) - 1, ""
    while lo <= hi:
        mid = (lo + hi) // 2
        if probe(cands[mid][1], model_type):
            found, hi = str(cands[mid][0]), mid - 1
        else:
            lo = mid + 1
    return found


def unregistered_model_type(model_id: str, load_error: str) -> str:
    """The model's ``model_type`` when the INSTALLED transformers does not register it, else "".

    Read from the model itself (its config.json; the load error as a fallback), never assumed."""
    try:
        from .llm_synth import _extract_model_type_from_error, _fetch_hf_config_model_type

        mt = _fetch_hf_config_model_type(model_id) or _extract_model_type_from_error(load_error) or ""
    except Exception:  # noqa: BLE001
        return ""
    if not mt:
        return ""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        return "" if mt in CONFIG_MAPPING_NAMES else mt
    except Exception:  # noqa: BLE001
        return ""
