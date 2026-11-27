# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""didt_runner config: load and strictly validate, all errors at once, before any device work."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# each_chip: the step's ids once per chip, all chips at once, each process seeing only its own chip.
RUNNERS = ("pytest", "each_chip")
CHECKS = ("nd_report", "run_to_run", "core_sweep")
TOP_FIELDS = ("version", "hang_secs", "reset", "smi", "collect", "fallbacks", "steps")
STEP_FIELDS = ("key", "runner", "ids", "opts", "env", "timeout", "check", "fallbacks", "note")
STEP_OPTIONAL = ("pin_id",)
FALLBACK_FIELDS = ("env", "opts")
# pin_core reruns the step's per-chip test (pin_id) on each failing device of the first attempt; a each_chip step
# reruns its own ids on that chip alone.
PIN_CORE_FIELDS = ("kind", "hang_reset")
FALLBACK_KINDS = ("env", "pin_core")
COMMAND_FIELDS = {"reset": ("cmd", "tries", "timeout"), "smi": ("cmd", "timeout"), "collect": ("cmd", "timeout")}
# Selection flags pick by substring (`iter2` matches `iter25`); exact node ids only.
BANNED_OPTS = ("-k", "-m")


class ConfigError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class Fallback:
    name: str
    env: dict[str, str]
    opts: list[str]
    kind: str = "env"
    hang_reset: dict | None = None


@dataclass
class StepCfg:
    key: str
    runner: str
    ids: list[str]
    opts: list[str]
    env: dict[str, str]
    timeout: float | None
    check: str
    fallbacks: list[Fallback]
    note: str
    pin_id: str | None = None


@dataclass
class Config:
    path: Path
    sha256: str
    hang_secs: int
    reset: dict
    smi: dict
    collect: dict
    steps: list[StepCfg]


def _fields(where: str, obj, required: tuple, optional: tuple = ()) -> list[str]:
    if not isinstance(obj, dict):
        return [f"{where}: must be an object"]
    errors = [f"{where}: unknown field {k!r}" for k in obj if k not in required + optional]
    return errors + [f"{where}: missing field {k!r}" for k in required if k not in obj]


def _env(where: str, env) -> list[str]:
    if not isinstance(env, dict):
        return [f"{where}: env must be an object"]
    return [f"{where}: env {k!r} must be a string" for k, v in env.items() if not isinstance(v, str)]


def _opts(where: str, opts) -> list[str]:
    if not isinstance(opts, list) or not all(isinstance(o, str) for o in opts):
        return [f"{where}: opts must be a list of strings"]
    return [f"{where}: {o.split('=')[0]!r} in opts" for o in opts if o.split("=")[0] in BANNED_OPTS]


def _check_step(where: str, s: dict, fallback_names) -> list[str]:
    errors = _fields(where, s, STEP_FIELDS, STEP_OPTIONAL)
    if s.get("runner", "pytest") not in RUNNERS:
        errors.append(f"{where}: unknown runner {s['runner']!r}")
    if s.get("check", "nd_report") not in CHECKS:
        errors.append(f"{where}: unknown check {s['check']!r}")
    if s.get("runner") == "each_chip":
        if "pin_id" in s:
            errors.append(f"{where}: 'pin_id' is for the pytest runner; each_chip reruns its own ids on the chip")
        if s.get("check") != "nd_report":
            errors.append(f"{where}: each_chip runner needs check 'nd_report'")
    ids = s.get("ids", [])
    if not isinstance(ids, list) or not ids:
        errors.append(f"{where}: ids must be a non-empty list")
    else:
        errors += [f"{where}: id without '::': {i!r}" for i in ids if "::" not in str(i)]
    if "pin_id" in s and "::" not in str(s["pin_id"]):
        errors.append(f"{where}: pin_id without '::': {s['pin_id']!r}")
    errors += _opts(where, s.get("opts", []))
    errors += _env(where, s.get("env", {}))
    t = s.get("timeout")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or t <= 0):
        errors.append(f"{where}: timeout must be seconds > 0 or null")
    errors += [f"{where}: unknown fallback {f!r}" for f in s.get("fallbacks", []) if f not in fallback_names]
    return errors


def _validate(raw) -> list[str]:
    errors = _fields("config", raw, TOP_FIELDS)
    if not isinstance(raw, dict):
        return errors
    if raw.get("version", 1) != 1:
        errors.append(f"config: version {raw['version']!r}, need 1")
    for name, fields in COMMAND_FIELDS.items():
        if name in raw:
            errors += _fields(name, raw[name], fields)
    fallbacks = raw.get("fallbacks", {})
    for name, fb in fallbacks.items():
        kind = fb.get("kind", "env") if isinstance(fb, dict) else "env"
        if kind not in FALLBACK_KINDS:
            errors.append(f"fallback {name}: unknown kind {kind!r}")
            continue
        if kind == "pin_core":
            errors += _fields(f"fallback {name}", fb, PIN_CORE_FIELDS)
            if isinstance(fb.get("hang_reset"), dict):
                errors += _fields(f"fallback {name} hang_reset", fb["hang_reset"], ("cmd", "timeout"))
            continue
        errors += _fields(f"fallback {name}", fb, FALLBACK_FIELDS, ("kind",))
        if isinstance(fb, dict):
            errors += _env(f"fallback {name}", fb.get("env", {})) + _opts(f"fallback {name}", fb.get("opts", []))
    seen = set()
    for i, s in enumerate(raw.get("steps", [])):
        key = s.get("key") if isinstance(s, dict) else None
        if key in seen:
            errors.append(f"steps[{i}]: duplicate key {key!r}")
        seen.add(key)
        errors += _check_step(key or f"steps[{i}]", s, fallbacks)
    return errors


def load(path: Path) -> Config:
    path = Path(path)
    data = path.read_bytes()
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as e:
        raise ConfigError([f"{path}: {e}"]) from e
    errors = _validate(raw)
    if errors:
        raise ConfigError(errors)
    fallbacks = {
        n: (
            Fallback(n, {}, [], "pin_core", dict(f["hang_reset"]))
            if f.get("kind") == "pin_core"
            else Fallback(n, dict(f["env"]), list(f["opts"]))
        )
        for n, f in raw["fallbacks"].items()
    }
    steps = [
        StepCfg(
            key=s["key"],
            runner=s["runner"],
            ids=list(s["ids"]),
            opts=list(s["opts"]),
            env=dict(s["env"]),
            timeout=s["timeout"],
            check=s["check"],
            fallbacks=[fallbacks[f] for f in s["fallbacks"]],
            note=s["note"],
            pin_id=s.get("pin_id"),
        )
        for s in raw["steps"]
    ]
    return Config(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        hang_secs=raw["hang_secs"],
        reset=raw["reset"],
        smi=raw["smi"],
        collect=raw["collect"],
        steps=steps,
    )
