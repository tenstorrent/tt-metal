# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-system tuning profiles: one file of deployment knobs, loaded by machine.

The model runs on machines that want different values for the same knob -- an
8-chip P150 host and a 32-chip Galaxy differ in pipeline depth, KV budget and
how far the weight prefetcher should run ahead. Those values used to be module
constants and ``os.environ`` reads scattered across ``tt/``, which made "what is
this machine actually running with?" unanswerable without grepping. They now
live in ``configs/system_configs.yaml`` as named profiles, and this module loads
one.

What belongs here: anything a *different machine* would want set differently.
What does not: model geometry (hidden size, layer/expert counts, head dims),
which comes from the checkpoint's own config -- duplicating it is how the two
drift apart.

Three layers of precedence, lowest first:

1. the profile in the YAML file (itself resolved through its ``extends`` chain),
2. per-field environment variables, so a one-off A/B needs no file edit,
3. explicit keyword arguments at the call site, which always win.

Typical use is implicit: :class:`~.model.DeepSeekV4Model` loads the profile
matching its mesh and publishes it with :func:`set_active_system_config`, so
leaf modules pick it up through :func:`active_system_config` without every
constructor growing a parameter. Explicit is also fine::

    cfg = load_system_config(profile="galaxy32")
    model = DeepSeekV4Model(config, loader, mesh, system_config=cfg)
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

import ttnn
from loguru import logger

# The shipped profile file. ``$DEEPSEEK_V4_SYSTEM_CONFIG`` points at another one
# (an operator's local tune) without touching the tree.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "system_configs.yaml"

# Selects a profile by name, overriding the device-count match.
PROFILE_ENV_VAR = "DEEPSEEK_V4_SYSTEM_PROFILE"
CONFIG_PATH_ENV_VAR = "DEEPSEEK_V4_SYSTEM_CONFIG"

_FALSEY = ("0", "", "false", "False", "no", "off")


# --------------------------------------------------------------------------- #
# Settings groups
#
# One frozen dataclass per YAML section. The field names ARE the accepted keys:
# the loader rejects anything else, so a typo in a config file fails at load
# with a suggestion instead of silently running an untuned machine.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeviceSettings:
    """How the mesh is opened. Consumed by the demo/test entry points, since the
    device exists before the model does."""

    fabric_config: str = "FABRIC_2D"
    num_command_queues: int = 2
    trace_region_size: int = 0
    # ``None`` = the whole system flattened to a 1xN line (what the submesh
    # pipeline is built against); ``[rows, cols]`` opens exactly that shape.
    mesh_shape: Optional[list[int]] = None
    worker_l1_size: int = 0

    @property
    def ttnn_fabric_config(self) -> Any:
        """``fabric_config`` as the ttnn enum member."""
        try:
            return getattr(ttnn.FabricConfig, self.fabric_config)
        except AttributeError as exc:
            valid = [n for n in dir(ttnn.FabricConfig) if n.isupper()]
            raise ValueError(f"unknown fabric_config {self.fabric_config!r}; expected one of {valid}") from exc

    def device_params(self) -> dict:
        """The ``open_mesh_device`` kwargs this profile asks for.

        Only non-default entries are emitted: a zero ``trace_region_size`` or
        ``worker_l1_size`` means "leave the ttnn default alone", which is not the
        same as passing 0.
        """
        params: dict = {
            "fabric_config": self.ttnn_fabric_config,
            "num_command_queues": self.num_command_queues,
        }
        if self.trace_region_size:
            params["trace_region_size"] = self.trace_region_size
        if self.worker_l1_size:
            params["worker_l1_size"] = self.worker_l1_size
        return params


@dataclass(frozen=True)
class PipelineSettings:
    """Layer placement across chips and the sockets that carry activations."""

    group_size: int = 1
    depth: int = 0
    socket_l1_bytes: int = 16384
    pcie_alignment: int = 64
    h2d_fifo_pages: int = 64
    d2h_fifo_bytes: int = 4032

    @property
    def h2d_fifo_bytes(self) -> int:
        """H2D packet FIFO size. Whole PCIe pages, so it is derived rather than
        configured directly -- a size that is not a multiple of the alignment
        cannot be filled by whole-page writes."""
        return self.h2d_fifo_pages * self.pcie_alignment


@dataclass(frozen=True)
class PrefetcherSettings:
    """DRISC weight prefetcher and its shared global circular buffer."""

    # ``None`` = enable wherever the device supports it.
    enabled: Optional[bool] = None
    num_prefetch_pages: int = 16
    num_prefetch_slabs: int = 2

    def resolve_enabled(self, device) -> bool:
        """Whether to prefetch on ``device``, honouring an explicit on/off."""
        if self.enabled is not None:
            return self.enabled
        return ttnn.experimental.is_tensor_prefetcher_supported(device)


@dataclass(frozen=True)
class MoESettings:
    """``fused_experts`` tuning. ``experts_block_size`` is the L1 knob: the
    gathered-activation CB it sizes is the op's dominant per-core consumer."""

    experts_block_size: int = 2
    routing_eps: float = 1.0e-20
    fused_num_cores: int = 64
    fused_dram_banks: int = 8


@dataclass(frozen=True)
class AttentionSettings:
    """SDPA-decode program config and the L1/DRAM tradeoffs around it."""

    sdpa_causal: bool = True
    sdpa_q_chunk_size: int = 0
    sdpa_k_chunk_size: int = 32
    sdpa_max_cores_per_head_batch: int = 2
    sdpa_exp_approx_mode: bool = False
    keep_qa_kv_weights_in_l1: bool = True
    # ``None`` = fuse wherever the prefetcher is off (see :meth:`resolve_fuse_qa_kv`).
    fuse_qa_kv_proj: Optional[bool] = False

    def resolve_fuse_qa_kv(self, use_prefetcher: bool) -> bool:
        """Whether to run q_a and kv as one matmul over their concatenated weight.

        ``None`` means "wherever it is possible", which is the L1 path only: the fused
        1536-wide weight cuts into 3-tile rows per B core and so shares no page size
        with the model's one decode GCB, which is why it cannot be prefetched. Forcing
        it on *with* the prefetcher is therefore rejected rather than quietly ignored --
        it would be a hang, not a slowdown.
        """
        if self.fuse_qa_kv_proj is None:
            return False
        if self.fuse_qa_kv_proj and use_prefetcher:
            raise ValueError(
                "attention.fuse_qa_kv_proj=true is incompatible with the weight prefetcher: the fused "
                "qa_kv weight has no page size in common with the shared decode GCB. Set it to null "
                "(fuse only where the prefetcher is off) or disable the prefetcher."
            )
        return self.fuse_qa_kv_proj

    def sdpa_program_config(self, device) -> Any:
        """The ``SDPAProgramConfig`` for ``device``, at this profile's settings.

        The compute grid is read off the device rather than configured: it is a
        property of the chip, and the L1 lever here is
        ``max_cores_per_head_batch`` (see the profile file).
        """
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=self.sdpa_q_chunk_size,
            k_chunk_size=self.sdpa_k_chunk_size,
            exp_approx_mode=self.sdpa_exp_approx_mode,
            max_cores_per_head_batch=self.sdpa_max_cores_per_head_batch,
        )


@dataclass(frozen=True)
class DecodeSettings:
    """Precision, paging and how many users a step serves."""

    weight_dtype: str = "bfloat4_b"
    block_size: int = 32
    batch: int = 1
    num_users: int = 2
    max_context: int = 131072
    # 0 = one ``max_context`` (the users share a single context's worth).
    total_context: int = 0
    max_new_tokens: int = 2048
    # 0 = every layer in the checkpoint.
    num_layers: int = 0
    traced: bool = True

    @property
    def ttnn_weight_dtype(self) -> Any:
        """``weight_dtype`` as the ttnn dtype object."""
        dtype = getattr(ttnn, self.weight_dtype, None)
        if not isinstance(dtype, ttnn.DataType):
            valid = [n for n in ("bfloat4_b", "bfloat8_b", "bfloat16", "float32") if hasattr(ttnn, n)]
            raise ValueError(f"unknown weight_dtype {self.weight_dtype!r}; expected one of {valid}")
        return dtype

    def resolve_total_context(self) -> int:
        """The shared block-pool budget, with 0 meaning one full context."""
        return self.total_context or self.max_context

    def resolve_num_layers(self, num_hidden_layers: int) -> int:
        """The layer cap, with 0 meaning the whole checkpoint stack."""
        return min(self.num_layers or num_hidden_layers, num_hidden_layers)


@dataclass(frozen=True)
class SystemConfig:
    """One machine's tuning profile: the sections above plus its identity."""

    name: str = "base"
    # The mesh size this profile is for. 0 = never auto-selected by device count
    # (a variant that has to be asked for by name).
    num_devices: int = 0
    description: str = ""
    device: DeviceSettings = field(default_factory=DeviceSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    prefetcher: PrefetcherSettings = field(default_factory=PrefetcherSettings)
    moe: MoESettings = field(default_factory=MoESettings)
    attention: AttentionSettings = field(default_factory=AttentionSettings)
    decode: DecodeSettings = field(default_factory=DecodeSettings)

    # -- introspection ------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Plain-data round-trip of this profile (for logging or re-serializing)."""
        return dataclasses.asdict(self)

    def summary(self) -> str:
        """One line naming the profile and the fields most likely to explain a
        performance or out-of-memory surprise."""
        return (
            f"system profile '{self.name}' ({self.num_devices or 'any'} devices): "
            f"PGS={self.pipeline.group_size} depth={self.pipeline.depth} "
            f"prefetch={self.prefetcher.enabled if self.prefetcher.enabled is not None else 'auto'}"
            f"/{self.prefetcher.num_prefetch_pages}p "
            f"experts_block={self.moe.experts_block_size} "
            f"fuse_qa_kv={'auto' if self.attention.fuse_qa_kv_proj is None else self.attention.fuse_qa_kv_proj} "
            f"dtype={self.decode.weight_dtype} batch={self.decode.batch} "
            f"users={self.decode.num_users} ctx={self.decode.max_context}"
        )

    def log(self) -> "SystemConfig":
        """Log :meth:`summary` and return self, so a load can be logged inline."""
        logger.info(self.summary())
        return self

    def with_overrides(self, **sections: dict) -> "SystemConfig":
        """A copy with per-section field overrides, e.g.
        ``cfg.with_overrides(pipeline={"group_size": 4})``.

        Values of ``None`` are dropped rather than applied, so a caller can pass
        an optional argument straight through without first checking it. To *set* a
        field to ``None`` (the "decide from the device" state of the tri-state
        fields), say so in the profile file or use its ``auto`` environment value.
        """
        return self._patch_sections(sections, drop_none=True)

    def _patch_sections(self, sections: dict, *, drop_none: bool) -> "SystemConfig":
        """``self`` with ``{section: {field: value}}`` applied.

        ``drop_none`` is what separates the public convenience of
        :meth:`with_overrides` from the environment overrides, where a ``None``
        parsed out of ``VAR=auto`` is a value the caller meant and must land.
        """
        patch: dict = {}
        for section_name, values in sections.items():
            if not values:
                continue
            current = getattr(self, section_name, None)
            if not dataclasses.is_dataclass(current):
                raise ValueError(f"{section_name!r} is not a settings section of SystemConfig")
            live = {k: v for k, v in values.items() if v is not None} if drop_none else dict(values)
            if not live:
                continue
            _reject_unknown(section_name, live.keys(), type(current))
            patch[section_name] = dataclasses.replace(current, **live)
        return dataclasses.replace(self, **patch) if patch else self


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_SECTION_TYPES: dict[str, type] = {
    "device": DeviceSettings,
    "pipeline": PipelineSettings,
    "prefetcher": PrefetcherSettings,
    "moe": MoESettings,
    "attention": AttentionSettings,
    "decode": DecodeSettings,
}
# Profile keys that are not settings sections.
_SCALAR_KEYS = ("num_devices", "description")


def _suggest(name: str, valid) -> str:
    """`` (did you mean 'x'?)`` when ``name`` is a near-miss for something valid."""
    import difflib

    close = difflib.get_close_matches(name, list(valid), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _reject_unknown(where: str, keys, dataclass_type: type) -> None:
    valid = {f.name for f in fields(dataclass_type)}
    for key in keys:
        if key not in valid:
            raise ValueError(
                f"unknown key {key!r} in section {where!r}{_suggest(key, valid)}; valid keys: {sorted(valid)}"
            )


def _coerce(value: Any, declared: Any, where: str) -> Any:
    """YAML scalar -> the field's declared type.

    YAML gets most of this right on its own; the cases that need help are ints
    written in float form (``1e6``) and the ``Optional`` fields, whose ``None``
    is meaningful ("auto") and must survive.
    """
    if value is None:
        return None
    if declared is bool:
        return value if isinstance(value, bool) else str(value) not in _FALSEY
    if declared is int:
        if isinstance(value, bool):
            return int(value)
        coerced = int(value)
        if isinstance(value, float) and coerced != value:
            raise ValueError(f"{where} wants an integer but got {value!r}")
        return coerced
    if declared is float:
        return float(value)
    if declared is str:
        return str(value)
    return value


def _build_section(section_name: str, raw: dict, base: Any) -> Any:
    """``base`` with ``raw``'s keys applied, type-coerced and validated."""
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError(f"section {section_name!r} must be a mapping, got {type(raw).__name__}")
    section_type = type(base)
    _reject_unknown(section_name, raw.keys(), section_type)
    declared = {f.name: f.type for f in fields(section_type)}
    values = {}
    for key, value in raw.items():
        # ``from __future__ import annotations`` makes f.type a string, so match
        # on the annotation text rather than the type object.
        annotation = str(declared[key])
        where = f"{section_name}.{key}"
        if "bool" in annotation:
            values[key] = _coerce(value, bool, where)
        elif "int" in annotation and "list" not in annotation:
            values[key] = _coerce(value, int, where)
        elif "float" in annotation:
            values[key] = _coerce(value, float, where)
        elif "str" in annotation:
            values[key] = _coerce(value, str, where)
        else:
            values[key] = value
    return dataclasses.replace(base, **values)


def _load_raw(path: Path) -> dict:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"system config file not found: {path}")
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict) or "profiles" not in doc:
        raise ValueError(f"{path} must be a mapping with a top-level 'profiles' key")
    profiles = doc["profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{path} has no profiles")
    return doc


def _resolve_chain(name: str, profiles: dict, _seen: Optional[tuple] = None) -> SystemConfig:
    """Build ``name``, applying its ``extends`` ancestors outermost-first."""
    _seen = _seen or ()
    if name in _seen:
        raise ValueError(f"circular 'extends' in system profiles: {' -> '.join((*_seen, name))}")
    if name not in profiles:
        raise ValueError(f"unknown system profile {name!r}{_suggest(name, profiles)}; available: {sorted(profiles)}")
    raw = dict(profiles[name] or {})
    parent = raw.pop("extends", None)
    base = _resolve_chain(parent, profiles, (*_seen, name)) if parent else SystemConfig()

    unknown = set(raw) - set(_SECTION_TYPES) - set(_SCALAR_KEYS)
    if unknown:
        valid = (*_SECTION_TYPES, *_SCALAR_KEYS, "extends")
        raise ValueError(
            f"profile {name!r} has unknown key(s) {sorted(unknown)}"
            f"{_suggest(sorted(unknown)[0], valid)}; valid: {sorted(valid)}"
        )

    patch: dict = {"name": name}
    if "num_devices" in raw:
        patch["num_devices"] = _coerce(raw["num_devices"], int, f"{name}.num_devices")
    if "description" in raw:
        patch["description"] = str(raw["description"])
    for section_name in _SECTION_TYPES:
        if section_name in raw:
            patch[section_name] = _build_section(section_name, raw[section_name], getattr(base, section_name))
    return dataclasses.replace(base, **patch)


# --------------------------------------------------------------------------- #
# Environment overrides
#
# (section, field, env var, parser). These keep the env-var names the code and
# the demos already used working, so existing scripts and tt-inference-server
# model specs need no change -- but the file is now the source of truth and the
# env var is the exception.
# --------------------------------------------------------------------------- #
def _env_bool(raw: str) -> bool:
    return raw not in _FALSEY


def _env_tristate(raw: str) -> Optional[bool]:
    """For the fields whose ``None`` means "decide from the device/config", so that
    state is reachable from a shell too: ``auto`` (or ``none``) restores it."""
    return None if raw.lower() in ("auto", "none", "null") else _env_bool(raw)


def _env_mesh_shape(raw: str) -> list[int]:
    parts = raw.replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"{'DEEPSEEK_V4_MESH_SHAPE'} must look like '1x8' or '8,4', got {raw!r}")
    return [int(p) for p in parts]


_ENV_OVERRIDES: tuple[tuple[str, str, str, Callable[[str], Any]], ...] = (
    ("device", "fabric_config", "DEEPSEEK_V4_FABRIC_CONFIG", str),
    ("device", "num_command_queues", "DEEPSEEK_V4_NUM_COMMAND_QUEUES", int),
    ("device", "trace_region_size", "DEEPSEEK_V4_TRACE_REGION_SIZE", int),
    ("device", "mesh_shape", "DEEPSEEK_V4_MESH_SHAPE", _env_mesh_shape),
    ("device", "worker_l1_size", "DEEPSEEK_V4_WORKER_L1_SIZE", int),
    ("pipeline", "group_size", "DEEPSEEK_V4_PIPELINE_GROUP_SIZE", int),
    ("pipeline", "depth", "DEEPSEEK_V4_PIPELINE_DEPTH", int),
    ("pipeline", "socket_l1_bytes", "DEEPSEEK_V4_SOCKET_L1_BYTES", int),
    ("pipeline", "pcie_alignment", "DEEPSEEK_V4_PCIE_ALIGNMENT", int),
    ("pipeline", "h2d_fifo_pages", "DEEPSEEK_V4_H2D_FIFO_PAGES", int),
    ("pipeline", "d2h_fifo_bytes", "DEEPSEEK_V4_D2H_FIFO_BYTES", int),
    ("prefetcher", "enabled", "DEEPSEEK_V4_PREFETCHER", _env_tristate),
    ("prefetcher", "num_prefetch_pages", "DEEPSEEK_V4_PREFETCH_PAGES", int),
    ("prefetcher", "num_prefetch_slabs", "DEEPSEEK_V4_PREFETCH_SLABS", int),
    ("moe", "experts_block_size", "DEEPSEEK_V4_EXPERTS_BLOCK_SIZE", int),
    ("attention", "sdpa_causal", "DEEPSEEK_V4_SDPA_CAUSAL", _env_bool),
    ("attention", "sdpa_q_chunk_size", "DEEPSEEK_V4_SDPA_Q_CHUNK", int),
    ("attention", "sdpa_k_chunk_size", "DEEPSEEK_V4_SDPA_K_CHUNK", int),
    ("attention", "sdpa_max_cores_per_head_batch", "DEEPSEEK_V4_SDPA_MAX_CORES_PER_HEAD", int),
    ("attention", "keep_qa_kv_weights_in_l1", "DEEPSEEK_V4_KEEP_WEIGHTS_IN_L1", _env_bool),
    ("attention", "fuse_qa_kv_proj", "DEEPSEEK_V4_FUSE_QA_KV", _env_tristate),
    ("decode", "weight_dtype", "DEEPSEEK_V4_WEIGHT_DTYPE", str),
    ("decode", "block_size", "DEEPSEEK_V4_BLOCK_SIZE", int),
    ("decode", "batch", "DEEPSEEK_V4_DECODE_BATCH", int),
    ("decode", "num_users", "DEEPSEEK_V4_NUM_USERS", int),
    ("decode", "max_context", "DEEPSEEK_V4_MAX_CONTEXT", int),
    ("decode", "total_context", "DEEPSEEK_V4_TOTAL_CONTEXT", int),
    ("decode", "max_new_tokens", "DEEPSEEK_V4_MAX_NEW_TOKENS", int),
    ("decode", "num_layers", "DEEPSEEK_V4_DECODE_LAYERS", int),
    ("decode", "traced", "DEEPSEEK_V4_TRACED_DECODE", _env_bool),
)


def _apply_env(cfg: SystemConfig, env: Optional[dict] = None) -> SystemConfig:
    """``cfg`` with every set environment override applied.

    An unset variable is left alone; an *empty* one is also left alone, so
    ``VAR=`` reads as "not overridden" rather than as 0 or the empty string.
    """
    env = os.environ if env is None else env
    patches: dict[str, dict] = {}
    for section_name, field_name, var, parse in _ENV_OVERRIDES:
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        try:
            patches.setdefault(section_name, {})[field_name] = parse(raw)
        except ValueError as exc:
            raise ValueError(f"{var}={raw!r} is not a valid {section_name}.{field_name}: {exc}") from exc
    if not patches:
        return cfg
    applied = {f"{s}.{f}" for s, vals in patches.items() for f in vals}
    logger.debug(f"system config: environment overrides {sorted(applied)}")
    # Not ``with_overrides``: ``VAR=auto`` parses to None, which is a value here.
    return cfg._patch_sections(patches, drop_none=False)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def available_profiles(path: Optional[Path | str] = None) -> dict[str, str]:
    """``{name: description}`` for every profile in the config file."""
    doc = _load_raw(Path(path or os.environ.get(CONFIG_PATH_ENV_VAR) or DEFAULT_CONFIG_PATH))
    return {name: (body or {}).get("description", "") for name, body in doc["profiles"].items()}


def load_system_config(
    profile: Optional[str] = None,
    *,
    num_devices: Optional[int] = None,
    mesh_device: Optional[Any] = None,
    variant: Optional[str] = None,
    path: Optional[Path | str] = None,
    apply_env: bool = True,
    **overrides: dict,
) -> SystemConfig:
    """The tuning profile for this machine.

    Selection, highest precedence first: an explicit ``profile``;
    ``$DEEPSEEK_V4_SYSTEM_PROFILE``; the unique profile whose ``num_devices``
    matches ``num_devices`` (or ``mesh_device``'s device count); the file's
    ``default_profile``.

    ``variant`` names a *workload* flavour of whatever machine profile was
    selected: ``variant="server"`` upgrades ``p150x8`` to ``p150x8_server`` if the
    file defines it, and is ignored otherwise. That keeps "which machine" (device
    count) and "what is it serving" (batched server, latency demo) as independent
    choices instead of one flat list of names to remember.

    ``overrides`` are per-section dicts applied last, after the file and the
    environment, e.g. ``load_system_config(pipeline={"group_size": 4})``.
    """
    path = Path(path or os.environ.get(CONFIG_PATH_ENV_VAR) or DEFAULT_CONFIG_PATH)
    doc = _load_raw(path)
    profiles = doc["profiles"]

    if num_devices is None and mesh_device is not None:
        num_devices = mesh_device.get_num_devices()

    name = profile or os.environ.get(PROFILE_ENV_VAR) or None
    if name is None and num_devices:
        matches = [
            key
            for key, body in profiles.items()
            if int((body or {}).get("num_devices", 0) or 0) == num_devices
            # A profile inheriting num_devices from a parent is a variant, not a
            # default for the machine; only an explicit match auto-selects.
        ]
        if len(matches) > 1:
            raise ValueError(
                f"{len(matches)} profiles claim num_devices={num_devices} ({sorted(matches)}); "
                f"disambiguate with {PROFILE_ENV_VAR} or profile="
            )
        if matches:
            name = matches[0]
        else:
            logger.warning(
                f"no system profile for a {num_devices}-device mesh; falling back to "
                f"{doc.get('default_profile', 'base')!r}. Add one to {path} for this machine."
            )
    if name is None:
        name = doc.get("default_profile", "base")

    # An explicitly named profile is taken literally -- a caller who asked for
    # 'p150x8_throughput' does not want it silently swapped for a variant.
    if variant and not profile and f"{name}_{variant}" in profiles:
        name = f"{name}_{variant}"
    print(f"Loaded profile: {name}")
    cfg = _resolve_chain(name, profiles)
    if apply_env:
        cfg = _apply_env(cfg)
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    if num_devices and cfg.num_devices and cfg.num_devices != num_devices:
        logger.warning(
            f"system profile '{cfg.name}' is written for {cfg.num_devices} devices but the mesh has "
            f"{num_devices}; its tuning may not hold"
        )
    return cfg


# --------------------------------------------------------------------------- #
# Process-wide active profile
#
# Leaf modules (attention's SDPA config, the MoE expert block size) need these
# values but sit several constructors below the entry point. Rather than thread
# a parameter through every one, the model publishes the profile it loaded and
# the leaves read it as their *default* -- an explicitly passed value still
# wins, so nothing here is load-bearing for a caller who is being explicit.
# --------------------------------------------------------------------------- #
_active: Optional[SystemConfig] = None


def set_active_system_config(cfg: Optional[SystemConfig]) -> Optional[SystemConfig]:
    """Publish ``cfg`` as the process-wide profile; returns the previous one."""
    global _active
    previous, _active = _active, cfg
    return previous


def active_system_config() -> SystemConfig:
    """The published profile, loading the default one on first use.

    Deliberately never raises for want of a published profile: a unit test that
    builds one layer directly should get documented defaults, not a
    configuration error.
    """
    global _active
    if _active is None:
        _active = load_system_config()
    return _active


def reset_active_system_config() -> None:
    """Forget the published profile (so the next read re-loads it)."""
    set_active_system_config(None)
