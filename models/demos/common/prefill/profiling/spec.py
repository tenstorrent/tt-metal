# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The per-model contract of the zone profiler: everything the shared tooling needs to know about one
model, in one frozen dataclass. Dependency-free on purpose (no ttnn, no pandas) so the model's
``utils/profiler_utils.py`` can build it at import time for free."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerClass:
    """One class of decoder layer, as tagged by the model's ``layerNN_<key>`` zone name.

    key:               the tag suffix (``"sliding"`` in ``layer04_sliding``); also the report's class key
    label:             human name for the report (``"Sliding-attention layer"``)
    full_model_count:  how many layers of this class the full model has — the projection multiplier
    """

    key: str
    label: str
    full_model_count: int


@dataclass(frozen=True)
class ZoneSpec:
    """What distinguishes one model's zone profile from another's.

    model_name:       report titles (``"GPT-OSS"``)
    signpost_prefix:  the signpost wire prefix; zones emit ``<prefix>_START <name>`` / ``<prefix>_END <name>``
    env_prefix:       env-var prefix; ``<prefix>_ZONES=1`` arms the zones, ``<prefix>_LEVEL`` picks the detail,
                      ``<prefix>_HOST_ZONES=0`` drops the cosmetic Tracy host zones
    host_zone_scope:  first argument of ``ttnn.start_tracy_zone`` for the host zones
    layer_classes:    the model's layer classes in report order (see :class:`LayerClass`)
    comm_keys:        zone names (matched on the last path element) that are communication
    mem_keys:         zone-name substrings that are KV-cache memory traffic
    root_zone:        the zone the harness wraps the profiled chunk in; only zones under it are reported
    """

    model_name: str
    signpost_prefix: str
    env_prefix: str
    host_zone_scope: str
    layer_classes: tuple[LayerClass, ...]
    comm_keys: tuple[str, ...]
    mem_keys: tuple[str, ...]
    root_zone: str = "profiled_chunk"

    def __post_init__(self):
        assert self.layer_classes, "a ZoneSpec needs at least one layer class"
        keys = [c.key for c in self.layer_classes]
        assert len(set(keys)) == len(keys), f"duplicate layer class keys: {keys}"

    # --- signposts ------------------------------------------------------------------------------
    @property
    def zone_start(self) -> str:
        return f"{self.signpost_prefix}_START"

    @property
    def zone_end(self) -> str:
        return f"{self.signpost_prefix}_END"

    # --- env vars -------------------------------------------------------------------------------
    @property
    def zones_env(self) -> str:
        return f"{self.env_prefix}_ZONES"

    @property
    def level_env(self) -> str:
        return f"{self.env_prefix}_LEVEL"

    @property
    def host_zones_env(self) -> str:
        return f"{self.env_prefix}_HOST_ZONES"

    # --- layer classes --------------------------------------------------------------------------
    @property
    def class_keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.layer_classes)

    @property
    def full_model_layers(self) -> int:
        return sum(c.full_model_count for c in self.layer_classes)

    def layer_class(self, key: str) -> LayerClass | None:
        for c in self.layer_classes:
            if c.key == key:
                return c
        return None

    # --- categorization -------------------------------------------------------------------------
    def cat(self, rel: str) -> str:
        """compute / comm / memory for a layer-relative zone path such as ``"mlp/dispatch"``.

        Drives the headline compute/communication/memory split, so it is deliberately simple: a zone is
        memory if its path contains a ``mem_keys`` substring, communication if its last element is a
        ``comm_keys`` name, compute otherwise.
        """
        if any(k in rel for k in self.mem_keys):
            return "memory"
        if any(rel.endswith(k) for k in self.comm_keys):
            return "comm"
        return "compute"
