# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The boundary between the model-independent placement core and one MoE implementation.

To support a new MoE implementation, subclass ``PlacementAdapter`` and register it in ``adapters/__init__.py``.
The implementation must already (or be made to):
  1. capture routing: dump each token's selected logical experts per MoE layer during an untraced prefill;
  2. remap on device: translate selected logical expert ids to physical slots with a per-layer map
     (e.g. ``moe_grouped_topk(expert_id_map=...)``), loaded from the placed cache;
  3. load a placed weight cache: expert weights stored in their physical slots, every other file unchanged;
  4. time a traced prefill of a given token sequence, and check a golden sequence for correctness.
Host-only stages (screen, calibrate, preset, held-out packing) need nothing but ``spec`` and the routing
captures, so they work for any model as soon as routing can be captured or imported.
"""

from __future__ import annotations

import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from ..spec import MoeSpec


class PlacementAdapter(ABC):
    family = "base"
    verified = False  # True once an on-device A/B has been run end to end with this adapter and model

    def __init__(self, spec: MoeSpec, chat_format: str = "hf"):
        self.spec = spec
        self.chat_format = chat_format

    # --- model data ------------------------------------------------------------------------------
    @abstractmethod
    def tokenizer_path(self) -> str:
        """Tokenizer (HF directory) used to pack the corpus."""

    @abstractmethod
    def original_cache(self) -> Path:
        """Weight cache in the original (logical) expert order."""

    def golden_sequence(self) -> Path | None:
        """A sequence directory (metadata.json token_ids) with a reference the perf run checks against."""
        return None

    # --- device stages ---------------------------------------------------------------------------
    @abstractmethod
    def capture(self, sequence: Path, out_dir: Path, log: Path) -> dict:
        """Untraced prefill of ``sequence`` on the original cache; write out_dir/routing.npz (routing.save)."""

    @abstractmethod
    def run(self, mode: str, sequence: Path, cache: Path, out_dir: Path, log: Path) -> int:
        """Timed prefill. mode: 'perf' (traced, golden check + cache digests), 'nopcc' (traced, timing only),
        'profile' (stage timing). Writes out_dir/measurement.json with status and iteration_totals
        [{iteration, whole_seconds}], and cache-digests.jsonl in perf mode. Returns 0 on success."""

    @abstractmethod
    def prepare_cache(self, preset: Path, destination: Path, log: Path) -> None:
        """Write a placed weight cache for ``preset`` to ``destination`` (must not exist)."""

    # --- device housekeeping ---------------------------------------------------------------------
    def check_device_free(self) -> None:
        """Refuse to start a timed run while another user's device workload is on this host."""
        me = os.getuid()
        for line in subprocess.check_output(["ps", "-eo", "uid=,pid=,args="], text=True).splitlines():
            f = line.strip().split(None, 2)
            if (
                len(f) == 3
                and int(f[0]) != me
                and re.search(r"(pytest|ttrun|tt-run|tt-metal|agentic_models|vllm)", f[2])
            ):
                raise RuntimeError(f"another user's device workload is running: {line}")

    def reset_device(self, log: Path) -> None:
        with Path(log).open("w") as f:
            subprocess.run(["tt-smi", "-glx_reset"], stdout=f, stderr=subprocess.STDOUT, timeout=900)

    def describe(self) -> dict:
        return dict(family=self.family, verified=self.verified, chat_format=self.chat_format, **self.spec.to_json())
