# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Supported model/SKU pairs, matching the main model e2e CI entries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    arch: str
    shape: tuple[int, int]
    mesh_name: str


HARDWARE = {
    "wh_n150": Hardware("wormhole_b0", (1, 1), "N150"),
    "bh_p150": Hardware("blackhole", (1, 1), "P150"),
    "wh_llmbox_perf": Hardware("wormhole_b0", (1, 8), "T3K"),
    "bh_quietbox_2": Hardware("blackhole", (1, 4), "P150x4"),
    "wh_galaxy_perf": Hardware("wormhole_b0", (4, 8), "TG"),
    "bh_galaxy": Hardware("blackhole", (4, 8), "TG"),
}


@dataclass(frozen=True)
class Profile:
    family: str
    hf_model: str
    skus: tuple[str, ...]

    def capacity(self, sku):
        if self.family == "gpt_oss":
            # The production demo supports throughput experts only on WH.
            return 128 if sku == "wh_galaxy_perf" else 1
        return 32


PROFILES = {
    "llama3.1-8b": Profile(
        "llama",
        "meta-llama/Llama-3.1-8B-Instruct",
        ("wh_n150", "bh_p150", "wh_llmbox_perf", "bh_quietbox_2"),
    ),
    "gemma-4-26b-a4b": Profile(
        "gemma",
        "google/gemma-4-26B-A4B-it",
        ("wh_llmbox_perf", "bh_quietbox_2"),
    ),
    "qwen3.6-27b": Profile("qwen", "Qwen/Qwen3.6-27B", ("bh_quietbox_2",)),
    "qwen3.6-35b-a3b": Profile("qwen", "Qwen/Qwen3.6-35B-A3B", ("bh_quietbox_2",)),
    "gpt-oss-120b": Profile("gpt_oss", "openai/gpt-oss-120b", ("wh_galaxy_perf", "bh_quietbox_2", "bh_galaxy")),
}


def select_sku(backend, arch, device_count, requested=None):
    profile = PROFILES[backend]
    if requested is None:
        candidates = [
            sku
            for sku in profile.skus
            if HARDWARE[sku].arch == arch and HARDWARE[sku].shape[0] * HARDWARE[sku].shape[1] == device_count
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Select --model-behavior-sku explicitly for {backend} on {arch}/{device_count} devices; "
                f"CI supports {profile.skus}"
            )
        requested = candidates[0]
    if requested not in profile.skus:
        raise ValueError(f"{backend} is not enabled on {requested} in CI; supported SKUs: {profile.skus}")
    hardware = HARDWARE[requested]
    if hardware.arch != arch or hardware.shape[0] * hardware.shape[1] > device_count:
        raise ValueError(f"{requested} requires {hardware.arch} mesh {hardware.shape}; found {arch}/{device_count}")
    return requested
