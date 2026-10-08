# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Kernel-visible de-duplication of reconfig test format variants.

``TestConfig`` rebuilds a ``FormatConfig`` from only ``unpack_A_src``, ``unpack_B_src``
and ``pack_dst`` (the rest is inferred), so many nominal ``(formats, dest_acc)``
variants build the same kernel with the same runtime formats. Keeping one variant
per kernel-visible key drops only exact duplicates.
"""

from typing import Callable, Hashable

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import data_formats, effective_dest_acc
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import DestAccumulation
from helpers.test_config import TestConfig

ReconfigFormats = tuple[DataFormat, DataFormat, DataFormat, DataFormat]


def _format_config_key(config: FormatConfig) -> Hashable:
    return tuple(sorted(vars(config).items()))


def kernel_visible_key(
    formats: ReconfigFormats, dest_acc: DestAccumulation
) -> Hashable:
    """The formats and Dest mode the kernel receives, inferred exactly as ``TestConfig`` does."""
    requested = FormatConfig(*formats, DataFormat.Float32)
    arch = get_chip_architecture()
    inferred = data_formats(
        input_format=requested.input_format,
        input_format_B=requested.input_format_B,
        output_format=requested.output_format,
        is_fp32_dest_acc_en=dest_acc,
        num_iterations=1,
        chip_arch=arch,
    )
    effective = effective_dest_acc(
        requested.input_format, requested.output_format, dest_acc, arch
    )
    return tuple(_format_config_key(config) for config in inferred), effective


def configured_key(configuration: TestConfig) -> Hashable:
    """The kernel-visible key of a constructed ``TestConfig``, to check ``kernel_visible_key`` against."""
    return (
        tuple(_format_config_key(config) for config in configuration.formats_config),
        configuration.dest_acc,
    )


def kernel_visible_variants(
    formats_list: list[ReconfigFormats],
    get_dest_acc: Callable[[ReconfigFormats], list[DestAccumulation]],
) -> dict[ReconfigFormats, list[DestAccumulation]]:
    """First ``(formats, dest_acc)`` variant of every kernel-visible key, grouped by formats.

    @note On Quasar, return the nominal variants unchanged: these WH/BH modules are
    collected there only to be deselected, and Quasar inference rejects their formats.
    """
    if get_chip_architecture() == ChipArchitecture.QUASAR:
        return {formats: get_dest_acc(formats) for formats in formats_list}
    seen = set()
    variants: dict[ReconfigFormats, list[DestAccumulation]] = {}
    for formats in formats_list:
        for dest_acc in get_dest_acc(formats):
            key = kernel_visible_key(formats, dest_acc)
            if key in seen:
                continue
            seen.add(key)
            variants.setdefault(formats, []).append(dest_acc)
    return variants
