# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Which L1 formats an architecture has, and the checks that depend on it.

Every unit needs this and none of them owns it. Its own module so the units
and :mod:`.arch` can both import it without a cycle. An architecture holds
units, so a unit cannot hold the architecture.

It carries the architecture's `name` so a refusal can say which device refused.
"""

from dataclasses import dataclass
from typing import FrozenSet, Sequence, Union

import torch
from helpers.format_config import DataFormat

from .l1_codec import (
    MODELLED_L1_FORMATS,
    _no_codec_message,
    pack_to_l1,
    unpack_from_l1,
)
from .storage import DEST_STORAGE_FORMATS, as_dest_acc


@dataclass(frozen=True)
class L1Support:
    """An architecture's L1 format set, and the questions asked of it."""

    #: Shown in refusals, so they name the device rather than the shared class.
    name: str

    #: L1 formats this architecture's unpacker can read.
    formats: FrozenSet[DataFormat]

    def __post_init__(self) -> None:
        if not self.formats:
            raise TypeError(
                f"{self.name} declares no L1 formats, so it can read nothing "
                f"from L1."
            )

    def supports(self, l1_format: DataFormat) -> bool:
        """Whether this golden can move `l1_format` through L1 on this arch.

        Two conditions, and not the same question: the architecture has to hold the
        format, *and* this golden has to have a codec for it. :meth:`check` says
        which of the two failed.
        """
        return l1_format in self.formats and l1_format in MODELLED_L1_FORMATS

    def check(self, l1_format: DataFormat) -> None:
        # Two failures, kept apart because they call for opposite responses:
        # a format the hardware cannot hold means the test asks the
        # impossible, and a missing codec means the model has a hole.
        if l1_format not in self.formats:
            raise ValueError(
                f"{self.name} cannot read {l1_format} from L1 on this " f"architecture."
            )
        if l1_format not in MODELLED_L1_FORMATS:
            raise ValueError(_no_codec_message(l1_format))

    @property
    def dest_formats(self) -> FrozenSet[DataFormat]:
        """Dest formats available on this architecture."""
        return DEST_STORAGE_FORMATS & self.formats

    def check_dest_format(self, dest_format: DataFormat) -> None:
        if dest_format not in self.dest_formats:
            raise ValueError(
                f"{dest_format} is not a Dest format on {self.name}. "
                f"Dest can hold {sorted(str(f) for f in self.dest_formats)}."
            )

    # ------------------------------------------------------------------
    # L1 access. Here rather than on a unit because both the unpacker and the
    # packer need it, and both need the same support check first.
    # ------------------------------------------------------------------

    def write(self, tensor: torch.Tensor, l1_format: DataFormat, **geometry) -> list:
        """Lay `tensor` out in L1 as `l1_format`. Where precision is lost."""
        self.check(l1_format)
        return pack_to_l1(tensor, l1_format, **_normalised_geometry(geometry))

    def read(
        self, l1_bytes: "L1Buffer", l1_format: DataFormat, **geometry
    ) -> torch.Tensor:
        """Read `l1_bytes` back as values, before any register conversion."""
        self.check(l1_format)
        if isinstance(l1_bytes, torch.Tensor):
            raise TypeError(
                "L1 holds bytes, not a tensor. Build a buffer with "
                f"pack_to_l1(tensor, {l1_format}) and pass that instead."
            )
        return unpack_from_l1(l1_bytes, l1_format, **_normalised_geometry(geometry))


L1Buffer = Union[Sequence[int], bytes]


def _normalised_geometry(geometry: dict) -> dict:
    """`geometry` with any ``dest_acc`` reduced to a bool.

    The codec tests it with a plain ``if`` and ``DestAccumulation`` has no
    ``__bool__``, so ``DestAccumulation.No`` would select the 32-bit SrcS slice
    layout, 1280 bytes where 1152 is correct, and nothing downstream notices.

    Every block reaches L1 through :meth:`L1Support.write` or
    :meth:`L1Support.read`, so normalising in both covers the lot.
    """
    if "dest_acc" not in geometry:
        return geometry
    return {**geometry, "dest_acc": as_dest_acc(geometry["dest_acc"])}
