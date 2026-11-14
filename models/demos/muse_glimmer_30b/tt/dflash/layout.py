# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Row layout of a 32-row DFlash2 step: which rows belong to which user slot."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RowLayout:
    """Row groups of a 32-row DFlash2 step: group g owns rows [g*R, (g+1)*R) for slot slots[g] (None: unused group).
    len(slots) * R <= 32."""

    R: int
    slots: tuple

    def __post_init__(self):
        object.__setattr__(self, "slots", tuple(self.slots))
        assert self.R >= 1, f"R must be >= 1, got {self.R}"
        assert len(self.slots) >= 1, "a layout needs at least one group"
        assert len(self.slots) * self.R <= 32, f"{len(self.slots)} groups x {self.R} rows exceed 32 rows"
        used = [s for s in self.slots if s is not None]
        assert len(used) == len(set(used)), f"duplicate slots in {self.slots}"

    @property
    def groups(self) -> int:
        return len(self.slots)

    @property
    def n_rows(self) -> int:
        return self.groups * self.R

    def rows(self, g: int) -> range:
        return range(g * self.R, (g + 1) * self.R)

    def slot_of_row(self, r: int):
        """Slot owning row r, or None (unused group, or a row past the groups)."""
        g = r // self.R
        return self.slots[g] if g < self.groups else None

    def group_of(self, slot: int) -> int:
        return self.slots.index(slot)
