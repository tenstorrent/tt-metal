# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only ownership proof for scheduler relocation of unfinished prefills."""

import torch


def plan_pending_prefill(*, tables, shadows, positions, pending, slots, starts):
    """Return a full destination-to-source permutation, without mutating state.

    Exclusive shared-pool live prefixes identify a pending continuation. A
    same-slot authoritative COW update remains valid; a moved COW prefix with
    no identity proof is rejected. Padding and newly appended pages are not
    part of the identity, but the caller validates all live pages afterwards.
    """
    width = len(positions)
    mapping = {}
    for row, (destination, start) in enumerate(zip(slots, starts, strict=True)):
        if not start:
            continue
        matches = []
        if tables is not None:
            live = (start + 31) // 32
            for source in pending:
                if positions[source] != start or pending[source] != start:
                    continue
                old = shadows.get(source)
                old = (old,) * len(tables) if isinstance(old, torch.Tensor) else old
                if (
                    old is not None
                    and len(old) == len(tables)
                    and all(
                        torch.equal(table[row, :live], previous[0, :live])
                        for table, previous in zip(tables, old, strict=True)
                    )
                ):
                    matches.append(source)
        if len(matches) > 1:
            raise ValueError("Ambiguous pending prefill ownership")
        if matches:
            source = matches[0]
        elif positions[destination] == start:
            # Existing same-slot COW support; never infer cross-slot identity
            # from a partial prefix or merely an equal committed position.
            source = destination
        else:
            raise ValueError("Cannot prove moved pending prefill ownership")
        if source in mapping.values():
            raise ValueError("Two prefill rows claim the same pending owner")
        mapping[destination] = source
    used = set(mapping.values())
    for destination, source in mapping.items():
        if destination != source and positions[destination] and destination not in used:
            raise ValueError("Pending prefill move would overwrite an unrelated live owner")
    # Preserve every unrelated holder. Fill the vacated ends of movement chains
    # with the empty destinations, completing cycles for fixed-address copies.
    for slot in range(width):
        if slot not in mapping and slot not in used:
            mapping[slot] = slot
            used.add(slot)
    remaining_destinations = [slot for slot in range(width) if slot not in mapping]
    remaining_sources = [slot for slot in range(width) if slot not in used]
    mapping.update(zip(remaining_destinations, remaining_sources, strict=True))
    return tuple(mapping[slot] for slot in range(width))
