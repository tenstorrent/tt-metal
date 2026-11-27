"""Validate page ownership and order same-batch prefix producers before readers."""

import torch


def prefill_dependencies(tokens, table, starts, ends, *, first_page, num_pages):
    writers = {}
    for row, (start, end) in enumerate(zip(starts, ends)):
        for block in range(start // 32, (end + 31) // 32):
            page = int(table[row, block])
            if not first_page <= page < num_pages:
                raise ValueError(f"Writable page {page} outside the request pool")
            if page in writers:
                raise ValueError(f"Prefill writes require exclusive physical pages: {page}")
            writers[page] = (row, block)
    dependencies = [{} for _ in starts]
    for row, start in enumerate(starts):
        for block, page in enumerate(table[row, : start // 32].tolist()):
            if page not in writers:
                continue
            owner, owner_block = writers[page]
            boundary = (block + 1) * 32
            if (
                owner == row
                or owner_block != block
                or ends[owner] < boundary
                or not torch.equal(tokens[row, :boundary], tokens[owner, :boundary])
            ):
                raise ValueError(f"Invalid cached-prefix alias on page {page}: reader {row}, writer {owner}")
            dependencies[row][owner] = max(boundary, dependencies[row].get(owner, 0))
    # Validate the whole graph before any device work. Legal aliases always
    # point toward an earlier starting producer, but retain an explicit check.
    pending = set(range(len(starts)))
    while pending:
        ready = {row for row in pending if not pending.intersection(dependencies[row])}
        if not ready:
            raise ValueError("Cyclic prefill page dependencies")
        pending.difference_update(ready)
    return dependencies


def ready_prefill_rows(starts, ends, dependencies):
    return [
        row
        for row, (start, end) in enumerate(zip(starts, ends))
        if start < end and all(starts[owner] >= boundary for owner, boundary in dependencies[row].items())
    ]
