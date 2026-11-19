# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Internal table lease; public raw tensor exposure revokes bulk-fill eligibility."""

from .optimized_decoder import PagedState


class _TableLease:
    def __init__(self, table, certified):
        self.table = table
        self.certified = certified

    def __deepcopy__(self, memo):
        # Device tensor wrappers may deepcopy while retaining device storage.
        # Share revocation across every state alias, just as shallow copy does.
        memo[id(self)] = self
        return self


class CertifiedPagedState(PagedState):
    def __init__(self, key, value, table, *, certified):
        self.key, self.value = key, value
        self._table_lease = _TableLease(table, certified)

    @property
    def page_table(self):
        # Any caller can retain this tensor and mutate it later, so exposure
        # permanently revokes this lease, including all shallow state copies.
        self._table_lease.certified = False
        return self._table_lease.table

    @page_table.setter
    def page_table(self, table):
        self._table_lease.certified = False
        self._table_lease = _TableLease(table, False)

    @property
    def page_table_shape(self):
        return tuple(self._table_lease.table.shape)

    @property
    def page_table_address(self):
        return int(self._table_lease.table.buffer_address())

    def _replace_with_fresh_private_table(self, table, *, certified):
        # Setup-only: never recertify a tensor previously exposed to callers.
        assert table is not self._table_lease.table
        self._table_lease.certified = False
        self._table_lease = _TableLease(table, certified)

    def __repr__(self):
        return f"CertifiedPagedState(table_shape={self.page_table_shape}, certified={self._table_lease.certified})"


def page_table_for_attention(state):
    # Internal read-only use only. Never return this handle from public APIs.
    return state._table_lease.table if isinstance(state, CertifiedPagedState) else state.page_table


def page_table_certified(state):
    return isinstance(state, CertifiedPagedState) and state._table_lease.certified


def certify_host_page_rows(rows):
    # CPU setup only. Negative -1 is a skip entry, not an owned physical page.
    values = [int(value) for row in rows for value in row if int(value) >= 0]
    return len(values) == len(set(values))
