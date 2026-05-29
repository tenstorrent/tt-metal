# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Custom pytest-xdist scheduler used together with the worker barrier plugin.

This scheduler is the master-side half of the "parallel within a file, serial
across files, reset before each file" pipeline. Its only responsibility is to hand each
worker its full, scope-ordered list of test indices in a single batch, then
mark the worker for shutdown. All cross-worker synchronisation and the
``tt-smi -r`` between files happen worker-side, in
``helpers.barrier_plugin.WorkerBarrierPlugin`` (see that module for the why).

Why "one big batch" instead of dispatching scope-by-scope:

Each xdist worker peeks at its *next* test index before running the current
one (it needs ``nextitem`` for ``pytest_runtest_teardown``). A barrier
scheduler that withholds the next scope until the previous one is fully
reported back deadlocks: the worker blocks at the peek of its (N+1)-th item
before ever running its N-th, so master never gets the N-th completion and
never releases the barrier. By front-loading everyone's queue (and tacking on
``SHUTDOWN`` so the very last peek doesn't block either) we sidestep that
class of deadlock entirely; the only synchronisation point that remains is
worker-side and runs *inside* ``pytest_runtest_setup`` -- well after the
previous test has already reported back to master.

Trade-off: workers idle at each barrier waiting for the slowest peer to
finish the current file. In exchange, hardware state is isolated between
files.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import pytest
from xdist.remote import Producer
from xdist.scheduler.loadfile import LoadFileScheduling


class LoadFileBarrierScheduling(LoadFileScheduling):
    """File-grouped scheduler with one-shot scope-ordered dispatch.

    Inherits ``_split_scope`` (split at the first ``::``) from
    ``LoadFileScheduling`` so all tests sharing a file path share a scope key.
    """

    def __init__(
        self,
        config: pytest.Config,
        log: Optional[Producer] = None,
    ) -> None:
        super().__init__(config, log)
        if log is None:
            self.log = Producer("loadfilebarriersched")
        else:
            self.log = log.loadfilebarriersched

    def schedule(self) -> None:
        """Dispatch every test to its worker in one shot, then shutdown.

        Each worker receives a list of indices ordered: all of scope A's
        round-robin slice for this worker, then scope B's, etc. The trailing
        ``node.shutdown()`` queues a ``SHUTDOWN`` marker behind the last test
        so the worker's final peek doesn't block.
        """
        assert self.collection_is_completed

        # LoadScopeScheduling.schedule() may be invoked multiple times (e.g.
        # when a new node joins after a crash). One-shot dispatch happens only
        # the first time; later calls just rebalance via the parent class.
        if self.collection is not None:
            return

        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return

        self.collection = list(next(iter(self.registered_collections.values())))
        if not self.collection:
            return

        # Build the scope-ordered workqueue (file path -> ordered nodeids).
        # OrderedDict preserves the order in which scopes first appear in the
        # collection, which is the order pytest's collector returned them.
        scoped: "OrderedDict[str, list[str]]" = OrderedDict()
        for nodeid in self.collection:
            scope = self._split_scope(nodeid)
            scoped.setdefault(scope, []).append(nodeid)

        nodes = list(self.nodes)
        if not nodes:
            return

        # per_node[node] = ordered list of nodeids for that worker, with all of
        # scope A's slice first, then scope B's, etc.
        per_node: dict = {n: [] for n in nodes}

        # Mirror into self.assigned_work in the same shape LoadScopeScheduling
        # uses (node -> scope -> {nodeid: completed_bool}); the parent class's
        # _pending_of() reads this and drives tests_finished / has_pending.
        for scope, nodeids in scoped.items():
            for i, nodeid in enumerate(nodeids):
                node = nodes[i % len(nodes)]
                per_node[node].append(nodeid)
                self.assigned_work[node].setdefault(scope, {})[nodeid] = False

        total_scopes = len(scoped)
        for node, ordered_nodeids in per_node.items():
            worker_collection = self.registered_collections[node]
            index_map = {nid: i for i, nid in enumerate(worker_collection)}
            indices = [index_map[nid] for nid in ordered_nodeids]
            self.log(
                f"Dispatching {len(indices)} tests across {total_scopes} scopes "
                f"to {node.gateway.id}"
            )
            if indices:
                node.send_runtest_some(indices)
            # SHUTDOWN goes after the last real index so the worker's final
            # torun.get() peek returns the SHUTDOWN sentinel instead of
            # blocking. The worker then runs its last test with nextitem=None
            # and exits the runtestloop cleanly.
            node.shutdown()

        # Empty out the workqueue: there is nothing left for the parent
        # class's _reschedule to dispatch.
        self.workqueue.clear()
