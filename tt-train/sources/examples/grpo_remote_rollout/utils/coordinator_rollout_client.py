# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trainer-facing client built on the asynchronous rollout coordinator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import RLock
from time import monotonic
from typing import Any

from .rollout_coordinator import SingleWorkerRolloutCoordinator
from .rollout_engine import PolicyVersion, PromptGroupLease, RolloutResult


class CoordinatorRolloutClient:
    """Adapt coordinator leases and cutovers to the remote-completer surface.

    The first implementation permits one submitted prompt group and one policy
    cutover at a time, matching the single-worker engine.  ``await_rollout`` is
    the lossless API: it returns tokens, behavior-policy log probabilities, and
    identity metadata in one :class:`RolloutResult`.  ``await_remote_generate``
    is a temporary compatibility shim for completers that consume only tokens;
    the complete result remains available through :attr:`last_result`.

    Weight publication has both a two-phase API and a blocking convenience
    wrapper.  Call ``begin_weight_update`` while the old-policy rollout is
    active, then ``await_weight_update`` at the version boundary to overlap
    transfer with generation.

    MULTI-WORKER EXTENSION: select a coordinator before creating each lease and
    keep pending leases in a ``lease_id -> coordinator`` map.  The public result
    and weight-update contracts do not otherwise need to change.
    """

    def __init__(
        self,
        *,
        coordinator: SingleWorkerRolloutCoordinator,
        max_new_tokens: int,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        self._coordinator = coordinator
        self._max_new_tokens = max_new_tokens
        self._id_factory = id_factory
        self._next_id = 0
        self._pending_lease_id: str | None = None
        self._pending_policy_version: PolicyVersion | None = None
        self._last_result: RolloutResult | None = None
        self._lock = RLock()

    @property
    def active_version(self) -> PolicyVersion:
        return self._coordinator.active_version

    @property
    def last_result(self) -> RolloutResult | None:
        """Most recently delivered full result, including behavior logprobs."""
        with self._lock:
            return self._last_result

    def submit_remote_generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        timeout: float | None = None,
    ) -> None:
        """Submit one already-expanded prompt group without waiting for it."""
        if max_new_tokens != self._max_new_tokens:
            raise ValueError(
                f"request max_new_tokens {max_new_tokens} does not match "
                f"rollout engine configuration {self._max_new_tokens}"
            )
        payload = [[int(token) for token in prompt] for prompt in prompts]
        with self._lock:
            if self._pending_lease_id is not None:
                raise RuntimeError(f"rollout lease {self._pending_lease_id!r} is still pending")
            if self._pending_policy_version is not None:
                raise RuntimeError(f"policy version {self._pending_policy_version} is still activating")
            identity = self._new_identity()
            lease = PromptGroupLease(
                lease_id=identity,
                group_id=identity,
                behavior_version=self._coordinator.active_version,
                payload=payload,
            )
            self._coordinator.submit(lease, timeout=timeout)
            self._pending_lease_id = lease.lease_id

    def await_rollout(self, *, timeout: float | None = None) -> RolloutResult:
        """Return the complete result for the one pending prompt-group lease."""
        with self._lock:
            lease_id = self._pending_lease_id
            if lease_id is None:
                raise RuntimeError("no rollout is pending")

        result = self._coordinator.receive_result(timeout=timeout)
        if result.lease_id != lease_id:
            raise RuntimeError(f"received lease {result.lease_id!r}; expected pending lease {lease_id!r}")

        with self._lock:
            self._pending_lease_id = None
            self._last_result = result
        return result

    def await_remote_generate(self, *, timeout: float | None = None) -> list[list[int]]:
        """Compatibility API returning tokens while retaining the full result."""
        result = self.await_rollout(timeout=timeout)
        return [list(tokens) for tokens in result.output.tokens]

    def begin_weight_update(
        self,
        weights: Any,
        *,
        version: PolicyVersion | None = None,
        timeout: float | None = None,
    ) -> PolicyVersion:
        """Start staging a newer policy, potentially alongside generation."""
        with self._lock:
            if self._pending_policy_version is not None:
                raise RuntimeError(f"policy version {self._pending_policy_version} is already activating")
            target = self._coordinator.active_version + 1 if version is None else version
            self._coordinator.begin_policy_cutover(target, weights, timeout=timeout)
            self._pending_policy_version = target
            return target

    def await_weight_update(
        self,
        version: PolicyVersion | None = None,
        *,
        timeout: float | None = None,
    ) -> PolicyVersion:
        """Wait until the staged version is active and new leases are safe."""
        with self._lock:
            pending = self._pending_policy_version
            if pending is None:
                raise RuntimeError("no policy update is pending")
            if version is not None and version != pending:
                raise ValueError(f"waiting for policy version {version}, but version {pending} is pending")

        self._coordinator.await_policy_activation(pending, timeout=timeout)
        with self._lock:
            self._pending_policy_version = None
        return pending

    def send_weights(
        self,
        weights: Any,
        *,
        version: PolicyVersion | None = None,
        timeout: float | None = None,
    ) -> PolicyVersion:
        """Stage and activate one policy update before returning."""
        deadline = None if timeout is None else monotonic() + timeout
        target = self.begin_weight_update(weights, version=version, timeout=timeout)
        remaining = None if deadline is None else max(0.0, deadline - monotonic())
        return self.await_weight_update(target, timeout=remaining)

    def close(self) -> None:
        self._coordinator.close()

    def shutdown(self) -> None:
        """Compatibility alias for the legacy rollout client."""
        self.close()

    def _new_identity(self) -> str:
        if self._id_factory is not None:
            identity = self._id_factory()
        else:
            identity = f"{self._coordinator.engine_id}-lease-{self._next_id}"
            self._next_id += 1
        if not identity:
            raise ValueError("id_factory must return a non-empty identity")
        return identity


__all__ = ["CoordinatorRolloutClient"]
