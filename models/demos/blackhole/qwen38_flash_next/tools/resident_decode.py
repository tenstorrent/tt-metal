"""The resident decode chain's fixed points, shared by the chat session and the timing tool.

The traced HEAD/TAIL chain replays one graph per GDN ring phase (position mod 4); the constants here pin that
structure, the guards keep host I/O out of a trace body, and the row helpers read the token row the device resolves.
"""
from __future__ import annotations

import inspect
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Sequence

import torch
from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import DIAGNOSTIC_INPUT_TOKEN_ID
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TP_SIZE, replicate_tensor_2d_mesh_mapper
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import LOCAL_VOCAB_SIZE, TOKEN_ROW_SHAPE

EXPECTED_CACHE_LOADS = 48


PLE_EMBEDDING_WIDTH = 2560


# The row is written ROW_MAJOR, so these 5,120 B (4 x 1,280 B) are also the
# wire bytes; the body tilizes the row with one op.
PLE_EMBEDDING_UPLOAD_BYTES = 5120


GDN_TRACE_FORBIDDEN_BODY_CALLS = (
    "ReadDeviceProfiler",
    "as_tensor",
    "copy_host_to_device_tensor",
    "dump_tensor",
    "from_device",
    "from_torch",
    "load_tensor",
    "synchronize_device",
    "to_device",
    "to_torch",
)


# The replay loops of the two chain modes are guarded once, around the whole
# loop.  The loop's own host I/O (the PLE row write: from_torch and
# copy_host_to_device_tensor; the blocking token-row readback: to_torch through
# from_device) stays callable; every other forbidden call, synchronize_device
# above all, fails at its call site anywhere inside the loop.
REPLAY_LOOP_HOST_IO_CALLS = ("copy_host_to_device_tensor", "from_device", "from_torch", "to_torch")


# One token per TP4 vocabulary owner: the warm pass checks the device-token
# embedding against the host-token path on every coordinate's real and
# sentinel rows.
SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS = tuple(
    DIAGNOSTIC_INPUT_TOKEN_ID + shard * LOCAL_VOCAB_SIZE for shard in range(TP_SIZE)
)


# The GDN conv ring binds its four slot buffers by conv_phase = position mod 4
# when a graph is captured (Qwen38TTNNGDNState.conv_window), so one HEAD and one
# TAIL graph per residue class are captured, each with its own phase binding,
# and the host picks trace[replay counter mod 4]; no host integer enters a trace.
SINGLE_TRACE_RESIDUE_CLASS_TRACES = 4


# Trace index (part, residue, regime): the QSA indexer regime is a key field
# with one value until the indexer regime split adds a second TAIL regime.
SINGLE_TRACE_TRACE_PARTS = ("head", "tail")


SINGLE_TRACE_INDEXER_REGIME = 0


# Warm positions run eagerly before the miss guard: one per vocabulary owner
# (SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS), which also covers positions 0-3 of
# the first compressed block and the four ring phases.
SINGLE_TRACE_WARM_POSITIONS = len(SEQUENTIAL_TRACE_WARM_EMBEDDING_TOKEN_IDS)


class ResidentDecodeError(RuntimeError):
    """The target-only resident timing contract was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def program_cache_count(mesh: Any) -> int:
    value = mesh.num_program_cache_entries()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResidentDecodeError("program-cache entry count must be an exact nonnegative integer")
    return value


@contextmanager
def forbid_trace_body_host_io_and_sync(*, phase: str, allowed: Sequence[str] = ()):
    """Fail at the call site if capture/replay crosses a host or sync boundary.

    ``allowed`` names the forbidden calls a replay loop itself must make (its
    PLE row write and blocking token-row readback); they stay bound.
    """

    if not isinstance(phase, str) or not phase:
        raise TypeError("GDN trace body phase must be a nonempty string")
    if any(name not in GDN_TRACE_FORBIDDEN_BODY_CALLS for name in allowed):
        raise ValueError(f"guard allowance must name forbidden calls only, got {tuple(allowed)}")
    originals: dict[str, Any] = {}
    blocked_attempts: list[str] = []

    def blocked(name: str):
        def reject(*_args: Any, **_kwargs: Any) -> None:
            blocked_attempts.append(name)
            raise ResidentDecodeError(f"{phase} called forbidden ttnn.{name}")

        return reject

    for name in GDN_TRACE_FORBIDDEN_BODY_CALLS:
        if name in allowed:
            continue
        value = getattr(ttnn, name, None)
        if not callable(value):
            raise ResidentDecodeError(f"GDN trace guard cannot bind callable ttnn.{name}")
        originals[name] = value
        setattr(ttnn, name, blocked(name))
    try:
        yield blocked_attempts
    finally:
        restoration_failures = []
        for name, original in originals.items():
            setattr(ttnn, name, original)
            if getattr(ttnn, name, None) is not original:
                restoration_failures.append(name)
        if restoration_failures:
            raise ResidentDecodeError(f"GDN trace guard failed to restore TTNN callables: {restoration_failures}")


def b5b_runtime_surface() -> dict[str, Any]:
    """Signature audit of the loaded binary for the HEAD/TAIL host loop; fails instead of falling back.

    Requires ``ttnn.record_event``, ``ttnn.event_synchronize``, ``ttnn.MeshEvent``
    and one non-blocking device-to-host read (``ttnn.from_device(..., blocking=False)``
    or ``Tensor.cpu(blocking=False)``); reports whether the binary exposes an event
    poll (the loop waits with ``event_synchronize`` because the pinned runtime has none).
    """

    def first_line(name: str) -> str | None:
        value = getattr(ttnn, name, None)
        return None if value is None else ((inspect.getdoc(value) or "").strip().splitlines() or [""])[0]

    from_device_doc = inspect.getdoc(getattr(ttnn, "from_device", None)) or ""
    cpu_doc = inspect.getdoc(getattr(ttnn.Tensor, "cpu", None)) or ""
    poll_names = sorted(
        f"{label}.{name}"
        for label, namespace in (("ttnn", ttnn), ("ttnn._ttnn.events", ttnn._ttnn.events))
        for name in dir(namespace)
        if "query" in name.lower() and "event" in name.lower()
    )
    mesh_event_members = sorted(name for name in dir(getattr(ttnn, "MeshEvent", object)) if not name.startswith("_"))
    surface = {
        "from_device_blocking_keyword": "blocking (bool" in from_device_doc,
        "tensor_cpu_blocking_keyword": "blocking: bool" in cpu_doc,
        "record_event": first_line("record_event"),
        "event_synchronize": first_line("event_synchronize"),
        "wait_for_event": first_line("wait_for_event"),
        "execute_trace": first_line("_ttnn_execute_trace"),
        "mesh_event_present": hasattr(ttnn, "MeshEvent"),
        "mesh_event_public_members": mesh_event_members,
        "event_poll_names": poll_names,
        "nonblocking_event_poll_exposed": bool(poll_names) or bool(mesh_event_members),
        "trace_alloc_tracking": bool(TRACE_ALLOC_TRACKING),
    }
    missing = [
        name
        for name in ("record_event", "event_synchronize", "_ttnn_execute_trace", "copy_host_to_device_tensor")
        if not callable(getattr(ttnn, name, None))
    ]
    if missing:
        raise ResidentDecodeError(f"the loaded binary has no callable ttnn.{missing}")
    if not surface["mesh_event_present"]:
        raise ResidentDecodeError("the loaded binary has no ttnn.MeshEvent")
    if "blocking: bool" not in (surface["execute_trace"] or ""):
        raise ResidentDecodeError(f"execute_trace has no blocking keyword: {surface['execute_trace']}")
    if surface["from_device_blocking_keyword"]:
        surface["nonblocking_read"] = "ttnn.from_device(local, blocking=False)"
    elif surface["tensor_cpu_blocking_keyword"]:
        surface["nonblocking_read"] = "Tensor.cpu(blocking=False)"
    else:
        raise ResidentDecodeError("the loaded binary exposes no non-blocking device-to-host read")
    return surface


def download_trace_tensor(tensor: Any, *, label: str) -> tuple[torch.Tensor, ...]:
    locals_in_order = tuple(ttnn.get_device_tensors(tensor))
    if len(locals_in_order) != 4:
        raise ResidentDecodeError(f"{label} has {len(locals_in_order)} device tensors, expected four")
    return tuple(ttnn.to_torch(local).contiguous() for local in locals_in_order)


def ple_host_row(row_mapper: Any, payload: bytearray) -> tuple[Any, torch.Tensor]:
    """The 5,120 B of one PLE row as a ROW_MAJOR bf16 host mesh tensor, width-sharded four ways.

    The persistent PLE input is interleaved DRAM ROW_MAJOR bf16 [1,1,1,640] per
    device; copy_host_to_device_tensor writes these four 1,280 B shards straight
    into it (same shape, dtype and layout, so the captured address is kept and
    no device allocation, eager copy or host tilize happens per token) and the
    captured body tilizes the row.  Returns the host tensor and its torch view.
    """

    if len(payload) != PLE_EMBEDDING_UPLOAD_BYTES:
        raise ResidentDecodeError(f"host PLE lookup must return {PLE_EMBEDDING_UPLOAD_BYTES} B")
    row = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(1, 1, 1, PLE_EMBEDDING_WIDTH)
    return ttnn.from_torch(row, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=row_mapper), row


def host_token_row(value: float) -> torch.Tensor:
    """Host image of a token row: FP32 [1,1,1,32] with ``value`` at column 0 and zeros elsewhere."""

    row = torch.zeros(TOKEN_ROW_SHAPE, dtype=torch.float32)
    row[..., 0] = float(value)
    return row


def token_row_value(token_row: Any) -> int:
    """The id in one device's replica of a token row: the whole per-token host readback (blocking)."""

    return token_row_host_value(ttnn.to_torch(ttnn.get_device_tensors(token_row)[0]))


def token_row_host_value(host: torch.Tensor) -> int:
    """The id in a token row already on the host (the evented read's tensor after its event completed)."""

    if host.dtype != torch.float32 or tuple(host.shape) != TOKEN_ROW_SHAPE:
        raise ResidentDecodeError(
            f"token row readback is {host.dtype} {tuple(host.shape)}, expected FP32 {TOKEN_ROW_SHAPE}"
        )
    value = host[0, 0, 0, 0].item()
    if not math.isfinite(value) or value != int(value):
        raise ResidentDecodeError(f"token row holds the non-integral value {value}")
    return int(value)


def require_token_row_holds(token_row: Any, expected: torch.Tensor, *, label: str) -> None:
    """Require all four replicas of ``token_row`` to equal the host row ``expected``."""

    for coordinate, actual in enumerate(download_trace_tensor(token_row, label=label)):
        if actual.dtype != torch.float32 or tuple(actual.shape) != TOKEN_ROW_SHAPE:
            raise ResidentDecodeError(
                f"{label} replica {coordinate} is {actual.dtype} {tuple(actual.shape)}, expected FP32 {TOKEN_ROW_SHAPE}"
            )
        if not torch.equal(actual, expected):
            # Column 0 is the id; report both sides so a hardware mismatch names the
            # resolved owner instead of only the fact of a difference.
            raise ResidentDecodeError(
                f"{label} replica {coordinate} differs from the expected token row "
                f"(column 0: device {actual[0, 0, 0, 0].item()!r}, expected {expected[0, 0, 0, 0].item()!r}; "
                f"nonzero device columns {torch.nonzero(actual.reshape(-1)).reshape(-1).tolist()})"
            )


def require_token_row_resolves(token_row: Any, candidates: Any, lm_head: Any, *, label: str) -> int:
    """Require every replica of ``token_row`` to hold the CPU resolve of ``candidates``; returns that id."""

    token_id = int(lm_head.resolve_greedy(candidates).reshape(-1)[0].item())
    require_token_row_holds(token_row, host_token_row(token_id), label=label)
    return token_id


def device_token_row(mesh: Any, token_id: int) -> Any:
    """Replicated host image of a token row, written into the persistent token row without a device allocation."""

    return ttnn.from_torch(
        host_token_row(token_id),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate_tensor_2d_mesh_mapper(mesh),
    )
