"""The MTP pass's on-device point-mass acceptance as the sampling chain serves it.

``QWEN38_MTP_DEVICE_ACCEPT=1`` lets an ``--mtp --sampling`` server decide its passes on the device: the pass loop's
third verify form runs ``fused/mtp_accept`` between the verify head and its tail, under the device sampler's law
(the temperature table's integer weights, exact prefix sums, one fp32 multiply per decision, no division: row ``j``
accepts ``d_{j+1}`` iff ``fl32(u_j * S_j) < w_j(d_{j+1})``, a tie rejects; the first rejection and the bonus row draw
with a second uniform).  The host-decided split form (``accept_pass``: the host sampler's fp32 law) stays the form
for every request this admission refuses and for servers without the switch.  A response names its arithmetic
(``mtp_acceptance_arithmetic``: ``device-theta`` or ``host-fp32``) and the server fingerprint carries
``-device-accept``: a seed reproduces one stream per arithmetic, never across them.

This module holds the switch, the per-request admission, the constants (the sampler tail's, ``rows = k + 1``: the
policy row, the uniforms row ``[u_0 .. u_{k-1}, v]``, the temperature table), the per-pass uniforms hook the pass
loop calls before the verify launch, and the re-derivation of a recorded pass with :func:`mtp_accept.accept_reference`
(the wired-loop gate: 0 mismatches expected, the same arithmetic on both sides).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.device_sampler import Qwen38DeviceSamplerPolicy, weight_table
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept, sampler_tail
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import UniformStream

SWITCH = "QWEN38_MTP_DEVICE_ACCEPT"
ARITHMETIC_DEVICE = "device-theta"
ARITHMETIC_HOST = "host-fp32"
# The constants are the sampler tail's (rows <= its MAX_ROWS) and the kernel holds k + 1 rows: k <= 4 today.
ADMITTED_DRAFTS = min(mtp_accept.MAX_DRAFTS, sampler_tail.MAX_ROWS - 1)


def device_accept_switch(environ: Mapping[str, str] = os.environ) -> bool:
    """``QWEN38_MTP_DEVICE_ACCEPT``: ``1`` on, ``0`` or unset off, anything else refused."""

    raw = environ.get(SWITCH, "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{SWITCH} must be 1 or 0, got {raw!r}")


def refusal(request: Any, drafts: int) -> str | None:
    """Why a request keeps the host-decided pass, or ``None`` when the device decides it: a greedy request (the
    greedy accept program), logprobs (no row normaliser per verify row), any penalty (the kernel reads no history),
    a policy the table cannot serve (``Qwen38DeviceSamplerPolicy.refusal``), more drafts than the constants hold."""

    if request is None:
        return "refused: greedy request"
    if request.logprobs or request.top_logprobs:
        return "refused: logprobs"
    p = request.parameters
    if p.presence_penalty != 0 or p.frequency_penalty != 0 or p.repetition_penalty != 1:
        return "refused: penalties"
    why = Qwen38DeviceSamplerPolicy.refusal(p)
    if why is not None:
        return f"refused: {why}"
    if drafts > ADMITTED_DRAFTS:
        return f"refused: k {drafts} above {ADMITTED_DRAFTS}"
    return None


class Qwen38DeviceAcceptance:
    """The chain's device acceptance for one draft count: its constants and the per-request protocol (the policy at
    request start, ``k + 1`` uniforms before every verify launch, from the request's splitmix64 stream)."""

    def __init__(self, drafts: int, constants: Any) -> None:
        if isinstance(drafts, bool) or type(drafts) is not int or not 1 <= drafts <= ADMITTED_DRAFTS:
            raise ValueError(f"device acceptance serves 1..{ADMITTED_DRAFTS} drafts, got {drafts!r}")
        for name in ("write_policy", "write_uniforms", "release", "mark_corruptible"):
            if not hasattr(constants, name):
                raise TypeError(f"constants need {name} (the sampler tail's constants with rows = k + 1)")
        self.drafts = drafts
        self.constants = constants

    @classmethod
    def build(cls, mesh, mesh_contract, *, drafts: int) -> "Qwen38DeviceAcceptance":
        if drafts > ADMITTED_DRAFTS:
            raise ValueError(f"{SWITCH} serves k <= {ADMITTED_DRAFTS}, got --mtp {drafts}")
        return cls(drafts, sampler_tail.Qwen38TTNNSamplerTailConstants.build(mesh, mesh_contract, rows=drafts + 1))

    @property
    def rows(self) -> int:
        return self.drafts + 1

    def admission(self, request: Any) -> str | None:
        return refusal(request, self.drafts)

    def begin_request(self, request: Any) -> None:
        """Request start (admitted requests only): the policy row, the request's stream restarted from its seed,
        the arithmetic named on the request."""

        why = self.admission(request)
        if why is not None:
            raise ValueError(f"device acceptance begins an admitted request only ({why})")
        policy = Qwen38DeviceSamplerPolicy.from_parameters(request.parameters)
        if policy is None:  # the admission checked the same rule
            raise ValueError("no device policy for an admitted request")
        self.constants.write_policy(policy)
        request.uniforms.clear()
        request.stream = UniformStream(request.parameters.seed)
        request.mtp_arithmetic = ARITHMETIC_DEVICE

    def before_verify(self, request: Any) -> Callable[[Sequence[int]], None]:
        """The pass loop's ``before_verify_sampled`` hook: ``[u_0 .. u_{k-1}, v]`` from the request's stream into
        the uniforms row, before the verify launch that reads them.  ``tokens`` is the pass's ``[t, d_1 .. d_k]``."""

        rows = self.rows

        def write(tokens: Sequence[int]) -> None:
            if len(tokens) != rows:
                raise ValueError(f"a {self.drafts}-draft pass has {rows} verify tokens, got {len(tokens)}")
            self.constants.write_uniforms([request.next_uniform() for _ in range(rows)])

        return write

    def mark_corruptible(self) -> None:
        self.constants.mark_corruptible()

    def release(self) -> None:
        self.constants.release()


def for_chain(mesh, mesh_contract, *, drafts: int | None, environ: Mapping[str, str] = os.environ):
    """The chain's device acceptance under the switch, or ``None`` (switch off, or no MTP chain)."""

    if drafts is None or not device_accept_switch(environ):
        return None
    return Qwen38DeviceAcceptance.build(mesh, mesh_contract, drafts=drafts)


# --- the gate's re-derivation ------------------------------------------------------------------------------------


def evaluated_rows(accepted: int, drafts: int) -> int:
    """The rows the kernel decided on: every draft row when all were accepted, else the rows up to the rejection."""

    return drafts if accepted == drafts else accepted + 1


def acceptance_probabilities(statistics: Sequence[float], drafts: int) -> list[float]:
    """``p_j(d_{j+1}) = w_j / S_j`` for the evaluated rows (the ledger's histogram input; the kernel divides nothing)."""

    accepted = int(statistics[mtp_accept.STAT_ACCEPTED])
    out = []
    for j in range(evaluated_rows(accepted, drafts)):
        w, s = float(statistics[mtp_accept.STAT_WEIGHT + j]), float(statistics[mtp_accept.STAT_TOTAL + j])
        if w < 0 or s <= 0:
            raise ValueError(f"row {j} was evaluated but its statistics lanes hold w={w} S={s}")
        out.append(w / s)
    return out


def guard_deviations(statistics: Sequence[float]) -> int:
    """The rows whose kept minimum did not clear the shard floor (the guard mask's set bits)."""

    return bin(int(statistics[mtp_accept.STAT_GUARD])).count("1")


def pass_uniforms(uniforms: Sequence[float], drafts: int, index: int) -> list[float]:
    """Pass ``index``'s ``k + 1`` uniforms from the request's ledger of draws."""

    rows = drafts + 1
    chunk = list(uniforms[index * rows : (index + 1) * rows])
    if len(chunk) != rows:
        raise ValueError(f"pass {index} needs {rows} uniforms, the ledger holds {len(uniforms)}")
    return chunk


@dataclass(frozen=True)
class PassCheck:
    index: int
    device: tuple[int, int]  # (a*, x*) the statistics row carries
    reference: tuple[int, int]  # accept_reference on the recorded rows, drafts and uniforms
    guard_deviations: int

    @property
    def ok(self) -> bool:
        return self.device == self.reference


def check_pass(
    record: Mapping[str, Any], policy: Qwen38DeviceSamplerPolicy, *, drafts: int, uniforms: Sequence[float], table=None
) -> PassCheck:
    """One recorded pass (``candidate_rows`` ``[(k+1) * 256]`` floats, ``tokens`` ``[t, d_1 .. d_k, ...]``,
    ``statistics`` 16 floats) against the host mirror."""

    rows = torch.tensor(record["candidate_rows"], dtype=torch.float32).reshape(drafts + 1, mtp_accept.ROW_LANES)
    draft_ids = [int(t) for t in record["tokens"][1 : drafts + 1]]
    statistics = record["statistics"]
    reference = mtp_accept.accept_reference(
        rows, draft_ids, policy, list(uniforms), table=table, sentinel=ZERO_EMBEDDING_TOKEN
    )
    device = (int(statistics[mtp_accept.STAT_ACCEPTED]), int(statistics[mtp_accept.STAT_TOKEN]))
    return PassCheck(
        int(record.get("index", 0)), device, (reference.accepted, reference.token), guard_deviations(statistics)
    )


def check_records(
    records: Sequence[Mapping[str, Any]], policy: Qwen38DeviceSamplerPolicy, *, drafts: int, uniforms: Sequence[float]
) -> dict[str, Any]:
    """Every recorded pass re-derived: the mismatch count is the wired-loop gate (0 expected)."""

    table = weight_table(policy.temperature)
    checks = [
        check_pass(record, policy, drafts=drafts, uniforms=pass_uniforms(uniforms, drafts, i), table=table)
        for i, record in enumerate(records)
    ]
    mismatches = [c for c in checks if not c.ok]
    return {
        "passes": len(checks),
        "mismatches": len(mismatches),
        "first_mismatch": None
        if not mismatches
        else {"index": mismatches[0].index, "device": mismatches[0].device, "reference": mismatches[0].reference},
        "guard_deviations": sum(c.guard_deviations for c in checks),
        "arithmetic": ARITHMETIC_DEVICE,
    }


__all__ = [
    "ADMITTED_DRAFTS",
    "ARITHMETIC_DEVICE",
    "ARITHMETIC_HOST",
    "SWITCH",
    "PassCheck",
    "Qwen38DeviceAcceptance",
    "acceptance_probabilities",
    "check_pass",
    "check_records",
    "device_accept_switch",
    "evaluated_rows",
    "for_chain",
    "guard_deviations",
    "pass_uniforms",
    "refusal",
]
