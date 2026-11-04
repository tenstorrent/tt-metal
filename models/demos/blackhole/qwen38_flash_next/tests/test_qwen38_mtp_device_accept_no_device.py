# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The MTP pass's device acceptance without a device: the switch, the admission table, the per-request protocol on
fake constants (the policy at request start, ``k + 1`` uniforms from the request's stream before every verify
launch), the statistics row's parsing and the ledger's counters, the pass record's fields, the gate's re-derivation
of a recorded pass and of the session's dump, the extension's pass-loop contract, the chain-open construction and the
fingerprint token."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.sampling import sampling_parameters
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_mtp_device_accept as da
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling as step
from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler as ds
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept as ma
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38SamplingParameters, UniformStream

K = 4


class FakeConstants:
    def __init__(self, rows: int):
        self.rows = rows
        self.policies: list = []
        self.uniforms: list[list[float]] = []
        self.released = False
        self.corruptible = False

    def write_policy(self, policy):
        self.policies.append(policy)
        return {"table": True, "scalars": True}

    def write_uniforms(self, uniforms):
        if not 1 <= len(uniforms) <= self.rows:
            raise ValueError(f"{len(uniforms)} draws for {self.rows} rows")
        self.uniforms.append(list(uniforms))

    def release(self):
        self.released = True

    def mark_corruptible(self):
        self.corruptible = True


def _request(profile=Qwen38SamplingParameters.official_thinking, **fields):
    return step.Qwen38SamplingRequest(profile(seed=11), **fields)


def _rows(seed: int) -> torch.Tensor:
    """``k + 1`` candidate rows of bf16 values and distinct ids, per shard ``[values | ids]``."""

    g = torch.Generator().manual_seed(seed)
    rows = torch.zeros(K + 1, ma.ROW_LANES)
    for r in range(K + 1):
        packs = rows[r].reshape(4, 2, 32)
        packs[:, 0] = (torch.randn(128, generator=g) * 3).to(torch.bfloat16).float().reshape(4, 32)
        packs[:, 1] = torch.randperm(248320, generator=g)[:128].float().reshape(4, 32)
    return rows


# --- the switch, the admission, the request protocol ------------------------------------------------------------------


def test_switch_parsing(expect_error):
    assert da.device_accept_switch({}) is False
    assert da.device_accept_switch({da.SWITCH: "0"}) is False
    assert da.device_accept_switch({da.SWITCH: "1"}) is True
    with expect_error(ValueError, match=da.SWITCH):
        da.device_accept_switch({da.SWITCH: "yes"})
    assert da.for_chain(None, None, drafts=4, environ={}) is None
    assert da.for_chain(None, None, drafts=None, environ={da.SWITCH: "1"}) is None


def test_admission_table():
    assert da.refusal(None, 4) == "refused: greedy request"
    assert da.refusal(_request(top_logprobs=3), 4) == "refused: logprobs"
    assert da.refusal(_request(Qwen38SamplingParameters.official_non_thinking), 4) == "refused: penalties"  # 1.5
    assert da.refusal(_request(), 4) is None
    assert da.refusal(_request(), 5).startswith("refused: k 5 above 4")
    hot = step.Qwen38SamplingRequest(sampling_parameters(temperature=5.0, seed=1))
    assert da.refusal(hot, 4).startswith("refused: temperature 5.0 above")
    assert da.ADMITTED_DRAFTS == 4


def test_request_protocol_on_fake_constants(expect_error):
    constants = FakeConstants(rows=5)
    acceptance = da.Qwen38DeviceAcceptance(4, constants)
    request = _request()
    request.uniforms.extend([0.5, 0.25])  # a previous request's ledger
    acceptance.begin_request(request)
    assert constants.policies == [ds.Qwen38DeviceSamplerPolicy.from_parameters(request.parameters)]
    assert request.uniforms == [] and request.mtp_arithmetic == da.ARITHMETIC_DEVICE
    hook = acceptance.before_verify(request)
    hook([7, 1, 2, 3, 4])
    hook([8, 5, 6, 7, 8])
    expected = UniformStream(request.parameters.seed)
    stream = [expected.next_uniform() for _ in range(10)]
    assert constants.uniforms == [stream[:5], stream[5:]]  # u_0 .. u_3, v per pass, in the stream's order
    assert request.uniforms == stream
    with expect_error(ValueError, match="verify tokens"):
        hook([1, 2, 3])
    with expect_error(ValueError, match="admitted request only"):
        acceptance.begin_request(_request(top_logprobs=1))
    acceptance.mark_corruptible()
    acceptance.release()
    assert constants.corruptible and constants.released
    with expect_error(ValueError, match="1..4 drafts"):
        da.Qwen38DeviceAcceptance(5, FakeConstants(6))


# --- the statistics row, the pass record, the gate --------------------------------------------------------------------


def test_accept_statistics_parse_the_program_row_and_count_the_guard_deviations(expect_error) -> None:
    alignment = (7, 42, *[ZERO_EMBEDDING_TOKEN] * 30)
    reference = ma.AcceptReference(1, 42, alignment, (5, 0), (9, 11), 0b101, True, 0.25, 3)
    row = reference.statistics_row()
    parsed = mtp_v2.parse_accept_statistics(row)
    assert parsed.row == tuple(row.tolist()) and len(parsed.row) == ma.STATS_LANES
    assert (parsed.weights, parsed.totals) == ((5, 0), (9, 11))
    assert (parsed.guard_mask, parsed.guard_deviations, parsed.resampled) == (0b101, 2, True)
    assert (parsed.accepted, parsed.token, parsed.theta, parsed.kept) == (1, 42, 0.25, 3)
    all_accepted = ma.AcceptReference(
        K, 9, (1, 2, 3, 4, 9, *[ZERO_EMBEDDING_TOKEN] * 27), (3, 3, 3, 3), (4, 4, 4, 4), 0, False, 0.5, 2
    )
    parsed = mtp_v2.parse_accept_statistics(all_accepted.statistics_row())
    assert parsed.weights == (3, 3, 3, 3) and parsed.totals == (4, 4, 4, 4) and not parsed.resampled
    assert parsed.guard_deviations == 0 and parsed.accepted == K and parsed.token == 9
    with expect_error(RuntimeError, match="lanes"):
        mtp_v2.parse_accept_statistics(torch.zeros(15))
    assert mtp_v2.DEVICE_ACCEPT_ARITHMETIC == da.ARITHMETIC_DEVICE == "device-theta"


def test_pass_record_device_fields_default_to_none() -> None:
    record = mtp_v2.Qwen38TTNNMTPPassRecord(0, 0, (1, 2, 3, 4, 5), 0, (), (), 0, 0, False, {})
    defaults = (record.decision, record.statistics, record.arithmetic, record.guard_deviations, record.candidate_rows)
    assert defaults == (None,) * 5
    record = mtp_v2.Qwen38TTNNMTPPassRecord(
        0, 0, (1, 2, 3, 4, 5), 2, (), (), 0, 0, False, {}, statistics=tuple(range(16)), arithmetic="device-theta"
    )
    assert record.arithmetic == mtp_v2.DEVICE_ACCEPT_ARITHMETIC and len(record.statistics) == 16


def test_gate_rederives_a_recorded_pass_and_the_counters_read_the_statistics_row(expect_error):
    policy = ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
    table = ds.weight_table(policy.temperature)
    rows = _rows(3)
    kept = [ma.row_kept_set(*ds.candidate_row_lanes(rows[r]), policy, table) for r in range(K + 1)]
    drafts = [kept[0].ids[0], kept[1].ids[0], 7, kept[3].ids[0]]  # row 2 rejects: id 7 is in no row
    uniforms = [n * 2.0**-24 for n in (0, 0, 0, 0, 5000000)]
    reference = ma.accept_reference(rows, drafts, policy, uniforms, table=table, sentinel=ZERO_EMBEDDING_TOKEN)
    assert reference.accepted == 2 and reference.resampled
    record = {
        "index": 0,
        "candidate_rows": rows.reshape(-1).tolist(),
        "tokens": [1234, *drafts, ZERO_EMBEDDING_TOKEN],
        "statistics": reference.statistics_row().tolist(),
    }
    summary = da.check_records([record], policy, drafts=K, uniforms=uniforms)
    assert summary == {
        "passes": 1,
        "mismatches": 0,
        "first_mismatch": None,
        "guard_deviations": 0,
        "arithmetic": "device-theta",
    }
    corrupted = dict(record, statistics=list(record["statistics"]))
    corrupted["statistics"][ma.STAT_TOKEN] = float(drafts[0])
    bad = da.check_records([corrupted], policy, drafts=K, uniforms=uniforms)
    assert bad["mismatches"] == 1 and bad["first_mismatch"]["index"] == 0
    with expect_error(ValueError, match="needs 5 uniforms"):
        da.pass_uniforms(uniforms[:4], K, 0)
    stats = step.Qwen38SampledDraftingStats()
    stats.record_device(record["statistics"], K)
    counters = (stats.passes, stats.accepted_drafts, stats.draws, stats.resampled, stats.guard_deviations)
    assert counters == (1, 2, 5, 1, 0)
    assert stats.acceptance_probabilities == [w / s for w, s in zip(reference.weights, reference.totals)]
    assert len(stats.acceptance_probabilities) == da.evaluated_rows(2, K) == 3
    assert stats.as_dict()["guard_deviations"] == 0
    accepted_all = ma.accept_reference(
        rows, [kept[j].ids[0] for j in range(K)], policy, [0.0] * 5, table=table, sentinel=ZERO_EMBEDDING_TOKEN
    )
    assert da.evaluated_rows(accepted_all.accepted, K) == K == len(accepted_all.weights)


def test_dump_record_shape_re_derives_under_the_gate(tmp_path: Path) -> None:
    """The records the session writes under the dump switch carry what ``check_records`` re-derives from (the rows the
    device decided on, the pass's tokens and its statistics) plus the pass's own uniforms; a consistent record (the
    statistics the host reference computes for those rows, drafts and uniforms) passes the gate with 0 mismatches."""

    parameters = sampling_parameters(top_p=0.95, seed=5)
    policy = ds.Qwen38DeviceSamplerPolicy.from_parameters(parameters)
    assert policy is not None
    uniforms: list[float] = []
    records: list[dict] = []
    for index in range(3):
        rows = _rows(100 + index)
        drafts = [int(v) for v in rows[:K, 32 : 32 + K].reshape(-1)[:K].tolist()]  # ids from the rows' first shard
        pass_uniforms = [(n + 7 * index) * 2**-24 for n in (1_000_003, 5_000_011, 9_000_017, 13_000_019, 16_000_023)]
        uniforms.extend(pass_uniforms)
        reference = ma.accept_reference(rows, drafts, policy, pass_uniforms, sentinel=ZERO_EMBEDDING_TOKEN)
        records.append(
            {
                "index": index,
                "candidate_rows": [float(v) for v in rows.reshape(-1).tolist()],
                "tokens": [11, *drafts],
                "statistics": [float(v) for v in reference.statistics_row().tolist()],
                "uniforms": pass_uniforms,
                "committed_tokens": 5 + 3 * index,
            }
        )
    chat = object.__new__(session.Qwen38ChatSession)
    chat.mtp = SimpleNamespace(drafts=K)
    chat.device_accept_dump = tmp_path
    chat.device_accept_records = records
    chat.requests_served = 3
    path = chat._write_device_accept_dump(SimpleNamespace(parameters=parameters, uniforms=uniforms))
    run = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "device-accept-00003.json" and set(run) == {"drafts", "parameters", "uniforms", "records"}
    assert run["drafts"] == K and run["parameters"] == {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
    assert len(run["records"]) == 3 and set(run["records"][0]) == set(records[0])
    assert len(run["records"][0]["candidate_rows"]) == (K + 1) * 256 and len(run["records"][0]["statistics"]) == 16
    summary = da.check_records(run["records"], policy, drafts=K, uniforms=run["uniforms"])
    assert summary["passes"] == 3 and summary["mismatches"] == 0 and summary["arithmetic"] == "device-theta"


# --- the extension, the chain open ------------------------------------------------------------------------------------


def test_extension_pass_loop_contract_without_a_device():
    ext = object.__new__(step.Qwen38SamplingChainExtension)
    ext.sampler = None
    ext.device_accept = None
    assert ext.device_acceptance_for(_request()) == (f"refused: {da.SWITCH} off", None)
    ext.device_accept = da.Qwen38DeviceAcceptance(4, FakeConstants(5))
    why, hook = ext.device_acceptance_for(_request(top_logprobs=2))
    assert why == "refused: logprobs" and hook is None
    request = _request()
    ext.begin_request(request)  # no device sampler on an --mtp server: the acceptance's request start still runs
    assert request.mtp_arithmetic == da.ARITHMETIC_DEVICE and ext.device_accept.constants.policies
    why, hook = ext.device_acceptance_for(request)
    assert why is None and callable(hook)
    hook([1, 2, 3, 4, 5])
    assert len(ext.device_accept.constants.uniforms[0]) == 5
    # release() and mark_corruptible() cover the acceptance constants (the extension owns them)
    assert "device_accept.release()" in inspect.getsource(step.Qwen38SamplingChainExtension.release)
    assert "device_accept.mark_corruptible()" in inspect.getsource(step.Qwen38SamplingChainExtension.mark_corruptible)
    assert "mtp_acceptance_arithmetic" in request.as_dict()
    refused = _request(top_logprobs=2)
    ext.begin_request(refused)
    assert refused.mtp_arithmetic is None  # the host decides; accept_pass names host-fp32 at its first pass


def test_chain_open_builds_the_acceptance_through_the_sampling_step_and_the_fingerprint_names_it():
    # the construction line in qwen38_chat_session calls sampling_step.device_acceptance_for_chain: the name must
    # survive the import-pruning hooks (it is used by another module only)
    assert step.device_acceptance_for_chain is da.for_chain
    assert step.Qwen38DeviceAcceptance is da.Qwen38DeviceAcceptance
    assert step.DEVICE_ACCEPT_SWITCH == da.SWITCH
    assert {"device_acceptance_for_chain", "Qwen38DeviceAcceptance", "DEVICE_ACCEPT_SWITCH"} <= set(step.__all__)
    opened = (
        inspect.getsource(session.Qwen38ChatSession.open)
        if hasattr(session.Qwen38ChatSession, "open")
        else inspect.getsource(session)
    )
    assert "device_accept=sampling_step.device_acceptance_for_chain(mesh, lm_head.mesh_contract, drafts=mtp)" in opened
    assert '"-device-accept" if mtp_sampled and device_accept_switch() else ""' in inspect.getsource(server)
    assert "request.mtp_arithmetic = ARITHMETIC_HOST" in inspect.getsource(step.accept_pass)
