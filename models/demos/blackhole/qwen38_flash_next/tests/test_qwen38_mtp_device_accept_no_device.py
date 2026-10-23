"""The MTP pass's device acceptance without a device: the switch, the admission table, the per-request protocol on
fake constants (the policy at request start, ``k + 1`` uniforms from the request's stream before every verify
launch), the ledger's counters from the statistics lanes, the gate's re-derivation of a recorded pass, the
extension's pass-loop contract, the chain-open construction and the fingerprint token."""

from __future__ import annotations

import inspect

import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_mtp_device_accept as da
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step
from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler as ds
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept as ma
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38SamplingParameters, UniformStream


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


def test_switch_parsing(expect_error):
    assert da.device_accept_switch({}) is False
    assert da.device_accept_switch({da.SWITCH: "0"}) is False
    assert da.device_accept_switch({da.SWITCH: "1"}) is True
    with expect_error(ValueError, match=da.SWITCH):  # allow-pytest.raises: reads the exception
        da.device_accept_switch({da.SWITCH: "yes"})
    assert da.for_chain(None, None, drafts=4, environ={}) is None
    assert da.for_chain(None, None, drafts=None, environ={da.SWITCH: "1"}) is None


def test_admission_table():
    assert da.refusal(None, 4) == "refused: greedy request"
    assert da.refusal(_request(top_logprobs=3), 4) == "refused: logprobs"
    assert (
        da.refusal(_request(Qwen38SamplingParameters.official_non_thinking), 4) == "refused: penalties"
    )  # presence 1.5
    assert da.refusal(_request(), 4) is None
    assert da.refusal(_request(), 5).startswith("refused: k 5 above 4")
    hot = step.Qwen38SamplingRequest(
        Qwen38SamplingParameters(temperature=5.0, top_p=1.0, top_k=20, presence_penalty=0.0, seed=1)
    )
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
    with expect_error(ValueError, match="verify tokens"):  # allow-pytest.raises: reads the exception
        hook([1, 2, 3])
    with expect_error(ValueError, match="admitted request only"):  # allow-pytest.raises: reads the exception
        acceptance.begin_request(_request(top_logprobs=1))
    acceptance.mark_corruptible()
    acceptance.release()
    assert constants.corruptible and constants.released
    with expect_error(ValueError, match="1..4 drafts"):  # allow-pytest.raises: reads the exception
        da.Qwen38DeviceAcceptance(5, FakeConstants(6))


def _rows(k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rows = torch.zeros(k + 1, ma.ROW_LANES)
    for r in range(k + 1):
        packs = rows[r].reshape(4, 2, 32)
        packs[:, 0] = (torch.randn(128, generator=g) * 3).to(torch.bfloat16).float().reshape(4, 32)
        packs[:, 1] = torch.randperm(248320, generator=g)[:128].float().reshape(4, 32)
    return rows


def test_gate_rederives_a_recorded_pass_and_the_counters_read_the_statistics_row(expect_error):
    k = 4
    policy = ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
    table = ds.weight_table(policy.temperature)
    rows = _rows(k, 3)
    kept = [ma.row_kept_set(*ma_lanes(rows, r), policy, table) for r in range(k + 1)]
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
    summary = da.check_records([record], policy, drafts=k, uniforms=uniforms)
    assert summary == {
        "passes": 1,
        "mismatches": 0,
        "first_mismatch": None,
        "guard_deviations": 0,
        "arithmetic": "device-theta",
    }
    corrupted = dict(record, statistics=list(record["statistics"]))
    corrupted["statistics"][ma.STAT_TOKEN] = float(drafts[0])
    bad = da.check_records([corrupted], policy, drafts=k, uniforms=uniforms)
    assert bad["mismatches"] == 1 and bad["first_mismatch"]["index"] == 0
    with expect_error(ValueError, match="needs 5 uniforms"):  # allow-pytest.raises: reads the exception
        da.pass_uniforms(uniforms[:4], k, 0)
    stats = step.Qwen38SampledDraftingStats()
    stats.record_device(record["statistics"], k)
    assert (stats.passes, stats.accepted_drafts, stats.draws, stats.resampled, stats.guard_deviations) == (
        1,
        2,
        5,
        1,
        0,
    )
    assert stats.acceptance_probabilities == [w / s for w, s in zip(reference.weights, reference.totals)]
    assert len(stats.acceptance_probabilities) == da.evaluated_rows(2, k) == 3
    assert stats.as_dict()["guard_deviations"] == 0
    accepted_all = ma.accept_reference(
        rows, [kept[j].ids[0] for j in range(k)], policy, [0.0] * 5, table=table, sentinel=ZERO_EMBEDDING_TOKEN
    )
    assert da.evaluated_rows(accepted_all.accepted, k) == k == len(accepted_all.weights)


def ma_lanes(rows: torch.Tensor, r: int):
    packs = rows[r].reshape(4, 2, 32)
    return packs[:, 0].reshape(-1).to(torch.float32), packs[:, 1].reshape(-1).to(torch.int64)


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


def test_the_sampling_step_module_re_exports_the_acceptance_for_the_chain_open():
    # the construction line in qwen38_chat_session calls sampling_step.device_acceptance_for_chain: the name must
    # survive the import-pruning hooks (it is used by another module only)
    assert step.device_acceptance_for_chain is da.for_chain
    assert step.Qwen38DeviceAcceptance is da.Qwen38DeviceAcceptance
    assert step.DEVICE_ACCEPT_SWITCH == da.SWITCH
    assert {"device_acceptance_for_chain", "Qwen38DeviceAcceptance", "DEVICE_ACCEPT_SWITCH"} <= set(step.__all__)


def test_chain_open_builds_the_acceptance_and_the_fingerprint_names_it():
    opened = (
        inspect.getsource(session.Qwen38ChatSession.open)
        if hasattr(session.Qwen38ChatSession, "open")
        else inspect.getsource(session)
    )
    assert "device_accept=sampling_step.device_acceptance_for_chain(mesh, lm_head.mesh_contract, drafts=mtp)" in opened
    served = inspect.getsource(server)
    assert '"-device-accept" if mtp_sampled and device_accept_switch() else ""' in served
    body = inspect.getsource(step.accept_pass)
    assert "request.mtp_arithmetic = ARITHMETIC_HOST" in body
