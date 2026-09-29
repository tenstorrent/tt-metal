# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-request drafting chains (``QWEN38_MTP_DRAFTS_PER_REQUEST=1``, ``extra_body.mtp_drafts``) without a device: the
server's switch and request field, the shared MTP layer generic state's invariants in mtp_v2 (identity-pinned sharing,
the sharing chain's release leaves the shared state to its owner, released exactly once), and the session's open /
request / release order pinned by source."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2

ROOT = Path(__file__).resolve().parents[1]
SESSION_SOURCE = ROOT / "tools" / "qwen38_chat_session.py"
SERVER_SOURCE = ROOT / "tools" / "qwen38_chat_server.py"
MTP_V2_SOURCE = ROOT / "ttnn" / "mtp_v2.py"


# -- the server: the switch and the request field -------------------------------------------------------------------


def test_switch_admits_the_pair_with_the_default_first_and_refuses_other_values(expect_error) -> None:
    assert server.MTP_DRAFTS_PER_REQUEST_VARIABLE == "QWEN38_MTP_DRAFTS_PER_REQUEST"
    assert server.MTP_DRAFTS_PER_REQUEST_PAIR == (4, 5)
    assert server.mtp_drafts_per_request_switch({}, drafts=4) == ()
    assert server.mtp_drafts_per_request_switch({"QWEN38_MTP_DRAFTS_PER_REQUEST": "0"}, drafts=4) == ()
    assert server.mtp_drafts_per_request_switch({"QWEN38_MTP_DRAFTS_PER_REQUEST": "1"}, drafts=4) == (4, 5)
    assert server.mtp_drafts_per_request_switch({"QWEN38_MTP_DRAFTS_PER_REQUEST": "1"}, drafts=5) == (5, 4)
    with expect_error(SystemExit, match="must be 0 or 1"):
        server.mtp_drafts_per_request_switch({"QWEN38_MTP_DRAFTS_PER_REQUEST": "yes"}, drafts=4)
    with expect_error(SystemExit, match="needs --mtp in"):
        server.mtp_drafts_per_request_switch({"QWEN38_MTP_DRAFTS_PER_REQUEST": "1"}, drafts=3)
    with expect_error(SystemExit, match="needs --mtp in"):
        server.mtp_drafts_per_request_switch({"QWEN38_MTP_DRAFTS_PER_REQUEST": "1"}, drafts=None)


def _document(**fields):
    return {"model": server.MODEL_ID, "messages": [{"role": "user", "content": "hi"}], **fields}


def test_request_field_is_refused_on_a_one_chain_server_and_admitted_against_the_captured_list() -> None:
    assert "mtp_drafts" in server.KNOWN_REQUEST_FIELDS
    assert server.parse_chat_request(_document())["mtp_drafts"] is None
    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        server.Qwen38ChatRequestRejected, match="not admitted by this server"
    ) as info:  # allow-pytest.raises: inspect the captured exception object
        server.parse_chat_request(_document(mtp_drafts=5))
    assert info.value.param == "mtp_drafts"
    admitted = (4, 5)
    assert server.parse_chat_request(_document(mtp_drafts=5), mtp_drafts_admitted=admitted)["mtp_drafts"] == 5
    assert server.parse_chat_request(_document(mtp_drafts=4), mtp_drafts_admitted=admitted)["mtp_drafts"] == 4
    # extra_body is the OpenAI client's carrier: merged under the top level like the other extension fields
    parsed = server.parse_chat_request(_document(extra_body={"mtp_drafts": 5}), mtp_drafts_admitted=admitted)
    assert parsed["mtp_drafts"] == 5
    for bad in (6, 3, True, "5", 4.0):
        with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
            server.Qwen38ChatRequestRejected, match=r"must be one of \[4, 5\]"
        ) as info:  # allow-pytest.raises: inspect the captured exception object
            server.parse_chat_request(_document(mtp_drafts=bad), mtp_drafts_admitted=admitted)
        assert info.value.param == "mtp_drafts"
    assert server.parse_chat_request(_document(mtp_drafts=None), mtp_drafts_admitted=admitted)["mtp_drafts"] is None


# -- mtp_v2: the shared MTP layer generic state ----------------------------------------------------------------------


def _verify_state(owner: object, alignment) -> mtp_v2.Qwen38TTNNVerifyState:
    """A verify state whose fields the sharing checks read (the rest are placeholders the checks never reach)."""

    return mtp_v2.Qwen38TTNNVerifyState(
        drafts=4,
        rows=5,
        rows_constants=None,
        qsa_chunk_constants=None,
        qsa_verify_constants=None,
        accept_constants=None,
        layers=(),
        token_row=None,
        draft_lanes=None,
        ple_rows=None,
        accepted=None,
        alignment=alignment,
        moe_rows=5,
        gdn_step_anchor_layers=frozenset(),
        split=None,
        _owner=owner,
    )


def _alignment(layer, generic_state, *, owns: bool = True) -> mtp_v2.Qwen38TTNNVerifyAlignment:
    return mtp_v2.Qwen38TTNNVerifyAlignment(
        layer,
        "input_mixer",
        "final_mixer",
        generic_state,
        SimpleNamespace(name="window"),
        "residual",
        owns_generic_state=owns,
    )


def test_alignment_owns_its_generic_state_unless_told_otherwise() -> None:
    assert mtp_v2.Qwen38TTNNVerifyAlignment.__dataclass_fields__["owns_generic_state"].default is True
    assert _alignment("layer", "history").owns_generic_state is True
    assert _alignment("layer", "history", owns=False).owns_generic_state is False


def test_sharing_validation_pins_the_owner_object_by_identity_and_the_ownership_flags(expect_error) -> None:
    owner_token = object()
    model = SimpleNamespace(_state_owner=owner_token)
    history = SimpleNamespace(name="committed history")
    owner = _verify_state(owner_token, _alignment("layer", history))
    sharing = _verify_state(owner_token, _alignment("layer", history, owns=False))
    validate = mtp_v2._validate_verify_state
    # the sharing checks run before the per-layer checks the placeholders cannot pass: every refusal below is theirs
    with expect_error(ValueError, match="two distinct verify states"):
        validate(model, owner, shared_with=owner)
    with expect_error(ValueError, match="must not own the MTP layer generic state; the owner must"):
        validate(
            model, owner, shared_with=_verify_state(owner_token, _alignment("layer", history))
        )  # an owner posing as the sharer
    with expect_error(ValueError, match="must not own the MTP layer generic state; the owner must"):
        validate(model, sharing, shared_with=_verify_state(owner_token, _alignment("layer", history, owns=False)))
    with expect_error(ValueError, match="is not the owner's object"):
        validate(
            model,
            _verify_state(owner_token, _alignment("layer", SimpleNamespace(name="a copy"), owns=False)),
            shared_with=owner,
        )
    with expect_error(ValueError, match="verify state sharing needs two distinct verify states with MTP alignment"):
        validate(model, _verify_state(owner_token, None), shared_with=owner)
    with expect_error(ValueError, match="not allocated by this model owner"):
        validate(model, _verify_state(object(), sharing.alignment), shared_with=owner)


def test_sharing_chain_release_leaves_the_generic_state_to_its_owner_released_once(monkeypatch) -> None:
    events: list[str] = []
    layer = SimpleNamespace(release_generic_state=lambda state: events.append(f"release generic {state.name}"))
    history = SimpleNamespace(name="history")
    owner_token = object()
    model = SimpleNamespace(_state_owner=owner_token, layers=())
    monkeypatch.setattr(mtp_v2, "_validate_verify_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(mtp_v2, "_deallocate", lambda *tensors: events.append(f"deallocate {tensors}"))
    monkeypatch.setattr(
        mtp_v2, "_release_layer_verify_state", lambda layer, state: events.append(f"release window {state.name}")
    )
    constants = SimpleNamespace(deallocate=lambda: None)

    def state(alignment):
        return mtp_v2.Qwen38TTNNVerifyState(
            drafts=4,
            rows=5,
            rows_constants=constants,
            qsa_chunk_constants=constants,
            qsa_verify_constants=constants,
            accept_constants=constants,
            layers=(),
            token_row="token_row",
            draft_lanes="draft_lanes",
            ple_rows=SimpleNamespace(release=lambda: None),
            accepted="accepted",
            alignment=alignment,
            moe_rows=5,
            gdn_step_anchor_layers=frozenset(),
            split=None,
            _owner=owner_token,
        )

    sharing = state(_alignment(layer, history, owns=False))
    owner = state(_alignment(layer, history))
    # the session's order: the sharing chain first, the owner last
    mtp_v2.release_verify_state(model, sharing)
    assert "release window window" in events and not [event for event in events if event.startswith("release generic")]
    mtp_v2.release_verify_state(model, owner)
    assert events.count("release generic history") == 1
    assert events.count("release window window") == 2


# -- the source pins: allocation, capture, request binding and release order ----------------------------------------


def _functions(source: Path) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, node)
    return found


def _segment(source: Path, node: ast.AST) -> str:
    return " ".join(ast.get_source_segment(source.read_text(encoding="utf-8"), node).split()).replace("( ", "(")


def test_allocate_verify_state_takes_the_shared_generic_state_as_its_last_keyword() -> None:
    functions = _functions(MTP_V2_SOURCE)
    allocate = functions["allocate_verify_state"]
    assert allocate.args.kwonlyargs[-1].arg == "alignment_generic_state"
    text = _segment(MTP_V2_SOURCE, allocate)
    assert "the committed history is one; the window is the chain's" in text
    assert (
        "generic_state = alignment_generic_state" in text
        and "owns_generic_state=alignment_generic_state is None" in text
    )
    release = _segment(MTP_V2_SOURCE, functions["release_verify_state"])
    assert "if alignment.owns_generic_state:" in release
    sharing = _segment(MTP_V2_SOURCE, functions["validate_verify_state_sharing"])
    assert "_validate_verify_state(model, owner)" in sharing and "shared_with=owner" in sharing


def test_open_allocates_the_alternates_before_the_snapshot_captures_every_chain_and_binds_per_request() -> None:
    functions = _functions(SESSION_SOURCE)
    opened = _segment(SESSION_SOURCE, functions["open"])
    assert "mtp_alternates: Sequence[int] = ()" in opened
    states = opened.index("alignment_generic_state=chain_mtp.alignment.generic_state")
    sharing_check = opened.index("mtp_v2.validate_verify_state_sharing(model, alternate_verify, chain_mtp.verify)")
    snapshot = opened.index("snapshot = model.allocate_generic_snapshot(")
    captures = opened.index("dram_after_previous = capture_mtp_chain(chain_mtp, dram_after_prefill_captures)")
    assert states < sharing_check < snapshot < captures  # states before any capture; the default chain captured first
    assert "components_shared=True" in opened and "chunk_extension=chain_mtp.chunk_extension" in opened
    assert "step_inputs=chain_mtp.step_inputs" in opened  # the decode tail trace bakes the default chain's step inputs
    assert "long_chunk_extension=chain_mtp.long_chunk_extension" in opened  # the 128-row twin is shared too
    assert "mtp_alternates and the device acceptance do not combine" in opened  # one k's accept constants
    assert "long_chunks=False, # the 128-row twin is the default chain's (shared)" in opened  # counted once
    assert "does not combine with `QWEN38_MTP_DEVICE_ACCEPT=1`" in (ROOT / "docs" / "SERVER.md").read_text(
        encoding="utf-8"
    )
    assert "MTP_WARM_ACCEPT_UNIFORMS[: target.drafts + 1]" in _segment(SESSION_SOURCE, functions["warm_mtp_chain"])
    warm = _segment(SESSION_SOURCE, functions["warm_mtp_chain"])
    assert (
        "[MTP_BOOTSTRAP_DRAFT_TOKEN] * target.drafts" in warm and "while warm_fed % RESIDUE_CLASSES != residue:" in warm
    )
    assert (
        "for target in [chain_mtp] + [" in opened and "if count != chain_mtp.drafts ]: warm_mtp_chain(target)" in opened
    )

    assert opened.index("warm_mtp_chain(target)") < opened.index(
        "dram_after_previous = capture_mtp_chain(chain_mtp, dram_after_prefill_captures)"
    )
    assert "for count in sorted(mtp_chains): if count != chain_mtp.drafts:" in opened
    helper = _segment(SESSION_SOURCE, functions["capture_mtp_chain"])
    for call in (
        "mtp_v2.capture_verify(",
        "mtp_v2.capture_commit(",
        "mtp_v2.capture_draft(",
        "mtp_v2.capture_verify_head(",
    ):
        assert call in helper and "target.verify" in helper
    complete = _segment(SESSION_SOURCE, functions["complete"])
    assert "mtp_drafts: int | None = None" in complete
    assert "self._bind_mtp(self.mtp_chains[mtp_drafts])" in complete
    assert complete.count("self._bind_mtp(self.mtp_default)") == 2  # the failure path and the return path
    bind = _segment(SESSION_SOURCE, functions["_bind_mtp"])
    assert "self.chain.mtp = chain_mtp" in bind and "while a pass loop is open" in bind
    close = _segment(SESSION_SOURCE, functions["close"])
    sharing_release = close.index("if chain_mtp is not self.mtp:")
    owner_release = close.index("mtp_v2.release_verify_state(model, self.mtp.verify)")
    assert sharing_release < owner_release  # the sharing chains first, the owner last
    assert close.count("step_inputs.deallocate()") == 1  # the shared step inputs: released once, by the owner
    construct = _segment(SESSION_SOURCE, functions["construct_chain"])
    assert "mtp_alternates: Sequence[int] = ()" in construct and "mtp_alternates=mtp_alternates" in construct
    admission = _segment(SESSION_SOURCE, functions["mtp_capacity_admission"])
    assert "components_shared: bool = False" in admission and '"components_shared": components_shared' in admission
    served = SERVER_SOURCE.read_text(encoding="utf-8")
    assert "mtp_alternates=mtp_drafts_admitted[1:]," in served
    assert 'mtp_drafts=request["mtp_drafts"],' in served and '"drafts_admitted": list(mtp_drafts_admitted),' in served
    assert '"mtp_chains": {' in served
    assert '"mtp_drafts_admitted": list(mtp_drafts_admitted),' in served  # READY names the admitted chains


# -- a failed capture never leaves the device with a capture open ---------------------------------------------------


def test_a_raising_capture_body_ends_and_releases_the_trace_before_the_error_propagates(
    expect_error, monkeypatch
) -> None:
    events: list[str] = []
    fake = SimpleNamespace(
        begin_trace_capture=lambda mesh, cq_id: events.append(f"begin cq{cq_id}") or 41,
        end_trace_capture=lambda mesh, trace_id, cq_id: events.append(f"end {trace_id} cq{cq_id}"),
        release_trace=lambda mesh, trace_id: events.append(f"release {trace_id}"),
    )
    monkeypatch.setattr(mtp_v2, "ttnn", fake)

    def failing_body():
        events.append("body")
        raise RuntimeError("program cache miss occurred, but cache misses are forbidden")

    with expect_error(RuntimeError, match="cache misses are forbidden"):
        mtp_v2._capture_trace("mesh", 0, failing_body)
    assert events == ["begin cq0", "body", "end 41 cq0", "release 41"]
    # the happy path: begun, recorded, ended, kept
    events.clear()
    assert mtp_v2._capture_trace("mesh", 1, lambda: events.append("body") or "output") == (41, "output")
    assert events == ["begin cq1", "body", "end 41 cq1"]
    # a release that fails does not mask the body's error
    fake.release_trace = lambda mesh, trace_id: (_ for _ in ()).throw(ValueError("already released"))
    with expect_error(RuntimeError, match="cache misses are forbidden"):
        mtp_v2._capture_trace("mesh", 0, failing_body)


def test_every_capture_site_goes_through_the_abort_safe_helper() -> None:
    text = MTP_V2_SOURCE.read_text(encoding="utf-8")
    assert "trace_id = ttnn.begin_trace_capture(model.mesh_device" not in text  # no bare capture left
    functions = _functions(MTP_V2_SOURCE)
    sites = [name for name in functions if name.startswith("capture_")]
    assert {"capture_verify", "capture_commit", "capture_draft", "capture_verify_head", "capture_verify_tail"} <= set(
        sites
    )
    for name in sites:
        segment = _segment(MTP_V2_SOURCE, functions[name])
        assert "_capture_trace(model.mesh_device, cq_id, _body)" in segment, name
        assert "with guard(" in segment, name
