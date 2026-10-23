"""The verify MoE row-count switch without a device: ``QWEN38_MTP_MOE_ROWS`` parsing, the override through the
admission (its states term keyed by the forced row count) and the chain open, the server's plumbing and health field."""

from __future__ import annotations

import inspect

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2


def test_switch_parsing(expect_error):
    assert mtp_v2.moe_rows_override({}) is None
    assert mtp_v2.moe_rows_override({mtp_v2.MOE_ROWS_SWITCH: ""}) is None
    for value in mtp_v2.MOE_ROWS_SWITCH_VALUES:
        assert mtp_v2.moe_rows_override({mtp_v2.MOE_ROWS_SWITCH: str(value)}) == value
    assert mtp_v2.MOE_ROWS_SWITCH_VALUES == (5, 6, 32)
    with expect_error(ValueError, match="must be one of"):  # allow-pytest.raises: reads the exception
        mtp_v2.moe_rows_override({mtp_v2.MOE_ROWS_SWITCH: "7"})
    with expect_error(ValueError, match="must be one of"):  # allow-pytest.raises: reads the exception
        mtp_v2.moe_rows_override({mtp_v2.MOE_ROWS_SWITCH: "six"})


def test_the_override_reaches_the_row_count_and_the_admission_states_term(expect_error):
    assert mtp_v2.ROWS6_HARDWARE_PROVEN is True
    assert mtp_v2.moe_rows_for(6) == 6  # k = 5 runs the 6-row form since its silicon proof (2026-09-26)
    assert mtp_v2.moe_rows_for(7) == 32 and mtp_v2.moe_rows_for(5) == 5
    assert mtp_v2.resolve_moe_rows(6, 6) == 6 and mtp_v2.resolve_moe_rows(6, None) == 6
    assert mtp_v2.resolve_moe_rows(6, 32) == 32  # the chunk form stays reachable through the switch
    with expect_error(ValueError, match="override"):  # allow-pytest.raises: reads the exception
        mtp_v2.resolve_moe_rows(6, 5)  # fewer MoE rows than the verify tile holds
    table = session.MTP_STATES_BEYOND_QSA_STATE_BYTES_PER_BANK_BY_MOE_ROWS
    assert set(table) == {5, 6, 32} and table[5] < table[6] < table[32]
    forced = session.mtp_capacity_admission(32768, drafts=5, verify_forms=1, moe_rows=32)
    default = session.mtp_capacity_admission(32768, drafts=5, verify_forms=1)
    assert forced["mtp_moe_rows"] == 32 and default["mtp_moe_rows"] == 6
    # the 6-row states term is measured (the 5-row term plus the 6-row open's growth over it); nothing is provisional
    assert default["mtp_states_estimate_provisional"] is False and forced["mtp_states_estimate_provisional"] is False
    assert session.MTP_STATES_ESTIMATE_PROVISIONAL_MOE_ROWS == frozenset()
    assert table[6] == table[5] + 744_064
    assert "[states estimate PROVISIONAL]" in inspect.getsource(
        session.Qwen38TracedChain.open
    )  # the growth gate names it
    remainders = "mtp_growth_remainders_bytes_per_bank"
    assert forced[remainders]["states_beyond_qsa_state"] == table[32]
    assert default[remainders]["states_beyond_qsa_state"] == table[6]
    assert default["required_free_bytes_per_bank"] < forced["required_free_bytes_per_bank"]  # 6 rows need less than 32
    with expect_error(ValueError, match="no MTP states estimate"):  # allow-pytest.raises: reads the exception
        session.mtp_capacity_admission(32768, drafts=5, verify_forms=1, moe_rows=7)


def test_open_and_server_plumb_the_forced_rows():
    opened = inspect.getsource(session.Qwen38TracedChain.open)
    assert "mtp_moe_rows: int | None = None" in opened
    assert "moe_rows=mtp_moe_rows,  # QWEN38_MTP_MOE_ROWS" in opened  # allocate_verify_state
    assert "moe_rows=mtp_moe_rows,\n            )" in opened  # the live admission
    constructed = inspect.getsource(session.construct_chain)
    assert "mtp_moe_rows=mtp_moe_rows" in constructed
    served = inspect.getsource(server)
    assert "mtp_moe_rows = mtp_v2.moe_rows_override()" in served
    assert "moe_rows=mtp_moe_rows" in served and "mtp_moe_rows=mtp_moe_rows" in served
    assert '"moe_rows": mtp_moe_rows' in served  # /health mtp
    assert "needs --mtp" in served
