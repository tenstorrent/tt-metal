"""The vision tower's terms in the MTP chain's DRAM admission (no device): the resident tower comes off the free side
where the reading predates its load, the largest image's transient activation is a required-side part with the growth
margin, and the default record is unchanged."""

from __future__ import annotations

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module

RESIDENT = 125_231_104  # the tower's BF16 weights per bank, measured on the line 2026-09-29
PEAK = 150_994_944  # 65,536 patches (the stock maximum image) x 18,432 bytes per padded patch over 8 banks


def _with_margin(remainder: int) -> int:
    return -(-remainder * (100 + session_module.MTP_GROWTH_ESTIMATE_MARGIN_PERCENT) // 100)


@pytest.fixture(autouse=True)
def _stub_pair(monkeypatch):
    monkeypatch.setattr(session_module, "packed_bf4_bytes_per_device", lambda *, ring_size: (80 << 20, 40 << 20))


def test_default_record_carries_zero_vision_terms_and_the_three_estimate_parts() -> None:
    record = session_module.mtp_capacity_admission(32768, drafts=4)
    assert record["vision_resident_bytes_per_bank"] == record["vision_peak_activation_bytes_per_bank"] == 0
    assert set(record["mtp_growth_estimate_bytes_per_bank"]) == {"components", "states", "traces"}
    assert "vision" not in record["decided_by"]["required_side"]


def test_resident_tower_comes_off_the_free_side_before_the_captures() -> None:
    for context in (32768, 65536):
        plain = session_module.mtp_capacity_admission(context, drafts=4, long_chunks=True)
        vision = session_module.mtp_capacity_admission(
            context, drafts=4, long_chunks=True, vision_resident_bytes_per_bank=RESIDENT
        )
        for key in ("free_bytes_per_bank_after_captures", "largest_contiguous_bytes_free_per_bank_after_captures"):
            assert plain[key] - vision[key] == RESIDENT, (context, key)
        assert vision["required_free_bytes_per_bank"] == plain["required_free_bytes_per_bank"]
        assert vision["headroom_bytes_per_bank"] == plain["headroom_bytes_per_bank"] - RESIDENT
        assert vision["decided_by"]["required_side"].endswith(" vision")
    view = {"free_bytes_per_bank": 1_700_000_000, "largest_contiguous_bytes_free_per_bank": 1_690_000_000}
    after_build = session_module.mtp_capacity_admission(
        32768, drafts=4, live=view, live_point="after_build", vision_resident_bytes_per_bank=RESIDENT
    )
    plain_after_build = session_module.mtp_capacity_admission(32768, drafts=4, live=view, live_point="after_build")
    assert (
        plain_after_build["free_bytes_per_bank_after_captures"] - after_build["free_bytes_per_bank_after_captures"]
        == RESIDENT
    )
    assert (
        after_build["resident_post_build_bytes_per_bank"] - plain_after_build["resident_post_build_bytes_per_bank"]
        == RESIDENT
    )
    # an after-captures read already holds the tower: nothing comes off
    after_captures = session_module.mtp_capacity_admission(
        32768, drafts=4, live=view, live_point="after_captures", vision_resident_bytes_per_bank=RESIDENT
    )
    plain_after_captures = session_module.mtp_capacity_admission(
        32768, drafts=4, live=view, live_point="after_captures"
    )
    for key in ("free_bytes_per_bank_after_captures", "largest_contiguous_bytes_free_per_bank_after_captures"):
        assert after_captures[key] == plain_after_captures[key] == view[key.replace("_after_captures", "")], key


def test_peak_activation_is_a_required_part_with_the_margin() -> None:
    plain = session_module.mtp_capacity_admission(32768, drafts=4)
    vision = session_module.mtp_capacity_admission(32768, drafts=4, vision_peak_activation_bytes_per_bank=PEAK)
    assert vision["mtp_growth_estimate_bytes_per_bank"]["vision_activation"] == _with_margin(PEAK)
    assert vision["required_free_bytes_per_bank"] - plain["required_free_bytes_per_bank"] == _with_margin(PEAK)
    assert vision["free_bytes_per_bank_after_captures"] == plain["free_bytes_per_bank_after_captures"]
    assert vision["fits"] == (
        vision["free_bytes_per_bank_after_captures"] >= vision["required_free_bytes_per_bank"]
        and vision["largest_contiguous_bytes_free_per_bank_after_captures"]
        >= vision["required_largest_contiguous_bytes_per_bank"]
    )


def test_a_tower_beside_the_mtp_chain_fits_at_32k_on_the_table_and_refuses_when_the_room_is_gone() -> None:
    both = session_module.mtp_capacity_admission(
        32768,
        drafts=4,
        long_chunks=True,
        vision_resident_bytes_per_bank=RESIDENT,
        vision_peak_activation_bytes_per_bank=PEAK,
    )
    assert both["fits"], both["decided_by"]
    tight = {"free_bytes_per_bank": RESIDENT + 1_000_000, "largest_contiguous_bytes_free_per_bank": 1_000_000}
    refused = session_module.mtp_capacity_admission(
        32768, drafts=4, live=tight, live_point="after_captures", vision_peak_activation_bytes_per_bank=PEAK
    )
    assert not refused["fits"] and "free_bytes_below_estimate" in refused["decided_by"]["shortfalls"]


@pytest.mark.parametrize("value", [True, -1, 1.5, "125"])
@pytest.mark.parametrize("name", ["vision_resident_bytes_per_bank", "vision_peak_activation_bytes_per_bank"])
def test_vision_terms_take_non_negative_ints(expect_error, name, value) -> None:
    with expect_error(ValueError, match=name):
        session_module.mtp_capacity_admission(32768, drafts=4, **{name: value})
