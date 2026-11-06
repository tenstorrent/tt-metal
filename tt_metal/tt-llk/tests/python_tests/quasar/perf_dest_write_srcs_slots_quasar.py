import pytest
from quasar.test_dest_write_srcs_slots_quasar import (
    CAUSE_CASES,
    CAUSE_IDS,
    ISOLATE_CASES,
    ISOLATE_IDS,
    MACRO_CASES,
    MACRO_IDS,
    run_slots,
)


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize("tiles", [3, 4])
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_perf_srcs_slots(perf_report, slots, tiles):
    run_slots(slots, tiles, False, perf=True, perf_report=perf_report)


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize("slots,per_instr,tiles", CAUSE_CASES, ids=CAUSE_IDS)
def test_perf_srcs_slots_stream_once(perf_report, slots, per_instr, tiles):
    run_slots(
        slots,
        tiles,
        False,
        perf=True,
        perf_report=perf_report,
        per_instr=per_instr,
        stream_once=True,
    )


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize("sfpu_only,tiles", ISOLATE_CASES, ids=ISOLATE_IDS)
def test_perf_srcs_isolate_sfpu(perf_report, sfpu_only, tiles):
    run_slots(
        1,
        tiles,
        False,
        perf=True,
        perf_report=perf_report,
        stream_once=not sfpu_only,
        sfpu_only=sfpu_only,
        writer_only=True,
    )


# One-face pack, so the bank stamps time the Dest writer.
@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize("sfpu_only,copy_mode", MACRO_CASES, ids=MACRO_IDS)
def test_perf_srcs_loadmacro(perf_report, sfpu_only, copy_mode):
    run_slots(
        1,
        4,
        False,
        perf=True,
        perf_report=perf_report,
        stream_once=not sfpu_only,
        sfpu_only=sfpu_only,
        copy_mode=copy_mode,
        writer_only=True,
    )
