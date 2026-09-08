# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Ordering test for the mid-kernel FP32 dest-acc handshake (``_llk_set_fp32_dest_acc_``), Blackhole.

The invariant: when MATH releases UNPACK/PACK, the three dest-acc config writes it just issued must
already be visible to them.

Section 1 exercises the real function end to end. Section 2 tests the ordering mechanism on a
replica of MATH's sequence, because the invariant cannot be stressed through the real function --
its own preamble leaves MATH's Tensix pipe empty, which hands the config writes a head start that
nothing outside the function can close.

The lever in section 2 is MOP issue occupancy placed *in front of* the config writes, so they land
late while the release path is untouched. Both directions and a position control are measured, since
each guards against a different way of reading a green arm that means nothing:

* the shipped replica must FAIL under occupancy -- otherwise the lever is inert and the fixed arm's
  zeros prove nothing;
* the drain placed BEFORE the release must hold at zero;
* the same drain placed AFTER the release must FAIL -- otherwise "the drain fixes it" is just "any
  added delay fixes it";
* RISC work between the writes and the release is the protective direction and must never create
  the race.
"""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

# Must match sources/fp32_dest_acc_cfg_occupancy_test.cpp
TRIALS = 4096


@pytest.mark.skipif(
    get_chip_architecture()
    not in (ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE),
    reason="_llk_set_fp32_dest_acc_ exists on Blackhole and Wormhole only.",
)
def test_fp32_dest_acc_cfg_occupancy():
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/fp32_dest_acc_cfg_occupancy_test.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )
    vals = [int(v) for v in configuration.run().result]

    trials, desync, plumbing, num_occ = vals[0], vals[1], vals[2], vals[3]
    sweep = [
        (vals[4 + 4 * i], vals[5 + 4 * i], vals[6 + 4 * i], vals[7 + 4 * i])
        for i in range(num_occ)
    ]
    k = 4 + 4 * num_occ
    real_trials, real_enabled, real_disabled = vals[k], vals[k + 1], vals[k + 2]
    num_dir, dir_occ = vals[k + 3], vals[k + 4]
    k += 5
    direction = [
        (vals[k + 3 * i], vals[k + 3 * i + 1], vals[k + 3 * i + 2])
        for i in range(num_dir)
    ]

    assert (
        desync == 0
    ), f"mailbox rendezvous desynced {desync} times; the result describes nothing"
    assert trials == TRIALS, f"kernel reported {trials} trials, expected {TRIALS}"

    # ---- Section 1: the real function ---------------------------------------------------------
    print(
        f"\n_llk_set_fp32_dest_acc_ over {real_trials} enable/disable cycles:"
        f"\n  field set after enable    : {real_enabled}/{real_trials}"
        f"\n  field clear after disable : {real_disabled}/{real_trials}"
    )
    assert real_enabled == real_trials and real_disabled == real_trials, (
        f"the real handshake did not publish dest-acc to PACK: enabled seen "
        f"{real_enabled}/{real_trials}, disabled seen {real_disabled}/{real_trials}"
    )

    # ---- Section 2: the ordering mechanism ----------------------------------------------------
    # Gated first: without a detector that can fire, every zero below is unfalsifiable.
    assert plumbing == TRIALS, (
        f"READ PATH IS WRONG: this arm never writes the field, so PACK must read it stale on all "
        f"{TRIALS} trials, but did so {plumbing} times. Fix the config read or mask before "
        f"interpreting anything else."
    )

    print(
        f"\nreplica of MATH's sequence, {TRIALS} trials per point:"
        f"\n\n  {'MOP slots':>9}  {'shipped':>12}  {'drain BEFORE':>13}  {'drain AFTER':>12}"
    )
    for depth, shipped, before, after in sweep:
        print(
            f"  {depth:>9}  {shipped:>6}/{TRIALS}  {before:>6}/{TRIALS}       {after:>6}/{TRIALS}"
        )

    # The lever must actually hurt the shipped replica, or nothing below is evidence.
    fired = [(d, s) for d, s, _, _ in sweep if s > 0]
    assert fired, (
        f"LEVER INERT: occupancy never made the shipped replica fail, up to depth {sweep[-1][0]}. "
        f"The zeros in the drain arms therefore prove nothing. Do not read this as the ordering "
        f"being correct -- widen the lever or report that it cannot reach it."
    )

    # The fix.
    leaks = [(d, b) for d, _, b, _ in sweep if b > 0]
    assert (
        not leaks
    ), "the drain before the release did NOT establish the ordering: " + ", ".join(
        f"depth {d} -> {b}/{TRIALS}" for d, b in leaks
    )

    # Position control: the same drain on the wrong side must not help.
    weak = [(d, a) for d, s, _, a in sweep if s > 0 and a == 0]
    assert not weak, (
        "POSITION CONTROL FAILED: the same drain placed AFTER the release also came back clean at "
        + ", ".join(f"depth {d}" for d, _ in weak)
        + ". Then the fixed arm's zero is added delay, not ordering, and this comparison is void."
    )
    print(
        f"  -> shipped fails from depth {fired[0][0]}; the drain before the release holds at 0 "
        f"everywhere;\n     the same drain after the release fails alike, so it is the ordering "
        f"that matters, not the delay."
    )

    # ---- Direction ----------------------------------------------------------------------------
    print(
        "\nRISC work between the config writes and the release (the protective direction):"
        f"\n\n  {'RISC nops':>9}  {'no occupancy':>14}  {f'{dir_occ} MOP slots':>14}"
    )
    for nops, clean, occupied in direction:
        print(f"  {nops:>9}  {clean:>7}/{TRIALS}  {occupied:>7}/{TRIALS}")

    created = [(n, c) for n, c, _ in direction if c > 0]
    assert not created, (
        "RISC delay between the config writes and the release CREATED the race at "
        + ", ".join(f"{n} nops -> {c}" for n, c in created)
        + ". That contradicts the direction this hazard runs in; explain it before relying on any "
        "of the above."
    )
    rescued = [n for n, _, o in direction if o == 0]
    note = (
        f"and rescues the occupied case from {min(rescued)} nops on"
        if rescued
        else "and does not rescue the occupied case at any level measured"
    )
    print(f"  -> never creates the race at any nop count, {note}.")
