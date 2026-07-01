# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Language Model prefill-mode PCC vs HuggingFace Qwen2 (last_hidden_state only)."""

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import (
    PCC_THRESHOLD,
    PREFILL_ISL_EXTENDED_SWEEP_LENGTHS,
    PREFILL_ISL_EXTENDED_TARGET,
    PREFILL_ISL_SWEEP_LENGTHS,
    SEQ_LEN,
    build_tt_lm,
    compare_prefill_hidden_pcc,
    prefill_isl_sweep_effective_lengths,
    print_prefill_isl_sweep_timing_table,
    print_prefill_pcc_isl_debug,
    reference_lm_forward,
    run_prefill_isl_sweep_timed,
    tt_prefill_hidden,
)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_prefill_hidden_state_pcc(mesh_device, vv_config, lm_state):
    torch.manual_seed(0)
    cfg = vv_config.decoder

    input_ids = torch.randint(0, cfg.vocab_size, (1, SEQ_LEN), dtype=torch.long)

    ref_prefill = reference_lm_forward(lm_state, input_ids, vv_config)

    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    kv_cache = lm_tt.alloc_kv_cache(SEQ_LEN + 8)
    tt_prefill = tt_prefill_hidden(lm_tt, input_ids, kv_cache)

    passed_p, pcc_p = comp_pcc(ref_prefill.to(torch.float32), tt_prefill, pcc=PCC_THRESHOLD)
    per_pos = [
        comp_pcc(ref_prefill[:, p].to(torch.float32), tt_prefill[:, p], pcc=PCC_THRESHOLD)[1] for p in range(SEQ_LEN)
    ]
    lows = sorted(range(SEQ_LEN), key=lambda i: per_pos[i])[:6]
    print(f"[test_lm_prefill_pcc] prefill PCC={pcc_p:.6f}")
    print(
        f"[test_lm_prefill_pcc] per-pos prefill PCC: last={per_pos[-1]:.5f} min={min(per_pos):.5f} "
        f"median={sorted(per_pos)[SEQ_LEN // 2]:.5f}  lows=" + ",".join(f"p{i}={per_pos[i]:.4f}" for i in lows)
    )
    assert passed_p, f"LM prefill last_hidden PCC {pcc_p:.6f} < {PCC_THRESHOLD}"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_prefill_hidden_state_pcc_isl_sweep(mesh_device, vv_config, lm_state):
    """Prefill PCC vs HuggingFace Qwen2 across multiple input sequence lengths."""
    cfg = vv_config.decoder
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    failures = []
    print("[test_lm_prefill_pcc]")

    for seq_len in PREFILL_ISL_SWEEP_LENGTHS:
        torch.manual_seed(0)
        input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)

        ref_prefill = reference_lm_forward(lm_state, input_ids, vv_config)
        kv_cache = lm_tt.alloc_kv_cache(seq_len + 8)
        tt_prefill = tt_prefill_hidden(lm_tt, input_ids, kv_cache)

        print_prefill_pcc_isl_debug(ref_prefill, tt_prefill, seq_len)
        passed_p, pcc_p, per_pos = compare_prefill_hidden_pcc(ref_prefill, tt_prefill, seq_len)

        print("--------------------------------------------------")
        print(f"Sequence Length : {seq_len}")
        print(f"Overall PCC     : {pcc_p:.5f}")
        print(f"Last Token PCC  : {per_pos[-1]:.5f}")
        print(f"Minimum PCC     : {min(per_pos):.5f}")
        print(f"Median PCC      : {sorted(per_pos)[seq_len // 2]:.5f}")
        print("--------------------------------------------------")

        if not passed_p:
            failures.append(f"sequence_length={seq_len} measured_pcc={pcc_p:.6f} threshold={PCC_THRESHOLD}")

    if failures:
        assert False, "LM prefill ISL sweep PCC below threshold:\n" + "\n".join(failures)

    print("PASS")


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_prefill_hidden_state_pcc_isl_sweep_extended_with_timing(mesh_device, vv_config, lm_state):
    """Extended prefill ISL sweep (32 … 64k) with HF/TT wall-time per length (diagnostic).

    Lengths above ``decoder.max_position_embeddings`` are skipped. KV cache is allocated per
    length as ``seq_len + 8`` (rounded up to 256 inside ``alloc_kv_cache``). Per-token PCC
    and debug output are skipped above 1024 — use ``test_lm_prefill_hidden_state_pcc_isl_sweep``
    for per-token diagnostics on shorter lengths.
    """
    effective, max_pos = prefill_isl_sweep_effective_lengths(vv_config, PREFILL_ISL_EXTENDED_SWEEP_LENGTHS)
    print(
        f"[test_lm_prefill_pcc] extended ISL sweep: lengths={effective} "
        f"(target={PREFILL_ISL_EXTENDED_TARGET}, max_position_embeddings={max_pos})"
    )

    results = run_prefill_isl_sweep_timed(mesh_device, lm_state, vv_config, PREFILL_ISL_EXTENDED_SWEEP_LENGTHS)
    print_prefill_isl_sweep_timing_table(results)

    errors = [r for r in results if r.get("status") == "error"]
    pcc_failures = [r for r in results if r.get("status") == "ok" and not r.get("pcc_pass")]

    if errors:
        print("\n[prefill ISL sweep] errors:")
        for row in errors:
            print(f"  ISL={row['seq_len']}: {row.get('error')}")

    if pcc_failures:
        print("\n[prefill ISL sweep] PCC below threshold:")
        for row in pcc_failures:
            print(f"  ISL={row['seq_len']}: overall_pcc={row['overall_pcc']:.6f}")

    assert not errors, "Extended ISL sweep hit runtime errors:\n" + "\n".join(
        f"ISL={r['seq_len']}: {r.get('error')}" for r in errors
    )
    assert not pcc_failures, "Extended ISL sweep PCC below threshold:\n" + "\n".join(
        f"ISL={r['seq_len']}: overall_pcc={r['overall_pcc']:.6f}" for r in pcc_failures
    )

    print("PASS")


@pytest.mark.timeout(18000)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_lm_prefill_hidden_state_pcc_isl_65536_with_timing(mesh_device, vv_config, lm_state):
    """Single-length prefill at max context (65536): timing + overall PCC (diagnostic).

    HF reference forward alone can take ~2h on CPU; use this instead of the full extended
    sweep when validating 64k in isolation.
    """
    effective, max_pos = prefill_isl_sweep_effective_lengths(vv_config, [PREFILL_ISL_EXTENDED_TARGET])
    if PREFILL_ISL_EXTENDED_TARGET not in effective:
        pytest.skip(f"ISL={PREFILL_ISL_EXTENDED_TARGET} exceeds max_position_embeddings={max_pos}")

    print(f"[test_lm_prefill_pcc] ISL={PREFILL_ISL_EXTENDED_TARGET} only " f"(max_position_embeddings={max_pos})")

    results = run_prefill_isl_sweep_timed(mesh_device, lm_state, vv_config, [PREFILL_ISL_EXTENDED_TARGET])
    print_prefill_isl_sweep_timing_table(results)

    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        assert False, "ISL 65536 sweep hit runtime errors:\n" + "\n".join(
            f"ISL={r['seq_len']}: {r.get('error')}" for r in errors
        )

    row = results[0]
    print(
        f"[test_lm_prefill_pcc] ISL={row['seq_len']} overall_PCC={row['overall_pcc']:.6f} "
        f"pcc_pass={'yes' if row['pcc_pass'] else 'no'} hf={row['hf_sec']:.1f}s tt={row['tt_sec']:.1f}s"
    )
    print("DONE")
