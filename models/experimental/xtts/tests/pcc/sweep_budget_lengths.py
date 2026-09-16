# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch sweep: run inference_fully_traced with min == max == N over a range of N and report clashes."""
import os

import pytest
import ttnn

from models.experimental.xtts.config import L1_SMALL_SIZE, SESSION_TRACE_REGION
from models.experimental.xtts.tests.pcc.test_empty_generation import SAMPLING, _inputs, _max_seq


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": SESSION_TRACE_REGION}], indirect=True
)
def test_sweep_budget_lengths(device, xtts_state_dict, reset_seeds):
    lo, hi = (int(x) for x in os.environ["XTTS_SWEEP_RANGE"].split("-"))
    tt, wav, spk_tt, padded, real_len, pad_to = _inputs(device, xtts_state_dict)
    failures = []
    for n in range(lo, hi + 1):
        try:
            wav_dev, codes, perf = tt.inference_fully_traced(
                padded,
                wav,
                spk_tt,
                _max_seq(pad_to, n),
                max_new_tokens=n,
                text_real_len=real_len,
                **dict(SAMPLING, min_new_tokens=n),
            )
            print(f"SWEEP n={n} codes={codes.shape[1]} samples={wav_dev.shape[1]} PASS", flush=True)
            if wav_dev.is_allocated():
                ttnn.deallocate(wav_dev)
        except Exception as e:  # noqa: BLE001
            text = str(e).strip()
            first = text.splitlines()[0][:220] if text else type(e).__name__
            print(f"SWEEP n={n} FAIL {first}", flush=True)
            failures.append(n)
            voc = tt.decoder.decoder
            for fn in (voc.generator.release_conditioning, voc.upsampler.release_cache):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass
    print(f"SWEEP DONE range={lo}-{hi} failures={failures}", flush=True)
    assert not failures, f"L1 clash at code budgets {failures}"
