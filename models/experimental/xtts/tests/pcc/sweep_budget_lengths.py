# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch sweep: run inference_fully_traced with min == max == N over a range of N and report clashes.

The program cache is cleared between lengths: every new vocoder length adds conv halo config
tensors in L1_SMALL (64 KB) that the cache keeps alive, and the third length in one process
runs out ("Out of Memory: Not enough space to allocate 1760 B L1_SMALL buffer").
"""
import os
import time

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
        t0 = time.perf_counter()
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
            print(
                f"SWEEP n={n} codes={codes.shape[1]} samples={wav_dev.shape[1]} PASS {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
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
        for k, v in getattr(tt.gpt, "_static_kv", []) or []:
            for t in (k, v):
                if t.is_allocated():
                    ttnn.deallocate(t)
        pos = getattr(tt.gpt, "_text_pos_full", None)
        if pos is not None and pos.is_allocated():
            ttnn.deallocate(pos)
        ttnn.synchronize_device(device)
        device.disable_and_clear_program_cache()
        device.enable_program_cache()
    print(f"SWEEP DONE range={lo}-{hi} failures={failures}", flush=True)
    assert not failures, f"L1 clash at code budgets {failures}"
