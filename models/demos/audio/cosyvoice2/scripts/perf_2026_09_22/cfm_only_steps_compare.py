"""Fast, isolated CFM-only measurement at steps=10 vs steps=5, at the REAL T values this
round's four sentences actually produce (T = 2 * (prompt_token_len + generated_token_len),
matching the encoder's upsample-by-2): T=510 (utt0), T=608 (utt1), T=660 (utt2), T=860
(utt3) -- taken directly from the 2026-09-22 warm-regression run's real token counts.

Random (not real-checkpoint) decoder weights -- kernel compile cost depends on tensor
SHAPE, not weight VALUES, and this script only measures time, so this is valid and lets
it skip checkpoint loading entirely (the full-pipeline version of this measurement timed
out at 20 minutes, dominated by an unrelated slow HiFT eager cold-cache conv-verification
pass at a length no earlier script had used -- this script sidesteps that by not touching
HiFT or the LLM at all).

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 900 /opt/venv/bin/python cfm_only_steps_compare.py
"""
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/home/user/tt-metal")

from models.demos.audio.cosyvoice2.tt.flow.decoder import (
    CausalConditionalCFMRef,
    CausalConditionalDecoderRef,
    TtCausalConditionalCFM,
    TtCausalConditionalDecoder,
)

device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=100_000_000)


def sync():
    ttnn.synchronize_device(device)


torch.manual_seed(0)
dec = CausalConditionalDecoderRef()
dec.eval()
cfm_ref = CausalConditionalCFMRef(dec)
tt_dec = TtCausalConditionalDecoder(device, dec)
tt_cfm = TtCausalConditionalCFM(device, tt_dec, cfm_ref.rand_noise, cfm_ref)

# (utt, T) -- T = 2 * (prompt_token_len + generated_token_len), from the real
# 2026-09-22 warm-regression run's real token counts for these four sentences.
UTTS = [(0, 510), (1, 608), (2, 660), (3, 860)]

try:
    results = {}
    for ui, t_len in UTTS:
        mu = torch.randn(1, t_len, 80) * 0.1
        cond = torch.randn(1, t_len, 80) * 0.1
        spks = torch.randn(1, 80) * 0.1
        mask = torch.ones(1, t_len, 1)

        def timed_solve(n_steps, n_warm=3):
            times = []
            for _ in range(1 + n_warm):
                sync()
                t0 = time.perf_counter()
                tt_cfm.forward(mu, mask, n_steps, spks, cond, use_trace=True)
                sync()
                times.append(time.perf_counter() - t0)
            tt_cfm.release_cfm_trace()
            return times  # [0]=cold(capture), [1:]=warm(replay)

        t10 = timed_solve(10)
        t5 = timed_solve(5)
        warm10 = sum(t10[1:]) / len(t10[1:])
        warm5 = sum(t5[1:]) / len(t5[1:])
        results[ui] = (t_len, t10, t5, warm10, warm5)
        print(
            f"[utt {ui}] T={t_len}  "
            f"STEPS=10: cold {t10[0]*1000:8.1f} ms  warm {[f'{x*1000:.1f}' for x in t10[1:]]} ms  mean {warm10*1000:.1f} ms  |  "
            f"STEPS=5: cold {t5[0]*1000:8.1f} ms  warm {[f'{x*1000:.1f}' for x in t5[1:]]} ms  mean {warm5*1000:.1f} ms  |  "
            f"speedup {warm10/warm5:.2f}x"
        )
        sys.stdout.flush()

    print("\n=== SUMMARY: real measured CFM-only warm time, steps=10 vs steps=5 ===")
    for ui, (t_len, t10, t5, warm10, warm5) in results.items():
        print(f"utt {ui} (T={t_len}): CFM@10 {warm10*1000:.1f} ms  CFM@5 {warm5*1000:.1f} ms  delta {(warm10-warm5)*1000:.1f} ms")
finally:
    tt_cfm.release_cfm_trace()
    ttnn.CloseDevice(device)
