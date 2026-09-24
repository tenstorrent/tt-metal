<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Golden-KV scripts

Placeholder. `generate_golden_kv_cache.py` and `verify_golden_kv.py` land with
[tt-blaze#4147](https://github.com/tenstorrent/tt-blaze/issues/4147) (reference model + golden KV),
mirroring `models/demos/gpt_oss_d_p/scripts/`.

The golden must round-trip weights through the **device dtype** (bf8_b cache) before PCC — a
full-precision golden leaves a spurious ~0.94–0.96 gap that reads as a real bug. And its RoPE frame
must match what blaze decode writes (Meta-interleaved), not what prefill happens to do.
