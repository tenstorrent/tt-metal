# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Kimi-K3 prefill stack.

K3 is a hybrid: 24 of its 93 layers are full-attention MLA layers, the other 69 are KDA linear
attention, and the residual stream between them is AttnRes rather than a running sum. Those three
differences do not fit behind a flag on `TtPrefillBlock` — that class builds `ttMLA` unconditionally
before its own `kv_only` early return, reads `self.mla.*` throughout `forward`, and returns the
residual the AttnRes walk has to own. So K3 lives here, as its own block / transformer / runtime,
and reuses the leaf modules unchanged: `ttMLA`, `TtMoe`, `TtFfn`, `TtDistributedRmsNorm`,
`TtParallelEmbedding`, `TtLMHead`, `RotarySetup`, `tt_ccl`. `models/demos/gpt_oss_d_p/tt/` and
`models/demos/minimax_m3/tt/` are the same arrangement.
"""
