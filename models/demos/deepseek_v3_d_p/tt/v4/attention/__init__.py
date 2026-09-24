# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash attention kinds for the pure-ttnn prefill: ``TtSWA`` (sliding window, layers 0-1), ``TtHCA``
(heavily compressed, odd layers; lives in ``tt/mla/heavily_compressed_attention.py``), ``TtCSA`` (compressed sparse,
even layers; plan M5). All three share TtHCA's stems, window carry, sinks, un-rope and grouped o-projection."""
