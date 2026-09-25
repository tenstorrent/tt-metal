# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.demos.gemma4.tt.generator import SDPA_CHUNK_ALIGN, align_num_cached_tokens_to_sdpa


def test_align_num_cached_tokens_to_sdpa_rounds_down():
    assert SDPA_CHUNK_ALIGN == 128
    # Nightly 31B QB2 crash: vLLM continuation at start_pos=48.
    assert align_num_cached_tokens_to_sdpa([48]) == [0]
    assert align_num_cached_tokens_to_sdpa([0, 128, 129, 256, 47]) == [0, 128, 128, 256, 0]
