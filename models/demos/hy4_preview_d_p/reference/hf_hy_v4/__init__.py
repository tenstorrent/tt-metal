# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""transformers 5.17.0 ``models/hy_v4`` (modeling_hy_v4.py, configuration_hy_v4.py), vendored because the shared
python_env has transformers 5.12.1, which lacks it. The only change is the imports: ``from ...x`` (package-relative in
transformers) became ``from transformers.x``; 5.12.1 has every name they import (DynamicIndexedLayer included).

One semantic fix, marked "bring-up fix" in modeling_hy_v4.py: RoPE. The 5.17 port applies rotate-half ("Non-interleave
RoPE") in the MLA and the indexer; Hy4 is served with interleaved (GPT-J pair) RoPE in both (SGLang
``srt/configs/hy_v4.py``: ``rope_interleave = True``, ``indexer_rope_interleave = True``, used by
``DeepseekV2AttentionMLA`` and the DSA indexer as ``is_neox_style=False``). With the port as released, the full model's
top-1 next-token accuracy on the canonical prompt was 0.237 (the smoke still answered "Paris"). Everything else is the
5.17.0 wheel's code (/localdev/dnijemcevic/bringup/hy4_preview_d_p/hf_code); main (2026-09-29) differs only in the KV
cache layout."""
