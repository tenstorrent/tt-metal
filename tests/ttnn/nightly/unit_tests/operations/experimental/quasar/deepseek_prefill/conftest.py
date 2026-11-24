# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Each test module here star-imports one test module of the original ttnn.experimental.deepseek_prefill ops and reruns it
# against the ttnn.experimental.quasar.deepseek_prefill copies. Only copies that keep the originals' Python signatures are
# swapped in: the moe_fused_swiglu and unified_routed_expert_moe copies no longer accept the originals' arguments.

import pytest
import ttnn

from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.conftest import (  # noqa: F401
    pytest_collection_modifyitems,
    pytest_configure,
)

QUASAR_DEEPSEEK_PREFILL_COPIES = (
    "attn_res_block_mix",
    "attn_res_weighted_reduce_nc",
    "csa_compressor",
    "extract",
    "insert",
    "masked_bincount",
    "mhc_split_sinkhorn",
    "moe_grouped_topk",
    "moe_hash_gate",
    "offset_cumsum",
    "post_combine_reduce",
    "rotary_embedding_indexed",
)


# Module scope so module-scoped fixtures of the original tests also see the copies, and the originals are restored
# before any other module runs in the same session.
@pytest.fixture(scope="module", autouse=True)
def quasar_deepseek_prefill_copies():
    with pytest.MonkeyPatch.context() as mp:
        for name in QUASAR_DEEPSEEK_PREFILL_COPIES:
            mp.setattr(
                ttnn.experimental.deepseek_prefill, name, getattr(ttnn.experimental.quasar.deepseek_prefill, name)
            )
        yield
