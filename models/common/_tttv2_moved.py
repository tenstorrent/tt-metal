# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

TTTV2_MOVED_MSG = (
    "TTTv2 left tt-metal. Install and import the tt_transformers package: "
    "https://github.com/tenstorrent/tt_transformers\n"
    "  models.common.modules      -> tt_transformers.modules\n"
    "  models.common.llm_runtime  -> tt_transformers.llm_runtime\n"
    "  models.common.models       -> tt_transformers.models\n"
    "Keepers that used to live under models.common.modules:\n"
    "  LazyWeight -> models.common.lazy_weight\n"
    "  tt_ccl     -> models.common.tt_ccl\n"
    "  MoE        -> models.common.moe\n"
    "models.common.sampling is unchanged (not TTTv2)."
)
