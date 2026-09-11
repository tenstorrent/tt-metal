# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

raise ImportError(
    "TTTv2 left tt-metal. Import from https://github.com/tenstorrent/tt_transformers "
    "(models.common.models -> tt_transformers.models). "
    "Keepers: models.common.{lazy_weight, tt_ccl, moe}."
)
