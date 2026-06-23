# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
try:
    from .nemotron_generator import NemotronHForCausalLM
except ModuleNotFoundError as _e:
    if "ttnn" not in str(_e) and "tt_lib" not in str(_e):
        raise
