# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
try:
    from .nemotron_generator import NemotronHForCausalLM
except ModuleNotFoundError:
    # ttnn is not available in sim-only environments (e.g. tt-lang venv)
    pass
