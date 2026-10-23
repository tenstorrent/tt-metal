# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch reference implementations (the PCC oracle, #47468).

Env-independent: no ttnn / checkpoint / hardware. The vendored HF model wrapper
will live here too once ``transformers`` ships ``diffusion_gemma``.
"""
