# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Config class for the in-tree Kimi-K2.7 reference config.

``config.json``'s ``auto_map`` resolves ``AutoConfig`` against this module, and transformers'
trust_remote_code loader imports it from a copy in its own module cache — so it has to be a real
module inside the K2.7 dir, not a path in the JSON. K2.7 is architecturally identical to K2.6, so
the class is re-exported instead of vendored twice. It is the in-tree DeepseekV3Config rather than
transformers' native one because native ``DeepseekV3Config`` no longer carries ``rope_theta`` as a
top-level field, which MLA's YaRN path reads.
"""

from models.demos.deepseek_v3_d_p.reference.kimi_k2_6.configuration_deepseek import DeepseekV3Config

__all__ = ["DeepseekV3Config"]
