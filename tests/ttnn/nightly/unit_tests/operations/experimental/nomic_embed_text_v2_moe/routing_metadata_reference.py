# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Re-export shim. Implementation lives in models.experimental.nomic_embed_text_v2_moe.routing_metadata_reference."""

import sys

from models.experimental.nomic_embed_text_v2_moe import routing_metadata_reference as _impl

sys.modules[__name__] = _impl
