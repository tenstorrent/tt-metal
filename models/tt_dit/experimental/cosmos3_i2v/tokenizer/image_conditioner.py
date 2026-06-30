# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Image preprocessing notes.

Phase 1 (tt-symbiote MVP): the `Cosmos3OmniDiffusersPipeline` handles
reference-image preprocessing itself (resize, crop, normalization) before
feeding the vision encoder. No custom code needed — pass a `PIL.Image` as
the `image=` kwarg to the pipeline.

Phase 2 (native tt-nn): implement the host-side preprocessing + a TTNN
patchifier here if we want to bypass diffusers and run native end-to-end.
"""

from __future__ import annotations
