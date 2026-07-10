# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""WavLM-large speaker-verification (the VibeVoice-report SIM model), torch-only."""

from .wavlm_standalone import embed, init_model

__all__ = ["init_model", "embed"]
