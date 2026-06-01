# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Coder-Next demo package for TT-NN (P150a)."""

from .tt.model_config import Qwen3CoderNextConfig
from .tt.deltanet import TtGatedDeltaNet
from .tt.attention import TtGatedAttention
from .tt.decoder import TtHybridDecoderLayer
from .tt.moe import TtMoE
from .tt.generator import Qwen3CoderNextGenerator
from .tt.model import TtQwen3CoderNextModel
