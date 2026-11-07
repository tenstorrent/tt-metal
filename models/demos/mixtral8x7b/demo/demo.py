# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Mixtral runs on the shared TTT text demo; this module collects it under the Mixtral folder.
from models.ttt_compat.demo.simple_text_demo import test_demo_text

__all__ = ["test_demo_text"]
