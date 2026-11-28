# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin the model identity for this demo's e2e gate at import time.

Setting HF_MODEL here (unconditionally) fixes the target to
Llama-3.1-8B-Instruct the instant collection begins, which is what resolves
the optimize tool's "HF_MODEL not fixed" rejection: the identity is bound at
invocation, not left to a caller or default.

No device work happens here — this only sets environment.
"""
import os

os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
os.environ.setdefault("MESH_DEVICE", "P150")
