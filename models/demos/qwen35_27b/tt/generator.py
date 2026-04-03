# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B Generator.

Thin wrapper around the framework Generator for text-only inference.
"""

from models.tt_transformers.tt.generator import Generator as TTTGenerator


class Generator(TTTGenerator):
    """Qwen3.5-27B text generator.

    Wraps the framework Generator with a simpler constructor.
    For text-only models, the framework Generator handles everything:
    prefill, decode, tracing, device sampling.
    """

    def __init__(self, model, model_args, mesh_device, tokenizer=None):
        super().__init__(
            model=[model],
            model_args=[model_args],
            mesh_device=mesh_device,
            tokenizer=tokenizer,
        )
