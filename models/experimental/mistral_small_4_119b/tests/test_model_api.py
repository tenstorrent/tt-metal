# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.mistral_small_4_119b.tt import model as model_mod


def test_get_vision_stack_class():
    cls = model_mod.get_vision_stack_class()
    assert cls.__name__ == "TtMistralVisionTransformer"


def test_full_model_reason_documents_gap():
    msg = model_mod.create_full_model_unsupported_reason()
    assert "Mistral4" in msg or "MoE" in msg
