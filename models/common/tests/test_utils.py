# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.common.tests.utils import stable_model_seed


def test_stable_model_seed_deterministic() -> None:
    assert stable_model_seed("llama-3") == stable_model_seed("llama-3")


def test_stable_model_seed_distinct() -> None:
    assert stable_model_seed("llama-3") != stable_model_seed("mistral-7b")


def test_stable_model_seed_uint32_range() -> None:
    seed = stable_model_seed("llama-3")
    assert 0 <= seed < 2**32
