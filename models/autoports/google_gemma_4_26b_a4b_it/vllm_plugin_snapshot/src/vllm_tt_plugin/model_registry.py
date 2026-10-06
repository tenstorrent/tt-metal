# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_tt_plugin.platform import (
    _should_pre_register_tt_test_models_from_cli,
    register_tt_models,
    register_tt_test_models,
)

__all__ = [
    "register_tt_models",
    "register_tt_models_from_plugin",
    "register_tt_test_models",
]


def register_tt_models_from_plugin() -> None:
    """Entry point used by ``vllm.general_plugins``."""
    register_tt_models(
        register_test_models=_should_pre_register_tt_test_models_from_cli()
    )
