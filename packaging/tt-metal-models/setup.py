# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time metadata for the `tt-metal-models` distribution.

Everything static lives in pyproject.toml. This file exists only to supply the two
pieces of metadata that are computed at build time:

* ``version`` -- taken from ``TT_METAL_MODELS_VERSION``, which ``build_wheel.py`` sets
  from the same setuptools-scm derivation the ``ttnn`` wheel uses, so that both
  artifacts built from one commit carry one version string.
* the ``ttnn`` pin -- ``ttnn==<that same version>``. The coupling between this tree and
  the runtime it calls is exact and undeclared in the source (``models/`` contains no
  version asserts), so a strict pin is the honest expression of it: skew becomes a
  resolver error instead of an ``AttributeError`` deep in model construction.

This file is copied into the staged build root by ``build_wheel.py``; it is not built
in place.
"""

import os

from setuptools import setup

VERSION_ENV_VAR = "TT_METAL_MODELS_VERSION"

# Runtime dependencies of the `models/` tree itself.
#
# Deliberately NOT inherited from tt_metal/python_env/requirements-dev.txt: that file is
# a development environment (pytest, black, jupyterlab, fiftyone, ...) and installing it
# to serve a model is what this package exists to avoid.
#
# Lower bounds, not pins, for the large ML frameworks. This package is installed
# alongside a vLLM fork that pins its own torch/transformers, and hard pins here would
# turn a working combination into a resolver conflict. The exact combination tt-metal
# tests against is in tt_metal/python_env/requirements-dev.txt. `ttnn` is the one
# genuine pin, added below.
INSTALL_REQUIRES = [
    # Core numerics and the tt-metal Python runtime's own stack. `ttnn` already brings
    # numpy/loguru/pyyaml/networkx/pandas; they are restated here because `models/`
    # imports them directly and should not depend on a transitive gift.
    "loguru>=0.6.0",
    "numpy>=1.24.4,<2",
    "pyyaml>=5.4",
    # Reference implementations and weight loading.
    "torch>=2.6",
    "transformers>=4.50",
    "huggingface-hub>=0.30.0",
    "safetensors>=0.4",
    # Tokenizers used by the Llama / Gemma / Mistral families.
    "sentencepiece>=0.2.0",
    "tiktoken>=0.7.0",
    "blobfile>=3.0.0",
    # Multimodal preprocessing: reached through the vision generators and, transitively,
    # through vLLM's pixtral/mistral3 chain.
    "torchvision>=0.21",
    "pillow>=10.0.0",
    # Small, widely imported utilities across models/.
    "einops>=0.6.1",
    "tqdm>=4.66.3",
    "pytz>=2023.3",
    "pydantic>=2.0",
]


def _version() -> str:
    version = os.environ.get(VERSION_ENV_VAR)
    if not version:
        raise SystemExit(
            f"{VERSION_ENV_VAR} is not set.\n"
            "This package is built through packaging/tt-metal-models/build_wheel.py, "
            "which derives the version from the repository so that it matches the ttnn "
            "wheel built from the same commit. Building this directory directly is not "
            "supported."
        )
    return version


def _install_requires(version: str) -> list:
    return [*INSTALL_REQUIRES, f"ttnn=={version}"]


if __name__ == "__main__":
    version = _version()
    setup(version=version, install_requires=_install_requires(version))
