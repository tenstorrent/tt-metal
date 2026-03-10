# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Re-export shim so that stage tests can use:
#   from .layer_norm_rm import layer_norm_rm
#
# Loads the operation module directly by file path to avoid circular imports
# that arise when ttnn.operations.__init__ registers bare module names.
import importlib.util as _util
import os as _os

_op_path = _os.path.join(
    _os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "..",  # tests/ttnn/unit_tests/operations/layer_norm_rm -> repo root
    "ttnn",
    "ttnn",
    "operations",
    "layer_norm_rm",
    "layer_norm_rm.py",
)
_op_path = _os.path.normpath(_op_path)
_spec = _util.spec_from_file_location("_layer_norm_rm_op", _op_path)
_mod = _util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
layer_norm_rm = _mod.layer_norm_rm

__all__ = ["layer_norm_rm"]
