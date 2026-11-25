# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Allowlisted deserialization for the Detectron2 Panoptic-DeepLab checkpoint (``model_final_bd324a.pkl``).

The checkpoint is a third-party ``.pkl`` (a dict of NumPy arrays), so it cannot be swapped for JSON.
Unrestricted ``pickle.load`` can execute arbitrary code, so load it with an allowlist of the globals that
plain NumPy arrays need and reject everything else.
"""

import pickle

_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),  # bytes round-trip emitted by protocol<=2 pickles of ndarray data
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Forbidden global in weights file: {module}.{name}")


def load_restricted_pickle(file):
    """Load a Detectron2 checkpoint from a binary file object using the allowlist above."""
    return RestrictedUnpickler(file, encoding="latin1").load()
