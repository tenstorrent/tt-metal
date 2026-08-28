# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The TT side of the component: stub loading, inputs, the HF golden, the pipeline.

No re-exports on purpose. `tt/stubs.py`, `tt/inputs.py` and `tt/reference.py` are
host-only (importable with no device and without `ttnn`); `tt/pipeline.py` is
not. Pulling them together here would make the cheap modules pay for the
expensive one.
"""
