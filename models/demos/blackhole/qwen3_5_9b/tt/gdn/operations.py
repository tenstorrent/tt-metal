# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GDN tensor operations placeholder.

The Gated DeltaNet device ops live in the experimental backend
(`models.experimental.gated_attention_gated_deltanet.tt`) and the frozen
prefill kernel (`tt/gdn_kernel/gdn_kernel_op.py`); they are imported directly
by prefill.py / decode.py / weights.py.
"""
