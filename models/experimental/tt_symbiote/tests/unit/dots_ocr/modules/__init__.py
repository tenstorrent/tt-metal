# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module-level PCC tests for dots.ocr (text + vision).

Each test in this package:

* Builds the matching HuggingFace PyTorch ``nn.Module`` with random weights
  via ``reference.architecture_factory.build_random_*``.
* Constructs the production ``TTNNModule`` from the same PyTorch state_dict
  via ``from_torch`` so reference and TT modules share numerical weights.
* Drives a forward pass at the captured input shape (per-device shape from
  the Phase 0 capture matrix).
* Gathers the TT output back to host and asserts PCC against the PyTorch
  reference at a per-module threshold (see PLAN §5.2, §5.5).

The tests are run with the same mesh fixture (``mesh_device_t3k_dp``) as the
Phase 2 op-level tests and use ``enable_trace=False`` per user decision.
"""
