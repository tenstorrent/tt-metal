# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Vendored HF reference modeling for the DFlash drafter.

``dflash.py`` is a verbatim copy of the upstream z-lab DFlash ``dflash.py`` (``Qwen3DFlashAttention`` /
``DFlashDraftModel``) as shipped in the Kimi-*-DFlash HF repos — same filename as upstream. It is vendored
here so the device-vs-HF PCC test (``tests/speculative_decoding/dflash/test_dflash.py``) references a stable, in-repo copy
of the reference model instead of a modeling file inside the (re-downloadable) model checkout.

Only the model CODE lives here — weights and config come from ``$DFLASH_HF_MODEL`` (the checkout).

Provenance / license: ``dflash.py`` is THIRD-PARTY model code (z-lab / Inco AI DFlash), not
Tenstorrent-authored. Confirm the upstream license permits redistribution before publishing this repo;
the SPDX header above applies to this ``__init__`` wrapper only, not to ``dflash.py``.
"""
