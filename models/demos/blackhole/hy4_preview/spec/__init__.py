# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview speculative decoding with the native MTP head: CPU-only accounting (accept, rollback, adaptive k, speedup model).

No ttnn import anywhere in this package: it is the exact specification the device code (task spec-mtp device stage) is tested against.
Design: $PROGRAM/spec-mtp/DESIGN.md.
"""
