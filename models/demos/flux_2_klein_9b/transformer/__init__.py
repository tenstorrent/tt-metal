# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end TTNN package for the FLUX.2 Klein 9B *transformer* component.

Deliberately empty of imports: `demo/` is run as `python -m
models.demos.flux_2_klein_9b.transformer.demo.demo_*`, so anything imported here
is paid for by every entrypoint (and would drag `ttnn` into host-only Gate 1
checks that do not need it).
"""
