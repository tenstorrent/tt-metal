# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
"""Command-R (c4ai-command-r-v01) TTNN port — org-internal (canada-quant/tt-metal).

Modules are imported lazily by models.tt_transformers.tt.model.Transformer when
args.model_type == "cohere" (family dispatch in Transformer.__init__). Nothing in
this package goes upstream without explicit owner approval.
"""
