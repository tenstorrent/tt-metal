# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SST-2 (GLUE): the prompt of a review and the response of its label."""

from string import Template

SST2_PROMPT_TEMPLATE = Template("""Review: $input\nOutput:""")
SST2_RESPONSE_TEMPLATE = Template('{"label": "$label"}')
SST2_LABELS = {0: "negative", 1: "positive"}
