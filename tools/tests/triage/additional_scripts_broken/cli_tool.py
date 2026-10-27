# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Test fixture: not a triage script at all, but an ordinary command line tool that parses arguments
# at import time. Discovery imports every .py in a directory it is given, so this exits during
# import - and SystemExit is not an Exception. Skipping it must not end the whole triage run.

import sys

sys.exit(2)
