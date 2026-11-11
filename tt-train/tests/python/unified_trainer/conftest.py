# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# pytest.ini selects --import-mode=importlib, which leaves this directory off sys.path; the shared fakes
# next to the tests (unified_trainer_fakes) need it there.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
