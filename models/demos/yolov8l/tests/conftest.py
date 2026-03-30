# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

# Repo pytest.ini sets timeout=300; YOLOv8l device forward + compile often exceeds that.
pytestmark = pytest.mark.timeout(1200)
