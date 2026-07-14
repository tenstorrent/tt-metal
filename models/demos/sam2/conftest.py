# Pytest test harness conftest for Tenstorrent SAM2 Bounty tests
import pytest
import sys
from pathlib import Path

# Add model root directory to sys.path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
