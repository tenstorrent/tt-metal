"""Make the perf_automation package importable for tests without install.

Adds the perf_automation root (parent of this tests/ dir) to sys.path so
`import agent` resolves regardless of pytest's invocation cwd.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
