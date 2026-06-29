import sys
from pathlib import Path

# Allow `from tt.xxx import ...` and `from models.demos.time_series_transformer.tt.xxx import ...`
_here = Path(__file__).resolve().parent.parent  # .../time_series_transformer/
_repo_root = _here.parents[2]  # .../tt-metal/

for p in [str(_here), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)
