"""Use the capacity verifier, without starting a model or native process."""

import sys
from pathlib import Path

OWNER = Path(__file__).resolve().parent
sys.path.insert(0, str(OWNER))
# The controller imports this public verifier from this entry module.
from verify_capacity import verify

__all__ = ["verify"]
