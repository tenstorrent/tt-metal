#!/usr/bin/env python3
"""Entry point script for the issue-verifier tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from issue_verifier.__main__ import main

sys.exit(main())
