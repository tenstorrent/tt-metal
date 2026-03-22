#!/usr/bin/env python3
"""Entry point script for the bug-checker tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bug_checker.__main__ import main

sys.exit(main())
