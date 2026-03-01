#!/usr/bin/env python3
"""Mock llama_models to work around import issue"""

import sys
from unittest.mock import MagicMock

# Create mock module structure
sys.modules["llama_models"] = MagicMock()
sys.modules["llama_models.llama3"] = MagicMock()
sys.modules["llama_models.llama3.api"] = MagicMock()
sys.modules["llama_models.llama3.api.datatypes"] = MagicMock()


# Create a mock ImageMedia class
class MockImageMedia:
    pass


sys.modules["llama_models.llama3.api.datatypes"].ImageMedia = MockImageMedia
