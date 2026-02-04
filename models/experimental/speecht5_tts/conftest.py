# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for SpeechT5 TTS tests.

Adds custom command-line options specific to TTS generation tests.
These options only apply to tests in this directory and subdirectories.
"""


def pytest_addoption(parser):
    """Add custom command-line options for SpeechT5 TTS tests."""
    parser.addoption(
        "--input-text",
        action="store",
        default=None,
        help="Input text to convert to speech",
    )
    parser.addoption(
        "--max-steps",
        action="store",
        type=int,
        default=None,
        help="Maximum number of generation steps",
    )
    parser.addoption(
        "--output-dir",
        action="store",
        default=None,
        help="Directory to save output audio files",
    )
