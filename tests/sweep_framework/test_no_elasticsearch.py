# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
"""Verify complete removal of Elasticsearch backend."""

import pytest
import sys
from pathlib import Path


def test_no_elasticsearch_import_in_framework():
    """Verify elasticsearch is not imported by framework modules."""
    # Clear any pre-existing imports
    mods_to_clear = [k for k in sys.modules.keys() if "elasticsearch" in k.lower()]
    for mod in mods_to_clear:
        del sys.modules[mod]

    # Import framework modules
    # Note: This test may need PYTHONPATH set correctly to run
    try:
        import os

        # Add sweep_framework to path
        sweep_path = Path(__file__).parent
        if str(sweep_path) not in sys.path:
            sys.path.insert(0, str(sweep_path))

        from framework import vector_source
        from framework import result_destination

        # Elasticsearch should not be loaded
        assert "elasticsearch" not in sys.modules, "elasticsearch should not be imported by framework"
    except ImportError as e:
        pytest.skip(f"Framework modules not available in test environment: {e}")


def test_elastic_config_removed():
    """Verify elastic_config.py has been removed."""
    elastic_config_path = Path(__file__).parent / "framework" / "elastic_config.py"
    assert not elastic_config_path.exists(), "elastic_config.py should be deleted"


def test_no_elastic_in_vector_sources():
    """Verify 'elastic' is not a supported vector source."""
    try:
        import sys
        from pathlib import Path

        sweep_path = Path(__file__).parent
        if str(sweep_path) not in sys.path:
            sys.path.insert(0, str(sweep_path))

        from framework.vector_source import VectorSourceFactory

        with pytest.raises(ValueError, match="Unknown vector source"):
            VectorSourceFactory.create_source("elastic")
    except ImportError as e:
        pytest.skip(f"Framework modules not available in test environment: {e}")


def test_no_elastic_in_result_destinations():
    """Verify 'elastic' is not a supported result destination."""
    try:
        import sys
        from pathlib import Path

        sweep_path = Path(__file__).parent
        if str(sweep_path) not in sys.path:
            sys.path.insert(0, str(sweep_path))

        from framework.result_destination import ResultDestinationFactory

        with pytest.raises(ValueError, match="Unknown result destination"):
            ResultDestinationFactory.create_destination("elastic")
    except ImportError as e:
        pytest.skip(f"Framework modules not available in test environment: {e}")


def test_no_elasticsearch_in_requirements():
    """Verify elasticsearch is not in requirements file."""
    requirements_path = Path(__file__).parent / "requirements-sweeps.txt"
    if requirements_path.exists():
        content = requirements_path.read_text()
        assert "elasticsearch" not in content.lower(), "elasticsearch should not be in requirements-sweeps.txt"


def test_no_elastic_references_in_python_files():
    """Verify no 'elastic' references in framework Python files (except comments)."""
    framework_dir = Path(__file__).parent / "framework"
    if not framework_dir.exists():
        pytest.skip("Framework directory not found")

    python_files = list(framework_dir.glob("*.py"))

    for py_file in python_files:
        if py_file.name == "__pycache__":
            continue

        content = py_file.read_text()
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if '"""' in line or "'''" in line:
                continue

            # Check for elastic references
            if "elastic" in line.lower():
                pytest.fail(f"Found 'elastic' reference in {py_file.name}:{line_num}\n" f"Line: {line}")


def test_deleted_utility_scripts():
    """Verify elasticsearch utility scripts have been removed."""
    sweep_dir = Path(__file__).parent

    # These scripts were Elasticsearch-dependent and should be deleted
    deleted_scripts = [
        sweep_dir / "sweeps_query.py",
        sweep_dir / "framework" / "export_to_sqlite.py",
    ]

    for script_path in deleted_scripts:
        assert not script_path.exists(), f"{script_path.name} should be deleted (was Elasticsearch-dependent)"


def test_vector_source_factory_supported_sources():
    """Verify VectorSourceFactory only supports file and vectors_export."""
    try:
        import sys
        from pathlib import Path

        sweep_path = Path(__file__).parent
        if str(sweep_path) not in sys.path:
            sys.path.insert(0, str(sweep_path))

        from framework.vector_source import VectorSourceFactory

        # Check SUPPORTED_SOURCES attribute if it exists
        if hasattr(VectorSourceFactory, "SUPPORTED_SOURCES"):
            assert VectorSourceFactory.SUPPORTED_SOURCES == {
                "file",
                "vectors_export",
            }, "VectorSourceFactory should only support 'file' and 'vectors_export'"
    except ImportError as e:
        pytest.skip(f"Framework modules not available in test environment: {e}")


def test_result_destination_factory_supported_destinations():
    """Verify ResultDestinationFactory only supports results_export and superset."""
    try:
        import sys
        from pathlib import Path

        sweep_path = Path(__file__).parent
        if str(sweep_path) not in sys.path:
            sys.path.insert(0, str(sweep_path))

        from framework.result_destination import ResultDestinationFactory

        # Check SUPPORTED_DESTINATIONS attribute if it exists
        if hasattr(ResultDestinationFactory, "SUPPORTED_DESTINATIONS"):
            assert ResultDestinationFactory.SUPPORTED_DESTINATIONS == {
                "results_export",
                "superset",
            }, "ResultDestinationFactory should only support 'results_export' and 'superset'"
    except ImportError as e:
        pytest.skip(f"Framework modules not available in test environment: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
